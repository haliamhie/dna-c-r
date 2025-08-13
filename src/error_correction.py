#!/usr/bin/env python3
"""
Error Correction Module for DNA Barcodes
Implements Hamming distance-based error correction with MiSeq-specific optimizations
Based on Johnson et al. (2023) and SSM principles
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class BarcodeErrorCorrector:
    """
    Error correction for 14bp barcodes with Hamming distance >= 4
    Handles MiSeq error rate of 0.1-0.4% per nucleotide
    """
    
    def __init__(self, reference_barcodes: List[str], min_hamming_dist: int = 4):
        """
        Initialize with reference barcode library
        
        Args:
            reference_barcodes: Known valid barcodes from micro_design
            min_hamming_dist: Minimum Hamming distance for error correction
        """
        self.reference_barcodes = set(reference_barcodes)
        self.min_hamming_dist = min_hamming_dist
        self.barcode_length = len(next(iter(reference_barcodes))) if reference_barcodes else 14
        
        # Build index for efficient lookup
        self._build_kmer_index()
        
    def _build_kmer_index(self, k: int = 7):
        """Build k-mer index for efficient neighbor finding (Shepherd method)"""
        self.kmer_index = defaultdict(set)
        for barcode in self.reference_barcodes:
            for i in range(len(barcode) - k + 1):
                kmer = barcode[i:i+k]
                self.kmer_index[kmer].add(barcode)
    
    def hamming_distance(self, s1: str, s2: str) -> int:
        """Calculate Hamming distance between two sequences"""
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance for handling indels
        Sequence-Levenshtein variant for DNA context
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Insertions, deletions, substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def find_nearest_valid_barcode(self, query: str, max_distance: int = 1) -> Optional[str]:
        """
        Find nearest valid barcode within max_distance edits
        Uses k-mer indexing for efficiency
        """
        if query in self.reference_barcodes:
            return query
        
        candidates = set()
        k = 7  # k-mer size
        
        # Find candidate barcodes sharing k-mers
        for i in range(len(query) - k + 1):
            kmer = query[i:i+k]
            if kmer in self.kmer_index:
                candidates.update(self.kmer_index[kmer])
        
        # Find closest match
        best_match = None
        min_dist = float('inf')
        
        for candidate in candidates:
            dist = self.hamming_distance(query, candidate)
            if dist <= max_distance and dist < min_dist:
                min_dist = dist
                best_match = candidate
        
        return best_match
    
    def correct_barcode_errors(self, observed_barcodes: List[str], 
                              counts: Optional[Dict[str, int]] = None) -> Dict[str, str]:
        """
        Correct errors in observed barcodes using reference library
        
        Args:
            observed_barcodes: Barcodes from sequencing
            counts: Read counts per barcode for abundance-based filtering
            
        Returns:
            Mapping of observed -> corrected barcodes
        """
        corrections = {}
        stats = {
            'exact_matches': 0,
            'single_error_corrections': 0,
            'multi_error_corrections': 0,
            'uncorrectable': 0
        }
        
        for obs_barcode in set(observed_barcodes):
            if obs_barcode in self.reference_barcodes:
                corrections[obs_barcode] = obs_barcode
                stats['exact_matches'] += 1
            else:
                # Try single error correction
                corrected = self.find_nearest_valid_barcode(obs_barcode, max_distance=1)
                if corrected:
                    corrections[obs_barcode] = corrected
                    stats['single_error_corrections'] += 1
                else:
                    # Try double error correction (with caution)
                    corrected = self.find_nearest_valid_barcode(obs_barcode, max_distance=2)
                    if corrected and counts:
                        # Only accept if abundance supports correction
                        if counts.get(obs_barcode, 0) > 10:  # Arbitrary threshold
                            corrections[obs_barcode] = corrected
                            stats['multi_error_corrections'] += 1
                        else:
                            stats['uncorrectable'] += 1
                    else:
                        stats['uncorrectable'] += 1
        
        logger.info(f"Error correction statistics: {stats}")
        return corrections
    
    def consensus_calling(self, barcode_reads: List[Tuple[str, int]], 
                         min_reads: int = 3) -> Optional[str]:
        """
        Generate consensus barcode from multiple reads
        Required for 315x average reuse rate
        
        Args:
            barcode_reads: List of (barcode, quality_score) tuples
            min_reads: Minimum reads required for consensus
            
        Returns:
            Consensus barcode or None if insufficient evidence
        """
        if len(barcode_reads) < min_reads:
            return None
        
        # Count occurrences of each barcode
        barcode_counts = Counter([br[0] for br in barcode_reads])
        
        # If clear majority, return it
        most_common = barcode_counts.most_common(1)[0]
        if most_common[1] > len(barcode_reads) * 0.5:
            return most_common[0]
        
        # Otherwise, build position-wise consensus
        consensus = []
        for pos in range(self.barcode_length):
            base_counts = Counter()
            for barcode, quality in barcode_reads:
                if len(barcode) > pos:
                    # Weight by quality if available
                    weight = quality if quality else 1
                    base_counts[barcode[pos]] += weight
            
            if base_counts:
                consensus.append(base_counts.most_common(1)[0][0])
            else:
                return None  # Inconsistent length
        
        consensus_barcode = ''.join(consensus)
        
        # Validate against reference
        if consensus_barcode in self.reference_barcodes:
            return consensus_barcode
        else:
            # Try error correction on consensus
            return self.find_nearest_valid_barcode(consensus_barcode, max_distance=1)
    
    def filter_by_abundance(self, barcode_counts: Dict[str, int], 
                           min_count: int = 5) -> Set[str]:
        """
        Filter barcodes by abundance to remove sequencing artifacts
        Based on Bartender algorithm principles
        
        Args:
            barcode_counts: Dictionary of barcode -> count
            min_count: Minimum count threshold
            
        Returns:
            Set of valid barcodes passing filter
        """
        # Calculate dynamic threshold based on distribution
        counts = list(barcode_counts.values())
        if not counts:
            return set()
        
        # Use median absolute deviation for robust threshold
        median_count = np.median(counts)
        mad = np.median([abs(c - median_count) for c in counts])
        threshold = max(min_count, median_count - 3 * mad)
        
        valid_barcodes = {
            barcode for barcode, count in barcode_counts.items()
            if count >= threshold
        }
        
        logger.info(f"Abundance filter: kept {len(valid_barcodes)}/{len(barcode_counts)} barcodes")
        return valid_barcodes
    
    def calculate_error_rates(self, observed_barcodes: List[str]) -> Dict:
        """
        Calculate empirical error rates from observed vs reference barcodes
        """
        error_stats = {
            'substitution_rate': 0,
            'insertion_rate': 0,
            'deletion_rate': 0,
            'position_specific_errors': [0] * self.barcode_length,
            'base_specific_errors': {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        }
        
        total_bases = 0
        total_errors = 0
        
        for obs in observed_barcodes:
            if obs in self.reference_barcodes:
                continue
                
            # Find closest reference
            closest = self.find_nearest_valid_barcode(obs, max_distance=3)
            if not closest:
                continue
            
            # Count errors by type
            if len(obs) == len(closest):
                # Substitutions only
                for i, (o, r) in enumerate(zip(obs, closest)):
                    total_bases += 1
                    if o != r:
                        total_errors += 1
                        error_stats['position_specific_errors'][i] += 1
                        if r in 'ACGT':
                            error_stats['base_specific_errors'][r] += 1
            
        if total_bases > 0:
            error_stats['substitution_rate'] = total_errors / total_bases
        
        return error_stats


class MiSeqErrorModel:
    """
    MiSeq-specific error model for quality-aware correction
    Handles cluster density and Q30 score considerations
    """
    
    def __init__(self, q30_threshold: float = 0.9):
        """
        Initialize MiSeq error model
        
        Args:
            q30_threshold: Expected Q30 score (typically >90% for MiSeq)
        """
        self.q30_threshold = q30_threshold
        self.phred_to_prob = lambda q: 10 ** (-q / 10)
        
    def calculate_error_probability(self, quality_scores: List[int]) -> float:
        """Calculate error probability from Phred quality scores"""
        if not quality_scores:
            return 0.001  # Default MiSeq error rate
        
        error_probs = [self.phred_to_prob(q) for q in quality_scores]
        return np.mean(error_probs)
    
    def should_trust_read(self, quality_scores: List[int], min_q: int = 30) -> bool:
        """Determine if read quality is sufficient for trust"""
        if not quality_scores:
            return False
        
        q30_fraction = sum(q >= min_q for q in quality_scores) / len(quality_scores)
        return q30_fraction >= self.q30_threshold
    
    def quality_weighted_consensus(self, sequences: List[Tuple[str, List[int]]]) -> str:
        """
        Generate quality-weighted consensus sequence
        
        Args:
            sequences: List of (sequence, quality_scores) tuples
            
        Returns:
            Consensus sequence
        """
        if not sequences:
            return ""
        
        seq_length = len(sequences[0][0])
        consensus = []
        
        for pos in range(seq_length):
            base_weights = defaultdict(float)
            
            for seq, quals in sequences:
                if len(seq) > pos and len(quals) > pos:
                    base = seq[pos]
                    # Weight by quality (higher quality = more weight)
                    weight = quals[pos] if pos < len(quals) else 30
                    base_weights[base] += weight
            
            # Select base with highest weighted count
            if base_weights:
                best_base = max(base_weights.items(), key=lambda x: x[1])[0]
                consensus.append(best_base)
            else:
                consensus.append('N')  # Unknown
        
        return ''.join(consensus)


def main():
    """Example usage of error correction module"""
    # Example reference barcodes
    reference = ['ATCGATCGATCGAT', 'GCTAGCTAGCTAGC', 'CGCGCGCGCGCGCG']
    
    corrector = BarcodeErrorCorrector(reference)
    
    # Example observed barcodes with errors
    observed = ['ATCGATCGATCGAT',  # Exact match
                'ATCGATCGATCGAC',  # Single error
                'GCCAGCTAGCTAGC']  # Single error
    
    corrections = corrector.correct_barcode_errors(observed)
    print(f"Corrections: {corrections}")
    
    # Test consensus calling
    reads = [('ATCGATCGATCGAT', 35), 
             ('ATCGATCGATCGAC', 30),
             ('ATCGATCGATCGAT', 40)]
    consensus = corrector.consensus_calling(reads)
    print(f"Consensus: {consensus}")


if __name__ == "__main__":
    main()