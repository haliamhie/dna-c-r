#!/usr/bin/env python3
"""
CRISPR Screening Analysis Module
Specialized for Optical Pooled Screening (OPS) with CRISPRmap methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CRISPRScreenAnalyzer:
    """
    Analyzer for Tel-Hai College CRISPR screening data
    Handles 10,000 unique barcodes with high reuse rate (315x average)
    """
    
    def __init__(self, barcode_length: int = 14, min_hamming_dist: int = 4):
        self.barcode_length = barcode_length
        self.min_hamming_dist = min_hamming_dist
        self.barcode_stats = {}
        
    def hamming_distance(self, s1: str, s2: str) -> int:
        """Calculate Hamming distance between two sequences"""
        if len(s1) != len(s2):
            raise ValueError("Sequences must be of equal length")
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    def validate_barcode_library(self, barcodes: List[str]) -> Dict:
        """
        Validate barcode library meets CRISPR screening requirements
        Based on Sequence Symmetry Minimization (SSM) principles
        """
        validation_results = {
            'total_barcodes': len(barcodes),
            'unique_barcodes': len(set(barcodes)),
            'correct_length': all(len(b) == self.barcode_length for b in barcodes),
            'min_hamming_distances': [],
            'error_correction_capable': False,
            'nucleotide_balance': {}
        }
        
        # Check minimum Hamming distances
        unique_barcodes = list(set(barcodes))
        for i, bc1 in enumerate(unique_barcodes[:100]):  # Sample for efficiency
            min_dist = float('inf')
            for j, bc2 in enumerate(unique_barcodes):
                if i != j:
                    dist = self.hamming_distance(bc1, bc2)
                    min_dist = min(min_dist, dist)
            validation_results['min_hamming_distances'].append(min_dist)
        
        # Check error correction capability
        min_observed = min(validation_results['min_hamming_distances'])
        validation_results['error_correction_capable'] = min_observed >= self.min_hamming_dist
        validation_results['can_correct_errors'] = min_observed // 2
        
        # Analyze nucleotide balance (should be ~25% each for synthetic random)
        all_bases = ''.join(barcodes)
        base_counts = Counter(all_bases)
        total_bases = len(all_bases)
        for base in 'ACGT':
            validation_results['nucleotide_balance'][base] = base_counts.get(base, 0) / total_bases
        
        return validation_results
    
    def analyze_start_stop_codons(self, sequences: List[str]) -> Dict:
        """
        Analyze start and stop codon frequency in payload sequences
        High frequency (82-84%) indicates loss-of-function or variant screening
        """
        start_codon = 'ATG'
        stop_codons = ['TAA', 'TAG', 'TGA']
        
        results = {
            'total_sequences': len(sequences),
            'with_start_codon': 0,
            'with_stop_codon': 0,
            'start_positions': [],
            'stop_positions': [],
            'functional_orfs': 0
        }
        
        for seq in sequences:
            # Check for start codon
            if start_codon in seq:
                results['with_start_codon'] += 1
                pos = seq.find(start_codon)
                results['start_positions'].append(pos)
                
            # Check for stop codons
            has_stop = False
            for stop in stop_codons:
                if stop in seq:
                    has_stop = True
                    pos = seq.find(stop)
                    results['stop_positions'].append(pos)
                    break
            if has_stop:
                results['with_stop_codon'] += 1
                
            # Check for functional ORF (start before stop)
            if start_codon in seq and has_stop:
                start_pos = seq.find(start_codon)
                stop_pos = min([seq.find(s) for s in stop_codons if s in seq])
                if start_pos < stop_pos:
                    results['functional_orfs'] += 1
        
        # Calculate percentages
        results['start_codon_freq'] = results['with_start_codon'] / results['total_sequences']
        results['stop_codon_freq'] = results['with_stop_codon'] / results['total_sequences']
        results['orf_frequency'] = results['functional_orfs'] / results['total_sequences']
        
        return results
    
    def calculate_barcode_reuse_statistics(self, barcode_counts: Dict[str, int]) -> Dict:
        """
        Analyze barcode reuse patterns for optical pooled screening
        Expected: ~315x average reuse for statistical power
        """
        counts = list(barcode_counts.values())
        
        return {
            'total_unique_barcodes': len(barcode_counts),
            'total_reads': sum(counts),
            'mean_reuse': np.mean(counts),
            'median_reuse': np.median(counts),
            'std_reuse': np.std(counts),
            'min_reuse': min(counts),
            'max_reuse': max(counts),
            'reuse_distribution': {
                '<100': sum(1 for c in counts if c < 100),
                '100-200': sum(1 for c in counts if 100 <= c < 200),
                '200-300': sum(1 for c in counts if 200 <= c < 300),
                '300-400': sum(1 for c in counts if 300 <= c < 400),
                '400-500': sum(1 for c in counts if 400 <= c < 500),
                '>500': sum(1 for c in counts if c >= 500)
            },
            'sufficient_coverage': np.mean(counts) > 100  # Minimum for statistical power
        }
    
    def identify_variant_type(self, design_df: pd.DataFrame) -> str:
        """
        Determine the type of CRISPR variant library based on sequence characteristics
        """
        # Analyze first 100 sequences for efficiency
        sample = design_df.head(100)
        
        # Extract payloads (sequences after barcode)
        payloads = sample['sequence'].str[14:]
        
        # Check for guide RNA features (typically 20bp with PAM)
        has_pam = payloads.str.contains('GG').mean() > 0.8  # NGG PAM for SpCas9
        
        # Check for regulatory elements
        has_promoter = payloads.str.contains('TATA|CAAT').mean() > 0.3
        
        # Check codon analysis results
        codon_results = self.analyze_start_stop_codons(payloads.tolist())
        high_stop_freq = codon_results['stop_codon_freq'] > 0.8
        
        if high_stop_freq:
            return "Saturation mutagenesis or loss-of-function screen"
        elif has_pam:
            return "CRISPR guide RNA library"
        elif has_promoter:
            return "Regulatory element screen"
        else:
            return "General variant library"
    
    def generate_quality_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive quality metrics for CRISPR screening data
        """
        metrics = {
            'total_reads': len(data),
            'unique_barcodes': data['barcode'].nunique() if 'barcode' in data.columns else 0,
            'valid_reads': data['barcode'].notna().sum() if 'barcode' in data.columns else 0,
            'failure_rate': 0,
            'coverage_uniformity': 0,
            'gc_content': {},
            'sequence_complexity': 0
        }
        
        if 'barcode' in data.columns:
            # Calculate failure rate
            metrics['failure_rate'] = data['barcode'].isna().mean()
            
            # Calculate coverage uniformity (Gini coefficient)
            if metrics['unique_barcodes'] > 0:
                barcode_counts = data['barcode'].value_counts().values
                sorted_counts = np.sort(barcode_counts)
                n = len(sorted_counts)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
                metrics['coverage_uniformity'] = 1 - gini  # Higher is more uniform
            
            # GC content analysis
            valid_barcodes = data['barcode'].dropna()
            if len(valid_barcodes) > 0:
                gc_counts = valid_barcodes.apply(lambda x: (x.count('G') + x.count('C')) / len(x))
                metrics['gc_content'] = {
                    'mean': gc_counts.mean(),
                    'std': gc_counts.std(),
                    'min': gc_counts.min(),
                    'max': gc_counts.max()
                }
        
        return metrics
    
    def reconstruct_clustering(self, barcodes: List[str], max_distance: int = 2) -> Dict[str, List[str]]:
        """
        Reconstruct barcode clustering using maximal distance criteria
        Similar to Bartender algorithm for high-complexity barcode data
        """
        clusters = {}
        assigned = set()
        
        # Sort barcodes by frequency if counts available
        sorted_barcodes = sorted(set(barcodes))
        
        for i, seed in enumerate(sorted_barcodes):
            if seed in assigned:
                continue
                
            cluster = [seed]
            assigned.add(seed)
            
            # Find all barcodes within max_distance
            for candidate in sorted_barcodes[i+1:]:
                if candidate not in assigned:
                    if self.hamming_distance(seed, candidate) <= max_distance:
                        cluster.append(candidate)
                        assigned.add(candidate)
            
            clusters[seed] = cluster
        
        logger.info(f"Created {len(clusters)} clusters from {len(barcodes)} barcodes")
        return clusters


def main():
    """Example usage of CRISPR analyzer"""
    analyzer = CRISPRScreenAnalyzer()
    
    # Load sample data
    logger.info("Initializing CRISPR Screen Analyzer for Tel-Hai dataset")
    logger.info(f"Expected: {analyzer.barcode_length}bp barcodes with Hamming distance >= {analyzer.min_hamming_dist}")
    
    # Example validation
    sample_barcodes = ['ATCGATCGATCGAT', 'GCTAGCTAGCTAGC', 'TATATATATATATA']
    validation = analyzer.validate_barcode_library(sample_barcodes)
    logger.info(f"Validation results: {validation}")


if __name__ == "__main__":
    main()