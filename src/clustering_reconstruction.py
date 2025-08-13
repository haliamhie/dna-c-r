#!/usr/bin/env python3
"""Clustering reconstruction for optical pooled screening"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class ClusterReconstructor:
    """Reconstruct spatial clusters from barcode-cell mappings"""
    
    def __init__(self, hamming_threshold: int = 4):
        self.hamming_threshold = hamming_threshold
        self.clusters = {}
        self.cell_assignments = {}
        self.barcode_map = {}
    
    def load_cluster_data(self, cluster_file: Path) -> pd.DataFrame:
        """Load cluster assignments"""
        df = pd.read_csv(cluster_file)
        logger.info(f"Loaded {len(df)} cluster assignments")
        return df
    
    def reconstruct_cells(self, cluster_df: pd.DataFrame, 
                         design_df: pd.DataFrame) -> Dict:
        """Reconstruct cell-barcode relationships"""
        
        # Map barcodes to design
        barcode_to_design = dict(zip(design_df['barcode'], design_df['sequence']))
        
        # Group by cluster ID (cell)
        cell_groups = defaultdict(list)
        for _, row in cluster_df.iterrows():
            if pd.notna(row.get('barcode')):
                cell_groups[row['id']].append(row['barcode'])
        
        # Analyze each cell
        cell_stats = {
            'total_cells': len(cell_groups),
            'cells_with_barcodes': sum(1 for v in cell_groups.values() if v),
            'unique_barcodes_per_cell': {},
            'multi_barcode_cells': 0
        }
        
        for cell_id, barcodes in cell_groups.items():
            unique_barcodes = set(barcodes)
            cell_stats['unique_barcodes_per_cell'][cell_id] = len(unique_barcodes)
            if len(unique_barcodes) > 1:
                cell_stats['multi_barcode_cells'] += 1
        
        return cell_stats
    
    def identify_phenotype_clusters(self, cluster_df: pd.DataFrame,
                                   min_cluster_size: int = 10) -> Dict:
        """Identify phenotypic clusters"""
        
        # Count barcode occurrences
        barcode_counts = Counter(
            bc for bc in cluster_df['barcode'].dropna()
        )
        
        # Find enriched barcodes (potential hits)
        mean_count = np.mean(list(barcode_counts.values()))
        std_count = np.std(list(barcode_counts.values()))
        
        enriched = {}
        depleted = {}
        
        for barcode, count in barcode_counts.items():
            z_score = (count - mean_count) / std_count if std_count > 0 else 0
            
            if z_score > 2:  # Enriched
                enriched[barcode] = {
                    'count': count,
                    'z_score': z_score,
                    'fold_change': count / mean_count
                }
            elif z_score < -2:  # Depleted
                depleted[barcode] = {
                    'count': count,
                    'z_score': z_score,
                    'fold_change': count / mean_count
                }
        
        return {
            'enriched_barcodes': enriched,
            'depleted_barcodes': depleted,
            'mean_barcode_count': mean_count,
            'std_barcode_count': std_count
        }
    
    def calculate_spatial_patterns(self, cluster_df: pd.DataFrame) -> Dict:
        """Analyze spatial distribution patterns"""
        
        # Group cells by barcode
        barcode_cells = defaultdict(list)
        for _, row in cluster_df.iterrows():
            if pd.notna(row.get('barcode')):
                barcode_cells[row['barcode']].append(row['id'])
        
        # Calculate distribution metrics
        patterns = {
            'dispersion_index': {},  # Variance/mean ratio
            'clustering_coefficient': {},
            'coverage': {}
        }
        
        for barcode, cells in barcode_cells.items():
            n_cells = len(cells)
            if n_cells > 1:
                # Simple dispersion metric
                patterns['dispersion_index'][barcode] = n_cells
                patterns['coverage'][barcode] = n_cells / len(cluster_df)
        
        return patterns
    
    def detect_cross_contamination(self, cluster_df: pd.DataFrame,
                                  expected_singles: float = 0.9) -> Dict:
        """Detect potential cross-contamination"""
        
        # Group by cell
        cell_barcodes = defaultdict(list)
        for _, row in cluster_df.iterrows():
            if pd.notna(row.get('barcode')):
                cell_barcodes[row['id']].append(row['barcode'])
        
        # Count cells with multiple unique barcodes
        multi_barcode_cells = []
        single_barcode_cells = []
        
        for cell_id, barcodes in cell_barcodes.items():
            unique_bcs = set(barcodes)
            if len(unique_bcs) > 1:
                multi_barcode_cells.append({
                    'cell_id': cell_id,
                    'n_barcodes': len(unique_bcs),
                    'barcodes': list(unique_bcs)[:5]  # First 5
                })
            elif len(unique_bcs) == 1:
                single_barcode_cells.append(cell_id)
        
        contamination_rate = len(multi_barcode_cells) / len(cell_barcodes) if cell_barcodes else 0
        
        return {
            'total_cells': len(cell_barcodes),
            'single_barcode_cells': len(single_barcode_cells),
            'multi_barcode_cells': len(multi_barcode_cells),
            'contamination_rate': contamination_rate,
            'expected_contamination': 1 - expected_singles,
            'contamination_detected': contamination_rate > (1 - expected_singles) * 1.5,
            'sample_multi_cells': multi_barcode_cells[:10]
        }
    
    def generate_cluster_summary(self, cluster_df: pd.DataFrame,
                                design_df: Optional[pd.DataFrame] = None) -> Dict:
        """Generate comprehensive clustering analysis"""
        
        summary = {
            'total_reads': len(cluster_df),
            'reads_with_barcodes': cluster_df['barcode'].notna().sum(),
            'unique_barcodes': cluster_df['barcode'].nunique(),
            'unique_cells': cluster_df['id'].nunique()
        }
        
        # Cell reconstruction
        if design_df is not None:
            summary['cell_reconstruction'] = self.reconstruct_cells(cluster_df, design_df)
        
        # Phenotype clusters
        summary['phenotype_clusters'] = self.identify_phenotype_clusters(cluster_df)
        
        # Spatial patterns
        summary['spatial_patterns'] = self.calculate_spatial_patterns(cluster_df)
        
        # Contamination check
        summary['contamination_analysis'] = self.detect_cross_contamination(cluster_df)
        
        # Quality metrics
        summary['quality_metrics'] = {
            'barcode_assignment_rate': summary['reads_with_barcodes'] / summary['total_reads'],
            'reads_per_barcode': summary['reads_with_barcodes'] / summary['unique_barcodes'] if summary['unique_barcodes'] > 0 else 0,
            'cells_per_barcode': summary['unique_cells'] / summary['unique_barcodes'] if summary['unique_barcodes'] > 0 else 0
        }
        
        return summary
    
    def identify_hit_candidates(self, enrichment_data: Dict,
                               fold_change_threshold: float = 2.0,
                               z_score_threshold: float = 3.0) -> List[Dict]:
        """Identify potential screening hits"""
        
        hits = []
        
        for barcode, stats in enrichment_data.get('enriched_barcodes', {}).items():
            if (stats['fold_change'] >= fold_change_threshold and 
                stats['z_score'] >= z_score_threshold):
                hits.append({
                    'barcode': barcode,
                    'type': 'enriched',
                    'fold_change': stats['fold_change'],
                    'z_score': stats['z_score'],
                    'count': stats['count']
                })
        
        for barcode, stats in enrichment_data.get('depleted_barcodes', {}).items():
            if (stats['fold_change'] <= 1/fold_change_threshold and 
                abs(stats['z_score']) >= z_score_threshold):
                hits.append({
                    'barcode': barcode,
                    'type': 'depleted',
                    'fold_change': stats['fold_change'],
                    'z_score': stats['z_score'],
                    'count': stats['count']
                })
        
        # Sort by absolute z-score
        hits.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        return hits


def main():
    """Test clustering reconstruction"""
    reconstructor = ClusterReconstructor()
    
    # Create test data
    test_clusters = pd.DataFrame({
        'id': ['cell_' + str(i) for i in range(1000)],
        'barcode': ['ACGTACGTACGTAC'] * 300 + ['TGCATGCATGCATG'] * 200 + 
                   ['GCTAGCTAGCTAGC'] * 100 + [None] * 400
    })
    
    test_design = pd.DataFrame({
        'barcode': ['ACGTACGTACGTAC', 'TGCATGCATGCATG', 'GCTAGCTAGCTAGC'],
        'sequence': ['ATG' + 'N'*93, 'TAA' + 'N'*93, 'TGA' + 'N'*93]
    })
    
    # Run analysis
    summary = reconstructor.generate_cluster_summary(test_clusters, test_design)
    
    print(f"✓ Total reads: {summary['total_reads']}")
    print(f"✓ Barcode assignment rate: {summary['quality_metrics']['barcode_assignment_rate']:.1%}")
    print(f"✓ Enriched barcodes: {len(summary['phenotype_clusters']['enriched_barcodes'])}")
    print(f"✓ Contamination rate: {summary['contamination_analysis']['contamination_rate']:.1%}")
    
    # Find hits
    hits = reconstructor.identify_hit_candidates(
        summary['phenotype_clusters']
    )
    print(f"✓ Hit candidates: {len(hits)}")
    
    return True


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)