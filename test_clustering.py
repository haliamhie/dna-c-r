#!/usr/bin/env python3
"""Test clustering reconstruction module"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from src.clustering_reconstruction import ClusterReconstructor

def test_clustering():
    """Test clustering reconstruction"""
    reconstructor = ClusterReconstructor(hamming_threshold=4)
    
    # Create realistic test data
    import numpy as np
    np.random.seed(42)
    
    # Simulate 7M reads with 55% failure rate
    n_reads = 10000  # Smaller for testing
    barcodes = ['ACGT' * 3 + 'AC'] * 2000 + \
               ['TGCA' * 3 + 'TG'] * 1500 + \
               ['GCTA' * 3 + 'GC'] * 1000 + \
               [None] * 5500  # 55% failure
    
    test_clusters = pd.DataFrame({
        'id': [f'cell_{i%1000}' for i in range(n_reads)],
        'barcode': barcodes[:n_reads]
    })
    
    test_design = pd.DataFrame({
        'barcode': ['ACGTACGTACGTAC', 'TGCATGCATGCATG', 'GCTAGCTAGCTAGC'],
        'sequence': ['ATG' + 'N'*93, 'TAA' + 'N'*93, 'TGA' + 'N'*93]
    })
    
    # Run comprehensive analysis
    summary = reconstructor.generate_cluster_summary(test_clusters, test_design)
    
    # Test cell reconstruction
    assert 'cell_reconstruction' in summary
    print(f"✓ Cells analyzed: {summary['cell_reconstruction']['total_cells']}")
    
    # Test phenotype identification
    assert 'phenotype_clusters' in summary
    enriched = summary['phenotype_clusters']['enriched_barcodes']
    print(f"✓ Enriched barcodes: {len(enriched)}")
    
    # Test contamination detection
    assert 'contamination_analysis' in summary
    contam = summary['contamination_analysis']
    print(f"✓ Contamination rate: {contam['contamination_rate']:.1%}")
    
    # Test hit identification
    hits = reconstructor.identify_hit_candidates(
        summary['phenotype_clusters'],
        fold_change_threshold=1.5,
        z_score_threshold=2.0
    )
    print(f"✓ Hit candidates identified: {len(hits)}")
    
    # Test quality metrics
    assert 'quality_metrics' in summary
    qm = summary['quality_metrics']
    print(f"✓ Assignment rate: {qm['barcode_assignment_rate']:.1%}")
    print(f"✓ Reads per barcode: {qm['reads_per_barcode']:.1f}")
    
    return True

if __name__ == "__main__":
    success = test_clustering()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)