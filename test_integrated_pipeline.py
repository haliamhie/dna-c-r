#!/usr/bin/env python3
"""Integration test for complete CRISPR pipeline"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from crispr_pipeline import CRISPRPipeline
import pandas as pd
import numpy as np


def create_test_data():
    """Create test datasets"""
    # Create test clusters
    np.random.seed(42)
    n_reads = 1000
    
    barcodes = (
        ['ACGTACGTACGTAC'] * 200 +
        ['TGCATGCATGCATG'] * 150 +
        ['GCTAGCTAGCTAGC'] * 100 +
        [None] * 550  # 55% failure rate
    )
    
    test_clusters = pd.DataFrame({
        'id': [f'cell_{i%100}' for i in range(n_reads)],
        'barcode': barcodes[:n_reads]
    })
    
    # Create test design
    test_design = pd.DataFrame({
        'barcode': ['ACGTACGTACGTAC', 'TGCATGCATGCATG', 'GCTAGCTAGCTAGC'],
        'sequence': [
            'ATG' + 'ACG' * 31,  # Start codon
            'TAA' + 'CGT' * 31,  # Stop codon
            'TGA' + 'GCT' * 31   # Stop codon
        ]
    })
    
    # Save test files
    test_dir = Path("/home/mch/dna/test_data")
    test_dir.mkdir(exist_ok=True)
    
    cluster_file = test_dir / "test_clusters.csv"
    design_file = test_dir / "test_design.csv"
    
    test_clusters.to_csv(cluster_file, index=False)
    test_design.to_csv(design_file, index=False)
    
    return cluster_file, design_file

def test_pipeline():
    """Test integrated pipeline"""
    
    # Create test data
    cluster_file, design_file = create_test_data()
    
    # Run pipeline with quick mode
    pipeline = CRISPRPipeline(output_dir=Path("/home/mch/dna/test_results"))
    
    print("Running integrated pipeline test...")
    results = pipeline.run_analysis(
        cluster_file,
        design_file,
        full_analysis=False  # Quick mode for testing
    )
    
    # Validate results
    assert 'data_summary' in results
    print(f"✓ Data loaded: {results['data_summary']['total_reads']} reads")
    
    assert 'quality_control' in results
    print(f"✓ QC completed: {results['quality_control']['summary']['failure_rate']:.1%} failure rate")
    
    assert 'error_correction' in results
    print(f"✓ Error correction: {results['error_correction']['correction_rate']:.1%} corrected")
    
    assert 'crispr_analysis' in results
    print(f"✓ CRISPR analysis: {results['crispr_analysis']['loss_of_function_rate']:.1%} LoF")
    
    assert 'clustering' in results
    print(f"✓ Clustering: {results['clustering']['unique_cells']} cells analyzed")
    
    assert 'runtime_seconds' in results
    print(f"✓ Runtime: {results['runtime_seconds']:.1f} seconds")
    
    # Save test results
    test_report = pipeline.output_dir / "test_results.json"
    import json
    with open(test_report, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to: {test_report}")
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    print(f"\nIntegration test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)