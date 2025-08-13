#!/usr/bin/env python3
"""Test visualization module"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.visualization import CRISPRVisualizer
import json

def test_visualization():
    """Test visualization functionality"""
    viz = CRISPRVisualizer()
    
    # Test with sample QC report
    qc_report = {
        'summary': {
            'total_reads': 10000,
            'valid_barcodes': 4500,
            'failed_barcodes': 5500,
            'failure_rate': 0.55
        },
        'failure_diagnosis': {
            'pcr_amplification_bias': 0.17,
            'rna_degradation': 0.10,
            'cell_segmentation_errors': 0.05
        },
        'coverage_statistics': {
            'mean_coverage': 315,
            'median_coverage': 280,
            'std_coverage': 120
        }
    }
    
    # Generate report
    output = viz._save_text_report(qc_report, "test_qc.json")
    html = viz.generate_html_report({'qc_report': qc_report})
    
    print(f"✓ Text report: {output}")
    print(f"✓ HTML report: {html}")
    print(f"✓ All outputs in: {viz.output_dir}")
    
    return output.exists() and html.exists()

if __name__ == "__main__":
    success = test_visualization()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)