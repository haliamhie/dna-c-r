#!/usr/bin/env python3
"""Test report generator module"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.report_generator import OpticalScreeningReport

def test_report_generator():
    """Test report generation"""
    report = OpticalScreeningReport()
    
    # Add comprehensive test data
    report.add_experimental_metadata({
        'institution': 'Tel-Hai College',
        'experiment_date': '2023-05-11'
    })
    
    report.add_sequencing_metrics({
        'platform': 'MiSeq',
        'total_reads': 7000000,
        'quality_scores': {'Q30': 0.92}
    })
    
    report.add_library_statistics({
        'total_barcodes': 10000,
        'actual_coverage': {'mean': 315, 'std': 120}
    })
    
    report.add_quality_control({
        'failure_rate': 0.555,
        'failure_diagnosis': {
            'pcr_amplification_bias': 0.17,
            'rna_degradation': 0.10
        },
        'recommendations': [
            'Optimize PCR conditions',
            'Improve RNA handling'
        ]
    })
    
    report.add_error_correction({
        'corrected_count': 2500000,
        'correction_rate': 0.357
    })
    
    report.add_screening_results({
        'lof_percentage': 0.82,
        'start_codon_count': 8200,
        'stop_codon_count': 8400,
        'frameshift_count': 1200,
        'hit_candidates': ['gene1', 'gene2', 'gene3']
    })
    
    # Generate reports
    json_path = report.save_json_report("phase3_test.json")
    md_path = report.save_markdown_report("phase3_test.md")
    tel_hai = report.generate_tel_hai_format()
    
    print(f"✓ JSON report: {json_path}")
    print(f"✓ Markdown report: {md_path}")
    print(f"✓ Tel-Hai format: {len(tel_hai)} sections")
    
    return json_path.exists() and md_path.exists()

if __name__ == "__main__":
    success = test_report_generator()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)