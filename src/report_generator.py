#!/usr/bin/env python3
"""Report generator for optical pooled CRISPR screening"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class OpticalScreeningReport:
    """Generate comprehensive reports for optical pooled screening"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("/home/mch/dna/reports")
        self.output_dir.mkdir(exist_ok=True)
        self.report_data = {}
        self.timestamp = datetime.now().isoformat()
    
    def add_experimental_metadata(self, metadata: Dict):
        """Add experimental metadata"""
        self.report_data['metadata'] = {
            'timestamp': self.timestamp,
            'experiment_type': 'Optical Pooled CRISPR Screening',
            'methodology': 'CRISPRmap',
            **metadata
        }
    
    def add_sequencing_metrics(self, metrics: Dict):
        """Add sequencing quality metrics"""
        self.report_data['sequencing'] = {
            'platform': metrics.get('platform', 'MiSeq'),
            'read_length': metrics.get('read_length', 150),
            'paired_end': metrics.get('paired_end', True),
            'total_reads': metrics.get('total_reads', 0),
            'quality_scores': metrics.get('quality_scores', {}),
            'adapter_contamination': metrics.get('adapter_contamination', 0)
        }
    
    def add_library_statistics(self, stats: Dict):
        """Add library design statistics"""
        self.report_data['library'] = {
            'total_barcodes': stats.get('total_barcodes', 10000),
            'barcode_length': stats.get('barcode_length', 14),
            'payload_length': stats.get('payload_length', 96),
            'hamming_distance': stats.get('hamming_distance', 4),
            'coverage_target': stats.get('coverage_target', 315),
            'actual_coverage': stats.get('actual_coverage', {})
        }
    
    def add_quality_control(self, qc_data: Dict):
        """Add QC analysis results"""
        self.report_data['quality_control'] = {
            'success_rate': 1 - qc_data.get('failure_rate', 0.555),
            'failure_causes': qc_data.get('failure_diagnosis', {}),
            'coverage_uniformity': qc_data.get('coverage_statistics', {}),
            'recommendations': qc_data.get('recommendations', [])
        }
    
    def add_error_correction(self, ec_results: Dict):
        """Add error correction results"""
        self.report_data['error_correction'] = {
            'sequences_corrected': ec_results.get('corrected_count', 0),
            'consensus_sequences': ec_results.get('consensus_count', 0),
            'uncorrectable': ec_results.get('uncorrectable_count', 0),
            'correction_rate': ec_results.get('correction_rate', 0)
        }
    
    def add_screening_results(self, results: Dict):
        """Add CRISPR screening analysis results"""
        self.report_data['screening'] = {
            'loss_of_function': results.get('lof_percentage', 0),
            'start_codons': results.get('start_codon_count', 0),
            'stop_codons': results.get('stop_codon_count', 0),
            'frameshifts': results.get('frameshift_count', 0),
            'saturation': results.get('saturation_level', 0),
            'hit_candidates': results.get('hit_candidates', [])
        }
    
    def add_clustering_analysis(self, clustering: Dict):
        """Add spatial clustering results"""
        self.report_data['clustering'] = {
            'total_clusters': clustering.get('total_clusters', 0),
            'cells_analyzed': clustering.get('cells_analyzed', 0),
            'phenotypes_detected': clustering.get('phenotypes', []),
            'spatial_patterns': clustering.get('spatial_patterns', {}),
            'cluster_quality': clustering.get('quality_metrics', {})
        }
    
    def generate_summary(self) -> Dict:
        """Generate executive summary"""
        summary = {
            'experiment_type': 'Optical Pooled CRISPR Screening',
            'timestamp': self.timestamp,
            'key_metrics': {}
        }
        
        if 'library' in self.report_data:
            summary['key_metrics']['total_barcodes'] = self.report_data['library']['total_barcodes']
        
        if 'quality_control' in self.report_data:
            qc = self.report_data['quality_control']
            summary['key_metrics']['success_rate'] = f"{qc['success_rate']:.1%}"
            summary['key_metrics']['main_failure_cause'] = max(
                qc.get('failure_causes', {}),
                key=lambda k: qc['failure_causes'].get(k, 0),
                default='Unknown'
            )
        
        if 'screening' in self.report_data:
            screen = self.report_data['screening']
            summary['key_metrics']['loss_of_function'] = f"{screen['loss_of_function']:.1%}"
            summary['key_metrics']['hit_candidates'] = len(screen.get('hit_candidates', []))
        
        if 'error_correction' in self.report_data:
            ec = self.report_data['error_correction']
            summary['key_metrics']['correction_rate'] = f"{ec['correction_rate']:.1%}"
        
        return summary
    
    def save_json_report(self, filename: str = None) -> Path:
        """Save report as JSON"""
        filename = filename or f"ops_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        output_path = self.output_dir / filename
        
        self.report_data['summary'] = self.generate_summary()
        
        with open(output_path, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def save_markdown_report(self, filename: str = None) -> Path:
        """Generate markdown report"""
        filename = filename or f"ops_report_{datetime.now():%Y%m%d_%H%M%S}.md"
        output_path = self.output_dir / filename
        
        lines = [
            "# Optical Pooled CRISPR Screening Report",
            f"Generated: {self.timestamp}",
            "",
            "## Executive Summary"
        ]
        
        summary = self.generate_summary()
        for key, value in summary.get('key_metrics', {}).items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        
        # Library Design
        if 'library' in self.report_data:
            lines.extend([
                "",
                "## Library Design",
                f"- Total Barcodes: {self.report_data['library']['total_barcodes']:,}",
                f"- Barcode Length: {self.report_data['library']['barcode_length']}bp",
                f"- Payload Length: {self.report_data['library']['payload_length']}bp",
                f"- Minimum Hamming Distance: {self.report_data['library']['hamming_distance']}"
            ])
        
        # Quality Control
        if 'quality_control' in self.report_data:
            qc = self.report_data['quality_control']
            lines.extend([
                "",
                "## Quality Control",
                f"- Success Rate: {qc['success_rate']:.1%}",
                "",
                "### Failure Analysis"
            ])
            for cause, rate in qc.get('failure_causes', {}).items():
                lines.append(f"- {cause.replace('_', ' ').title()}: {rate:.1%}")
        
        # Screening Results
        if 'screening' in self.report_data:
            screen = self.report_data['screening']
            lines.extend([
                "",
                "## Screening Results",
                f"- Loss of Function Rate: {screen['loss_of_function']:.1%}",
                f"- Start Codons: {screen['start_codons']:,}",
                f"- Stop Codons: {screen['stop_codons']:,}"
            ])
            if 'frameshift_count' in screen:
                lines.append(f"- Frameshifts: {screen['frameshift_count']:,}")
            lines.append(f"- Hit Candidates: {len(screen.get('hit_candidates', []))}")
        
        # Error Correction
        if 'error_correction' in self.report_data:
            ec = self.report_data['error_correction']
            lines.extend([
                "",
                "## Error Correction",
                f"- Sequences Corrected: {ec['sequences_corrected']:,}",
                f"- Consensus Sequences: {ec['consensus_sequences']:,}",
                f"- Correction Rate: {ec['correction_rate']:.1%}"
            ])
        
        # Recommendations
        if 'quality_control' in self.report_data:
            recs = self.report_data['quality_control'].get('recommendations', [])
            if recs:
                lines.extend([
                    "",
                    "## Recommendations"
                ])
                for rec in recs[:5]:
                    lines.append(f"- {rec}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Markdown report saved to {output_path}")
        return output_path
    
    def generate_tel_hai_format(self) -> Dict:
        """Generate report in Tel-Hai College format"""
        return {
            'institution': 'Tel-Hai College',
            'experiment_type': 'CRISPR Optical Pooled Screening',
            'date': self.timestamp,
            'barcode_library': {
                'size': self.report_data.get('library', {}).get('total_barcodes', 10000),
                'design': '14bp barcodes with 96bp payloads',
                'hamming_distance': 4
            },
            'sequencing': self.report_data.get('sequencing', {}),
            'quality_metrics': self.report_data.get('quality_control', {}),
            'screening_results': self.report_data.get('screening', {}),
            'summary': self.generate_summary()
        }


def main():
    """Test report generation"""
    report = OpticalScreeningReport()
    
    # Add test data
    report.add_experimental_metadata({
        'institution': 'Tel-Hai College',
        'date': '2023-05-11',
        'researcher': 'DNA Analysis Pipeline'
    })
    
    report.add_library_statistics({
        'total_barcodes': 10000,
        'barcode_length': 14,
        'payload_length': 96,
        'hamming_distance': 4
    })
    
    report.add_quality_control({
        'failure_rate': 0.555,
        'failure_diagnosis': {
            'pcr_amplification_bias': 0.17,
            'rna_degradation': 0.10,
            'cell_segmentation_errors': 0.05
        }
    })
    
    report.add_screening_results({
        'lof_percentage': 0.82,
        'start_codon_count': 8200,
        'stop_codon_count': 8400
    })
    
    # Generate reports
    json_path = report.save_json_report("test_report.json")
    md_path = report.save_markdown_report("test_report.md")
    
    print(f"✓ JSON report: {json_path}")
    print(f"✓ Markdown report: {md_path}")
    
    return json_path.exists() and md_path.exists()


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)