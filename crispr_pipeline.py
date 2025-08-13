#!/usr/bin/env python3
"""Main CRISPR screening analysis pipeline"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader, load_telhai_data
from src.crispr_analyzer import CRISPRScreenAnalyzer as CRISPRAnalyzer
from src.error_correction import BarcodeErrorCorrector as ErrorCorrector
from src.quality_control import QualityControlPipeline as QualityController
from src.visualization import CRISPRVisualizer
from src.report_generator import OpticalScreeningReport
from src.clustering_reconstruction import ClusterReconstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CRISPRPipeline:
    """Orchestrate CRISPR screening analysis"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("/home/mch/dna/results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.loader = DataLoader()
        self.analyzer = None  # Will be initialized with data
        self.corrector = None  # Will be initialized with reference barcodes
        self.qc = QualityController(expected_failure_rate=0.555)
        self.visualizer = CRISPRVisualizer(self.output_dir / "plots")
        self.reporter = OpticalScreeningReport(self.output_dir / "reports")
        self.reconstructor = ClusterReconstructor()
        
        self.results = {}
    
    def run_analysis(self, cluster_file: Path, design_file: Path,
                    full_analysis: bool = True) -> Dict:
        """Run complete analysis pipeline"""
        
        logger.info("Starting CRISPR screening analysis pipeline")
        start_time = datetime.now()
        
        # Phase 1: Load data
        logger.info("Phase 1: Loading data")
        cluster_df = self.loader.load(cluster_file)
        design_df = self.loader.load(design_file)
        
        self.results['data_summary'] = {
            'cluster_file': str(cluster_file),
            'design_file': str(design_file),
            'total_reads': len(cluster_df),
            'unique_barcodes': cluster_df['barcode'].nunique() if 'barcode' in cluster_df else 0,
            'design_barcodes': len(design_df)
        }
        
        # Initialize analyzer with data
        if 'barcode' in design_df:
            self.analyzer = CRISPRAnalyzer(barcode_length=14, min_hamming_dist=4)
            self.corrector = ErrorCorrector(design_df['barcode'].tolist())
        
        # Phase 2: Quality control
        logger.info("Phase 2: Running quality control")
        qc_report = self.qc.generate_qc_report(cluster_df, design_df['barcode'].tolist() if 'barcode' in design_df else [])
        self.results['quality_control'] = qc_report
        
        # Phase 3: Error correction
        logger.info("Phase 3: Performing error correction")
        if 'barcode' in cluster_df and self.corrector:
            barcodes_with_errors = cluster_df['barcode'].dropna().tolist()
            corrections = self.corrector.correct_barcode_errors(barcodes_with_errors)
            correction_results = {
                'corrected_count': sum(1 for k, v in corrections.items() if k != v),
                'total_barcodes': len(barcodes_with_errors),
                'correction_rate': sum(1 for k, v in corrections.items() if k != v) / len(barcodes_with_errors) if barcodes_with_errors else 0
            }
            self.results['error_correction'] = correction_results
        
        # Phase 4: CRISPR analysis
        logger.info("Phase 4: Analyzing CRISPR screening data")
        if 'sequence' in design_df and self.analyzer:
            sequences = design_df['sequence'].tolist()
            payloads = [seq[14:] if len(seq) > 14 else seq for seq in sequences]
            codon_analysis = self.analyzer.analyze_start_stop_codons(payloads)
            crispr_stats = {
                'loss_of_function_rate': codon_analysis['start_codon_freq'],
                'start_codons': sum(1 for p in payloads if p[:3] in ['ATG', 'GTG', 'TTG']),
                'stop_codons': sum(1 for p in payloads if p[:3] in ['TAA', 'TAG', 'TGA'])
            }
            self.results['crispr_analysis'] = crispr_stats
        
        # Phase 5: Clustering reconstruction
        logger.info("Phase 5: Reconstructing clusters")
        cluster_summary = self.reconstructor.generate_cluster_summary(
            cluster_df, design_df
        )
        self.results['clustering'] = cluster_summary
        
        # Phase 6: Hit identification
        logger.info("Phase 6: Identifying screening hits")
        if 'phenotype_clusters' in cluster_summary:
            hits = self.reconstructor.identify_hit_candidates(
                cluster_summary['phenotype_clusters']
            )
            self.results['hit_candidates'] = hits
        
        if full_analysis:
            # Phase 7: Generate visualizations
            logger.info("Phase 7: Creating visualizations")
            self._generate_visualizations()
            
            # Phase 8: Generate reports
            logger.info("Phase 8: Generating reports")
            self._generate_reports()
        
        # Calculate runtime
        runtime = (datetime.now() - start_time).total_seconds()
        self.results['runtime_seconds'] = runtime
        
        logger.info(f"Pipeline completed in {runtime:.1f} seconds")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _generate_visualizations(self):
        """Generate all visualizations"""
        
        # Barcode distribution
        if 'clustering' in self.results:
            barcode_counts = {}
            if 'phenotype_clusters' in self.results['clustering']:
                enriched = self.results['clustering']['phenotype_clusters'].get('enriched_barcodes', {})
                for bc, stats in enriched.items():
                    barcode_counts[bc] = stats['count']
                
                if barcode_counts:
                    self.visualizer.plot_barcode_distribution(barcode_counts)
        
        # Quality metrics
        if 'quality_control' in self.results:
            self.visualizer.plot_quality_metrics(self.results['quality_control'])
        
        # Generate HTML report with plots
        self.visualizer.generate_html_report(self.results)
    
    def _generate_reports(self):
        """Generate comprehensive reports"""
        
        # Add metadata
        self.reporter.add_experimental_metadata({
            'pipeline_version': '2.0',
            'analysis_date': datetime.now().isoformat()
        })
        
        # Add library stats
        if 'data_summary' in self.results:
            self.reporter.add_library_statistics({
                'total_barcodes': self.results['data_summary']['design_barcodes']
            })
        
        # Add QC results
        if 'quality_control' in self.results:
            self.reporter.add_quality_control(self.results['quality_control'])
        
        # Add error correction
        if 'error_correction' in self.results:
            self.reporter.add_error_correction(self.results['error_correction'])
        
        # Add screening results
        if 'crispr_analysis' in self.results:
            analysis = self.results['crispr_analysis']
            self.reporter.add_screening_results({
                'lof_percentage': analysis.get('loss_of_function_rate', 0),
                'start_codon_count': analysis.get('start_codons', 0),
                'stop_codon_count': analysis.get('stop_codons', 0)
            })
        
        # Add clustering
        if 'clustering' in self.results:
            self.reporter.add_clustering_analysis({
                'total_clusters': self.results['clustering'].get('unique_cells', 0)
            })
        
        # Save reports
        self.reporter.save_json_report("pipeline_report.json")
        self.reporter.save_markdown_report("pipeline_report.md")
    
    def _save_results(self):
        """Save pipeline results"""
        output_file = self.output_dir / f"pipeline_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save summary
        summary_file = self.output_dir / "latest_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"CRISPR Pipeline Analysis Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Runtime: {self.results.get('runtime_seconds', 0):.1f} seconds\n\n")
            
            if 'data_summary' in self.results:
                ds = self.results['data_summary']
                f.write(f"Data:\n")
                f.write(f"  Total reads: {ds['total_reads']:,}\n")
                f.write(f"  Unique barcodes: {ds['unique_barcodes']:,}\n\n")
            
            if 'quality_control' in self.results:
                qc = self.results['quality_control']
                if 'summary' in qc:
                    f.write(f"Quality Control:\n")
                    f.write(f"  Failure rate: {qc['summary'].get('failure_rate', 0):.1%}\n\n")
            
            if 'hit_candidates' in self.results:
                f.write(f"Screening Results:\n")
                f.write(f"  Hit candidates: {len(self.results['hit_candidates'])}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='CRISPR Screening Analysis Pipeline')
    parser.add_argument('--clusters', type=Path, 
                       default=Path("/home/mch/dna/DNA-Data for Telhai/2023-05-11/clusters.csv"),
                       help='Path to clusters CSV file')
    parser.add_argument('--design', type=Path,
                       default=Path("/home/mch/dna/updated_data/micro_design.csv"),
                       help='Path to design CSV file')
    parser.add_argument('--output', type=Path,
                       default=Path("/home/mch/dna/results"),
                       help='Output directory')
    parser.add_argument('--quick', action='store_true',
                       help='Skip visualization and report generation')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = CRISPRPipeline(args.output)
    results = pipeline.run_analysis(
        args.clusters,
        args.design,
        full_analysis=not args.quick
    )
    
    # Print summary
    print("\nAnalysis Complete!")
    print(f"Results saved to: {args.output}")
    print(f"Runtime: {results.get('runtime_seconds', 0):.1f} seconds")
    
    if 'hit_candidates' in results:
        print(f"Hit candidates identified: {len(results['hit_candidates'])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())