#!/usr/bin/env python3
"""Optimized CRISPR screening analysis pipeline"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Optional
import duckdb
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
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


class OptimizedCRISPRPipeline:
    """Optimized CRISPR screening analysis for large datasets"""
    
    def __init__(self, output_dir: Path = None, use_duckdb: bool = True):
        self.output_dir = output_dir or Path("/home/mch/dna/results")
        self.output_dir.mkdir(exist_ok=True)
        self.use_duckdb = use_duckdb
        
        # Initialize components
        self.loader = DataLoader()
        self.analyzer = None
        self.corrector = None
        self.qc = QualityController(expected_failure_rate=0.555)
        self.visualizer = CRISPRVisualizer(self.output_dir / "plots")
        self.reporter = OpticalScreeningReport(self.output_dir / "reports")
        self.reconstructor = ClusterReconstructor()
        
        self.results = {}
        self.con = None  # DuckDB connection
    
    def run_analysis(self, cluster_file: Path, design_file: Path,
                    sample_size: Optional[int] = None,
                    full_analysis: bool = True) -> Dict:
        """Run optimized analysis pipeline"""
        
        logger.info("Starting OPTIMIZED CRISPR screening analysis")
        start_time = datetime.now()
        
        # Use DuckDB for large files
        if self.use_duckdb and cluster_file.suffix == '.csv' and cluster_file.stat().st_size > 100_000_000:
            logger.info("Using DuckDB backend for large file processing")
            results = self._run_duckdb_analysis(cluster_file, design_file, sample_size, full_analysis)
        else:
            results = self._run_pandas_analysis(cluster_file, design_file, sample_size, full_analysis)
        
        # Calculate runtime
        runtime = (datetime.now() - start_time).total_seconds()
        self.results['runtime_seconds'] = runtime
        
        logger.info(f"Pipeline completed in {runtime:.1f} seconds")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _run_duckdb_analysis(self, cluster_file: Path, design_file: Path,
                            sample_size: Optional[int], full_analysis: bool) -> Dict:
        """Run analysis using DuckDB for better performance"""
        
        # Check if DuckDB database exists
        db_path = Path("/home/mch/dna/artifacts/dna.duckdb")
        if db_path.exists():
            logger.info("Using existing DuckDB database")
            self.con = duckdb.connect(str(db_path))
        else:
            logger.info("Creating temporary DuckDB database")
            self.con = duckdb.connect(':memory:')
            
            # Load clusters into DuckDB
            logger.info("Loading clusters into DuckDB...")
            self.con.execute(f"""
                CREATE TABLE clusters AS 
                SELECT * FROM read_csv_auto('{cluster_file}', 
                    sample_size={sample_size if sample_size else -1})
            """)
        
        # Load design data
        design_df = pd.read_csv(design_file)
        
        # Get statistics using SQL
        logger.info("Phase 1: Computing statistics with SQL")
        stats = self.con.execute("""
            SELECT 
                COUNT(*) as total_reads,
                COUNT(DISTINCT id) as unique_cells,
                COUNT(DISTINCT barcode) as unique_barcodes,
                COUNT(barcode) as reads_with_barcodes,
                COUNT(*) - COUNT(barcode) as failed_reads
            FROM clusters
        """).fetchone()
        
        self.results['data_summary'] = {
            'total_reads': stats[0],
            'unique_cells': stats[1],
            'unique_barcodes': stats[2],
            'reads_with_barcodes': stats[3],
            'failed_reads': stats[4],
            'failure_rate': stats[4] / stats[0] if stats[0] > 0 else 0
        }
        
        # Phase 2: Quality control with sampling
        logger.info("Phase 2: Quality control with intelligent sampling")
        sample_df = self.con.execute("""
            SELECT * FROM clusters 
            USING SAMPLE 10000 ROWS
        """).fetchdf()
        
        if 'barcode' in design_df:
            self.analyzer = CRISPRAnalyzer(barcode_length=14, min_hamming_dist=4)
            self.corrector = ErrorCorrector(design_df['barcode'].tolist())
        
        qc_report = self.qc.generate_qc_report(sample_df, design_df['barcode'].tolist() if 'barcode' in design_df else [])
        qc_report['summary']['total_reads'] = stats[0]  # Use actual total
        qc_report['summary']['failure_rate'] = stats[4] / stats[0] if stats[0] > 0 else 0
        self.results['quality_control'] = qc_report
        
        # Phase 3: Barcode frequency analysis (optimized)
        logger.info("Phase 3: Optimized barcode frequency analysis")
        barcode_freq = self.con.execute("""
            SELECT 
                barcode,
                COUNT(*) as count,
                COUNT(DISTINCT id) as unique_cells
            FROM clusters
            WHERE barcode IS NOT NULL
            GROUP BY barcode
            ORDER BY count DESC
            LIMIT 1000
        """).fetchdf()
        
        # Calculate enrichment statistics
        if len(barcode_freq) > 0:
            mean_count = barcode_freq['count'].mean()
            std_count = barcode_freq['count'].std()
            
            enriched = {}
            depleted = {}
            
            for _, row in barcode_freq.iterrows():
                z_score = (row['count'] - mean_count) / std_count if std_count > 0 else 0
                
                if z_score > 2:
                    enriched[row['barcode']] = {
                        'count': row['count'],
                        'z_score': z_score,
                        'fold_change': row['count'] / mean_count
                    }
                elif z_score < -2:
                    depleted[row['barcode']] = {
                        'count': row['count'],
                        'z_score': z_score,
                        'fold_change': row['count'] / mean_count
                    }
            
            self.results['clustering'] = {
                'phenotype_clusters': {
                    'enriched_barcodes': enriched,
                    'depleted_barcodes': depleted,
                    'mean_barcode_count': mean_count,
                    'std_barcode_count': std_count
                },
                'unique_cells': stats[1],
                'unique_barcodes': stats[2]
            }
            
            # Identify hits
            hits = self.reconstructor.identify_hit_candidates(
                self.results['clustering']['phenotype_clusters']
            )
            self.results['hit_candidates'] = hits
        
        # Phase 4: CRISPR analysis
        if 'sequence' in design_df and self.analyzer:
            logger.info("Phase 4: CRISPR screening analysis")
            sequences = design_df['sequence'].tolist()
            payloads = [seq[14:] if len(seq) > 14 else seq for seq in sequences]
            codon_analysis = self.analyzer.analyze_start_stop_codons(payloads)
            self.results['crispr_analysis'] = {
                'loss_of_function_rate': codon_analysis['start_codon_freq'],
                'start_codons': sum(1 for p in payloads if p[:3] in ['ATG', 'GTG', 'TTG']),
                'stop_codons': sum(1 for p in payloads if p[:3] in ['TAA', 'TAG', 'TGA'])
            }
        
        # Error correction on sample
        if self.corrector:
            logger.info("Phase 5: Error correction on sample")
            sample_barcodes = sample_df['barcode'].dropna().head(1000).tolist()
            corrections = self.corrector.correct_barcode_errors(sample_barcodes)
            self.results['error_correction'] = {
                'sample_size': len(sample_barcodes),
                'corrected_count': sum(1 for k, v in corrections.items() if k != v),
                'correction_rate': sum(1 for k, v in corrections.items() if k != v) / len(sample_barcodes) if sample_barcodes else 0
            }
        
        if full_analysis:
            self._generate_visualizations()
            self._generate_reports()
        
        # Close connection
        if self.con:
            self.con.close()
        
        return self.results
    
    def _run_pandas_analysis(self, cluster_file: Path, design_file: Path,
                           sample_size: Optional[int], full_analysis: bool) -> Dict:
        """Run standard pandas-based analysis"""
        
        logger.info("Phase 1: Loading data")
        
        # Load with sampling if specified
        if sample_size and cluster_file.suffix == '.csv':
            # Read header to get column names
            header_df = pd.read_csv(cluster_file, nrows=0)
            # Calculate skip pattern for sampling
            total_rows = sum(1 for _ in open(cluster_file)) - 1  # Subtract header
            skip_rows = max(1, total_rows // sample_size) if sample_size < total_rows else 1
            cluster_df = pd.read_csv(cluster_file, skiprows=lambda i: i > 0 and i % skip_rows != 0)
            logger.info(f"Sampled {len(cluster_df)} rows from {total_rows} total")
        else:
            cluster_df = self.loader.load(cluster_file)
        
        design_df = self.loader.load(design_file)
        
        # Continue with standard analysis...
        self.results['data_summary'] = {
            'total_reads': len(cluster_df),
            'unique_barcodes': cluster_df['barcode'].nunique() if 'barcode' in cluster_df else 0,
            'design_barcodes': len(design_df)
        }
        
        # Initialize components
        if 'barcode' in design_df:
            self.analyzer = CRISPRAnalyzer(barcode_length=14, min_hamming_dist=4)
            self.corrector = ErrorCorrector(design_df['barcode'].tolist())
        
        # Run QC
        logger.info("Phase 2: Quality control")
        qc_report = self.qc.generate_qc_report(cluster_df, design_df['barcode'].tolist() if 'barcode' in design_df else [])
        self.results['quality_control'] = qc_report
        
        # Clustering (optimized)
        logger.info("Phase 3: Optimized clustering")
        if len(cluster_df) > 100000:
            # Use sampling for large datasets
            sample_df = cluster_df.sample(min(100000, len(cluster_df)))
            cluster_summary = self.reconstructor.generate_cluster_summary(sample_df, design_df)
            cluster_summary['note'] = 'Based on 100k sample'
        else:
            cluster_summary = self.reconstructor.generate_cluster_summary(cluster_df, design_df)
        
        self.results['clustering'] = cluster_summary
        
        # CRISPR analysis
        if 'sequence' in design_df and self.analyzer:
            logger.info("Phase 4: CRISPR analysis")
            sequences = design_df['sequence'].tolist()
            payloads = [seq[14:] if len(seq) > 14 else seq for seq in sequences]
            codon_analysis = self.analyzer.analyze_start_stop_codons(payloads)
            self.results['crispr_analysis'] = {
                'loss_of_function_rate': codon_analysis['start_codon_freq'],
                'start_codons': sum(1 for p in payloads if p[:3] in ['ATG', 'GTG', 'TTG']),
                'stop_codons': sum(1 for p in payloads if p[:3] in ['TAA', 'TAG', 'TGA'])
            }
        
        if full_analysis:
            self._generate_visualizations()
            self._generate_reports()
        
        return self.results
    
    def _generate_visualizations(self):
        """Generate visualizations"""
        if 'clustering' in self.results:
            barcode_counts = {}
            if 'phenotype_clusters' in self.results['clustering']:
                enriched = self.results['clustering']['phenotype_clusters'].get('enriched_barcodes', {})
                for bc, stats in enriched.items():
                    barcode_counts[bc] = stats['count']
                
                if barcode_counts:
                    self.visualizer.plot_barcode_distribution(barcode_counts)
        
        if 'quality_control' in self.results:
            self.visualizer.plot_quality_metrics(self.results['quality_control'])
        
        html_path = self.visualizer.generate_html_report(self.results)
        
        # Auto-open visualization
        self._open_visualization(html_path)
    
    def _open_visualization(self, html_path: Path):
        """Automatically open visualization in browser"""
        import webbrowser
        import platform
        import subprocess
        
        if html_path and html_path.exists():
            url = f"file://{html_path.absolute()}"
            
            # Try different methods to open
            try:
                # Method 1: webbrowser (most portable)
                webbrowser.open(url)
                logger.info(f"Opening visualization: {url}")
            except:
                # Method 2: system-specific commands
                system = platform.system()
                try:
                    if system == "Darwin":  # macOS
                        subprocess.run(["open", str(html_path)])
                    elif system == "Linux":
                        # Try common Linux browsers
                        for browser in ["xdg-open", "firefox", "chromium", "google-chrome"]:
                            try:
                                subprocess.run([browser, str(html_path)], check=False)
                                break
                            except FileNotFoundError:
                                continue
                    elif system == "Windows":
                        subprocess.run(["start", "", str(html_path)], shell=True)
                except:
                    pass
            
            # Also display the image files directly if matplotlib available
            try:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                
                # Find and display PNG files
                png_files = list(self.visualizer.output_dir.glob("*.png"))
                if png_files:
                    fig, axes = plt.subplots(1, len(png_files), figsize=(15, 6))
                    if len(png_files) == 1:
                        axes = [axes]
                    
                    for ax, png_file in zip(axes, png_files[:3]):  # Max 3 images
                        img = mpimg.imread(str(png_file))
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(png_file.stem.replace('_', ' ').title())
                    
                    plt.tight_layout()
                    plt.show(block=False)  # Non-blocking show
                    logger.info("Displaying visualization plots")
            except:
                pass
    
    def _generate_reports(self):
        """Generate reports"""
        self.reporter.add_experimental_metadata({
            'pipeline_version': '2.0-optimized',
            'analysis_date': datetime.now().isoformat()
        })
        
        if 'data_summary' in self.results:
            self.reporter.add_library_statistics({
                'total_barcodes': self.results['data_summary'].get('unique_barcodes', 0)
            })
        
        if 'quality_control' in self.results:
            self.reporter.add_quality_control(self.results['quality_control'])
        
        if 'error_correction' in self.results:
            self.reporter.add_error_correction(self.results['error_correction'])
        
        if 'crispr_analysis' in self.results:
            self.reporter.add_screening_results(self.results['crispr_analysis'])
        
        self.reporter.save_json_report("optimized_report.json")
        self.reporter.save_markdown_report("optimized_report.md")
    
    def _save_results(self):
        """Save pipeline results"""
        output_file = self.output_dir / f"optimized_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Optimized CRISPR Screening Analysis')
    parser.add_argument('--clusters', type=Path, 
                       default=Path("/home/mch/dna/DNA-Data for Telhai/2023-05-11/clusters.csv"),
                       help='Path to clusters CSV file')
    parser.add_argument('--design', type=Path,
                       default=Path("/home/mch/dna/updated_data/micro_design.csv"),
                       help='Path to design CSV file')
    parser.add_argument('--output', type=Path,
                       default=Path("/home/mch/dna/results"),
                       help='Output directory')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for quick analysis (e.g., 100000)')
    parser.add_argument('--no-duckdb', action='store_true',
                       help='Disable DuckDB backend')
    parser.add_argument('--quick', action='store_true',
                       help='Skip visualization and reports')
    
    args = parser.parse_args()
    
    # Run optimized pipeline
    pipeline = OptimizedCRISPRPipeline(
        args.output,
        use_duckdb=not args.no_duckdb
    )
    
    results = pipeline.run_analysis(
        args.clusters,
        args.design,
        sample_size=args.sample,
        full_analysis=not args.quick
    )
    
    # Print summary
    print("\n" + "="*50)
    print("OPTIMIZED ANALYSIS COMPLETE")
    print("="*50)
    print(f"Runtime: {results.get('runtime_seconds', 0):.1f} seconds")
    print(f"Total reads: {results['data_summary']['total_reads']:,}")
    print(f"Failure rate: {results['data_summary'].get('failure_rate', 0):.1%}")
    
    if 'hit_candidates' in results:
        print(f"Hit candidates: {len(results['hit_candidates'])}")
    
    print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())