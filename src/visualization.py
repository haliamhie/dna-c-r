#!/usr/bin/env python3
"""Visualization module for CRISPR screening analysis"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("Matplotlib/Seaborn not available. Text-based visualizations only.")


class CRISPRVisualizer:
    """Visualization tools for CRISPR screening data"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("/home/mch/dna/reports")
        self.output_dir.mkdir(exist_ok=True)
        
        if HAS_PLOTTING:
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_barcode_distribution(self, barcode_counts: Dict[str, int], 
                                 title: str = "Barcode Frequency Distribution") -> Optional[Path]:
        """Plot barcode frequency distribution"""
        if not HAS_PLOTTING:
            return self._text_histogram(barcode_counts, title)
        
        counts = list(barcode_counts.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(counts, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Read Count')
        ax1.set_ylabel('Number of Barcodes')
        ax1.set_title('Read Count Distribution')
        ax1.set_yscale('log')
        
        # Cumulative distribution
        sorted_counts = sorted(counts, reverse=True)
        cumsum = np.cumsum(sorted_counts) / np.sum(sorted_counts)
        ax2.plot(range(len(cumsum)), cumsum)
        ax2.set_xlabel('Barcode Rank')
        ax2.set_ylabel('Cumulative Fraction of Reads')
        ax2.set_title('Cumulative Read Distribution')
        ax2.set_xscale('log')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        output_path = self.output_dir / "barcode_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_hamming_distances(self, distances: List[int], 
                              min_required: int = 4) -> Optional[Path]:
        """Plot Hamming distance distribution"""
        if not HAS_PLOTTING:
            return self._text_stats(distances, "Hamming Distances")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        unique, counts = np.unique(distances, return_counts=True)
        colors = ['red' if d < min_required else 'green' for d in unique]
        
        ax.bar(unique, counts, color=colors, edgecolor='black')
        ax.axvline(x=min_required, color='blue', linestyle='--', 
                  label=f'Min Required = {min_required}')
        ax.set_xlabel('Hamming Distance')
        ax.set_ylabel('Frequency')
        ax.set_title('Hamming Distance Distribution Between Barcodes')
        ax.legend()
        
        output_path = self.output_dir / "hamming_distances.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_quality_metrics(self, qc_report: Dict) -> Optional[Path]:
        """Create quality control visualization"""
        if not HAS_PLOTTING:
            return self._save_text_report(qc_report, "qc_metrics.txt")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Failure causes pie chart
        if 'failure_diagnosis' in qc_report:
            failures = qc_report['failure_diagnosis']
            non_zero = {k: v for k, v in failures.items() if v > 0}
            if non_zero:
                axes[0, 0].pie(non_zero.values(), labels=non_zero.keys(), 
                             autopct='%1.1f%%')
                axes[0, 0].set_title('Failure Cause Breakdown')
        
        # Coverage statistics
        if 'coverage_statistics' in qc_report:
            stats = qc_report['coverage_statistics']
            metrics = ['mean_coverage', 'median_coverage', 'std_coverage']
            values = [stats.get(m, 0) for m in metrics]
            axes[0, 1].bar(metrics, values)
            axes[0, 1].set_title('Coverage Statistics')
            axes[0, 1].set_ylabel('Reads')
        
        # Success/Failure rate
        if 'summary' in qc_report:
            summary = qc_report['summary']
            sizes = [summary.get('valid_barcodes', 0), 
                    summary.get('failed_barcodes', 0)]
            axes[1, 0].pie(sizes, labels=['Valid', 'Failed'], 
                         colors=['green', 'red'],
                         autopct='%1.1f%%')
            axes[1, 0].set_title('Overall Success Rate')
        
        # Text summary
        axes[1, 1].axis('off')
        summary_text = self._format_summary(qc_report)
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                       verticalalignment='center')
        
        plt.suptitle('Quality Control Metrics', fontsize=14)
        plt.tight_layout()
        
        output_path = self.output_dir / "qc_metrics.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_nucleotide_composition(self, sequences: List[str]) -> Optional[Path]:
        """Visualize nucleotide composition"""
        if not sequences:
            return None
        
        if not HAS_PLOTTING:
            return self._text_composition(sequences)
        
        # Calculate composition
        length = len(sequences[0]) if sequences else 0
        position_counts = []
        
        for pos in range(length):
            counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
            for seq in sequences:
                if len(seq) > pos:
                    base = seq[pos]
                    if base in counts:
                        counts[base] += 1
            position_counts.append(counts)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        positions = range(length)
        bottom = np.zeros(length)
        colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
        
        for base in ['A', 'C', 'G', 'T']:
            values = [pc[base] for pc in position_counts]
            ax.bar(positions, values, bottom=bottom, 
                  label=base, color=colors[base], alpha=0.8)
            bottom += values
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Count')
        ax.set_title('Nucleotide Composition by Position')
        ax.legend()
        
        output_path = self.output_dir / "nucleotide_composition.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_html_report(self, analysis_results: Dict) -> Path:
        """Generate HTML report with all visualizations"""
        html_content = [
            "<html><head><title>CRISPR Screening Analysis Report</title>",
            "<style>body {font-family: Arial; margin: 20px;}",
            "img {max-width: 800px; margin: 10px 0;}",
            "table {border-collapse: collapse; margin: 10px 0;}",
            "th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}",
            "</style></head><body>",
            "<h1>CRISPR Screening Analysis Report</h1>"
        ]
        
        # Summary section
        if 'summary' in analysis_results:
            html_content.append("<h2>Summary</h2>")
            html_content.append("<table>")
            for key, value in analysis_results['summary'].items():
                html_content.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            html_content.append("</table>")
        
        # Quality metrics
        if 'qc_report' in analysis_results:
            html_content.append("<h2>Quality Control</h2>")
            qc = analysis_results['qc_report']
            if 'summary' in qc:
                html_content.append("<h3>QC Summary</h3><table>")
                for key, value in qc['summary'].items():
                    html_content.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                html_content.append("</table>")
        
        # Add plots if they exist
        for plot_file in self.output_dir.glob("*.png"):
            html_content.append(f"<h2>{plot_file.stem.replace('_', ' ').title()}</h2>")
            html_content.append(f"<img src='{plot_file.name}' />")
        
        html_content.append("</body></html>")
        
        output_path = self.output_dir / "analysis_report.html"
        with open(output_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        return output_path
    
    def _text_histogram(self, data: Dict, title: str) -> Path:
        """Create text-based histogram"""
        counts = list(data.values())
        hist, bins = np.histogram(counts, bins=10)
        
        output = [title, "=" * 50]
        max_count = max(hist)
        
        for i, count in enumerate(hist):
            bar_length = int(40 * count / max_count) if max_count > 0 else 0
            bar = "*" * bar_length
            label = f"{bins[i]:.0f}-{bins[i+1]:.0f}"
            output.append(f"{label:10s} | {bar} ({count})")
        
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(output))
        
        return output_path
    
    def _text_stats(self, data: List, title: str) -> Path:
        """Generate text statistics"""
        output = [
            title,
            "=" * 50,
            f"Count: {len(data)}",
            f"Mean: {np.mean(data):.2f}",
            f"Median: {np.median(data):.2f}",
            f"Std: {np.std(data):.2f}",
            f"Min: {min(data)}",
            f"Max: {max(data)}"
        ]
        
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(output))
        
        return output_path
    
    def _text_composition(self, sequences: List[str]) -> Path:
        """Text-based nucleotide composition"""
        total_bases = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        
        for seq in sequences:
            for base in seq:
                if base in total_bases:
                    total_bases[base] += 1
        
        total = sum(total_bases.values())
        output = ["Nucleotide Composition", "=" * 50]
        
        for base, count in total_bases.items():
            pct = 100 * count / total if total > 0 else 0
            bar_length = int(40 * pct / 100)
            bar = "*" * bar_length
            output.append(f"{base}: {bar} {pct:.1f}% ({count})")
        
        output_path = self.output_dir / "nucleotide_composition.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(output))
        
        return output_path
    
    def _save_text_report(self, data: Dict, filename: str) -> Path:
        """Save dictionary as formatted text"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return output_path
    
    def _format_summary(self, qc_report: Dict) -> str:
        """Format QC report as summary text"""
        lines = []
        if 'summary' in qc_report:
            s = qc_report['summary']
            lines.append(f"Total Reads: {s.get('total_reads', 0):,}")
            lines.append(f"Valid: {s.get('valid_barcodes', 0):,}")
            lines.append(f"Failed: {s.get('failed_barcodes', 0):,}")
            lines.append(f"Failure Rate: {s.get('failure_rate', 0):.1%}")
        
        if 'recommendations' in qc_report:
            lines.append("\nRecommendations:")
            for rec in qc_report['recommendations'][:3]:
                lines.append(f"• {rec[:50]}...")
        
        return '\n'.join(lines)


def main():
    """Test visualization functionality"""
    viz = CRISPRVisualizer()
    
    # Test data
    test_counts = {f"BC{i:04d}": np.random.poisson(100) for i in range(100)}
    test_distances = [4, 5, 6, 4, 7, 3, 8, 5, 6, 7]
    
    # Generate visualizations
    viz.plot_barcode_distribution(test_counts)
    viz.plot_hamming_distances(test_distances)
    
    print(f"Visualizations saved to {viz.output_dir}")


if __name__ == "__main__":
    main()