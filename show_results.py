#!/usr/bin/env python3
"""Display CRISPR analysis results with visualizations"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np

def create_dashboard():
    """Create a comprehensive dashboard of results"""
    
    results_dir = Path("/home/mch/dna/results")
    plots_dir = results_dir / "plots"
    reports_dir = results_dir / "reports"
    
    # Load latest results
    json_files = list(results_dir.glob("optimized_results_*.json"))
    if not json_files:
        print("No results found. Run the pipeline first.")
        return
    
    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_json) as f:
        data = json.load(f)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Tel-Hai CRISPR Screening Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Layout: 2x3 grid
    # [Summary Stats] [QC Metrics Plot]
    # [Key Findings ] [Barcode Dist Plot]
    # [Failure Analysis] [HTML Report Info]
    
    # 1. Summary Statistics (top left)
    ax1 = plt.subplot(2, 3, 1)
    ax1.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*30}
    Total Reads: {data['data_summary']['total_reads']:,}
    Unique Cells: {data['data_summary'].get('unique_cells', 'N/A'):,}
    Unique Barcodes: {data['data_summary']['unique_barcodes']:,}
    
    Failure Rate: {data['data_summary']['failure_rate']:.1%}
    Success Rate: {(1-data['data_summary']['failure_rate']):.1%}
    
    Runtime: {data['runtime_seconds']:.1f} seconds
    """
    
    ax1.text(0.1, 0.9, summary_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 2. QC Metrics Plot (top right)
    ax2 = plt.subplot(2, 3, 2)
    qc_png = plots_dir / "qc_metrics.png"
    if qc_png.exists():
        img = mpimg.imread(str(qc_png))
        ax2.imshow(img)
        ax2.axis('off')
        ax2.set_title('Quality Control Metrics')
    else:
        ax2.text(0.5, 0.5, 'QC Metrics\n(Run with full analysis)', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # 3. Key Findings (middle left)
    ax3 = plt.subplot(2, 3, 4)
    ax3.axis('off')
    
    if 'crispr_analysis' in data:
        crispr = data['crispr_analysis']
        findings_text = f"""
    KEY FINDINGS
    {'='*30}
    Loss of Function: {crispr['loss_of_function_rate']:.1%}
    Start Codons: {crispr['start_codons']}
    Stop Codons: {crispr['stop_codons']}
    
    Hit Candidates: {len(data.get('hit_candidates', []))}
    Error Correction: {data.get('error_correction', {}).get('correction_rate', 0):.1%}
    """
    else:
        findings_text = "No CRISPR analysis data"
    
    ax3.text(0.1, 0.9, findings_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # 4. Barcode Distribution Plot (middle right)
    ax4 = plt.subplot(2, 3, 5)
    barcode_png = plots_dir / "barcode_distribution.png"
    if barcode_png.exists():
        img = mpimg.imread(str(barcode_png))
        ax4.imshow(img)
        ax4.axis('off')
        ax4.set_title('Barcode Distribution')
    else:
        # Create a simple bar chart from data
        if 'clustering' in data and 'phenotype_clusters' in data['clustering']:
            enriched = data['clustering']['phenotype_clusters'].get('enriched_barcodes', {})
            if enriched:
                barcodes = list(enriched.keys())[:10]
                counts = [enriched[bc]['count'] for bc in barcodes]
                ax4.bar(range(len(barcodes)), counts)
                ax4.set_xlabel('Top Barcodes')
                ax4.set_ylabel('Count')
                ax4.set_title('Top Enriched Barcodes')
            else:
                ax4.text(0.5, 0.5, 'No enriched barcodes found', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_xticks([])
                ax4.set_yticks([])
        else:
            ax4.text(0.5, 0.5, 'Barcode Distribution\n(Run with full analysis)', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_xticks([])
            ax4.set_yticks([])
    
    # 5. Failure Analysis (bottom left)
    ax5 = plt.subplot(2, 3, 3)
    
    if 'quality_control' in data and 'failure_diagnosis' in data['quality_control']:
        failures = data['quality_control']['failure_diagnosis']
        non_zero = {k: v for k, v in failures.items() if v > 0}
        
        if non_zero:
            labels = [k.replace('_', ' ').title() for k in non_zero.keys()]
            sizes = list(non_zero.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
            
            ax5.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax5.set_title('Failure Cause Breakdown')
        else:
            ax5.text(0.5, 0.5, 'No failure analysis data', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_xticks([])
            ax5.set_yticks([])
    else:
        ax5.text(0.5, 0.5, 'Failure Analysis\n(Not available)', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_xticks([])
        ax5.set_yticks([])
    
    # 6. Report Links (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    report_text = f"""
    GENERATED REPORTS
    {'='*30}
    
    📊 HTML Report:
    {plots_dir}/analysis_report.html
    
    📄 Markdown Report:
    {reports_dir}/optimized_report.md
    
    📁 JSON Data:
    {latest_json.name}
    
    Pipeline: Optimized v2.0
    Using: DuckDB backend
    """
    
    ax6.text(0.1, 0.9, report_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    
    # Save and show
    dashboard_path = results_dir / "dashboard.png"
    plt.savefig(dashboard_path, dpi=100, bbox_inches='tight')
    print(f"\n✅ Dashboard saved to: {dashboard_path}")
    
    plt.show()
    
    # Also try to open HTML report
    html_report = plots_dir / "analysis_report.html"
    if html_report.exists():
        import webbrowser
        webbrowser.open(f"file://{html_report.absolute()}")
        print(f"📊 Opening HTML report in browser...")

if __name__ == "__main__":
    create_dashboard()