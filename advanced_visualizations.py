#!/usr/bin/env python3
"""Advanced visualizations for CRISPR screening data"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
from collections import Counter
import logomaker
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class AdvancedCRISPRVisualizer:
    """Create detailed structural and statistical visualizations"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("/home/mch/dna/visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_sequence_logo(self, sequences, title="DNA Sequence Logo"):
        """Create sequence logo showing conservation patterns"""
        
        if not sequences:
            return None
        
        # Convert sequences to position frequency matrix
        seq_length = len(sequences[0])
        matrix = np.zeros((seq_length, 4))  # ACGT
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        for seq in sequences:
            for pos, base in enumerate(seq[:seq_length]):
                if base in base_map:
                    matrix[pos, base_map[base]] += 1
        
        # Normalize to frequencies
        matrix = matrix / len(sequences)
        
        # Create DataFrame for logomaker
        df = pd.DataFrame(matrix, columns=['A', 'C', 'G', 'T'])
        
        # Calculate information content
        info_content = []
        for i in range(len(df)):
            row = df.iloc[i]
            # Shannon entropy
            entropy = -sum([p * np.log2(p) if p > 0 else 0 for p in row])
            info = 2 - entropy  # Max 2 bits for DNA
            info_content.append(info)
            # Scale heights by information content
            df.iloc[i] = row * info
        
        # Create logo
        fig, ax = plt.subplots(figsize=(16, 4))
        logo = logomaker.Logo(df, ax=ax, color_scheme='classic')
        ax.set_ylabel('Information Content (bits)')
        ax.set_xlabel('Position')
        ax.set_title(title)
        
        # Add conservation score
        avg_conservation = np.mean(info_content)
        ax.text(0.02, 0.98, f'Avg Conservation: {avg_conservation:.2f} bits',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        output_path = self.output_dir / "sequence_logo.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_hamming_distance_heatmap(self, barcodes, sample_size=100):
        """Create heatmap of Hamming distances between barcodes"""
        
        # Sample if too many
        if len(barcodes) > sample_size:
            barcodes = np.random.choice(barcodes, sample_size, replace=False)
        
        n = len(barcodes)
        distance_matrix = np.zeros((n, n))
        
        # Calculate Hamming distances
        for i in range(n):
            for j in range(i+1, n):
                dist = sum(c1 != c2 for c1, c2 in zip(barcodes[i], barcodes[j]))
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Create clustered heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Cluster barcodes by similarity
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        dendro = dendrogram(linkage_matrix, no_plot=True)
        order = dendro['leaves']
        
        # Reorder matrix
        distance_matrix_ordered = distance_matrix[order, :][:, order]
        
        # Plot heatmap
        sns.heatmap(distance_matrix_ordered, cmap='viridis', 
                   cbar_kws={'label': 'Hamming Distance'},
                   square=True, ax=ax)
        
        ax.set_title(f'Hamming Distance Matrix ({n} barcodes)')
        ax.set_xlabel('Barcode Index')
        ax.set_ylabel('Barcode Index')
        
        # Add statistics
        avg_dist = np.mean(distance_matrix[np.triu_indices(n, k=1)])
        min_dist = np.min(distance_matrix[np.triu_indices(n, k=1)])
        
        stats_text = f'Avg Distance: {avg_dist:.1f}\nMin Distance: {min_dist:.0f}'
        ax.text(1.02, 0.5, stats_text, transform=ax.transAxes,
               va='center', bbox=dict(boxstyle='round', facecolor='white'))
        
        output_path = self.output_dir / "hamming_heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_3d_barcode_network(self, barcodes, distance_threshold=4):
        """Create 3D network of barcode relationships"""
        
        # Sample for performance
        if len(barcodes) > 50:
            barcodes = list(np.random.choice(barcodes, 50, replace=False))
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, bc in enumerate(barcodes):
            G.add_node(i, barcode=bc)
        
        # Add edges based on Hamming distance
        for i in range(len(barcodes)):
            for j in range(i+1, len(barcodes)):
                dist = sum(c1 != c2 for c1, c2 in zip(barcodes[i], barcodes[j]))
                if dist <= distance_threshold:
                    G.add_edge(i, j, weight=1/dist if dist > 0 else 1)
        
        # 3D layout using spring layout
        pos = nx.spring_layout(G, dim=3, weight='weight', iterations=50)
        
        # Extract positions
        edge_trace = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_trace.append(go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none'
            ))
        
        # Node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]
        
        # Color by degree
        node_degrees = [G.degree(node) for node in G.nodes()]
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=10,
                color=node_degrees,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Connections"),
                line=dict(width=2, color='white')
            ),
            text=[f"BC{i}" for i in range(len(barcodes))],
            textposition="top center",
            hovertemplate='Barcode: %{text}<br>Connections: %{marker.color}<extra></extra>'
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title=f"3D Barcode Network (Hamming distance ≤ {distance_threshold})",
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, showticklabels=False, title=''),
                bgcolor='rgb(240,240,240)'
            ),
            height=700
        )
        
        output_path = self.output_dir / "barcode_network_3d.html"
        fig.write_html(str(output_path))
        
        return output_path
    
    def create_mutation_pattern_analysis(self, reference_seq, variant_seqs):
        """Analyze and visualize mutation patterns"""
        
        # Track mutations by position and type
        mutations = {'substitution': [], 'insertion': [], 'deletion': []}
        position_mutations = Counter()
        
        for variant in variant_seqs[:1000]:  # Sample for performance
            # Simple mutation detection
            for i, (ref, var) in enumerate(zip(reference_seq, variant)):
                if ref != var:
                    mutations['substitution'].append(i)
                    position_mutations[i] += 1
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mutation Frequency by Position',
                'Mutation Type Distribution',
                'Nucleotide Transition Matrix',
                'Mutation Hotspots'
            ),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'heatmap'}, {'type': 'scatter'}]]
        )
        
        # 1. Mutation frequency by position
        positions = list(range(len(reference_seq)))
        frequencies = [position_mutations.get(i, 0) for i in positions]
        
        fig.add_trace(
            go.Bar(x=positions, y=frequencies, name='Mutations',
                  marker_color='indianred'),
            row=1, col=1
        )
        
        # 2. Mutation type pie chart
        mut_types = ['Substitution', 'Insertion', 'Deletion']
        mut_counts = [len(mutations['substitution']), 
                     len(mutations['insertion']), 
                     len(mutations['deletion'])]
        
        fig.add_trace(
            go.Pie(labels=mut_types, values=mut_counts,
                  hole=0.3),
            row=1, col=2
        )
        
        # 3. Nucleotide transition matrix
        transitions = np.zeros((4, 4))
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        bases = ['A', 'C', 'G', 'T']
        
        for variant in variant_seqs[:100]:
            for ref, var in zip(reference_seq, variant):
                if ref in base_map and var in base_map and ref != var:
                    transitions[base_map[ref], base_map[var]] += 1
        
        fig.add_trace(
            go.Heatmap(z=transitions, x=bases, y=bases,
                      colorscale='Blues',
                      text=transitions.astype(int),
                      texttemplate='%{text}',
                      textfont={"size": 10}),
            row=2, col=1
        )
        
        # 4. Mutation hotspots
        hotspots = sorted(position_mutations.items(), key=lambda x: x[1], reverse=True)[:20]
        if hotspots:
            hotspot_pos, hotspot_freq = zip(*hotspots)
            
            fig.add_trace(
                go.Scatter(x=list(hotspot_pos), y=list(hotspot_freq),
                          mode='markers+text',
                          marker=dict(size=15, color=hotspot_freq,
                                    colorscale='Reds', showscale=True),
                          text=[f"Pos {p}" for p in hotspot_pos],
                          textposition="top center"),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Comprehensive Mutation Pattern Analysis",
            showlegend=False,
            height=800
        )
        
        output_path = self.output_dir / "mutation_analysis.html"
        fig.write_html(str(output_path))
        
        return output_path
    
    def create_gc_content_landscape(self, sequences):
        """Create GC content landscape visualization"""
        
        # Calculate GC content for windows
        window_size = 10
        gc_landscape = []
        
        for seq in sequences[:100]:  # Sample
            gc_windows = []
            for i in range(0, len(seq) - window_size, window_size):
                window = seq[i:i+window_size]
                gc = (window.count('G') + window.count('C')) / len(window)
                gc_windows.append(gc)
            gc_landscape.append(gc_windows)
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Heatmap
        sns.heatmap(gc_landscape, cmap='RdYlBu_r', 
                   cbar_kws={'label': 'GC Content'},
                   ax=ax1, vmin=0, vmax=1)
        ax1.set_xlabel(f'Position (windows of {window_size}bp)')
        ax1.set_ylabel('Sequence')
        ax1.set_title('GC Content Landscape')
        
        # Average GC profile
        avg_gc = np.mean(gc_landscape, axis=0)
        positions = np.arange(len(avg_gc)) * window_size
        
        ax2.plot(positions, avg_gc, linewidth=2, color='darkblue')
        ax2.fill_between(positions, avg_gc, alpha=0.3)
        ax2.set_xlabel('Position (bp)')
        ax2.set_ylabel('Avg GC Content')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        # Add CpG island detection
        cpg_threshold = 0.6
        cpg_islands = positions[avg_gc > cpg_threshold]
        if len(cpg_islands) > 0:
            ax2.scatter(cpg_islands, [cpg_threshold]*len(cpg_islands),
                       color='red', s=20, marker='v', label=f'CpG Islands (>{cpg_threshold})')
            ax2.legend()
        
        plt.tight_layout()
        
        output_path = self.output_dir / "gc_landscape.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_barcode_diversity_sunburst(self, cluster_data):
        """Create sunburst chart showing barcode diversity hierarchy"""
        
        # Prepare hierarchical data
        data = []
        
        # Root
        data.append(dict(ids='Total', labels='All Barcodes', parents=''))
        
        # Categories
        categories = {
            'High Frequency': [],
            'Medium Frequency': [],
            'Low Frequency': [],
            'Unique': []
        }
        
        # Categorize barcodes by frequency
        barcode_counts = cluster_data['barcode'].value_counts()
        
        for barcode, count in barcode_counts.items():
            if pd.notna(barcode):
                if count > 100:
                    categories['High Frequency'].append((barcode, count))
                elif count > 10:
                    categories['Medium Frequency'].append((barcode, count))
                elif count > 1:
                    categories['Low Frequency'].append((barcode, count))
                else:
                    categories['Unique'].append((barcode, count))
        
        # Build hierarchy
        colors = []
        for cat_name, barcodes in categories.items():
            # Add category
            data.append(dict(
                ids=cat_name,
                labels=f"{cat_name}<br>({len(barcodes)} types)",
                parents='Total',
                values=sum(count for _, count in barcodes)
            ))
            
            # Add top barcodes in each category
            for bc, count in sorted(barcodes, key=lambda x: x[1], reverse=True)[:5]:
                bc_id = f"{cat_name}_{bc[:8]}"
                data.append(dict(
                    ids=bc_id,
                    labels=f"{bc[:8]}...<br>({count} reads)",
                    parents=cat_name,
                    values=count
                ))
        
        # Convert to DataFrame
        df = pd.DataFrame(data[1:])  # Skip root for Plotly
        
        # Create sunburst
        fig = go.Figure(go.Sunburst(
            ids=df['ids'],
            labels=df['labels'],
            parents=df['parents'],
            values=df['values'],
            branchvalues="total",
            marker=dict(colorscale='Viridis'),
            textinfo="label+percent parent"
        ))
        
        fig.update_layout(
            title="Barcode Diversity Hierarchy",
            height=600,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        output_path = self.output_dir / "barcode_sunburst.html"
        fig.write_html(str(output_path))
        
        return output_path


def create_advanced_dashboard(clusters_path, design_path):
    """Create comprehensive advanced visualizations"""
    
    print("Creating advanced visualizations...")
    
    # Load data
    clusters_df = pd.read_csv(clusters_path, nrows=100000)  # Sample for demo
    design_df = pd.read_csv(design_path)
    
    # Initialize visualizer
    viz = AdvancedCRISPRVisualizer()
    
    # 1. Sequence Logo
    print("1. Creating sequence logo...")
    if 'sequence' in design_df:
        sequences = design_df['sequence'].head(1000).tolist()
        viz.create_sequence_logo(sequences, "CRISPR Payload Sequence Conservation")
    
    # 2. Hamming Distance Heatmap
    print("2. Creating Hamming distance heatmap...")
    if 'barcode' in design_df:
        barcodes = design_df['barcode'].head(100).tolist()
        viz.create_hamming_distance_heatmap(barcodes)
    
    # 3. 3D Barcode Network
    print("3. Creating 3D barcode network...")
    if 'barcode' in design_df:
        barcodes = design_df['barcode'].head(50).tolist()
        viz.create_3d_barcode_network(barcodes)
    
    # 4. Mutation Pattern Analysis
    print("4. Creating mutation pattern analysis...")
    if 'sequence' in design_df and len(design_df) > 1:
        reference = design_df['sequence'].iloc[0]
        variants = design_df['sequence'].iloc[1:].tolist()
        viz.create_mutation_pattern_analysis(reference, variants)
    
    # 5. GC Content Landscape
    print("5. Creating GC content landscape...")
    if 'sequence' in design_df:
        sequences = design_df['sequence'].head(100).tolist()
        viz.create_gc_content_landscape(sequences)
    
    # 6. Barcode Diversity Sunburst
    print("6. Creating barcode diversity sunburst...")
    viz.create_barcode_diversity_sunburst(clusters_df)
    
    print(f"\n✅ Advanced visualizations saved to: {viz.output_dir}")
    print("\nGenerated files:")
    for file in viz.output_dir.glob("*"):
        print(f"  - {file.name}")
    
    return viz.output_dir


if __name__ == "__main__":
    clusters_path = Path("/home/mch/dna/DNA-Data for Telhai/2023-05-11/clusters.csv")
    design_path = Path("/home/mch/dna/updated_data/micro_design.csv")
    
    create_advanced_dashboard(clusters_path, design_path)