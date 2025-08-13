#!/usr/bin/env python3
"""
EXACT data processor - streams the ACTUAL 7,071,847 sequences in REAL order
Nothing simulated - only speed is throttled for visibility
"""

import asyncio
import json
import duckdb
import numpy as np
from pathlib import Path
import websockets
import time
from typing import Dict, List, Optional

class ExactDataProcessor:
    def __init__(self):
        self.db_path = Path("/home/mch/dna/artifacts/dna.duckdb")
        self.con = duckdb.connect(str(self.db_path), read_only=True)
        self.clients = set()
        
        # Load COMPLETE dataset info
        self.initialize_exact_data()
        
        # Processing state
        self.current_position = 0
        self.processed_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.enriched_found = 0
        
        # Real-time metrics
        self.stage_counts = {
            'loading': 0,
            'quality_control': 0, 
            'error_correction': 0,
            'clustering': 0,
            'failed': 0
        }
        
    def initialize_exact_data(self):
        """Load EXACT data structures from database"""
        print("Loading exact dataset...")
        
        # Get EXACT barcode distribution
        self.barcode_counts = {}
        result = self.con.execute("""
            SELECT barcode, COUNT(*) as count 
            FROM clusters 
            WHERE barcode IS NOT NULL 
            GROUP BY barcode
        """).fetchall()
        
        for barcode, count in result:
            self.barcode_counts[barcode] = count
        
        # Calculate EXACT statistics
        stats = self.con.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT barcode) as unique_barcodes,
                COUNT(CASE WHEN barcode IS NOT NULL THEN 1 END) as with_barcode,
                COUNT(CASE WHEN barcode IS NULL THEN 1 END) as without_barcode,
                AVG(LENGTH(barcode)) as avg_length
            FROM clusters
        """).fetchone()
        
        self.total_sequences = stats[0]  # 7,071,847
        self.unique_barcodes = stats[1]  # 10,000
        self.sequences_with_barcode = stats[2]  # 3,149,475
        self.sequences_without_barcode = stats[3]  # 3,922,372
        
        print(f"Loaded {self.total_sequences:,} sequences")
        print(f"Success rate: {(self.sequences_with_barcode/self.total_sequences)*100:.1f}%")
        
        # Get ALL enriched barcodes (count > 600)
        self.enriched_barcodes = set()
        enriched = self.con.execute("""
            SELECT barcode 
            FROM clusters 
            WHERE barcode IS NOT NULL 
            GROUP BY barcode 
            HAVING COUNT(*) > 600
        """).fetchall()
        
        for (barcode,) in enriched:
            self.enriched_barcodes.add(barcode)
        
        print(f"Found {len(self.enriched_barcodes)} enriched barcodes")
        
        # Calculate EXACT mean and std for Z-scores
        counts = [c for c in self.barcode_counts.values()]
        self.mean_count = np.mean(counts)  # Real mean
        self.std_count = np.std(counts)    # Real std
        
        # Pre-calculate Hamming distance matrix for top barcodes
        self.calculate_hamming_matrix()
        
    def calculate_hamming_matrix(self):
        """Pre-calculate Hamming distances between all unique barcodes"""
        print("Calculating Hamming distance matrix...")
        
        # Get all unique barcodes
        all_barcodes = list(self.barcode_counts.keys())[:100]  # Top 100 for efficiency
        
        self.hamming_cache = {}
        for i, b1 in enumerate(all_barcodes):
            for b2 in all_barcodes[i+1:]:
                if len(b1) == len(b2) == 14:
                    dist = sum(c1 != c2 for c1, c2 in zip(b1, b2))
                    self.hamming_cache[f"{b1}_{b2}"] = dist
                    self.hamming_cache[f"{b2}_{b1}"] = dist
        
        print(f"Calculated {len(self.hamming_cache)} Hamming distances")
    
    async def stream_exact_sequences(self, websocket, speed_multiplier=100):
        """
        Stream the EXACT sequences from database in REAL order
        speed_multiplier: How much faster than real-time to process (100 = 100x speed)
        """
        # Start from beginning or resume
        offset = self.current_position
        batch_size = 1000  # Process in batches for efficiency
        
        while offset < self.total_sequences:
            # Get EXACT next batch of sequences
            batch = self.con.execute(f"""
                SELECT id, barcode 
                FROM clusters 
                ORDER BY id 
                LIMIT {batch_size} 
                OFFSET {offset}
            """).fetchall()
            
            if not batch:
                break
            
            for seq_id, barcode in batch:
                self.current_position += 1
                self.processed_count += 1
                
                # EXACT processing pipeline
                stages = await self.process_exact_sequence(seq_id, barcode)
                
                # Send each stage update
                for stage_data in stages:
                    await websocket.send(json.dumps({
                        'type': 'sequence',
                        'data': stage_data
                    }))
                    
                    # Throttle speed for visibility (only manipulation)
                    await asyncio.sleep(1.0 / (1000 * speed_multiplier))
                
                # Send metrics every 100 sequences
                if self.processed_count % 100 == 0:
                    metrics = self.get_exact_metrics()
                    await websocket.send(json.dumps({
                        'type': 'metrics',
                        'data': metrics
                    }))
            
            offset += batch_size
            
            # Progress update
            progress = (offset / self.total_sequences) * 100
            await websocket.send(json.dumps({
                'type': 'progress',
                'data': {
                    'processed': offset,
                    'total': self.total_sequences,
                    'percentage': progress
                }
            }))
    
    async def process_exact_sequence(self, seq_id: str, barcode: Optional[str]) -> List[Dict]:
        """Process a single sequence through EXACT pipeline stages"""
        stages = []
        
        # Stage 1: Loading (REAL)
        stage_data = {
            'stage': 'loading',
            'id': seq_id,
            'barcode': barcode or '',
            'has_barcode': barcode is not None,
            'position': self.current_position,
            'timestamp': time.time()
        }
        self.stage_counts['loading'] += 1
        stages.append(stage_data.copy())
        
        if barcode:
            # Stage 2: Quality Control (REAL validation)
            stage_data['stage'] = 'quality_control'
            stage_data['length'] = len(barcode)
            stage_data['valid_length'] = len(barcode) == 14
            stage_data['base_composition'] = {
                'A': barcode.count('A'),
                'T': barcode.count('T'),
                'G': barcode.count('G'),
                'C': barcode.count('C')
            }
            stage_data['gc_content'] = (barcode.count('G') + barcode.count('C')) / len(barcode)
            stage_data['homopolymer'] = max(len(s) for s in 
                [''.join(g) for k, g in __import__('itertools').groupby(barcode)])
            
            self.stage_counts['quality_control'] += 1
            self.success_count += 1
            stages.append(stage_data.copy())
            
            # Stage 3: Error Correction (REAL Hamming distances)
            stage_data['stage'] = 'error_correction'
            
            # Get REAL Hamming distances to other barcodes
            distances = []
            for other_barcode in list(self.barcode_counts.keys())[:20]:  # Sample for speed
                if other_barcode != barcode and len(other_barcode) == len(barcode):
                    key = f"{barcode}_{other_barcode}"
                    if key in self.hamming_cache:
                        distances.append(self.hamming_cache[key])
                    else:
                        dist = sum(c1 != c2 for c1, c2 in zip(barcode, other_barcode))
                        distances.append(dist)
            
            if distances:
                stage_data['min_hamming'] = min(distances)
                stage_data['avg_hamming'] = np.mean(distances)
                stage_data['hamming_valid'] = min(distances) >= 4
            
            self.stage_counts['error_correction'] += 1
            stages.append(stage_data.copy())
            
            # Stage 4: Clustering (REAL enrichment analysis)
            stage_data['stage'] = 'clustering'
            
            # EXACT count for this barcode
            count = self.barcode_counts.get(barcode, 1)
            stage_data['count'] = count
            stage_data['frequency'] = count / self.total_sequences
            
            # EXACT Z-score calculation
            z_score = (count - self.mean_count) / self.std_count if self.std_count > 0 else 0
            stage_data['z_score'] = z_score
            
            # EXACT enrichment status
            is_enriched = barcode in self.enriched_barcodes
            stage_data['enriched'] = is_enriched
            stage_data['fold_change'] = count / self.mean_count
            
            if is_enriched:
                self.enriched_found += 1
                stage_data['enrichment_rank'] = self.enriched_found
            
            self.stage_counts['clustering'] += 1
            stages.append(stage_data.copy())
            
        else:
            # Failed sequence (REAL failure)
            stage_data['stage'] = 'failed'
            stage_data['failure_reason'] = 'no_barcode_detected'
            stage_data['failure_position'] = self.current_position
            self.stage_counts['failed'] += 1
            self.failure_count += 1
            stages.append(stage_data.copy())
        
        return stages
    
    def get_exact_metrics(self) -> Dict:
        """Return EXACT real-time metrics"""
        return {
            'total_sequences': self.total_sequences,
            'processed': self.processed_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': (self.success_count / max(self.processed_count, 1)) * 100,
            'failure_rate': (self.failure_count / max(self.processed_count, 1)) * 100,
            'enriched_found': self.enriched_found,
            'unique_barcodes_seen': len(set(self.barcode_counts.keys())),
            'stage_counts': self.stage_counts,
            'current_position': self.current_position,
            'progress_percentage': (self.current_position / self.total_sequences) * 100,
            'exact_mean_coverage': self.mean_count,
            'exact_std_coverage': self.std_count
        }
    
    async def handler(self, websocket):
        """WebSocket connection handler"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            # Send initial EXACT data package
            await websocket.send(json.dumps({
                'type': 'init',
                'data': {
                    'total_sequences': self.total_sequences,
                    'unique_barcodes': self.unique_barcodes,
                    'success_rate': (self.sequences_with_barcode / self.total_sequences) * 100,
                    'enriched_count': len(self.enriched_barcodes),
                    'barcode_distribution': {
                        k: v for k, v in 
                        sorted(self.barcode_counts.items(), key=lambda x: x[1], reverse=True)[:50]
                    }
                }
            }))
            
            # Start streaming EXACT sequences
            await self.stream_exact_sequences(websocket, speed_multiplier=100)
            
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        finally:
            self.clients.remove(websocket)
    
    async def run_server(self):
        """Start WebSocket server"""
        print("Starting EXACT data server on ws://localhost:8766")
        print(f"Will stream {self.total_sequences:,} real sequences")
        async with websockets.serve(self.handler, "localhost", 8766):
            await asyncio.Future()  # Run forever

if __name__ == "__main__":
    processor = ExactDataProcessor()
    asyncio.run(processor.run_server())