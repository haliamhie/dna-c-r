#!/usr/bin/env python3
"""
Production CRISPR Analysis Pipeline with NVIDIA AI Classification
Processes all 7M sequences with intelligent batching and caching
"""

import os
import json
import asyncio
import aiohttp
import duckdb
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib
from collections import defaultdict
import time

class NVIDIABatchClassifier:
    """Optimized NVIDIA classifier with batching and caching"""
    
    def __init__(self, cache_dir: Path = Path("/home/mch/dna/ai_cache")):
        self.api_key = os.environ.get('NVIDIA_API_KEY')
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found")
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self.requests_per_minute = 50  # Adjust based on API limits
        self.last_request_time = 0
        self.request_delay = 60 / self.requests_per_minute
        
        # Classification cache
        self.classification_cache = {}
        self.load_cache()
        
    def load_cache(self):
        """Load previous classifications from cache"""
        cache_file = self.cache_dir / "classifications.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.classification_cache = json.load(f)
            print(f"Loaded {len(self.classification_cache)} cached classifications")
    
    def save_cache(self):
        """Save classifications to cache"""
        cache_file = self.cache_dir / "classifications.json"
        with open(cache_file, 'w') as f:
            json.dump(self.classification_cache, f, indent=2)
    
    def get_cache_key(self, barcode: str, count: int, z_score: float) -> str:
        """Generate cache key for classification"""
        return f"{barcode}_{count}_{z_score:.2f}"
    
    async def classify_batch(self, sequences: List[Tuple[str, int, float]]) -> List[Dict]:
        """Classify a batch of sequences with caching and rate limiting"""
        results = []
        to_classify = []
        
        # Check cache first
        for seq, count, z_score in sequences:
            cache_key = self.get_cache_key(seq, count, z_score)
            if cache_key in self.classification_cache:
                results.append(self.classification_cache[cache_key])
            else:
                to_classify.append((seq, count, z_score))
                results.append(None)  # Placeholder
        
        # Classify uncached sequences
        if to_classify:
            print(f"Classifying {len(to_classify)} new sequences...")
            
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.request_delay:
                await asyncio.sleep(self.request_delay - time_since_last)
            
            # Batch API call
            classifications = await self._call_nvidia_api(to_classify)
            
            # Update cache and results
            result_idx = 0
            for i, classification in enumerate(classifications):
                # Find the placeholder position
                while results[result_idx] is not None:
                    result_idx += 1
                
                results[result_idx] = classification
                
                # Cache the result
                seq, count, z_score = to_classify[i]
                cache_key = self.get_cache_key(seq, count, z_score)
                self.classification_cache[cache_key] = classification
            
            self.last_request_time = time.time()
            
            # Save cache periodically
            if len(self.classification_cache) % 100 == 0:
                self.save_cache()
        
        return results
    
    async def _call_nvidia_api(self, sequences: List[Tuple[str, int, float]]) -> List[Dict]:
        """Make batch API call to NVIDIA"""
        url = f"{self.base_url}/chat/completions"
        
        # Create batch prompt
        batch_data = []
        for seq, count, z_score in sequences:
            gc_content = (seq.count('G') + seq.count('C')) / len(seq) * 100
            batch_data.append({
                "barcode": seq,
                "count": count,
                "z_score": round(z_score, 2),
                "gc_content": round(gc_content, 1)
            })
        
        prompt = f"""
        Analyze these CRISPR screening barcodes for enrichment.
        For each, classify as: HIGHLY_ENRICHED, ENRICHED, NEUTRAL, DEPLETED, or HIGHLY_DEPLETED
        
        Data:
        {json.dumps(batch_data, indent=2)}
        
        Return JSON array with classification, confidence (0-1), and brief reason for each.
        Format: [{{"barcode": "...", "classification": "...", "confidence": 0.0, "reason": "..."}}]
        """
        
        payload = {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are a genomics expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        result_text = data["choices"][0]["message"]["content"]
                        
                        # Parse JSON response
                        import re
                        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                        if json_match:
                            results = json.loads(json_match.group())
                            return results
                        else:
                            # Fallback if parsing fails
                            return [{"classification": "UNKNOWN", "confidence": 0, "reason": "Parse error"} 
                                   for _ in sequences]
                    else:
                        print(f"API Error: {response.status}")
                        return [{"classification": "ERROR", "confidence": 0, "reason": str(response.status)} 
                               for _ in sequences]
        except Exception as e:
            print(f"Exception: {e}")
            return [{"classification": "ERROR", "confidence": 0, "reason": str(e)} 
                   for _ in sequences]

class AIEnhancedPipeline:
    """Complete pipeline with AI classification"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.con = duckdb.connect(str(db_path))
        self.classifier = NVIDIABatchClassifier()
        self.results_dir = Path("/home/mch/dna/ai_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def get_sequences_for_classification(self, min_count: int = 100) -> List[Tuple[str, int, float]]:
        """Get sequences that need classification"""
        # Get barcode statistics
        stats = self.con.execute("""
            SELECT 
                barcode,
                COUNT(*) as count
            FROM clusters
            WHERE barcode IS NOT NULL
            GROUP BY barcode
            HAVING COUNT(*) >= ?
            ORDER BY count DESC
        """, [min_count]).fetchall()
        
        # Calculate z-scores
        all_counts = [s[1] for s in stats]
        mean_count = np.mean(all_counts)
        std_count = np.std(all_counts)
        
        sequences = []
        for barcode, count in stats:
            z_score = (count - mean_count) / std_count if std_count > 0 else 0
            sequences.append((barcode, count, z_score))
        
        return sequences
    
    async def classify_all_sequences(self, batch_size: int = 10, limit: int = None):
        """Classify all sequences with AI"""
        sequences = self.get_sequences_for_classification()
        if limit:
            sequences = sequences[:limit]
        print(f"\nTotal sequences to classify: {len(sequences)}")
        
        # Process in batches
        all_results = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
            
            results = await self.classifier.classify_batch(batch)
            
            # Combine sequences with classifications
            for (barcode, count, z_score), classification in zip(batch, results):
                all_results.append({
                    "barcode": barcode,
                    "count": count,
                    "z_score": z_score,
                    **classification
                })
            
            # Show sample results
            if i == 0:  # First batch
                print("\nSample classifications:")
                for r in results[:3]:
                    print(f"  {r.get('classification', 'UNKNOWN')}: {r.get('confidence', 0):.1%}")
        
        # Save results
        self.classifier.save_cache()
        return all_results
    
    def analyze_classifications(self, results: List[Dict]):
        """Analyze classification results"""
        # Group by classification
        by_class = defaultdict(list)
        for r in results:
            by_class[r.get('classification', 'UNKNOWN')].append(r)
        
        print("\n" + "="*80)
        print("CLASSIFICATION SUMMARY")
        print("="*80)
        
        for classification in ['HIGHLY_ENRICHED', 'ENRICHED', 'NEUTRAL', 'DEPLETED', 'HIGHLY_DEPLETED', 'UNKNOWN', 'ERROR']:
            items = by_class[classification]
            if items:
                avg_z = np.mean([r['z_score'] for r in items])
                avg_count = np.mean([r['count'] for r in items])
                print(f"\n{classification}: {len(items)} sequences")
                print(f"  Average count: {avg_count:.0f}")
                print(f"  Average Z-score: {avg_z:.2f}")
                
                if classification == 'HIGHLY_ENRICHED' and items:
                    print(f"  Top 3 barcodes:")
                    for r in sorted(items, key=lambda x: x['count'], reverse=True)[:3]:
                        print(f"    {r['barcode']}: count={r['count']}, Z={r['z_score']:.2f}")
        
        return by_class
    
    def export_results(self, results: List[Dict], by_class: Dict):
        """Export results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full results JSON
        json_file = self.results_dir / f"ai_classifications_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to: {json_file}")
        
        # Enriched sequences only
        enriched = by_class['HIGHLY_ENRICHED'] + by_class['ENRICHED']
        if enriched:
            enriched_file = self.results_dir / f"enriched_barcodes_{timestamp}.txt"
            with open(enriched_file, 'w') as f:
                f.write("barcode\tcount\tz_score\tclassification\tconfidence\n")
                for r in sorted(enriched, key=lambda x: x['count'], reverse=True):
                    f.write(f"{r['barcode']}\t{r['count']}\t{r['z_score']:.2f}\t")
                    f.write(f"{r['classification']}\t{r.get('confidence', 0):.2%}\n")
            print(f"Enriched barcodes saved to: {enriched_file}")
        
        # Summary report
        report_file = self.results_dir / f"classification_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write("# AI Classification Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- Total sequences analyzed: {len(results)}\n")
            f.write(f"- Highly enriched: {len(by_class['HIGHLY_ENRICHED'])}\n")
            f.write(f"- Enriched: {len(by_class['ENRICHED'])}\n")
            f.write(f"- Neutral: {len(by_class['NEUTRAL'])}\n")
            f.write(f"- Depleted: {len(by_class['DEPLETED'] + by_class['HIGHLY_DEPLETED'])}\n")
            f.write("\n## Top Enriched Barcodes\n\n")
            
            for r in sorted(enriched, key=lambda x: x['count'], reverse=True)[:20]:
                f.write(f"- `{r['barcode']}`: {r['count']} counts, Z={r['z_score']:.2f}\n")
        
        print(f"Report saved to: {report_file}")
        
        return json_file, enriched_file, report_file

async def main():
    """Run the AI-enhanced pipeline"""
    print("="*80)
    print("AI-ENHANCED CRISPR ANALYSIS PIPELINE")
    print("Powered by NVIDIA NIM")
    print("="*80)
    
    db_path = Path("/home/mch/dna/artifacts/dna.duckdb")
    pipeline = AIEnhancedPipeline(db_path)
    
    # Run classification (limit to top 50 for demo)
    results = await pipeline.classify_all_sequences(batch_size=5, limit=50)
    
    # Analyze results
    by_class = pipeline.analyze_classifications(results)
    
    # Export results
    files = pipeline.export_results(results, by_class)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Classified {len(results)} sequences")
    print(f"Files saved in: /home/mch/dna/ai_results/")
    
    return results, by_class

if __name__ == "__main__":
    asyncio.run(main())