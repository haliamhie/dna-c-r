#!/usr/bin/env python3
"""
NVIDIA NIM-powered CRISPR Barcode Classifier
Uses NVIDIA's AI models for advanced sequence classification
"""

import os
import json
import requests
import numpy as np
from typing import List, Dict, Tuple
import asyncio
import aiohttp
from pathlib import Path

class NVIDIASequenceClassifier:
    def __init__(self):
        self.api_key = os.environ.get('NVIDIA_API_KEY')
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment")
        
        # NVIDIA NIM endpoints
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Available models
        self.models = {
            "embedding": "nvidia/nv-embedqa-e5-v5",  # For sequence embeddings
            "llm": "meta/llama-3.1-8b-instruct",     # For analysis
            "bio": "nvidia/bionemo-dnabert"          # DNA-specific (if available)
        }
        
    async def get_sequence_embedding(self, sequence: str) -> np.ndarray:
        """Get embedding vector for DNA sequence"""
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "input": [f"DNA sequence: {sequence}"],
            "model": self.models["embedding"],
            "encoding_format": "float"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return np.array(data["data"][0]["embedding"])
                else:
                    print(f"Error: {response.status}")
                    return None
    
    async def classify_enrichment(self, sequence: str, count: int, z_score: float) -> Dict:
        """Classify if a barcode is enriched using NVIDIA AI"""
        url = f"{self.base_url}/chat/completions"
        
        prompt = f"""
        Analyze this CRISPR screening barcode for enrichment:
        
        Sequence: {sequence}
        Observation count: {count}
        Z-score: {z_score:.2f}
        GC content: {(sequence.count('G') + sequence.count('C')) / len(sequence) * 100:.1f}%
        
        Context: In CRISPR screening, enriched barcodes indicate functional importance.
        Typical threshold: count > 600, z-score > 2.
        
        Classify as: HIGHLY_ENRICHED, ENRICHED, NEUTRAL, DEPLETED, or HIGHLY_DEPLETED
        Provide confidence score (0-1) and brief reasoning.
        
        Return JSON format: {{"classification": "...", "confidence": 0.0, "reason": "..."}}
        """
        
        payload = {
            "model": self.models["llm"],
            "messages": [
                {"role": "system", "content": "You are a genomics expert analyzing CRISPR screening data."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    result_text = data["choices"][0]["message"]["content"]
                    try:
                        # Parse JSON from response
                        import re
                        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group())
                    except:
                        pass
                    return {"classification": "UNKNOWN", "confidence": 0, "reason": result_text}
                else:
                    print(f"Error: {response.status}")
                    return {"classification": "ERROR", "confidence": 0, "reason": str(response.status)}
    
    def calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between two sequences"""
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    async def batch_classify(self, sequences: List[Tuple[str, int, float]]) -> List[Dict]:
        """Classify multiple sequences in parallel"""
        tasks = []
        for seq, count, z_score in sequences:
            task = self.classify_enrichment(seq, count, z_score)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def find_motifs(self, enriched_sequences: List[str]) -> Dict:
        """Find common motifs in enriched sequences using AI"""
        url = f"{self.base_url}/chat/completions"
        
        seq_sample = enriched_sequences[:20]  # Sample for API limits
        
        prompt = f"""
        Analyze these enriched CRISPR barcodes for common motifs:
        
        {chr(10).join(seq_sample)}
        
        Identify:
        1. Common subsequences (3-6bp)
        2. Position-specific nucleotide preferences
        3. GC content patterns
        4. Potential secondary structure elements
        
        Return findings as structured JSON.
        """
        
        payload = {
            "model": self.models["llm"],
            "messages": [
                {"role": "system", "content": "You are a bioinformatics expert specializing in sequence analysis."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"motifs": data["choices"][0]["message"]["content"]}
                else:
                    return {"error": f"Status {response.status}"}

class CRISPRClassifierPipeline:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.classifier = NVIDIASequenceClassifier()
        
    async def analyze_sequences(self):
        """Run full analysis pipeline with NVIDIA AI"""
        import duckdb
        
        # Connect to database
        con = duckdb.connect(str(self.db_path))
        
        # Get top enriched sequences
        enriched = con.execute("""
            SELECT barcode, COUNT(*) as count
            FROM clusters
            WHERE barcode IS NOT NULL
            GROUP BY barcode
            HAVING COUNT(*) > 500
            ORDER BY count DESC
            LIMIT 50
        """).fetchall()
        
        print(f"Analyzing {len(enriched)} potentially enriched barcodes...")
        
        # Calculate z-scores
        all_counts = con.execute("""
            SELECT COUNT(*) as count
            FROM clusters
            WHERE barcode IS NOT NULL
            GROUP BY barcode
        """).fetchall()
        
        counts = [c[0] for c in all_counts]
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        # Prepare sequences for classification
        sequences_to_classify = []
        for barcode, count in enriched:
            z_score = (count - mean_count) / std_count if std_count > 0 else 0
            sequences_to_classify.append((barcode, count, z_score))
        
        # Run AI classification
        print("\nClassifying with NVIDIA AI...")
        classifications = await self.classifier.batch_classify(sequences_to_classify[:10])  # Limit for demo
        
        # Display results
        print("\n" + "="*80)
        print("NVIDIA AI CLASSIFICATION RESULTS")
        print("="*80)
        
        for i, (seq_data, classification) in enumerate(zip(sequences_to_classify[:10], classifications)):
            barcode, count, z_score = seq_data
            print(f"\nBarcode: {barcode}")
            print(f"Count: {count}, Z-score: {z_score:.2f}")
            print(f"Classification: {classification.get('classification', 'UNKNOWN')}")
            print(f"Confidence: {classification.get('confidence', 0):.2%}")
            print(f"Reasoning: {classification.get('reason', 'N/A')[:100]}...")
        
        # Find motifs in enriched sequences
        enriched_seqs = [s[0] for s in enriched if s[1] > 600]
        if enriched_seqs:
            print("\n" + "="*80)
            print("MOTIF ANALYSIS")
            print("="*80)
            motifs = await self.classifier.find_motifs(enriched_seqs)
            print(motifs.get("motifs", "No motifs found"))
        
        con.close()
        
        return classifications

async def main():
    """Test NVIDIA-powered classifier with real CRISPR data"""
    print("NVIDIA NIM-Powered CRISPR Classifier")
    print(f"API Key: {os.environ.get('NVIDIA_API_KEY', 'NOT FOUND')[:15]}...")
    
    # Test basic connectivity
    classifier = NVIDIASequenceClassifier()
    
    # Test with a real barcode
    test_barcode = "GTCTTTCTGCTCGT"
    test_count = 899
    test_zscore = 8.11
    
    print(f"\nTesting with barcode: {test_barcode}")
    result = await classifier.classify_enrichment(test_barcode, test_count, test_zscore)
    print(f"Classification: {result}")
    
    # Run full pipeline if database exists
    db_path = Path("/home/mch/dna/artifacts/dna.duckdb")
    if db_path.exists():
        print("\n" + "="*80)
        print("RUNNING FULL ANALYSIS ON REAL DATA")
        print("="*80)
        pipeline = CRISPRClassifierPipeline(db_path)
        await pipeline.analyze_sequences()
    else:
        print(f"\nDatabase not found at {db_path}")

if __name__ == "__main__":
    asyncio.run(main())