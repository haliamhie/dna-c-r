#!/usr/bin/env python3
"""
CRISPR Analysis Backend API Server
FastAPI-based backend with NVIDIA AI integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import aiohttp
import duckdb
import numpy as np
from pathlib import Path
import json
import os
from datetime import datetime
import hashlib
import time
from collections import defaultdict
import uvicorn
from dnabert_classifier import DNABERTClassifier, CRISPRAnalysisEngine
from unified_classifier import UnifiedCRISPRClassifier, ModelTrainer

# Initialize FastAPI app
app = FastAPI(
    title="CRISPR AI Analysis API",
    description="Backend for AI-powered CRISPR sequence analysis",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class SequenceRequest(BaseModel):
    barcode: str
    count: Optional[int] = None
    z_score: Optional[float] = None
    model: Optional[str] = "ensemble"

class ClassificationResult(BaseModel):
    barcode: str
    count: int
    z_score: float
    classification: str
    confidence: float
    reason: str
    gc_content: float
    length: int
    timestamp: str

class AnalysisStatus(BaseModel):
    total_sequences: int
    processed: int
    classified: int
    enriched: int
    depleted: int
    processing_rate: float
    estimated_time_remaining: float

# Global state
class AnalysisEngine:
    def __init__(self):
        self.db_path = Path("/home/mch/dna/artifacts/dna.duckdb")
        self.con = duckdb.connect(str(self.db_path), read_only=True)
        
        # Initialize DNABERT classifier
        self.dnabert_engine = CRISPRAnalysisEngine(self.db_path)
        self.dnabert_classifier = self.dnabert_engine.classifier
        
        # Initialize unified classifier
        self.unified_classifier = UnifiedCRISPRClassifier()
        
        self.cache_dir = Path("/home/mch/dna/backend/cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.classification_cache = {}
        self.load_cache()
        self.websocket_clients = set()
        
        # Performance metrics
        self.start_time = None
        self.sequences_processed = 0
        self.api_calls_made = 0
        self.cache_hits = 0
        
    def load_cache(self):
        cache_file = self.cache_dir / "classifications.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.classification_cache = json.load(f)
    
    def save_cache(self):
        cache_file = self.cache_dir / "classifications.json"
        with open(cache_file, 'w') as f:
            json.dump(self.classification_cache, f)
    
    async def classify_sequence(self, barcode: str, count: int = None, z_score: float = None) -> Dict:
        """Classify a single sequence using DNABERT genomics model"""
        
        # Check cache first
        cache_key = f"{barcode}_{count}_{z_score:.2f}" if count and z_score else barcode
        if cache_key in self.classification_cache:
            self.cache_hits += 1
            cached_result = self.classification_cache[cache_key]
            # Ensure it has model field
            if 'model' not in cached_result:
                cached_result['model'] = 'DNABERT'
            return cached_result
        
        # Use DNABERT for classification
        self.api_calls_made += 1
        result = await self.dnabert_engine.analyze_sequence(barcode, count, z_score)
        
        # Cache result
        self.classification_cache[cache_key] = result
        if len(self.classification_cache) % 50 == 0:
            self.save_cache()
        
        return result
    
    async def _nvidia_classify(self, barcode: str, count: int, z_score: float) -> Dict:
        """Call NVIDIA NIM API for classification
        
        Note: For production genomics applications, NVIDIA recommends using
        specialized models from BioNeMo platform:
        - DNABERT: For DNA sequence analysis and mutation effects
        - ESM-2: For protein sequence modeling
        - scBERT: For single-cell RNA sequencing analysis
        
        This implementation uses Llama 3.1 as a demonstration of the API integration.
        For production CRISPR analysis, consider using DNABERT through BioNeMo.
        """
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        gc_content = (barcode.count('G') + barcode.count('C')) / len(barcode) * 100
        
        prompt = f"""
        Analyze this CRISPR barcode:
        Sequence: {barcode}
        Count: {count}
        Z-score: {z_score:.2f}
        GC%: {gc_content:.1f}
        
        Classify as: HIGHLY_ENRICHED, ENRICHED, NEUTRAL, DEPLETED, or HIGHLY_DEPLETED
        Return JSON: {{"classification": "...", "confidence": 0.0, "reason": "..."}}
        """
        
        payload = {
            "model": "meta/llama-3.1-8b-instruct",  # Demo model - use DNABERT for production
            "messages": [
                {"role": "system", "content": "You are a genomics expert. Respond with JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        result_text = data["choices"][0]["message"]["content"]
                        
                        import re
                        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group())
                        else:
                            result = {"classification": "UNKNOWN", "confidence": 0, "reason": "Parse error"}
                        
                        return {
                            "barcode": barcode,
                            "count": count,
                            "z_score": z_score,
                            "gc_content": gc_content,
                            "length": len(barcode),
                            "timestamp": datetime.now().isoformat(),
                            **result
                        }
                    else:
                        return {
                            "barcode": barcode,
                            "count": count,
                            "z_score": z_score,
                            "gc_content": gc_content,
                            "length": len(barcode),
                            "classification": "ERROR",
                            "confidence": 0,
                            "reason": f"API error {response.status}",
                            "timestamp": datetime.now().isoformat()
                        }
        except Exception as e:
            return {
                "barcode": barcode,
                "count": count,
                "z_score": z_score,
                "gc_content": gc_content,
                "length": len(barcode),
                "classification": "ERROR",
                "confidence": 0,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics"""
        stats = self.con.execute("""
            SELECT 
                COUNT(*) as total_sequences,
                COUNT(DISTINCT barcode) as unique_barcodes,
                COUNT(CASE WHEN barcode IS NOT NULL THEN 1 END) as sequences_with_barcode,
                COUNT(CASE WHEN barcode IS NULL THEN 1 END) as sequences_without_barcode
            FROM clusters
        """).fetchone()
        
        return {
            "total_sequences": stats[0],
            "unique_barcodes": stats[1],
            "sequences_with_barcode": stats[2],
            "sequences_without_barcode": stats[3],
            "success_rate": (stats[2] / stats[0]) * 100 if stats[0] > 0 else 0
        }
    
    def get_top_sequences(self, limit: int = 100) -> List[Dict]:
        """Get top sequences by count"""
        sequences = self.con.execute("""
            SELECT barcode, COUNT(*) as count
            FROM clusters
            WHERE barcode IS NOT NULL
            GROUP BY barcode
            ORDER BY count DESC
            LIMIT ?
        """, [limit]).fetchall()
        
        # Calculate z-scores
        all_counts = self.con.execute("""
            SELECT COUNT(*) as c FROM clusters 
            WHERE barcode IS NOT NULL 
            GROUP BY barcode
        """).fetchall()
        counts = [c[0] for c in all_counts]
        mean = np.mean(counts)
        std = np.std(counts)
        
        results = []
        for barcode, count in sequences:
            z_score = (count - mean) / std if std > 0 else 0
            gc_content = (barcode.count('G') + barcode.count('C')) / len(barcode) * 100
            results.append({
                "barcode": barcode,
                "count": count,
                "z_score": z_score,
                "gc_content": gc_content,
                "length": len(barcode)
            })
        
        return results
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.sequences_processed / elapsed if elapsed > 0 else 0
        else:
            elapsed = 0
            rate = 0
        
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.api_calls_made, 1) * 100
        
        return {
            "sequences_processed": self.sequences_processed,
            "api_calls_made": self.api_calls_made,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "processing_rate": rate,
            "elapsed_time": elapsed,
            "avg_time_per_sequence": elapsed / max(self.sequences_processed, 1)
        }

# Initialize engine
engine = AnalysisEngine()

# API Endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "CRISPR AI Analysis API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": [
            "/api/stats",
            "/api/sequences/top",
            "/api/classify",
            "/api/batch-classify",
            "/api/performance",
            "/ws/live"
        ]
    }

@app.get("/api/stats")
async def get_stats():
    """Get database statistics"""
    return engine.get_database_stats()

@app.get("/api/sequences/top")
async def get_top_sequences(limit: int = 100):
    """Get top sequences by count"""
    return engine.get_top_sequences(limit)

@app.post("/api/classify")
async def classify_sequence(request: SequenceRequest):
    """Classify a single sequence"""
    engine.sequences_processed += 1
    if not engine.start_time:
        engine.start_time = time.time()
    
    # Use unified classifier with specified model
    result = await engine.unified_classifier.classify(
        request.barcode,
        request.count if request.count else 100,
        request.z_score if request.z_score else 0.0,
        model=request.model
    )
    return result

@app.post("/api/batch-classify")
async def batch_classify(sequences: List[SequenceRequest]):
    """Classify multiple sequences"""
    results = []
    for seq in sequences:
        engine.sequences_processed += 1
        if not engine.start_time:
            engine.start_time = time.time()
        
        result = await engine.classify_sequence(
            seq.barcode,
            seq.count,
            seq.z_score
        )
        results.append(result)
    
    return results

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    return engine.get_performance_metrics()

@app.get("/api/export/{format}")
async def export_results(format: str):
    """Export classification results"""
    if format not in ["json", "csv", "tsv"]:
        raise HTTPException(status_code=400, detail="Invalid format")
    
    # Get all cached classifications
    results = list(engine.classification_cache.values())
    
    if format == "json":
        return JSONResponse(content=results)
    
    # CSV/TSV export
    import csv
    import io
    
    output = io.StringIO()
    delimiter = ',' if format == 'csv' else '\t'
    writer = csv.DictWriter(output, fieldnames=results[0].keys() if results else [], delimiter=delimiter)
    writer.writeheader()
    writer.writerows(results)
    
    return FileResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type=f"text/{format}",
        filename=f"classifications.{format}"
    )

@app.post("/api/classify-compare")
async def classify_with_comparison(request: SequenceRequest):
    """Classify with all models for comparison"""
    result = await engine.unified_classifier.classify_with_all_models(
        request.barcode,
        request.count if request.count else 100,
        request.z_score if request.z_score else 0.0
    )
    return result

@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": list(engine.unified_classifier.models.keys()) + 
                 ['deepcrispr', 'azimuth', 'crispor'],
        "default": "ensemble"
    }

@app.get("/api/models/performance")
async def get_model_performance():
    """Get performance metrics for all models"""
    return engine.unified_classifier.get_performance_report()

@app.post("/api/models/train")
async def train_models(sample_size: int = 1000):
    """Train ML models on database sequences"""
    trainer = ModelTrainer(engine.db_path)
    results = await trainer.train_models(sample_size)
    return results

@app.post("/api/models/recommend")
async def recommend_model(requirements: Dict):
    """Get model recommendations based on requirements"""
    recommendations = engine.unified_classifier.get_model_recommendations(requirements)
    return {"recommended_models": recommendations}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates"""
    await websocket.accept()
    engine.websocket_clients.add(websocket)
    
    try:
        while True:
            # Send performance metrics every second
            await asyncio.sleep(1)
            metrics = engine.get_performance_metrics()
            await websocket.send_json({"type": "metrics", "data": metrics})
    except:
        engine.websocket_clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)