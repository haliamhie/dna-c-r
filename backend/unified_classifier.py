#!/usr/bin/env python3
"""
Unified Classifier System
Integrates all classification models with model selection
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Literal
from pathlib import Path
import json
from datetime import datetime

# Import all classifiers
from dnabert_classifier import DNABERTClassifier
from ml_classifiers import CRISPRMLClassifier, SpecializedGenomicModels

ModelType = Literal[
    "ensemble", "dnabert", "xgboost", "random_forest", 
    "neural_network", "svm", "gradient_boost",
    "deepcrispr", "azimuth", "crispor"
]

class UnifiedCRISPRClassifier:
    """
    Unified interface for all CRISPR classification models
    """
    
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        self.classification_history = []
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all available models"""
        
        # DNABERT model
        try:
            self.models['dnabert'] = DNABERTClassifier()
            print("✓ DNABERT model loaded")
        except Exception as e:
            print(f"✗ DNABERT failed to load: {e}")
        
        # ML models
        ml_models = [
            'ensemble', 'random_forest', 'xgboost', 
            'neural_network', 'svm', 'gradient_boost'
        ]
        
        for model_name in ml_models:
            try:
                self.models[model_name] = CRISPRMLClassifier(model_type=model_name)
                print(f"✓ {model_name} model loaded")
            except Exception as e:
                print(f"✗ {model_name} failed to load: {e}")
        
        # Initialize performance tracking
        for model_name in self.models.keys():
            self.performance_metrics[model_name] = {
                'total_predictions': 0,
                'avg_time': 0,
                'avg_confidence': 0,
                'classifications': {}
            }
    
    async def classify(self, 
                       sequence: str, 
                       count: int, 
                       z_score: float,
                       model: ModelType = "ensemble") -> Dict:
        """
        Classify a sequence using specified model
        """
        
        start_time = time.time()
        
        # Route to appropriate classifier
        if model == 'dnabert' and 'dnabert' in self.models:
            result = await self.models['dnabert'].classify_with_dnabert(
                sequence, count, z_score
            )
        elif model in ['deepcrispr', 'azimuth', 'crispor']:
            # Use specialized genomic models
            result = self._classify_with_specialized(sequence, count, z_score, model)
        elif model in self.models:
            # Use ML models
            result = self.models[model].predict(sequence, count, z_score)
        else:
            # Fallback to ensemble
            result = self.models.get('ensemble', 
                                    self.models['random_forest']).predict(
                sequence, count, z_score
            )
        
        # Track performance
        elapsed_time = time.time() - start_time
        self._update_metrics(model, result, elapsed_time)
        
        # Add metadata
        result['model_used'] = model
        result['processing_time'] = elapsed_time
        result['timestamp'] = datetime.now().isoformat()
        
        # Store in history
        self.classification_history.append({
            'sequence': sequence,
            'model': model,
            'result': result['classification'],
            'confidence': result.get('confidence', 0),
            'time': elapsed_time
        })
        
        return result
    
    async def classify_with_all_models(self, 
                                       sequence: str, 
                                       count: int, 
                                       z_score: float) -> Dict:
        """
        Classify using all available models for comparison
        """
        
        results = {}
        
        # Test all models
        all_models = list(self.models.keys()) + ['deepcrispr', 'azimuth', 'crispor']
        
        for model_name in all_models:
            try:
                result = await self.classify(sequence, count, z_score, model_name)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {
                    'classification': 'ERROR',
                    'confidence': 0,
                    'error': str(e)
                }
        
        # Calculate consensus
        classifications = [r['classification'] for r in results.values() 
                         if r['classification'] != 'ERROR']
        
        if classifications:
            from collections import Counter
            consensus = Counter(classifications).most_common(1)[0][0]
            consensus_count = Counter(classifications)[consensus]
            consensus_confidence = consensus_count / len(classifications)
        else:
            consensus = 'UNKNOWN'
            consensus_confidence = 0
        
        return {
            'individual_results': results,
            'consensus': consensus,
            'consensus_confidence': consensus_confidence,
            'models_agree': len(set(classifications)) == 1 if classifications else False
        }
    
    def _classify_with_specialized(self, sequence: str, count: int, 
                                  z_score: float, model: str) -> Dict:
        """
        Classify using specialized genomic models
        """
        
        # Get score from specialized model
        if model == 'deepcrispr':
            score = SpecializedGenomicModels.deepcrispr_score(sequence)
        elif model == 'azimuth':
            score = SpecializedGenomicModels.azimuth_score(sequence)
        elif model == 'crispor':
            score = SpecializedGenomicModels.crispor_score(sequence)
        else:
            score = 0.5
        
        # Convert score to classification
        if score >= 0.8:
            classification = 'HIGHLY_ENRICHED'
        elif score >= 0.6:
            classification = 'ENRICHED'
        elif score >= 0.4:
            classification = 'NEUTRAL'
        elif score >= 0.2:
            classification = 'DEPLETED'
        else:
            classification = 'HIGHLY_DEPLETED'
        
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
        
        return {
            'classification': classification,
            'confidence': score,
            'score': score,
            'model': model,
            'gc_content': gc_content,
            'reason': f'{model.upper()} score: {score:.3f}'
        }
    
    def _update_metrics(self, model: str, result: Dict, elapsed_time: float):
        """
        Update performance metrics for a model
        """
        
        if model not in self.performance_metrics:
            self.performance_metrics[model] = {
                'total_predictions': 0,
                'avg_time': 0,
                'avg_confidence': 0,
                'classifications': {}
            }
        
        metrics = self.performance_metrics[model]
        
        # Update counts
        metrics['total_predictions'] += 1
        
        # Update average time (running average)
        n = metrics['total_predictions']
        metrics['avg_time'] = ((n - 1) * metrics['avg_time'] + elapsed_time) / n
        
        # Update average confidence
        confidence = result.get('confidence', 0)
        metrics['avg_confidence'] = ((n - 1) * metrics['avg_confidence'] + confidence) / n
        
        # Track classification distribution
        classification = result.get('classification', 'UNKNOWN')
        if classification not in metrics['classifications']:
            metrics['classifications'][classification] = 0
        metrics['classifications'][classification] += 1
    
    def get_performance_report(self) -> Dict:
        """
        Get comprehensive performance report for all models
        """
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, metrics in self.performance_metrics.items():
            if metrics['total_predictions'] > 0:
                report['models'][model_name] = {
                    'total_predictions': metrics['total_predictions'],
                    'avg_processing_time_ms': metrics['avg_time'] * 1000,
                    'avg_confidence': metrics['avg_confidence'],
                    'classification_distribution': metrics['classifications'],
                    'throughput_per_second': 1 / metrics['avg_time'] if metrics['avg_time'] > 0 else 0
                }
        
        # Add comparative analysis
        if len(report['models']) > 1:
            speeds = {m: d['avg_processing_time_ms'] 
                     for m, d in report['models'].items()}
            report['fastest_model'] = min(speeds, key=speeds.get)
            
            confidences = {m: d['avg_confidence'] 
                          for m, d in report['models'].items()}
            report['most_confident_model'] = max(confidences, key=confidences.get)
        
        return report
    
    def get_model_recommendations(self, requirements: Dict) -> List[str]:
        """
        Get model recommendations based on requirements
        """
        
        recommendations = []
        
        # Speed requirement
        if requirements.get('max_latency_ms'):
            max_latency = requirements['max_latency_ms']
            for model, metrics in self.performance_metrics.items():
                if metrics['total_predictions'] > 0:
                    if metrics['avg_time'] * 1000 <= max_latency:
                        recommendations.append(model)
        
        # Confidence requirement
        if requirements.get('min_confidence'):
            min_conf = requirements['min_confidence']
            recommendations = [m for m in recommendations 
                             if self.performance_metrics[m]['avg_confidence'] >= min_conf]
        
        # Interpretability requirement
        if requirements.get('interpretability'):
            if 'dnabert' in recommendations:
                recommendations = ['dnabert']  # Most interpretable
        
        # Production readiness
        if requirements.get('production_ready'):
            # Prefer well-tested models
            production_models = ['xgboost', 'random_forest', 'ensemble']
            recommendations = [m for m in recommendations if m in production_models]
        
        if not recommendations:
            recommendations = ['ensemble']  # Default recommendation
        
        return recommendations


class ModelTrainer:
    """
    Train ML models on CRISPR data
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        
    async def train_models(self, sample_size: int = 1000) -> Dict:
        """
        Train all ML models on database sequences
        """
        
        import duckdb
        con = duckdb.connect(str(self.db_path), read_only=True)
        
        # Get training data
        print(f"Loading {sample_size} training sequences...")
        
        sequences_data = con.execute(f"""
            WITH stats AS (
                SELECT 
                    AVG(cnt) as mean_count,
                    STDDEV(cnt) as std_count
                FROM (
                    SELECT COUNT(*) as cnt
                    FROM clusters
                    WHERE barcode IS NOT NULL
                    GROUP BY barcode
                )
            )
            SELECT 
                barcode,
                COUNT(*) as count,
                (COUNT(*) - stats.mean_count) / stats.std_count as z_score
            FROM clusters, stats
            WHERE barcode IS NOT NULL
            GROUP BY barcode, stats.mean_count, stats.std_count
            LIMIT {sample_size}
        """).fetchall()
        
        sequences = [row[0] for row in sequences_data]
        counts = [row[1] for row in sequences_data]
        z_scores = [row[2] for row in sequences_data]
        
        # Generate labels based on rules
        labels = []
        for count, z_score in zip(counts, z_scores):
            if z_score >= 2.0 and count > 600:
                labels.append('HIGHLY_ENRICHED')
            elif z_score >= 1.0 and count > 300:
                labels.append('ENRICHED')
            elif z_score <= -2.0:
                labels.append('HIGHLY_DEPLETED')
            elif z_score <= -1.0:
                labels.append('DEPLETED')
            else:
                labels.append('NEUTRAL')
        
        # Train models
        print("Training ML models...")
        
        results = {}
        models_to_train = ['ensemble', 'random_forest', 'xgboost', 'neural_network']
        
        for model_type in models_to_train:
            print(f"  Training {model_type}...")
            
            try:
                classifier = CRISPRMLClassifier(model_type=model_type)
                
                start_time = time.time()
                classifier.train(sequences, labels, counts, z_scores)
                training_time = time.time() - start_time
                
                # Save model
                model_dir = Path(f"/home/mch/dna/models/{model_type}")
                classifier.save_models(model_dir)
                
                results[model_type] = {
                    'status': 'success',
                    'training_time': training_time,
                    'samples_used': len(sequences),
                    'model_saved': str(model_dir)
                }
                
                print(f"    ✓ Completed in {training_time:.2f}s")
                
            except Exception as e:
                results[model_type] = {
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"    ✗ Failed: {e}")
        
        con.close()
        
        return results


# Quick test
async def test_unified_classifier():
    """Test the unified classifier system"""
    
    classifier = UnifiedCRISPRClassifier()
    
    test_sequence = "GTCTTTCTGCTCGT"
    count = 800
    z_score = 2.5
    
    print("=" * 60)
    print("Unified Classifier Test")
    print("=" * 60)
    
    # Test individual model
    print("\n1. Testing XGBoost model:")
    result = await classifier.classify(test_sequence, count, z_score, model='xgboost')
    print(f"   Classification: {result['classification']}")
    print(f"   Confidence: {result.get('confidence', 0):.2%}")
    
    # Test all models
    print("\n2. Testing all models:")
    all_results = await classifier.classify_with_all_models(test_sequence, count, z_score)
    
    for model, result in all_results['individual_results'].items():
        print(f"   {model:15} -> {result['classification']:15} (conf: {result.get('confidence', 0):.2%})")
    
    print(f"\n   Consensus: {all_results['consensus']} ({all_results['consensus_confidence']:.2%})")
    print(f"   All models agree: {all_results['models_agree']}")
    
    # Performance report
    print("\n3. Performance Report:")
    report = classifier.get_performance_report()
    for model, metrics in report['models'].items():
        print(f"   {model}: {metrics['avg_processing_time_ms']:.2f}ms avg")


if __name__ == "__main__":
    asyncio.run(test_unified_classifier())