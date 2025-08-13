#!/usr/bin/env python3
"""
Machine Learning Classifiers for CRISPR Sequence Analysis
Implements multiple specialized ML models for genomic classification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class GenomicFeatureExtractor:
    """
    Extract biologically relevant features from DNA sequences
    Based on established genomic analysis methods
    """
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, sequence: str) -> np.ndarray:
        """Extract comprehensive genomic features from a DNA sequence"""
        features = []
        
        # 1. Basic composition features
        features.extend(self._get_nucleotide_composition(sequence))
        
        # 2. K-mer frequencies (2-mer to 4-mer)
        features.extend(self._get_kmer_frequencies(sequence, k=2))
        features.extend(self._get_kmer_frequencies(sequence, k=3))
        features.extend(self._get_kmer_frequencies(sequence, k=4))
        
        # 3. Structural features
        features.extend(self._get_structural_features(sequence))
        
        # 4. Thermodynamic features
        features.extend(self._get_thermodynamic_features(sequence))
        
        # 5. CRISPR-specific features
        features.extend(self._get_crispr_features(sequence))
        
        # 6. Sequence complexity metrics
        features.extend(self._get_complexity_features(sequence))
        
        return np.array(features)
    
    def _get_nucleotide_composition(self, seq: str) -> List[float]:
        """Get basic nucleotide composition"""
        length = len(seq)
        return [
            seq.count('A') / length,
            seq.count('T') / length,
            seq.count('G') / length,
            seq.count('C') / length,
            (seq.count('G') + seq.count('C')) / length,  # GC content
            (seq.count('A') + seq.count('T')) / length,  # AT content
        ]
    
    def _get_kmer_frequencies(self, seq: str, k: int) -> List[float]:
        """Calculate k-mer frequencies"""
        from itertools import product
        
        # Generate all possible k-mers
        nucleotides = ['A', 'T', 'G', 'C']
        all_kmers = [''.join(p) for p in product(nucleotides, repeat=k)]
        
        # Count k-mers in sequence
        kmer_counts = {kmer: 0 for kmer in all_kmers}
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
        
        # Normalize by total possible k-mers
        total = len(seq) - k + 1
        if total > 0:
            return [count / total for count in kmer_counts.values()]
        return [0] * len(all_kmers)
    
    def _get_structural_features(self, seq: str) -> List[float]:
        """Calculate DNA structural features"""
        features = []
        
        # Melting temperature approximation (Wallace rule for short sequences)
        tm = 2 * (seq.count('A') + seq.count('T')) + 4 * (seq.count('G') + seq.count('C'))
        features.append(tm / 100.0)  # Normalize
        
        # DNA shape features (simplified)
        # Minor groove width tendency
        at_runs = self._count_homopolymer_runs(seq, ['A', 'T'])
        gc_runs = self._count_homopolymer_runs(seq, ['G', 'C'])
        features.append(at_runs / len(seq))
        features.append(gc_runs / len(seq))
        
        # Bendability (based on dinucleotides)
        bendable_dimers = ['AA', 'TT', 'TA', 'CA', 'TG']
        bend_score = sum(seq.count(dimer) for dimer in bendable_dimers)
        features.append(bend_score / (len(seq) - 1))
        
        return features
    
    def _get_thermodynamic_features(self, seq: str) -> List[float]:
        """Calculate thermodynamic properties"""
        # Nearest-neighbor thermodynamic parameters (simplified)
        nn_params = {
            'AA': -1.0, 'TT': -1.0, 'AT': -0.88, 'TA': -0.58,
            'CA': -1.45, 'TG': -1.45, 'AC': -1.44, 'GT': -1.44,
            'GA': -1.30, 'TC': -1.30, 'AG': -1.28, 'CT': -1.28,
            'CG': -2.17, 'GC': -2.24, 'GG': -1.84, 'CC': -1.84
        }
        
        energy = 0
        for i in range(len(seq) - 1):
            dimer = seq[i:i+2]
            energy += nn_params.get(dimer, 0)
        
        features = [
            energy / (len(seq) - 1),  # Average energy
            abs(energy) / (len(seq) - 1)  # Stability measure
        ]
        
        return features
    
    def _get_crispr_features(self, seq: str) -> List[float]:
        """CRISPR-specific features based on known design rules"""
        features = []
        
        # PAM proximity features (assuming NGG PAM)
        pam_sites = seq.count('GG')
        features.append(pam_sites / len(seq))
        
        # Seed region GC content (positions 1-10 typically)
        seed_region = seq[:10] if len(seq) >= 10 else seq
        seed_gc = (seed_region.count('G') + seed_region.count('C')) / len(seed_region)
        features.append(seed_gc)
        
        # Position-specific nucleotide preferences
        # Based on empirical CRISPR efficiency studies
        position_scores = 0
        if len(seq) >= 14:
            # Favorable nucleotides at specific positions
            if seq[0] in ['G', 'A']: position_scores += 0.1
            if seq[3] == 'C': position_scores += 0.1
            if seq[8] == 'A': position_scores += 0.1
            if seq[13] == 'G': position_scores += 0.1
        features.append(position_scores)
        
        # Self-complementarity (potential for secondary structure)
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        rev_comp = ''.join(complement.get(base, 'N') for base in seq[::-1])
        self_comp_score = sum(1 for i in range(min(len(seq), len(rev_comp))) 
                             if seq[i] == rev_comp[i])
        features.append(self_comp_score / len(seq))
        
        return features
    
    def _get_complexity_features(self, seq: str) -> List[float]:
        """Calculate sequence complexity metrics"""
        features = []
        
        # Shannon entropy
        from collections import Counter
        counts = Counter(seq)
        length = len(seq)
        entropy = -sum((count/length) * np.log2(count/length) 
                      for count in counts.values() if count > 0)
        features.append(entropy / 2.0)  # Normalize by max entropy (2 bits for 4 bases)
        
        # Linguistic complexity
        unique_kmers = set()
        for k in [2, 3]:
            for i in range(len(seq) - k + 1):
                unique_kmers.add(seq[i:i+k])
        possible_kmers = 4**2 + 4**3  # 2-mers and 3-mers
        features.append(len(unique_kmers) / possible_kmers)
        
        # Compression ratio (simplified)
        unique_chars = len(set(seq))
        features.append(unique_chars / 4.0)
        
        return features
    
    def _count_homopolymer_runs(self, seq: str, bases: List[str]) -> int:
        """Count homopolymer runs for specified bases"""
        max_run = 0
        current_run = 0
        
        for base in seq:
            if base in bases:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run


class CRISPRMLClassifier:
    """
    Multi-model ML classifier for CRISPR sequence analysis
    Implements ensemble of specialized genomic ML models
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.feature_extractor = GenomicFeatureExtractor()
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize specialized ML models for genomic classification"""
        
        # 1. Random Forest - good for genomic data with many features
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # 2. XGBoost - state-of-the-art for genomic classification
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # 3. Support Vector Machine - effective for sequence classification
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # 4. Neural Network - can capture complex genomic patterns
        self.models['neural_net'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42
        )
        
        # 5. Gradient Boosting - robust to outliers
        self.models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def train(self, sequences: List[str], labels: List[str], 
              counts: List[int], z_scores: List[float]):
        """Train the ML models on CRISPR sequence data"""
        
        # Extract features
        print(f"Extracting features from {len(sequences)} sequences...")
        X = []
        for i, seq in enumerate(sequences):
            features = self.feature_extractor.extract_features(seq)
            # Add count and z-score as additional features
            features = np.append(features, [counts[i], z_scores[i]])
            X.append(features)
        
        X = np.array(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
        # Train each model
        print("Training models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_scaled, y)
            
            # Calculate cross-validation score
            scores = cross_val_score(model, X_scaled, y, cv=5)
            print(f"    CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        self.is_trained = True
        
    def predict(self, sequence: str, count: int, z_score: float) -> Dict:
        """Predict classification for a single sequence"""
        
        if not self.is_trained:
            # Return rule-based classification if models not trained
            return self._rule_based_classification(sequence, count, z_score)
        
        # Extract features
        features = self.feature_extractor.extract_features(sequence)
        features = np.append(features, [count, z_score])
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            pred_proba = model.predict_proba(X_scaled)[0]
            
            predictions[name] = self.label_encoder.inverse_transform([pred])[0]
            confidences[name] = float(np.max(pred_proba))
        
        # Ensemble prediction (majority vote with confidence weighting)
        if self.model_type == 'ensemble':
            class_votes = {}
            for name, pred in predictions.items():
                if pred not in class_votes:
                    class_votes[pred] = 0
                class_votes[pred] += confidences[name]
            
            final_prediction = max(class_votes, key=class_votes.get)
            final_confidence = class_votes[final_prediction] / sum(class_votes.values())
        else:
            # Use specific model
            final_prediction = predictions.get(self.model_type, 'NEUTRAL')
            final_confidence = confidences.get(self.model_type, 0.5)
        
        # Generate explanation
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
        
        return {
            'classification': final_prediction,
            'confidence': final_confidence,
            'model': self.model_type,
            'model_predictions': predictions,
            'model_confidences': confidences,
            'gc_content': gc_content,
            'reason': self._generate_explanation(final_prediction, count, z_score, gc_content)
        }
    
    def _rule_based_classification(self, sequence: str, count: int, z_score: float) -> Dict:
        """Fallback rule-based classification"""
        
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
        
        # Classification based on statistical and biological rules
        if z_score >= 2.0 and count > 600:
            classification = 'HIGHLY_ENRICHED'
            confidence = 0.9
        elif z_score >= 1.0 and count > 300:
            classification = 'ENRICHED'
            confidence = 0.8
        elif z_score <= -2.0:
            classification = 'HIGHLY_DEPLETED'
            confidence = 0.9
        elif z_score <= -1.0:
            classification = 'DEPLETED'
            confidence = 0.8
        else:
            classification = 'NEUTRAL'
            confidence = 0.7
        
        # Adjust based on GC content
        if 40 <= gc_content <= 60:
            confidence += 0.05
        else:
            confidence -= 0.05
        
        confidence = min(max(confidence, 0.0), 1.0)
        
        return {
            'classification': classification,
            'confidence': confidence,
            'model': 'rule_based',
            'gc_content': gc_content,
            'reason': self._generate_explanation(classification, count, z_score, gc_content)
        }
    
    def _generate_explanation(self, classification: str, count: int, 
                             z_score: float, gc_content: float) -> str:
        """Generate human-readable explanation for classification"""
        
        reasons = []
        
        if z_score >= 2.0:
            reasons.append("Highly significant positive z-score")
        elif z_score >= 1.0:
            reasons.append("Significant positive z-score")
        elif z_score <= -2.0:
            reasons.append("Highly significant negative z-score")
        elif z_score <= -1.0:
            reasons.append("Significant negative z-score")
        else:
            reasons.append("Neutral z-score")
        
        if count > 600:
            reasons.append("High sequence abundance")
        elif count > 300:
            reasons.append("Moderate sequence abundance")
        elif count < 100:
            reasons.append("Low sequence abundance")
        
        if 40 <= gc_content <= 60:
            reasons.append("Optimal GC content for CRISPR")
        elif gc_content < 30:
            reasons.append("Low GC content")
        elif gc_content > 70:
            reasons.append("High GC content")
        
        return "; ".join(reasons)
    
    def save_models(self, path: Path):
        """Save trained models to disk"""
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save each model
        for name, model in self.models.items():
            joblib.dump(model, save_dir / f"{name}_model.pkl")
        
        # Save scaler and label encoder
        joblib.dump(self.scaler, save_dir / "scaler.pkl")
        joblib.dump(self.label_encoder, save_dir / "label_encoder.pkl")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'saved_at': datetime.now().isoformat()
        }
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load_models(self, path: Path):
        """Load trained models from disk"""
        load_dir = Path(path)
        
        # Load models
        for name in self.models.keys():
            model_path = load_dir / f"{name}_model.pkl"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
        
        # Load scaler and label encoder
        scaler_path = load_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        encoder_path = load_dir / "label_encoder.pkl"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
        
        # Load metadata
        metadata_path = load_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.is_trained = metadata.get('is_trained', False)


# Specialized genomic models wrapper
class SpecializedGenomicModels:
    """
    Wrapper for specialized genomic ML models
    These would typically use pre-trained models from:
    - DeepCRISPR: Deep learning for CRISPR/Cas9 activity prediction
    - Azimuth: Microsoft's CRISPR scoring model
    - CRISPOR: CRISPR efficiency prediction
    """
    
    @staticmethod
    def deepcrispr_score(sequence: str) -> float:
        """
        Simulate DeepCRISPR scoring
        In production, this would call the actual DeepCRISPR model
        """
        # DeepCRISPR uses CNN + RNN architecture
        # Trained on large-scale CRISPR screening data
        
        # Simplified scoring based on DeepCRISPR principles
        score = 0.5
        
        # Favorable features from DeepCRISPR paper
        if sequence[0] == 'G': score += 0.1
        if sequence[-1] == 'G': score += 0.1
        
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        if 0.4 <= gc_content <= 0.6:
            score += 0.2
        
        # Penalize homopolymers
        for base in ['A', 'T', 'G', 'C']:
            if base * 4 in sequence:
                score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    @staticmethod
    def azimuth_score(sequence: str) -> float:
        """
        Simulate Azimuth (Rule Set 2) scoring
        Based on Doench et al. 2016
        """
        # Azimuth uses gradient-boosted regression trees
        # with position-specific and position-independent features
        
        score = 0.5
        
        # Position-specific nucleotide preferences (simplified)
        position_weights = {
            (0, 'G'): 0.05, (0, 'A'): 0.03,
            (3, 'C'): 0.04,
            (7, 'G'): 0.03,
            (9, 'C'): 0.04,
            (13, 'G'): 0.05
        }
        
        for (pos, base), weight in position_weights.items():
            if pos < len(sequence) and sequence[pos] == base:
                score += weight
        
        # Dinucleotide features
        favorable_dimers = ['GC', 'CC', 'GG']
        unfavorable_dimers = ['TT', 'AA']
        
        for dimer in favorable_dimers:
            score += sequence.count(dimer) * 0.02
        
        for dimer in unfavorable_dimers:
            score -= sequence.count(dimer) * 0.02
        
        return min(max(score, 0.0), 1.0)
    
    @staticmethod
    def crispor_score(sequence: str) -> float:
        """
        Simulate CRISPOR scoring
        Aggregates multiple scoring methods
        """
        # CRISPOR combines multiple models:
        # Doench 2014, Doench 2016 (Azimuth), Moreno-Mateos, etc.
        
        # Weighted average of different scoring methods
        deepcrispr = SpecializedGenomicModels.deepcrispr_score(sequence)
        azimuth = SpecializedGenomicModels.azimuth_score(sequence)
        
        # Additional CRISPOR-specific features
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        
        # Off-target potential (simplified)
        off_target_penalty = 0
        if sequence[:10].count('G') + sequence[:10].count('C') > 7:
            off_target_penalty = 0.1  # High GC in seed region
        
        # Weighted combination
        score = (deepcrispr * 0.3 + azimuth * 0.4 + 
                (0.5 if 0.3 <= gc_content <= 0.7 else 0.2) * 0.3 - 
                off_target_penalty)
        
        return min(max(score, 0.0), 1.0)


def test_ml_classifiers():
    """Test the ML classification system"""
    
    # Test sequences with known characteristics
    test_data = [
        ("GTCTTTCTGCTCGT", 800, 2.5, "HIGHLY_ENRICHED"),
        ("GGCGCTTCATGGTC", 400, 1.2, "ENRICHED"),
        ("ACGTACGTACGTAC", 200, 0.0, "NEUTRAL"),
        ("TTTTTTTTTTTTTT", 50, -1.5, "DEPLETED"),
        ("AAAAAAAACCCCCC", 20, -2.5, "HIGHLY_DEPLETED"),
    ]
    
    # Initialize classifier
    classifier = CRISPRMLClassifier(model_type='ensemble')
    
    print("=" * 60)
    print("CRISPR ML Classification System Test")
    print("=" * 60)
    
    for seq, count, z_score, expected in test_data:
        result = classifier.predict(seq, count, z_score)
        
        print(f"\nSequence: {seq}")
        print(f"Expected: {expected}")
        print(f"Predicted: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"GC Content: {result['gc_content']:.1f}%")
        print(f"Reason: {result['reason']}")
        
        # Test specialized models
        print("\nSpecialized Model Scores:")
        print(f"  DeepCRISPR: {SpecializedGenomicModels.deepcrispr_score(seq):.3f}")
        print(f"  Azimuth: {SpecializedGenomicModels.azimuth_score(seq):.3f}")
        print(f"  CRISPOR: {SpecializedGenomicModels.crispor_score(seq):.3f}")
    
    print("\n" + "=" * 60)
    print("Test Complete")


if __name__ == "__main__":
    test_ml_classifiers()