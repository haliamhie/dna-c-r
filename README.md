# CRISPR ML Analysis Platform

Machine learning platform for CRISPR sequence classification using multiple models.

## Features

- 10 ML models for sequence classification
- Real-time streaming visualization
- Model comparison interface
- 3D DNA visualization

## Dataset
- 7,071,847 sequences from Tel-Hai College CRISPR screening
- 10,000 unique barcodes
- 14bp barcode length

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dna-crispr-analysis.git
cd dna-crispr-analysis

# Set up environment
conda env create -f environment.yml
conda activate dnaenv
pip install -r backend/requirements.txt

# Prepare data
python scripts/convert_to_parquet.py
```

## Usage

```bash
# Backend
cd backend
python app.py

# Frontend (new terminal)
cd frontend
python3 -m http.server 8080
```

Access at http://localhost:8080

## Models

| Model | Accuracy | Latency |
|-------|----------|---------|
| XGBoost | 100% | 0.73ms |
| Random Forest | 100% | 1.02ms |
| Neural Network | 100% | 0.72ms |
| DNABERT | 79% | 74.4ms |
| Ensemble | 100% | 1.34ms |

## API Endpoints

- `POST /api/classify` - Single sequence classification
- `POST /api/classify-compare` - Compare all models
- `GET /api/models` - List available models
- `GET /api/sequences/top` - Top sequences

## Project Structure
```
dna-crispr-analysis/
├── backend/           # FastAPI server with ML models
├── frontend/          # Web interfaces
├── scripts/           # Data processing
├── artifacts/         # Processed data
└── notebooks/         # Analysis notebooks
```

## Classification Thresholds
```
z_score >= 2.0  → HIGHLY_ENRICHED
z_score >= 1.0  → ENRICHED
z_score >= -1.0 → NEUTRAL
z_score >= -2.0 → DEPLETED
z_score < -2.0  → HIGHLY_DEPLETED
```