"""
DNA Barcode Analysis Pipeline for Tel-Hai CRISPR Screening
Optical Pooled Screening (OPS) Analysis Framework
May 2023 Dataset - Helmsley Science Building
"""

__version__ = "1.0.0"
__author__ = "DNA Analysis Pipeline"
__project__ = "Tel-Hai College CRISPR Screen"

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Experimental parameters from research
BARCODE_LENGTH = 14
PAYLOAD_LENGTH = 96
CONSTRUCT_LENGTH = 110
EXPECTED_BARCODES = 10000
MIN_HAMMING_DISTANCE = 4
AVG_REUSE_RATE = 315
VALID_READ_RATE = 0.445
PLATFORM = "Illumina MiSeq"