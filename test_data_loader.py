#!/usr/bin/env python3
"""Test data loader module"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader, load_telhai_data

def test_loader():
    """Test data loading functionality"""
    loader = DataLoader()
    
    # Test with micro_design
    design_path = Path("/home/mch/dna/updated_data/micro_design.csv")
    if design_path.exists():
        df = loader.load(design_path)
        validation = loader.validate_barcode_data(df)
        summary = loader.get_data_summary(df)
        
        print(f"✓ Loaded {df.shape[0]} barcodes")
        print(f"✓ Validation: {validation['barcode_length_consistent']}")
        print(f"✓ Memory: {summary['memory_usage_mb']:.1f}MB")
        return True
    return False

if __name__ == "__main__":
    success = test_loader()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)