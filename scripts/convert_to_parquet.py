#!/usr/bin/env python3
"""
Convert CSV to Parquet format with partitioning for efficient processing.
Sanitizes barcode sequences and creates DuckDB database.
"""

import os
import sys
import pandas as pd
import duckdb
from pathlib import Path
from tqdm import tqdm

def main():
    # Setup paths
    root = Path('/home/mch/dna')
    data_dir = root / 'DNA-Data for Telhai' / '2023-05-11'
    art = root / 'artifacts'
    art.mkdir(exist_ok=True)
    
    parq_dir = art / 'clusters_parquet'
    parq_dir.mkdir(exist_ok=True)
    
    clusters = data_dir / 'clusters.csv'
    
    if not clusters.exists():
        print(f"Error: {clusters} not found")
        sys.exit(1)
    
    print(f"Converting {clusters} to Parquet format...")
    
    # Get total rows for progress bar
    with open(clusters, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header
    
    # Stream read and sanitize barcode
    chunksize = 500_000
    chunks = pd.read_csv(clusters, chunksize=chunksize)
    
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        for i, chunk in enumerate(chunks):
            if 'barcode' in chunk.columns:
                # Sanitize barcode: uppercase and keep only ACGT
                chunk['barcode'] = (chunk['barcode']
                                    .astype('string')
                                    .str.upper()
                                    .str.replace(r"[^ACGT]", "", regex=True))
            
            out = parq_dir / f'part_{i:05d}.parquet'
            chunk.to_parquet(out, index=False)
            pbar.update(len(chunk))
    
    print(f"Parquet files written to {parq_dir}")
    
    # Create DuckDB database with view
    db_path = art / 'dna.duckdb'
    print(f"Creating DuckDB database at {db_path}...")
    
    con = duckdb.connect(str(db_path))
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute(
        f"CREATE OR REPLACE VIEW clusters AS SELECT * FROM read_parquet('{str(parq_dir / '*.parquet')}')"
    )
    
    # Verify
    count = con.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    print(f"DuckDB view created successfully with {count:,} rows")
    
    # Create some useful summary views
    con.execute("""
        CREATE OR REPLACE VIEW barcode_stats AS
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT barcode) as unique_barcodes,
            COUNT(CASE WHEN barcode IS NULL OR barcode = '' THEN 1 END) as null_barcodes,
            COUNT(CASE WHEN barcode IS NOT NULL AND barcode != '' THEN 1 END) as valid_barcodes
        FROM clusters
    """)
    
    con.execute("""
        CREATE OR REPLACE VIEW barcode_lengths AS
        SELECT 
            LENGTH(barcode) as barcode_length,
            COUNT(*) as count
        FROM clusters
        WHERE barcode IS NOT NULL AND barcode != ''
        GROUP BY 1
        ORDER BY 2 DESC
    """)
    
    print("Additional views created: barcode_stats, barcode_lengths")
    
    con.close()
    print("Conversion complete!")

if __name__ == "__main__":
    main()