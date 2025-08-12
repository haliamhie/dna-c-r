#!/usr/bin/env python3
"""
Create Jupyter notebooks with prefilled cells for DNA sequence analysis.
"""

from pathlib import Path
import nbformat as nbf

def create_inventory_notebook():
    """Create notebook for data inventory and file exploration."""
    cells = [
        nbf.v4.new_markdown_cell("""# 01 · Data Inventory
        
This notebook explores the DNA sequence data structure and file organization.

## Project Structure
- **Raw Data**: `DNA-Data for Telhai/2023-05-11/`
- **Processed Data**: `artifacts/`
- **Logs**: `logs/`
- **Scratch Space**: `scratch/`
"""),
        nbf.v4.new_code_cell("""# Import libraries and set paths
from pathlib import Path
import os
import pandas as pd

root = Path("/home/mch/dna")
DATA_DIR = root / "DNA-Data for Telhai" / "2023-05-11"
ARTIFACTS = root / "artifacts"
LOGS = root / "logs"

print(f"Root directory: {root}")
print(f"Data directory: {DATA_DIR}")
print(f"Data exists: {DATA_DIR.exists()}")"""),
        
        nbf.v4.new_code_cell("""# Check raw data files
clusters_csv = DATA_DIR / "clusters.csv"
zip_dir = DATA_DIR / "7zip"

print(f"\\nClusters CSV:")
print(f"  Path: {clusters_csv}")
print(f"  Exists: {clusters_csv.exists()}")
if clusters_csv.exists():
    size_mb = clusters_csv.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    
print(f"\\n7zip directory:")
print(f"  Path: {zip_dir}")
if zip_dir.exists():
    for f in zip_dir.iterdir():
        print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")"""),
        
        nbf.v4.new_code_cell("""# Quick preview of clusters.csv
if clusters_csv.exists():
    df_sample = pd.read_csv(clusters_csv, nrows=10)
    print(f"Shape (first 10 rows): {df_sample.shape}")
    print(f"Columns: {list(df_sample.columns)}")
    print("\\nFirst 5 rows:")
    display(df_sample.head())
    
    # Check for null values
    print(f"\\nNull values in sample:")
    print(df_sample.isnull().sum())"""),
        
        nbf.v4.new_code_cell("""# Check processed data
parquet_dir = ARTIFACTS / "clusters_parquet"
duckdb_file = ARTIFACTS / "dna.duckdb"

print("Processed data status:")
print(f"  Parquet directory exists: {parquet_dir.exists()}")
if parquet_dir.exists():
    parquet_files = list(parquet_dir.glob("*.parquet"))
    print(f"  Number of parquet files: {len(parquet_files)}")
    if parquet_files:
        total_size = sum(f.stat().st_size for f in parquet_files) / (1024*1024)
        print(f"  Total size: {total_size:.2f} MB")

print(f"\\n  DuckDB file exists: {duckdb_file.exists()}")
if duckdb_file.exists():
    print(f"  Size: {duckdb_file.stat().st_size / (1024*1024):.2f} MB")""")
    ]
    
    return cells

def create_convert_notebook():
    """Create notebook for data conversion process."""
    cells = [
        nbf.v4.new_markdown_cell("""# 02 · Data Conversion: CSV → Parquet → DuckDB

This notebook handles the conversion of raw CSV data to optimized Parquet format and creates a DuckDB database for fast queries.

## Conversion Strategy
1. **Stream Processing**: Read CSV in chunks to handle large file
2. **Data Sanitization**: Clean and standardize barcode sequences
3. **Partitioning**: Split into multiple Parquet files for parallel processing
4. **Database Views**: Create DuckDB views for efficient querying
"""),
        
        nbf.v4.new_code_cell("""# Import required libraries
import duckdb
import pandas as pd
from pathlib import Path
import sys

root = Path("/home/mch/dna")
sys.path.append(str(root / "scripts"))

# Paths
DATA_DIR = root / "DNA-Data for Telhai" / "2023-05-11"
ARTIFACTS = root / "artifacts"
clusters_csv = DATA_DIR / "clusters.csv"
parquet_dir = ARTIFACTS / "clusters_parquet"
db_path = ARTIFACTS / "dna.duckdb"

print(f"Source CSV: {clusters_csv}")
print(f"Target Parquet: {parquet_dir}")
print(f"Target DB: {db_path}")"""),
        
        nbf.v4.new_code_cell("""# Option 1: Run the conversion script
# Uncomment to run the full conversion (may take a few minutes)
# !python /home/mch/dna/scripts/convert_to_parquet.py 2>&1 | tee /home/mch/dna/logs/conversion.log"""),
        
        nbf.v4.new_code_cell("""# Option 2: Manual conversion with progress tracking
from tqdm.notebook import tqdm

def convert_csv_to_parquet(csv_path, parquet_dir, chunksize=500_000):
    \"\"\"Convert CSV to Parquet with chunking and progress bar.\"\"\"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    # Count total rows for progress bar
    with open(csv_path, 'r') as f:
        total_rows = sum(1 for _ in f) - 1
    
    chunks = pd.read_csv(csv_path, chunksize=chunksize)
    
    with tqdm(total=total_rows, desc="Converting") as pbar:
        for i, chunk in enumerate(chunks):
            # Sanitize barcode column
            if 'barcode' in chunk.columns:
                chunk['barcode'] = (chunk['barcode']
                                    .astype('string')
                                    .str.upper()
                                    .str.replace(r"[^ACGT]", "", regex=True))
            
            # Write to parquet
            out_path = parquet_dir / f'part_{i:05d}.parquet'
            chunk.to_parquet(out_path, index=False)
            pbar.update(len(chunk))
    
    return i + 1  # Return number of parts created

# Uncomment to run conversion
# num_parts = convert_csv_to_parquet(clusters_csv, parquet_dir)
# print(f"Created {num_parts} parquet files")"""),
        
        nbf.v4.new_code_cell("""# Connect to DuckDB and create/verify views
con = duckdb.connect(str(db_path))

# Install and load parquet extension
con.execute("INSTALL parquet; LOAD parquet;")

# Create main view
con.execute(
    "CREATE OR REPLACE VIEW clusters AS SELECT * FROM read_parquet(?)",
    [str(parquet_dir / "*.parquet")]
)

# Verify the view
result = con.sql("SELECT COUNT(*) as total_rows FROM clusters")
print("Database view created successfully!")
result.show()"""),
        
        nbf.v4.new_code_cell("""# Create additional analytical views
con.execute(\"\"\"
    CREATE OR REPLACE VIEW barcode_stats AS
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT barcode) as unique_barcodes,
        COUNT(CASE WHEN barcode IS NULL OR barcode = '' THEN 1 END) as null_barcodes,
        COUNT(CASE WHEN barcode IS NOT NULL AND barcode != '' THEN 1 END) as valid_barcodes
    FROM clusters
\"\"\")

con.execute(\"\"\"
    CREATE OR REPLACE VIEW barcode_lengths AS
    SELECT 
        LENGTH(barcode) as barcode_length,
        COUNT(*) as count
    FROM clusters
    WHERE barcode IS NOT NULL AND barcode != ''
    GROUP BY 1
    ORDER BY 2 DESC
\"\"\")

print("Analytical views created:")
print("- barcode_stats: Overall statistics")
print("- barcode_lengths: Distribution of barcode lengths")

# Show stats
con.sql("SELECT * FROM barcode_stats").show()"""),
        
        nbf.v4.new_code_cell("""# Close connection
con.close()
print("Database setup complete!")""")
    ]
    
    return cells

def create_eda_notebook():
    """Create notebook for exploratory data analysis."""
    cells = [
        nbf.v4.new_markdown_cell("""# 03 · Exploratory Data Analysis

This notebook performs comprehensive EDA on the DNA sequence data using DuckDB for efficient queries.

## Analysis Sections
1. Basic statistics and data quality
2. Barcode sequence patterns
3. UUID distribution analysis
4. Data completeness assessment
"""),
        
        nbf.v4.new_code_cell("""# Setup and connect to database
import duckdb
import pandas as pd
import os
from pathlib import Path

root = Path("/home/mch/dna")
db_path = root / "artifacts" / "dna.duckdb"

# Connect to database
con = duckdb.connect(str(db_path))
print(f"Connected to: {db_path}")

# Verify tables/views
tables = con.sql("SHOW TABLES").df()
print(f"\\nAvailable views: {list(tables['name'])}")"""),
        
        nbf.v4.new_markdown_cell("""## 1. Basic Statistics"""),
        
        nbf.v4.new_code_cell("""# Overall statistics
print("=== OVERALL STATISTICS ===")
con.sql("SELECT * FROM barcode_stats").show()"""),
        
        nbf.v4.new_code_cell("""# Sample of data
print("=== SAMPLE DATA (10 rows) ===")
con.sql(\"\"\"
    SELECT id, barcode, LENGTH(barcode) as barcode_len
    FROM clusters
    WHERE barcode IS NOT NULL AND barcode != ''
    LIMIT 10
\"\"\").show()"""),
        
        nbf.v4.new_markdown_cell("""## 2. Barcode Analysis"""),
        
        nbf.v4.new_code_cell("""# Barcode length distribution
print("=== BARCODE LENGTH DISTRIBUTION ===")
con.sql(\"\"\"
    SELECT barcode_length, count, 
           ROUND(100.0 * count / SUM(count) OVER (), 2) as percentage
    FROM barcode_lengths
    LIMIT 15
\"\"\").show()"""),
        
        nbf.v4.new_code_cell("""# Most common barcodes
print("=== TOP 10 MOST FREQUENT BARCODES ===")
con.sql(\"\"\"
    SELECT barcode, COUNT(*) as frequency,
           ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM clusters WHERE barcode IS NOT NULL), 4) as percentage
    FROM clusters
    WHERE barcode IS NOT NULL AND barcode != ''
    GROUP BY barcode
    ORDER BY frequency DESC
    LIMIT 10
\"\"\").show()"""),
        
        nbf.v4.new_code_cell("""# Nucleotide composition
print("=== NUCLEOTIDE COMPOSITION IN BARCODES ===")
con.sql(\"\"\"
    WITH nucleotide_counts AS (
        SELECT 
            SUM(LENGTH(barcode) - LENGTH(REPLACE(barcode, 'A', ''))) as A_count,
            SUM(LENGTH(barcode) - LENGTH(REPLACE(barcode, 'C', ''))) as C_count,
            SUM(LENGTH(barcode) - LENGTH(REPLACE(barcode, 'G', ''))) as G_count,
            SUM(LENGTH(barcode) - LENGTH(REPLACE(barcode, 'T', ''))) as T_count,
            SUM(LENGTH(barcode)) as total_bases
        FROM clusters
        WHERE barcode IS NOT NULL AND barcode != ''
    )
    SELECT 
        A_count, 
        ROUND(100.0 * A_count / total_bases, 2) as A_pct,
        C_count,
        ROUND(100.0 * C_count / total_bases, 2) as C_pct,
        G_count,
        ROUND(100.0 * G_count / total_bases, 2) as G_pct,
        T_count,
        ROUND(100.0 * T_count / total_bases, 2) as T_pct,
        total_bases
    FROM nucleotide_counts
\"\"\").show()"""),
        
        nbf.v4.new_markdown_cell("""## 3. UUID Analysis"""),
        
        nbf.v4.new_code_cell("""# Check UUID format compliance
print("=== UUID FORMAT ANALYSIS ===")
con.sql(\"\"\"
    SELECT 
        COUNT(*) as total_ids,
        COUNT(CASE WHEN id ~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' 
                   THEN 1 END) as valid_uuid_format,
        COUNT(CASE WHEN id NOT SIMILAR TO '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' 
                   THEN 1 END) as invalid_uuid_format
    FROM clusters
\"\"\").show()"""),
        
        nbf.v4.new_code_cell("""# Check for duplicate IDs
print("=== DUPLICATE ID CHECK ===")
con.sql(\"\"\"
    WITH id_counts AS (
        SELECT id, COUNT(*) as count
        FROM clusters
        GROUP BY id
        HAVING COUNT(*) > 1
    )
    SELECT 
        COUNT(*) as duplicate_ids,
        SUM(count) as total_duplicate_rows
    FROM id_counts
\"\"\").show()"""),
        
        nbf.v4.new_markdown_cell("""## 4. Data Quality Summary"""),
        
        nbf.v4.new_code_cell("""# Comprehensive data quality report
print("=== DATA QUALITY REPORT ===\\n")

# Get stats
stats = con.sql("SELECT * FROM barcode_stats").df().iloc[0]

print(f"Total Records: {stats['total_rows']:,}")
print(f"Unique Barcodes: {stats['unique_barcodes']:,}")
print(f"Records with Barcode: {stats['valid_barcodes']:,} ({100*stats['valid_barcodes']/stats['total_rows']:.1f}%)")
print(f"Records without Barcode: {stats['null_barcodes']:,} ({100*stats['null_barcodes']/stats['total_rows']:.1f}%)")

# Barcode uniqueness
if stats['valid_barcodes'] > 0:
    duplication_rate = 1 - (stats['unique_barcodes'] / stats['valid_barcodes'])
    print(f"\\nBarcode Duplication Rate: {100*duplication_rate:.2f}%")
    print(f"Average occurrences per barcode: {stats['valid_barcodes']/stats['unique_barcodes']:.2f}")"""),
        
        nbf.v4.new_code_cell("""# Export summary statistics to CSV (optional)
# summary_df = con.sql("SELECT * FROM barcode_stats").df()
# summary_df.to_csv(root / "artifacts" / "data_summary.csv", index=False)
# print("Summary exported to artifacts/data_summary.csv")"""),
        
        nbf.v4.new_code_cell("""# Close database connection
con.close()
print("\\nAnalysis complete!")""")
    ]
    
    return cells

def main():
    """Create all notebooks."""
    root = Path('/home/mch/dna')
    nb_dir = root / 'notebooks'
    nb_dir.mkdir(exist_ok=True)
    
    # Create notebooks
    notebooks = [
        ('01_inventory', create_inventory_notebook()),
        ('02_convert', create_convert_notebook()),
        ('03_eda', create_eda_notebook())
    ]
    
    for name, cells in notebooks:
        nb = nbf.v4.new_notebook()
        nb['cells'] = cells
        
        # Add kernel spec
        nb['metadata'] = {
            'kernelspec': {
                'display_name': 'Python (dnaenv)',
                'language': 'python',
                'name': 'dnaenv'
            },
            'language_info': {
                'name': 'python',
                'version': '3.11'
            }
        }
        
        # Write notebook
        output_path = nb_dir / f'{name}.ipynb'
        nbf.write(nb, output_path)
        print(f"Created: {output_path}")
    
    print(f"\nAll notebooks created in {nb_dir}")

if __name__ == "__main__":
    main()