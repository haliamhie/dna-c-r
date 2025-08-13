#!/usr/bin/env python3
import duckdb
from pathlib import Path

root = Path('/home/mch/dna')
db_path = root / 'artifacts' / 'dna.duckdb'

con = duckdb.connect(str(db_path))

# Create analytical views
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

print('Created analytical views: barcode_stats, barcode_lengths')

# Test the views
stats = con.execute('SELECT * FROM barcode_stats').fetchone()
print(f'\nDatabase Statistics:')
print(f'  Total rows: {stats[0]:,}')
print(f'  Unique barcodes: {stats[1]:,}')
print(f'  Null/empty barcodes: {stats[2]:,}')
print(f'  Valid barcodes: {stats[3]:,}')

con.close()
print('\nDatabase setup complete!')