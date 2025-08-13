#!/usr/bin/env python3
"""Deep analysis of DNA barcode patterns and distributions"""

import duckdb
import pandas as pd
from pathlib import Path
import numpy as np

# Connect to database
db_path = Path('/home/mch/dna/artifacts/dna.duckdb')
con = duckdb.connect(str(db_path))

print('=' * 70)
print('ADVANCED DNA SEQUENCE ANALYSIS')
print('=' * 70)

# 1. Barcode Distribution Analysis
print('\n1. BARCODE FREQUENCY DISTRIBUTION')
print('-' * 40)
distribution = con.execute("""
    WITH barcode_freqs AS (
        SELECT barcode, COUNT(*) as frequency
        FROM clusters
        WHERE barcode IS NOT NULL AND barcode != ''
        GROUP BY barcode
    ),
    freq_bins AS (
        SELECT 
            CASE 
                WHEN frequency < 200 THEN '< 200'
                WHEN frequency < 300 THEN '200-299'
                WHEN frequency < 400 THEN '300-399'
                WHEN frequency < 500 THEN '400-499'
                WHEN frequency < 600 THEN '500-599'
                ELSE '600+'
            END as freq_range,
            COUNT(*) as barcode_count
        FROM barcode_freqs
        GROUP BY freq_range
        ORDER BY MIN(frequency)
    )
    SELECT * FROM freq_bins
""").fetchdf()
print("Frequency Range | Number of Barcodes")
print("-" * 40)
for _, row in distribution.iterrows():
    print(f"{row['freq_range']:12s} | {row['barcode_count']:,}")

# 2. Position-specific nucleotide analysis
print('\n2. POSITION-SPECIFIC NUCLEOTIDE FREQUENCY')
print('-' * 40)
print("Position | A% | C% | G% | T% | Entropy")
print("-" * 40)

for pos in range(1, 15):
    pos_analysis = con.execute(f"""
        SELECT 
            SUM(CASE WHEN SUBSTRING(barcode, {pos}, 1) = 'A' THEN 1 ELSE 0 END) as A,
            SUM(CASE WHEN SUBSTRING(barcode, {pos}, 1) = 'C' THEN 1 ELSE 0 END) as C,
            SUM(CASE WHEN SUBSTRING(barcode, {pos}, 1) = 'G' THEN 1 ELSE 0 END) as G,
            SUM(CASE WHEN SUBSTRING(barcode, {pos}, 1) = 'T' THEN 1 ELSE 0 END) as T,
            COUNT(*) as total
        FROM clusters
        WHERE barcode IS NOT NULL AND barcode != ''
    """).fetchone()
    
    total = pos_analysis[4]
    freqs = [pos_analysis[i]/total for i in range(4)]
    
    # Calculate Shannon entropy
    entropy = -sum([f * np.log2(f) if f > 0 else 0 for f in freqs])
    
    print(f"   {pos:2d}    | {freqs[0]*100:3.0f} | {freqs[1]*100:3.0f} | {freqs[2]*100:3.0f} | {freqs[3]*100:3.0f} | {entropy:.3f}")

# 3. Dinucleotide analysis
print('\n3. DINUCLEOTIDE FREQUENCY ANALYSIS')
print('-' * 40)
dinucleotides = con.execute("""
    WITH dinucs AS (
        SELECT 
            SUBSTRING(barcode, 1, 2) as dinuc FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 2, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 3, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 4, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 5, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 6, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 7, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 8, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 9, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 10, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 11, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 12, 2) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 13, 2) FROM clusters WHERE barcode IS NOT NULL
    )
    SELECT dinuc, COUNT(*) as count
    FROM dinucs
    GROUP BY dinuc
    ORDER BY count DESC
    LIMIT 10
""").fetchdf()

print("Top 10 Most Common Dinucleotides:")
for _, row in dinucleotides.iterrows():
    pct = row['count'] / (3149475 * 13) * 100  # 13 dinucleotides per 14bp sequence
    print(f"  {row['dinuc']}: {row['count']:,} ({pct:.2f}%)")

# 4. GC content distribution
print('\n4. PER-BARCODE GC CONTENT DISTRIBUTION')
print('-' * 40)
gc_dist = con.execute("""
    WITH gc_calc AS (
        SELECT 
            barcode,
            (LENGTH(barcode) - LENGTH(REPLACE(REPLACE(barcode, 'G', ''), 'C', ''))) * 100.0 / LENGTH(barcode) as gc_pct
        FROM clusters
        WHERE barcode IS NOT NULL AND barcode != ''
    )
    SELECT 
        ROUND(gc_pct / 10) * 10 as gc_range,
        COUNT(*) as count
    FROM gc_calc
    GROUP BY gc_range
    ORDER BY gc_range
""").fetchdf()

print("GC% Range | Count")
print("-" * 40)
for _, row in gc_dist.iterrows():
    print(f"  {int(row['gc_range']):2d}-{int(row['gc_range'])+9:2d}%  | {row['count']:,}")

# 5. Sequence motif search
print('\n5. COMMON SEQUENCE MOTIFS (3-mers)')
print('-' * 40)
motifs = con.execute("""
    WITH trimers AS (
        SELECT SUBSTRING(barcode, 1, 3) as motif FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 4, 3) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 7, 3) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 10, 3) FROM clusters WHERE barcode IS NOT NULL
        UNION ALL
        SELECT SUBSTRING(barcode, 12, 3) FROM clusters WHERE barcode IS NOT NULL
    )
    SELECT motif, COUNT(*) as count
    FROM trimers
    GROUP BY motif
    ORDER BY count DESC
    LIMIT 10
""").fetchdf()

print("Top 10 Most Common 3-mer Motifs:")
for _, row in motifs.iterrows():
    print(f"  {row['motif']}: {row['count']:,}")

# 6. Palindromic sequences
print('\n6. PALINDROMIC SEQUENCE ANALYSIS')
print('-' * 40)
palindromes = con.execute("""
    SELECT COUNT(*) as palindrome_count
    FROM clusters
    WHERE barcode IS NOT NULL 
    AND barcode = REVERSE(REPLACE(REPLACE(REPLACE(REPLACE(barcode, 'A', 'X'), 'T', 'A'), 'X', 'T'), 'G', 'Y'), 'C', 'G'), 'Y', 'C')
""").fetchone()
print(f"Palindromic sequences: {palindromes[0]:,}")

# 7. Homopolymer runs
print('\n7. HOMOPOLYMER RUN DETECTION')
print('-' * 40)
for base in ['A', 'T', 'G', 'C']:
    runs = con.execute(f"""
        SELECT 
            COUNT(CASE WHEN barcode LIKE '%{base*3}%' THEN 1 END) as triple,
            COUNT(CASE WHEN barcode LIKE '%{base*4}%' THEN 1 END) as quad,
            COUNT(CASE WHEN barcode LIKE '%{base*5}%' THEN 1 END) as penta
        FROM clusters
        WHERE barcode IS NOT NULL
    """).fetchone()
    print(f"{base}-runs: 3x={runs[0]:,}, 4x={runs[1]:,}, 5x={runs[2]:,}")

# 8. UUID pattern analysis
print('\n8. UUID PATTERN ANALYSIS')
print('-' * 40)
uuid_patterns = con.execute("""
    SELECT 
        SUBSTRING(id, 1, 8) as prefix,
        COUNT(*) as count
    FROM clusters
    GROUP BY prefix
    ORDER BY count DESC
    LIMIT 5
""").fetchdf()
print("Top 5 UUID Prefixes:")
for _, row in uuid_patterns.iterrows():
    print(f"  {row['prefix']}: {row['count']:,} IDs")

# 9. Barcode complexity score
print('\n9. SEQUENCE COMPLEXITY ANALYSIS')
print('-' * 40)
complexity = con.execute("""
    WITH complexity_scores AS (
        SELECT 
            barcode,
            LENGTH(barcode) - LENGTH(REPLACE(REPLACE(REPLACE(REPLACE(barcode, 'A', ''), 'T', ''), 'G', ''), 'C', '')) as unique_bases,
            CASE 
                WHEN barcode LIKE '%AAA%' OR barcode LIKE '%TTT%' OR barcode LIKE '%GGG%' OR barcode LIKE '%CCC%' THEN 1 
                ELSE 0 
            END as has_repeat
        FROM clusters
        WHERE barcode IS NOT NULL AND barcode != ''
    )
    SELECT 
        AVG(unique_bases) as avg_unique,
        SUM(has_repeat) * 100.0 / COUNT(*) as pct_with_repeats
    FROM complexity_scores
""").fetchone()
print(f"Average unique nucleotides per barcode: {complexity[0]:.1f}/4")
print(f"Barcodes with 3+ repeats: {complexity[1]:.1f}%")

con.close()

print('\n' + '=' * 70)
print('ANALYSIS COMPLETE')
print('=' * 70)