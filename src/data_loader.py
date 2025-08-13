#!/usr/bin/env python3
"""
Unified Data Loader for Tel-Hai CRISPR Screening Pipeline
Handles CSV, Parquet, DuckDB, and compressed formats
"""

import pandas as pd
import duckdb
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator
import logging
from tqdm import tqdm
import hashlib
import zipfile
import gzip

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader with support for multiple formats and chunked reading
    Optimized for Tel-Hai's 7M read dataset with 55.5% failure rate
    """
    
    def __init__(self, chunk_size: int = 500_000):
        """
        Initialize data loader
        
        Args:
            chunk_size: Default chunk size for large file processing
        """
        self.chunk_size = chunk_size
        self.supported_formats = {'.csv', '.parquet', '.duckdb', '.zip', '.gz', '.7z'}
        self._file_cache = {}
        
    def detect_format(self, file_path: Path) -> str:
        """Detect file format from extension and magic bytes"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check extension
        suffix = file_path.suffix.lower()
        
        # Check for compressed files
        if suffix in ['.zip', '.gz', '.7z']:
            return 'compressed'
        elif suffix == '.csv':
            return 'csv'
        elif suffix == '.parquet':
            return 'parquet'
        elif suffix == '.duckdb':
            return 'duckdb'
        else:
            # Try to detect from content
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header.startswith(b'PAR1'):
                    return 'parquet'
                elif header.startswith(b'SQLite'):
                    return 'duckdb'
                else:
                    return 'csv'  # Default assumption
    
    def load_csv(self, file_path: Path, 
                 columns: Optional[List[str]] = None,
                 chunked: bool = False) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Load CSV file with optional chunking for large files
        
        Args:
            file_path: Path to CSV file
            columns: Specific columns to load
            chunked: Return iterator for chunked reading
            
        Returns:
            DataFrame or iterator of DataFrames
        """
        logger.info(f"Loading CSV: {file_path}")
        
        if chunked:
            return pd.read_csv(
                file_path,
                usecols=columns,
                chunksize=self.chunk_size,
                low_memory=False
            )
        else:
            # For large files, show progress
            if file_path.stat().st_size > 100_000_000:  # 100MB
                # Count lines first
                with open(file_path, 'r') as f:
                    total_lines = sum(1 for _ in f)
                
                # Read with progress bar
                chunks = []
                with tqdm(total=total_lines, desc="Loading CSV") as pbar:
                    for chunk in pd.read_csv(file_path, usecols=columns, 
                                            chunksize=self.chunk_size):
                        chunks.append(chunk)
                        pbar.update(len(chunk))
                
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.read_csv(file_path, usecols=columns)
    
    def load_parquet(self, file_path: Path, 
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load Parquet file or directory of Parquet files"""
        logger.info(f"Loading Parquet: {file_path}")
        
        if file_path.is_dir():
            # Load all parquet files in directory
            parquet_files = sorted(file_path.glob("*.parquet"))
            dfs = []
            for pf in tqdm(parquet_files, desc="Loading Parquet files"):
                dfs.append(pd.read_parquet(pf, columns=columns))
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.read_parquet(file_path, columns=columns)
    
    def load_duckdb(self, db_path: Path, 
                   query: str = None,
                   table: str = None) -> pd.DataFrame:
        """
        Load data from DuckDB database
        
        Args:
            db_path: Path to DuckDB file
            query: SQL query to execute
            table: Table name to load (if no query provided)
            
        Returns:
            Query results as DataFrame
        """
        logger.info(f"Loading from DuckDB: {db_path}")
        
        con = duckdb.connect(str(db_path), read_only=True)
        
        try:
            if query:
                result = con.execute(query).fetchdf()
            elif table:
                result = con.execute(f"SELECT * FROM {table}").fetchdf()
            else:
                # List available tables
                tables = con.execute("SHOW TABLES").fetchdf()
                logger.info(f"Available tables: {tables['name'].tolist()}")
                raise ValueError("Please specify either 'query' or 'table' parameter")
        finally:
            con.close()
        
        return result
    
    def load_compressed(self, file_path: Path) -> pd.DataFrame:
        """Load compressed files (zip, gz)"""
        logger.info(f"Loading compressed file: {file_path}")
        
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt') as f:
                return pd.read_csv(f)
        elif file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                # Assume single CSV inside
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if csv_files:
                    with z.open(csv_files[0]) as f:
                        return pd.read_csv(f)
                else:
                    raise ValueError("No CSV file found in zip archive")
        else:
            raise ValueError(f"Unsupported compression format: {file_path.suffix}")
    
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Universal loader that auto-detects format
        
        Args:
            file_path: Path to file
            **kwargs: Additional arguments passed to specific loader
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        file_format = self.detect_format(file_path)
        
        if file_format == 'csv':
            return self.load_csv(file_path, **kwargs)
        elif file_format == 'parquet':
            return self.load_parquet(file_path, **kwargs)
        elif file_format == 'duckdb':
            return self.load_duckdb(file_path, **kwargs)
        elif file_format == 'compressed':
            return self.load_compressed(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    def validate_barcode_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate loaded barcode data meets CRISPR screening requirements
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation report
        """
        validation = {
            'has_required_columns': False,
            'barcode_length_consistent': False,
            'id_format_valid': False,
            'data_types_correct': False,
            'no_duplicate_ids': False,
            'barcode_valid_chars': False,
            'issues': []
        }
        
        # Check required columns
        required_cols = {'id', 'barcode'}
        if 'barcode' in df.columns:
            validation['has_required_columns'] = True
        else:
            validation['issues'].append("Missing 'barcode' column")
            return validation
        
        # Check barcode length (should be 14 for Tel-Hai)
        valid_barcodes = df[df['barcode'].notna()]['barcode']
        if len(valid_barcodes) > 0:
            lengths = valid_barcodes.str.len()
            if lengths.nunique() == 1 and lengths.iloc[0] == 14:
                validation['barcode_length_consistent'] = True
            else:
                validation['issues'].append(f"Inconsistent barcode lengths: {lengths.unique()}")
        
        # Check valid DNA characters
        if len(valid_barcodes) > 0:
            invalid_chars = valid_barcodes.str.contains(r'[^ACGT]', na=False)
            if not invalid_chars.any():
                validation['barcode_valid_chars'] = True
            else:
                validation['issues'].append("Invalid characters in barcodes (not ACGT)")
        
        # Check for duplicate IDs
        if 'id' in df.columns:
            if df['id'].nunique() == len(df):
                validation['no_duplicate_ids'] = True
            else:
                duplicates = df['id'].value_counts()
                duplicates = duplicates[duplicates > 1]
                validation['issues'].append(f"Duplicate IDs found: {len(duplicates)}")
        
        return validation
    
    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 checksum for data integrity"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def verify_checksum(self, file_path: Path, expected_md5: str = None) -> bool:
        """
        Verify file integrity using MD5 checksum
        
        Args:
            file_path: Path to file
            expected_md5: Expected MD5 hash (or looks for .md5 file)
            
        Returns:
            True if checksum matches
        """
        if not expected_md5:
            # Look for .md5 file
            md5_file = file_path.with_suffix(file_path.suffix + '.md5')
            if md5_file.exists():
                with open(md5_file, 'r') as f:
                    expected_md5 = f.read().strip().split()[0]
        
        if expected_md5:
            actual_md5 = self.calculate_md5(file_path)
            match = actual_md5 == expected_md5
            if match:
                logger.info(f"Checksum verified: {file_path.name}")
            else:
                logger.warning(f"Checksum mismatch for {file_path.name}")
            return match
        else:
            logger.warning(f"No checksum available for {file_path.name}")
            return True  # Assume OK if no checksum available
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for loaded data"""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': df.isnull().sum().to_dict()
        }
        
        if 'barcode' in df.columns:
            summary['barcode_stats'] = {
                'unique': df['barcode'].nunique(),
                'null_rate': df['barcode'].isnull().mean(),
                'most_common': df['barcode'].value_counts().head().to_dict() if not df['barcode'].isnull().all() else {}
            }
        
        return summary


# Convenience functions
def load_telhai_data() -> Dict[str, pd.DataFrame]:
    """Load all Tel-Hai CRISPR screening data"""
    loader = DataLoader()
    data = {}
    
    # Load micro_design if available
    design_path = Path("/home/mch/dna/updated_data/micro_design.csv")
    if design_path.exists():
        data['micro_design'] = loader.load(design_path)
        logger.info(f"Loaded micro_design: {data['micro_design'].shape}")
    
    # Load clusters from DuckDB
    db_path = Path("/home/mch/dna/artifacts/dna.duckdb")
    if db_path.exists():
        data['clusters'] = loader.load_duckdb(db_path, table='clusters')
        logger.info(f"Loaded clusters: {data['clusters'].shape}")
    
    return data


def main():
    """Test data loader functionality"""
    loader = DataLoader()
    
    # Test with sample data
    test_file = Path("/home/mch/dna/updated_data/micro_design.csv")
    if test_file.exists():
        print(f"Testing with: {test_file}")
        
        # Load data
        df = loader.load(test_file)
        print(f"Loaded shape: {df.shape}")
        
        # Validate
        validation = loader.validate_barcode_data(df)
        print(f"Validation: {validation}")
        
        # Summary
        summary = loader.get_data_summary(df)
        print(f"Summary: Columns={summary['columns']}, Memory={summary['memory_usage_mb']:.1f}MB")
    else:
        print("Test file not found")


if __name__ == "__main__":
    main()