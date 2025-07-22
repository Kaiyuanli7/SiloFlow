#!/usr/bin/env python3
"""
Parquet File Inspector
=====================

A simple script to inspect Parquet files by displaying:
- Column names
- First 5 rows of data
- Basic file information

Usage:
    python parquet_inspector.py
"""

import pandas as pd
import sys
from pathlib import Path
import argparse
import os


def convert_parquet_to_csv(file_path: str, output_dir: str = None, max_rows: int = None) -> str:
    """
    Convert a Parquet file to CSV format for easier reading.
    
    Args:
        file_path (str): Path to the Parquet file
        output_dir (str): Directory to save CSV file (default: same directory as parquet)
        max_rows (int): Maximum number of rows to convert (for large files)
    
    Returns:
        str: Path to the created CSV file
    """
    try:
        # Check if file exists
        if not Path(file_path).exists():
            print(f"Error: File '{file_path}' does not exist.")
            return None
        
        print(f"\n🔄 CONVERTING PARQUET TO CSV:")
        print(f"{'='*60}")
        
        # Read the Parquet file
        print(f"Reading Parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Limit rows if specified
        if max_rows and len(df) > max_rows:
            print(f"Limiting to first {max_rows:,} rows (original: {len(df):,} rows)")
            df = df.head(max_rows)
        
        # Determine output path
        parquet_path = Path(file_path)
        if output_dir:
            output_path = Path(output_dir) / f"{parquet_path.stem}.csv"
        else:
            output_path = parquet_path.with_suffix('.csv')
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to CSV
        print(f"Converting to CSV: {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Get file sizes
        parquet_size = parquet_path.stat().st_size / 1024 / 1024
        csv_size = output_path.stat().st_size / 1024 / 1024
        
        print(f"✅ Conversion complete!")
        print(f"   Parquet size: {parquet_size:.2f} MB")
        print(f"   CSV size: {csv_size:.2f} MB")
        print(f"   Compression ratio: {csv_size/parquet_size:.1f}x larger")
        print(f"   CSV saved to: {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error converting Parquet to CSV: {e}")
        return None


def inspect_parquet_file(file_path: str, convert_to_csv: bool = False, csv_max_rows: int = None) -> None:
    """
    Inspect a Parquet file and display its contents.
    
    Args:
        file_path (str): Path to the Parquet file
        convert_to_csv (bool): Whether to also convert to CSV
        csv_max_rows (int): Maximum rows to include in CSV conversion
    """
    try:
        # Check if file exists
        if not Path(file_path).exists():
            print(f"Error: File '{file_path}' does not exist.")
            return
        
        # Check if file is a Parquet file
        if not file_path.lower().endswith('.parquet'):
            print(f"Warning: File '{file_path}' doesn't have a .parquet extension.")
            print("Attempting to read anyway...")
        
        print(f"\n{'='*80}")
        print(f"INSPECTING PARQUET FILE: {file_path}")
        print(f"{'='*80}")
        
        # Read the Parquet file
        df = pd.read_parquet(file_path)
        
        # Display basic file information
        print(f"\n📊 FILE INFORMATION:")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"   File size: {Path(file_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        # Display column information with better formatting
        print(f"\n📋 COLUMN NAMES ({len(df.columns)} columns):")
        print(f"{'='*80}")
        print(f"{'#':<3} {'Column Name':<35} {'Data Type':<20} {'Non-Null':<10} {'Null':<8} {'Sample Values'}")
        print(f"{'='*80}")
        
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            null_count = df[col].isnull().sum()
            
            # Get sample values (first non-null value)
            sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
            if isinstance(sample_val, str) and len(str(sample_val)) > 20:
                sample_val = str(sample_val)[:17] + "..."
            elif isinstance(sample_val, (int, float)):
                sample_val = f"{sample_val:.2f}" if isinstance(sample_val, float) else str(sample_val)
            
            print(f"{i:2d}. {col:<35} {dtype:<20} {non_null_count:<10} {null_count:<8} {sample_val}")
        
        # Display first 5 rows with better formatting
        print(f"\n📈 FIRST 5 ROWS:")
        print(f"{'='*80}")
        
        # Format the display for better readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        pd.set_option('display.min_rows', 5)
        
        # Create a more readable format
        sample_df = df.head(5).copy()
        
        # Format datetime columns for better display
        datetime_cols = sample_df.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            sample_df[col] = sample_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format numeric columns to reduce decimal places
        numeric_cols = sample_df.select_dtypes(include=['float64']).columns
        for col in numeric_cols:
            sample_df[col] = sample_df[col].round(2)
        
        # Display with better formatting
        print(sample_df.to_string(index=True, max_colwidth=25))
        
        # If there are many columns, also show a transposed view for better readability
        if len(df.columns) > 10:
            print(f"\n📊 TRANSPOSED VIEW (First 5 rows, showing all columns):")
            print(f"{'='*80}")
            
            # Create transposed view for wide datasets
            transposed = sample_df.T
            print(transposed.to_string(max_colwidth=40))
        
        # Display data types summary
        print(f"\n🔍 DATA TYPES SUMMARY:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Display data quality summary
        print(f"\n✅ DATA QUALITY SUMMARY:")
        total_cells = df.shape[0] * df.shape[1]
        total_nulls = df.isnull().sum().sum()
        completeness = ((total_cells - total_nulls) / total_cells) * 100
        
        print(f"   Total cells: {total_cells:,}")
        print(f"   Null cells: {total_nulls:,}")
        print(f"   Data completeness: {completeness:.1f}%")
        
        # Show unique values for key columns (if they exist)
        key_columns = ['granary_id', 'heap_id', 'detection_time']
        print(f"\n🔑 KEY COLUMNS UNIQUE VALUES:")
        for col in key_columns:
            if col in df.columns:
                unique_count = df[col].nunique()
                print(f"   {col}: {unique_count:,} unique values")
                if unique_count <= 10:
                    unique_vals = df[col].unique()[:5]  # Show first 5 unique values
                    print(f"     Sample values: {list(unique_vals)}")
        
        print(f"\n{'='*80}")
        print("Inspection complete!")
        print(f"{'='*80}")
        
        # Convert to CSV if requested
        if convert_to_csv:
            print(f"\n" + "="*80)
            csv_path = convert_parquet_to_csv(file_path, max_rows=csv_max_rows)
            if csv_path:
                print(f"\n📄 CSV FILE CREATED: {csv_path}")
                print(f"You can now open this CSV file in Excel, Google Sheets, or any text editor.")
                print(f"File size: {Path(csv_path).stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        print("Make sure the file is a valid Parquet file.")


def interactive_mode():
    """Run the script in interactive mode."""
    print("Parquet File Inspector")
    print("=" * 30)
    print("Enter the path to a Parquet file to inspect it.")
    print("Type 'quit' or 'exit' to close the script.\n")
    
    while True:
        try:
            # Get file path from user
            file_path = input("Enter Parquet file path: ").strip()
            
            # Check for exit commands
            if file_path.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Skip empty input
            if not file_path:
                print("Please enter a valid file path.")
                continue
            
            # Ask if user wants to convert to CSV
            convert_choice = input("Convert to CSV for easier reading? (y/n): ").strip().lower()
            convert_to_csv = convert_choice in ['y', 'yes']
            
            csv_max_rows = None
            if convert_to_csv:
                max_rows_input = input("Maximum rows to convert (press Enter for all): ").strip()
                if max_rows_input:
                    try:
                        csv_max_rows = int(max_rows_input)
                        print(f"Will convert first {csv_max_rows:,} rows to CSV.")
                    except ValueError:
                        print("Invalid number, will convert all rows.")
            
            # Inspect the file
            inspect_parquet_file(file_path, convert_to_csv=convert_to_csv, csv_max_rows=csv_max_rows)
            
            # Ask if user wants to continue
            print()
            continue_choice = input("Inspect another file? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '']:
                print("Goodbye!")
                break
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


def main():
    """Main function to handle command line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Inspect Parquet files by displaying column names and first 5 rows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parquet_inspector.py                    # Interactive mode
  python parquet_inspector.py file.parquet       # Inspect specific file
  python parquet_inspector.py -i                 # Interactive mode (explicit)
  python parquet_inspector.py file.parquet -c    # Inspect and convert to CSV
  python parquet_inspector.py file.parquet -c --csv-max-rows 1000  # Convert first 1000 rows
        """
    )
    
    parser.add_argument(
        'file_path',
        nargs='?',
        help='Path to the Parquet file to inspect'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '-c', '--convert-csv',
        action='store_true',
        help='Convert Parquet to CSV after inspection'
    )
    
    parser.add_argument(
        '--csv-max-rows',
        type=int,
        help='Maximum number of rows to include in CSV conversion (for large files)'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided or interactive mode requested, run interactively
    if args.interactive or args.file_path is None:
        interactive_mode()
    else:
        # Inspect the specified file
        inspect_parquet_file(
            args.file_path, 
            convert_to_csv=args.convert_csv,
            csv_max_rows=args.csv_max_rows
        )


if __name__ == "__main__":
    main() 