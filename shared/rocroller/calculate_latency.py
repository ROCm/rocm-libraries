#!/usr/bin/env python3
"""
Script to process CSV files in build/output/ and calculate median latency.
"""

import os
import re
import argparse
import json
import pandas as pd
import numpy as np
from tabulate import tabulate


def find_csv_files(directory):
    """Find all stats_ui_output CSV files in the given directory."""
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("stats_ui_output_agent_") and file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def extract_specific_lines(csv_file):
    """Extract instruction data from lines 3-6 of a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        instructions = []
        
        # Get rows 2-5 (lines 3-6 in 1-based counting)
        for i in range(2, 6):
            if i < len(df):
                instruction = df.iloc[i]['Instruction'] if 'Instruction' in df.columns else 'N/A'
                instructions.append(str(instruction)[:40])  # Truncate to 40 chars
            else:
                instructions.append('N/A')
        
        return instructions
    except Exception as e:
        print(f"Error reading lines from {csv_file}: {e}")
        return ['N/A'] * 4


def calculate_median_latency(csv_file):
    """Calculate median latency from s_waitcnt instructions."""
    try:
        df = pd.read_csv(csv_file)
        
        # Filter for s_waitcnt instructions
        if 'Instruction' not in df.columns or 'Latency' not in df.columns or 'Hitcount' not in df.columns:
            return None
            
        waitcnt_df = df[df['Instruction'].str.contains('s_waitcnt', na=False)]
        
        if waitcnt_df.empty:
            return None
        
        # Get the last s_waitcnt instruction
        last_row = waitcnt_df.iloc[-1]
        latency = int(last_row['Latency'])
        
        # Calculate mode of hitcounts
        hitcounts = waitcnt_df['Hitcount'].astype(int)
        mode_hitcount = hitcounts.mode()[0] if not hitcounts.empty else 1
        
        return latency / mode_hitcount

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None


def load_env_vars(csv_file):
    """Load environment variables from env_vars.json in the same directory as the CSV."""
    dir_path = os.path.dirname(csv_file)
    env_vars_path = os.path.join(dir_path, 'env_vars.json')
    
    if os.path.exists(env_vars_path):
        try:
            with open(env_vars_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading env_vars.json from {dir_path}: {e}")
    
    return {}


def process_all_files(directory, show_instructions=False, show_env_vars=False):
    """Process all CSV files and return results as a DataFrame."""
    csv_files = find_csv_files(directory)
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return None
    
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    results = []
    env_var_keys = []  # To track the order of env var keys
    
    for csv_file in csv_files:
        dir_name = os.path.basename(os.path.dirname(csv_file))
        
        env_vars = load_env_vars(csv_file)
        
        if not env_var_keys and env_vars:
            env_var_keys = list(env_vars.keys())
        
        result = {
            'Directory': dir_name,
            'Median Latency': calculate_median_latency(csv_file),
            'Full Path': os.path.abspath(csv_file),
            **{k: int(v) for k, v in env_vars.items()}
        }
        
        if show_instructions:
            instructions = extract_specific_lines(csv_file)
            for i, instr in enumerate(instructions, 3):
                result[f'Line {i} Instruction'] = instr
        
        for key, value in env_vars.items():
            result[key] = int(value)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    if env_var_keys:
        df = df.sort_values(env_var_keys)
    
    return df


def main():
    """Main function to process all CSV files."""
    parser = argparse.ArgumentParser(
        description="Calculate median latency from CSV files in a directory"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="output",
        help="Directory to search for CSV files",
    )
    parser.add_argument(
        "-i", "--instructions",
        action="store_true",
        help="Show instruction columns",
    )
    parser.add_argument(
        "-p", "--path",
        action="store_true",
        help="Show full path column",
    )
    parser.add_argument(
        "-e", "--env-vars",
        action="store_true",
        help="Show environment variables from env_vars.json",
    )
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"Directory {args.directory} not found!")
        return

    # Process all files
    df = process_all_files(args.directory, args.instructions, args.env_vars)
    
    if df is None or df.empty:
        return
    
    # Format median latency column
    df['Median Latency'] = df['Median Latency'].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
    )
    
    # Select columns to display
    display_columns = ['Directory', 'Median Latency']
    
    if args.instructions:
        # Add instruction columns
        instruction_cols = [col for col in df.columns if col.startswith('Line ')]
        display_columns.extend(instruction_cols)
    
    if args.env_vars:
        # Add all environment variable columns from env_vars.json
        env_cols = ['WRITE', 'INSTR_WIDTH', 'BYTE_STRIDE', 'ITERS']
        # Add any env columns that exist in the dataframe
        for col in env_cols:
            if col in df.columns:
                display_columns.append(col)
    
    if args.path:
        # Add path column at the end
        display_columns.append('Full Path')
    
    # Display the table
    table = tabulate(
        df[display_columns], 
        headers='keys', 
        tablefmt='pipe',
        showindex=False,
        numalign='right',
        stralign='left'
    )
    
    print(table)


if __name__ == "__main__":
    main()
