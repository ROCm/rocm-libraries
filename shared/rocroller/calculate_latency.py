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
        for i in range(1, 5):
            if i < len(df):
                instruction = df.iloc[i]['Instruction'] if 'Instruction' in df.columns else 'N/A'
                instructions.append(str(instruction)[:40])  # Truncate to 40 chars
            else:
                instructions.append('N/A')
        
        return instructions
    except Exception as e:
        print(f"Error reading lines from {csv_file}: {e}")
        return ['N/A'] * 4


def get_metric(csv_file):
    """Analyze CSV data and return a metric as (column_name, value) tuple."""
    COLUMN_NAME = 'Latency'
    ret = (COLUMN_NAME, None)
    try:
        df = pd.read_csv(csv_file)
                    
        filtered_df = df[df['Instruction'].str.contains('s_waitcnt', na=False)]

        if True:
            # Average latency
            row = filtered_df.iloc[1:].copy()
            row['Avg Latency'] = row['Latency'] / row['Hitcount']
            # latency = row['Avg Latency'].mode().iloc[0]
            latency = row['Avg Latency'].median()
            return (COLUMN_NAME, latency)
        else:
            # First s_waitcnt
            row = filtered_df.iloc[0]
            latency = row['Latency'] / row['Hitcount']
            return (COLUMN_NAME, latency)
            

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return ret


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


def process_all_files(directory, show_instructions=False, show_path=False, show_env_vars=False):
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
        
        metric_name, metric_value = get_metric(csv_file)
        
        result = {
            'Directory': dir_name,
            metric_name: metric_value,
            'Full Path': os.path.abspath(csv_file),
            **{k: int(v) for k, v in env_vars.items()}
        }
        
        if show_instructions:
            lines = extract_specific_lines(csv_file)
            for i, line in enumerate(lines, 3):
                result[f'Row {i}'] = line
        
        for key, value in env_vars.items():
            result[key] = int(value)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    if env_var_keys:
        df = df.sort_values(env_var_keys)
    
    # Get the metric column name from the first result
    metric_column = None
    if results:
        for key in results[0].keys():
            if key not in ['Directory', 'Full Path'] and key not in env_var_keys and not key.startswith('Row '):
                metric_column = key
                break
    
    column_headers = ['Directory']
    if metric_column:
        column_headers.append(metric_column)
    if show_env_vars:
        column_headers += env_var_keys
    if show_path:
        column_headers.append('Full Path')
    if show_instructions:
        column_headers += [f'Row {i}' for i in range(3, 7)]
    return df, list(column_headers)


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

    df, display_columns = process_all_files(args.directory, show_instructions=args.instructions, show_path=args.path, show_env_vars=args.env_vars)
    
    if df is None or df.empty:
        return
        
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
