#!/usr/bin/env python3

"""
compare_logs.py - Compare test results between two log folders

This script compares execution time and code generation time between two test sessions.
It reads from local.json files and generates comparison tables.
"""

import argparse
import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Optional


def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return ""


def compare_assembly_files(base_dir: str, compared_dir: str) -> bool:
    """Compare all assembly files in two directories."""
    base_asm_files = sorted(Path(base_dir).glob("*.s"))
    compared_asm_files = sorted(Path(compared_dir).glob("*.s"))

    # If different number of files, they're different
    if len(base_asm_files) != len(compared_asm_files):
        return True

    # Compare each file
    for base_file, compared_file in zip(base_asm_files, compared_asm_files):
        base_hash = calculate_file_hash(str(base_file))
        compared_hash = calculate_file_hash(str(compared_file))

        if base_hash != compared_hash:
            return True

    return False


def collect_exe_time_results(log_folder: str) -> Dict[str, Dict]:
    """Collect exe_time results from local.json."""
    exe_time_dir = Path(log_folder) / "exe_time"
    local_json = exe_time_dir / "local.json"

    if not local_json.exists():
        print(f"Warning: {local_json} not found", file=sys.stderr)
        return {}

    try:
        with open(local_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {local_json}: {e}", file=sys.stderr)
        return {}

    results = {}

    # Process each test case
    for test_case, times in data.items():
        if not times:
            continue

        # Get best (minimum) time
        best_time = min(times)

        # Determine assembly directory
        # test_case format: "mfma.yaml" or "sia3/mfma.yaml"
        # Remove .yaml extension and use as directory name
        test_name = test_case.replace('.yaml', '')
        asm_dir = exe_time_dir / test_name

        results[test_case] = {
            'time_us': best_time,
            'asm_dir': str(asm_dir) if asm_dir.exists() else None
        }

    return results


def collect_codegen_time_results(log_folder: str) -> Dict[str, float]:
    """Collect codegen_time results from local.json."""
    codegen_time_dir = Path(log_folder) / "codegen_time"
    local_json = codegen_time_dir / "local.json"

    if not local_json.exists():
        print(f"Warning: {local_json} not found", file=sys.stderr)
        return {}

    try:
        with open(local_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {local_json}: {e}", file=sys.stderr)
        return {}

    results = {}

    # Process each test case
    for test_case, times in data.items():
        if not times:
            continue

        # Get best (minimum) time
        best_time = min(times)
        results[test_case] = best_time

    return results


def generate_comparison_table(base_results: Dict, compared_results: Dict,
                              base_folder: str, compared_folder: str,
                              check_asm: bool = False) -> str:
    """Generate comparison table."""
    output = []

    # Header
    output.append(f"base folder: {base_folder}")
    output.append(f"compared folder: {compared_folder}")
    output.append("")

    # Table header
    if check_asm:
        header = f"{'testcase':<40} {'base':>12} {'compared':>12} {'improved':>12} {'asm_changed':>12}"
    else:
        header = f"{'testcase':<40} {'base':>12} {'compared':>12} {'improved':>12}"

    output.append(header)
    output.append("-" * len(header))

    # Get all test cases
    all_tests = sorted(set(base_results.keys()) | set(compared_results.keys()))

    for test_name in all_tests:
        base_val = base_results.get(test_name)
        compared_val = compared_results.get(test_name)

        if base_val is None:
            base_str = "N/A"
            # For codegen_time it's float, for exe_time it's dict
            if isinstance(compared_val, float):
                compared_str = f"{compared_val:.3f}"
            else:
                compared_str = f"{compared_val['time_us']:.3f}"
            improved_str = "NEW"
            asm_changed_str = "-"
        elif compared_val is None:
            base_str = f"{base_val:.3f}" if isinstance(base_val, float) else f"{base_val['time_us']:.3f}"
            compared_str = "N/A"
            improved_str = "REMOVED"
            asm_changed_str = "-"
        else:
            # Extract time values
            if isinstance(base_val, float):
                base_time = base_val
                compared_time = compared_val
            else:
                base_time = base_val['time_us']
                compared_time = compared_val['time_us']

            base_str = f"{base_time:.3f}"
            compared_str = f"{compared_time:.3f}"

            # Calculate improvement (positive means improvement)
            # base_time could be 0, handle that case
            if base_time == 0 or compared_time == 0:
                improvement = 0
            else:
                improvement = ((base_time - compared_time) / base_time) * 100
            improved_str = f"{improvement:+.3f}%"

            # Check assembly if needed
            asm_changed_str = "-"
            if check_asm and isinstance(base_val, dict) and isinstance(compared_val, dict):
                if base_val.get('asm_dir') and compared_val.get('asm_dir'):
                    asm_changed = compare_assembly_files(base_val['asm_dir'], compared_val['asm_dir'])
                    asm_changed_str = "yes" if asm_changed else "no"

        if check_asm:
            line = f"{test_name:<40} {base_str:>12} {compared_str:>12} {improved_str:>12} {asm_changed_str:>12}"
        else:
            line = f"{test_name:<40} {base_str:>12} {compared_str:>12} {improved_str:>12}"

        output.append(line)

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description='Compare test results between two log folders')
    parser.add_argument('--base-folder', required=True, help='Base log folder path')
    parser.add_argument('--compared-folder', required=True, help='Compared log folder path')
    parser.add_argument('--output-prefix', default='comparison', help='Output file prefix (default: comparison)')

    args = parser.parse_args()

    # Validate folders exist
    if not os.path.exists(args.base_folder):
        print(f"Error: Base folder not found: {args.base_folder}", file=sys.stderr)
        return 1

    if not os.path.exists(args.compared_folder):
        print(f"Error: Compared folder not found: {args.compared_folder}", file=sys.stderr)
        return 1

    # Get folder names for display
    base_name = os.path.basename(args.base_folder.rstrip('/'))
    compared_name = os.path.basename(args.compared_folder.rstrip('/'))

    # Collect exe_time results from local.json
    print("Reading exe_time results from local.json...")
    base_exe_results = collect_exe_time_results(args.base_folder)
    compared_exe_results = collect_exe_time_results(args.compared_folder)

    # Collect codegen_time results from local.json
    print("Reading codegen_time results from local.json...")
    base_codegen_results = collect_codegen_time_results(args.base_folder)
    compared_codegen_results = collect_codegen_time_results(args.compared_folder)

    # Generate comparison tables
    print("\nGenerating comparison tables...")

    # exe_time comparison
    exe_comparison = generate_comparison_table(
        base_exe_results, compared_exe_results,
        base_name, compared_name, check_asm=True
    )

    # codegen_time comparison
    codegen_comparison = generate_comparison_table(
        base_codegen_results, compared_codegen_results,
        base_name, compared_name, check_asm=False
    )

    # Write output file
    output_file = f"{args.output_prefix}_report.txt"
    with open(output_file, 'w') as f:
        f.write("--------------------------\n")
        f.write("EXECUTION TIME COMPARISON:\n")
        f.write("--------------------------\n")
        f.write(exe_comparison)
        f.write("\n\n\n")

        f.write("--------------------------------\n")
        f.write("CODE GENERATION TIME COMPARISON:\n")
        f.write("--------------------------------\n")
        f.write(codegen_comparison)
        f.write("\n\n")

    print(f"\nComparison report saved to: {output_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

