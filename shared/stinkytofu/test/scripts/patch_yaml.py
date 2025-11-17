#!/usr/bin/env python3

"""
patch_yaml.py - Patch YAML config files for test variants

This script modifies YAML configuration files to create test variants:
- Patch GlobalParameters for codegen-time testing
- Modify ScheduleIterAlg for SIA variants
- Enable sparse matrix support
"""

import argparse
import sys
import yaml
from typing import Any, Dict


def patch_codegen_params(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Patch GlobalParameters for codegen-time testing.

    These settings optimize for code generation profiling:
    - Disable validation and warmups
    - Single benchmark run
    - Enable Python profiling
    """
    if 'GlobalParameters' not in data:
        data['GlobalParameters'] = {}

    codegen_params = {
        'SleepPercent': 0,
        'NumElementsToValidate': 0,
        'NumWarmups': 0,
        'EnqueuesPerSync': 1,
        'SyncsPerBenchmark': 1,
        'NumBenchmarks': 1,
        'KernelTime': False,
        'MaxWorkspaceSize': 5000000000,
        'CpuThreads': 0,
        'PythonProfile': True,
    }

    data['GlobalParameters'].update(codegen_params)
    return data


def patch_schedule_iter_alg(data: Dict[str, Any], value: int) -> Dict[str, Any]:
    """
    Patch ScheduleIterAlg in ForkParameters.

    This controls the scheduling algorithm:
    - 1: SIA1 (stinkytofu default)
    - 3: SIA3 (alternative scheduling)
    """
    if 'BenchmarkProblems' not in data:
        return data

    for problem_set in data['BenchmarkProblems']:
        if isinstance(problem_set, list):
            for item in problem_set:
                if isinstance(item, dict) and 'ForkParameters' in item:
                    fork_params = item['ForkParameters']
                    # Find and update ScheduleIterAlg
                    found = False
                    for param in fork_params:
                        if isinstance(param, dict) and 'ScheduleIterAlg' in param:
                            param['ScheduleIterAlg'] = [value]
                            found = True

                    # If not found, add it
                    if not found:
                        fork_params.append({'ScheduleIterAlg': [value]})

    return data


def patch_enable_sparse(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enable sparse matrix support.

    Modifies problem type to enable sparse operations.
    """
    if 'BenchmarkProblems' not in data:
        return data

    for problem_set in data['BenchmarkProblems']:
        if isinstance(problem_set, list):
            for item in problem_set:
                if isinstance(item, dict) and 'Sparse' in item:
                    item['Sparse'] = 1

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Patch YAML config files for test variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch for codegen testing
  %(prog)s --input mfma.yaml --output mfma-codegen.yaml --patch-codegen

  # Create SIA3 variant
  %(prog)s --input mfma.yaml --output mfma-sia3.yaml --schedule-iter-alg 3

  # Combine multiple patches
  %(prog)s --input mfma.yaml --output mfma-sia3-codegen.yaml \\
           --patch-codegen --schedule-iter-alg 3
        """
    )

    parser.add_argument('--input', required=True,
                       help='Input YAML file')
    parser.add_argument('--output', required=True,
                       help='Output YAML file')
    parser.add_argument('--patch-codegen', action='store_true',
                       help='Patch GlobalParameters for codegen testing')
    parser.add_argument('--schedule-iter-alg', type=int, metavar='N',
                       help='Set ScheduleIterAlg value (e.g., 1, 3)')
    parser.add_argument('--enable-sparse', action='store_true',
                       help='Enable sparse matrix support')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')

    args = parser.parse_args()

    # Check if any patch is specified
    if not any([args.patch_codegen, args.schedule_iter_alg is not None, args.enable_sparse]):
        print("Warning: No patches specified, file will be copied unchanged", file=sys.stderr)

    # Load YAML
    try:
        with open(args.input, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file {args.input}: {e}", file=sys.stderr)
        return 1

    # Apply patches
    if args.patch_codegen:
        if args.verbose:
            print(f"Patching GlobalParameters for codegen testing...")
        data = patch_codegen_params(data)

    if args.schedule_iter_alg is not None:
        if args.verbose:
            print(f"Setting ScheduleIterAlg to {args.schedule_iter_alg}...")
        data = patch_schedule_iter_alg(data, args.schedule_iter_alg)

    if args.enable_sparse:
        if args.verbose:
            print(f"Enabling sparse matrix support...")
        data = patch_enable_sparse(data)

    # Save YAML
    try:
        with open(args.output, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        if args.verbose:
            print(f"Successfully wrote patched YAML to {args.output}")
    except Exception as e:
        print(f"Error writing YAML file {args.output}: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

