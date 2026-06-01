#!/usr/bin/env python3
################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# TensileLite COMPLETE Parameter Coverage Analysis
# Tracks ALL 300+ parameters across all architectures
#
################################################################################

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple

# Add Tensile parent directory to path
tensile_root = Path(__file__).parent.parent.parent
if str(tensile_root) not in sys.path:
    sys.path.insert(0, str(tensile_root))

import yaml

def load_tensile_modules():
    """Lazy load Tensile modules"""
    global DEFAULT_YAML_LOADER, SUPPORTED_ISA, validParameters, _defaultProblemType, globalParameters
    from Tensile.CustomYamlLoader import DEFAULT_YAML_LOADER
    from Tensile.Common.Architectures import SUPPORTED_ISA
    from Tensile.Common.ValidParameters import validParameters
    from Tensile.SolutionStructs.Problem import _defaultProblemType
    from Tensile.Common.GlobalParameters import globalParameters
    return DEFAULT_YAML_LOADER, SUPPORTED_ISA, validParameters, _defaultProblemType, globalParameters

# Initialize globals
DEFAULT_YAML_LOADER = None
SUPPORTED_ISA = None
validParameters = None
_defaultProblemType = None
globalParameters = None


def normalize_value(value):
    """Normalize parameter values for consistent comparison"""
    if isinstance(value, list):
        return tuple(value)
    # Normalize 0/1 to False/True for boolean parameters
    if value == 0:
        return False
    elif value == 1:
        return True
    return value


def is_trackable_valid_values(valid_values):
    """Determine if valid values are trackable (finite set, not a range)"""
    if not isinstance(valid_values, list):
        return False

    # Skip if it's a range-like structure (too many values)
    if len(valid_values) > 100:
        return False

    # Check if all values are simple (not complex nested structures)
    for v in valid_values:
        if isinstance(v, list) and len(v) > 4:  # Skip complex matrix instructions
            return False

    return True


def extract_all_parameters():
    """Extract all parameters from Tensile source files"""
    all_params = {}

    # 1. Extract Solution parameters from validParameters
    for param_name, valid_values in validParameters.items():
        # Determine if we can track value coverage
        trackable = is_trackable_valid_values(valid_values)

        all_params[param_name] = {
            'category': 'solution',
            'valid_values': valid_values if trackable else None,
            'default': None,
            'trackable': trackable
        }

    # 2. Extract Problem parameters from _defaultProblemType
    for param_name, default_value in _defaultProblemType.items():
        if param_name in all_params:
            # Already exists (some overlap)
            all_params[param_name]['default'] = default_value
        else:
            # Infer valid values for boolean parameters
            valid_values = None
            trackable = False

            if isinstance(default_value, bool):
                valid_values = [False, True]
                trackable = True
            elif param_name in ['UseBias', 'UseScaleAlphaVec']:
                valid_values = [0, 1, 2, 3]
                trackable = True
            elif param_name in ['Sparse']:
                valid_values = [0, 1, 2]
                trackable = True
            elif param_name in ['MXBlockA', 'MXBlockB']:
                valid_values = [0, 16, 32]
                trackable = True
            elif param_name == 'UseScaleAB':
                valid_values = ['', 'Scalar', 'Vector']
                trackable = True

            all_params[param_name] = {
                'category': 'problem',
                'valid_values': valid_values,
                'default': default_value,
                'trackable': trackable
            }

    # 3. Extract Global parameters from globalParameters
    for param_name, default_value in globalParameters.items():
        if param_name in all_params:
            continue

        # Most global params are not trackable (numeric ranges, strings, etc.)
        valid_values = None
        trackable = False

        if isinstance(default_value, bool):
            valid_values = [False, True]
            trackable = True

        all_params[param_name] = {
            'category': 'global',
            'valid_values': valid_values,
            'default': default_value,
            'trackable': trackable
        }

    return all_params


def get_arch_from_path(filepath: Path) -> str:
    """Extract architecture from file path"""
    path_str = str(filepath)

    # Check for gfx subdirectories
    if '/gfx11/' in path_str:
        return 'GFX11'
    elif '/gfx1250/' in path_str:
        return 'GFX1250'
    elif '/gfx12/' in path_str:
        return 'GFX12'
    elif '/gfx90a/' in path_str:
        return 'GFX90A'
    elif '/gfx94x/' in path_str or '/gfx942/' in path_str or '/gfx940/' in path_str:
        return 'GFX94X'
    elif '/gfx950/' in path_str:
        return 'GFX950'

    return None


def scan_yaml_file(filepath: Path, param_registry: Dict) -> Dict:
    """Scan YAML file and extract ALL parameter values"""
    try:
        with open(filepath, 'r') as f:
            data = yaml.load(f, Loader=DEFAULT_YAML_LOADER)

        if not data:
            return None

        # Collect all parameter values from the file
        param_values = defaultdict(set)

        # Check GlobalParameters
        global_params = data.get('GlobalParameters', {})
        if global_params and isinstance(global_params, dict):
            for param, value in global_params.items():
                if param in param_registry:
                    param_values[param].add(normalize_value(value))

        # Check BenchmarkProblems
        benchmark_problems = data.get('BenchmarkProblems', [])
        for problem_group in benchmark_problems:
            if not problem_group or not isinstance(problem_group, list):
                continue

            # First element is ProblemType
            if len(problem_group) >= 1 and isinstance(problem_group[0], dict):
                problem_type = problem_group[0]
                if problem_type:
                    for param, value in problem_type.items():
                        if param in param_registry:
                            param_values[param].add(normalize_value(value))

            # Process solution groups
            for solution_group in problem_group[1:]:
                if not isinstance(solution_group, dict):
                    continue

                # Check all parameter sections
                for section in ['InitialSolutionParameters', 'BenchmarkCommonParameters',
                              'ForkParameters', 'JoinParameters', 'BenchmarkFinalParameters']:
                    params = solution_group.get(section, [])
                    if isinstance(params, dict):
                        params = [params]

                    if isinstance(params, list):
                        for param_dict in params:
                            if isinstance(param_dict, dict):
                                for param, values in param_dict.items():
                                    if param in param_registry:
                                        if isinstance(values, list):
                                            for v in values:
                                                param_values[param].add(normalize_value(v))
                                        else:
                                            param_values[param].add(normalize_value(values))

        return param_values

    except Exception as e:
        return None


def analyze_coverage(test_dir: Path, param_registry: Dict, verbose: bool = False):
    """Analyze parameter value coverage across architectures"""

    # Architecture -> parameter -> set of values
    arch_coverage = defaultdict(lambda: defaultdict(set))
    arch_file_counts = defaultdict(int)

    yaml_files = list(test_dir.rglob('*.yaml'))

    if verbose:
        print(f"Scanning {len(yaml_files)} YAML files...")

    for yaml_path in yaml_files:
        arch = get_arch_from_path(yaml_path)
        if not arch:
            continue

        param_values = scan_yaml_file(yaml_path, param_registry)
        if not param_values:
            continue

        arch_file_counts[arch] += 1

        for param, values in param_values.items():
            arch_coverage[arch][param].update(values)

    return arch_coverage, arch_file_counts


def generate_report(arch_coverage: Dict, arch_file_counts: Dict, param_registry: Dict, output_file: str):
    """Generate formatted coverage report for ALL parameters across all architectures"""

    archs = sorted(arch_coverage.keys())

    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 180 + "\n")
        f.write("COMPLETE PARAMETER COVERAGE REPORT - ALL PARAMETERS ACROSS ALL ARCHITECTURES\n")
        f.write("=" * 180 + "\n\n")

        # Architecture summary
        f.write("ARCHITECTURES ANALYZED:\n")
        for arch in archs:
            f.write(f"  {arch}: {arch_file_counts[arch]} test files\n")
        f.write("\n")

        # Separate trackable and non-trackable parameters
        trackable_params = {k: v for k, v in param_registry.items() if v['trackable']}
        non_trackable_params = {k: v for k, v in param_registry.items() if not v['trackable']}

        # Section 1: Trackable parameters with cross-architecture coverage
        f.write("=" * 180 + "\n")
        f.write("SECTION 1: TRACKABLE PARAMETERS (Value Coverage Across Architectures)\n")
        f.write("=" * 180 + "\n\n")

        # Build table header
        header = f"{'Parameter':<35} {'Category':<10} {'Default':<10} {'Valid Values':<25}"
        for arch in archs:
            header += f" {arch:<15}"
        f.write(header + "\n")
        f.write("-" * 180 + "\n")

        # Coverage statistics per architecture
        arch_stats = {arch: {'full': 0, 'partial': 0, 'missing': 0} for arch in archs}

        for param in sorted(trackable_params.keys()):
            param_info = param_registry[param]
            valid_values = param_info['valid_values']
            category = param_info['category']
            default = str(param_info.get('default', 'N/A'))[:8]

            if not valid_values:
                continue

            # Convert to set, handling nested lists
            try:
                valid_set = set(valid_values if not any(isinstance(v, list) for v in valid_values)
                              else [tuple(v) if isinstance(v, list) else v for v in valid_values])
            except TypeError:
                continue

            # Format valid values
            valid_str = str(valid_values)[:23]
            if len(str(valid_values)) > 23:
                valid_str += ".."

            row = f"{param:<35} {category:<10} {default:<10} {valid_str:<25}"

            # Add status for each architecture
            for arch in archs:
                used_values = arch_coverage[arch].get(param, set())
                missing_values = valid_set - used_values

                if len(used_values) == 0:
                    status = "✗ MISSING"
                    arch_stats[arch]['missing'] += 1
                elif len(missing_values) == 0:
                    status = "✓ FULL"
                    arch_stats[arch]['full'] += 1
                else:
                    status = f"⚠️ {len(used_values)}/{len(valid_set)}"
                    arch_stats[arch]['partial'] += 1

                row += f" {status:<15}"

            f.write(row + "\n")

        # Summary statistics
        f.write("\n" + "=" * 180 + "\n")
        f.write("TRACKABLE PARAMETERS SUMMARY\n")
        f.write("=" * 180 + "\n\n")
        f.write(f"{'Architecture':<15} {'Full Coverage':<18} {'Partial Coverage':<18} {'Not Tested':<18} {'Coverage %':<15}\n")
        f.write("-" * 90 + "\n")

        total_trackable = len([p for p in trackable_params.keys() if trackable_params[p]['valid_values']])

        for arch in archs:
            stats = arch_stats[arch]
            tested = stats['full'] + stats['partial']
            pct = int(tested / total_trackable * 100) if total_trackable > 0 else 0
            f.write(f"{arch:<15} {stats['full']:<18} {stats['partial']:<18} {stats['missing']:<18} {pct}%\n")

        # Detailed breakdown of missing values for partial/missing coverage
        f.write("\n" + "=" * 180 + "\n")
        f.write("DETAILED MISSING VALUES BREAKDOWN (Trackable Parameters)\n")
        f.write("=" * 180 + "\n\n")

        for param in sorted(trackable_params.keys()):
            param_info = param_registry[param]
            valid_values = param_info['valid_values']

            if not valid_values:
                continue

            # Convert to set, handling nested lists
            try:
                valid_set = set(valid_values if not any(isinstance(v, list) for v in valid_values)
                              else [tuple(v) if isinstance(v, list) else v for v in valid_values])
            except TypeError:
                continue

            # Check if any architecture has missing values
            has_missing = False
            arch_details = {}

            for arch in archs:
                used_values = arch_coverage[arch].get(param, set())
                missing_values = valid_set - used_values

                if len(missing_values) > 0:
                    has_missing = True
                    arch_details[arch] = {
                        'used': used_values,
                        'missing': missing_values
                    }

            # Only show parameters with missing values on at least one arch
            if has_missing:
                f.write(f"\n{param}\n")
                f.write(f"  Category: {param_info['category']}, Default: {param_info.get('default', 'N/A')}\n")
                f.write(f"  Valid Values: {sorted(list(valid_set), key=str)}\n")

                for arch in archs:
                    if arch in arch_details:
                        used = sorted(list(arch_details[arch]['used']), key=str)
                        missing = sorted(list(arch_details[arch]['missing']), key=str)
                        f.write(f"    {arch:<10}: ✓ Tested {used}  |  ✗ Missing {missing}\n")
                    else:
                        used_values = arch_coverage[arch].get(param, set())
                        if len(used_values) > 0:
                            f.write(f"    {arch:<10}: ✓ FULL COVERAGE\n")
                        else:
                            f.write(f"    {arch:<10}: ✗ NOT TESTED\n")

        # Section 2: Non-trackable parameters
        f.write("\n\n" + "=" * 180 + "\n")
        f.write("SECTION 2: NON-TRACKABLE PARAMETERS (Usage Across Architectures)\n")
        f.write("=" * 180 + "\n\n")

        header = f"{'Parameter':<35} {'Category':<10} {'Default':<15}"
        for arch in archs:
            header += f" {arch:<15}"
        f.write(header + "\n")
        f.write("-" * 180 + "\n")

        for param in sorted(non_trackable_params.keys()):
            param_info = param_registry[param]
            category = param_info['category']
            default = str(param_info.get('default', 'N/A'))[:13]

            row = f"{param:<35} {category:<10} {default:<15}"

            for arch in archs:
                used_values = arch_coverage[arch].get(param, set())
                if len(used_values) > 0:
                    status = "✓ USED"
                else:
                    status = "---"
                row += f" {status:<15}"

            f.write(row + "\n")

        # Overall summary
        f.write("\n" + "=" * 180 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 180 + "\n\n")
        f.write(f"Total Parameters: {len(param_registry)}\n")
        f.write(f"  Trackable (finite value sets): {len(trackable_params)}\n")
        f.write(f"  Non-trackable (ranges/complex): {len(non_trackable_params)}\n\n")

        for arch in archs:
            tested_all = sum(1 for p in param_registry.keys() if len(arch_coverage[arch].get(p, set())) > 0)
            f.write(f"{arch}: {tested_all}/{len(param_registry)} parameters used ({int(tested_all/len(param_registry)*100)}%)\n")

        # Parameters not tested on ANY architecture
        f.write("\n" + "=" * 180 + "\n")
        f.write("PARAMETERS NOT TESTED ON ANY ARCHITECTURE\n")
        f.write("=" * 180 + "\n\n")

        untested_trackable = []
        untested_non_trackable = []

        for param, param_info in sorted(param_registry.items()):
            # Check if tested on any architecture
            tested_anywhere = False
            for arch in archs:
                if len(arch_coverage[arch].get(param, set())) > 0:
                    tested_anywhere = True
                    break

            if not tested_anywhere:
                if param_info['trackable']:
                    untested_trackable.append((param, param_info))
                else:
                    untested_non_trackable.append((param, param_info))

        # Trackable parameters not tested
        if untested_trackable:
            f.write(f"TRACKABLE PARAMETERS (not tested anywhere): {len(untested_trackable)}\n\n")
            f.write(f"{'Parameter':<40} {'Category':<12} {'Default':<15} {'Valid Values':<50}\n")
            f.write("-" * 120 + "\n")

            for param, param_info in untested_trackable:
                category = param_info['category']
                default = str(param_info.get('default', 'N/A'))[:13]
                valid_str = str(param_info['valid_values'])
                if len(valid_str) > 48:
                    valid_str = valid_str[:45] + "..."
                f.write(f"{param:<40} {category:<12} {default:<15} {valid_str:<50}\n")

            f.write("\n")

        # Non-trackable parameters not tested
        if untested_non_trackable:
            f.write(f"NON-TRACKABLE PARAMETERS (not tested anywhere): {len(untested_non_trackable)}\n\n")
            f.write(f"{'Parameter':<40} {'Category':<12} {'Default':<15}\n")
            f.write("-" * 70 + "\n")

            for param, param_info in untested_non_trackable:
                category = param_info['category']
                default = str(param_info.get('default', 'N/A'))[:13]
                f.write(f"{param:<40} {category:<12} {default:<15}\n")

            f.write("\n")

        # Summary
        total_untested = len(untested_trackable) + len(untested_non_trackable)
        f.write(f"Total untested parameters: {total_untested}/{len(param_registry)} ({int(total_untested/len(param_registry)*100)}%)\n")
        f.write(f"  Trackable: {len(untested_trackable)}\n")
        f.write(f"  Non-trackable: {len(untested_non_trackable)}\n")



def main():
    parser = argparse.ArgumentParser(
        description='Analyze TensileLite parameter coverage across all architectures'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='Tensile/Tests/common',
        help='Directory containing test YAML files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='complete_parameter_coverage.txt',
        help='Output filename'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress information'
    )

    args = parser.parse_args()

    load_tensile_modules()

    if args.verbose:
        print("=" * 60)
        print("TensileLite COMPLETE Parameter Coverage Analysis")
        print("=" * 60)
        print("\nExtracting all parameters...")

    param_registry = extract_all_parameters()

    trackable_count = sum(1 for p in param_registry.values() if p['trackable'])
    non_trackable_count = len(param_registry) - trackable_count

    if args.verbose:
        print(f"  Total parameters: {len(param_registry)}")
        print(f"    - Trackable (finite value sets): {trackable_count}")
        print(f"    - Non-trackable (ranges/complex): {non_trackable_count}")

        solution_count = sum(1 for p in param_registry.values() if p['category'] == 'solution')
        problem_count = sum(1 for p in param_registry.values() if p['category'] == 'problem')
        global_count = sum(1 for p in param_registry.values() if p['category'] == 'global')
        print(f"  By category:")
        print(f"    - Solution: {solution_count}")
        print(f"    - Problem: {problem_count}")
        print(f"    - Global: {global_count}")

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Error: Test directory '{test_dir}' does not exist")
        return 1

    arch_coverage, arch_file_counts = analyze_coverage(test_dir, param_registry, args.verbose)

    if args.verbose:
        print(f"\nFound {len(arch_coverage)} architectures:")
        for arch in sorted(arch_coverage.keys()):
            print(f"  {arch}: {arch_file_counts[arch]} test files")

    generate_report(arch_coverage, arch_file_counts, param_registry, args.output)

    print(f"\n✓ Report generated: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
