#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tensile Kernel Count Validator for gfx950

Calculates expected kernel counts from YAML ForkParameters and validates
against actual Tensile output to detect regressions where solutions are
incorrectly rejected.

Usage:
  # Generate baseline from YAML files
  ./kernel_count_validator.py --generate-baseline --yaml-dir Tensile/Tests/common

  # Validate a Tensile run output
  ./kernel_count_validator.py --validate --log build_output.log

  # Check a single YAML
  ./kernel_count_validator.py --check-yaml path/to/test.yaml
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml


class KernelCountValidator:
    """Calculate and validate kernel counts for gfx950"""

    def __init__(self, baseline_path: Path = None):
        self.baseline_path = baseline_path or Path("kernel_count_baseline_gfx950.json")
        self.baseline = self._load_baseline()

    def _load_baseline(self) -> Dict:
        """Load existing baseline or return empty dict"""
        if self.baseline_path.exists():
            with open(self.baseline_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_baseline(self):
        """Save baseline to file"""
        with open(self.baseline_path, 'w') as f:
            json.dump(self.baseline, f, indent=2, sort_keys=True)
        print(f"✓ Baseline saved to {self.baseline_path}")

    def _is_gfx950_yaml(self, yaml_data: Dict) -> bool:
        """Check if YAML targets gfx950"""
        # Check GlobalParameters Architecture
        global_params = yaml_data.get("GlobalParameters", {})
        arch = global_params.get("Architecture", "")
        if arch == "gfx950":
            return True

        # Check ISA
        isa = global_params.get("ISA", [])
        if [9, 5, 0] in isa or [[9, 5, 0]] == isa:
            return True

        # Check TestParameters marks (skip-gfx950 means NOT for gfx950)
        test_params = yaml_data.get("TestParameters", {})
        marks = test_params.get("marks", [])
        if "skip-gfx950" in marks:
            return False

        # If it skips many archs but not gfx950, assume it's for gfx950
        skip_marks = [m for m in marks if m.startswith("skip-gfx")]
        if skip_marks and "skip-gfx950" not in skip_marks:
            return True

        return False

    def _calculate_fork_combinations(self, fork_params: List[Dict]) -> int:
        """Calculate total combinations from ForkParameters"""
        if not fork_params:
            return 1

        total = 1
        for param_dict in fork_params:
            for key, values in param_dict.items():
                if isinstance(values, list):
                    # Handle nested lists (e.g., MatrixInstruction: [[32,64,32,...]])
                    if values and isinstance(values[0], list):
                        count = len(values)
                    else:
                        count = len(values) if values else 1
                    total *= count
        return total

    def _extract_problem_signature(self, problem_type: Dict) -> str:
        """Create a readable signature for a problem type"""
        sig_parts = []

        # Data type
        dt = problem_type.get("DataType", "?")
        sig_parts.append(dt)

        # Transpose
        ta = "T" if problem_type.get("TransposeA", False) else "N"
        tb = "T" if problem_type.get("TransposeB", False) else "N"
        sig_parts.append(f"{ta}{tb}")

        # Special features
        if problem_type.get("Batched", False):
            sig_parts.append("Batched")
        if problem_type.get("UseBias", 0):
            sig_parts.append("Bias")
        if problem_type.get("MXBlockA", 0) or problem_type.get("MXBlockB", 0):
            sig_parts.append("MX")

        return "_".join(sig_parts)

    def calculate_yaml_counts(self, yaml_path: Path) -> Dict[str, int]:
        """Calculate expected kernel counts for all problems in a YAML"""
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        if not self._is_gfx950_yaml(yaml_data):
            return {}

        benchmark_problems = yaml_data.get("BenchmarkProblems", [])
        if not benchmark_problems:
            return {}

        problem_counts = {}
        problem_idx = 0

        for problem_group in benchmark_problems:
            if not problem_group or len(problem_group) < 2:
                continue

            # problem_group[0] is ProblemType
            # problem_group[1] is BenchmarkProblemSizeGroup with ForkParameters
            problem_type = problem_group[0]
            size_group = problem_group[1]

            # Get ForkParameters
            fork_params = size_group.get("ForkParameters", [])
            if not fork_params:
                continue

            # Calculate combinations
            count = self._calculate_fork_combinations(fork_params)

            # Create signature
            sig = self._extract_problem_signature(problem_type)
            key = f"problem_{problem_idx}_{sig}"
            problem_counts[key] = count
            problem_idx += 1

        return problem_counts

    def generate_baseline(self, yaml_dir: Path):
        """Generate baseline from all YAML files in directory"""
        yaml_files = []
        for pattern in ["**/*.yaml", "**/*.yml"]:
            yaml_files.extend(yaml_dir.glob(pattern))

        if not yaml_files:
            print(f"No YAML files found in {yaml_dir}")
            return

        print(f"Scanning {len(yaml_files)} YAML files for gfx950 configs...")

        new_baseline = {}
        processed = 0

        for yaml_path in sorted(yaml_files):
            try:
                counts = self.calculate_yaml_counts(yaml_path)
                if counts:
                    # Use relative path from yaml_dir
                    try:
                        rel_path = yaml_path.relative_to(yaml_dir)
                    except ValueError:
                        rel_path = yaml_path

                    # Store with architecture key
                    new_baseline[str(rel_path)] = {
                        "gfx950": counts
                    }
                    processed += 1

                    total = sum(counts.values())
                    print(f"  ✓ {rel_path}: {len(counts)} problem(s), {total} total kernels")
            except Exception as e:
                print(f"  ✗ {yaml_path.name}: Error - {e}")

        if new_baseline:
            self.baseline = new_baseline
            self._save_baseline()
            print(f"\n✓ Processed {processed} gfx950 YAML files")
        else:
            print("✗ No gfx950 YAML files found")

    def validate_tensile_log(self, log_path: Path, arch: str = "gfx950") -> bool:
        """Validate Tensile output against baseline"""
        with open(log_path, 'r') as f:
            log_content = f.read()

        # Extract config file from log
        config_match = re.search(r'Config:\s+(.+\.ya?ml)', log_content)
        if not config_match:
            print("✗ Could not find Config path in log")
            return False

        config_path = Path(config_match.group(1)).name

        # Find matching baseline entry
        baseline_entry = None
        for key in self.baseline.keys():
            if Path(key).name == config_path:
                baseline_data = self.baseline[key]
                # Check if architecture key exists
                if arch in baseline_data:
                    baseline_entry = baseline_data[arch]
                elif isinstance(baseline_data, dict) and "problem_" in str(list(baseline_data.keys())[0] if baseline_data else ""):
                    # Old format without arch key
                    baseline_entry = baseline_data
                break

        if not baseline_entry:
            print(f"✗ No baseline found for {config_path} (arch: {arch})")
            return False

        # Extract "Actual Solutions: X / Y" from log
        solution_pattern = r'# Actual Solutions:\s+(\d+)\s+/\s+(\d+)'
        matches = re.findall(solution_pattern, log_content)

        if not matches:
            print("✗ Could not find solution counts in log")
            return False

        # Tensile may have multiple benchmark steps
        all_passed = True
        for idx, (actual_str, attempted_str) in enumerate(matches):
            actual = int(actual_str)
            attempted = int(attempted_str)

            # Try to match with baseline problems
            problem_keys = list(baseline_entry.keys())
            if idx < len(problem_keys):
                problem_key = problem_keys[idx]
                expected = baseline_entry[problem_key]

                if actual < expected:
                    print(f"✗ {config_path}::{problem_key} [{arch}]")
                    print(f"  Expected: {expected} kernels")
                    print(f"  Actual:   {actual} kernels")
                    print(f"  Missing:  {expected - actual} kernels (REGRESSION)")
                    all_passed = False
                elif actual > expected:
                    print(f"⚠ {config_path}::{problem_key} [{arch}]")
                    print(f"  Expected: {expected} kernels")
                    print(f"  Actual:   {actual} kernels")
                    print(f"  Extra:    {actual - expected} kernels (baseline may need update)")
                else:
                    print(f"✓ {config_path}::{problem_key} [{arch}]: {actual}/{expected} kernels")

        return all_passed

    def check_yaml(self, yaml_path: Path):
        """Check a single YAML and print expected counts"""
        counts = self.calculate_yaml_counts(yaml_path)

        if not counts:
            print(f"✗ {yaml_path.name}: Not a gfx950 YAML or no problems found")
            return

        print(f"Expected kernel counts for {yaml_path.name}:")
        print()

        total = 0
        for problem, count in counts.items():
            print(f"  {problem}: {count} kernels")
            total += count

        print()
        print(f"Total: {total} kernels across {len(counts)} problem(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Tensile kernel generation counts for gfx950",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate baseline from YAML files"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate Tensile log output against baseline"
    )

    parser.add_argument(
        "--check-yaml",
        action="store_true",
        help="Check a single YAML file and print expected counts"
    )

    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=Path("Tensile/Tests/common"),
        help="Directory containing YAML files (default: Tensile/Tests/common)"
    )

    parser.add_argument(
        "--log",
        type=Path,
        help="Tensile log file to validate"
    )

    parser.add_argument(
        "--yaml",
        type=Path,
        help="Single YAML file to check"
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("kernel_count_baseline_gfx950.json"),
        help="Baseline file path (default: kernel_count_baseline_gfx950.json)"
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="gfx950",
        help="Architecture to validate (default: gfx950)"
    )

    args = parser.parse_args()

    validator = KernelCountValidator(baseline_path=args.baseline)

    if args.generate_baseline:
        if not args.yaml_dir.exists():
            print(f"✗ Directory not found: {args.yaml_dir}")
            sys.exit(1)
        validator.generate_baseline(args.yaml_dir)

    elif args.validate:
        if not args.log:
            print("✗ --log required for validation")
            sys.exit(1)
        if not args.log.exists():
            print(f"✗ Log file not found: {args.log}")
            sys.exit(1)

        success = validator.validate_tensile_log(args.log, arch=args.arch)
        sys.exit(0 if success else 1)

    elif args.check_yaml:
        if not args.yaml:
            print("✗ --yaml required for check-yaml")
            sys.exit(1)
        if not args.yaml.exists():
            print(f"✗ YAML file not found: {args.yaml}")
            sys.exit(1)

        validator.check_yaml(args.yaml)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
