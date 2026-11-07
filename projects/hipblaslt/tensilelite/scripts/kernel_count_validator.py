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
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml


class KernelCountValidator:
    """Calculate and validate kernel counts for gfx950"""

    def __init__(self, baseline_path: Path = None):
        if baseline_path is None:
            # Default to baseline file in same directory as this script
            baseline_path = Path(__file__).resolve().parent / "kernel_count_baseline_gfx950.json"
        self.baseline_path = baseline_path
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

    def _is_ductile_backend(self, yaml_data: Dict) -> bool:
        """Check if YAML uses ductile backend (evolutionary search, not exhaustive)."""
        backend = yaml_data.get("Backend", {})
        if isinstance(backend, dict):
            name = backend.get("Name", "")
            return name.lower() == "ductile"
        return False

    def _calculate_fork_combinations(self, fork_params: List[Dict]) -> int:
        """Calculate total combinations from ForkParameters.

        Mirrors the logic in BenchmarkStructs.constructForkPermutations:
        - Extract 'Groups' entries (pre-defined parameter combinations)
        - For each cartesian product of group entries, multiply fork params
          NOT covered by the group
        - Sum across all group combinations
        """
        if not fork_params:
            return 1

        # Separate Groups from regular fork params
        regular_params = {}  # name -> list of values
        param_groups = []    # list of lists of dicts

        for param_dict in fork_params:
            for key, values in param_dict.items():
                if key == "Groups":
                    # Groups is a list of lists of dicts
                    # e.g. [[{MI: [...], WG: [...]}, {MI: [...], WG: [...]}], [{DTL: 0}, {DTL: 1}]]
                    if isinstance(values, list):
                        param_groups = self._expand_groups(values)
                else:
                    # Regular fork param
                    if isinstance(values, list):
                        regular_params[key] = len(values)
                    else:
                        regular_params[key] = 1

        # If no groups, just multiply all regular params
        if not param_groups:
            total = 1
            for count in regular_params.values():
                total *= count
            return total

        # With groups: sum over all group combinations
        # For each group combination, multiply params not in the group
        total_permutations = 0
        for groups_combo in self._cartesian_product(param_groups):
            # Collect all param names covered by this group combination
            covered_names = set()
            for g in groups_combo:
                if isinstance(g, dict):
                    covered_names.update(g.keys())

            # Multiply only uncovered params
            combo_perms = 1
            for name, count in regular_params.items():
                if name not in covered_names:
                    combo_perms *= count

            total_permutations += combo_perms

        return total_permutations

    def _expand_groups(self, groups: List) -> List[List[Dict]]:
        """Expand Groups structure into list of lists of dicts.

        Groups can be:
        - List of dicts: [[{MI: ..., WG: ...}, {MI: ..., WG: ...}], [{DTL: 0}, ...]]
        - Each inner list represents choices at that level
        """
        expanded = []
        for group in groups:
            if isinstance(group, list):
                # List of entries at this level
                entries = []
                for entry in group:
                    if isinstance(entry, dict):
                        entries.append(entry)
                if entries:
                    expanded.append(entries)
        return expanded

    def _cartesian_product(self, lists: List[List]) -> List[List]:
        """Simple cartesian product implementation."""
        if not lists:
            return [[]]
        result = [[]]
        for lst in lists:
            result = [r + [x] for r in result for x in lst]
        return result

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

        # Skip ductile backend - it uses evolutionary search, not exhaustive enumeration
        if self._is_ductile_backend(yaml_data):
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

    def parse_actual_counts_from_log(self, log_content: str) -> List[Tuple[int, int]]:
        """Parse actual and attempted solution counts from Tensile log output.

        Returns list of (actual, attempted) tuples (one per problem/benchmark step).

        Uses the "after SolutionStructs" line which shows:
        - attempted = total from ForkParameters expansion
        - actual = solutions that passed validation

        The "after KernelWriter" line has attempted=actual since KernelWriter
        starts with the already-filtered set.
        """
        # Parse "# Actual Solutions: X / Y after SolutionStructs" from output
        # This gives us the true attempted count from ForkParameters
        pattern = r'# Actual Solutions:\s+(\d+)\s+/\s+(\d+)\s+after SolutionStructs'
        matches = re.findall(pattern, log_content)

        if not matches:
            # Fall back to KernelWriter line
            pattern = r'# Actual Solutions:\s+(\d+)\s+/\s+(\d+)\s+after KernelWriter'
            matches = re.findall(pattern, log_content)

        if not matches:
            # Fall back to any "Actual Solutions" line
            pattern = r'# Actual Solutions:\s+(\d+)\s+/\s+(\d+)'
            matches = re.findall(pattern, log_content)

        return [(int(actual), int(attempted)) for actual, attempted in matches]

    def run_tensile_and_get_actual(self, yaml_path: Path, arch: str = "gfx950",
                                    tensile_path: Optional[Path] = None,
                                    output_dir: Optional[Path] = None,
                                    env: Optional[Dict[str, str]] = None) -> List[int]:
        """Run Tensile on a YAML file and extract actual solution counts.

        Returns list of actual counts (one per problem/benchmark step).
        """
        if tensile_path is None:
            # Find Tensile script relative to this script
            script_dir = Path(__file__).parent.parent
            tensile_path = script_dir / "tensilelite" / "bin" / "Tensile"

        if not tensile_path.exists():
            raise FileNotFoundError(f"Tensile script not found: {tensile_path}")

        # Use temp dir if no output dir specified
        use_temp = output_dir is None
        if use_temp:
            output_dir = Path(tempfile.mkdtemp(prefix="tensile_run_"))

        try:
            cmd = [
                sys.executable,
                str(tensile_path),
                str(yaml_path),
                str(output_dir),
                "--gpu-targets", arch,
                "--cpu-only",
                "--build-only",
            ]

            # Use provided env or inherit current environment
            run_env = os.environ.copy()
            if env:
                run_env.update(env)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per YAML
                env=run_env,
            )

            if result.returncode != 0 and result.stderr:
                # Check for common errors
                if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
                    print(f"\n  ✗ Environment error - rocisa not properly installed")
                    print(f"    Run: pip install -e rocisa/")
                    return []

            return self.parse_actual_counts_from_log(result.stdout)

        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout running {yaml_path.name}")
            return []
        except Exception as e:
            print(f"  ✗ Error running {yaml_path.name}: {e}")
            return []
        finally:
            if use_temp and output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    def _log_name_to_baseline_key(self, log_name: str) -> Optional[str]:
        """Convert log filename back to baseline key.

        E.g., gemm_fp32_tn.log -> gemm/fp32_tn.yaml
              ::gemm_fp32_tn.log -> gemm/fp32_tn.yaml (handles :: prefix from bash script)
        """
        # Remove .log extension and any leading :: or colons
        base = log_name.lstrip(":").replace(".log", "")
        # Try to find matching baseline key
        for key in self.baseline.keys():
            # Convert key to log-style name for comparison
            key_as_log = key.replace("/", "_").replace("\\", "_").replace(".yaml", "").replace(".yml", "")
            if key_as_log == base:
                return key
        return None

    def update_baseline_from_log(self, log_path: Path, arch: str = "gfx950",
                                   save: bool = True) -> bool:
        """Update baseline with actual counts parsed from a Tensile log file.

        Returns True if baseline was updated, False otherwise.
        """
        with open(log_path, 'r') as f:
            log_content = f.read()

        # Extract config file path from log
        config_match = re.search(r'#\s+Config:\s+(.+\.ya?ml)', log_content)
        if not config_match:
            print(f"  ✗ Could not find Config path in {log_path.name}")
            return False

        config_path = Path(config_match.group(1))
        config_name = config_path.name

        # Find the relative path key in baseline
        # First try matching via log filename (handles unique log names like gemm_fp32_tn.log)
        rel_key = self._log_name_to_baseline_key(log_path.name)

        # Fallback: match by YAML filename from log content
        if not rel_key:
            for key in self.baseline.keys():
                if Path(key).name == config_name:
                    rel_key = key
                    break

        if not rel_key:
            print(f"  ✗ No baseline entry for {config_name}")
            return False

        # Parse actual counts from log - returns list of (actual, attempted) tuples
        counts = self.parse_actual_counts_from_log(log_content)
        if not counts:
            print(f"  ✗ No solution counts found in {log_path.name}")
            return False

        # Update baseline
        if arch not in self.baseline[rel_key]:
            self.baseline[rel_key][arch] = {}

        arch_data = self.baseline[rel_key][arch]
        problem_keys = list(arch_data.keys())

        updated = 0
        for idx, problem_key in enumerate(problem_keys):
            if idx >= len(counts):
                break

            actual, attempted = counts[idx]
            current = arch_data[problem_key]

            # Handle both old format (just int) and new format (dict with expected/actual)
            if isinstance(current, int):
                # Old format: the int was our calculated expected
                arch_data[problem_key] = {
                    "expected": attempted,  # Use Tensile's attempted count as expected
                    "actual": actual,
                    "calculated": current   # Keep our calculation for reference
                }
            else:
                # Update existing dict
                arch_data[problem_key]["expected"] = attempted
                arch_data[problem_key]["actual"] = actual

            updated += 1

        if save:
            self._save_baseline()
        print(f"  ✓ {config_name}: {updated} problem(s) updated")
        return True

    def update_baseline_from_logs_dir(self, logs_dir: Path, arch: str = "gfx950"):
        """Update baseline with actual counts from all log files in a directory.

        Expects log files with .log extension containing Tensile output.
        """
        log_files = list(logs_dir.glob("*.log"))
        if not log_files:
            print(f"No .log files found in {logs_dir}")
            return

        print(f"Processing {len(log_files)} log files from {logs_dir}...")

        updated = 0
        failed = 0

        for log_path in sorted(log_files):
            try:
                if self.update_baseline_from_log(log_path, arch=arch, save=False):
                    updated += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ {log_path.name}: Error - {e}")
                failed += 1

        if updated > 0:
            self._save_baseline()

        print(f"\n✓ Updated from {updated} logs, {failed} failed")

    def verify_from_log(self, log_path: Path, arch: str = "gfx950") -> Tuple[bool, str]:
        """Verify kernel counts from a log file against baseline.

        Returns (passed, message) tuple.
        """
        with open(log_path, 'r') as f:
            log_content = f.read()

        # Extract config file path from log
        config_match = re.search(r'#\s+Config:\s+(.+\.ya?ml)', log_content)
        if not config_match:
            return True, f"[SKIP] No Config path in {log_path.name}"

        config_path = Path(config_match.group(1))
        config_name = config_path.name

        # Find baseline entry
        # First try matching via log filename (handles unique log names like gemm_fp32_tn.log)
        rel_key = self._log_name_to_baseline_key(log_path.name)

        # Fallback: match by YAML filename from log content
        if not rel_key:
            for key in self.baseline.keys():
                if Path(key).name == config_name:
                    rel_key = key
                    break

        if not rel_key:
            return True, f"[SKIP] No baseline for {config_name}"

        # Parse counts from log
        counts = self.parse_actual_counts_from_log(log_content)
        if not counts:
            return True, f"[SKIP] No solution counts in {log_path.name}"

        # Compare against baseline
        arch_data = self.baseline.get(rel_key, {}).get(arch, {})
        if not arch_data:
            return True, f"[SKIP] No baseline for {config_name} ({arch})"

        problem_keys = list(arch_data.keys())
        all_passed = True
        messages = []

        for idx, problem_key in enumerate(problem_keys):
            if idx >= len(counts):
                break

            actual_run, _ = counts[idx]
            entry = arch_data[problem_key]

            # Get baseline actual count
            if isinstance(entry, dict):
                baseline_actual = entry.get("actual", entry.get("expected", 0))
            else:
                baseline_actual = entry

            if actual_run < baseline_actual:
                all_passed = False
                messages.append(
                    f"[FAIL] {config_name}::{problem_key}: "
                    f"got {actual_run}, expected {baseline_actual} "
                    f"(regression of {baseline_actual - actual_run})"
                )
            elif actual_run > baseline_actual:
                messages.append(
                    f"[WARN] {config_name}::{problem_key}: "
                    f"got {actual_run}, baseline {baseline_actual} "
                    f"(+{actual_run - baseline_actual})"
                )
            else:
                messages.append(
                    f"[PASS] {config_name}::{problem_key}: {actual_run} solutions"
                )

        return all_passed, "; ".join(messages)

    def verify_from_logs_dir(self, logs_dir: Path, arch: str = "gfx950") -> bool:
        """Verify kernel counts from all logs in a directory against baseline.

        Returns True if all checks pass, False if any regression detected.
        """
        log_files = list(logs_dir.glob("*.log"))
        if not log_files:
            print(f"No .log files found in {logs_dir}")
            return True

        print(f"Verifying {len(log_files)} log files against baseline...")
        print()

        passed_count = 0
        failed_count = 0
        skipped_count = 0
        all_passed = True

        for log_path in sorted(log_files):
            try:
                passed, message = self.verify_from_log(log_path, arch=arch)
                if "[SKIP]" in message:
                    skipped_count += 1
                elif passed:
                    passed_count += 1
                    if "[WARN]" in message:
                        print(f"  {message}")
                    else:
                        # Only print PASS in verbose mode or summarize
                        pass
                else:
                    failed_count += 1
                    all_passed = False
                    print(f"  {message}")
            except Exception as e:
                print(f"  [ERROR] {log_path.name}: {e}")
                failed_count += 1
                all_passed = False

        print()
        print(f"Results: {passed_count} passed, {failed_count} failed, {skipped_count} skipped")

        if all_passed:
            print("✓ All kernel count checks passed")
        else:
            print("✗ Kernel count regression detected!")

        return all_passed

    def _find_baseline_key(self, yaml_path: Path) -> Optional[str]:
        """Find the baseline key for a YAML file.

        Matches by filename, handling paths like:
        - Input: tensilelite/Tests/common/gemm/bf16_tn.yaml
        - Baseline key: gemm/bf16_tn.yaml
        """
        config_name = yaml_path.name
        yaml_str = str(yaml_path)

        # Try exact filename match first
        for key in self.baseline.keys():
            if Path(key).name == config_name:
                # If multiple matches possible, prefer one that matches more of the path
                if key in yaml_str or yaml_str.endswith(key):
                    return key

        # Fallback: just match by filename
        for key in self.baseline.keys():
            if Path(key).name == config_name:
                return key

        return None

    def _run_single_yaml(self, yaml_path: Path, arch: str, tensile_path: Path,
                         work_base: Path) -> Tuple[Path, bool, str]:
        """Run Tensile on a single YAML and verify against baseline.

        Returns (yaml_path, passed, message)
        """
        work_dir = work_base / f"work_{os.getpid()}_{yaml_path.stem}"

        try:
            work_dir.mkdir(parents=True, exist_ok=True)

            # Inherit current environment (including LD_LIBRARY_PATH, PYTHONNOUSERSITE, VIRTUAL_ENV)
            env = os.environ.copy()

            # Use the same Python executable that's running this script
            python_exe = sys.executable

            cmd = [
                python_exe,
                str(tensile_path),
                str(yaml_path),
                str(work_dir),
                "--gpu-targets", arch,
                "--cpu-only",
                "--build-only",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            output = result.stdout + result.stderr

            # Parse and verify
            counts = self.parse_actual_counts_from_log(output)
            if not counts:
                # Debug: show why no counts
                if result.returncode != 0:
                    return yaml_path, True, f"[SKIP] Tensile failed (rc={result.returncode})"
                return yaml_path, True, f"[SKIP] No solution counts in output"

            # Find baseline
            rel_key = self._find_baseline_key(yaml_path)
            if not rel_key:
                # Return special marker for missing baseline (valid YAML but no baseline)
                total_actual = sum(c[0] for c in counts)
                return yaml_path, None, f"[MISSING] No baseline entry ({total_actual} solutions generated)"

            arch_data = self.baseline.get(rel_key, {}).get(arch, {})
            if not arch_data:
                total_actual = sum(c[0] for c in counts)
                return yaml_path, None, f"[MISSING] No baseline for {arch} ({total_actual} solutions generated)"

            # Check each problem
            problem_keys = list(arch_data.keys())
            all_passed = True
            messages = []

            for idx, problem_key in enumerate(problem_keys):
                if idx >= len(counts):
                    break

                actual_run, attempted = counts[idx]
                entry = arch_data[problem_key]

                if isinstance(entry, dict):
                    baseline_actual = entry.get("actual", entry.get("expected", 0))
                else:
                    baseline_actual = entry

                if actual_run < baseline_actual:
                    all_passed = False
                    diff = baseline_actual - actual_run
                    messages.append(
                        f"{problem_key}: actual={actual_run}, baseline={baseline_actual} "
                        f"(REGRESSION: {diff} fewer solutions)"
                    )
                elif actual_run > baseline_actual:
                    diff = actual_run - baseline_actual
                    messages.append(
                        f"{problem_key}: actual={actual_run}, baseline={baseline_actual} "
                        f"(+{diff} new solutions)"
                    )

            if all_passed:
                if messages:
                    return yaml_path, True, f"[WARN] " + "; ".join(messages)
                total_actual = sum(c[0] for c in counts[:len(problem_keys)])
                return yaml_path, True, f"[PASS] {total_actual} solutions"
            else:
                return yaml_path, False, f"[FAIL] " + "; ".join(messages)

        except subprocess.TimeoutExpired:
            return yaml_path, False, "[FAIL] Timeout"
        except Exception as e:
            return yaml_path, False, f"[FAIL] {e}"
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    def run_and_verify(self, yaml_dir: Path, arch: str = "gfx950",
                       parallel: int = 1, tensile_path: Optional[Path] = None) -> bool:
        """Run Tensile on all YAMLs and verify kernel counts against baseline.

        This is the main entry point for kernel count regression testing.

        Args:
            yaml_dir: Directory containing YAML test files
            arch: Target architecture
            parallel: Number of parallel jobs
            tensile_path: Path to Tensile script (auto-detected if None)

        Returns:
            True if all checks pass, False if any regression detected
        """
        # Find Tensile script
        if tensile_path is None:
            script_dir = Path(__file__).parent.parent
            tensile_path = script_dir / "tensilelite" / "bin" / "Tensile"

        if not tensile_path.exists():
            print(f"✗ Tensile script not found: {tensile_path}")
            return False

        # Collect YAMLs
        yaml_files = []
        for pattern in ["**/*.yaml", "**/*.yml"]:
            yaml_files.extend(yaml_dir.glob(pattern))

        # Filter to target arch YAMLs (excluding ductile)
        valid_yamls = []
        for yaml_path in sorted(yaml_files):
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                if self._is_gfx950_yaml(yaml_data) and not self._is_ductile_backend(yaml_data):
                    valid_yamls.append(yaml_path)
            except Exception:
                pass

        if not valid_yamls:
            print(f"No valid {arch} YAML files found in {yaml_dir}")
            return True

        print(f"Running kernel count verification on {len(valid_yamls)} YAML files...")
        print(f"  Architecture: {arch}")
        print(f"  Parallel jobs: {parallel}")
        print()

        # Create temp work directory
        work_base = Path(tempfile.mkdtemp(prefix="kernel_count_"))

        try:
            passed_count = 0
            failed_count = 0
            skipped_count = 0
            missing_count = 0
            failed_yamls = []
            missing_yamls = []

            if parallel > 1:
                # Parallel execution
                with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                    futures = {
                        executor.submit(
                            self._run_single_yaml, yaml_path, arch, tensile_path, work_base
                        ): yaml_path
                        for yaml_path in valid_yamls
                    }

                    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                        yaml_path, passed, message = future.result()
                        config_name = yaml_path.name

                        if "[SKIP]" in message:
                            skipped_count += 1
                        elif "[MISSING]" in message:
                            missing_count += 1
                            missing_yamls.append((yaml_path, message))
                            print(f"  [{i}/{len(valid_yamls)}] {config_name}: {message}")
                        elif passed:
                            passed_count += 1
                            print(f"  [{i}/{len(valid_yamls)}] {config_name}: {message}")
                        else:
                            failed_count += 1
                            failed_yamls.append((config_name, message))
                            print(f"  [{i}/{len(valid_yamls)}] {config_name}: {message}")
            else:
                # Sequential execution
                for i, yaml_path in enumerate(valid_yamls, 1):
                    config_name = yaml_path.name
                    print(f"  [{i}/{len(valid_yamls)}] {config_name}...", end=" ", flush=True)

                    yaml_path_result, passed, message = self._run_single_yaml(
                        yaml_path, arch, tensile_path, work_base
                    )

                    if "[SKIP]" in message:
                        skipped_count += 1
                        print(message)
                    elif "[MISSING]" in message:
                        missing_count += 1
                        missing_yamls.append((yaml_path, message))
                        print(message)
                    elif passed:
                        passed_count += 1
                        print(message)
                    else:
                        failed_count += 1
                        failed_yamls.append((config_name, message))
                        print(message)

            # Summary
            print()
            print("=" * 60)
            print(f"Results: {passed_count} PASS, {failed_count} FAIL, {missing_count} MISSING, {skipped_count} SKIP")
            print("=" * 60)

            if missing_yamls:
                print("\nNew YAMLs missing from baseline:")
                for yaml_path, msg in missing_yamls:
                    print(f"  {yaml_path.name}: {msg}")
                print("\nTo add them to baseline, run:")
                for yaml_path, _ in missing_yamls:
                    print(f"  python3 scripts/kernel_count_validator.py --add-yaml --yaml {yaml_path}")

            if failed_yamls:
                print("\nFailed tests:")
                for name, msg in failed_yamls:
                    print(f"  {name}: {msg}")
                print()
                print("✗ KERNEL COUNT REGRESSION DETECTED")
                return False
            elif missing_yamls:
                print()
                print("⚠ NEW YAMLS NEED BASELINE - run --add-yaml for each")
                return False
            else:
                print("\n✓ ALL KERNEL COUNT CHECKS PASSED")
                return True

        finally:
            if work_base.exists():
                shutil.rmtree(work_base, ignore_errors=True)

    def _yaml_to_log_name(self, yaml_path: Path, yaml_dir: Path) -> str:
        """Convert YAML path to unique log filename preserving directory structure.

        E.g., gemm/fp32_tn.yaml -> gemm_fp32_tn.log
              benchmark_runs/fp32_tn.yaml -> benchmark_runs_fp32_tn.log
        """
        try:
            rel_path = yaml_path.relative_to(yaml_dir)
        except ValueError:
            rel_path = Path(yaml_path.name)

        # Replace path separators with underscores, remove .yaml extension
        return str(rel_path).replace("/", "_").replace("\\", "_").replace(".yaml", ".log").replace(".yml", ".log")

    def generate_run_script(self, yaml_dir: Path, output_script: Path,
                            arch: str = "gfx950", logs_dir: Path = None,
                            parallel: int = 1):
        """Generate a bash script to run Tensile on all YAML files.

        The script runs each YAML with --cpu-only --build-only and saves logs
        to the specified directory. Log filenames include the directory path
        to avoid collisions (e.g., gemm_fp32_tn.log vs benchmark_runs_fp32_tn.log).

        Args:
            parallel: Number of parallel jobs (default: 1 for sequential)
        """
        yaml_files = []
        for pattern in ["**/*.yaml", "**/*.yml"]:
            yaml_files.extend(yaml_dir.glob(pattern))

        if not yaml_files:
            print(f"No YAML files found in {yaml_dir}")
            return

        # Filter to only gfx950 YAMLs (excluding ductile)
        valid_yamls = []
        for yaml_path in sorted(yaml_files):
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                if self._is_gfx950_yaml(yaml_data) and not self._is_ductile_backend(yaml_data):
                    valid_yamls.append(yaml_path)
            except Exception:
                pass

        if not valid_yamls:
            print("No valid gfx950 YAML files found")
            return

        if logs_dir is None:
            logs_dir = Path("tensile_logs")

        # Find Tensile script path
        script_dir = Path(__file__).parent.parent
        tensile_path = script_dir / "tensilelite" / "bin" / "Tensile"

        if parallel > 1:
            # Generate parallel script using xargs
            # Note: We pass yaml_file|log_name pairs to preserve unique log names
            lines = [
                "#!/bin/bash",
                "# Auto-generated script to run Tensile on all gfx950 YAML files",
                f"# Generated for {len(valid_yamls)} YAML files (parallel: {parallel} jobs)",
                "",
                f"LOGS_DIR=\"{logs_dir}\"",
                f"TENSILE=\"{tensile_path}\"",
                f"ARCH=\"{arch}\"",
                f"PARALLEL_JOBS={parallel}",
                "",
                "mkdir -p \"$LOGS_DIR\"",
                "",
                "run_yaml() {",
                "    # Input format: yaml_file|log_name",
                "    yaml_file=\"${1%|*}\"",
                "    log_name=\"${1#*|}\"",
                "    work_dir=\"/tmp/tensile_work_$$_$(basename \"$yaml_file\" .yaml)\"",
                "",
                "    echo \"Running: $yaml_file -> $log_name\"",
                "    mkdir -p \"$work_dir\"",
                "    python3 \"$TENSILE\" \"$yaml_file\" \"$work_dir\" \\",
                "        --gpu-targets \"$ARCH\" --cpu-only --build-only \\",
                "        > \"$LOGS_DIR/$log_name\" 2>&1 || true",
                "    rm -rf \"$work_dir\"",
                "}",
                "export -f run_yaml",
                f"export TENSILE=\"{tensile_path}\"",
                f"export LOGS_DIR=\"{logs_dir}\"",
                f"export ARCH=\"{arch}\"",
                "",
                f"echo \"Running {len(valid_yamls)} YAML files with $PARALLEL_JOBS parallel jobs...\"",
                "echo \"Logs will be saved to: $LOGS_DIR\"",
                "echo",
                "",
                "cat << 'YAML_LIST' | xargs -P $PARALLEL_JOBS -I {} bash -c 'run_yaml \"{}\"'",
            ]

            # Add each YAML file with its unique log name
            for yaml_path in valid_yamls:
                log_name = self._yaml_to_log_name(yaml_path, yaml_dir)
                lines.append(f"{yaml_path}|{log_name}")

            lines.extend([
                "YAML_LIST",
                "",
                "echo",
                "echo \"Done! Logs saved to: $LOGS_DIR\"",
                f"echo \"Run: python3 scripts/kernel_count_validator.py --verify --logs-dir $LOGS_DIR\"",
            ])
        else:
            # Generate sequential script
            lines = [
                "#!/bin/bash",
                "# Auto-generated script to run Tensile on all gfx950 YAML files",
                f"# Generated for {len(valid_yamls)} YAML files",
                "",
                "set -e",
                "",
                f"LOGS_DIR=\"{logs_dir}\"",
                f"TENSILE=\"{tensile_path}\"",
                f"ARCH=\"{arch}\"",
                f"TOTAL_YAMLS={len(valid_yamls)}",
                "",
                "mkdir -p \"$LOGS_DIR\"",
                "mkdir -p /tmp/tensile_work",
                "",
                "COUNT=0",
                "run_yaml() {",
                "    local yaml_file=\"$1\"",
                "    local log_name=\"$2\"",
                "    local work_dir=\"/tmp/tensile_work/$(basename \"$yaml_file\" .yaml)\"",
                "    COUNT=$((COUNT + 1))",
                "",
                "    echo \"[$COUNT/$TOTAL_YAMLS] Running: $yaml_file -> $log_name\"",
                "    mkdir -p \"$work_dir\"",
                "    python3 \"$TENSILE\" \"$yaml_file\" \"$work_dir\" \\",
                "        --gpu-targets \"$ARCH\" --cpu-only --build-only \\",
                "        > \"$LOGS_DIR/$log_name\" 2>&1 || true",
                "    rm -rf \"$work_dir\"",
                "}",
                "",
                "echo \"Running $TOTAL_YAMLS YAML files...\"",
                "echo \"Logs will be saved to: $LOGS_DIR\"",
                "echo",
                "",
            ]

            # Add each YAML file with its unique log name
            for yaml_path in valid_yamls:
                log_name = self._yaml_to_log_name(yaml_path, yaml_dir)
                lines.append(f"run_yaml \"{yaml_path}\" \"{log_name}\"")

            lines.extend([
                "",
                "echo",
                "echo \"Done! Logs saved to: $LOGS_DIR\"",
                f"echo \"Run: python3 scripts/kernel_count_validator.py --verify --logs-dir $LOGS_DIR\"",
            ])

        with open(output_script, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        os.chmod(output_script, 0o755)
        print(f"✓ Generated {output_script} with {len(valid_yamls)} YAML files")
        if parallel > 1:
            print(f"  Parallel jobs: {parallel}")
        print(f"  Run it with: ./{output_script}")
        print(f"  Then verify with: python3 scripts/kernel_count_validator.py --verify --logs-dir {logs_dir}")

    def add_yaml_to_baseline(self, yaml_path: Path, yaml_base_dir: Path = None,
                              arch: str = "gfx950", tensile_path: Optional[Path] = None) -> bool:
        """Run Tensile on a single YAML and add/update its baseline entry.

        Args:
            yaml_path: Path to the YAML file
            yaml_base_dir: Base directory for computing relative path (default: Tests/common)
            arch: Target architecture
            tensile_path: Path to Tensile script (auto-detected if None)

        Returns:
            True if successful, False otherwise
        """
        if not yaml_path.exists():
            print(f"✗ YAML file not found: {yaml_path}")
            return False

        # Find Tensile script
        if tensile_path is None:
            script_dir = Path(__file__).parent.parent
            tensile_path = script_dir / "tensilelite" / "bin" / "Tensile"

        if not tensile_path.exists():
            print(f"✗ Tensile script not found: {tensile_path}")
            return False

        # Determine base directory for relative path
        if yaml_base_dir is None:
            # Try to find Tests/common in the path
            yaml_str = str(yaml_path.resolve())
            if "Tests/common/" in yaml_str:
                idx = yaml_str.index("Tests/common/")
                yaml_base_dir = Path(yaml_str[:idx + len("Tests/common")])
            else:
                yaml_base_dir = yaml_path.parent

        # Check if it's a valid gfx950 YAML
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
        except Exception as e:
            print(f"✗ Failed to parse YAML: {e}")
            return False

        if not self._is_gfx950_yaml(yaml_data):
            print(f"✗ {yaml_path.name} is not a gfx950-compatible YAML")
            return False

        if self._is_ductile_backend(yaml_data):
            print(f"✗ {yaml_path.name} uses ductile backend (not supported)")
            return False

        # Calculate expected counts from ForkParameters
        expected_counts = self.calculate_yaml_counts(yaml_path)

        print(f"Running Tensile on {yaml_path.name}...")

        # Run Tensile
        work_dir = Path(tempfile.mkdtemp(prefix="tensile_add_"))
        try:
            cmd = [
                sys.executable,
                str(tensile_path),
                str(yaml_path),
                str(work_dir),
                "--gpu-targets", arch,
                "--cpu-only",
                "--build-only",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=os.environ.copy(),
            )

            output = result.stdout + result.stderr

            # Parse actual counts
            counts = self.parse_actual_counts_from_log(output)
            if not counts:
                print(f"✗ No solution counts in Tensile output")
                if result.returncode != 0:
                    print(f"  Tensile exited with code {result.returncode}")
                return False

            # Compute relative path for baseline key
            try:
                rel_path = yaml_path.resolve().relative_to(yaml_base_dir.resolve())
            except ValueError:
                rel_path = Path(yaml_path.name)

            rel_path_str = str(rel_path)

            # Update baseline
            if rel_path_str not in self.baseline:
                self.baseline[rel_path_str] = {}
            if arch not in self.baseline[rel_path_str]:
                self.baseline[rel_path_str][arch] = {}

            # Get problem keys from expected counts, or generate them
            if expected_counts:
                problem_keys = list(expected_counts.keys())
            else:
                # No ForkParameters - generate problem keys from count
                problem_keys = [f"problem_{i}" for i in range(len(counts))]

            for idx, problem_key in enumerate(problem_keys):
                if idx >= len(counts):
                    break

                actual, attempted = counts[idx]
                expected = expected_counts.get(problem_key, attempted) if expected_counts else attempted

                self.baseline[rel_path_str][arch][problem_key] = {
                    "expected": attempted,
                    "actual": actual,
                }

            self._save_baseline()

            total_actual = sum(c[0] for c in counts[:len(problem_keys)])
            total_expected = sum(c[1] for c in counts[:len(problem_keys)])
            print(f"✓ Added {rel_path_str}: {total_actual}/{total_expected} solutions")
            return True

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout running Tensile")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    def run_and_update_baseline(self, yaml_dir: Path, arch: str = "gfx950",
                                 tensile_path: Optional[Path] = None):
        """Run Tensile on all YAML files and update baseline with actual counts."""
        yaml_files = []
        for pattern in ["**/*.yaml", "**/*.yml"]:
            yaml_files.extend(yaml_dir.glob(pattern))

        if not yaml_files:
            print(f"No YAML files found in {yaml_dir}")
            return

        print(f"Running Tensile on {len(yaml_files)} YAML files for {arch}...")
        print("This may take a while...\n")

        updated = 0
        skipped = 0
        failed = 0

        for yaml_path in sorted(yaml_files):
            try:
                # First check if this is a valid gfx950 YAML
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)

                if not self._is_gfx950_yaml(yaml_data):
                    continue

                if self._is_ductile_backend(yaml_data):
                    skipped += 1
                    continue

                # Calculate expected counts
                expected_counts = self.calculate_yaml_counts(yaml_path)
                if not expected_counts:
                    continue

                # Run Tensile and get actual counts
                print(f"  Running {yaml_path.name}...", end=" ", flush=True)
                actual_counts = self.run_tensile_and_get_actual(
                    yaml_path, arch=arch, tensile_path=tensile_path
                )

                if not actual_counts:
                    print("FAILED (no output)")
                    failed += 1
                    continue

                # Update baseline with both expected and actual
                try:
                    rel_path = yaml_path.relative_to(yaml_dir)
                except ValueError:
                    rel_path = yaml_path

                rel_path_str = str(rel_path)
                if rel_path_str not in self.baseline:
                    self.baseline[rel_path_str] = {}
                if arch not in self.baseline[rel_path_str]:
                    self.baseline[rel_path_str][arch] = {}

                problem_keys = list(expected_counts.keys())
                for idx, problem_key in enumerate(problem_keys):
                    expected = expected_counts[problem_key]
                    actual = actual_counts[idx] if idx < len(actual_counts) else 0

                    self.baseline[rel_path_str][arch][problem_key] = {
                        "expected": expected,
                        "actual": actual
                    }

                total_expected = sum(expected_counts.values())
                total_actual = sum(actual_counts[:len(problem_keys)])
                print(f"OK ({total_actual}/{total_expected} solutions)")
                updated += 1

            except Exception as e:
                print(f"  ✗ {yaml_path.name}: Error - {e}")
                failed += 1

        # Save updated baseline
        if updated > 0:
            self._save_baseline()

        print(f"\n✓ Updated {updated} files, skipped {skipped} ductile, {failed} failed")


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
        "--run-and-update",
        action="store_true",
        help="Run Tensile on all YAML files and update baseline with actual counts"
    )

    parser.add_argument(
        "--update-from-log",
        action="store_true",
        help="Update baseline with actual counts from a Tensile log file (--log required)"
    )

    parser.add_argument(
        "--update-from-logs-dir",
        action="store_true",
        help="Update baseline with actual counts from all .log files in a directory (--logs-dir required)"
    )

    parser.add_argument(
        "--logs-dir",
        type=Path,
        help="Directory containing Tensile log files"
    )

    parser.add_argument(
        "--generate-run-script",
        action="store_true",
        help="Generate a bash script to run Tensile on all YAML files"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify kernel counts from logs against baseline (use with --logs-dir)"
    )

    parser.add_argument(
        "--run-and-verify",
        action="store_true",
        help="Run Tensile on all YAMLs and verify kernel counts (single command)"
    )

    parser.add_argument(
        "--add-yaml",
        action="store_true",
        help="Run Tensile on a single YAML and add/update its baseline entry (use with --yaml)"
    )

    parser.add_argument(
        "--output-script",
        type=Path,
        default=Path("run_all_yamls.sh"),
        help="Output path for generated run script (default: run_all_yamls.sh)"
    )

    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        help="Number of parallel jobs for --generate-run-script (default: 1)"
    )

    parser.add_argument(
        "--tensile-path",
        type=Path,
        default=None,
        help="Path to Tensile script (default: auto-detect)"
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
        default=None,
        help="Baseline file path (default: scripts/kernel_count_baseline_gfx950.json)"
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

    elif args.run_and_update:
        if not args.yaml_dir.exists():
            print(f"✗ Directory not found: {args.yaml_dir}")
            sys.exit(1)
        validator.run_and_update_baseline(
            args.yaml_dir,
            arch=args.arch,
            tensile_path=args.tensile_path
        )

    elif args.update_from_log:
        if not args.log:
            print("✗ --log required for --update-from-log")
            sys.exit(1)
        if not args.log.exists():
            print(f"✗ Log file not found: {args.log}")
            sys.exit(1)
        validator.update_baseline_from_log(args.log, arch=args.arch)

    elif args.update_from_logs_dir:
        if not args.logs_dir:
            print("✗ --logs-dir required for --update-from-logs-dir")
            sys.exit(1)
        if not args.logs_dir.exists():
            print(f"✗ Directory not found: {args.logs_dir}")
            sys.exit(1)
        validator.update_baseline_from_logs_dir(args.logs_dir, arch=args.arch)

    elif args.generate_run_script:
        if not args.yaml_dir.exists():
            print(f"✗ Directory not found: {args.yaml_dir}")
            sys.exit(1)
        validator.generate_run_script(
            args.yaml_dir,
            args.output_script,
            arch=args.arch,
            logs_dir=args.logs_dir or Path("tensile_logs"),
            parallel=args.parallel
        )

    elif args.verify:
        if not args.logs_dir:
            print("✗ --logs-dir required for --verify")
            sys.exit(1)
        if not args.logs_dir.exists():
            print(f"✗ Directory not found: {args.logs_dir}")
            sys.exit(1)
        success = validator.verify_from_logs_dir(args.logs_dir, arch=args.arch)
        sys.exit(0 if success else 1)

    elif args.add_yaml:
        if not args.yaml:
            print("✗ --yaml required for --add-yaml")
            sys.exit(1)
        if not args.yaml.exists():
            print(f"✗ YAML file not found: {args.yaml}")
            sys.exit(1)
        success = validator.add_yaml_to_baseline(
            args.yaml,
            yaml_base_dir=args.yaml_dir if args.yaml_dir.exists() else None,
            arch=args.arch,
            tensile_path=args.tensile_path
        )
        sys.exit(0 if success else 1)

    elif args.run_and_verify:
        if args.yaml:
            # Single YAML file
            if not args.yaml.exists():
                print(f"✗ YAML file not found: {args.yaml}")
                sys.exit(1)

            tensile_path = args.tensile_path
            if tensile_path is None:
                script_dir = Path(__file__).parent.parent
                tensile_path = script_dir / "tensilelite" / "bin" / "Tensile"

            work_base = Path(tempfile.mkdtemp(prefix="kernel_count_"))
            try:
                print(f"Running kernel count verification on {args.yaml.name}...")
                yaml_path, passed, message = validator._run_single_yaml(
                    args.yaml, args.arch, tensile_path, work_base
                )
                print(f"  {yaml_path.name}: {message}")
                print()
                if "[MISSING]" in message:
                    print("⚠ YAML is valid but has no baseline entry")
                    print(f"\nTo add it to baseline, run:")
                    print(f"  python3 scripts/kernel_count_validator.py --add-yaml --yaml {args.yaml}")
                    sys.exit(1)
                elif passed:
                    print("✓ PASS")
                else:
                    print("✗ FAIL")
                sys.exit(0 if passed else 1)
            finally:
                if work_base.exists():
                    shutil.rmtree(work_base, ignore_errors=True)
        else:
            # Directory of YAMLs
            if not args.yaml_dir.exists():
                print(f"✗ Directory not found: {args.yaml_dir}")
                sys.exit(1)
            success = validator.run_and_verify(
                args.yaml_dir,
                arch=args.arch,
                parallel=args.parallel,
                tensile_path=args.tensile_path
            )
            sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
