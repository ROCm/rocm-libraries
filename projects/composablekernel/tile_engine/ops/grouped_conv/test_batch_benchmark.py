#!/usr/bin/env python3
"""Test batch benchmark with first 2 kernels x 5 problems = 10 measurements."""

import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

# We'll use the existing forward_bf16.json config but limit to first 2 kernels
# This avoids config format issues

print("Testing batch benchmark with small dataset")
print("=" * 80)
print("Config: forward_bf16.json (limited to first 2 kernels)")
print("Problems: Using forward_training_small (5 problems)")
print("Batch size: 2 (both kernels in one subprocess per problem)")
print()

# Run benchmark
cmd = [
    sys.executable,
    str(_THIS_DIR / "grouped_conv_full_benchmark.py"),
    str(_THIS_DIR / "configs/forward_bf16.json"),
    "--arch", "gfx950",
    "--problems", "forward_training_small",
    "--csv", str(_THIS_DIR / "test_batch_results.csv"),
    "--workers", "4",
    "--batch-size", "2",
    "--kernel-timeout", "10",
    "--max-kernels", "2",  # Limit to first 2 kernels
]

print(f"Command: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, cwd=_THIS_DIR)

if result.returncode == 0:
    print("\n" + "=" * 80)
    print("✓ Batch benchmark test PASSED")
    print("=" * 80)

    # Show results
    csv_file = _THIS_DIR / "test_batch_results.csv"
    if csv_file.exists():
        with open(csv_file) as f:
            lines = f.readlines()
            print(f"\nResults ({len(lines) - 1} measurements):")
            print("".join(lines[:6]))  # Header + first 5 rows
else:
    print("\n" + "=" * 80)
    print(f"✗ Batch benchmark test FAILED (exit code {result.returncode})")
    print("=" * 80)

sys.exit(result.returncode)
