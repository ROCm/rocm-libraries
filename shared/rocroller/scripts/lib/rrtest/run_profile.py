#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Run and time tests for a given profile.

This utility helps identify fast/slow tests to create appropriate test profiles
(e.g., a "smoke" profile with only fast tests).

Usage:
    python3 run_profile.py <profile_name> [options]
    python3 run_profile.py precheckin
    python3 run_profile.py codecov --build-dir build
    python3 run_profile.py precheckin --sort-by time
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rrtest import get_test_commands


@dataclass
class TestResult:
    """Results from running a test command."""

    command: list[str]
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
    success: bool


def run_test_command(command: list[str], cwd: Path | None = None) -> TestResult:
    """
    Run a test command and capture timing and results.

    Args:
        command: Full command line as list
        cwd: Working directory

    Returns:
        TestResult with timing and output information
    """
    print(f"\n{'=' * 70}")
    print(f"Command: {' '.join(command)}")
    if cwd:
        print(f"Working directory: {cwd}")
    print(f"{'=' * 70}")

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )

        duration = time.time() - start_time

        print(f"✓ Completed in {duration:.2f}s (exit code: {result.returncode})")

        return TestResult(
            command=command,
            returncode=result.returncode,
            duration_seconds=duration,
            stdout=result.stdout,
            stderr=result.stderr,
            success=(result.returncode == 0),
        )

    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ Failed after {duration:.2f}s: {e}")

        return TestResult(
            command=command,
            returncode=-1,
            duration_seconds=duration,
            stdout="",
            stderr=str(e),
            success=False,
        )


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def print_summary(results: list[TestResult], sort_by: str = "order"):
    """
    Print a summary of test results.

    Args:
        results: List of TestResult objects
        sort_by: How to sort results ("order", "time", "name")
    """
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    # Sort results
    if sort_by == "time":
        sorted_results = sorted(results, key=lambda r: r.duration_seconds, reverse=True)
    elif sort_by == "name":
        sorted_results = sorted(results, key=lambda r: r.command[0])
    else:  # order
        sorted_results = results

    total_time = sum(r.duration_seconds for r in results)
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed

    print(
        f"\nTotal: {len(results)} commands(s) | "
        f"Passed: {passed} | Failed: {failed} | "
        f"Time: {format_duration(total_time)}"
    )
    print()

    # Print individual results
    for i, result in enumerate(sorted_results, 1):
        status = "✓" if result.success else "✗"
        print(
            f"{i:2d}. {status} {result.command[0]:40s} {format_duration(result.duration_seconds):>10s}"
        )

    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run and time tests for a given profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_profile.py precheckin
  python3 run_profile.py codecov --build-dir build
  python3 run_profile.py precheckin --sort-by time
  python3 run_profile.py codecov --max-time 60
        """,
    )

    parser.add_argument("profile", help="Profile name from test-profiles.yaml")
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory (default: current directory)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["order", "time", "name"],
        default="order",
        help="Sort results by: order (default), time, or name",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        help="Skip remaining commands if total time exceeds this (seconds)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show stdout/stderr from test runs",
    )

    args = parser.parse_args()

    try:
        # Get commands for the profile
        print(f"Loading profile: {args.profile}")
        commands = get_test_commands(args.profile)
        print(f"Found {len(commands)} command(s) to run")

        # Track wall-clock time
        start_time = time.time()

        # Run each command and collect results
        results = []
        total_test_time = 0.0

        for framework, commands in commands.items():
            # Check if we've exceeded max time
            if args.max_time and total_test_time > args.max_time:
                print(
                    f"\n⚠ Skipping remaining tests (exceeded --max-time {args.max_time}s)"
                )
                break

            for command in commands:
                result = run_test_command(command, cwd=args.build_dir)
                results.append(result)
                total_test_time += result.duration_seconds

            # Show verbose output if requested
            if args.verbose:
                if result.stdout:
                    print("\nSTDOUT:")
                    print(result.stdout)
                if result.stderr:
                    print("\nSTDERR:")
                    print(result.stderr)

        # Calculate total elapsed time
        total_elapsed = time.time() - start_time

        # Print summary
        print_summary(results, sort_by=args.sort_by)

        # Print timing information
        print(f"\n{'=' * 70}")
        print(f"Test execution time: {format_duration(total_test_time)}")
        print(f"Total elapsed time:  {format_duration(total_elapsed)}")
        if total_elapsed > total_test_time:
            overhead = total_elapsed - total_test_time
            print(f"Overhead:            {format_duration(overhead)}")
        print(f"{'=' * 70}")

        # Exit with error if any tests failed
        failed_count = sum(1 for r in results if not r.success)
        sys.exit(1 if failed_count > 0 else 0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
