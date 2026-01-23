#!/usr/bin/env python3
"""Run rocroller tests in parallel shards.

Usage: run-tests-sharded.py [BUILD_DIR] [NUM_SHARDS] [GPU_FILTER]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_shard(shard_index, test_exe, test_type, num_shards, gpu_filter, build_dir):
    """Run a single shard of tests."""
    test_prefix = "G" if test_type == "gtest" else "C"
    shard_tag = f"[{test_prefix}{shard_index:02d}]"
    print(f"{shard_tag} Starting shard")

    # Set environment variables
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["OMP_NUM_THREADS"] = "2"

    # Build command based on test type
    if test_type == "gtest":
        # Google Test sharding
        env["GTEST_TOTAL_SHARDS"] = str(num_shards)
        env["GTEST_SHARD_INDEX"] = str(shard_index)

        cmd = [test_exe]
        if gpu_filter:
            cmd.append(gpu_filter)
        cmd.extend(
            ["--gtest_output=xml:test_report/gtest_shard_{}.xml".format(shard_index)]
        )
    else:
        # Catch2 sharding
        cmd = [
            test_exe,
            "--shard-count",
            str(num_shards),
            "--shard-index",
            str(shard_index),
            "-r",
            "junit",
            "-o",
            "test_report/catch2_shard_{}.xml".format(shard_index),
        ]

    try:
        # Run process and capture output
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Read and print output line by line with prefix
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(f"{shard_tag} {line}", flush=True)

        # Wait for process to complete
        return_code = process.wait()

        if return_code == 0:
            print(f"{shard_tag} Completed successfully")
            return (shard_index, test_type, 0)
        else:
            print(f"{shard_tag} Failed with exit code {return_code}")
            return (shard_index, test_type, return_code)

    except Exception as e:
        print(f"{shard_tag} Exception: {e}")
        return (shard_index, test_type, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Run rocroller tests in parallel shards"
    )
    parser.add_argument(
        "build_dir",
        nargs="?",
        default="build",
        help="Path to build directory (default: build)",
    )
    parser.add_argument(
        "num_shards",
        nargs="?",
        type=int,
        default=4,
        help="Number of parallel shards (default: 4)",
    )
    parser.add_argument(
        "gpu_filter",
        nargs="?",
        default="",
        help="GPU filter for Google Test (e.g., --gtest_filter=-*GPU*)",
    )

    args = parser.parse_args()

    # Resolve build directory
    build_dir = Path(args.build_dir).resolve()
    if not build_dir.exists():
        print(f"ERROR: Build directory does not exist: {build_dir}")
        sys.exit(1)

    # Check test executables exist
    gtest_exe = build_dir / "test" / "rocroller-tests"
    catch2_exe = build_dir / "test" / "rocroller-tests-catch"

    if not gtest_exe.exists():
        print(f"ERROR: Google Test executable not found: {gtest_exe}")
        sys.exit(1)

    if not catch2_exe.exists():
        print(f"ERROR: Catch2 executable not found: {catch2_exe}")
        sys.exit(1)

    print("=" * 50)
    print("Running sharded tests")
    print(f"Build directory: {build_dir}")
    print(f"Number of shards: {args.num_shards}")
    print(f"GPU filter: {args.gpu_filter if args.gpu_filter else 'none'}")
    print("=" * 50)

    # Create test report directory
    test_report_dir = build_dir / "test_report"
    test_report_dir.mkdir(exist_ok=True)

    # Run all shards
    failed_shards = []

    # Run Google Test shards first
    print("\n--- Running Google Test shards ---")
    gtest_tasks = []
    for i in range(args.num_shards):
        gtest_tasks.append(
            (i, str(gtest_exe), "gtest", args.num_shards, args.gpu_filter, build_dir)
        )

    with ProcessPoolExecutor(max_workers=args.num_shards) as executor:
        futures = [executor.submit(run_shard, *task) for task in gtest_tasks]

        for future in as_completed(futures):
            try:
                shard_index, test_type, exit_code = future.result()
                if exit_code != 0:
                    failed_shards.append((shard_index, test_type, exit_code))
            except Exception as e:
                print(f"ERROR: Unexpected exception: {e}")
                failed_shards.append((-1, "unknown", 1))

    # Run Catch2 shards after Google Test completes
    print("\n--- Running Catch2 shards ---")
    catch2_tasks = []
    for i in range(args.num_shards):
        catch2_tasks.append(
            (i, str(catch2_exe), "catch2", args.num_shards, args.gpu_filter, build_dir)
        )

    with ProcessPoolExecutor(max_workers=args.num_shards) as executor:
        futures = [executor.submit(run_shard, *task) for task in catch2_tasks]

        for future in as_completed(futures):
            try:
                shard_index, test_type, exit_code = future.result()
                if exit_code != 0:
                    failed_shards.append((shard_index, test_type, exit_code))
            except Exception as e:
                print(f"ERROR: Unexpected exception: {e}")
                failed_shards.append((-1, "unknown", 1))

    print()

    # Report results
    if failed_shards:
        print("FAILED: One or more test shards failed:")
        for shard_index, test_type, exit_code in failed_shards:
            print(f"  - {test_type} shard {shard_index}: exit code {exit_code}")
        sys.exit(1)
    else:
        print("SUCCESS: All test shards completed successfully.")
        print()
        print(f"Test results written to: {test_report_dir}/")

        # List generated XML files
        xml_files = sorted(test_report_dir.glob("*.xml"))
        if xml_files:
            for xml_file in xml_files:
                size = xml_file.stat().st_size
                print(f"  - {xml_file.name} ({size:,} bytes)")

        sys.exit(0)


if __name__ == "__main__":
    main()
