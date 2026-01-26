#!/usr/bin/env python3
"""Run rocroller tests in parallel shards.

Usage: run-tests-sharded.py [BUILD_DIR] [GPU_FILTER]

Number of shards is auto-detected as ncores/2.
"""

import argparse
import os
import pty
import select
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ShardInfo:
    """Information about a running test shard."""

    process: subprocess.Popen
    master_fd: int
    shard_tag: str
    shard_index: int
    test_type: str
    output_buffer: str = ""


def get_gpu_arch():
    """Detect GPU architecture (e.g., gfx1201) if possible."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        # Look for gfx architecture in output
        for line in result.stdout.splitlines():
            if "Name:" in line and "gfx" in line.lower():
                # Extract gfx architecture (e.g., gfx1201)
                parts = line.lower().split()
                for part in parts:
                    if part.startswith("gfx"):
                        return part.strip()
    except:
        pass
    return None


def get_available_cpus():
    """Get the number of CPUs available, respecting cgroups and affinity."""
    try:
        result = subprocess.run(
            ["nproc"], capture_output=True, text=True, check=True, timeout=1
        )
        return int(result.stdout.strip())
    except:
        pass

    return os.cpu_count() or 4


def start_shard(shard_index, test_exe, test_type, num_shards, gpu_filter, build_dir):
    """Start a shard of tests and return process info."""
    test_prefix = "G" if test_type == "gtest" else "C"
    shard_tag = f"[{test_prefix}{shard_index:02d}]"
    print(f"{shard_tag} Starting shard", flush=True)

    # Set environment variables
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["OMP_NUM_THREADS"] = "2"

    # Build command based on test type
    if test_type == "gtest":
        env["GTEST_TOTAL_SHARDS"] = str(num_shards)
        env["GTEST_SHARD_INDEX"] = str(shard_index)

        cmd = [test_exe]
        cmd.append("--gtest_shuffle")
        if gpu_filter:
            cmd.append(gpu_filter)
        cmd.extend(
            ["--gtest_output=xml:test_report/gtest_shard_{}.xml".format(shard_index)]
        )
    else:
        cmd = [
            test_exe,
            "--order",
            "rand",
            "--shard-count",
            str(num_shards),
            "--shard-index",
            str(shard_index),
            "-r",
            "junit",
            "-o",
            "test_report/catch2_shard_{}.xml".format(shard_index),
        ]

    # Log the command being executed
    cmd_str = " ".join(cmd)
    print(f"{shard_tag} Running command in {build_dir}:", flush=True)
    print(f"{shard_tag}   {cmd_str}", flush=True)

    # Use pseudo-TTY to ensure line-buffered output from subprocess
    master_fd, slave_fd = pty.openpty()

    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=build_dir,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )

    # Close slave fd in parent process
    os.close(slave_fd)

    return ShardInfo(
        process=process,
        master_fd=master_fd,
        shard_tag=shard_tag,
        shard_index=shard_index,
        test_type=test_type,
        output_buffer="",
    )


def run_shards(shard_infos):
    """Run multiple shards and multiplex their I/O using select."""
    active_shards = {info.master_fd: info for info in shard_infos}
    failed_shards = []

    while active_shards:
        fds = list(active_shards.keys())

        ready, _, _ = select.select(fds, [], [], 0.1)

        for fd in ready:
            info = active_shards[fd]

            try:
                chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
                if chunk:
                    info.output_buffer += chunk

                    # Process complete lines
                    while "\n" in info.output_buffer:
                        line, info.output_buffer = info.output_buffer.split("\n", 1)
                        line = line.rstrip("\r")
                        if line:
                            print(f"{info.shard_tag} {line}", flush=True)
            except OSError:
                # File descriptor closed or error
                pass

        # Check which processes have finished
        finished_fds = []
        for fd, info in active_shards.items():
            if info.process.poll() is not None:
                finished_fds.append(fd)

        # Handle finished processes
        for fd in finished_fds:
            info = active_shards[fd]

            # Read any remaining output
            try:
                while True:
                    chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    info.output_buffer += chunk
            except OSError:
                pass

            # Print any remaining buffered output
            if info.output_buffer:
                for line in info.output_buffer.split("\n"):
                    line = line.rstrip("\r")
                    if line:
                        print(f"{info.shard_tag} {line}", flush=True)

            # Close file descriptor
            os.close(fd)

            # Get return code
            return_code = info.process.wait()

            if return_code == 0:
                print(f"{info.shard_tag} Completed successfully", flush=True)
            else:
                print(
                    f"{info.shard_tag} Failed with exit code {return_code}", flush=True
                )
                failed_shards.append((info.shard_index, info.test_type, return_code))

            # Remove from active shards
            del active_shards[fd]

    return failed_shards


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
        "gpu_filter",
        nargs="?",
        default="",
        help="GPU filter for Google Test (e.g., --gtest_filter=-*GPU*)",
    )

    args = parser.parse_args()

    # Auto-detect number of available CPUs (respecting cgroups) and use half
    num_cores = get_available_cpus()
    num_shards = max(1, num_cores // 2)

    # Cap shards to 8 on gfx1201 due to hanging issues
    gpu_arch = get_gpu_arch()
    if gpu_arch and "gfx1201" in gpu_arch:
        if num_shards > 8:
            print(f"Detected {gpu_arch}, capping shards to 8 (was {num_shards})")
            num_shards = 8

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
    print(f"Available cores: {num_cores}")
    if gpu_arch:
        print(f"GPU architecture: {gpu_arch}")
    print(f"Number of shards: {num_shards} (auto-detected)")
    print(f"GPU filter: {args.gpu_filter if args.gpu_filter else 'none'}")
    print("=" * 50)

    # Create test report directory
    test_report_dir = build_dir / "test_report"
    test_report_dir.mkdir(exist_ok=True)

    # Run all shards
    failed_shards = []

    # Run Google Test shards first
    print("\n--- Running Google Test shards ---")
    gtest_infos = []
    for i in range(num_shards):
        try:
            info = start_shard(
                i, str(gtest_exe), "gtest", num_shards, args.gpu_filter, build_dir
            )
            gtest_infos.append(info)
        except Exception as e:
            print(f"ERROR: Failed to start gtest shard {i}: {e}", flush=True)
            failed_shards.append((i, "gtest", 1))

    # Run all gtest shards and wait for completion
    gtest_failures = run_shards(gtest_infos)
    failed_shards.extend(gtest_failures)

    # Run Catch2 shards after Google Test completes
    print("\n--- Running Catch2 shards ---")
    catch2_infos = []
    for i in range(num_shards):
        try:
            info = start_shard(
                i,
                str(catch2_exe),
                "catch2",
                num_shards,
                args.gpu_filter,
                build_dir,
            )
            catch2_infos.append(info)
        except Exception as e:
            print(f"ERROR: Failed to start catch2 shard {i}: {e}", flush=True)
            failed_shards.append((i, "catch2", 1))

    # Run all catch2 shards and wait for completion
    catch2_failures = run_shards(catch2_infos)
    failed_shards.extend(catch2_failures)

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
