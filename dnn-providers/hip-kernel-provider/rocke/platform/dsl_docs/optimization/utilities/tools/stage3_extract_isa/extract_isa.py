# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ISA extraction tool for FlyDSL and CK Tile kernels."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def extract_flydsl_isa(kernel_path: Path, output_dir: Path) -> Path:
    """Extract ISA from a FlyDSL kernel.

    Args:
        kernel_path: Path to FlyDSL kernel Python file
        output_dir: Directory to save ISA files

    Returns:
        Path to extracted ISA file
    """
    print(f"Extracting FlyDSL ISA from: {kernel_path}")

    # Set up environment for ISA dump
    env = os.environ.copy()
    env["FLYDSL_DUMP_IR"] = "1"
    env["FLYDSL_DUMP_DIR"] = str(output_dir)
    env["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    env["FLYDSL_DEBUG_ENABLE_DEBUG_INFO"] = "1"

    # Run the kernel to trigger ISA dump
    result = subprocess.run(
        [sys.executable, str(kernel_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"Error running FlyDSL kernel:\n{result.stderr}")
        return None

    # Find the generated ISA file
    isa_files = list(output_dir.glob("*/*_final_isa.s"))
    if not isa_files:
        print(f"No ISA file found in {output_dir}")
        return None

    isa_file = isa_files[0]
    print(f"✓ ISA extracted to: {isa_file}")
    return isa_file


def extract_cktile_isa(so_path: Path, output_dir: Path) -> Path:
    """Extract ISA from a CK Tile .so file.

    Args:
        so_path: Path to CK Tile .so file
        output_dir: Directory to save ISA files

    Returns:
        Path to extracted ISA file
    """
    print(f"Extracting CK Tile ISA from: {so_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract code object from .so
    result = subprocess.run(
        ["roc-obj-extract", str(so_path)],
        cwd=output_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error extracting code object:\n{result.stderr}")
        return None

    # Find extracted .co file
    co_files = list(output_dir.glob("*.co"))
    if not co_files:
        print(f"No .co file found in {output_dir}")
        return None

    co_file = co_files[0]
    print(f"  Code object extracted: {co_file}")

    # Disassemble with llvm-objdump
    isa_file = output_dir / "kernel.s"
    llvm_objdump = Path("/opt/rocm/llvm/bin/llvm-objdump")

    if not llvm_objdump.exists():
        # Try system llvm-objdump
        llvm_objdump = "llvm-objdump"

    result = subprocess.run(
        [str(llvm_objdump), "-d", "--mcpu=gfx950", str(co_file)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error disassembling:\n{result.stderr}")
        return None

    isa_file.write_text(result.stdout)
    print(f"✓ ISA extracted to: {isa_file}")

    return isa_file


def count_instructions(isa_file: Path) -> dict:
    """Count key instructions in an ISA file.

    Args:
        isa_file: Path to ISA file

    Returns:
        Dictionary with instruction counts
    """
    content = isa_file.read_text()

    counts = {
        "v_mfma": content.count("v_mfma"),
        "ds_read": content.count("ds_read"),
        "ds_write": content.count("ds_write"),
        "buffer_load": content.count("buffer_load"),
        "global_load": content.count("global_load"),
        "s_waitcnt": content.count("s_waitcnt"),
        "s_barrier": content.count("s_barrier"),
        "s_sched_barrier": content.count("s_sched_barrier"),
        "s_sched_group_barrier": content.count("s_sched_group_barrier"),
        "s_setprio": content.count("s_setprio"),
        "v_bitop3": content.count("v_bitop3"),  # XOR swizzle indicator
        "v_xor": content.count("v_xor"),
        "total_lines": len(content.split("\n")),
    }

    return counts


def print_summary(isa_file: Path, counts: dict):
    """Print a summary of ISA statistics.

    Args:
        isa_file: Path to ISA file
        counts: Dictionary with instruction counts
    """
    print(f"\n{'=' * 60}")
    print(f"ISA Summary: {isa_file.name}")
    print(f"{'=' * 60}")
    print(f"  v_mfma:                  {counts['v_mfma']:6d}")
    print(f"  ds_read:                 {counts['ds_read']:6d}")
    print(f"  ds_write:                {counts['ds_write']:6d}")
    print(f"  buffer_load:             {counts['buffer_load']:6d}")
    print(f"  global_load:             {counts['global_load']:6d}")
    print(f"  s_waitcnt:               {counts['s_waitcnt']:6d}")
    print(f"  s_barrier:               {counts['s_barrier']:6d}")
    print(f"  s_sched_barrier:         {counts['s_sched_barrier']:6d}")
    print(f"  s_sched_group_barrier:   {counts['s_sched_group_barrier']:6d}")
    print(f"  s_setprio:               {counts['s_setprio']:6d}")
    print(f"  v_bitop3 (XOR swizzle):  {counts['v_bitop3']:6d}")
    print(f"  v_xor:                   {counts['v_xor']:6d}")
    print(f"  Total lines:             {counts['total_lines']:6d}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ISA from FlyDSL or CK Tile kernels"
    )
    parser.add_argument("--flydsl", type=str, help="Path to FlyDSL kernel Python file")
    parser.add_argument("--cktile", type=str, help="Path to CK Tile .so file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--summary", action="store_true", help="Print instruction summary"
    )

    args = parser.parse_args()

    if not args.flydsl and not args.cktile:
        parser.error("Must specify at least one of --flydsl or --cktile")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract FlyDSL ISA
    if args.flydsl:
        flydsl_isa = extract_flydsl_isa(Path(args.flydsl), output_dir / "flydsl")
        if flydsl_isa and args.summary:
            counts = count_instructions(flydsl_isa)
            print_summary(flydsl_isa, counts)

    # Extract CK Tile ISA
    if args.cktile:
        cktile_isa = extract_cktile_isa(Path(args.cktile), output_dir / "cktile")
        if cktile_isa and args.summary:
            counts = count_instructions(cktile_isa)
            print_summary(cktile_isa, counts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
