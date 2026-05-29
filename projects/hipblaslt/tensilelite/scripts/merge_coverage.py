#!/usr/bin/env python3
# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Merge LLVM C++ coverage with Python coverage into unified reports."""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_llvm_tools(rocm_path: str = "/opt/rocm"):
    """Find llvm-profdata and llvm-cov."""
    llvm_bin = Path(rocm_path) / "llvm" / "bin"
    profdata = llvm_bin / "llvm-profdata"
    cov = llvm_bin / "llvm-cov"

    if not profdata.exists():
        profdata = shutil.which("llvm-profdata")
    if not cov.exists():
        cov = shutil.which("llvm-cov")

    if not profdata or not cov:
        sys.exit("Error: llvm-profdata and llvm-cov required. Install ROCm or LLVM.")

    return str(profdata), str(cov)


def merge_profraw(profdata: str, profraw_dir: Path, output: Path):
    """Merge .profraw files into .profdata."""
    profraw_files = list(profraw_dir.glob("*.profraw"))
    if not profraw_files:
        print(f"Warning: No .profraw files found in {profraw_dir}")
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [profdata, "merge", "-sparse", "-o", str(output)] + [str(f) for f in profraw_files]
    subprocess.run(cmd, check=True)
    return output


def export_lcov(llvm_cov: str, profdata: Path, objects: list, output: Path, source_dir: Path):
    """Export LLVM coverage to LCOV format."""
    cmd = [
        llvm_cov, "export",
        "-format=lcov",
        f"-instr-profile={profdata}",
    ]
    for obj in objects:
        cmd.extend(["-object", str(obj)])

    # Filter out build directories and external paths
    cmd.append("--ignore-filename-regex=.*/(build|_deps|site-packages)/.*")
    cmd.append("--ignore-filename-regex=.*/shared/origami/.*")
    cmd.append("--ignore-filename-regex=.*/shared/stinkytofu/.*")
    cmd.append("--ignore-filename-regex=.*\\.hpp$")

    with open(output, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)

    return output


def convert_python_coverage(coverage_xml: Path, output: Path):
    """Convert Python coverage.xml to LCOV using coverage tool."""
    # coverage can export to lcov directly
    cmd = ["coverage", "lcov", "-o", str(output)]
    subprocess.run(cmd, check=True, cwd=coverage_xml.parent)
    return output


def merge_lcov(lcov_files: list, output: Path):
    """Merge multiple LCOV files."""
    if not shutil.which("lcov"):
        sys.exit("Error: lcov required. Install with: apt install lcov")

    cmd = ["lcov"]
    for f in lcov_files:
        cmd.extend(["-a", str(f)])
    cmd.extend(["-o", str(output)])

    subprocess.run(cmd, check=True)
    return output


def filter_lcov(lcov_file: Path, output: Path, ignore_patterns: list):
    """Filter LCOV file to remove unwanted paths."""
    if not shutil.which("lcov"):
        sys.exit("Error: lcov required. Install with: apt install lcov")

    cmd = ["lcov", "--remove", str(lcov_file)]
    for pattern in ignore_patterns:
        cmd.append(pattern)
    cmd.extend(["-o", str(output)])

    subprocess.run(cmd, check=True)
    return output


def generate_html(lcov_file: Path, output_dir: Path, source_dir: Path, title: str = "TensileLite Coverage"):
    """Generate HTML report from LCOV."""
    if not shutil.which("genhtml"):
        sys.exit("Error: genhtml required. Install with: apt install lcov")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "genhtml", str(lcov_file),
        "-o", str(output_dir),
        "--title", title,
        "--legend",
        "--branch-coverage",
        "--ignore-errors", "source",  # Ignore missing source files
        "-p", str(source_dir),  # Strip this prefix from paths
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: genhtml completed with errors:\n{result.stderr}")
        # Try again without branch coverage if it fails
        cmd_simple = [
            "genhtml", str(lcov_file),
            "-o", str(output_dir),
            "--title", title,
            "--ignore-errors", "source",
        ]
        subprocess.run(cmd_simple, check=False)


def convert_to_cobertura(lcov_file: Path, output: Path):
    """Convert LCOV to Cobertura XML for CI tools."""
    try:
        import lcov_cobertura  # noqa: F401
    except ImportError:
        # Try running as command
        if shutil.which("lcov_cobertura"):
            subprocess.run(["lcov_cobertura", str(lcov_file), "-o", str(output)], check=True)
        else:
            print("Warning: lcov_cobertura not available, skipping Cobertura XML")
            return None
    else:
        subprocess.run(["lcov_cobertura", str(lcov_file), "-o", str(output)], check=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Merge C++ and Python coverage")
    parser.add_argument("--profraw-dir", type=Path, required=True,
                        help="Directory containing .profraw files")
    parser.add_argument("--cpp-objects", nargs="*", default=[],
                        help="C++ object files/libraries for coverage")
    parser.add_argument("--python-coverage-dir", type=Path, default=Path("."),
                        help="Directory with .coverage file")
    parser.add_argument("--source-dir", type=Path, default=Path("."),
                        help="Source directory for filtering")
    parser.add_argument("--output-dir", type=Path, default=Path("coverage_combined"),
                        help="Output directory for reports")
    parser.add_argument("--rocm-path", default="/opt/rocm",
                        help="Path to ROCm installation")
    args = parser.parse_args()

    profdata_tool, cov_tool = find_llvm_tools(args.rocm_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge profraw files
    print("Merging C++ profraw files...")
    profdata = merge_profraw(
        profdata_tool,
        args.profraw_dir,
        args.output_dir / "cpp.profdata"
    )

    lcov_files = []

    # Step 2: Export C++ coverage to LCOV
    if profdata:
        print("Exporting C++ coverage to LCOV...")
        cpp_objects = []
        for pattern in args.cpp_objects:
            cpp_objects.extend(glob.glob(pattern))

        if cpp_objects:
            cpp_lcov = export_lcov(
                cov_tool,
                profdata,
                cpp_objects,
                args.output_dir / "cpp.lcov",
                args.source_dir
            )
            lcov_files.append(cpp_lcov)

    # Step 3: Convert Python coverage to LCOV
    print("Converting Python coverage to LCOV...")
    python_lcov = args.output_dir / "python.lcov"
    try:
        convert_python_coverage(args.python_coverage_dir, python_lcov)
        lcov_files.append(python_lcov)
    except subprocess.CalledProcessError:
        print("Warning: Failed to convert Python coverage")

    if not lcov_files:
        sys.exit("Error: No coverage data to merge")

    # Step 4: Merge LCOV files
    print("Merging coverage data...")
    merged_lcov = args.output_dir / "merged.lcov"
    if len(lcov_files) == 1:
        shutil.copy(lcov_files[0], merged_lcov)
    else:
        merge_lcov(lcov_files, merged_lcov)

    # Step 5: Filter out unwanted paths
    print("Filtering coverage data...")
    combined_lcov = args.output_dir / "combined.lcov"
    ignore_patterns = [
        "*/shared/origami/*",
        "*/stinkytofu/*",
        "*.hpp",
        "*.inc",
    ]
    filter_lcov(merged_lcov, combined_lcov, ignore_patterns)

    # Step 6: Generate reports
    print("Generating HTML report...")
    generate_html(combined_lcov, args.output_dir / "html", args.source_dir)

    print("Generating Cobertura XML...")
    convert_to_cobertura(combined_lcov, args.output_dir / "coverage.xml")

    print(f"\nCoverage reports generated in {args.output_dir}/")
    print(f"  HTML:      {args.output_dir}/html/index.html")
    print(f"  LCOV:      {args.output_dir}/combined.lcov")
    print(f"  Cobertura: {args.output_dir}/coverage.xml")


if __name__ == "__main__":
    main()
