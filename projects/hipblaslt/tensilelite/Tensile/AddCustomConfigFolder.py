################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Batch-add custom.config metadata to all .s files in a folder.

Each assembly file is matched to its CustomKernel entry in the YAML by
basename (without the .s suffix).  The YAML may declare many kernels across
multiple BenchmarkProblemSizeGroups; see AddCustomConfig._parse_tensile_yaml.

Usage:
    python -m Tensile.AddCustomConfigFolder <folder> --yaml <test.yaml>

Examples:
    python -m Tensile.AddCustomConfigFolder Tensile/CustomKernels/hipkittens \\
        --yaml Tensile/Tests/custom/custom_hipkittens.yaml

    python -m Tensile.AddCustomConfigFolder Tensile/CustomKernels/hipkittens \\
        --yaml Tensile/Tests/custom/custom_hipkittens.yaml --dry-run

    python -m Tensile.AddCustomConfigFolder Tensile/CustomKernels/hipkittens \\
        --yaml Tensile/Tests/custom/custom_hipkittens.yaml --skip-existing
"""

import argparse
import os
import sys

from Tensile.AddCustomConfig import (
    list_custom_kernels_in_yaml,
    process_asm_file,
)
from Tensile.CustomKernels import iterCustomKernelFiles


def iter_asm_files(folder, recursive=True):
    """Yield .s assembly files under *folder*."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise RuntimeError(f"Not a directory: {folder}")

    if recursive:
        yield from iterCustomKernelFiles(folder)
        return

    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".s"):
            yield os.path.join(folder, fname)


def process_folder(
    folder,
    yaml_path,
    origin=None,
    repository=None,
    version="1.0.0",
    dry_run=False,
    skip_existing=False,
    recursive=True,
):
    """Process every .s file under *folder*.

    Returns:
        dict with keys updated, skipped, failed (list of messages),
        and unmatched_yaml (kernel names in YAML with no .s file).
    """
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.isfile(yaml_path):
        raise RuntimeError(f"YAML not found: {yaml_path}")

    yaml_kernels = set(list_custom_kernels_in_yaml(yaml_path))
    asm_paths = list(iter_asm_files(folder, recursive=recursive))
    asm_kernels = {os.path.basename(p)[:-2] for p in asm_paths}

    results = {
        "updated": [],
        "skipped": [],
        "failed": [],
        "unmatched_yaml": sorted(yaml_kernels - asm_kernels),
        "unmatched_asm": sorted(asm_kernels - yaml_kernels),
    }

    for filepath in asm_paths:
        try:
            status = process_asm_file(
                filepath,
                yaml_path=yaml_path,
                origin=origin,
                repository=repository,
                version=version,
                dry_run=dry_run,
                skip_existing=skip_existing,
            )
        except RuntimeError as e:
            results["failed"].append(f"{filepath}: {e}")
            continue

        if status == "updated":
            results["updated"].append(filepath)
        elif status == "skipped":
            results["skipped"].append(filepath)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Add custom.config metadata to all .s files in a folder",
        epilog="Each .s file is matched to a CustomKernel entry in the YAML "
               "by basename. Origin defaults to the parent directory name of "
               "each file unless --origin is set."
    )
    parser.add_argument(
        "folder",
        help="Directory containing custom kernel .s files",
    )
    parser.add_argument(
        "--yaml",
        required=True,
        help="Tensile test YAML with ForkParameters for all kernels",
    )
    parser.add_argument(
        "--origin",
        default=None,
        help="Override origin for every file (default: each file's parent directory)",
    )
    parser.add_argument(
        "--repository",
        default=None,
        help="Source repository URL",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Kernel version (default: 1.0.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be injected without modifying files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip .s files that already contain a custom.config block",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only process .s files directly in the given folder",
    )

    args = parser.parse_args()

    try:
        results = process_folder(
            args.folder,
            args.yaml,
            origin=args.origin,
            repository=args.repository,
            version=args.version,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
            recursive=not args.no_recursive,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    action = "Would update" if args.dry_run else "Updated"
    print(f"\nSummary:")
    print(f"  {action}: {len(results['updated'])}")
    print(f"  Skipped:  {len(results['skipped'])}")
    print(f"  Failed:   {len(results['failed'])}")

    if results["failed"]:
        print("\nFailures:", file=sys.stderr)
        for msg in results["failed"]:
            print(f"  {msg}", file=sys.stderr)

    if results["unmatched_yaml"]:
        print("\nYAML kernels with no matching .s file:", file=sys.stderr)
        for name in results["unmatched_yaml"]:
            print(f"  {name}", file=sys.stderr)

    if results["unmatched_asm"]:
        print("\n.s files with no matching YAML entry:", file=sys.stderr)
        for name in results["unmatched_asm"]:
            print(f"  {name}", file=sys.stderr)

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
