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

"""CI enforcement script for custom kernel metadata validation.

Validates that all custom kernel .s files contain a custom.config block
inside their .amdgpu_metadata YAML section.

Exit codes:
    0 - All kernels pass validation
    1 - One or more validation failures

Usage:
    python -m Tensile.ValidateMetadata [--strict] [--custom-kernels-root DIR]

    --strict    Treat missing or incomplete metadata as errors.
                Without --strict, validation failures produce warnings only.
"""

import argparse
import os
import sys

from Tensile.CustomKernels import iterCustomKernelFiles, validateCustomKernelMetadata


def validate_all(root, strict=False):
    """Validates all kernels under a directory for embedded metadata.

    With ``strict=True``, validation failures are returned as errors;
    otherwise they are returned as warnings.

    Returns (errors, warnings) as lists of message strings.
    """
    errors = []
    warnings = []

    for path in iterCustomKernelFiles(root):
        kernel_dir = os.path.dirname(path)
        name = os.path.basename(path)[:-2]

        valid, msg = validateCustomKernelMetadata(name, kernel_dir)
        if not valid:
            (errors if strict else warnings).append(f"{path}: {msg}")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate custom kernel embedded metadata for CI enforcement"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing metadata as errors",
    )
    parser.add_argument(
        "--custom-kernels-root",
        type=str,
        help="Root CustomKernels directory (default: auto-detect)",
    )
    args = parser.parse_args()

    if args.custom_kernels_root:
        root = args.custom_kernels_root
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CustomKernels")

    if not os.path.isdir(root):
        print(f"ERROR: CustomKernels directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    errors, warnings = validate_all(root, strict=args.strict)

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    total_kernels = sum(1 for _ in iterCustomKernelFiles(root))

    print(f"\nSummary: {total_kernels} kernel(s), {len(errors)} error(s), {len(warnings)} warning(s)")

    if errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
