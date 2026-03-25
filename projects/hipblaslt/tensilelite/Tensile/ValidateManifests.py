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

"""CI enforcement script for kernel manifest validation.

Validates that all custom kernel directories have a manifest.yaml and that
every kernel's ContentHash matches its .s file on disk.

Exit codes:
    0 - All kernels pass validation
    1 - One or more validation failures

Usage:
    python -m Tensile.ValidateManifests [--strict] [--custom-kernels-root DIR]

    --strict    Treat missing manifests as errors (Phase 4 enforcement).
                Without --strict, missing manifests produce warnings only.
"""

import argparse
import os
import sys

from Tensile.CustomKernels import (
    readManifest,
    computeContentHash,
    MANIFEST_FILENAME,
)


def validate_directory(directory, strict=False):
    """Validates all kernels in a directory against its manifest.

    Returns (errors, warnings) as lists of message strings.
    """
    errors = []
    warnings = []

    manifest = None
    try:
        manifest = readManifest(directory)
    except RuntimeError as e:
        errors.append(f"{directory}: {e}")
        return errors, warnings

    s_files = sorted(f for f in os.listdir(directory) if f.endswith(".s"))
    if not s_files:
        return errors, warnings

    if manifest is None:
        msg = f"{directory}: no {MANIFEST_FILENAME} found ({len(s_files)} kernel(s))"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)
        return errors, warnings

    kernels = manifest.get("Kernels", {})

    for fname in s_files:
        name = fname[:-2]
        filepath = os.path.join(directory, fname)

        if name not in kernels:
            msg = f"{directory}: kernel '{name}' not listed in manifest"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
            continue

        entry = kernels[name]
        expected_hash = entry.get("ContentHash")
        if not expected_hash:
            errors.append(f"{directory}: kernel '{name}' has no ContentHash in manifest")
            continue

        actual_hash = _compute_hash_for_file(filepath)
        if actual_hash != expected_hash:
            errors.append(
                f"{directory}: kernel '{name}' content hash mismatch "
                f"(manifest={expected_hash}, actual={actual_hash})"
            )

    for name in kernels:
        s_path = os.path.join(directory, name + ".s")
        if not os.path.isfile(s_path):
            warnings.append(f"{directory}: manifest lists '{name}' but no .s file found")

    return errors, warnings


def _compute_hash_for_file(filepath):
    import hashlib
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def validate_all(root, strict=False):
    """Validates all subdirectories under the custom kernels root.

    Returns (total_errors, total_warnings).
    """
    total_errors = []
    total_warnings = []

    for entry in sorted(os.listdir(root)):
        subdir = os.path.join(root, entry)
        if not os.path.isdir(subdir):
            continue

        errs, warns = validate_directory(subdir, strict=strict)
        total_errors.extend(errs)
        total_warnings.extend(warns)

    return total_errors, total_warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate custom kernel manifests for CI enforcement"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing manifests as errors (Phase 4 enforcement)",
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

    total_dirs = sum(
        1 for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and any(f.endswith(".s") for f in os.listdir(os.path.join(root, d)))
    )

    print(f"\nSummary: {total_dirs} directories, {len(errors)} error(s), {len(warnings)} warning(s)")

    if errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
