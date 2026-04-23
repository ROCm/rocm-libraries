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

"""Adds a custom.config block to an external custom kernel .s file.

External kernels (non-Tensile) need a custom.config section in their
.amdgpu_metadata YAML to carry source/version/feature metadata AND the
Tensile interface (args, macrotile, threads, grid, ProblemType, etc.).

Auto-detected from the .s file (override with explicit flags):
    - origin:         parent directory name
    - wavefront-size: .wavefront_size from amdhsa.kernels metadata
    - threads:        .reqd_workgroup_size from amdhsa.kernels metadata

Usage:
    python -m Tensile.AddCustomConfig <file.s> \\
        --problem-type-file <problem_type.yaml> \\
        --args-file <args.yaml> \\
        --macrotile 256,256,64 \\
        --grid TilesX,TilesY,One \\
        --matrix-instruction 16,16,32,1

Examples:
    # Full config (origin, threads, wavefront-size auto-detected)
    python -m Tensile.AddCustomConfig CustomKernels/aiter/kernel.s \\
        --problem-type-file pt.yaml \\
        --args-file args.yaml \\
        --macrotile 256,256,64 \\
        --grid TilesX,TilesY,One \\
        --matrix-instruction 16,16,32,1

    # Provenance-only (no build config -- kernel won't be usable
    # through CustomKernels: list path until interface is added)
    python -m Tensile.AddCustomConfig CustomKernels/aiter/kernel.s

    # Preview without modifying the file
    python -m Tensile.AddCustomConfig kernel.s --dry-run
"""

import argparse
import json
import os
import sys


FEATURE_FLAGS = [
    "SupportsUserArgs",
    "SupportsBias",
    "SupportsActivation",
    "SupportsScaleAlpha",
    "SupportsGSU",
]


def build_custom_config_yaml(origin, repository=None, version="1.0.0",
                              features=None, problem_type=None,
                              kernel_args=None, macrotile=None,
                              threads=None, grid=None,
                              matrix_instruction=None, wavefront_size=64,
                              kern_args_version=0):
    """Builds the custom.config YAML block string."""
    if features is None:
        features = {f: False for f in FEATURE_FLAGS}

    lines = ["custom.config:"]
    lines.append("  Source:")
    lines.append(f"    Origin: {origin}")
    if repository:
        lines.append(f"    Repository: {repository}")
    lines.append(f"  Version: {version}")
    lines.append("  Features:")
    for flag in FEATURE_FLAGS:
        val = str(features.get(flag, False)).lower()
        lines.append(f"    {flag}: {val}")

    lines.append("  InternalSupportParams:")
    lines.append(f"    KernArgsVersion: {kern_args_version}")

    if problem_type:
        lines.append("  ProblemType:")
        for k, v in problem_type.items():
            lines.append(f"    {k}: {v}")

    if kernel_args is not None and macrotile is not None:
        lines.append("  CustomKernel:")
        lines.append(f"    args: {json.dumps(kernel_args)}")
        lines.append(f"    macrotile: {list(macrotile)}")
        if threads:
            lines.append(f"    threads: {list(threads)}")
        if grid:
            lines.append(f"    grid: {list(grid)}")

    if matrix_instruction:
        lines.append(f"  MatrixInstruction: {list(matrix_instruction)}")
        if len(matrix_instruction) >= 4 and macrotile and threads:
            lines.append("  EnableMatrixInstruction: True")
            num_threads = threads[0] * threads[1] * threads[2]
            num_waves = max(1, num_threads // wavefront_size)
            wgM = int(num_waves ** 0.5)
            while wgM > 0 and num_waves % wgM != 0:
                wgM -= 1
            wgM = max(1, wgM)
            wgN = num_waves // wgM
            mi_wave_tile = [max(1, macrotile[0] // (matrix_instruction[0] * wgM)),
                            max(1, macrotile[1] // (matrix_instruction[1] * wgN))]
            lines.append(f"  MIWaveTile: {mi_wave_tile}")

    lines.append(f"  WavefrontSize: {wavefront_size}")

    return "\n".join(lines)


def has_custom_config(filepath):
    """Checks whether the file already has a custom.config in .amdgpu_metadata."""
    in_metadata = False
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped == ".amdgpu_metadata":
                in_metadata = True
                continue
            if in_metadata and stripped == "...":
                break
            if in_metadata and stripped.startswith("custom.config"):
                return True
    return False


def inject_custom_config(filepath, config_yaml, dry_run=False):
    """Injects the custom.config block after the --- line in .amdgpu_metadata."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    in_metadata = False
    insert_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == ".amdgpu_metadata":
            in_metadata = True
            continue
        if in_metadata and stripped == "---":
            insert_idx = i + 1
            break

    if insert_idx is None:
        print("ERROR: No .amdgpu_metadata / --- section found in the file.",
              file=sys.stderr)
        print("The file must have an .amdgpu_metadata section to inject into.",
              file=sys.stderr)
        return False

    config_lines = [l + "\n" for l in config_yaml.split("\n")]

    if dry_run:
        print("--- custom.config block that would be inserted ---")
        print(config_yaml)
        print(f"--- at line {insert_idx + 1} of {filepath} ---")
        return True

    new_lines = lines[:insert_idx] + config_lines + lines[insert_idx:]

    with open(filepath, "w") as f:
        f.writelines(new_lines)

    print(f"Injected custom.config into {filepath} at line {insert_idx + 1}")
    return True


def _parse_int_list(s):
    return [int(x.strip()) for x in s.split(",")]


def _parse_string_list(s):
    return [x.strip() for x in s.split(",")]


def _load_yaml_file(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _detect_from_metadata(filepath):
    """Auto-detect origin, wavefront_size, and threads from the .s file."""
    import re
    import yaml

    detected = {}

    detected["origin"] = os.path.basename(os.path.dirname(os.path.abspath(filepath)))

    in_metadata = False
    yaml_lines = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "---":
                in_metadata = True
                continue
            if stripped == "...":
                break
            if in_metadata:
                yaml_lines.append(line)

    if not yaml_lines:
        return detected

    try:
        metadata = yaml.safe_load("\n".join(yaml_lines))
    except Exception:
        return detected

    if not metadata:
        return detected

    kernels = metadata.get("amdhsa.kernels", [])
    if kernels:
        k = kernels[0]
        wgs = k.get(".reqd_workgroup_size")
        if wgs and isinstance(wgs, list):
            detected["threads"] = wgs
        wf = k.get(".wavefront_size")
        if wf:
            detected["wavefront_size"] = int(wf)

    return detected


def main():
    parser = argparse.ArgumentParser(
        description="Add custom.config metadata to an external custom kernel .s file",
        epilog="Auto-detected from the .s file: origin (parent directory), "
               "wavefront-size, threads (.reqd_workgroup_size). "
               "Override any auto-detected value by passing the flag explicitly."
    )
    parser.add_argument("file", help="Path to the .s assembly file")
    parser.add_argument("--origin", default=None,
                        help="Source origin name (default: auto-detect from parent directory)")
    parser.add_argument("--repository", default=None,
                        help="Source repository URL")
    parser.add_argument("--version", default="1.0.0",
                        help="Kernel version (default: 1.0.0)")

    feat_group = parser.add_argument_group("Feature flags")
    feat_group.add_argument("--supports-user-args", action="store_true")
    feat_group.add_argument("--supports-bias", action="store_true")
    feat_group.add_argument("--supports-activation", action="store_true")
    feat_group.add_argument("--supports-scale-alpha", action="store_true")
    feat_group.add_argument("--supports-gsu", action="store_true")

    iface_group = parser.add_argument_group("Tensile interface (for CustomKernels: list path)")
    iface_group.add_argument("--problem-type-file", default=None,
                             help="YAML file with ProblemType dict (e.g. {OperationType: GEMM, DataType: b, ...})")
    iface_group.add_argument("--args-file", default=None,
                             help="YAML file with kernel args list")
    iface_group.add_argument("--macrotile", default=None,
                             help="Macro tile dimensions, comma-separated (e.g. 256,256,64)")
    iface_group.add_argument("--threads", default=None,
                             help="Thread dimensions, comma-separated (default: auto-detect from .reqd_workgroup_size)")
    iface_group.add_argument("--grid", default=None,
                             help="Grid mapping, comma-separated (e.g. TilesX,TilesY,One)")
    iface_group.add_argument("--matrix-instruction", default=None,
                             help="Matrix instruction, comma-separated (e.g. 16,16,32,1)")
    iface_group.add_argument("--wavefront-size", type=int, default=None,
                             help="Wavefront size (default: auto-detect from .wavefront_size)")
    iface_group.add_argument("--kern-args-version", type=int, default=0,
                             help="KernArgsVersion for InternalSupportParams (default: 0)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be injected without modifying the file")

    args = parser.parse_args()

    filepath = os.path.abspath(args.file)
    if not os.path.isfile(filepath):
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    if not filepath.endswith(".s"):
        print(f"WARNING: {filepath} does not end with .s", file=sys.stderr)

    if has_custom_config(filepath):
        print(f"ERROR: {filepath} already has a custom.config block.", file=sys.stderr)
        print("Remove the existing block first if you want to regenerate it.",
              file=sys.stderr)
        sys.exit(1)

    detected = _detect_from_metadata(filepath)

    origin = args.origin or detected.get("origin")
    if not origin:
        print("ERROR: Could not detect origin. Pass --origin explicitly.", file=sys.stderr)
        sys.exit(1)

    wavefront_size = args.wavefront_size or detected.get("wavefront_size", 64)
    threads = _parse_int_list(args.threads) if args.threads else detected.get("threads")

    auto_info = []
    if not args.origin and "origin" in detected:
        auto_info.append(f"origin={detected['origin']}")
    if args.wavefront_size is None and "wavefront_size" in detected:
        auto_info.append(f"wavefront_size={detected['wavefront_size']}")
    if not args.threads and "threads" in detected:
        auto_info.append(f"threads={detected['threads']}")
    if auto_info:
        print(f"Auto-detected: {', '.join(auto_info)}")

    features = {
        "SupportsUserArgs": args.supports_user_args,
        "SupportsBias": args.supports_bias,
        "SupportsActivation": args.supports_activation,
        "SupportsScaleAlpha": args.supports_scale_alpha,
        "SupportsGSU": args.supports_gsu,
    }

    problem_type = None
    if args.problem_type_file:
        problem_type = _load_yaml_file(args.problem_type_file)

    kernel_args = None
    if args.args_file:
        kernel_args = _load_yaml_file(args.args_file)

    macrotile = _parse_int_list(args.macrotile) if args.macrotile else None
    grid = _parse_string_list(args.grid) if args.grid else None
    matrix_instruction = _parse_int_list(args.matrix_instruction) if args.matrix_instruction else None

    config_yaml = build_custom_config_yaml(
        origin=origin,
        repository=args.repository,
        version=args.version,
        features=features,
        problem_type=problem_type,
        kernel_args=kernel_args,
        macrotile=macrotile,
        threads=threads,
        grid=grid,
        matrix_instruction=matrix_instruction,
        wavefront_size=wavefront_size,
        kern_args_version=args.kern_args_version,
    )

    if not inject_custom_config(filepath, config_yaml, dry_run=args.dry_run):
        sys.exit(1)


if __name__ == "__main__":
    main()
