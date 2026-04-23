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

The config is extracted directly from a Tensile test YAML that already
has the kernel's ForkParameters (CustomKernel, MatrixInstruction, etc.).

Auto-detected from the .s file (override with CLI flags):
    - origin:         parent directory name
    - wavefront-size: .wavefront_size from amdhsa.kernels metadata
    - threads:        .reqd_workgroup_size from amdhsa.kernels metadata

Usage:
    python -m Tensile.AddCustomConfig <file.s> --yaml <test.yaml>

Examples:
    # Extract config from test YAML and inject into .s file
    python -m Tensile.AddCustomConfig CustomKernels/aiter/kernel.s \\
        --yaml Tests/custom/custom_aiter_bf16.yaml

    # Provenance-only (no --yaml -- kernel won't be usable
    # through CustomKernels: list path until interface is added)
    python -m Tensile.AddCustomConfig CustomKernels/aiter/kernel.s

    # Preview without modifying the file
    python -m Tensile.AddCustomConfig kernel.s --yaml test.yaml --dry-run
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


def _parse_tensile_yaml(path, kernel_name=None):
    """Extract ProblemType, CustomKernel, and MatrixInstruction from a Tensile test YAML.

    Args:
        path: Path to the Tensile test YAML file.
        kernel_name: If provided, match this kernel name in the ForkParameters.
                     If None, use the first CustomKernel found.

    Returns:
        dict with ProblemType, CustomKernel, MatrixInstruction, WavefrontSize (if found).
    """
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)

    bp = data["BenchmarkProblems"][0]
    problem_type = bp[0]
    bench = bp[1]

    config = {"ProblemType": problem_type}

    fork_params = bench.get("ForkParameters", [])
    for entry in fork_params:
        if not isinstance(entry, dict):
            continue

        if "CustomKernel" in entry:
            for ck in entry["CustomKernel"]:
                if not isinstance(ck, dict) or "name" not in ck:
                    continue
                if kernel_name and ck["name"] != kernel_name:
                    continue
                config["CustomKernel"] = {
                    k: v for k, v in ck.items() if k != "name"
                }
                break

        if "MatrixInstruction" in entry:
            mi_list = entry["MatrixInstruction"]
            if mi_list and isinstance(mi_list[0], list):
                config["MatrixInstruction"] = mi_list[0][:4]

        if "WavefrontSize" in entry:
            wf_list = entry["WavefrontSize"]
            if wf_list:
                config["WavefrontSize"] = wf_list[0]

    return config


def build_custom_config_yaml(origin, config, repository=None, version="1.0.0"):
    """Builds the custom.config YAML block string.

    Args:
        origin: Source origin name (e.g. "aiter", "wave").
        config: Dict with ProblemType, CustomKernel, MatrixInstruction, etc.
                May be None for provenance-only injection.
        repository: Optional source repository URL.
        version: Kernel version string.
    """
    features = config.get("Features", {}) if config else {}
    for flag in FEATURE_FLAGS:
        features.setdefault(flag, False)

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

    isp = {}
    if config:
        isp = config.get("InternalSupportParams", {})
    isp.setdefault("KernArgsVersion", 0)
    lines.append("  InternalSupportParams:")
    for k, v in isp.items():
        lines.append(f"    {k}: {v}")

    if config and "ProblemType" in config:
        lines.append("  ProblemType:")
        for k, v in config["ProblemType"].items():
            lines.append(f"    {k}: {v}")

    if config and "CustomKernel" in config:
        ck = config["CustomKernel"]
        lines.append("  CustomKernel:")
        if "args" in ck:
            lines.append(f"    args: {json.dumps(ck['args'])}")
        if "macrotile" in ck:
            lines.append(f"    macrotile: {list(ck['macrotile'])}")
        if "threads" in ck:
            lines.append(f"    threads: {list(ck['threads'])}")
        if "grid" in ck:
            lines.append(f"    grid: {list(ck['grid'])}")

    if config and "MatrixInstruction" in config:
        mi = config["MatrixInstruction"]
        lines.append(f"  MatrixInstruction: {list(mi)}")

        macrotile = config.get("CustomKernel", {}).get("macrotile")
        threads = config.get("CustomKernel", {}).get("threads")
        wavefront_size = config.get("WavefrontSize", 64)
        if len(mi) >= 4 and macrotile and threads:
            lines.append("  EnableMatrixInstruction: True")
            num_threads = threads[0] * threads[1] * threads[2]
            num_waves = max(1, num_threads // wavefront_size)
            wgM = int(num_waves ** 0.5)
            while wgM > 0 and num_waves % wgM != 0:
                wgM -= 1
            wgM = max(1, wgM)
            wgN = num_waves // wgM
            mi_wave_tile = [max(1, macrotile[0] // (mi[0] * wgM)),
                            max(1, macrotile[1] // (mi[1] * wgN))]
            lines.append(f"  MIWaveTile: {mi_wave_tile}")

    wavefront_size = config.get("WavefrontSize", 64) if config else 64
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


def _detect_from_metadata(filepath):
    """Auto-detect origin, wavefront_size, and threads from the .s file."""
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
               "Override any auto-detected value with CLI flags."
    )
    parser.add_argument("file", help="Path to the .s assembly file")
    parser.add_argument("--yaml", default=None,
                        help="Tensile test YAML with ForkParameters "
                             "(ProblemType, CustomKernel, MatrixInstruction extracted automatically)")
    parser.add_argument("--origin", default=None,
                        help="Source origin name (default: auto-detect from parent directory)")
    parser.add_argument("--repository", default=None,
                        help="Source repository URL")
    parser.add_argument("--version", default="1.0.0",
                        help="Kernel version (default: 1.0.0)")
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

    config = None
    if args.yaml:
        kernel_name = os.path.basename(filepath)[:-2]
        config = _parse_tensile_yaml(args.yaml, kernel_name)

    if config:
        ck = config.get("CustomKernel", {})
        if "threads" not in ck and "threads" in detected:
            ck["threads"] = detected["threads"]
            config["CustomKernel"] = ck
        if "WavefrontSize" not in config and "wavefront_size" in detected:
            config["WavefrontSize"] = detected["wavefront_size"]

    auto_info = []
    if not args.origin and "origin" in detected:
        auto_info.append(f"origin={detected['origin']}")
    if "wavefront_size" in detected:
        auto_info.append(f"wavefront_size={detected['wavefront_size']}")
    if "threads" in detected:
        auto_info.append(f"threads={detected['threads']}")
    if auto_info:
        print(f"Auto-detected: {', '.join(auto_info)}")

    config_yaml = build_custom_config_yaml(
        origin=origin,
        config=config,
        repository=args.repository,
        version=args.version,
    )

    if not inject_custom_config(filepath, config_yaml, dry_run=args.dry_run):
        sys.exit(1)


if __name__ == "__main__":
    main()
