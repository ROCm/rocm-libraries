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
import os
import sys

from Tensile.Common.Utilities import deriveWaveParams


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
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse Tensile YAML '{path}': {e}") from e

    try:
        bp = data["BenchmarkProblems"][0]
        problem_type = bp[0]
        bench = bp[1]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Tensile YAML '{path}' does not contain BenchmarkProblems[0] "
            "with ProblemType and ForkParameters"
        ) from e

    config = {"ProblemType": problem_type}

    fork_params = bench.get("ForkParameters", [])
    available_kernels = []
    for entry in fork_params:
        if not isinstance(entry, dict):
            continue

        if "CustomKernel" in entry:
            for ck in entry["CustomKernel"]:
                if not isinstance(ck, dict) or "name" not in ck:
                    continue
                available_kernels.append(ck["name"])
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

    if kernel_name and "CustomKernel" not in config:
        raise RuntimeError(
            f"Kernel '{kernel_name}' not found in {path}. "
            f"Available: {available_kernels or 'none'}"
        )

    if "CustomKernel" not in config:
        raise RuntimeError(f"No CustomKernel entry found in {path}")

    return config


def _fmt_yaml_scalar(value):
    """Format a scalar for the compact YAML style used in custom.config."""
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def _fmt_yaml_inline(value):
    """Format simple YAML values inline without quoting strings."""
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_yaml_inline(v) for v in value) + "]"
    if isinstance(value, dict):
        pairs = ", ".join(
            f"{k}: {_fmt_yaml_inline(v)}" for k, v in value.items()
        )
        return "{ " + pairs + " }"
    return _fmt_yaml_scalar(value)


def _fmt_yaml_args(args, indent=4):
    """Format an args list as multi-line YAML flow mappings.

    Produces output like:
        args: [ { type: address, semantic: AddressD, padding: 8 },
                { type: address, semantic: AddressA, padding: 8 } ]
    """
    if not args:
        return "args: []"
    prefix = " " * indent + "args: "
    continuation = " " * len(prefix)
    formatted = []
    for arg in args:
        pairs = ", ".join(
            f"{k}: {_fmt_yaml_inline(v)}" for k, v in arg.items()
        )
        formatted.append("{ " + pairs + " }")
    if len(formatted) == 1:
        return prefix + "[ " + formatted[0] + " ]"
    first = prefix + "[ " + formatted[0] + ","
    middle = [continuation + "  " + f + "," for f in formatted[1:-1]]
    last = continuation + "  " + formatted[-1] + " ]"
    return "\n".join([first] + middle + [last])


def build_custom_config_yaml(origin, config, repository=None, version="1.0.0"):
    """Builds the custom.config YAML block string.

    Args:
        origin: Source origin name (e.g. "aiter", "wave").
        config: Dict with ProblemType, CustomKernel, MatrixInstruction, etc.
                May be None for provenance-only injection. Treated as read-only.
        repository: Optional source repository URL.
        version: Kernel version string.
    """
    # Take local copies so we never mutate caller-supplied dicts.
    features = dict((config or {}).get("Features", {}))
    for flag in FEATURE_FLAGS:
        features.setdefault(flag, False)

    isp = dict((config or {}).get("InternalSupportParams", {}))
    isp.setdefault("KernArgsVersion", 0)

    wavefront_size = (config or {}).get("WavefrontSize", 64)

    lines = ["custom.config:"]
    lines.append("  Source:")
    lines.append(f"    Origin: {origin}")
    if repository:
        lines.append(f"    Repository: {repository}")
    lines.append(f"  Version: {version}")
    lines.append("  Features:")
    for flag in FEATURE_FLAGS:
        lines.append(f"    {flag}: {str(features[flag]).lower()}")

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
        for key, value in ck.items():
            if key == "args":
                lines.append(_fmt_yaml_args(value))
            else:
                lines.append(f"    {key}: {_fmt_yaml_inline(value)}")

    if config and "MatrixInstruction" in config:
        mi = config["MatrixInstruction"]
        lines.append(f"  MatrixInstruction: {_fmt_yaml_inline(mi)}")

        macrotile = config.get("CustomKernel", {}).get("macrotile")
        threads = config.get("CustomKernel", {}).get("threads")
        if len(mi) >= 4 and macrotile and threads:
            lines.append("  EnableMatrixInstruction: True")
            num_threads = threads[0] * threads[1] * threads[2]
            _, mi_wave_tile = deriveWaveParams(mi, num_threads, macrotile, wavefront_size)
            lines.append(f"  MIWaveTile: {_fmt_yaml_inline(mi_wave_tile)}")

    lines.append(f"  WavefrontSize: {wavefront_size}")

    return "\n".join(lines)


def _read_asm_file(filepath):
    """Reads an .s file once, returning all info needed by the pipeline.

    Returns a dict with:
        lines:              list of raw line strings (with newlines)
        has_custom_config:  bool
        insert_idx:         line index after '---' in .amdgpu_metadata (or None)
        origin:             parent directory name
        detected:           dict with auto-detected threads and wavefront_size
    """
    import yaml

    with open(filepath, "r") as f:
        lines = f.readlines()

    result = {
        "lines": lines,
        "has_custom_config": False,
        "insert_idx": None,
        "origin": os.path.basename(os.path.dirname(os.path.abspath(filepath))),
        "detected": {},
    }

    in_metadata = False
    in_yaml = False
    yaml_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == ".amdgpu_metadata":
            in_metadata = True
            continue
        if in_metadata and stripped == "---":
            in_yaml = True
            result["insert_idx"] = i + 1
            continue
        if in_metadata and stripped == "...":
            break
        if in_metadata and stripped.startswith("custom.config"):
            result["has_custom_config"] = True
        if in_yaml:
            yaml_lines.append(line)

    if not yaml_lines:
        return result

    try:
        metadata = yaml.safe_load("\n".join(yaml_lines))
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse .amdgpu_metadata YAML in {filepath}: {e}") from e

    if not metadata:
        return result

    kernels = metadata.get("amdhsa.kernels", [])
    if kernels:
        k = kernels[0]
        wgs = k.get(".reqd_workgroup_size")
        if wgs and isinstance(wgs, list):
            result["detected"]["threads"] = wgs
        wf = k.get(".wavefront_size")
        if wf:
            result["detected"]["wavefront_size"] = int(wf)

    return result


def inject_custom_config(file_info, filepath, config_yaml, dry_run=False):
    """Injects the custom.config block using pre-read file data."""
    insert_idx = file_info["insert_idx"]

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

    lines = file_info["lines"]
    new_lines = lines[:insert_idx] + config_lines + lines[insert_idx:]

    with open(filepath, "w") as f:
        f.writelines(new_lines)

    print(f"Injected custom.config into {filepath} at line {insert_idx + 1}")
    return True


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

    try:
        file_info = _read_asm_file(filepath)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if file_info["has_custom_config"]:
        print(f"ERROR: {filepath} already has a custom.config block.", file=sys.stderr)
        print("Remove the existing block first if you want to regenerate it.",
              file=sys.stderr)
        sys.exit(1)

    detected = file_info["detected"]
    origin = args.origin or file_info["origin"]
    if not origin:
        print("ERROR: Could not detect origin. Pass --origin explicitly.", file=sys.stderr)
        sys.exit(1)

    config = None
    if args.yaml:
        kernel_name = os.path.basename(filepath)[:-2]
        try:
            config = _parse_tensile_yaml(args.yaml, kernel_name)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    if config:
        ck = config.get("CustomKernel", {})
        if "threads" not in ck and "threads" in detected:
            ck["threads"] = detected["threads"]
            config["CustomKernel"] = ck
        if "WavefrontSize" not in config and "wavefront_size" in detected:
            config["WavefrontSize"] = detected["wavefront_size"]

    auto_info = []
    if not args.origin and file_info["origin"]:
        auto_info.append(f"origin={file_info['origin']}")
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

    if not inject_custom_config(file_info, filepath, config_yaml, dry_run=args.dry_run):
        sys.exit(1)


if __name__ == "__main__":
    main()
