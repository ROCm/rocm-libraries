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

"""Generates manifest.yaml files for custom kernel directories.

Usage:
    python -m Tensile.GenerateManifest [--directory DIR] [--origin ORIGIN]

If no arguments are given, generates manifests for all subdirectories under
CustomKernels/ that contain .s files.
"""

import argparse
import hashlib
import os
import re
import sys
import yaml

MANIFEST_FILENAME = "manifest.yaml"
MANIFEST_VERSION = 1

_ORIGIN_MAP = {
    "tensile": "tensile",
    "aiter":   "aiter",
    "ck":      "composable_kernel",
    "rocroller": "rocroller",
}


def _compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _extract_isa(filepath):
    """Extracts the target ISA from a .s file.

    Checks for .amdgcn_target directive first, then falls back to
    header comment patterns like '-mcpu=gfxNNN'.
    """
    with open(filepath) as f:
        for line in f:
            m = re.search(r'\.amdgcn_target\s+"amdgcn-amd-amdhsa--(\w+)"', line)
            if m:
                return m.group(1)
            m = re.search(r'-mcpu=(\w+)', line)
            if m:
                return m.group(1)
            if line.startswith(".text") or line.startswith(".section"):
                break
    return None


def _extract_custom_config(filepath):
    """Extracts the custom.config YAML block from a Tensile-generated .s file."""
    contents = ""
    in_metadata = False
    with open(filepath) as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped == "---":
                in_metadata = True
                continue
            if stripped == "...":
                in_metadata = False
                continue
            if in_metadata:
                contents += line

    if not contents:
        return None

    try:
        parsed = yaml.safe_load(contents)
        if isinstance(parsed, dict) and "custom.config" in parsed:
            return parsed["custom.config"]
    except yaml.YAMLError:
        pass
    return None


def _detect_features_from_name(name):
    """Infers feature flags from kernel naming conventions."""
    features = {}
    features["SupportsUserArgs"] = "UserArgs" in name
    features["SupportsBias"] = "Bias" in name or "_B_BIAS_" in name or "BiasSB" in name
    features["SupportsActivation"] = False
    features["SupportsScaleAlpha"] = "SAV" in name
    features["SupportsGSU"] = name.startswith("CustomGSUs_") or "GSUM" in name
    return features


def generate_tensile_manifest(directory):
    """Generates manifest.yaml for a directory of Tensile-generated kernels."""
    s_files = sorted(f for f in os.listdir(directory) if f.endswith(".s"))
    if not s_files:
        return None

    kernels = {}
    common_isa = None

    for fname in s_files:
        filepath = os.path.join(directory, fname)
        name = fname[:-2]
        content_hash = _compute_sha256(filepath)
        isa = _extract_isa(filepath)

        if common_isa is None:
            common_isa = isa
        elif common_isa != isa:
            common_isa = ""

        entry = {
            "Version": "1.0.0",
            "ContentHash": content_hash,
        }

        if isa and common_isa != isa:
            entry["Target"] = {"ISA": isa}

        config = _extract_custom_config(filepath)
        if config:
            schedule_params = {}
            if "InternalSupportParams" in config:
                isp = config["InternalSupportParams"]
                if "KernArgsVersion" in isp:
                    schedule_params["KernArgsVersion"] = isp["KernArgsVersion"]
            if schedule_params:
                entry["ScheduleParams"] = schedule_params

        features = _detect_features_from_name(name)
        entry["Features"] = features
        kernels[name] = entry

    manifest = {"MetaVersion": MANIFEST_VERSION}

    manifest["Source"] = {"Origin": "tensile"}
    if common_isa:
        manifest["Target"] = {"ISA": common_isa}

    manifest["Kernels"] = kernels
    return manifest


def generate_external_manifest(directory, origin):
    """Generates manifest.yaml for a directory of external kernels (aiter, ck, rocroller)."""
    s_files = sorted(f for f in os.listdir(directory) if f.endswith(".s"))
    if not s_files:
        return None

    kernels = {}
    common_isa = None

    for fname in s_files:
        filepath = os.path.join(directory, fname)
        name = fname[:-2]
        content_hash = _compute_sha256(filepath)
        isa = _extract_isa(filepath)

        if common_isa is None:
            common_isa = isa
        elif common_isa != isa:
            common_isa = ""

        entry = {
            "Version": "1.0.0",
            "ContentHash": content_hash,
        }

        if isa and common_isa != isa:
            entry["Target"] = {"ISA": isa}

        entry["Features"] = {
            "SupportsUserArgs": False,
            "SupportsBias": False,
            "SupportsActivation": False,
            "SupportsScaleAlpha": False,
            "SupportsGSU": False,
        }

        kernels[name] = entry

    manifest = {"MetaVersion": MANIFEST_VERSION}

    source = {"Origin": _ORIGIN_MAP.get(origin, origin)}
    if origin == "aiter":
        source["Repository"] = "https://github.com/ROCm/aiter"
    elif origin == "ck":
        source["Repository"] = "https://github.com/ROCm/composable_kernel"
    elif origin == "rocroller":
        source["Repository"] = "https://github.com/ROCm/rocroller"
    manifest["Source"] = source

    if common_isa:
        manifest["Target"] = {"ISA": common_isa}

    manifest["Kernels"] = kernels
    return manifest


def _write_manifest(directory, manifest):
    path = os.path.join(directory, MANIFEST_FILENAME)
    with open(path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, width=120)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate manifest.yaml for custom kernel directories")
    parser.add_argument("--directory", type=str, help="Specific directory to process")
    parser.add_argument("--origin", type=str, help="Origin name (tensile, aiter, ck, rocroller)")
    parser.add_argument("--custom-kernels-root", type=str,
                        help="Root CustomKernels directory (default: auto-detect)")
    args = parser.parse_args()

    if args.custom_kernels_root:
        root = args.custom_kernels_root
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CustomKernels")

    if args.directory:
        directory = os.path.abspath(args.directory)
        origin = args.origin or os.path.basename(directory)
        if origin == "tensile":
            manifest = generate_tensile_manifest(directory)
        else:
            manifest = generate_external_manifest(directory, origin)
        if manifest:
            path = _write_manifest(directory, manifest)
            print(f"Generated {path} ({len(manifest['Kernels'])} kernels)")
    else:
        for entry in sorted(os.listdir(root)):
            subdir = os.path.join(root, entry)
            if not os.path.isdir(subdir):
                continue

            s_files = [f for f in os.listdir(subdir) if f.endswith(".s")]
            if not s_files:
                continue

            if entry == "tensile":
                manifest = generate_tensile_manifest(subdir)
            else:
                manifest = generate_external_manifest(subdir, entry)

            if manifest:
                path = _write_manifest(subdir, manifest)
                print(f"Generated {path} ({len(manifest['Kernels'])} kernels)")


if __name__ == "__main__":
    main()
