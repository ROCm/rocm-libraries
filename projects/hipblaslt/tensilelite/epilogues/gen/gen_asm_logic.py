# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Register a hand-written .s file as a custom kernel and emit a 12-element
LibraryLogic YAML that references it via CustomKernelName.

Usage:
    python epilogues/gen/gen_asm_logic.py <asm_file> [--chip CHIP] [--out-dir DIR]
        [--kernel-name NAME] [--kernargs-version N]
"""

import argparse
import os
import re
import sys
import yaml

_TENSILELITE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TENSILELITE not in sys.path:
    sys.path.insert(0, _TENSILELITE)

# All gfx950 device IDs (see Tensile/Common/Architectures.py).
_GFX950_DEVICE_IDS = [
    "Device 74a1",
    "Device 75a0", "Device 75b0",
    "Device 75a2", "Device 75b2",
    "Device 75a3", "Device 75b3",
    "Device 75a8", "Device 75b8",
]

# Maximum length enforced to keep filenames sane.
_MAX_NAME_LEN = 64


def _extract_meta_payload(lines):
    """Return the YAML payload lines from inside the .amdgpu_metadata block."""
    metaLines = []
    inMeta = False
    for ln in lines:
        if ".amdgpu_metadata" in ln and ".end_amdgpu_metadata" not in ln:
            inMeta = True
            continue
        if ".end_amdgpu_metadata" in ln:
            break
        if inMeta:
            metaLines.append(ln)

    # Skip the opening delimiter and stop at the closing delimiter.
    payloadLines = []
    seenOpen = False
    for ln in metaLines:
        stripped = ln.strip()
        if stripped in ("---", "...") and not seenOpen:
            seenOpen = True
            continue
        if stripped in ("---", "...") and seenOpen:
            break
        if seenOpen:
            payloadLines.append(ln)
    return payloadLines


def parse_asm(asmPath):
    """Extract arch, symbol, and amdhsa.kernels metadata from a .s file."""
    with open(asmPath) as fh:
        text = fh.read()

    archMatch = re.search(r'\.amdgcn_target\s+"amdgcn-amd-amdhsa--([^"]+)"', text)
    if not archMatch:
        raise RuntimeError("could not find .amdgcn_target directive in " + asmPath)
    arch = archMatch.group(1).split(":")[0]

    symMatch = re.search(r'(?m)^\s*\.globl\s+(\S+)', text)
    if not symMatch:
        raise RuntimeError("could not find .globl directive in " + asmPath)
    symbol = symMatch.group(1)

    payloadLines = _extract_meta_payload(text.splitlines())
    doc = yaml.safe_load("\n".join(payloadLines)) or {}
    kernel = (doc.get("amdhsa.kernels") or [{}])[0]

    return {
        "arch":                arch,
        "symbol":              symbol,
        "kernargSize":         kernel.get(".kernarg_segment_size", 0),
        "wavefrontSize":       kernel.get(".wavefront_size", 64),
        "maxFlatWorkgroupSize":kernel.get(".max_flat_workgroup_size", 256),
        "name":                kernel.get(".name", symbol),
        "args":                kernel.get(".args", []),
    }


def _inject_custom_config(lines, kernargsVersion):
    """Inject custom.config after the opening --- of .amdgpu_metadata."""
    out = []
    seenOpen = False
    injected = any("custom.config:" in ln for ln in lines)
    for ln in lines:
        stripped = ln.strip()
        if stripped == "---" and not seenOpen and any(".amdgpu_metadata" in x for x in out[-3:]):
            out.append(ln)
            seenOpen = True
            if not injected:
                out += [
                    "custom.config:",
                    "  InternalSupportParams:",
                    f"    KernArgsVersion: {kernargsVersion}",
                ]
                injected = True
            continue
        if stripped == "---" and seenOpen:
            out.append("...")
            continue
        out.append(ln)
    if not injected:
        raise RuntimeError(
            "failed to locate .amdgpu_metadata opening delimiter to inject custom.config")
    return out


def transform_asm(text, name, chip, kernargsVersion):
    """Rewrite target directive, rename symbol, and inject custom.config."""
    # Rewrite target so the assembler matches the requested chip.
    text = re.sub(
        r'\.amdgcn_target\s+"amdgcn-amd-amdhsa--[^"]+"',
        f'.amdgcn_target "amdgcn-amd-amdhsa--{chip}"',
        text,
    )

    # Rename the kernel symbol if needed (whole-word replace is safe; symbol is unique).
    oldSymMatch = re.search(r'(?m)^\s*\.globl\s+(\S+)', text)
    oldSymbol = oldSymMatch.group(1) if oldSymMatch else name
    if name != oldSymbol:
        text = re.sub(r'\b' + re.escape(oldSymbol) + r'\b', name, text)

    out = _inject_custom_config(text.splitlines(), kernargsVersion)
    return "\n".join(out) + "\n"


def register_asm(text, name):
    """Write the transformed assembly into the Tensile CustomKernels directory."""
    from Tensile import CUSTOM_KERNEL_PATH
    destPath = os.path.join(CUSTOM_KERNEL_PATH, name + ".s")
    with open(destPath, "w") as fh:
        fh.write(text)
    print(f"Registered custom kernel → {destPath}")
    return destPath


def setup_tensile(chip):
    """Initialise TensileLite global state for the requested chip."""
    from pathlib import Path
    from Tensile.Toolchain.Validators import validateToolchain
    from Tensile.Toolchain.Component import Assembler
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.GlobalParameters import assignGlobalParameters
    gfx = chip.split(":")[0]
    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa(gfx)
    isaInfoMap = makeIsaInfoMap([isa], cxx)
    assignGlobalParameters({}, isaInfoMap)
    assembler = Assembler(Path(cxx), co_version="6")
    return assembler, isaInfoMap, isa


def _make_solution_config(chip, isa, isaInfoMap):
    """Return the config dict for a minimal valid GEMM solution."""
    from Tensile.Common.GlobalParameters import defaultInternalSupportParams
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters, validateMIParameters)

    problemType = {
        "OperationType": "GEMM", "DataType": "b", "DestDataType": "b",
        "ComputeDataType": "s", "HighPrecisionAccumulate": True,
        "TransposeA": True, "TransposeB": False, "UseBeta": True,
        "Batched": True, "StridedBatched": True, "GroupedGemm": False,
        "UseBias": 0, "UseScaleAB": "", "UseScaleCD": False,
        "UseScaleAlphaVec": 0, "Sparse": 0,
    }
    # gfx950 MFMA bf16 uses a 32-element K dimension; older arches use 16.
    instK = 32 if chip.startswith("gfx95") else 16
    mi9 = [16, 16, instK, 1, 1, 1, 1, 1, 1]
    miParams = matrixInstructionToMIParameters(
        mi9, isa, 64, problemType, workGroup=None, isaInfoMap=isaInfoMap)

    config = {
        "ProblemType": problemType,
        "InternalSupportParams": defaultInternalSupportParams,
        "ISA": [isa.major, isa.minor, isa.patch],
        "CodeObjectVersion": "6",
        "KernelLanguage": "Assembly",
        "GlobalSplitU": 1,
        "DepthU": 64,
    }
    config.update(miParams)
    if not validateMIParameters(config, isaInfoMap):
        raise RuntimeError("MI parameter validation failed")
    return config


def build_dummy_solution(chip, assembler, isaInfoMap, isa):
    """Build a minimal valid GEMM solution used only for dispatch metadata."""
    from Tensile.SolutionStructs.Solution import Solution

    config = _make_solution_config(chip, isa, isaInfoMap)
    solution = Solution(
        config, splitGSU=False, printSolutionRejectionReason=True,
        printIndexAssignmentInfo=False, assembler=assembler, isaInfoMap=isaInfoMap)

    if not solution["Valid"]:
        raise RuntimeError("dummy solution was rejected")
    return solution


def _to_primitive(v):
    """Recursively convert Tensile-typed objects to YAML-safe Python primitives."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, list):
        return [_to_primitive(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_primitive(val) for k, val in v.items()}
    # DataType and similar objects expose .value as an int.
    if hasattr(v, "value") and isinstance(v.value, int):
        return v.value
    # ActivationType and other Tensile enums: stringify (yields e.g. "None").
    if hasattr(v, "__module__") and "Tensile" in str(getattr(v, "__module__", "")):
        return str(v)
    # SemanticVersion / ISA namedtuple: convert to [major, minor, patch].
    if hasattr(v, "major"):
        return [v.major, v.minor, v.patch]
    return str(v)


def _problem_type_state(pt):
    return {k: _to_primitive(pt[k]) for k in sorted(pt.keys())}


def _solution_state(sol, name):
    d = {k: _to_primitive(v) for k, v in dict(sol).items()}

    # ISA namedtuple → [major, minor, patch].
    if hasattr(d.get("ISA"), "major"):
        d["ISA"] = [d["ISA"].major, d["ISA"].minor, d["ISA"].patch]

    # ProblemType lives in the outer YAML element; drop it from the solution dict.
    d.pop("ProblemType", None)

    # InternalSupportParams namedtuple → plain dict.
    isp = d.get("InternalSupportParams")
    if isp is not None and not isinstance(isp, dict):
        try:
            d["InternalSupportParams"] = dict(isp._asdict())
        except AttributeError:
            d["InternalSupportParams"] = dict(vars(isp))

    # Reference the custom kernel by name so the loader reads it from CustomKernels/.
    d.update({"SolutionIndex": 0, "Valid": True, "KernelNameMin": name,
              "BaseName": name, "CustomKernelName": name})

    # Type coercions required to pass TensileCreateLibrary validation.
    if "BufferStore" in d:
        d["BufferStore"] = bool(d["BufferStore"])
    if "GlobalReadPerMfma" in d:
        d["GlobalReadPerMfma"] = float(d["GlobalReadPerMfma"])
    if "StaggerUStride" in d:
        d["StaggerUStride"] = int(d["StaggerUStride"])

    # Ensure the dispatch block size equals NumThreads to avoid launch failures.
    num_threads = d.get("NumThreads", 0)
    if num_threads > 0:
        d["WorkGroup"] = [num_threads, 1, 1]

    return d


def _write_logic_yaml(logic, outDir, chip, name):
    """Write the logic list to a YAML file and round-trip verify it."""
    logicDir = os.path.join(outDir, chip, "Equality")
    os.makedirs(logicDir, exist_ok=True)
    outPath = os.path.join(logicDir, name + ".yaml")

    with open(outPath, "w") as fh:
        yaml.dump(logic, fh, default_flow_style=None, sort_keys=True, allow_unicode=True)

    with open(outPath) as fh:
        check = yaml.safe_load(fh)
    assert isinstance(check, list) and len(check) == 12, "yaml round-trip failed"
    assert check[5][0]["CustomKernelName"] == name, \
        "customKernelName mismatch after round-trip"
    return outPath


def generate(asmPath, chipOverride, outDir, kernelNameOverride, kernargsVersion):
    """Register the .s file and write a LibraryLogic YAML for it.

    Returns the path to the written YAML.
    """
    info = parse_asm(asmPath)
    chip = chipOverride or info["arch"]
    name = kernelNameOverride or info["symbol"]

    if len(name) > _MAX_NAME_LEN:
        raise RuntimeError(
            f"kernel name '{name}' is {len(name)} characters; max is {_MAX_NAME_LEN}")

    with open(asmPath) as fh:
        rawText = fh.read()
    register_asm(transform_asm(rawText, name, chip, kernargsVersion), name)

    assembler, isaInfoMap, isa = setup_tensile(chip)
    sol = build_dummy_solution(chip, assembler, isaInfoMap, isa)

    deviceIds = _GFX950_DEVICE_IDS if chip == "gfx950" else []
    logic = [
        {"MinimumRequiredVersion": "5.0.0"},
        chip, chip, deviceIds,
        _problem_type_state(sol["ProblemType"]),
        [_solution_state(sol, name)],
        [0],
        # Placeholder dispatch size; callers relying on Equality routing should replace [128, 1, 1, 1] with the real (M, N, batch, K) tuple.
        [[[128, 1, 1, 1], [0, 0]]],
        None, None,
        "DeviceEfficiency",
        "Equality",
    ]

    outPath = _write_logic_yaml(logic, outDir, chip, name)
    print(f"Wrote LibraryLogic YAML → {outPath}")
    return outPath


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asm_file",
                        help="Path to the hand-written .s assembly file.")
    parser.add_argument("--chip", default=None,
                        help="GPU architecture override (default: read from .s).")
    parser.add_argument("--out-dir", default="/tmp/asm_logic",
                        help="Root output directory for logic YAMLs "
                             "(default: /tmp/asm_logic).")
    parser.add_argument("--kernel-name", default=None,
                        help="Override the kernel symbol name used in the library.")
    parser.add_argument("--kernargs-version", type=int, default=2,
                        help="KernArgsVersion injected into custom.config (default: 2).")
    args = parser.parse_args()
    generate(args.asm_file, args.chip, args.out_dir, args.kernel_name,
             args.kernargs_version)


if __name__ == "__main__":
    main()
