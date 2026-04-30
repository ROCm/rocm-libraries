################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from . import CUSTOM_KERNEL_PATH
from Tensile.Common.ValidParameters import checkParametersAreValid, validParameters, newMIValidParameters

import re
import yaml

import os

# ---------------------------------------------------------------------------
# Mapping from amdgpu_metadata arg .name -> CustomArgSemantic string
# ---------------------------------------------------------------------------
_METADATA_NAME_TO_SEMANTIC = {
    "Gemm info":              "GemmInfo",
    "kernel info":            "InternalArgs",
    "kernel info0":           "InternalArgs",
    "kernel info1":           "InternalArgs1",
    "internalArgs":           "InternalArgs",
    "numWG":                  "NumWorkGroups",
    "D":                      "AddressD",
    "C":                      "AddressC",
    "A":                      "AddressA",
    "B":                      "AddressB",
    "MetaData":               "AddressMetadata",
    "AddressDbg":             "DebugBuffer",
    "AddressWS":              "AddressWorkspace",
    "AddressFlags":           "AddressFlags",
    "alpha":                  "Alpha",
    "beta":                   "Beta",
    "AddressScaleA":          "AddressScaleA",
    "AddressScaleB":          "AddressScaleB",
    "AddressScaleC":          "AddressScaleC",
    "AddressScaleD":          "AddressScaleD",
    "AddressMXScaleA":        "AddressMXScaleA",
    "AddressMXScaleB":        "AddressMXScaleB",
    "AddressScaleAlphaVec":   "AddressScaleAlphaVec",
    "bias":                   "AddressBias",
    "biasType":               "BiasType",
    "StrideBias":             "StrideBias",
    "factorDim":              "FactorDim",
    "E":                      "AddressE",
    "activationType":         "ActivationTypeArg",
    "AddrAmaxOut":            "AddressAmaxOut",
    "AmaxWS":                 "AmaxWS",
    "AmaxSync":               "AmaxSync",
    "dstD":                   "AddressTD",
    "Synchronizer":           "Synchronizer",
    "GSUSync":                "GSUSync",
    "ItersPerTile":           "ItersPerTile",
    "MagicNumberItersPerTile": "MagicNumberItersPerTile",
    "MagicShiftItersPerTile": "MagicShiftItersPerTile",
    "TotalIters":             "TotalIters",
    "SKItersPerWG":           "SKItersPerWG",
    "skGrid":                 "SKGrid",
    "skTiles":                "SKTilesAndSplit",
}

_ACTIVATION_ARG_INDEX = {
    "activationAlpha": 0,
    "activationBeta":  1,
    "activationGamma": 2,
    "activationDelta": 3,
}

# Top-level custom.config keys that are not in validParameters but must survive
# the parameter-validation/strip pass in getCustomKernelConfig.
#
# These two are consumed structurally by Tensile (ProblemType drives the
# solution; InternalSupportParams threads through to the kernel writer) and
# would otherwise be popped because they're not tunable parameters.
#
# Provenance-only keys (Source, Version, Features) are deliberately NOT in this
# set: getCustomKernelConfig drops them so they don't pollute the solution
# dict, while validateCustomKernelMetadata still sees them via its independent
# readCustomKernelConfig call.
_PASSTHROUGH_KEYS = {"ProblemType", "InternalSupportParams"}

def isCustomKernelConfig(config):
    if "CustomKernel" in config and config["CustomKernel"]["name"]:
        if config["CustomKernel"].get("generated", False):
            return False
        return True
    return bool(config.get("CustomKernelName", ""))

def getCustomKernelFilepath(name, directory=CUSTOM_KERNEL_PATH):
    flat = os.path.join(directory, (name + ".s"))
    if os.path.isfile(flat):
        return flat
    for path in iterCustomKernelFiles(directory):
        if os.path.basename(path) == (name + ".s"):
            return path
    return flat

def iterCustomKernelFiles(directory=CUSTOM_KERNEL_PATH):
    """Yield custom kernel assembly files using the same recursive discovery as the loader."""
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for fname in sorted(files):
            if fname.endswith(".s"):
                yield os.path.join(root, fname)

def getAllCustomKernelNames(directory=CUSTOM_KERNEL_PATH):
    return [os.path.basename(path)[:-2] for path in iterCustomKernelFiles(directory)]

def getCustomKernelContents(name, directory=CUSTOM_KERNEL_PATH):
    try:
        with open(getCustomKernelFilepath(name, directory)) as f:
            return f.read()
    except OSError as e:
        raise RuntimeError("Failed to find custom kernel: {}".format(os.path.join(directory, name))) from e

def _readEmbeddedYaml(name, directory=CUSTOM_KERNEL_PATH):
    """Parse the YAML payload between '---' and '...' inside .amdgpu_metadata.

    The .s files emitted under this branch contain exactly one such block;
    we stop at the first '...' and return whatever yaml.safe_load returns
    (typically a mapping with 'custom.config' and 'amdhsa.kernels' keys).
    """
    contents = getCustomKernelContents(name, directory)
    inYaml = False
    yamlLines = []
    for line in contents.splitlines():
        if line == "---":
            inYaml = True
            continue
        if line == "..." and inYaml:
            break
        if inYaml:
            yamlLines.append(line)
    try:
        return yaml.safe_load("\n".join(yamlLines))
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse YAML for custom kernel '{name}': {e}") from e

def readCustomKernelConfig(name, directory=CUSTOM_KERNEL_PATH):
    parsed = _readEmbeddedYaml(name, directory)
    if not isinstance(parsed, dict) or "custom.config" not in parsed:
        raise RuntimeError(f"Custom kernel '{name}' has no custom.config in its .amdgpu_metadata")
    config = parsed["custom.config"]
    if not isinstance(config, dict):
        raise RuntimeError(f"Custom kernel '{name}' custom.config must be a YAML mapping")
    return config

def _metadataArgToCustomArg(metaArg, kernelName=None):
    """Convert a single amdgpu_metadata .args entry to a CustomKernel arg dict.

    Raises RuntimeError with an actionable message if the arg name is not
    recognized (which can only happen when auto-inferring a CustomKernel
    block for a kernel that does not declare one explicitly).
    """
    name = metaArg.get(".name")
    size = metaArg.get(".size")
    valueKind = metaArg.get(".value_kind")
    if name is None or size is None or valueKind is None:
        raise RuntimeError(
            f"amdgpu_metadata arg entry missing required field "
            f"(.name/.size/.value_kind): {metaArg}"
        )

    if valueKind == "global_buffer":
        argType = "address"
    elif size == 8:
        argType = "float64"
    else:
        argType = "uint32"

    if name in _METADATA_NAME_TO_SEMANTIC:
        return {"type": argType, "semantic": _METADATA_NAME_TO_SEMANTIC[name]}

    if name in _ACTIVATION_ARG_INDEX:
        actType = "float64" if size > 4 else "float32"
        entry = {"type": actType, "semantic": "ActivationArg"}
        idx = _ACTIVATION_ARG_INDEX[name]
        if idx:
            entry["index"] = idx
        return entry

    m = re.match(r"SizesFree(\d+)", name)
    if m:
        return {"type": argType, "semantic": "SizeFree%s" % m.group(1)}

    m = re.match(r"SizesSum(\d+)", name)
    if m:
        idx = int(m.group(1))
        return {"type": argType, "semantic": "SizeSum" if idx == 0 else "SizeSum%d" % idx}

    m = re.match(r"stride([A-Z])(\d+)", name)
    if m:
        return {"type": argType, "semantic": "Stride%s%s" % (m.group(1), m.group(2))}

    m = re.match(r"strideMetadata(\d+)", name)
    if m:
        return {"type": argType, "semantic": "StrideMetadata%s" % m.group(1)}

    m = re.match(r"StrideE(\d+)", name)
    if m:
        return {"type": argType, "semantic": "StrideE%s" % m.group(1)}

    m = re.match(r"(MagicNumberSize|MagicShiftSize)(\w)", name)
    if m:
        from Tensile.Common.Constants import INDEX_CHARS
        idx = INDEX_CHARS.index(m.group(2))
        return {"type": argType, "semantic": m.group(1), "index": idx}

    where = f" in kernel '{kernelName}'" if kernelName else ""
    raise RuntimeError(
        f"Unknown amdgpu_metadata arg name '{name}'{where} while auto-inferring "
        f"a CustomKernel block. Either add an explicit CustomKernel: section "
        f"to custom.config, or extend _METADATA_NAME_TO_SEMANTIC in "
        f"Tensile/CustomKernels.py."
    )

_HEADER_SEMANTIC_ORDER = {
    "GemmInfo": 0, "InternalArgs": 1, "InternalArgs1": 2, "NumWorkGroups": 3,
}

def _buildCustomKernelFromMetadata(kernelName, fullYaml, kernelConfig):
    """Build a CustomKernel dict from the amdgpu_metadata and custom.config sections."""
    if not isinstance(fullYaml, dict):
        raise RuntimeError(f"Custom kernel '{kernelName}' has no parseable .amdgpu_metadata YAML")
    kernels = fullYaml.get("amdhsa.kernels") or []
    if not kernels:
        raise RuntimeError(
            f"Custom kernel '{kernelName}' has no amdhsa.kernels entries; cannot "
            f"auto-infer a CustomKernel block. Add an explicit CustomKernel: "
            f"section to custom.config."
        )
    kernelMeta = kernels[0]
    if ".args" not in kernelMeta:
        raise RuntimeError(
            f"Custom kernel '{kernelName}' amdhsa.kernels[0] has no .args; cannot "
            f"auto-infer a CustomKernel block. Add an explicit CustomKernel: "
            f"section to custom.config."
        )

    args = [_metadataArgToCustomArg(a, kernelName) for a in kernelMeta[".args"]]

    # UseUniversalArgs kernels expect a header (GemmInfo, InternalArgs, ...) at
    # the start of the kernel argument buffer, followed by the data args.  The
    # .amdgpu_metadata declares them at interior offsets, so reorder here to
    # match the actual runtime layout the assembly prologue depends on.
    isp = kernelConfig.get("InternalSupportParams", {})
    if isp.get("UseUniversalArgs", True):
        headerArgs = [a for a in args if a["semantic"] in _HEADER_SEMANTIC_ORDER]
        dataArgs   = [a for a in args if a["semantic"] not in _HEADER_SEMANTIC_ORDER]
        headerArgs.sort(key=lambda a: _HEADER_SEMANTIC_ORDER[a["semantic"]])
        args = headerArgs + dataArgs

    mi = kernelConfig.get("MatrixInstruction", kernelConfig.get("MIBlock", [0,0,0,0]))
    if len(mi) >= 9 and "MIWaveTile" not in kernelConfig:
        wt = [mi[5], mi[6]]
        wg = [mi[7], mi[8]]
    else:
        wt = kernelConfig.get("MIWaveTile", [1, 1])
        wg = kernelConfig.get("MIWaveGroup", [1, 1])
    depthU = kernelConfig.get("DepthU", 0)
    macrotile = [mi[0] * wt[0] * wg[0], mi[1] * wt[1] * wg[1], depthU]

    # Fallback: parse macrotile from kernel name if computed values are zero
    if macrotile[0] == 0 or macrotile[1] == 0:
        m = re.search(r'MT(\d+)x(\d+)x(\d+)', kernelName)
        if m:
            macrotile = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    threads = [kernelMeta.get(".max_flat_workgroup_size", 256), 1, 1]

    streamK = kernelConfig.get("StreamK", 0)
    hasNumWGArg = any(a.get("semantic") == "NumWorkGroups" for a in args)

    if streamK:
        batched = kernelConfig.get("ProblemType", {}).get("Batched", False)
        grid = ["StreamKWithBatch" if batched else "StreamKNoBatch", "One", "One"]
    elif hasNumWGArg:
        # Version >= 1 kernels receive numWorkGroups as arg and decompose
        # the flat 1-D work-group index internally.
        grid = ["TilesXYBatchGSU", "One", "One"]
    else:
        # Version 0 kernels rely on hardware gridDim for tile decomposition,
        # so the grid must be multi-dimensional.
        grid = ["TilesX", "TilesY", "Batch"]

    return {
        "name": kernelName,
        "args": args,
        "macrotile": macrotile,
        "threads": threads,
        "grid": grid,
        "workspaceType": "None",
        "workspaceSizePerElemC": 0,
        "workspaceSizePerElemBias": 0,
    }

def getCustomKernelConfig(
    kernelName: str, internalSupportParams: dict, directory: str = CUSTOM_KERNEL_PATH
) -> dict:
    """
    Retrieves and validates the configuration for a custom kernel.

    Args:
        kernelName: The name of the custom kernel.
        internalSupportParams: A dictionary of internal support parameters to be merged with the kernel configuration.
        directory: The directory where custom kernel files are located. Defaults to CUSTOM_KERNEL_PATH.

    Returns:
        dict: The validated configuration dictionary for the custom kernel.

    Raises:
        RuntimeError: If the custom kernel configuration is missing required fields or if there is an error reading the configuration.
    """
    kernelConfig = readCustomKernelConfig(kernelName, directory)
    if "InternalSupportParams" not in kernelConfig:
        raise RuntimeError(f"Custom kernel {kernelName} config must have 'InternalSupportParams'")

    if "KernArgsVersion" not in kernelConfig["InternalSupportParams"]:
        raise RuntimeError(f"Custom kernel {kernelName} config must have 'KernArgsVersion'")

    kernelIsp = kernelConfig["InternalSupportParams"]
    for key in internalSupportParams:
        if key not in kernelIsp:
            kernelIsp[key] = internalSupportParams[key]

    # Compute a merged validParameters set locally; do NOT mutate the global
    # validParameters dict (that leaks state across calls and into unit tests
    # via the precomputed _expectedParamTypes cache in test_validateParameterTypes).
    mergedValid = {**validParameters, **newMIValidParameters}

    for k, v in kernelConfig.items():
        if k in _PASSTHROUGH_KEYS:
            continue
        if k in mergedValid:
            checkParametersAreValid((k, [v]), mergedValid)

    metadata_keys = [k for k in kernelConfig if k not in mergedValid and k not in _PASSTHROUGH_KEYS]
    for k in metadata_keys:
        kernelConfig.pop(k)

    kernelConfig["KernelLanguage"] = "Assembly"

    if "CustomKernel" not in kernelConfig:
        fullYaml = _readEmbeddedYaml(kernelName, directory)
        kernelConfig["CustomKernel"] = _buildCustomKernelFromMetadata(kernelName, fullYaml, kernelConfig)

    kernelConfig["CustomKernel"]["name"] = kernelName
    kernelConfig["CustomKernel"].setdefault("workspaceType", "None")
    kernelConfig["CustomKernel"].setdefault("workspaceSizePerElemC", 0)
    kernelConfig["CustomKernel"].setdefault("workspaceSizePerElemBias", 0)
    kernelConfig["CustomKernelName"] = kernelName

    return kernelConfig


################################################################################
# Embedded metadata validation functions
################################################################################

_EXTERNAL_HINT = (
    "External kernels carry their full Tensile-side interface in custom.config. "
    "To inject one from a Tensile test YAML, run:\n"
    "  python -m Tensile.AddCustomConfig <file.s> --yaml <test.yaml>\n"
)

_TENSILE_HINT = (
    "Tensile-generated kernels only need InternalSupportParams.KernArgsVersion in "
    "custom.config; ProblemType and tuning state come from the consuming logic file "
    "or test YAML. If the field is missing, regenerate the kernel with the current "
    "kernel writer."
)

def _requiredFieldsMissing(config, fields):
    return [field for field in fields if field not in config]

def _missingMetadataMessage(kind, name, filepath, missing):
    hint = _EXTERNAL_HINT if kind == "External" else _TENSILE_HINT
    return (
        f"{kind} kernel '{name}' has custom.config but is missing required fields:\n"
        f"  Missing: {', '.join(missing)}\n"
        f"  File: {filepath}\n"
        f"{hint}"
    )

def validateCustomKernelMetadata(name, directory=CUSTOM_KERNEL_PATH):
    """Validates that a kernel has an embedded custom.config with required fields.

    Tensile-generated kernels (no Source.Origin) only need
    InternalSupportParams.KernArgsVersion -- the only field
    `getCustomKernelConfig` actually requires at runtime. Their ProblemType and
    tuning state live in the consuming logic file or test YAML and are merged
    on top of custom.config there.

    External kernels (Source.Origin present) carry their full Tensile-side
    interface and provenance in custom.config: Source, Features, Version,
    InternalSupportParams.KernArgsVersion, ProblemType, MatrixInstruction, and
    a CustomKernel block with args/macrotile/threads/grid.

    Whether failures are reported as errors or warnings is the caller's
    decision (see ValidateMetadata.validate_all and
    Toolchain.Assembly.validateCustomKernelMetadataAtBuild).

    Returns:
        (bool, str): A tuple of (is_valid, message).
    """
    filepath = getCustomKernelFilepath(name, directory)

    try:
        config = readCustomKernelConfig(name, directory)
    except RuntimeError as e:
        # Use the external hint here -- a kernel without any custom.config is
        # almost always an external kernel that hasn't been migrated yet.
        return False, (
            f"Cannot read custom.config for '{name}' ({filepath}): {e}\n"
            f"{_EXTERNAL_HINT}"
        )

    is_external = "Source" in config
    missing = []

    if "InternalSupportParams" not in config:
        missing.append("InternalSupportParams")
    elif not isinstance(config["InternalSupportParams"], dict):
        missing.append("InternalSupportParams (mapping)")
    elif "KernArgsVersion" not in config["InternalSupportParams"]:
        missing.append("InternalSupportParams.KernArgsVersion")

    if is_external:
        source = config.get("Source")
        if not isinstance(source, dict) or "Origin" not in source:
            missing.append("Source.Origin")
        if "Features" not in config or not isinstance(config["Features"], dict):
            missing.append("Features")
        missing.extend(_requiredFieldsMissing(
            config,
            ["Version", "ProblemType", "CustomKernel", "MatrixInstruction"],
        ))

    if "CustomKernel" in config:
        custom_kernel = config["CustomKernel"]
        if not isinstance(custom_kernel, dict):
            missing.append("CustomKernel (mapping)")
        else:
            ck_missing = _requiredFieldsMissing(custom_kernel, ["args", "macrotile", "threads", "grid"])
            missing.extend(f"CustomKernel.{field}" for field in ck_missing)

    if missing:
        kind = "External" if is_external else "Tensile"
        return False, _missingMetadataMessage(kind, name, filepath, missing)

    return True, f"Kernel '{name}' metadata is valid"
