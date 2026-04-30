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
    for root, _, files in os.walk(directory):
        if (name + ".s") in files:
            return os.path.join(root, (name + ".s"))
    return flat

def getAllCustomKernelNames(directory=CUSTOM_KERNEL_PATH):
    names = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".s"):
                names.append(fname[:-2])
    return names

def getCustomKernelContents(name, directory=CUSTOM_KERNEL_PATH):
    try:
        with open(getCustomKernelFilepath(name, directory)) as f:
            return f.read()
    except:
        raise RuntimeError("Failed to find custom kernel: {}".format(os.path.join(directory, name)))

def getCustomKernelConfigAndAssembly(name, directory=CUSTOM_KERNEL_PATH):
    contents  = getCustomKernelContents(name, directory)
    config = "\n"    #Yaml configuration properties
    assembly = ""
    inConfig = False
    for line in contents.splitlines():
        if   line == "---": inConfig = True                          #Beginning of yaml section
        elif line == "...": inConfig = False                         #End of yaml section
        elif      inConfig: config   += line + "\n"
        else              : assembly += line + "\n"; config += "\n"  #Second statement to keep line numbers consistent for yaml errors

    return (config, assembly)

def readCustomKernelConfig(name, directory=CUSTOM_KERNEL_PATH):
    rawConfig, _ = getCustomKernelConfigAndAssembly(name, directory)
    try:
        return yaml.safe_load(rawConfig)["custom.config"]
    except yaml.scanner.ScannerError as e:
        raise RuntimeError("Failed to read configuration for custom kernel: {0}\nDetails:\n{1}".format(name, e))

def _readFullYaml(name, directory=CUSTOM_KERNEL_PATH):
    """Read and return the full parsed YAML (all sections) from a custom kernel .s file."""
    rawConfig, _ = getCustomKernelConfigAndAssembly(name, directory)
    try:
        return yaml.safe_load(rawConfig)
    except yaml.scanner.ScannerError as e:
        raise RuntimeError("Failed to read YAML for custom kernel: {0}\nDetails:\n{1}".format(name, e))

def _metadataArgToCustomArg(metaArg):
    """Convert a single amdgpu_metadata .args entry to a CustomKernel arg dict."""
    name = metaArg[".name"]
    size = metaArg[".size"]
    valueKind = metaArg[".value_kind"]

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

    raise RuntimeError("Unknown metadata arg name: '%s'" % name)

_HEADER_SEMANTIC_ORDER = {
    "GemmInfo": 0, "InternalArgs": 1, "InternalArgs1": 2, "NumWorkGroups": 3,
}

def _buildCustomKernelFromMetadata(kernelName, fullYaml, kernelConfig):
    """Build a CustomKernel dict from the amdgpu_metadata and custom.config sections."""
    kernelMeta = fullYaml["amdhsa.kernels"][0]

    args = [_metadataArgToCustomArg(a) for a in kernelMeta[".args"]]

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

    validParameters.update(newMIValidParameters)

    for k, v in kernelConfig.items():
        if k != "ProblemType":
            checkParametersAreValid((k, [v]), validParameters)

    kernelConfig["KernelLanguage"] = "Assembly"

    if "CustomKernel" not in kernelConfig:
        fullYaml = _readFullYaml(kernelName, directory)
        kernelConfig["CustomKernel"] = _buildCustomKernelFromMetadata(kernelName, fullYaml, kernelConfig)

    kernelConfig["CustomKernel"]["name"] = kernelName
    kernelConfig["CustomKernel"].setdefault("workspaceType", "None")
    kernelConfig["CustomKernel"].setdefault("workspaceSizePerElemC", 0)
    kernelConfig["CustomKernel"].setdefault("workspaceSizePerElemBias", 0)
    kernelConfig["CustomKernelName"] = kernelName

    return kernelConfig
