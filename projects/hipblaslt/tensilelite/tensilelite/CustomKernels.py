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

from .resources import custom_kernel_names, custom_kernel_text
from .Common.ValidParameters import checkParametersAreValid, validParameters, newMIValidParameters

from .ExecutionPolicy import isPersistent, isPersistentDataParallel

import re
import yaml

import os

# rocRoller kernels embed their full kernel graph in .amdgpu_metadata, which the
# pure-Python SafeLoader takes over a second per kernel to parse.
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

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
    "betapad":                "Beta",
    "cuCount":                "ComputeUnits",
    "ComputeUnits":           "ComputeUnits",
    "AddressScaleA":          "AddressScaleA",
    "AddressScaleB":          "AddressScaleB",
    "AddressScaleZeroA":      "AddressScaleZeroA",
    "groupSize":              "ScaleBlockSizeA",
    "AddressScaleC":          "AddressScaleC",
    "AddressScaleD":          "AddressScaleD",
    "AddressMXScaleA":        "AddressMXScaleA",
    "AddressMXScaleB":        "AddressMXScaleB",
    "MXSA":                   "AddressMXScaleA",
    "MXSB":                   "AddressMXScaleB",
    "AddressScaleAlphaVec":   "AddressScaleAlphaVec",
    "bias":                   "AddressBias",
    "biasType":               "BiasType",
    "StrideBias":             "StrideBias",
    "gate":                   "AddressGateResidual",
    "gateType":               "GateResidualType",
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
    "PersistentGrid":         "PersistentGrid",
    "MagicNumberItersPerTile": "MagicNumberItersPerTile",
    "MagicShiftItersPerTile": "MagicShiftItersPerTile",
    "TotalIters":             "TotalIters",
    "SKItersPerWG":           "SKItersPerWG",
    "skGrid":                 "SKGrid",
    "skTiles":                "SKTilesAndSplit",
    "ESMRuntimeSupported":    "ESMRuntimeSupported",
    "batchOffsetD":           "BatchOffsetD",
    "batchOffsetC":           "BatchOffsetC",
    "batchOffsetA":           "BatchOffsetA",
    "batchOffsetB":           "BatchOffsetB",
}

# rocRoller Stream-K hoists its work decomposition into kernargs whose names
# carry no numeric suffix, so they are safe to match by name here. Its four
# libdivide constants DO carry an unstable suffix and are classified by
# expression instead, in RocRollerStreamK.custom_arg_for_metadata.
from .RocRollerStreamK import STABLE_NAME_TO_SEMANTIC as _ROCROLLER_SK_NAMES

_METADATA_NAME_TO_SEMANTIC.update(_ROCROLLER_SK_NAMES)

_ACTIVATION_ARG_INDEX = {
    "activationAlpha": 0,
    "activationBeta":  1,
    "activationGamma": 2,
    "activationDelta": 3,
}

# Top-level custom.config keys that are not in validParameters but must survive
# the parameter-validation/strip pass in getCustomKernelConfig.
#
# These two are consumed structurally by TensileLite (ProblemType drives the
# solution; InternalSupportParams threads through to the kernel writer) and
# would otherwise be popped because they're not tunable parameters.
#
# Provenance-only keys (Source, Version, Features) are deliberately NOT in this
# set: getCustomKernelConfig drops them so they don't pollute the solution
# dict, while validateCustomKernelMetadata still sees them via its independent
# readCustomKernelConfig call.
_PASSTHROUGH_KEYS = {"ProblemType", "InternalSupportParams", "KernelLanguage", "CustomKernelName"}

def isCustomKernelConfig(config):
    # CustomKernel may be absent, None, or the -1 placeholder for an unset parameter,
    # so check that it is a populated dict before reaching into it.
    ck = config.get("CustomKernel")
    if isinstance(ck, dict) and ck.get("name"):
        return not ck.get("generated", False)
    return bool(config.get("CustomKernelName", ""))

def customKernelNameOf(config):
    """The custom kernel a solution state (or fork-parameter entry) names, else ""."""
    ck = config.get("CustomKernel")
    if isinstance(ck, dict) and ck.get("name") and not ck.get("generated", False):
        return ck["name"]
    if config.get("name") and not config.get("generated", False):
        return config["name"]
    return config.get("CustomKernelName", "") or ""


def isExternalCustomKernel(name, directory=None):
    """Whether a kernel was captured from outside Tensile.

    An external capture declares a ``Source`` block in its embedded
    ``custom.config`` naming where it came from (``rocroller``, ``aiter``, ...);
    a Tensile-generated kernel has no such block. This is the same split
    ``validateCustomKernelMetadata`` keys its required-field set on.

    Unreadable metadata counts as not external, which keeps the caller on the
    conservative branch instead of failing the run.
    """
    if not name:
        return False
    try:
        config = readCustomKernelConfig(name, directory)
    except (RuntimeError, OSError):
        return False
    return "Source" in config


def mxScalesArePackedForSolutions(solutions, directory=None):
    """Whether the benchmark client should allocate MX scales with no padding.

    A micro-scaled GEMM carries one scale byte per 32 elements along the
    summation dimension, so one row of scales needs ceil(K/32) bytes. Tensile's
    own generated kernels for gfx950 read a row pitch rounded up to a multiple
    of 8 and need that slack allocated. An externally captured kernel -- every
    rocRoller kernel, for instance -- instead indexes the caller's buffer with
    the unrounded pitch, which is the layout the hipBLASLt public API specifies
    and the only one a production caller ever allocates.

    The benchmark client allocates one scale buffer per problem and shares it
    across every kernel it measures, so the unpadded layout is safe only when
    the whole run agrees on it. Return True only when ``solutions`` is non-empty
    and every one of them is an external capture; a run containing even one
    Tensile-generated kernel keeps the padded layout that kernel requires.
    """
    solutions = list(solutions or [])
    if not solutions:
        return False
    return all(
        isExternalCustomKernel(customKernelNameOf(s), directory) for s in solutions
    )


def supportsUserSgprKernargPreload(rocmVersion):
    """Return whether a ROCm version passes TensileLite's preload gate.

    AMD's ROCm 6.0 compiler branch added descriptor and codegen support in
    September 2023 for feature-enabled targets. HIP recorded 6.0.32650 on
    September 29, and hipBLASLt adopted it as its 6.x floor on October 6.
    That floor is historical compatibility policy, not a complete capability
    test: target ISA, assembler, and firmware also matter.

    HIP's patch field is a build number, not globally monotonic. Official
    ROCm 7 releases can report a patch below 32650, so later major releases
    remain eligible. A locally built ROCm 6.x toolchain reporting a low build
    (for example, 6.4.0) remains ambiguous and is treated as unsupported.
    """
    return rocmVersion.major > 6 or (
        rocmVersion.major == 6 and rocmVersion.patch >= 32650
    )

_DEFAULT_CUSTOM_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CustomKernels")


def _custom_kernel_dir(directory=None):
    """Resolve the CustomKernels tree; None means the packaged source directory."""
    return _DEFAULT_CUSTOM_KERNEL_DIR if directory is None else directory


def getCustomKernelFilepath(name, directory=None):
    directory = _custom_kernel_dir(directory)
    flat = os.path.join(directory, (name + ".s"))
    if os.path.isfile(flat):
        return flat
    for path in iterCustomKernelFiles(directory):
        if os.path.basename(path) == (name + ".s"):
            return path
    return flat

def iterCustomKernelFiles(directory=None):
    """Yield custom kernel assembly files using the same recursive discovery as the loader."""
    directory = _custom_kernel_dir(directory)
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for fname in sorted(files):
            if fname.endswith(".s"):
                yield os.path.join(root, fname)

def getAllCustomKernelNames(directory=None):
    if directory is None:
        return custom_kernel_names()
    # Sorted in alphabetical order so that custom-kernel enumeration (notably the CustomKernels: ["*"]
    # wildcard) does not depend on os.listdir order, which varies with the
    # filesystem and with how the package was installed. iterCustomKernelFiles
    # walks vendor subdirectories (aiter/, tensile/, ...).
    return sorted(os.path.basename(path)[:-2] for path in iterCustomKernelFiles(directory))

def getCustomKernelContents(name, directory=None):
    if directory is None:
        try:
            return custom_kernel_text(name)
        except ValueError:
            raise
        except Exception as error:
            raise RuntimeError(f"Failed to find custom kernel: {name}") from error
    try:
        with open(getCustomKernelFilepath(name, directory)) as f:
            return f.read()
    except OSError as e:
        raise RuntimeError("Failed to find custom kernel: {}".format(os.path.join(directory, name))) from e

def getCustomKernelSource(name, rocmVersion, directory=None):
    contents = getCustomKernelContents(name, directory)
    if supportsUserSgprKernargPreload(rocmVersion):
        return contents
    return "".join(
        line
        for line in contents.splitlines(keepends=True)
        if "amdhsa_user_sgpr_kernarg_preload" not in line
    )

_CUSTOM_CONFIG_KEY = re.compile(r"custom\.config:(\s|$)")

def _embeddedYamlLines(contents):
    """Lines between '---' and the first '...' of the .amdgpu_metadata block."""
    inYaml = False
    for line in contents.splitlines():
        if line == "---":
            inYaml = True
            continue
        if line == "..." and inYaml:
            return
        if inYaml:
            yield line

def _customConfigYaml(contents):
    """Text of the top-level custom.config entry, or None if it is not found.

    The entry runs until the next line that starts in column 0 with something
    other than a comment.

    The scan continues to the end of the block, and a second column-0
    "custom.config:" line makes the result None. PyYAML keeps the last of two
    duplicate keys, and the same line can sit inside a multi-line quoted scalar
    of an earlier key, so the first match is not safe to slice on its own; the
    whole block decides instead.

    The text ends with a newline so that a block scalar ending the entry keeps
    its final newline, as it does in a whole-block parse.
    """
    section = None
    inEntry = False
    for line in _embeddedYamlLines(contents):
        if _CUSTOM_CONFIG_KEY.match(line):
            if section is not None:
                return None
            section = [line]
            inEntry = True
        elif inEntry:
            if line and line[0] not in " \t#":
                inEntry = False
            else:
                section.append(line)
    return None if section is None else "\n".join(section) + "\n"

def _readEmbeddedYaml(name, directory=None, customConfigOnly=False):
    """Parse the YAML payload between '---' and '...' inside .amdgpu_metadata.

    The .s files emitted under this branch contain exactly one such block;
    we stop at the first '...' and return whatever the safe loader returns
    (typically a mapping with 'custom.config' and 'amdhsa.kernels' keys).

    customConfigOnly parses just the custom.config entry, a few KB inside a
    block that can be 1 MB (rocRoller's serialized kernel graph). The rest of
    the block is not parsed, so errors in it go unnoticed. The whole block is
    parsed when custom.config is not found in its usual form, appears more than
    once, or the slice does not parse.
    """
    contents = getCustomKernelContents(name, directory)
    if customConfigOnly:
        text = _customConfigYaml(contents)
        if text is not None:
            try:
                return yaml.load(text, Loader=SafeLoader)
            except yaml.YAMLError:
                # The slice can cut valid YAML: a flow collection closed in
                # column 0, or an alias to an anchor defined outside the entry.
                # The whole block parses those, and reports a real error in
                # custom.config.
                pass
    try:
        # The final newline keeps the last block scalar's own newline, which
        # the slice above keeps as well.
        return yaml.load("\n".join(_embeddedYamlLines(contents)) + "\n", Loader=SafeLoader)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse YAML for custom kernel '{name}': {e}") from e

def readCustomKernelConfig(name, directory=None):
    parsed = _readEmbeddedYaml(name, directory, customConfigOnly=True)
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
    elif name in ("batchOffsetD", "batchOffsetC", "batchOffsetA", "batchOffsetB"):
        argType = "uint64"
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
        tensor = "Gate" if m.group(1) == "G" else m.group(1)
        return {"type": argType, "semantic": "Stride%s%s" % (tensor, m.group(2))}

    m = re.match(r"strideMetadata(\d+)", name)
    if m:
        return {"type": argType, "semantic": "StrideMetadata%s" % m.group(1)}

    m = re.match(r"strideMXS([AB])(\d+)", name)
    if m:
        return {"type": argType, "semantic": "StrideScale%s%s" % (m.group(1), m.group(2))}

    m = re.match(r"StrideE(\d+)", name)
    if m:
        return {"type": argType, "semantic": "StrideE%s" % m.group(1)}

    m = re.match(r"(MagicNumberSize|MagicShiftSize)(\w)", name)
    if m:
        from .Common.Constants import INDEX_CHARS
        idx = INDEX_CHARS.index(m.group(2))
        return {"type": argType, "semantic": m.group(1), "index": idx}

    # rocRoller suffixes (Subtract_21, MagicMultiple_26, WGM_18) are not
    # stable. The expression tree selects the launch semantic.
    # Stream-K's MagicMultiple_/MagicShifts_ constants carry the same kind of
    # unstable numeric suffix, and the same constant changes suffix between
    # tiles, so they are classified by the divisor their expression reads.
    #
    # This runs before the workgroup-mapping hook because the two overlap on
    # MagicMultiple over the N tile count. The Stream-K classifier is the
    # strict one -- it only claims the unsigned form -- so putting it first
    # resolves the overlap without it ever taking a workgroup-mapping
    # argument. A kernel never carries both: rocRoller forces
    # workgroupMappingDim = -1 for every Stream-K kernel.
    from .RocRollerStreamK import custom_arg_for_metadata as sk_custom_arg_for_metadata
    skArg = sk_custom_arg_for_metadata(metaArg)
    if skArg is not None:
        return skArg

    from .RocRollerWgm import custom_arg_for_metadata
    wgmArg = custom_arg_for_metadata(metaArg)
    if wgmArg is not None:
        return wgmArg

    where = f" in kernel '{kernelName}'" if kernelName else ""
    raise RuntimeError(
        f"Unknown amdgpu_metadata arg name '{name}'{where} while auto-inferring "
        f"a CustomKernel block. Either add an explicit CustomKernel: section "
        f"to custom.config, or extend _METADATA_NAME_TO_SEMANTIC in "
        f"tensilelite/CustomKernels.py."
    )

_HEADER_SEMANTIC_ORDER = {
    "GemmInfo": 0, "InternalArgs": 1, "InternalArgs1": 2, "NumWorkGroups": 3,
}

def validateCustomPersistentArgs(kernelConfig):
    """Validate the scheduling payload of a versioned persistent descriptor."""
    version = kernelConfig.get("InternalSupportParams", {}).get("PersistentLoopArgsVersion", 0)
    if type(version) is not int or version not in (0, 1, 2):
        raise ValueError("Unsupported PersistentLoopArgsVersion")
    outer_version = kernelConfig.get("InternalSupportParams", {}).get("KernArgsVersion", 3)
    if type(outer_version) is not int or outer_version not in (0, 1, 2, 3):
        raise ValueError("Unsupported KernArgsVersion")
    if version in (1, 2) and outer_version != 3:
        raise ValueError(f"PersistentLoopArgsVersion={version} requires KernArgsVersion=3")
    descriptor = kernelConfig.get("CustomKernel")
    if not isinstance(descriptor, dict) or not descriptor.get("name"):
        if version in (1, 2) and kernelConfig.get("CustomKernelName"):
            raise ValueError(f"Persistent argument layout version {version} requires a custom-kernel argument descriptor")
        return
    args = descriptor.get("args", [])
    persistentGridArgIndices = [i for i, arg in enumerate(args) if arg.get("semantic") == "PersistentGrid"]
    if version == 0:
        if persistentGridArgIndices:
            raise ValueError("PersistentGrid requires PersistentLoopArgsVersion=1")
        return
    if version == 2:
        expected = ["ItersPerTile", "MagicNumberItersPerTile", "MagicShiftItersPerTile",
                    "SKItersPerWG", "SKGrid", "SKTilesAndSplit"]
        forbidden = {"PersistentGrid", "TotalIters", "AddressSynchronizer", "Synchronizer", "GSUSync"}
        scheduling = [(i, arg) for i, arg in enumerate(args)
                      if arg.get("semantic") in set(expected) | forbidden]
        if ([arg.get("semantic") for _, arg in scheduling] != expected
                or any(arg.get("type") != "uint32" or arg.get("padding", 0)
                       for _, arg in scheduling)
                or any(right[0] != left[0] + 1
                       for left, right in zip(scheduling, scheduling[1:]))):
            raise ValueError("StreamK version-2 scheduling arguments must be six adjacent uint32 values")
        for semantic in ("AddressWorkspace", "AddressFlags"):
            pointers = [arg for arg in args if arg.get("semantic") == semantic]
            if (len(pointers) != 1 or pointers[0].get("type") != "address"
                    or pointers[0].get("padding", 0)):
                raise ValueError(f"StreamK version-2 requires one {semantic} pointer")
        # An omitted workspace declaration is derived from the authoritative
        # problem compute type by Solution._assignCustomKernelParameters.
        if descriptor.get("workspaceType", "None") not in ("None", "StreamKWithReduction"):
            raise ValueError("StreamK version-2 requires tree partial workspace")
        return
    legacy = {"MagicNumberItersPerTile", "MagicShiftItersPerTile", "TotalIters",
              "SKItersPerWG", "SKGrid", "SKTilesAndSplit", "AddressWorkspace", "AddressFlags",
              "AddressSynchronizer", "Synchronizer", "GSUSync"}
    scheduling = [(i, arg) for i, arg in enumerate(args)
                  if arg.get("semantic") in legacy | {"ItersPerTile", "PersistentGrid"}]
    if (len(scheduling) != 2
            or [arg.get("semantic") for _, arg in scheduling] != ["ItersPerTile", "PersistentGrid"]
            or scheduling[1][0] != scheduling[0][0] + 1
            or any(arg.get("type") != "uint32" or arg.get("padding", 0) for _, arg in scheduling)):
        raise ValueError("DataParallel version-1 scheduling arguments must be adjacent uint32 ItersPerTile and PersistentGrid")
    if (descriptor.get("workspaceType", "None") != "None"
            or descriptor.get("workspaceSizePerElemC", 0)
            or descriptor.get("workspaceSizePerElemBias", 0)):
        raise ValueError("DataParallel custom kernels with argument layout version 1 cannot require partial workspace")


def _hasRocRollerWgmArgs(args):
    """True when the metadata expressions are rocRoller's workgroup-mapping kernargs.

    The kernel name is not consulted. A kernel that merely ends in ``_WGM``
    does not match, and neither does a rocRoller kernel that does not hoist
    these libdivide arguments.
    """
    from .RocRollerWgm import WGM_ROLE_ORDER

    roles = frozenset(WGM_ROLE_ORDER)
    return any(arg.get("semantic") in roles for arg in args)


# Compute-type letters as they appear in a custom.config ProblemType, mapped
# to the byte width of one accumulator element. A rocRoller Stream-K kernel
# stores partial tiles in the accumulator type, so this is what sizes its
# workspace.
_COMPUTE_TYPE_BYTES = {
    "s": 4, "f32": 4, "f32_r": 4, "Float": 4, "Single": 4,
    "d": 8, "f64": 8, "f64_r": 8, "Double": 8,
    "h": 2, "f16": 2, "Half": 2,
    "b": 2, "bf16": 2, "BFloat16": 2,
    "i": 4, "i32": 4, "Int32": 4,
}


def _computeTypeBytes(kernelConfig, default=4):
    """Bytes per accumulator element, from the config's ProblemType."""
    problemType = kernelConfig.get("ProblemType") or {}
    key = problemType.get("ComputeDataType", problemType.get("DataType"))
    if isinstance(key, str):
        return _COMPUTE_TYPE_BYTES.get(key, default)
    return default


# gfx950 (CDNA) occupancy model. Four SIMDs per CU, a 512-entry unified
# VGPR+AGPR file per SIMD allocated in blocks of eight, at most eight waves
# resident per SIMD, and 160 KiB of LDS per CU.
#
# Every constant here was pinned by measurement: the model was checked
# against hipModuleOccupancyMaxActiveBlocksPerMultiprocessor over 4552
# gfx950 kernels (Tensile assembly, rocRoller custom kernels, and synthetic
# kernels built to sweep AGPR and LDS usage) and agreed on all of them, and
# every constant was ablated to confirm no other value fits. The device
# properties corroborate them: maxSharedMemoryPerMultiProcessor = 163840 and
# regsPerBlock = 131072 = 512 x 64 lanes x 4 SIMDs.
#
# maxBlocksPerMultiProcessor is 32 on this part and never binds on its own:
# it equals this model's own ceiling of 8 waves/SIMD x 4 SIMDs with a
# single-wave workgroup, so no extra clamp is needed.
#
# gfx942 has 64 KiB of LDS per CU; this table is gfx950-specific.
_CU_SIMDS = 4
_VGPRS_PER_SIMD = 512
_VGPR_ALLOC_GRANULE = 8
_MAX_WAVES_PER_SIMD = 8
_LDS_BYTES_PER_CU = 160 * 1024


def rocRollerOccupancy(kernelMeta):
    """Workgroups resident per CU, from a kernel's static resource usage.

    This mirrors hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, which is
    what the rocRoller JIT path queries on the loaded kernel
    (rocroller/gemm.cpp:633-638) and feeds to origami's Stream-K grid
    selection. Recording it at capture time is what lets the rocRoller-off
    path reach the same grid without loading the code object.

    Returns 0 when the metadata does not carry enough to decide, which
    callers must read as "unknown" rather than as "zero workgroups".
    """
    if not isinstance(kernelMeta, dict):
        return 0
    waveSize = kernelMeta.get(".wavefront_size", 64)
    flat = kernelMeta.get(".max_flat_workgroup_size", 0)
    if not waveSize or not flat:
        return 0

    wavesPerGroup = -(-int(flat) // int(waveSize))
    if wavesPerGroup <= 0:
        return 0

    # .agpr_count is deliberately NOT added. On gfx90a and later LLVM reports
    # .vgpr_count as NumVGPRsForWavesPerEU, which is already the unified
    # arch-plus-acc allocation; .agpr_count is an informational subset of it.
    # Adding the two undercounts occupancy for every kernel that uses AGPRs --
    # measured against HIP, the sum disagrees on 45 of 4552 kernels and
    # .vgpr_count alone on none.
    vgprs = int(kernelMeta.get(".vgpr_count", 0))
    granule = _VGPR_ALLOC_GRANULE
    allocated = max(-(-vgprs // granule) * granule, granule)
    wavesPerSimd = min(_VGPRS_PER_SIMD // allocated, _MAX_WAVES_PER_SIMD)
    if wavesPerSimd <= 0:
        return 0
    byWaves = (wavesPerSimd * _CU_SIMDS) // wavesPerGroup

    lds = int(kernelMeta.get(".group_segment_fixed_size", 0))
    byLds = _LDS_BYTES_PER_CU // lds if lds > 0 else byWaves

    return max(1, min(byWaves, byLds))


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

    metadataArgs = kernelMeta[".args"]
    args = [_metadataArgToCustomArg(a, kernelName) for a in metadataArgs]
    if args and args[-1]["semantic"] == "ScaleBlockSizeA":
        last = metadataArgs[-1]
        padding = kernelMeta[".kernarg_segment_size"] - last[".offset"] - last[".size"]
        if padding:
            args[-1]["padding"] = padding

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

    from .RocRollerStreamK import is_streamk_metadata as _isRocRollerStreamK

    isRocRollerSK = _isRocRollerStreamK(args)

    mi = kernelConfig.get("MatrixInstruction", kernelConfig.get("MIBlock", [0,0,0,0]))
    if len(mi) >= 9 and "MIWaveTile" not in kernelConfig:
        wt = [mi[5], mi[6]]
        wg = [mi[7], mi[8]]
    else:
        wt = kernelConfig.get("MIWaveTile", [1, 1])
        wg = kernelConfig.get("MIWaveGroup", [1, 1])
    depthU = kernelConfig.get("DepthU", 0)
    macrotile = [mi[0] * wt[0] * wg[0], mi[1] * wt[1] * wg[1], depthU]

    # Fallback: parse macrotile from kernel name if computed values are zero.
    # rocRoller spells its workgroup tile WGT_<m>x<n>x<k> rather than
    # MT<m>x<n>x<k>, and a captured rocRoller kernel carries no
    # MatrixInstruction/DepthU for the computation above to use. The tile is
    # load-bearing for Stream-K -- it is the divisor in every hoisted kernarg
    # and it sizes the partial-tile workspace -- so a zero here must not
    # survive.
    if macrotile[0] == 0 or macrotile[1] == 0:
        m = re.search(r'(?:MT|WGT_)(\d+)x(\d+)x(\d+)', kernelName)
        if m:
            macrotile = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    if isRocRollerSK and (
        macrotile[0] <= 0 or macrotile[1] <= 0 or macrotile[2] <= 0
    ):
        raise RuntimeError(
            f"Custom kernel '{kernelName}' is a rocRoller Stream-K kernel but "
            f"its macrotile resolved to {macrotile}. Stream-K divides by the "
            f"macrotile to derive every hoisted kernarg, so it cannot be "
            f"inferred later; give the kernel a WGT_<m>x<n>x<k> name or "
            f"declare MatrixInstruction/MIWaveTile/DepthU in custom.config."
        )

    threads = [kernelMeta.get(".max_flat_workgroup_size", 256), 1, 1]

    hasNumWGArg = any(a.get("semantic") == "NumWorkGroups" for a in args)

    # DataParallel assigns whole tiles to persistent workgroups. Ordinary GEMM
    # uses strategy None and reaches the tile-grid branches below.
    if isRocRollerSK:
        # rocRoller decomposes the work itself from the hoisted kernargs and
        # reads only workgroup X, so the launch is a flat grid of exactly the
        # numWGs the kernargs were computed for. It must not go through
        # StreamKNoBatch, whose grid comes from Tensile's own Stream-K
        # decomposition -- a different partitioning from rocRoller's.
        grid = ["RocRollerStreamK", "One", "One"]
    elif isPersistentDataParallel(kernelConfig) or isp.get("PersistentLoopArgsVersion", 0) in (1, 2):
        grid = ["PersistentGrid", "One", "One"]
    elif isPersistent(kernelConfig):
        # sk.grid already counts every batch's tiles, so the launch must not
        # multiply by batch again.
        grid = ["StreamKNoBatch", "One", "One"]
    elif hasNumWGArg:
        # Version >= 1 kernels receive numWorkGroups as arg and decompose
        # the flat 1-D work-group index internally.
        grid = ["TilesXYBatchGSU", "One", "One"]
    elif _hasRocRollerWgmArgs(args):
        # rocRoller workgroup mapping consumes the flattened tile index in
        # workgroup X and does not read workgroup Y. A TilesX x TilesY launch
        # repeats the first row of tiles and never runs the remaining ids.
        # Presence of the libdivide kernargs is the signal, not the file name.
        grid = ["TilesXY", "One", "Batch"]
    else:
        # Version 0 kernels rely on hardware gridDim for tile decomposition,
        # so the grid must be multi-dimensional. Like generated kernels, they
        # take their GSU slices along gridDim.y.
        grid = ["TilesX", "TilesYGSU", "Batch"]

    # Workspace is left unset here and derived in
    # Solution._assignCustomKernelParameters, which sees the logic file's
    # ProblemType rather than the advisory copy in custom.config.
    #
    # A rocRoller Stream-K kernel is the exception: it stores partial tiles
    # unconditionally, and the auto-derivation only fires for a PartialsBuffer
    # GlobalSplitU algorithm, which a handwritten custom kernel does not have.
    # Leaving it "None" would hand the kernel a zero-byte workspace and fault
    # on its first partial-tile store, so declare it here.
    workspaceType = "None"
    workspaceSizePerElemC = 0
    if isRocRollerSK:
        workspaceType = "RocRollerStreamK"
        workspaceSizePerElemC = _computeTypeBytes(kernelConfig)

    return {
        "name": kernelName,
        "args": args,
        "macrotile": macrotile,
        "threads": threads,
        "grid": grid,
        "occupancy": rocRollerOccupancy(kernelMeta),
        "workspaceType": workspaceType,
        "workspaceSizePerElemC": workspaceSizePerElemC,
        "workspaceSizePerElemBias": 0,
    }

def getCustomKernelConfig(
    kernelName: str, internalSupportParams: dict, directory: str = None
) -> dict:
    """
    Retrieves and validates the configuration for a custom kernel.

    Args:
        kernelName: The name of the custom kernel.
        internalSupportParams: A dictionary of internal support parameters to be merged with the kernel configuration.
        directory: Optional directory where custom kernel files are located.
            Defaults to bundled package resources.

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
    # Missing metadata describes a prebuilt legacy payload, even when the
    # consuming solution was regenerated with DataParallel argument layout v1.
    kernelIsp.setdefault("PersistentLoopArgsVersion", 0)
    for key in internalSupportParams:
        if key not in kernelIsp:
            kernelIsp[key] = internalSupportParams[key]

    from .ExecutionPolicy import ALIASES, SELECTORS, normalize_execution_policy
    if SELECTORS.intersection(kernelConfig):
        kernelConfig = normalize_execution_policy(kernelConfig, regenerate=False)
    else:
        # A minimal custom.config can inherit its selectors from the consuming
        # logic file. Preserve explicit shared controls until that merge.
        for old, new in ALIASES.items():
            if old in kernelConfig:
                if new in kernelConfig and kernelConfig[new] != kernelConfig[old]:
                    raise ValueError(f"Conflicting {old} and {new}")
                kernelConfig[new] = kernelConfig.pop(old)

    # Compute a merged validParameters set locally; do NOT mutate the global
    # validParameters dict (that leaks state across calls and into unit tests
    # via the precomputed _expectedParamTypes cache in test_validateParameterTypes).
    mergedValid = {**validParameters, **newMIValidParameters}
    for k, v in kernelConfig.items():
        if k in _PASSTHROUGH_KEYS:
            continue
        if k in mergedValid:
            checkParametersAreValid((k, [v]), mergedValid)

    metadata_keys = [
        k for k in kernelConfig
        if k not in mergedValid
        and k not in _PASSTHROUGH_KEYS
    ]
    for k in metadata_keys:
        kernelConfig.pop(k)

    kernelConfig["KernelLanguage"] = "Assembly"

    if "CustomKernel" not in kernelConfig:
        # Only a kernel without an explicit CustomKernel block needs
        # amdhsa.kernels; the rocRoller kernels embed about 1 MB of it, so it is
        # parsed here and not for every kernel.
        fullYaml = _readEmbeddedYaml(kernelName, directory)
        kernelConfig["CustomKernel"] = _buildCustomKernelFromMetadata(kernelName, fullYaml, kernelConfig)

    kernelConfig["CustomKernel"]["name"] = kernelName
    kernelConfig["CustomKernel"].setdefault("workspaceType", "None")
    kernelConfig["CustomKernel"].setdefault("workspaceSizePerElemC", 0)
    kernelConfig["CustomKernel"].setdefault("workspaceSizePerElemBias", 0)
    # 0 means "not recorded". An explicitly written CustomKernel block keeps
    # whatever it declares; only rocRoller Stream-K reads this.
    kernelConfig["CustomKernel"].setdefault("occupancy", 0)
    kernelConfig["CustomKernelName"] = kernelName

    validateCustomPersistentArgs(kernelConfig)

    return kernelConfig


################################################################################
# Embedded metadata validation functions
################################################################################

_EXTERNAL_HINT = (
    "External kernels carry their full TensileLite-side interface in custom.config. "
    "To inject one from a TensileLite test YAML, run:\n"
    "  python -m tensilelite.AddCustomConfig <file.s> --yaml <test.yaml>\n"
)

_TENSILE_HINT = (
    "TensileLite-generated kernels only need InternalSupportParams.KernArgsVersion in "
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

def validateCustomKernelMetadata(name, directory=None):
    """Validates that a kernel has an embedded custom.config with required fields.

    TensileLite-generated kernels (no Source.Origin) only need
    InternalSupportParams.KernArgsVersion -- the only field
    `getCustomKernelConfig` actually requires at runtime. Their ProblemType and
    tuning state live in the consuming logic file or test YAML and are merged
    on top of custom.config there.

    External kernels (Source.Origin present) carry their full TensileLite-side
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
        kind = "External" if is_external else "TensileLite"
        return False, _missingMetadataMessage(kind, name, filepath, missing)

    return True, f"Kernel '{name}' metadata is valid"
