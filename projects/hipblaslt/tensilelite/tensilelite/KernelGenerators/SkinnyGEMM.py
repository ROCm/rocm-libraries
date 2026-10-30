# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Skinny MXFP8 GEMM kernels emitted by the tensilelite_codegen module.

The emitter produces a complete assembly file; this module describes how
Tensile launches it (arguments, tile shape, grid) and which problems it serves
(the Assert keys), so that it behaves like any other custom kernel.
"""

from copy import deepcopy
from typing import Any, Dict, List

from .Errors import GeneratorKernelRejected

# The knobs of tensilelite_codegen.SkinnyGEMMConfig, in binding order.
PARAMETERS = ("bm", "tn", "lanesK", "waves", "prefetch", "upIters",
              "partialSteps", "group", "numXcds", "grid3d", "useBeta")

# Limits of the emitted kernel's 32-bit index arithmetic.
MAX_EXTENT = 2**20
MAX_WIDE = 2**30
# Alignment of ldA and ldB: the operand extent is alignUp(k, 4) and the tail loads are dword masked.
# ldC and ldD need none: the kernel relies on unaligned buffer access.
LD_MULTIPLE = 4

_WAVE_SIZE = 64

_BOOL_PARAMETERS = ("grid3d", "useBeta")

_ARGS = (
    {"type": "address", "semantic": "AddressD"},
    {"type": "address", "semantic": "AddressA"},
    {"type": "address", "semantic": "AddressB"},
    {"type": "address", "semantic": "AddressMXScaleA"},
    {"type": "address", "semantic": "AddressMXScaleB"},
    # The split-K workspace: unused with a single split, passed as a null pointer.
    {"type": "uint64", "semantic": "ConstantZero"},
    # C is read by the useBeta variant and by the split-K reduce, never by the other variants.
    {"type": "address", "semantic": "AddressC"},
    {"type": "int32", "semantic": "StrideD0Bytes"},
    {"type": "int32", "semantic": "StrideA0Bytes"},
    {"type": "int32", "semantic": "StrideB0Bytes"},
    {"type": "int32", "semantic": "SizeFree0"},
    {"type": "int32", "semantic": "SizeFree1"},
    {"type": "int32", "semantic": "SizeSum"},
    # Split-K chunk (defaults to whole-K iterations when GSU is 1).
    {"type": "int32", "semantic": "Spk"},
    {"type": "int32", "semantic": "StrideC0Bytes"},
    {"type": "float32", "semantic": "Alpha"},
    {"type": "float32", "semantic": "Beta"},
    # The launched workgroup counts, filled by the host.
    {"type": "int32", "semantic": "NumWorkGroupsX"},
    {"type": "int32", "semantic": "NumWorkGroupsY"},
    {"type": "int32", "semantic": "NumWorkGroupsZ"},
)


def args() -> List[Dict[str, str]]:
    return [dict(arg) for arg in _ARGS]


def _codegen():
    # Imported lazily so that importing Tensile never requires the module.
    try:
        import tensilelite_codegen
    except ImportError as error:
        raise RuntimeError(
            "the SkinnyGEMM generator needs the tensilelite_codegen module, built with "
            "TENSILELITE_ENABLE_CODEGEN=ON and TENSILELITE_ENABLE_JIT_PYTHON=ON, "
            "on PYTHONPATH"
        ) from error
    return tensilelite_codegen


def _isInt(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validateParams(params: Any) -> Dict[str, Any]:
    """Checks the shape of generatorParams and returns a copy; the emitter checks the knob ranges."""
    if not isinstance(params, dict):
        raise ValueError(f"the SkinnyGEMM generatorParams must be a dict, got {type(params).__name__}")
    missing = [key for key in PARAMETERS if key not in params]
    if missing:
        raise ValueError(f"the SkinnyGEMM generatorParams is missing {', '.join(missing)}")
    extra = sorted(str(key) for key in params if key not in PARAMETERS)
    if extra:
        raise ValueError(f"the SkinnyGEMM generatorParams has unknown keys {', '.join(extra)}")
    for key in PARAMETERS:
        _validateParam(key, params[key])
    return {key: params[key] for key in PARAMETERS}


def _validateParam(key: str, value: Any) -> None:
    if key in _BOOL_PARAMETERS:
        if not isinstance(value, bool):
            raise ValueError(f"the SkinnyGEMM generatorParams {key} must be a bool, got {value!r}")
        return
    if not _isInt(value):
        raise ValueError(f"the SkinnyGEMM generatorParams {key} must be an int, got {value!r}")
    if key == "partialSteps" and value not in (0, 1):
        raise ValueError(f"the SkinnyGEMM generatorParams partialSteps must be 0 or 1, got {value}")
    if key == "upIters" and value < 0:
        raise ValueError(f"the SkinnyGEMM generatorParams upIters must not be negative, got {value}")


def _config(params: Dict[str, Any]):
    return _codegen().SkinnyGEMMConfig(**params)


def kernelName(params: Dict[str, Any]) -> str:
    params = validateParams(params)
    try:
        return _codegen().skinnyGEMMKernelName(_config(params))
    except ValueError as error:
        raise GeneratorKernelRejected(f"invalid SkinnyGEMM generatorParams {params}: {error}") from error


def emit(params: Dict[str, Any], arch: str) -> str:
    params = validateParams(params)
    try:
        return _codegen().emitSkinnyGEMM(_config(params), arch)
    except ValueError as error:
        raise GeneratorKernelRejected(f"invalid SkinnyGEMM generatorParams {params}: {error}") from error


def _tileShape(params: Dict[str, Any]) -> List[int]:
    lanesK = params["lanesK"]
    columns = params["waves"] * (_WAVE_SIZE // lanesK) * params["tn"]
    return [columns, params["bm"], _WAVE_SIZE * lanesK]


def _grid(params: Dict[str, Any]) -> List[str]:
    if params["grid3d"]:
        return ["TilesLong", "TilesShort", "Batch"]
    return ["ComputeUnits", "One", "One"]


def _asserts(params: Dict[str, Any]) -> Dict[str, Any]:
    kStep = _WAVE_SIZE * params["lanesK"]
    upIters = params["upIters"]
    minK = (upIters - 1) * kStep if upIters > 0 else 0
    maxK = min(upIters * kStep, MAX_EXTENT) if upIters > 0 else MAX_EXTENT
    asserts = {
        "AssertSizeEqual": {2: 1},
        "AssertStrideAEqual": {0: 1},
        "AssertStrideBEqual": {0: 1},
        "AssertStrideDEqual": {0: 1},
        "AssertStrideAMultiple": {1: LD_MULTIPLE},
        "AssertStrideBMultiple": {1: LD_MULTIPLE},
        "AssertSizeGreaterThan": {3: minK},
        "AssertSizeLessThan": {0: MAX_EXTENT + 1, 1: MAX_WIDE + 1, 3: maxK + 1},
        "AssertNoBias": True,
        "AssertNoActivation": True,
        "AssertNoScaleAB": True,
        "AssertNoScaleAlphaVec": True,
    }
    # Without useBeta the kernel ignores C and beta, so it only serves beta == 0; with it, C is read
    # with the vector widths of D.
    if params["useBeta"]:
        asserts["AssertStrideCEqual"] = {0: 1}
    else:
        asserts["AssertBetaZero"] = True
    # A partial last step handles any K; otherwise K must fill whole steps.
    if params["partialSteps"] == 0:
        asserts["AssertSummationElementMultiple"] = kStep
    return asserts


def kernelConfig(params: Dict[str, Any]) -> Dict[str, Any]:
    """The custom-kernel config of the emitted kernel, without the generator keys."""
    params = validateParams(params)
    name = kernelName(params)
    tile = _tileShape(params)
    config = {
        "KernelLanguage": "Assembly",
        "InternalSupportParams": {"KernArgsVersion": 0, "SupportUserGSU": False},
        "GlobalSplitU": 1,
        # The kernel issues no MFMA; these only describe it to Tensile.
        "MatrixInstruction": [16, 16, 16, 1],
        "EnableMatrixInstruction": True,
        "MIWaveTile": [1, 1],
        "StaggerU": 0,
        "WavefrontSize": _WAVE_SIZE,
        "CustomKernelName": name,
        "CustomKernel": {
            "name": name,
            "args": args(),
            "macrotile": tile,
            "threads": [_WAVE_SIZE * params["waves"], 1, 1],
            "grid": _grid(params),
            "workspaceType": "None",
            "workspaceSizePerElemC": 0,
            "workspaceSizePerElemBias": 0,
        },
    }
    config.update(deepcopy(_asserts(params)))
    return config
