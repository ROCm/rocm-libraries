# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Convolution backward-data (dgrad) implicit-GEMM dispatcher family.

Mirrors :mod:`rocke.dispatch.families.conv` for the dgrad direction.  All
convolutions — stride=1 and strided — share a single ABI with eight kernel
parameters::

    (A=dY, B=W, D=dX, A_bytes, B_bytes, D_bytes, sub_gemm_buf, num_sub_gemms)

``sub_gemm_buf`` carries the tilde-decomposition record(s) precomputed by
:func:`~rocke.instances.common.conv_implicit_gemm_dgrad.pack_sub_gemm_buffer`.
For stride=1 this holds exactly one record and the in-kernel binary search
degenerates trivially.

Grid layout
-----------
The grid is always 1-D in X (``flat_tiles``):

    flat_tiles = sub_gemms[-1].block_end
    grid = (flat_tiles, 1, split_k)

Candidates (fp16 dgrad)
-----------------------
* ``cdna_mem_64x64``      — 64x64x32, mfma16x16x16, mem (gfx942 + gfx950).
* ``cdna_hiperf_64x64``   — 64x64x64, mfma32x32x8,  mem (gfx942 + gfx950);
  larger atom, higher occupancy at wide C/K.
* ``cdna_hiperf_gfx950``  — 64x64x64, mfma32x32x16, mem (gfx950 only;
  gfx942 lacks the 32x32x16 f16 atom).
* ``rdna_wmma_32x32``     — 32x32x16, wmma16x16x16,  mem (RDNA wave32).

``auto`` ranks: gfx950_hiperf (priority 10) → cdna_hiperf (20) →
cdna_mem (30) → rdna_wmma (10 on RDNA).
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from typing import Callable, List, Sequence, Tuple

from rocke.core.arch import ArchTarget
from rocke.helpers.manifest import conv_args_signature
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    SubGemmParams,
    enumerate_sub_gemms,
    is_valid_dgrad_spec,
    pack_sub_gemm_buffer,
)
from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)

_FAMILY = "conv_implicit_gemm_dgrad"
_ALGORITHM = "implicit_gemm_dgrad"
CONV_DGRAD_ABI_VERSION = "hipkg-conv-implicit-gemm-dgrad/v1"

# Fixed extended signature: base (A,B,D,bytes) + sub_gemm_buf + num_sub_gemms.
# The dtype placeholder 'f16' is fine here — only the ptr width (8 bytes) and
# i32 sizes matter for ABI packing; the actual element type is in the kernel IR.
_DGRAD_SIGNATURE_FP16 = conv_args_signature("fp16") + [
    {"name": "sub_gemm_buf", "type": "ptr<i32, global>", "size_bytes": 8},
    {"name": "num_sub_gemms", "type": "i32", "size_bytes": 4},
]
_DGRAD_SIGNATURE_BF16 = conv_args_signature("bf16") + [
    {"name": "sub_gemm_buf", "type": "ptr<i32, global>", "size_bytes": 8},
    {"name": "num_sub_gemms", "type": "i32", "size_bytes": 4},
]
_DGRAD_SIGNATURE_FP32 = conv_args_signature("fp32") + [
    {"name": "sub_gemm_buf", "type": "ptr<i32, global>", "size_bytes": 8},
    {"name": "num_sub_gemms", "type": "i32", "size_bytes": 4},
]


@dataclass(frozen=True)
class ConvDgradRequest(OperatorRequest):
    """Normalized 2-D backward-data convolution request (NHWC implicit-GEMM)."""

    N: int
    C: int
    K: int
    Hi: int
    Wi: int
    Y: int
    X: int
    arch: str
    G: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    op: str = "conv_dgrad"
    dtype: str = "fp16"
    layout: str = "NHWC"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = _dgrad_dtype(self.dtype)
        d["layout"] = self.layout.upper()
        return d


def _dgrad_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d in ("fp16", "f16"):
        return "fp16"
    if d in ("fp32", "f32"):
        return "fp32"
    return d


def _problem(req: ConvDgradRequest) -> ConvProblem:
    return ConvProblem(
        N=int(req.N),
        Hi=int(req.Hi),
        Wi=int(req.Wi),
        C=int(req.C),
        K=int(req.K),
        Y=int(req.Y),
        X=int(req.X),
        sH=int(req.stride_h),
        sW=int(req.stride_w),
        pH=int(req.pad_h),
        pW=int(req.pad_w),
        dH=int(req.dilation_h),
        dW=int(req.dilation_w),
    )


def _request_errors(req: OperatorRequest) -> List[str]:
    if not isinstance(req, ConvDgradRequest):
        return [f"expected ConvDgradRequest, got {type(req).__name__}"]
    errors: List[str] = []
    if req.op != "conv_dgrad":
        errors.append(f"unsupported op {req.op!r}")
    for f in ("N", "C", "K", "Hi", "Wi", "Y", "X"):
        if int(getattr(req, f)) <= 0:
            errors.append(f"{f} must be positive")
    if int(req.G) != 1:
        errors.append("only groups=1 (G=1) dgrad is implemented")
    if _dgrad_dtype(req.dtype) not in ("fp16", "bf16", "fp32"):
        errors.append(f"unsupported dtype {req.dtype!r}; fp16/bf16/fp32 only")
    if req.layout.upper() != "NHWC":
        errors.append(f"unsupported layout {req.layout!r}; NHWC only")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    if errors:
        return errors
    p = _problem(req)
    if p.Ho <= 0 or p.Wo <= 0:
        errors.append(
            f"degenerate output spatial dims Ho={p.Ho} Wo={p.Wo} "
            "(filter larger than padded input)"
        )
    return errors


def _arch_family_supported(req: ConvDgradRequest, arch_family: str) -> Tuple[bool, str]:
    target = ArchTarget.from_gfx(req.arch)
    if target.family != arch_family:
        return False, (
            f"{arch_family!r}-family candidate does not support "
            f"{target.family!r}-family arch {req.arch}"
        )
    return True, "ok"


def _selector_matches(
    req: ConvDgradRequest, candidate: KernelCandidate
) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


_CDNA_GFX950_ONLY = ("gfx950",)
_CDNA_ALL = ("gfx90a", "gfx942", "gfx950")
_RDNA_WMMA = ("gfx1151", "gfx1201")

# ---- per-candidate spec factories ----------------------------------------


def _spec_cdna_mem(req: ConvDgradRequest, name: str) -> DgradConvSpec:
    """64x64x32, mfma16x16x16 — works on gfx942 and gfx950."""
    dtype = _dgrad_dtype(req.dtype)
    return DgradConvSpec(
        problem=_problem(req),
        name=name,
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        pipeline="mem",
        epilogue="default",
    )


def _spec_cdna_hiperf(req: ConvDgradRequest, name: str) -> DgradConvSpec:
    """64x64x64, mfma32x32x8 — gfx942 and gfx950; wider atom, higher throughput."""
    dtype = _dgrad_dtype(req.dtype)
    return DgradConvSpec(
        problem=_problem(req),
        name=name,
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=8,
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        pipeline="mem",
        epilogue="default",
    )


def _spec_cdna_hiperf_gfx950(req: ConvDgradRequest, name: str) -> DgradConvSpec:
    """64x64x64, mfma32x32x16 — gfx950 only (atom not present on gfx942)."""
    dtype = _dgrad_dtype(req.dtype)
    return DgradConvSpec(
        problem=_problem(req),
        name=name,
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        pipeline="mem",
        epilogue="default",
    )


def _spec_cdna_fp32(req: ConvDgradRequest, name: str) -> DgradConvSpec:
    """64x64x16, mfma16x16x4 — fp32 CDNA path (gfx942/gfx950)."""
    dtype = _dgrad_dtype(req.dtype)
    return DgradConvSpec(
        problem=_problem(req),
        name=name,
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=64,
        tile_n=64,
        tile_k=16,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=4,
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        pipeline="mem",
        epilogue="default",
    )


def _spec_rdna_wmma(req: ConvDgradRequest, name: str) -> DgradConvSpec:
    """32x32x16, wmma16x16x16 — RDNA wave32 targets (gfx1151, gfx1201)."""
    dtype = _dgrad_dtype(req.dtype)
    return DgradConvSpec(
        problem=_problem(req),
        name=name,
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=32,
        tile_n=32,
        tile_k=16,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        pipeline="mem",
        epilogue="default",
    )


# ---- grid and signature ---------------------------------------------------


def _grid(spec: DgradConvSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    """1-D flat-tile grid over all sub-GEMMs, Z = split_k."""
    sub_gemms = spec.compute_sub_gemms()
    flat_tiles = sub_gemms[-1].block_end
    return (flat_tiles, 1, max(spec.split_k, 1))


def _signature(spec: DgradConvSpec) -> Sequence[dict]:
    dtype = spec.data.dtype_a
    if dtype == "bf16":
        return _DGRAD_SIGNATURE_BF16
    if dtype == "fp32":
        return _DGRAD_SIGNATURE_FP32
    return _DGRAD_SIGNATURE_FP16


# ---- candidate factory ---------------------------------------------------


def _make_candidate(
    *,
    name: str,
    spec_id: str,
    priority: int,
    spec_fn: Callable[[ConvDgradRequest, str], DgradConvSpec],
    arch_family: str,
    arches: Tuple[str, ...],
    dtype_filter: tuple = ("fp16", "bf16"),
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvDgradRequest)
        ok, why = _arch_family_supported(req, arch_family)
        if not ok:
            return False, why
        if _dgrad_dtype(req.dtype) not in dtype_filter:
            return False, (
                f"candidate {name!r} only supports dtypes {dtype_filter}, "
                f"got {req.dtype!r}"
            )
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = spec_fn(req, name)
        return is_valid_dgrad_spec(spec, arch=req.arch)

    def select(req: OperatorRequest) -> DgradConvSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvDgradRequest)
        return spec_fn(req, name)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id=spec_id,
        abi_version=CONV_DGRAD_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=arches, dtypes=dtype_filter),
        _supports=support,
        select_spec=select,
        signature=_signature,
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


CONV_DGRAD_REGISTRY = CandidateRegistry(_FAMILY)
CONV_DGRAD_REGISTRY.extend(
    (
        _make_candidate(
            name="conv_dgrad_igemm_cdna_hiperf_gfx950",
            spec_id="cdna_hiperf_gfx950_64x64",
            priority=10,
            spec_fn=_spec_cdna_hiperf_gfx950,
            arch_family="cdna",
            arches=_CDNA_GFX950_ONLY,
            dtype_filter=("fp16", "bf16"),
        ),
        _make_candidate(
            name="conv_dgrad_igemm_cdna_hiperf",
            spec_id="cdna_hiperf_64x64",
            priority=20,
            spec_fn=_spec_cdna_hiperf,
            arch_family="cdna",
            arches=_CDNA_ALL,
            dtype_filter=("fp16", "bf16"),
        ),
        _make_candidate(
            name="conv_dgrad_igemm_cdna_mem",
            spec_id="cdna_mem_64x64",
            priority=30,
            spec_fn=_spec_cdna_mem,
            arch_family="cdna",
            arches=_CDNA_ALL,
            dtype_filter=("fp16", "bf16"),
        ),
        _make_candidate(
            name="conv_dgrad_igemm_cdna_fp32",
            spec_id="cdna_fp32_64x64",
            priority=10,
            spec_fn=_spec_cdna_fp32,
            arch_family="cdna",
            arches=_CDNA_ALL,
            dtype_filter=("fp32",),
        ),
        _make_candidate(
            name="conv_dgrad_igemm_rdna_wmma",
            spec_id="rdna_wmma_32x32",
            priority=10,
            spec_fn=_spec_rdna_wmma,
            arch_family="rdna",
            arches=_RDNA_WMMA,
            dtype_filter=("fp16", "bf16"),
        ),
    )
)


def conv_dgrad_candidates() -> Tuple[KernelCandidate, ...]:
    return CONV_DGRAD_REGISTRY.candidates()


def _kernel_id(
    req: ConvDgradRequest,
    candidate: KernelCandidate,
    spec: DgradConvSpec,
) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash({"kernel_name": spec.kernel_name()}, n=16)
    return KernelId(
        op="conv_dgrad",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def conv_dgrad_sweep_space(req: OperatorRequest) -> Sequence[DgradConvSpec]:
    if _request_errors(req):
        return ()
    specs = []
    seen: set = set()
    for candidate in CONV_DGRAD_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        h = spec.kernel_name()
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def dispatch_conv_dgrad(
    req: ConvDgradRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered dgrad implicit-GEMM candidate for ``req``."""
    candidate = CONV_DGRAD_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, spec)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=(
            f"selected {candidate.name} for {req.dtype} dgrad on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
