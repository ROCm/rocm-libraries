# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Convolution (forward implicit-GEMM) dispatcher family.

Worked implementation mirroring :mod:`rocke.dispatch.gemm.bf16_rcr`. Backed by
:mod:`rocke.instances.common.conv_implicit_gemm` (the NHWC x KRSC -> NHWK
forward implicit-GEMM emitter).

The implicit-GEMM problem reduces to a GEMM of
``M = N*Ho*Wo, N_gemm = K, K_gemm = R*S*C`` (see ``ConvProblem``). The support
predicate therefore composes:

* family request errors (positive dims, known arch, Ho/Wo > 0),
* the arch-family gate (CDNA candidates off RDNA and vice versa),
* the instance's ``is_valid_spec`` (tile/atom/arch + WMMA narrowing), and
* a no-padding GEMM-divisibility check on the DERIVED GEMM dims (these
  shape-generic conv kernels do not pad, mirroring the GEMM family).

Candidate set (f16 forward conv):

* ``cdna_cshuffle``   64x64x64, 32x32x16 atom, cshuffle epilogue (gfx950 only --
  gfx942 has no 32x32x16 f16 atom),
* ``cdna_mem``        64x64x32, 16x16x16 atom, mem pipeline (gfx942 + gfx950),
* ``rdna_wmma``       16x16x16 atom, mem pipeline (RDNA wave32).

``auto`` ranks by priority: cshuffle (10) before mem (20); the arch-family +
atom + divisibility gates decide which are actually selectable for a request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence, Tuple

from ...core.arch import ArchTarget
from ...instances.common.conv_implicit_gemm import (
    ConvProblem,
    ImplicitGemmConvSpec,
    is_valid_spec as _conv_is_valid_spec,
)
from ..core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)

_FAMILY = "conv_implicit_gemm"
_ALGORITHM = "implicit_gemm_fwd"
CONV_ABI_VERSION = "hipkg-conv-implicit-gemm/v1"


@dataclass(frozen=True)
class ConvRequest(OperatorRequest):
    """Normalized 2D forward convolution request (NHWC implicit-GEMM)."""

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
    op: str = "conv"
    dtype: str = "fp16"
    layout: str = "NHWC"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = _conv_dtype(self.dtype)
        d["layout"] = self.layout.upper()
        return d


def _conv_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d == "fp16":
        return "f16"
    return d


def _problem(req: ConvRequest) -> ConvProblem:
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


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, ConvRequest):
        return [f"expected ConvRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "conv":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("N", "C", "K", "Hi", "Wi", "Y", "X"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if int(req.G) != 1:
        errors.append("only groups=1 (G=1) forward conv is implemented")
    if _conv_dtype(req.dtype) != "f16":
        errors.append(f"unsupported dtype {req.dtype!r}; f16 only")
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


def _arch_family_supported(req: ConvRequest, arch_family: str) -> Tuple[bool, str]:
    target = ArchTarget.from_gfx(req.arch)
    if target.family != arch_family:
        return False, (
            f"{arch_family!r}-family candidate does not support "
            f"{target.family!r}-family arch {req.arch}"
        )
    return True, "ok"


def _selector_matches(req: ConvRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


# ---- per-candidate spec factories -----------------------------------------


def _spec_cdna_cshuffle(req: ConvRequest, name: str) -> ImplicitGemmConvSpec:
    return ImplicitGemmConvSpec(
        problem=_problem(req),
        name=name,
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
        epilogue="cshuffle",
    )


def _spec_cdna_mem(req: ConvRequest, name: str) -> ImplicitGemmConvSpec:
    return ImplicitGemmConvSpec(
        problem=_problem(req),
        name=name,
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


def _spec_rdna_wmma(req: ConvRequest, name: str) -> ImplicitGemmConvSpec:
    return ImplicitGemmConvSpec(
        problem=_problem(req),
        name=name,
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


def _gemm_dims_divide(req: ConvRequest, spec: ImplicitGemmConvSpec) -> Tuple[bool, str]:
    """No-padding divisibility on the DERIVED implicit-GEMM dims.

    These shape-generic conv kernels do not pad, so the implicit-GEMM
    ``(M, N_gemm, K_gemm)`` must be exactly divisible by ``(tile_m, tile_n,
    tile_k)`` -- the same gate the GEMM family applies, on the conv-derived dims.
    """
    p = spec.problem
    checks = (
        ("M", p.M, spec.tile_m),
        ("N", p.N_gemm, spec.tile_n),
        ("K", p.K_gemm, spec.tile_k),
    )
    for dim_name, dim, tile in checks:
        if dim % tile:
            return False, (
                f"implicit-GEMM {dim_name}={dim} not divisible by tile={tile}"
            )
    return True, "ok"


def _make_candidate(
    *,
    name: str,
    spec_id: str,
    priority: int,
    spec_fn: Callable[[ConvRequest, str], ImplicitGemmConvSpec],
    arch_family: str,
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, ConvRequest)
        ok, why = _arch_family_supported(req, arch_family)
        if not ok:
            return False, why
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        spec = spec_fn(req, name)
        ok, why = _conv_is_valid_spec(spec, arch=req.arch)
        if not ok:
            return False, why
        return _gemm_dims_divide(req, spec)

    def select(req: OperatorRequest) -> ImplicitGemmConvSpec:
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, ConvRequest)
        return spec_fn(req, name)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id=spec_id,
        abi_version=CONV_ABI_VERSION,
        priority=priority,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


def _grid(spec: ImplicitGemmConvSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    p = spec.problem
    gm = (p.M + spec.tile_m - 1) // spec.tile_m
    gn = (p.N_gemm + spec.tile_n - 1) // spec.tile_n
    # grid_order "NM": x=n-tiles, y=m-tiles (matches the shipped conv manifest).
    return (gn, gm, 1)


CONV_REGISTRY = CandidateRegistry(_FAMILY)
CONV_REGISTRY.extend(
    (
        _make_candidate(
            name="conv_igemm_cdna_cshuffle",
            spec_id="cdna_cshuffle_64x64",
            priority=10,
            spec_fn=_spec_cdna_cshuffle,
            arch_family="cdna",
        ),
        _make_candidate(
            name="conv_igemm_cdna_mem",
            spec_id="cdna_mem_64x64",
            priority=20,
            spec_fn=_spec_cdna_mem,
            arch_family="cdna",
        ),
        _make_candidate(
            name="conv_igemm_rdna_wmma",
            spec_id="rdna_wmma_32x32",
            priority=10,
            spec_fn=_spec_rdna_wmma,
            arch_family="rdna",
        ),
    )
)


def conv_candidates() -> Tuple[KernelCandidate, ...]:
    return CONV_REGISTRY.candidates()


def _kernel_id(
    req: ConvRequest, candidate: KernelCandidate, spec: ImplicitGemmConvSpec
) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    # ImplicitGemmConvSpec is nested (ConvProblem, acc_epilogue); hash its name.
    spec_hash = stable_json_hash({"kernel_name": spec.kernel_name()}, n=16)
    return KernelId(
        op="conv",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def conv_sweep_space(req: OperatorRequest) -> Sequence[ImplicitGemmConvSpec]:
    if _request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in CONV_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        h = spec.kernel_name()
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def dispatch_conv(req: ConvRequest, *, ranker: Ranker | None = None) -> DispatchResult:
    """Select a registered forward implicit-GEMM conv candidate for ``req``."""
    candidate = CONV_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} for f16 forward conv on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
