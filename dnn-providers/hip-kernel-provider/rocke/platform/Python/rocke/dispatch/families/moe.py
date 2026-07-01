# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused MoE dispatcher family (single-launch mega-kernel).

Worked implementation mirroring :mod:`rocke.dispatch.gemm.bf16_rcr`, backed by
:mod:`rocke.instances.common.moe_fused_mega` (f16/bf16) and
:mod:`rocke.instances.common.moe_fused_mega_fp8` (fp8 e4m3 block-scale).

SCOPE -- what this dispatcher decides
-------------------------------------
The fused-MoE mega-kernel has a STATIC tile geometry (locked by BUILD_SPEC:
``tile_m=16, tile_n_inter=256, tile_k_gu=32``); the MoE problem dims
(num_tokens / hidden / intermediate / num_experts / top_k) are RUNTIME kernel
args, not selection knobs. The load-bearing dispatch decision is therefore the
*element path*: the f16/bf16 mega-kernel vs the fp8 block-scale mega-kernel.

The candidate set is two element-path kernels, selected by request dtype:

* ``mega_f16``  : f16/bf16 mega-kernel (atom 16x16x32),
* ``mega_fp8``  : fp8 e4m3 block-scale mega-kernel (hero atom 16x16x128).

Arch coverage: MoE is CDNA-only (the mega-kernel atoms are MFMA), and both
shipped configs are gfx950-tuned -- the f16 path's 16x16x32 atom does not exist
on gfx942, and the fp8 path's 16x16x128 hero atom is a gfx950-only scaled
intrinsic (it is not even in the generic MMA catalog; the instance builder skips
the catalog guard for ``atom.k==128``). So the support predicate gates to the
CDNA family AND to gfx950, and validates the f16 atom against the per-arch MMA
catalog where it is expressible.

DEFERRED -- the MoE component pipeline
--------------------------------------
The non-fused MoE component kernels (``moe_sorting``, ``moe_gemm_fused``,
``moe_smoothquant``) and the multi-launch ``fused_moe`` path are separate
algorithms; only the single-launch mega-kernel is dispatched here. Adding them
is a candidate-registration follow-on (same recipe).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, Tuple

from ...core.arch import ArchTarget
from ...instances.common.moe_fused_mega import FusedMegaKernelSpec
from ...instances.common.moe_fused_mega_fp8 import FusedMegaKernelSpecFp8
from ..core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)

_FAMILY = "moe_fused_mega"
MOE_ABI_VERSION = "hipkg-moe-fused-mega/v1"

# Both mega configs are gfx950-tuned (see module docstring).
_SUPPORTED_ARCHES = ("gfx950",)


@dataclass(frozen=True)
class MoeRequest(OperatorRequest):
    """Normalized fused mixture-of-experts request."""

    num_tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    arch: str
    op: str = "moe"
    dtype: str = "fp16"
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = _moe_dtype(self.dtype)
        return d


def _moe_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d in ("f16", "half"):
        return "fp16"
    if d in ("fp8", "f8", "fp8e4m3", "e4m3"):
        return "fp8e4m3"
    return d


_F16_DTYPES = ("fp16", "bf16")
_FP8_DTYPES = ("fp8e4m3",)


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, MoeRequest):
        return [f"expected MoeRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "moe":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("num_tokens", "hidden", "intermediate", "num_experts", "top_k"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if int(req.top_k) > int(req.num_experts):
        errors.append("top_k must be <= num_experts")
    dt = _moe_dtype(req.dtype)
    if dt not in _F16_DTYPES + _FP8_DTYPES:
        errors.append(f"unsupported dtype {req.dtype!r}; one of fp16/bf16/fp8")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def _arch_supported(req: MoeRequest) -> Tuple[bool, str]:
    target = ArchTarget.from_gfx(req.arch)
    if target.family != "cdna":
        return False, (
            f"MoE mega-kernel is CDNA-only; {req.arch} is {target.family}-family"
        )
    if req.arch not in _SUPPORTED_ARCHES:
        return False, (
            f"MoE mega-kernel is tuned for {_SUPPORTED_ARCHES} only (got {req.arch}); "
            "the 16x16x32 / 16x16x128 atoms are gfx950-specific"
        )
    return True, "ok"


def _selector_matches(req: MoeRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


def _spec_f16(req: MoeRequest):
    dt = _moe_dtype(req.dtype)
    return FusedMegaKernelSpec(name=f"moe_{dt}", dtype=dt)


def _spec_fp8(req: MoeRequest):
    return FusedMegaKernelSpecFp8(name="moe_fp8")


def _make_candidate(*, name, spec_id, dtypes, spec_fn, priority) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, MoeRequest)
        if _moe_dtype(req.dtype) not in dtypes:
            return False, (
                f"candidate {name} supports {dtypes}, not {_moe_dtype(req.dtype)!r}"
            )
        ok, why = _arch_supported(req)
        if not ok:
            return False, why
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        # f16 path: validate the 16x16x32 atom against the per-arch catalog
        # (gfx942 lacks it -> rejected even though arch-family is CDNA).
        if _moe_dtype(req.dtype) in _F16_DTYPES:
            target = ArchTarget.from_gfx(req.arch)
            dt = _moe_dtype(req.dtype)
            if not target.mma.has_shape(
                family="mma",
                a_dtype=dt,
                b_dtype=dt,
                c_dtype="fp32",
                m=16,
                n=16,
                k=32,
            ):
                return False, f"unsupported {dt} 16x16x32 MoE atom on {req.arch}"
        return True, "ok"

    def select(req: OperatorRequest):
        ok, why = support(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, MoeRequest)
        return spec_fn(req)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=spec_id,
        spec_id=spec_id,
        abi_version=MOE_ABI_VERSION,
        priority=priority,
        supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),  # grid is runtime (num_m_blocks, inter)
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if support(req)[0] else (),
    )
    return candidate


MOE_REGISTRY = CandidateRegistry(_FAMILY)
MOE_REGISTRY.extend(
    (
        _make_candidate(
            name="moe_fused_mega_f16",
            spec_id="mega_f16",
            dtypes=_F16_DTYPES,
            spec_fn=_spec_f16,
            priority=10,
        ),
        _make_candidate(
            name="moe_fused_mega_fp8",
            spec_id="mega_fp8",
            dtypes=_FP8_DTYPES,
            spec_fn=_spec_fp8,
            priority=10,
        ),
    )
)


def moe_candidates() -> Tuple[KernelCandidate, ...]:
    return MOE_REGISTRY.candidates()


def _struct(spec) -> dict:
    """The selection-parity structural identity for a MoE mega spec."""
    if isinstance(spec, FusedMegaKernelSpecFp8):
        atom_k = spec.gate_up_k
        path = "fp8"
    else:
        atom_k = spec.warp_tile_k
        path = "f16"
    return {
        "path": path,
        "tile_m": spec.tile_m,
        "tile_n_inter": spec.tile_n_inter,
        "tile_k_gu": spec.tile_k_gu,
        "atom_k": atom_k,
        "block_size": int(spec.block_size),
    }


def _kernel_id(req: MoeRequest, candidate: KernelCandidate, spec) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(_struct(spec), n=16)
    return KernelId(
        op="moe",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def moe_sweep_space(req: OperatorRequest) -> Sequence[object]:
    if _request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in MOE_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        h = stable_json_hash(_struct(spec), n=16)
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def dispatch_moe(req: MoeRequest, *, ranker: Ranker | None = None) -> DispatchResult:
    """Select the fused-MoE mega-kernel element path for ``req``."""
    candidate = MOE_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} for {req.dtype} fused MoE on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
