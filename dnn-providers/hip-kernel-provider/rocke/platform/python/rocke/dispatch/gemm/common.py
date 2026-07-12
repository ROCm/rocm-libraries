# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Common helpers for GEMM-family dispatcher cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Tuple

from ...core.arch import ArchTarget
from ...helpers.split_k import select_split_k
from ...instances.common.gemm_universal import UniversalGemmSpec
from ..core import KernelCandidate, OperatorRequest


@dataclass(frozen=True)
class GemmRequest(OperatorRequest):
    """Normalized GEMM request shared by GEMM-family dispatchers.

    Phase 1 uses this for FP16 RCR UniversalGemm. Later GEMM-family cases can
    either extend this type or define more specialized request types beside
    their case module.
    """

    M: int
    N: int
    K: int
    arch: str
    op: str = "gemm"
    dtype: str = "fp16"
    layout: str = "RCR"
    trans_a: bool = False
    trans_b: bool = True
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = normalize_dtype(self.dtype)
        d["layout"] = self.layout.upper()
        d["algorithm"] = normalize_selector(self.algorithm)
        d["spec_id"] = normalize_selector(self.spec_id)
        return d


def normalize_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d in ("f16", "half"):
        return "fp16"
    return d


def normalize_selector(value: str) -> str:
    return value.strip().lower()


def basic_gemm_request_errors(req: OperatorRequest) -> list[str]:
    """Common shape/op checks shared by GEMM-family dispatchers."""
    if not isinstance(req, GemmRequest):
        return [f"expected GemmRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "gemm":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("M", "N", "K"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def rcr_request_errors(req: OperatorRequest, *, dtype: str) -> list[str]:
    """Shared RCR-layout request validation parametrized by element dtype.

    Used by every RCR UniversalGemm family module (fp16, bf16, ...) so the
    layout/transpose checks and the single-dtype gate live in one place. The
    family passes the canonical dtype it implements (e.g. ``"fp16"``, ``"bf16"``).
    """
    errors = basic_gemm_request_errors(req)
    if errors:
        return errors
    assert isinstance(req, GemmRequest)
    if normalize_dtype(req.dtype) != dtype:
        errors.append(
            f"unsupported dtype {req.dtype!r}; this family supports {dtype} only"
        )
    if req.layout.upper() != "RCR":
        errors.append(f"unsupported layout {req.layout!r}; RCR only")
    if req.trans_a or not req.trans_b:
        errors.append("RCR expects A row-major and B logically transposed")
    return errors


def arch_family_supported(req: GemmRequest, arch_family: str) -> Tuple[bool, str]:
    """Gate a GEMM candidate to its intended arch family (``cdna`` or ``rdna``).

    ``ArchTarget.family`` is the SSOT split (``cdna`` for gfx90a/gfx942/gfx950,
    ``rdna`` for gfx11xx/gfx12xx). Candidates are tuned for one micro-arch family
    only; this predicate keeps an RDNA candidate from matching a CDNA request
    just because the rebuilt spec happens to satisfy the generic config checks
    (and vice versa).
    """
    target = ArchTarget.from_gfx(req.arch)
    if target.family != arch_family:
        return False, (
            f"{arch_family!r}-family candidate does not support "
            f"{target.family!r}-family arch {req.arch}"
        )
    return True, "ok"


def apply_split_k(req: GemmRequest, spec: UniversalGemmSpec) -> UniversalGemmSpec:
    """Return ``spec`` with a split-K degree chosen for ``req`` on its arch.

    Split-K is engaged only for skinny / tall-N decode shapes whose base grid
    leaves the CU-rich CDNA device idle, and only on the MFMA (CDNA) family the
    kernel's atomic-add epilogue supports. For any shape that already fills the
    device -- and on RDNA, where the split-K epilogue is not wired -- the chosen
    degree is ``1`` and this returns the spec **unchanged** (so the default /
    square-GEMM path stays byte-identical). The ``ROCKE_GEMM_SPLIT_K`` env flag
    overrides the heuristic (see :mod:`rocke.helpers.split_k`).
    """
    target = ArchTarget.from_gfx(req.arch)
    if target.family != "cdna":
        return spec
    t = spec.tile
    decision = select_split_k(
        M=req.M,
        N=req.N,
        K=req.K,
        tile_m=t.tile_m,
        tile_n=t.tile_n,
        tile_k=t.tile_k,
    )
    if decision.split_k <= 1:
        return spec
    return replace(spec, trait=replace(spec.trait, split_k=decision.split_k))


def selector_matches(req: GemmRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = normalize_selector(req.algorithm)
    spec_id = normalize_selector(req.spec_id)
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"
