# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Common helpers for GEMM-family dispatcher cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

from ...core.arch import ArchTarget
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


def selector_matches(req: GemmRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = normalize_selector(req.algorithm)
    spec_id = normalize_selector(req.spec_id)
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"
