# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Arch-neutral vocabulary for the GDN decode dispatcher family.

Holds everything that describes a *problem* rather than a chip: the request
dataclass, the family identity, the dimension vocabulary the registry may gate
on, and the shared request/selector validation. Arch-specific capability and
tile selection live in the per-arch modules (``gfx950.py``), so adding another
arch does not touch this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from kernels.gfx950.gdn_decode import (
    # Re-exported, never redeclared. The kernel owns what it covers; dispatch's
    # job is to state that coverage, not to restate it -- a copy drifts in the
    # direction that fails silently, rejecting a shape the kernel has since
    # learned to run.
    GDN_DTYPES,
)
from rocke.dispatch.core import (
    OperatorRequest,
    selector_matches as _shared_selector_matches,
)

FAMILY = "gdn_decode"

# Bumped when the kernel's argument contract changes, so a cached compile from
# an older layout can never be reused against a newer launcher.
GDN_ABI_VERSION = "rocke-gdn-decode/v1"

_DTYPE_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "f16": "f16",
    "fp16": "f16",
    "float16": "f16",
}


def normalize_dtype(dtype: str) -> str:
    d = str(dtype).strip().lower()
    return _DTYPE_ALIASES.get(d, d)


@dataclass(frozen=True)
class GdnDecodeRequest(OperatorRequest):
    """One gated-delta-rule single-token decode step.

    ``batch`` is the number of *active* sequences this step. Tile selection is
    keyed on ``batch * num_v_heads`` -- the "work" -- not on batch alone: every
    workgroup does identical work and the grid is their product, so the two are
    interchangeable. Keying on the product is what keeps a tensor-parallel
    deployment, which divides ``num_v_heads`` across ranks, on the right tile.

    ``gate_kind`` selects the forget-gate granularity: ``"gdn"`` (one scalar
    decay per head) or ``"kda"`` (a per-channel decay). It reaches the spec and
    therefore the kernel name, so the two never share a compile-cache entry.

    ``seq_len`` is not a field. This family is decode-only -- one token per
    sequence -- and a request carrying any other sequence length would be a
    different operator, so it is rejected rather than silently accepted.
    """

    batch: int
    arch: str
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    op: str = "gdn_decode"
    dtype: str = "bf16"
    state_dtype: str = "bf16"
    use_qk_l2norm: bool = True
    algorithm: str = "auto"
    gate_kind: str = "gdn"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = normalize_dtype(self.dtype)
        d["state_dtype"] = normalize_dtype(self.state_dtype)
        return d

    def dims(self) -> dict:
        return {
            "batch": int(self.batch),
            "num_k_heads": int(self.num_k_heads),
            "num_v_heads": int(self.num_v_heads),
            "head_k_dim": int(self.head_k_dim),
            "head_v_dim": int(self.head_v_dim),
        }


GDN_DIM_VOCABULARY = (
    "batch",
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
)


def request_errors(req: OperatorRequest) -> list:
    """Shape-level rejections that are independent of any candidate or arch."""
    if not isinstance(req, GdnDecodeRequest):
        return [f"expected GdnDecodeRequest, got {type(req).__name__}"]
    errors = []
    if req.batch <= 0:
        errors.append(f"batch must be positive, got {req.batch}")
    if req.num_k_heads <= 0 or req.num_v_heads <= 0:
        errors.append("head counts must be positive")
    elif req.num_v_heads % req.num_k_heads:
        errors.append(
            f"num_v_heads {req.num_v_heads} must be a multiple of "
            f"num_k_heads {req.num_k_heads}"
        )
    if req.head_k_dim <= 0 or req.head_v_dim <= 0:
        errors.append("head dims must be positive")
    if normalize_dtype(req.dtype) not in GDN_DTYPES:
        errors.append(f"unsupported dtype {req.dtype!r}")
    if normalize_dtype(req.state_dtype) not in GDN_DTYPES:
        errors.append(f"unsupported state_dtype {req.state_dtype!r}")
    if req.gate_kind not in ("gdn", "kda"):
        errors.append(
            f"unsupported gate_kind {req.gate_kind!r} (expected 'gdn' or 'kda')"
        )
    if req.gate_kind == "kda" and (req.head_k_dim != 128 or req.head_v_dim != 128):
        errors.append(
            "NOT_YET_IMPLEMENTED: KDA decode currently requires "
            "head_k_dim == head_v_dim == 128"
        )
    return errors


# Shared pin-selector, re-exported under this family's name (see dispatch core).
selector_matches = _shared_selector_matches
