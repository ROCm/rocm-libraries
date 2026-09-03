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
from typing import Tuple

from rocke.dispatch.core import KernelCandidate, OperatorRequest

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

    ``batch`` is the number of *active* sequences this step, which is the only
    dimension the tuned tile selection depends on: it sets how much natural
    parallelism the launch already has, and therefore whether the kernel needs
    to manufacture more by splitting each head's V dimension across workgroups.

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
    if normalize_dtype(req.dtype) not in ("bf16", "f16"):
        errors.append(f"unsupported dtype {req.dtype!r}")
    if normalize_dtype(req.state_dtype) not in ("bf16", "f16"):
        errors.append(f"unsupported state_dtype {req.state_dtype!r}")
    return errors


def selector_matches(
    req: GdnDecodeRequest, candidate: KernelCandidate
) -> Tuple[bool, str]:
    """Honour an explicit ``algorithm`` / ``spec_id`` pin on the request."""
    algorithm = req.algorithm.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    spec_id = req.spec_id.strip().lower()
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"
