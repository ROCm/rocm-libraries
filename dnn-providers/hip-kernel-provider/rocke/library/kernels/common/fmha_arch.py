# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Arch-awareness helpers for the MFMA-tiled FMHA forward instances.

The MFMA-tiled attention body (:func:`rocke.helpers.mfma_attention.
mfma_attention_fwd_inner_body`) and the warp-distributed scalar bodies
in the ``fmha_*`` / ``sage_attention`` / ``sparse_attention`` instances
all run on the *narrow* MFMA atom catalog:

* f16 ``16x16x16`` (``MFMA_ATTN_BLOCK_M == MFMA_ATTN_BLOCK_K == 16``)
* bf16 ``16x16x16``

Both of those atoms exist on **gfx942 (CDNA3)** and **gfx950 (CDNA4)**,
so the canonical f16 / bf16 forward path is arch-portable. This helper
makes that fact explicit and machine-checkable: it sources the legal
atom set from :class:`rocke.core.arch.ArchTarget` (the same catalog
``instances/common/gemm_universal.is_valid_spec`` consults) and returns
a structured ``(ok, reason)`` instead of letting an arch-incompatible
config reach comgr (where a missing intrinsic HARD-CRASHES with
``LLVM ERROR: Cannot select intrinsic``).

The MFMA-tiled FMHA bodies never select the *wide* gfx950-only atoms
(``16x16x32`` / ``32x32x16``) on the f16 / bf16 path -- the helper pins
``BLOCK_K = 16`` -- so the validation below is a forward-looking guard
for the day a caller wires a wide-atom or fp8 attention variant through
these instances. Keeping the predicate arch-aware now means those
variants reject gfx942 cleanly instead of crashing the backend.
"""

from __future__ import annotations

from typing import Tuple


__all__ = [
    "FMHA_MFMA_ATTN_BLOCK",
    "validate_fmha_mfma_atom",
]


# The narrow attention MFMA tile both attention helpers pin (the K-dim
# of the QK / PV atom is 16; head_size is sliced into 16-wide atoms).
FMHA_MFMA_ATTN_BLOCK = 16


def validate_fmha_mfma_atom(dtype: str, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for the MFMA-tiled FMHA atom on ``arch``.

    Mirrors :func:`rocke.instances.common.gemm_universal.is_valid_spec`'s
    catalog query: the f16 / bf16 ``16x16x16`` warp-tile atom the
    attention body uses must be in the target's MMA catalog. gfx942 and
    gfx950 both ship it, so the standard f16 / bf16 forward is accepted
    on both; an unknown arch (or a future dtype with no narrow atom on
    the target) is rejected with a structured reason.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    # Map the FMHA activation dtype string to the catalog dtype name.
    if dtype in ("f16", "fp16"):
        a_name = "f16"
    elif dtype == "bf16":
        a_name = "bf16"
    else:
        # validate_common_spec already restricts dtype to {f16,fp16,bf16};
        # this branch is defensive for callers that skip it.
        return False, f"unsupported FMHA dtype {dtype!r} for MFMA atom selection"

    blk = FMHA_MFMA_ATTN_BLOCK
    if not target.mma.has_shape(
        a_dtype=a_name, b_dtype=a_name, c_dtype="fp32", m=blk, n=blk, k=blk
    ):
        return False, (
            f"FMHA MFMA atom {a_name} {blk}x{blk}x{blk} not in {arch} catalog"
        )
    return True, "ok"
