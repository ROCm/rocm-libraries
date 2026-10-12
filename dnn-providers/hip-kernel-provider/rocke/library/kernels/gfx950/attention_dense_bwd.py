# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense SDPA backward main kernel on gfx950.

:class:`Gfx950AttentionDenseBwdSpec` is the shared
:class:`~kernels.common.attention_dense_bwd_spec.AttentionDenseBwdSpec` plus
the gfx950 body knob ``body_variant``:

* ``"general"`` (built): the shared backward body of the layout-general
  family behind the dense shell (compile-time strides and lengths, buffer
  range-checked loads, compile-time band classification, runtime batch);
* ``"trload"`` (registered, not built): the trload body (x4 LDS DMA with a
  source-side XOR, a two-deep ring, ``ds_read_tr16_b64`` operand reads, a
  larger key tile from the 160 KiB LDS) as its own emitter.

The dense backward files never import the dense forward modules, so the
forward golden IR cannot move with them.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernels.common._attention_dense_bwd_shell import emit_dense_bwd_main
from kernels.common.attention_dense_bwd_spec import AttentionDenseBwdSpec

__all__ = ["Gfx950AttentionDenseBwdSpec", "build_attention_dense_bwd"]


@dataclass(frozen=True)
class Gfx950AttentionDenseBwdSpec(AttentionDenseBwdSpec):
    """Dense backward spec on gfx950 (see the module docstring)."""

    body_variant: str = "general"

    ARCH = "gfx950"
    BODY_VARIANTS = ("general", "trload")


def build_attention_dense_bwd(spec: Gfx950AttentionDenseBwdSpec, *, arch: str):
    """Build the gfx950 dense backward main kernel (``rocke.attn_dense_bwd.v1``).

    The spec is validated for ``arch`` first (an illegal spec raises
    ``ValueError``); knob values whose emission is not built raise
    ``NotImplementedError`` before any IR is emitted.
    """
    if not isinstance(spec, Gfx950AttentionDenseBwdSpec):
        raise TypeError(
            f"build_attention_dense_bwd on gfx950 takes a Gfx950AttentionDenseBwdSpec, "
            f"got {type(spec).__name__}"
        )
    return emit_dense_bwd_main(spec, arch)
