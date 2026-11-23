# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense SDPA backward main kernel on gfx942.

:class:`Gfx942AttentionDenseBwdSpec` is
:class:`~kernels.common.attention_dense_bwd_spec.ArchAttentionDenseBwdSpec`
bound to gfx942: the shared dense spec plus the body knob ``body_variant``:

* ``"general"`` (built): the shared backward body of the layout-general
  family behind the dense shell (compile-time strides and lengths, buffer
  range-checked loads, compile-time band classification, runtime batch);
* ``"xt"`` (registered, not built): the XT body (K, K^T, V register
  resident, XT LDS transposes, a stage table with one-deep register prefetch)
  as its own emitter.

The dense backward files never import the dense forward modules, so the
forward golden IR cannot move with them.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernels.common._attention_dense_bwd_shell import emit_dense_bwd_main
from kernels.common.attention_dense_bwd_spec import (
    ArchAttentionDenseBwdSpec,
    check_dense_bwd_spec_class,
)

__all__ = ["Gfx942AttentionDenseBwdSpec", "build_attention_dense_bwd"]


@dataclass(frozen=True)
class Gfx942AttentionDenseBwdSpec(ArchAttentionDenseBwdSpec, arch="gfx942"):
    """Dense backward spec on gfx942 (see the module docstring)."""


def build_attention_dense_bwd(spec: Gfx942AttentionDenseBwdSpec, *, arch: str):
    """Build the gfx942 dense backward main kernel (``rocke.attn_dense_bwd.v1``).

    The spec is validated for ``arch`` first (an illegal spec raises
    ``ValueError``); knob values whose emission is not built raise
    ``NotImplementedError`` before any IR is emitted.
    """
    check_dense_bwd_spec_class(spec, Gfx942AttentionDenseBwdSpec)
    return emit_dense_bwd_main(spec, arch)
