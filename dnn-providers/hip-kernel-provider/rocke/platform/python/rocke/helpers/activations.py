# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared scalar activation primitives (exp2-based, AMDGPU-lowerable).

The transcendental activations used by the fused-epilogue ops
(:mod:`helpers.fuse`) and the elementwise instance
(:mod:`instances.common.elementwise`) reduce to the same two f32
building blocks: an AMDGPU-lowerable ``tanh`` and an ``exp2``-based
``sigmoid``. The core ``math.tanh`` operation expands to stable f32
arithmetic plus ``exp2`` instead of emitting ``llvm.tanh.f32``; sigmoid
also avoids ``math.exp`` because the AMDGPU backend does not lower those
intrinsics on its own.

These were previously duplicated across ``fuse.py`` and ``elementwise.py``;
they live here so both call sites share one canonical operation sequence.
"""

from __future__ import annotations

from ..core.ir import IRBuilder, Value


__all__ = ["_sigmoid_via_exp2", "_tanh_via_exp2"]


def _sigmoid_via_exp2(b: IRBuilder, x: Value) -> Value:
    """1 / (1 + e^-x), implemented via exp2.

    ``exp(-x) = exp2(-x * log2(e))``. Avoids ``math.exp`` (which the
    AMDGPU backend does not lower on its own).
    """
    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one = b.const_f32(1.0)
    return b.rcp(b.fadd(one, b.exp2(b.fmul(c_neg_log2e, x))))


def _tanh_via_exp2(b: IRBuilder, x: Value) -> Value:
    """Return the stable AMDGPU-lowerable f32 ``tanh`` operation."""
    return b.tanh(x)
