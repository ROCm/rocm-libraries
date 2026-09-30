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


__all__ = [
    "SOFTPLUS_THRESHOLD",
    "LN2",
    "LOG2E",
    "_sigmoid_via_exp2",
    "_tanh_via_exp2",
]


# softplus(x) = log1p(exp(x)). Above this x the two agree to f32 precision, so
# the branch returns x directly -- and exp(20) ~ 4.9e8 is close enough to the
# clamped exp2 range that evaluating it buys nothing but risk. Every emitter
# AND every host reference must switch at the SAME point: an oracle that keeps
# its own copy stops being independent of the kernel it checks, and the
# disagreement between the two thresholds is far below any tolerance.
SOFTPLUS_THRESHOLD = 20.0
LN2 = 0.6931471805599453
LOG2E = 1.4426950408889634


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
