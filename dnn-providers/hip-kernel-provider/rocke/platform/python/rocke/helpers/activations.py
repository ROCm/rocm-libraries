# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared scalar activation primitives (exp2-based, AMDGPU-lowerable).

The transcendental activations used by the fused-epilogue ops
(:mod:`helpers.fuse`) and the elementwise instance
(:mod:`instances.common.elementwise`) reduce to the same two f32
building blocks: an ``exp2``-based ``tanh`` and ``sigmoid``. Both
deliberately avoid ``llvm.tanh.f32`` / ``math.exp`` because the AMDGPU
backend does not lower them on its own (it produces a
``CODEGEN_BC_TO_RELOCATABLE`` failure in ``amd_comgr``); ``exp2`` is
the one transcendental the backend always lowers.

These were previously duplicated byte-for-byte across ``fuse.py`` and
``elementwise.py``; they live here so both call sites share the exact
same constant pool and op chain. The emitted IR is identical to the
former inline bodies (same ops, same order).
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
    """tanh(x) = (e^{2x} - 1) / (e^{2x} + 1), via exp2.

    Deliberately avoids ``math.tanh`` / ``llvm.tanh.f32`` to keep the
    body AMDGPU-lowerable without extra runtime libraries.
    """
    c_2log2e = b.const_f32(2.0 * 1.4426950408889634)
    one = b.const_f32(1.0)
    e2x = b.exp2(b.fmul(c_2log2e, x))
    return b.fmul(b.fsub(e2x, one), b.rcp(b.fadd(e2x, one)))
