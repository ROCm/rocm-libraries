# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The capability gate shared by the wave32 WMMA forward convolutions.

``conv_direct_fwd_wmma`` and ``conv_fwd_small_wmma`` (small C / small K) run
on the same hardware: a wave32 target with TDM and the 16x16x32 fp16 / bf16
WMMA atom. :func:`wmma_fwd_gate` checks exactly that, once, for all three
families; the shape rules of each family stay in its own ``*_reason``.

The C++ engine mirrors this module in
``platform/cpp/instances/common/conv_fwd_wmma_gate.cpp`` with the same checks
in the same order and the same rejection messages.
"""

from __future__ import annotations

from rocke.core.arch import ArchTarget

from kernels.common.conv_direct_grouped import DirectConvProblem

__all__ = ["WMMA_K", "WMMA_M", "WMMA_N", "wmma_ab_dtype", "wmma_fwd_gate"]

# The fp32-accumulating WMMA atom every forward family issues: M x N x K.
WMMA_M = 16
WMMA_N = 16
WMMA_K = 32


def wmma_ab_dtype(dtype: str) -> str:
    """The WMMA operand dtype of a conv ``dtype`` ("fp16" or "bf16")."""
    return "bf16" if dtype == "bf16" else "f16"


def wmma_fwd_gate(
    p: DirectConvProblem, arch: str, name: str, wave_size: int
) -> tuple[str | None, ArchTarget | None]:
    """``(why, target)``: why kernel ``name`` cannot run ``p`` on ``arch`` at
    the spec's ``wave_size`` (``None`` when the target qualifies), and the
    resolved :class:`ArchTarget` so callers need not resolve it again. The
    target is ``None`` only when ``arch`` is unknown."""
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return str(e), None
    if target.wave_size != 32 or wave_size != 32:
        return f"{name} is a wave32 kernel", target
    if not target.memory.has_tdm:
        return f"no TDM on {arch}", target
    if p.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {p.dtype!r}", target
    ab = wmma_ab_dtype(p.dtype)
    if not target.mma.has_shape(
        family="wmma",
        a_dtype=ab,
        b_dtype=ab,
        c_dtype="fp32",
        m=WMMA_M,
        n=WMMA_N,
        k=WMMA_K,
    ):
        return f"missing wmma_f32_{WMMA_M}x{WMMA_N}x{WMMA_K}_{ab} on {arch}", target
    return None, target
