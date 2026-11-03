# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The capability gate shared by the wave32 WMMA forward convolutions.

``conv_direct_fwd_wmma`` and ``conv_fwd_small_wmma`` (small C / small K) run
on the same hardware: a wave32 target with TDM and the 16x16x32 fp16 / bf16
WMMA atom. :func:`wmma_fwd_gate` checks exactly that, once, for every
forward family, together with the padding each takes (one for both axes, or
a column padding of its own); the other shape rules of each family stay in
its own ``*_reason``, which ends with :func:`wmma_fwd_limits_reason`. The
halo helpers give the input window a tile of output pixels reads, for the
families that stage one per channel chunk (direct, small-group).

The C++ engine mirrors this module in
``platform/cpp/instances/common/conv_fwd_wmma_gate.cpp`` with the same checks
in the same order and the same rejection messages.
"""

from __future__ import annotations

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import normalize_scheduler_strategy
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    one_padding_reason,
    pad_w_reason,
)

__all__ = [
    "WMMA_K",
    "WMMA_M",
    "WMMA_N",
    "wmma_ab_dtype",
    "wmma_fwd_gate",
    "wmma_fwd_halo_cols",
    "wmma_fwd_halo_extent",
    "wmma_fwd_lead_cols",
    "wmma_fwd_limits_reason",
]

# The fp32-accumulating WMMA atom every forward family issues: M x N x K.
WMMA_M = 16
WMMA_N = 16
WMMA_K = 32
_I32_MAX = (1 << 31) - 1


def wmma_ab_dtype(dtype: str) -> str:
    """The WMMA operand dtype of a conv ``dtype`` ("fp16" or "bf16")."""
    return "bf16" if dtype == "bf16" else "f16"


def wmma_fwd_gate(
    p: DirectConvProblem,
    arch: str,
    name: str,
    wave_size: int,
    *,
    per_axis_pad: bool = False,
) -> tuple[str | None, ArchTarget | None]:
    """``(why, target)``: why kernel ``name`` cannot run ``p`` on ``arch`` at
    the spec's ``wave_size`` (``None`` when the target and the padding
    qualify), and the resolved :class:`ArchTarget` so callers need not resolve
    it again. The target is ``None`` only when ``arch`` is unknown.
    ``per_axis_pad`` lets a kernel that offsets columns by ``PAD_W`` take a
    column padding of its own."""
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
    return (pad_w_reason(p) if per_axis_pad else one_padding_reason(p)), target


def wmma_fwd_limits_reason(
    p: DirectConvProblem, sched: str | None, n_blocks: int
) -> str | None:
    """The checks a forward validator ends with: a known scheduler strategy,
    i32 byte offsets into X, W and Y, and an i32 flat block id."""
    try:
        normalize_scheduler_strategy(sched)
    except ValueError as e:
        return str(e)
    tensor_bytes = (
        2 * p.N * p.H * p.W * p.total_c,
        2 * p.total_k * p.KH * p.KW * p.cpg,
        2 * p.N * p.Ho * p.Wo * p.total_k,
    )
    if max(tensor_bytes) > _I32_MAX:
        return f"tensors of {max(tensor_bytes)} B exceed the i32 byte offsets"
    if n_blocks > _I32_MAX:
        return f"{n_blocks} workgroups exceed the i32 block id"
    return None


def wmma_fwd_halo_extent(stride: int, tile: int, k: int, dil: int) -> int:
    """Input rows (columns) a tile of ``tile`` output rows (columns) reads
    through a ``k``-tap filter axis at ``stride`` and dilation ``dil``."""
    return stride * (tile - 1) + (k - 1) * dil + 1


def wmma_fwd_halo_cols(p: DirectConvProblem, tile_w: int, phase_split: bool) -> int:
    """Columns of one (phase) sub-halo: the whole halo row, or with
    ``phase_split`` the columns of one of the ``stride`` column phases."""
    if phase_split:
        return tile_w + ((p.KW - 1) * p.dil_w) // p.stride
    return wmma_fwd_halo_extent(p.stride, tile_w, p.KW, p.dil_w)


def wmma_fwd_lead_cols(pad_w: int, stride: int, phase_split: bool) -> int:
    """Most leading zero columns a (phase) sub-halo gets at the left edge."""
    return ceil_div(pad_w, stride) if phase_split else pad_w
