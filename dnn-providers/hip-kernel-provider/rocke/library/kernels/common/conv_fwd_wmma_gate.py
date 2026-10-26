# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The capability gate and shared facts of the wave32 WMMA forward convolutions.

What each family reads from here:

- :func:`wmma_fwd_gate` (a wave32 target with TDM and the 16x16x32 fp16 /
  bf16 WMMA atom, plus the padding a family takes: one for both axes, or a
  column padding of its own): ``conv_direct_fwd_wmma``,
  ``conv_fwd_small_wmma`` (small C / small K),
  ``conv_direct_fwd_gstream_wmma`` (small groups) and
  ``conv_direct_fwd_tapouter_wmma`` (large dilations);
- :func:`wmma_fwd_limits_reason`, the end of a validator (scheduler, i32
  byte offsets, i32 block id): ``conv_direct_fwd_wmma``,
  ``conv_direct_fwd_gstream_wmma`` and ``conv_direct_fwd_tapouter_wmma``;
- the halo helpers (the input window a tile of output pixels reads, for the
  families that stage one per channel chunk): ``conv_direct_fwd_wmma`` and
  ``conv_direct_fwd_gstream_wmma``;
- :func:`wmma_fwd_tensor_bytes_reason`, the i32 byte-offset check of the
  tail: also ``conv_fwd_small_wmma``, which ends with its own scheduler check;
- ``CH_PAD``, the LDS row skew of a staged X / W tile: ``conv_direct_fwd_wmma``,
  ``conv_fwd_small_wmma``, ``conv_direct_fwd_gstream_wmma`` and
  ``conv_direct_fwd_tapouter_wmma``;
- the WMMA atom (``WMMA_M`` / ``WMMA_N`` / ``WMMA_K``, :func:`wmma_ab_dtype`,
  :func:`wmma_fwd_op`): all of the above and ``conv_direct_dw_fwd_wmma``
  (depthwise), which uses neither the gate nor the validator tail: it checks
  its arch and shapes itself;
- ``CH_PAD``, ``WMMA_K``, ``I32_MAX`` and :func:`wmma_fwd_op` also serve the
  dgrad WMMA families that stage like the forward ones
  (``conv_direct_dgrad_wmma``, ``conv_dgrad_patch_wmma`` and
  ``conv_dgrad_smallc_wmma``).

The other shape rules of each family stay in its own ``*_reason``.

The C++ engine mirrors this module in
``library/cpp/instances/common/conv_fwd_wmma_gate.cpp`` with the same checks
in the same order and the same rejection messages.
"""

from __future__ import annotations

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import scheduler_strategy_reason
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    I32_MAX,
    DirectConvProblem,
    one_padding_reason,
    pad_w_reason,
    problem_dims_reason,
)

__all__ = [
    "CH_PAD",
    "I32_MAX",
    "WMMA_K",
    "WMMA_M",
    "WMMA_N",
    "wmma_ab_dtype",
    "wmma_fwd_gate",
    "wmma_fwd_halo_cols",
    "wmma_fwd_halo_extent",
    "wmma_fwd_lead_cols",
    "wmma_fwd_limits_reason",
    "wmma_fwd_op",
    "wmma_fwd_tensor_bytes_reason",
]

# The fp32-accumulating WMMA atom every forward family issues: M x N x K.
WMMA_M = 16
WMMA_N = 16
WMMA_K = 32
CH_PAD = 8  # channels of row skew past each LDS row of a staged tile (16 bytes)


def wmma_ab_dtype(dtype: str) -> str:
    """The WMMA operand dtype of a conv ``dtype`` ("fp16" or "bf16")."""
    return "bf16" if dtype == "bf16" else "f16"


def wmma_fwd_op(dtype: str, arch: str):
    """The ``WMMA_M x WMMA_N x WMMA_K`` fp32-accumulating WMMA of conv
    ``dtype`` on ``arch`` (the op every forward family issues)."""
    ab = wmma_ab_dtype(dtype)
    return ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma",
        a_dtype=ab,
        b_dtype=ab,
        c_dtype="fp32",
        m=WMMA_M,
        n=WMMA_N,
        k=WMMA_K,
    )


def wmma_fwd_gate(
    p: DirectConvProblem,
    arch: str,
    name: str,
    wave_size: int,
    *,
    per_axis_pad: bool = False,
) -> tuple[str | None, ArchTarget | None]:
    """``(why, target)``: why kernel ``name`` cannot run ``p`` on ``arch`` at
    the spec's ``wave_size`` (``None`` when the target, the problem extents
    (:func:`problem_dims_reason`) and the padding qualify), and the resolved
    :class:`ArchTarget` so callers need not resolve it again. The target is
    ``None`` only when ``arch`` is unknown. ``per_axis_pad`` lets a kernel
    that offsets columns by ``PAD_W`` take a column padding of its own."""
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
    why = problem_dims_reason(p)
    if why is None:
        why = pad_w_reason(p) if per_axis_pad else one_padding_reason(p)
    return why, target


def wmma_fwd_limits_reason(
    p: DirectConvProblem, sched: str | None, n_blocks: int
) -> str | None:
    """The checks a forward validator ends with: a known scheduler strategy,
    i32 byte offsets into X, W and Y, and an i32 flat block id."""
    why = scheduler_strategy_reason(sched) or wmma_fwd_tensor_bytes_reason(p)
    if why is not None:
        return why
    if n_blocks > I32_MAX:
        return f"{n_blocks} workgroups exceed the i32 block id"
    return None


def wmma_fwd_tensor_bytes_reason(p: DirectConvProblem) -> str | None:
    """Why the i32 byte offsets cannot address X, W or Y of ``p``."""
    tensor_bytes = (
        2 * p.N * p.H * p.W * p.total_c,
        2 * p.total_k * p.KH * p.KW * p.cpg,
        2 * p.N * p.Ho * p.Wo * p.total_k,
    )
    if max(tensor_bytes) > I32_MAX:
        return f"tensors of {max(tensor_bytes)} B exceed the i32 byte offsets"
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
