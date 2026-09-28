# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Workspace-cast kernel for the two-stage wgrad path.

Stage 2 of the two-stage backward-weight convolution.  Stage 1
(``conv_implicit_gemm_wgrad`` with ``two_stage=True``) atomic-adds its f32
partial sums into a scratch buffer of shape ``[groups, wg_M, wg_N]``; by the
time this kernel runs the reduction over ``split_k`` is already complete, done
by the hardware atomics.  All that is left is to convert f32 -> ``dtype_d`` and
write ``dW``.

That is why this kernel has no loop: the scratch and ``dW`` have exactly the
same element count and the same layout, so the whole thing is one load, one
convert, one store per element.  (It did once carry a sequential reduction over
a ``[groups * split_k, wg_M, wg_N]`` scratch.  The grid here is sized by the
*output*, which for a wgrad is a filter-sized handful of elements, so that loop
ran on a few CTAs with one dependent load per iteration and cost more than the
Stage 1 GEMM it was reducing.)

For grouped convolutions the group slabs are contiguous in both buffers, so
``block_id_z`` indexes them identically on each side.

Kernel signature::

    ws_ptr  : f32 global ptr, readonly      — scratch [groups * wg_M * wg_N]
    dw_ptr  : dtype_d global ptr, writeonly — weight gradient [groups * wg_M * wg_N]
    wg_M    : i32   — per-group output-channel dimension (K // groups)
    wg_N    : i32   — per-group filter-spatial × input-channel (Y*X * C//groups)
    ws_bytes: i32   — scratch buffer byte size (ABI boundary)
    dw_bytes: i32   — dW buffer byte size (ABI boundary)
    groups  : i32   — number of convolution groups (1 for non-grouped)

Grid: ``(ceil(wg_N / tile_n), ceil(wg_M / tile_m), groups)``
Block: ``(block_size, 1, 1)`` where ``block_size = tile_m * tile_n`` (flat)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.io import store_scalar_from_f32
from rocke.helpers.spec import SignatureBuilder, kernel_name_join
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm_wgrad import (
    _DEFAULT_WS_REPLICAS,
    _wg_M,
    _wg_N,
)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# Default tile sizes for the cast kernel.  Each workgroup handles one
# (tile_m x tile_n) patch of the (wg_M x wg_N) output space.
_DEFAULT_TILE_M = 4
_DEFAULT_TILE_N = 64  # block_size = tile_m * tile_n = 256


@dataclass(frozen=True)
class WgradReduceSpec:
    """Configuration for the two-stage wgrad workspace-cast kernel.

    Args:
        problem:    The convolution problem (needed for wg_M / wg_N).
        dtype_d:    Output dtype for dW ("fp32", "fp16", "bf16").
        tile_m:     Workgroup tile height over the M dimension.
        tile_n:     Workgroup tile width over the N dimension.
        name:       Kernel base name.
        groups:     Number of convolution groups.  Grid z = groups; the CTA at
                    block_id_z=g casts group g's scratch slab into the
                    corresponding dW slab.
    """

    problem: ConvProblem
    dtype_d: str = "fp16"
    tile_m: int = _DEFAULT_TILE_M
    tile_n: int = _DEFAULT_TILE_N
    name: str = "conv_wgrad_ws_cast"
    groups: int = 1
    # Must match ``WgradConvSpec.ws_replicas`` of the Stage 1 kernel that filled
    # the scratch. Folding fewer slabs than Stage 1 wrote silently drops part of
    # the sum; folding more reads past the buffer. Shares Stage 1's default
    # constant rather than restating it, so the two cannot drift.
    ws_replicas: int = _DEFAULT_WS_REPLICAS

    @property
    def block_size(self) -> int:
        return self.tile_m * self.tile_n

    @property
    def wg_M(self) -> int:
        return _wg_M(self.problem)

    @property
    def wg_N(self) -> int:
        return _wg_N(self.problem)

    def kernel_name(self) -> str:
        p = self.problem
        parts = [p.short(), f"t{self.tile_m}x{self.tile_n}"]
        # Only tagged above the default: the replica count changes the emitted
        # fold, and the name is what the launcher and artifact cache key on.
        if self.ws_replicas > 1:
            parts.append(f"wsr{self.ws_replicas}")
        parts.append(self.dtype_d)
        return kernel_name_join(self.name, *parts)


def build_conv_wgrad_workspace_reduce(
    spec: WgradReduceSpec,
    *,
    arch: str = "gfx950",  # noqa: ARG001 — reserved for future arch dispatch
) -> KernelDef:
    """Build the IR for the Stage 2 workspace-cast kernel.

    Each workgroup covers a ``(tile_m, tile_n)`` patch of ``(wg_M, wg_N)``.
    Within the patch, each thread owns one ``(m, n)`` element: it loads the f32
    scratch value, converts it to ``dtype_d``, and stores it to ``dW``.

    The scratch already holds the complete sum over ``split_k`` -- Stage 1's
    f32 atomics did the reduction -- so there is nothing to accumulate here and
    no summation order to fix.
    """
    tile_m = spec.tile_m
    tile_n = spec.tile_n
    BS = spec.block_size  # tile_m * tile_n — one thread per output element
    _is_fp32_out = spec.dtype_d in ("fp32", "f32")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    ws_ptr = b.param(
        "ws_ptr", PtrType(F32, "global"), noalias=True, readonly=True, align=16
    )
    # For fp32 output the ptr element type is f32; for fp16/bf16 use io_ir_type.
    if _is_fp32_out:
        dw_pty = PtrType(F32, "global")
    else:
        from rocke.helpers.io import io_ir_type

        dw_pty = PtrType(io_ir_type(spec.dtype_d), "global")
    dw_ptr = b.param("dw_ptr", dw_pty, noalias=True, writeonly=True, align=16)
    wg_M_param = b.param("wg_M", I32)
    wg_N_param = b.param("wg_N", I32)
    _ws_bytes = b.param(
        "ws_bytes", I32
    )  # noqa: F841 — ABI boundary; no bounds check performed
    _dw_bytes = b.param(
        "dw_bytes", I32
    )  # noqa: F841 — ABI boundary; no bounds check performed
    _groups = b.param(
        "groups", I32
    )  # noqa: F841 — ABI boundary; the group index comes from block_id_z

    # Thread flat index within the workgroup.
    tid = b.thread_id_x()

    # Grid is (ceil(wg_N/tile_n), ceil(wg_M/tile_m), groups):
    #   blockIdx.x — N tiles, blockIdx.y — M tiles (local within group),
    #   blockIdx.z — group index.
    blk_m = b.block_id_y()
    blk_n = b.block_id_x()
    grp_id = b.block_id_z()

    # Each thread in the flat block owns one (m_local, n_local) element.
    t_m = b.div(tid, b.const_i32(tile_n))  # row within tile
    t_n = b.mod(tid, b.const_i32(tile_n))  # col within tile

    # Per-group (m, n) coordinates.
    c_m = b.add(b.mul(blk_m, b.const_i32(tile_m)), t_m)
    c_n = b.add(b.mul(blk_n, b.const_i32(tile_n)), t_n)

    # OOB guard — threads outside [0, wg_M) x [0, wg_N) do nothing.
    in_bounds = b.land(b.cmp_lt(c_m, wg_M_param), b.cmp_lt(c_n, wg_N_param))
    with b.scf_if(in_bounds):
        #   dw_off  = grp_id * wg_M*wg_N + c_m * wg_N + c_n
        #   ws_off  = (grp_id * R + r) * wg_M*wg_N + c_m * wg_N + c_n
        # dW is one slab per group; the scratch is R slabs per group. At R == 1
        # the two coincide and the fold collapses to a single load.
        grp_stride = b.mul(wg_M_param, wg_N_param)
        elem_in_slab = b.add(b.mul(c_m, wg_N_param), c_n)
        dw_off = b.add(b.mul(grp_id, grp_stride), elem_in_slab)

        reps = spec.ws_replicas
        if reps == 1:
            total = b.global_load_f32(ws_ptr, dw_off)
        else:
            # R is compile-time, so this is a flat unrolled fold -- R
            # independent loads issued before the first add consumes one.
            ws_base = b.add(
                b.mul(b.mul(grp_id, b.const_i32(reps)), grp_stride), elem_in_slab
            )
            partials = [
                b.global_load_f32(
                    ws_ptr,
                    (
                        ws_base
                        if r == 0
                        else b.add(ws_base, b.mul(b.const_i32(r), grp_stride))
                    ),
                )
                for r in range(reps)
            ]
            total = partials[0]
            for partial in partials[1:]:
                total = b.fadd(total, partial)

        if _is_fp32_out:
            # Scratch is already f32 — plain store, no conversion needed.
            b.global_store(dw_ptr, dw_off, total, align=4)
        else:
            store_scalar_from_f32(b, dw_ptr, dw_off, total, dtype=spec.dtype_d)

    return b.kernel


def wgrad_reduce_grid(spec: WgradReduceSpec) -> Tuple[int, int, int]:
    """Return the ``(x, y, z)`` grid dimensions for this reduce spec."""
    return (
        _ceil_div(spec.wg_N, spec.tile_n),
        _ceil_div(spec.wg_M, spec.tile_m),
        spec.groups,
    )


def wgrad_reduce_signature(spec: WgradReduceSpec) -> list:
    """Return the kernel signature list for :class:`KernelLauncher`."""
    return (
        SignatureBuilder()
        .ptr("ws_ptr", "fp32")
        .ptr("dw_ptr", spec.dtype_d)
        .scalar("wg_M", "i32")
        .scalar("wg_N", "i32")
        .scalar("ws_bytes", "i32")
        .scalar("dw_bytes", "i32")
        .scalar("groups", "i32")
        .build()
    )
