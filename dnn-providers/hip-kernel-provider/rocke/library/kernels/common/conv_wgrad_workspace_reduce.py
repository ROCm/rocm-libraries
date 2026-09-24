# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Workspace-reduce kernel for two-stage deterministic wgrad.

Stage 2 of the two-stage backward-weight convolution.  Reads the f32 partial
sums written by Stage 1 (``conv_implicit_gemm_wgrad`` with ``two_stage=True``)
from a flat workspace buffer of shape ``[groups * split_k, wg_M, wg_N]`` and
reduces (sums) along the ``split_k`` axis in a fixed sequential order, then
stores the result as ``dtype_d`` to the weight-gradient output ``dW``.

For grouped convolutions (``groups > 1``), ``block_id_z`` encodes the group
index.  Each group's ``split_k`` workspace slices are laid out contiguously at
offset ``grp_id * split_k`` in the first dimension.  The dW output for group
``g`` starts at flat offset ``g * wg_M * wg_N`` (per-group slab).

The fixed iteration order ``k_id = 0, 1, ..., split_k - 1`` guarantees
bit-exact reproducibility across runs, machines, and GPU models (for a fixed
kernel binary and a fixed problem descriptor).

Kernel signature::

    ws_ptr  : f32 global ptr, readonly   — workspace [groups * split_k * wg_M * wg_N]
    dw_ptr  : dtype_d global ptr, writeonly — weight gradient output [groups * wg_M * wg_N]
    wg_M    : i32   — per-group output-channel dimension (K // groups)
    wg_N    : i32   — per-group filter-spatial × input-channel (Y*X * C//groups)
    split_k : i32   — number of K partitions written by Stage 1 (per group)
    ws_bytes: i32   — workspace buffer byte size (ABI boundary)
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
from kernels.common.conv_implicit_gemm_wgrad import _wg_M, _wg_N


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# Default tile sizes for the reduction kernel.  Each workgroup handles one
# (tile_m x tile_n) patch of the (wg_M x wg_N) output space.
_DEFAULT_TILE_M = 4
_DEFAULT_TILE_N = 64  # block_size = tile_m * tile_n = 256


@dataclass(frozen=True)
class WgradReduceSpec:
    """Configuration for the two-stage wgrad workspace-reduce kernel.

    Args:
        problem:    The convolution problem (needed for wg_M / wg_N).
        dtype_d:    Output dtype for dW ("fp32", "fp16", "bf16").
        tile_m:     Workgroup tile height over the M dimension.
        tile_n:     Workgroup tile width over the N dimension.
        name:       Kernel base name.
        groups:     Number of convolution groups.  Grid z = groups; each CTA
                    at block_id_z=g reduces the split_k workspace slices for
                    group g into the corresponding dW slab.
    """

    problem: ConvProblem
    dtype_d: str = "fp16"
    tile_m: int = _DEFAULT_TILE_M
    tile_n: int = _DEFAULT_TILE_N
    name: str = "conv_wgrad_ws_reduce"
    groups: int = 1

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
        return kernel_name_join(
            self.name,
            p.short(),
            f"t{self.tile_m}x{self.tile_n}",
            self.dtype_d,
        )


def build_conv_wgrad_workspace_reduce(
    spec: WgradReduceSpec,
    *,
    arch: str = "gfx950",  # noqa: ARG001 — reserved for future arch dispatch
) -> KernelDef:
    """Build the IR for the Stage 2 workspace-reduce kernel.

    Each workgroup covers a ``(tile_m, tile_n)`` patch of ``(wg_M, wg_N)``.
    Within the patch, each thread owns one ``(m, n)`` element.  It iterates
    sequentially over ``k_id = 0 .. split_k - 1``, accumulates the f32
    partial sums from the workspace, then stores the result as ``dtype_d``
    to ``dW``.

    The sequential loop order is the determinism guarantee: floating-point
    addition is not associative, so fixing the iteration order fixes the
    result bit-exactly.
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
    split_k_param = b.param("split_k", I32)
    _ws_bytes = b.param(
        "ws_bytes", I32
    )  # noqa: F841 — ABI boundary; no bounds check performed
    _dw_bytes = b.param(
        "dw_bytes", I32
    )  # noqa: F841 — ABI boundary; no bounds check performed
    groups_param = b.param("groups", I32)

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
        # Sequential reduction over split_k slices for this group.
        # Workspace layout: [groups * split_k, wg_M, wg_N] (f32).
        # Stage 1 writes blockIdx.z = grp_id * split_k + kid, so group g's
        # slices occupy indices [g*split_k, g*split_k+split_k) in the flat dim.
        # ws_off = (grp_id * split_k + kid) * wg_M * wg_N + c_m * wg_N + c_n.
        c0 = b.const_i32(0)
        c1 = b.const_i32(1)
        acc_init = b.const_f32(0.0)

        for_op = b.scf_for_iter(
            c0,
            split_k_param,
            c1,
            [("acc", acc_init)],
            iv_name="kid",
        )
        with for_op as (kid, iter_vars):
            acc_in = iter_vars[0]
            # ws_off = (grp_id * split_k + kid) * wg_M * wg_N + c_m * wg_N + c_n
            grp_stride = b.mul(wg_M_param, wg_N_param)
            slice_idx = b.add(b.mul(grp_id, split_k_param), kid)
            slice_base = b.mul(slice_idx, grp_stride)
            elem_off = b.add(slice_base, b.add(b.mul(c_m, wg_N_param), c_n))
            partial = b.global_load_f32(ws_ptr, elem_off)
            new_acc = b.fadd(acc_in, partial)
            b.scf_yield(new_acc)

        total = for_op.results[0]

        # dw_off = grp_id * wg_M * wg_N + c_m * wg_N + c_n
        grp_dw_base = b.mul(grp_id, b.mul(wg_M_param, wg_N_param))
        dw_off = b.add(grp_dw_base, b.add(b.mul(c_m, wg_N_param), c_n))
        if _is_fp32_out:
            # Accumulator is already f32 — plain store, no conversion needed.
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
        .scalar("split_k", "i32")
        .scalar("ws_bytes", "i32")
        .scalar("dw_bytes", "i32")
        .scalar("groups", "i32")
        .build()
    )
