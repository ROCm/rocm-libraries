# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Workspace-fold kernel for the two-stage wgrad path.

Stage 2 of the two-stage backward-weight convolution.  Stage 1
(``conv_implicit_gemm_wgrad`` with ``two_stage=True``) f32-atomic-adds its
partial sums into a scratch buffer of shape ``[groups * R, wg_M, wg_N]``, where
``R`` is ``ws_replicas``: each of a group's ``split_k`` slices picks one of that
group's ``R`` replica slabs (by ``block_id_z % R``) and atomically accumulates
into it.  So by the time this kernel runs the reduction over ``split_k`` is
complete *within* each slab, but the ``R`` slabs of a group still have to be
summed.  This kernel folds them and converts f32 -> ``dtype_d`` into ``dW``.

``R`` is a compile-time constant, so the fold is flat and unrolled -- ``R``
independent loads, ``R - 1`` adds, one convert, one store per output element,
no loop.  (It did once carry a *sequential* reduction over a
``[groups * split_k, wg_M, wg_N]`` scratch, i.e. a loop over the reduction
degree.  The grid here is sized by the *output*, which for a wgrad is a
filter-sized handful of elements, so that loop ran on a few CTAs with one
dependent load per iteration and cost more than the Stage 1 GEMM it was
reducing.  The replica count is deliberately decoupled from ``split_k`` so this
stage stays a fixed handful of loads however deep the split gets.)

The replicas exist for Stage 1's sake, not this kernel's: a dW-sized scratch is
a few dozen cache lines, and pointing every CTA's atomics at it serialises them
in L2.  See the ``ws_replicas`` field docs on ``WgradConvSpec``.

``R`` here MUST match the Stage 1 spec's ``ws_replicas`` -- folding fewer slabs
silently drops part of the sum, folding more reads past the buffer.

For grouped convolutions the group slabs are contiguous in both buffers, so
``block_id_z`` indexes group ``g``'s ``R`` scratch slabs and its single ``dW``
slab from the same index.

Kernel signature::

    ws_ptr  : f32 global ptr, readonly      — scratch [groups * R * wg_M * wg_N]
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
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm_wgrad import (
    _DEFAULT_WS_REPLICAS,
    _wg_M,
    _wg_N,
)


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
    # Elements per thread along N ("v<n>", 4: one 16-byte f32 load per slab
    # and one 8-byte bf16 / 16-byte f32 store per thread). Needs wg_N % vec
    # == 0 (the launcher checks). Default 1: the scalar kernel, unchanged.
    vec: int = 1
    # r5: 1 = the fold also stores zeros back into every scratch element it
    # read ("rz1"), leaving the scratch zero for the next launch -- the
    # launcher then skips its per-launch scratch clear (the allocation clear
    # covers the first). vec > 1 path only. Default 0: unchanged.
    rezero: int = 0
    # The Stage 1 slabs hold bf16 partials (WgradConvSpec.ws_store 2): load
    # bf16, fold in f32. Needs vec > 1. Default False: unchanged.
    ws_bf16: bool = False
    # Deterministic many-slab fold (det-wgrad): > 0 = the R slabs are folded
    # by a loop instead of the flat unroll -- ``fold_loop`` slab loads per
    # iteration, each thread summing its slabs strictly in slab order -- for
    # the per-slice slabs of ws_store at large split degrees, where R loads
    # per thread would not fit the registers. Needs vec > 1. Tags ``fl<n>``.
    # Default 0: unchanged.
    fold_loop: int = 0
    # With fold_loop: the slabs are spread over ``slab_par`` thread groups
    # of tile_m * tile_n threads (group p folds slabs p, p + P, ... in
    # order); the P partials are then summed in group order through LDS.
    # The workgroup is tile_m * tile_n * slab_par threads. Tags ``sp<n>``.
    # Default 1.
    slab_par: int = 1

    @property
    def block_size(self) -> int:
        return self.tile_m * self.tile_n * max(1, self.slab_par)

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
        if self.vec > 1:
            parts.append(f"v{self.vec}")
        if self.rezero:
            parts.append(f"rz{self.rezero}")
        if self.ws_bf16:
            parts.append("wsb")
        if self.fold_loop:
            parts.append(f"fl{self.fold_loop}")
        if self.slab_par > 1:
            parts.append(f"sp{self.slab_par}")
        parts.append(self.dtype_d)
        return kernel_name_join(self.name, *parts)


def build_conv_wgrad_workspace_reduce(
    spec: WgradReduceSpec,
    *,
    arch: str = "gfx950",  # noqa: ARG001 — reserved for future arch dispatch
) -> KernelDef:
    """Build the IR for the Stage 2 workspace-cast kernel.

    Each workgroup covers a ``(tile_m, tile_n)`` patch of ``(wg_M, wg_N)``.
    Within the patch, each thread owns one ``(m, n)`` element: it loads that
    element from each of its group's ``spec.ws_replicas`` f32 scratch slabs,
    sums them, converts to ``dtype_d``, and stores to ``dW``.

    Stage 1's f32 atomics already reduced over ``split_k`` within each slab, so
    the only accumulation left is the fixed fold across the replicas.  At
    ``ws_replicas == 1`` that collapses to a single load and the kernel is a
    pure cast.
    """
    tile_m = spec.tile_m
    tile_n = spec.tile_n
    BS = spec.block_size  # tile_m * tile_n — one thread per output element
    _is_fp32_out = spec.dtype_d in ("fp32", "f32")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    if spec.rezero and spec.vec <= 1:
        raise ValueError("WgradReduceSpec.rezero needs vec > 1")
    if spec.rezero and spec.ws_bf16:
        raise ValueError("WgradReduceSpec.rezero stores f32 zeros; not with ws_bf16 scratch")
    ws_ptr = (
        b.param("ws_ptr", PtrType(F32, "global"), noalias=True, align=16)
        if spec.rezero
        else b.param(
            "ws_ptr", PtrType(F32, "global"), noalias=True, readonly=True, align=16
        )
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
    if spec.fold_loop:
        _emit_loop_fold(b, spec, ws_ptr, dw_ptr, wg_M_param, wg_N_param, tid)
        return b.kernel

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

    if spec.vec > 1:
        _emit_vec_fold(b, spec, ws_ptr, dw_ptr, wg_M_param, wg_N_param, c_m, c_n, grp_id)
        return b.kernel

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


def _emit_vec_fold(b, spec, ws_ptr, dw_ptr, wg_M_param, wg_N_param, c_m, c_n, grp_id):
    """``vec`` consecutive N elements per thread (wg_N % vec == 0): one
    ``vec``-wide f32 load per slab, the fold, one ``vec``-wide store."""
    from rocke.helpers.io import io_ir_type, pack_f32_to

    v = spec.vec
    c_n = b.mul(c_n, b.const_i32(v))
    in_bounds = b.land(b.cmp_lt(c_m, wg_M_param), b.cmp_lt(c_n, wg_N_param))
    with b.scf_if(in_bounds):
        grp_stride = b.mul(wg_M_param, wg_N_param)
        elem_in_slab = b.add(b.mul(c_m, wg_N_param), c_n)
        dw_off = b.add(b.mul(grp_id, grp_stride), elem_in_slab)
        reps = spec.ws_replicas
        ws_base = (
            dw_off
            if reps == 1
            else b.add(b.mul(b.mul(grp_id, b.const_i32(reps)), grp_stride), elem_in_slab)
        )
        if spec.ws_bf16:
            from rocke.core.ir import BF16

            parts = [
                b.global_load_vN(
                    ws_ptr,
                    ws_base if r == 0 else b.add(ws_base, b.mul(b.const_i32(r), grp_stride)),
                    BF16,
                    v,
                )
                for r in range(reps)
            ]
            tot = [b.cast_to_f32(b.vec_extract(parts[0], i)) for i in range(v)]
            for pv in parts[1:]:
                tot = [b.fadd(t, b.cast_to_f32(b.vec_extract(pv, i))) for i, t in enumerate(tot)]
        else:
            parts = [
                b.global_load_vN(
                    ws_ptr,
                    ws_base if r == 0 else b.add(ws_base, b.mul(b.const_i32(r), grp_stride)),
                    F32,
                    v,
                )
                for r in range(reps)
            ]
            tot = [b.vec_extract(parts[0], i) for i in range(v)]
            for pv in parts[1:]:
                tot = [b.fadd(t, b.vec_extract(pv, i)) for i, t in enumerate(tot)]
        if spec.rezero:
            z = b.zero_vec_f32(v)
            for r in range(reps):
                off = ws_base if r == 0 else b.add(ws_base, b.mul(b.const_i32(r), grp_stride))
                b.global_store_vN(ws_ptr, off, z, v)
        if spec.dtype_d in ("fp32", "f32"):
            b.global_store_vN(dw_ptr, dw_off, b.vec_pack(tot, F32), v)
        else:
            b.global_store_vN(dw_ptr, dw_off, pack_f32_to(b, tot, dtype=spec.dtype_d), v)


def _emit_loop_fold(b, spec, ws_ptr, dw_ptr, wg_M_param, wg_N_param, tid):
    """``fold_loop``: deterministic fold of R = ws_replicas slabs (ungrouped).

    Thread (p, e): e = tid % (tile_m * tile_n) owns ``vec`` consecutive N
    elements of the tile (as the vector fold), p = tid // (tile_m * tile_n)
    folds slabs p, p + P, p + 2P, ... in that order into an f32 accumulator.
    The P partials are added in p order (LDS). Every association is fixed at
    compile time: the result is bitwise reproducible.
    """
    from rocke.helpers.io import pack_f32_to

    if spec.vec <= 1:
        raise ValueError("WgradReduceSpec.fold_loop needs vec > 1")
    if spec.rezero or spec.groups > 1:
        raise ValueError("WgradReduceSpec.fold_loop: no rezero, ungrouped only")
    v = spec.vec
    R = spec.ws_replicas
    P = max(1, spec.slab_par)
    U = spec.fold_loop
    TE = spec.tile_m * spec.tile_n
    p_v = b.div(tid, b.const_i32(TE)) if P > 1 else None
    e_v = b.mod(tid, b.const_i32(TE)) if P > 1 else tid
    t_m = b.div(e_v, b.const_i32(spec.tile_n))
    t_n = b.mod(e_v, b.const_i32(spec.tile_n))
    c_m = b.add(b.mul(b.block_id_y(), b.const_i32(spec.tile_m)), t_m)
    c_n = b.mul(b.add(b.mul(b.block_id_x(), b.const_i32(spec.tile_n)), t_n), b.const_i32(v))
    in_bounds = b.land(b.cmp_lt(c_m, wg_M_param), b.cmp_lt(c_n, wg_N_param))
    slab = b.mul(wg_M_param, wg_N_param)
    elem = b.add(b.mul(c_m, wg_N_param), c_n)
    # Out-of-bounds threads read element 0 (in range) and store nothing; the
    # LDS exchange needs every thread at the barrier.
    elem_safe = b.select(in_bounds, elem, b.const_i32(0))
    base = elem_safe if P == 1 else b.add(elem_safe, b.mul(p_v, slab))
    step_slabs = P * U  # slabs per loop iteration (all groups)
    n_main = R // step_slabs
    if spec.ws_bf16:
        from rocke.core.ir import BF16

        def load(off):
            pv = b.global_load_vN(ws_ptr, off, BF16, v)
            return [b.cast_to_f32(b.vec_extract(pv, i)) for i in range(v)]
    else:
        def load(off):
            pv = b.global_load_vN(ws_ptr, off, F32, v)
            return [b.vec_extract(pv, i) for i in range(v)]

    acc = None
    if n_main:
        stride_it = b.mul(slab, b.const_i32(step_slabs))
        init = b.zero_vec_f32(v)
        loop = b.scf_for_iter(
            b.const_i32(0), b.const_i32(n_main), b.const_i32(1),
            [("fold_acc", init)], iv_name="fold_it",
        )
        with loop as (it, (acc_v,)):
            off0 = b.add(base, b.mul(it, stride_it))
            vals = [
                load(off0 if u == 0 else b.add(off0, b.mul(slab, b.const_i32(u * P))))
                for u in range(U)
            ]
            tot = [b.vec_extract(acc_v, i) for i in range(v)]
            for vv in vals:
                tot = [b.fadd(t, x) for t, x in zip(tot, vv)]
            b.scf_yield(b.vec_pack(tot, F32))
        acc = [b.vec_extract(loop.results[0], i) for i in range(v)]
    # Remainder slabs r = n_main * P * U + u * P + p (< R), in order.
    rem = R - n_main * step_slabs
    if rem:
        zero = b.const_f32(0.0)
        r0 = n_main * step_slabs
        for u in range(-(-rem // P)):
            r_first = r0 + u * P  # slab of group 0
            off = b.add(base, b.mul(slab, b.const_i32(r_first)))
            if r_first + P <= R:
                vals = load(off)
            else:
                ok = b.cmp_lt(p_v, b.const_i32(R - r_first)) if P > 1 else None
                # clamp the slab to the last one for the groups past R
                off_c = off if ok is None else b.select(ok, off, b.add(elem_safe, b.mul(slab, b.const_i32(R - 1))))
                vals = load(off_c)
                if ok is not None:
                    vals = [b.select(ok, x, zero) for x in vals]
            acc = vals if acc is None else [b.fadd(t, x) for t, x in zip(acc, vals)]
    if P > 1:
        sm = b.smem_alloc_f32([P, TE * v], name_hint="fold_part")
        b.smem_store_vN_f32(sm, [p_v, b.mul(e_v, b.const_i32(v))], b.vec_pack(acc, F32), v)
        b.sync()
        with b.scf_if(b.land(in_bounds, b.cmp_eq(p_v, b.const_i32(0)))):
            tot = None
            for q in range(P):
                pv = b.smem_load_vN_f32(sm, b.const_i32(q), b.mul(e_v, b.const_i32(v)), n=v)
                xs = [b.vec_extract(pv, i) for i in range(v)]
                tot = xs if tot is None else [b.fadd(t, x) for t, x in zip(tot, xs)]
            _fold_store(b, spec, dw_ptr, elem, tot, v, pack_f32_to)
    else:
        with b.scf_if(in_bounds):
            _fold_store(b, spec, dw_ptr, elem, acc, v, pack_f32_to)


def _fold_store(b, spec, dw_ptr, off, tot, v, pack_f32_to):
    if spec.dtype_d in ("fp32", "f32"):
        b.global_store_vN(dw_ptr, off, b.vec_pack(tot, F32), v)
    else:
        b.global_store_vN(dw_ptr, off, pack_f32_to(b, tot, dtype=spec.dtype_d), v)


def wgrad_reduce_grid(spec: WgradReduceSpec) -> Tuple[int, int, int]:
    """Return the ``(x, y, z)`` grid dimensions for this reduce spec."""
    return (
        ceil_div(spec.wg_N, spec.tile_n * spec.vec),
        ceil_div(spec.wg_M, spec.tile_m),
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
