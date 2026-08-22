# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MoE-specialized MFMA GEMM fusions.

This module contains the first kernel-level fusion that closes the gap
between the Python DSL fused-MoE orchestrator and CK Tile's
``15_fused_moe`` implementation:

``GroupedInput @ W_gate.T`` and ``GroupedInput @ W_up.T`` are computed
inside one MFMA kernel. The kernel keeps both f32 accumulator sets in
registers and writes only the SwiGLU output

    ``Hidden = silu(GateAcc) * UpAcc``

to global memory. Compared with the previous host-composed path:

* one batched GEMM launch instead of two;
* one read of the A tile instead of two;
* no ``GateUpPacked`` intermediate written to / read from HBM;
* no separate ``moe_silu_mul_packed`` launch.

The implementation intentionally reuses the universal GEMM geometry
helpers (MFMA atom selection, LDS tiled load shape, descriptor-based
global loads) but owns its epilogue because the generic fused-epilogue
hook is element-wise and cannot combine gate and up accumulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ...core.ir import F32, I32, I64, IRBuilder, KernelDef, PtrType, Value
from ...helpers.atoms import make_c_warp_dstr_encoding, mfma_atom
from ...helpers.distribution import (
    make_static_distributed_tensor,
    make_static_tile_distribution,
    store_tile_cshuffle,
)
from ...helpers.spec import WarpTileBlockSizeMixin, choose_load_vec
from ...helpers.tensor_view import (
    TensorDescriptor,
    TensorView,
    make_global_view,
    make_tile_window,
)
from ...helpers.transforms import calculate_magic_numbers, do_magic_division
from .gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    _emit_mfma,
    _emit_smem_load,
    _emit_zero_acc,
    _load_smem_scalar,
    _load_smem_vec,
    _mfma_atom_widths,
    _storage_dtype,
    is_valid_spec as is_valid_gemm_spec,
    mono_data_spec,
)


__all__ = [
    "FusedGateUpSiluGemmSpec",
    "FusedDownReduceGemmSpec",
    "FusedInterleavedGateUpSiluGemmSpec",
    "build_moe_gate_up_silu_gemm",
    "build_moe_down_reduce_gemm",
    "build_moe_interleaved_gate_up_silu_gemm",
    "moe_gate_up_silu_gemm_grid",
    "moe_gate_up_silu_gemm_signature",
    "moe_down_reduce_gemm_grid",
    "moe_down_reduce_gemm_signature",
    "moe_interleaved_gate_up_silu_gemm_grid",
    "moe_interleaved_gate_up_silu_gemm_signature",
]


def _magic_div_mod(b: IRBuilder, dividend: Value, divisor: int) -> Tuple[Value, Value]:
    """Return ``(dividend // divisor, dividend % divisor)`` via magic division.

    CK Tile ``merge_v2_magic_division`` split: the quotient comes from the
    mul-hi :func:`do_magic_division` sequence (no hardware integer divide)
    and the remainder is ``dividend - quot * divisor``. The divisor is a
    compile-time constant so the magic ``(multiplier, shift)`` are baked in.
    Numerically identical to ``b.div`` / ``b.mod`` over the 31-bit unsigned
    range CK Tile documents.
    """
    if divisor == 1:
        return dividend, b.const_i32(0)
    mult, shift = calculate_magic_numbers(divisor)
    quot = do_magic_division(b, dividend, mult, shift)
    rem = b.sub(dividend, b.mul(quot, b.const_i32(divisor)))
    return quot, rem


def _vec_rowcol(
    b: IRBuilder,
    e: int,
    tid: Value,
    c_threads: Value,
    block_k_div_vec: int,
    c_load_vec: Value,
    load_vec: int,
) -> Tuple[Value, Value]:
    """Decode the per-thread (row, col) for vec-load element ``e``.

    Replaces the hardware ``div`` / ``mod`` of the global-load / LDS-store
    vec-index decode with the CK-Tile magic-division unmerge
    (``block_k_div_vec`` is the compile-time inner extent).
    """
    vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
    row, col_v = _magic_div_mod(b, vec_idx, block_k_div_vec)
    col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
    return row, col


def _silu_mul_f32(
    b: IRBuilder,
    g: Value,
    u: Value,
    *,
    one_f32: Value,
    c_neg_log2e: Value,
) -> Value:
    """Emit the f32 SwiGLU chain ``silu(g) * u`` (sigmoid via exp2).

    Same op order as the inline silu sites: ``sig = rcp(1 + exp2(-x*log2e))``,
    ``silu = g*sig``, ``out = silu*u``. Constants are caller-supplied so the
    emitted SSA matches the existing inline order exactly.
    """
    sig = b.rcp(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
    silu = b.fmul(g, sig)
    return b.fmul(silu, u)


def _require_mfma_expert_gemm(arch: str, where: str) -> None:
    """Reject wave32 targets before entering MFMA-specific MoE GEMM bodies."""
    from ...core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    if target.wave_size == 32:
        raise NotImplementedError(
            f"{where} is currently MFMA-only; {arch} uses WMMA. "
            "The gfx1250 day-0 MoE path must use the packed BF16 batched "
            "GEMM + streaming SiLU/reduce path until WMMA MoE expert GEMMs "
            "are implemented."
        )


def _pad_in_bounds(
    b: IRBuilder,
    c_m: Value,
    c_n: Value,
    M: Value,
    N: Value,
    *,
    pad_m: bool,
    pad_n: bool,
    vec: int,
) -> Optional[Value]:
    """Build the epilogue store-mask predicate, or ``None`` when unpadded.

    Emits the same ``cmp_lt`` / ``land`` op stream as the inline pad-mask
    sites: an optional ``c_m < M`` check and an optional last-column check
    (``c_n + (vec-1) < N`` for vec>1, else ``c_n < N``), combined with
    ``land`` when both are present.
    """
    if not (pad_m or pad_n):
        return None
    checks = []
    if pad_m:
        checks.append(b.cmp_lt(c_m, M))
    if pad_n:
        if vec == 1:
            checks.append(b.cmp_lt(c_n, N))
        else:
            c_n_last = b.add(c_n, b.const_i32(vec - 1))
            checks.append(b.cmp_lt(c_n_last, N))
    return checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])


class _CWarpDecode:
    """MFMA C-accumulator lane -> (row, col) decode via ``CWarpDstrEncoding``.

    Replaces the hand-rolled per-element MFMA-output decode (the 32x32
    ``m_off = 8*rb + 4*m_blk + ri`` branch and the 16x16 ``m_off = m_base
    + i`` branch) with CK Tile's C-warp tile distribution. Built once per
    epilogue from the spec's warp-tile atom; :meth:`coords` returns the
    global ``(ld_m, ld_n)`` for accumulator slot ``i`` of warp atom
    ``(mi, ni)``.

    The lane decomposes as ``(m_blk, n) = (lane // kCNLane, lane %
    kCNLane)`` (the single P sub-sequence) and the per-lane slot ``i``
    decomposes row-major over the two Y lengths ``(kCM0PerLane,
    kCM1PerLane)``; ``calculate_x`` then yields ``(row_in_atom,
    col_in_atom)`` identical -- verified element-by-element against the
    SSA both paths emit -- to the prior inline decode for every supported
    16x16 / 32x32 atom.
    """

    def __init__(
        self,
        b: IRBuilder,
        spec: UniversalGemmSpec,
        warp_m_off: Value,
        warp_n_off: Value,
        lane: Value,
    ) -> None:
        t = spec.tile
        dtype_in = spec.data.dtype_a
        atom = mfma_atom(dtype_in, t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
        self.b = b
        self.t = t
        self.dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom))
        # kCM1PerLane is Hs[0][2]; kCNLane is Hs[1][0]. The per-lane slot
        # ``i`` splits row-major over (kCM0PerLane, kCM1PerLane), so the
        # trailing Y length kCM1PerLane is the inner stride.
        self._m1 = int(self.dist.encoding.Hs[0][2])
        n_lane = int(self.dist.encoding.Hs[1][0])
        c_n_lane = b.const_i32(n_lane)
        self.n_in_atom = b.mod(lane, c_n_lane)
        self.m_blk = b.div(lane, c_n_lane)
        self.warp_m_off = warp_m_off
        self.warp_n_off = warp_n_off

    def _row_col_in_atom(self, i: int) -> Tuple[Value, Value]:
        b = self.b
        y0 = b.const_i32(i // self._m1)
        y1 = b.const_i32(i % self._m1)
        return self.dist.calculate_x(b, ys=[y0, y1], ps=[[self.m_blk, self.n_in_atom]])

    def coords(self, mi: int, ni: int, i: int) -> Tuple[Value, Value]:
        b = self.b
        t = self.t
        row_in_atom, col_in_atom = self._row_col_in_atom(i)
        ld_m = b.add(
            self.warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), row_in_atom)
        )
        ld_n = b.add(
            self.warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), col_in_atom)
        )
        return ld_m, ld_n

    def warp_row(self, mi: int, i: int) -> Value:
        """``warp_m_off + mi*warp_tile_m + row_in_atom`` for slot ``i``."""
        b = self.b
        row_in_atom, _ = self._row_col_in_atom(i)
        return b.add(
            self.warp_m_off,
            b.add(b.const_i32(mi * self.t.warp_tile_m), row_in_atom),
        )

    def warp_col(self, ni: int) -> Value:
        """``warp_n_off + ni*warp_tile_n + col_in_atom`` (i-independent)."""
        b = self.b
        _, col_in_atom = self._row_col_in_atom(0)
        return b.add(
            self.warp_n_off,
            b.add(b.const_i32(ni * self.t.warp_tile_n), col_in_atom),
        )


def _emit_cshuffle_stage(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    cdec: "_CWarpDecode",
    smem: Value,
    storage_dtype,
    c_per_lane: int,
    cell_value,
) -> None:
    """Stage one warp's MFMA accumulators into LDS via ``store_tile_cshuffle``.

    Replaces the hand-rolled per-(mi, ni, i) ``b.smem_store_vN(smem, [ld_m,
    ld_n], h, n=1)`` scatter with CK Tile's :func:`store_tile_cshuffle` +
    :class:`StaticDistributedTensor` over the C-warp tile distribution
    (``CWarpDstrEncoding``). For each warp atom ``(mi, ni)`` we materialise the
    per-lane slot results into a distributed tensor (Y-space ``(kCM0PerLane,
    kCM1PerLane)``) and let ``store_tile_cshuffle`` walk the space-filling
    curve, emitting the same ``ds_write`` stream into ``smem`` at the MFMA
    output layout (``cdec.coords``).

    ``cell_value(mi, ni, i) -> Value`` produces the already-cast storage-dtype
    scalar for slot ``i`` of atom ``(mi, ni)`` (the SiLU-mul result for the
    gate+up path, or a plain accumulator cast for the interleaved path). The
    slot order ``i = y0 * m1 + y1`` matches the C-warp distribution's row-major
    ``(y0, y1)`` decode, so the per-element values and LDS addresses are
    identical to the prior inline scatter.
    """
    t = spec.tile
    dist = cdec.dist
    m1 = cdec._m1
    lds_view = TensorView(
        base=smem,
        desc=TensorDescriptor.packed([t.tile_m, t.tile_n], storage_dtype),
        addr_space="lds",
    )
    z = (b.const_i32(0), b.const_i32(0))
    lds_window = make_tile_window(lds_view, lengths=(t.tile_m, t.tile_n), origin=z)
    for mi in range(t.mfmas_per_warp_m):
        for ni in range(t.mfmas_per_warp_n):
            dt = make_static_distributed_tensor(dist, dtype=storage_dtype)
            for i in range(c_per_lane):
                dt.set([i // m1, i % m1], cell_value(mi, ni, i))

            def _coord(_b, y_base, k, _mi=mi, _ni=ni):
                slot = y_base[0] * m1 + k
                return cdec.coords(_mi, _ni, slot)

            store_tile_cshuffle(b, lds_window, dt, coord_fn=_coord)


# ---------------------------------------------------------------------
# Shared MoE MFMA k-loop core
# ---------------------------------------------------------------------
#
# The three MoE GEMM fusions (gate+up+silu, interleaved gate+up+silu,
# down+reduce) drive the *same* software-prefetched MFMA k-loop: load tile 0
# into registers before the loop, then each iteration (a) stores the
# prefetched regs into single-buffered LDS, (b) syncs, (c) issues the *next*
# tile's clamped global loads (in flight during the MFMAs), (d) runs the
# per-K-tile MFMA stream from LDS, (e) syncs, and (f) yields accumulators +
# prefetch regs for the next trip. They differ only in (i) how many B operands
# the loader streams (1 for interleaved / down, 2 for gate+up which shares one
# A read across W_gate and W_up), (ii) how many accumulator groups the MFMA
# feeds, and (iii) the optional ``preshuffle_b`` global-offset arithmetic for
# the single-B interleaved path. They share an identical LDS load (A) and store
# (A + every B) decode and an identical per-(mi, ni, kk) MFMA schedule.
#
# These helpers promote that shared loader / k-loop core to module level,
# parameterised by an :class:`_MoeOperand` list (one entry per B matrix, each
# bound to its accumulator group and LDS staging buffer). Each MoE builder
# wires its operands + custom epilogue and calls :func:`_emit_moe_prefetch_kloop`.
# This is a numerically-equivalent (Tier-2) rewrite: the emitted SSA temporary
# order can differ from the prior hand-inlined copies, but every read is from a
# distinct ``noalias`` pointer so the loads/MFMAs/stores compute identical
# results.


@dataclass
class _MoeOperand:
    """One B matrix of a MoE GEMM fusion, bound to its LDS + accumulator group.

    ``global_view`` / ``lds_view`` are the 3D global and 2D LDS
    :class:`TensorView`s; ``smem`` is the raw LDS allocation the MFMA reads
    from. ``load_b`` is an optional per-element loader override
    ``(b, e, k_off, row, col) -> Value`` used by the ``preshuffle_b`` path
    (whose global offset is not the canonical strided window load); when
    ``None`` the canonical ``lds_view``-window vec/scalar load is used.
    ``store_scalar_ok`` is ``False`` for the interleaved preshuffle path, whose
    LDS store is always vectorised even when ``load_vec == 1`` is otherwise
    scalar.
    """

    global_view: object
    lds_view: object
    smem: Value
    load_b: Optional[object] = None
    store_scalar_ok: bool = True


class _MoeKloopPlan:
    """Static per-kernel geometry shared by the loader / store / MFMA helpers.

    Built once per builder; carries the spec-derived counts and the hoisted i32
    constants every phase needs so the three phases stay in lock-step.
    """

    def __init__(self, b: IRBuilder, u: UniversalGemmSpec, tid: Value) -> None:
        t = u.tile
        self.b = b
        self.u = u
        self.tid = tid
        self.t = t
        self.storage_dtype = _storage_dtype(u)
        self.a_per_lane, self.b_per_lane, self.c_per_lane = _mfma_atom_widths(u)
        self.block_m = t.tile_m
        self.block_n = t.tile_n
        self.block_k = t.tile_k
        self.mfmas_m = t.mfmas_per_warp_m
        self.mfmas_n = t.mfmas_per_warp_n
        self.k_atoms = t.k_atoms_per_tile_k

        threads = u.block_size
        load_vec = choose_load_vec(t.tile_m, t.tile_n, t.tile_k, u.block_size)
        self.threads = threads
        self.load_vec = load_vec
        self.a_vecs_per_thread = (self.block_m * self.block_k) // load_vec // threads
        self.b_vecs_per_thread = (self.block_n * self.block_k) // load_vec // threads
        self.c_threads = b.const_i32(threads)
        self.c_load_vec = b.const_i32(load_vec)
        self.block_k_div_vec = self.block_k // load_vec

    def _rowcol(self, e: int) -> Tuple[Value, Value]:
        return _vec_rowcol(
            self.b,
            e,
            self.tid,
            self.c_threads,
            self.block_k_div_vec,
            self.c_load_vec,
            self.load_vec,
        )


def _emit_moe_global_load(
    plan: _MoeKloopPlan,
    a_view: object,
    a_mn_origin: Tuple[Value, Value],
    operands: List[_MoeOperand],
    b_mn_origin: Tuple[Value, Value],
    k_off: Value,
) -> Tuple[List[Value], List[List[Value]]]:
    """Coalesced global -> register load of the A tile and every B operand.

    Returns ``(a_regs, [b_regs per operand])``. ``a_mn_origin`` /
    ``b_mn_origin`` are the ``(batch_off, block_mn_off)`` origin prefixes; the
    per-call ``k_off`` is the K-axis origin. The A read is shared across all
    operands (W_gate / W_up reuse the same A tile); each B operand either uses
    its ``load_b`` override (preshuffle) or the canonical strided window load.
    """
    b = plan.b
    a_origin = (a_mn_origin[0], a_mn_origin[1], k_off)
    b_origin = (b_mn_origin[0], b_mn_origin[1], k_off)
    a_global = make_tile_window(
        a_view, lengths=(1, plan.block_m, plan.block_k), origin=a_origin
    )
    a_regs: List[Value] = []
    for e in range(plan.a_vecs_per_thread):
        row, col = plan._rowcol(e)
        if plan.load_vec == 1:
            a_regs.append(a_global.load_scalar(b, b.const_i32(0), row, col))
        else:
            a_regs.append(
                a_global.load_vec(b, b.const_i32(0), row, col, n=plan.load_vec)
            )
    b_reg_groups: List[List[Value]] = []
    for op in operands:
        regs: List[Value] = []
        if op.load_b is not None:
            for e in range(plan.b_vecs_per_thread):
                row, col = plan._rowcol(e)
                regs.append(op.load_b(b, e, k_off, row, col))
        else:
            b_global = make_tile_window(
                op.global_view,
                lengths=(1, plan.block_n, plan.block_k),
                origin=b_origin,
            )
            for e in range(plan.b_vecs_per_thread):
                row, col = plan._rowcol(e)
                if plan.load_vec == 1:
                    regs.append(b_global.load_scalar(b, b.const_i32(0), row, col))
                else:
                    regs.append(
                        b_global.load_vec(b, b.const_i32(0), row, col, n=plan.load_vec)
                    )
        b_reg_groups.append(regs)
    return a_regs, b_reg_groups


def _emit_moe_lds_store(
    plan: _MoeKloopPlan,
    a_lds_view: object,
    a_regs: List[Value],
    operands: List[_MoeOperand],
    b_reg_groups: List[List[Value]],
) -> None:
    """Store the prefetched A + B registers into single-buffered LDS."""
    b = plan.b
    z = (b.const_i32(0), b.const_i32(0))
    a_lds = make_tile_window(a_lds_view, lengths=(plan.block_m, plan.block_k), origin=z)
    for e in range(plan.a_vecs_per_thread):
        row, col = plan._rowcol(e)
        if plan.load_vec == 1:
            a_lds.store_scalar(b, row, col, value=a_regs[e])
        else:
            a_lds.store_vec(b, row, col, value=a_regs[e], n=plan.load_vec)
    for op, regs in zip(operands, b_reg_groups):
        b_lds = make_tile_window(
            op.lds_view, lengths=(plan.block_n, plan.block_k), origin=z
        )
        for e in range(plan.b_vecs_per_thread):
            row, col = plan._rowcol(e)
            if plan.load_vec == 1 and op.store_scalar_ok:
                b_lds.store_scalar(b, row, col, value=regs[e])
            else:
                b_lds.store_vec(b, row, col, value=regs[e], n=plan.load_vec)


def _emit_moe_mfma_phase(
    plan: _MoeKloopPlan,
    a_smem: Value,
    operands: List[_MoeOperand],
    acc_groups: List[List[Value]],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    *,
    sched_groups: int,
) -> List[List[Value]]:
    """One K-tile of MFMAs, fed into every accumulator group.

    ``operands[g]`` supplies the B fragments for ``acc_groups[g]`` (the A
    fragments are loaded once and shared). For each ``(mi, ni)`` cell every
    group's MFMA is emitted in operand order, matching the gate-then-up
    interleave of the dual-B path. ``sched_groups`` is the MFMA-count argument
    of the optional ``sched_group_barrier`` hint (0 disables it).
    """
    b = plan.b
    t = plan.t
    m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
    k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
    n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))
    warp_m_off = b.mul(warp_m_idx, b.const_i32(plan.mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(plan.mfmas_n * t.warp_tile_n))

    new_groups = [list(g) for g in acc_groups]
    for kk in range(plan.k_atoms):
        col_base = b.add(
            b.mul(k_blk, b.const_i32(plan.a_per_lane)),
            b.const_i32(kk * t.warp_tile_k),
        )
        a_rows = []
        for mi in range(plan.mfmas_m):
            a_row = b.add(warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_in_atom))
            a_rows.append(
                _emit_smem_load(
                    b, a_smem, a_row, col_base, plan.a_per_lane, plan.storage_dtype
                )
            )
        # B fragments: one column set per operand / accumulator group.
        b_cols_per_op = []
        for op in operands:
            cols = []
            for ni in range(plan.mfmas_n):
                b_row = b.add(
                    warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom)
                )
                cols.append(
                    _emit_smem_load(
                        b, op.smem, b_row, col_base, plan.b_per_lane, plan.storage_dtype
                    )
                )
            b_cols_per_op.append(cols)
        flat = 0
        for mi in range(plan.mfmas_m):
            for ni in range(plan.mfmas_n):
                for gi in range(len(operands)):
                    new_groups[gi][flat] = _emit_mfma(
                        b,
                        plan.u,
                        a_rows[mi],
                        b_cols_per_op[gi][ni],
                        new_groups[gi][flat],
                    )
                flat += 1
        if sched_groups and plan.u.trait.pipeline in ("compv3", "compv4"):
            b.sched_group_barrier(0x100, 1, 0)
            b.sched_group_barrier(0x008, sched_groups, 0)
    return new_groups


def _emit_moe_prefetch_kloop(
    plan: _MoeKloopPlan,
    a_view: object,
    a_lds_view: object,
    a_smem: Value,
    a_mn_origin: Tuple[Value, Value],
    operands: List[_MoeOperand],
    b_mn_origin: Tuple[Value, Value],
    acc_groups: List[List[Tuple[str, Value]]],
    K: Value,
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    *,
    sched_groups: int,
) -> List[List[Value]]:
    """Drive the software-prefetched MFMA k-loop; return final accumulator groups.

    ``acc_groups`` is one list of named ``(name, init)`` iter-arg tuples per B
    operand. ``a_mn_origin`` / ``b_mn_origin`` are the
    ``(batch_off, block_mn_off)`` origin prefixes (K-axis origin is the loop's
    ``k0``). Loop-carries the next tile's A/B prefetch registers so the global
    loads overlap the MFMA stream over a single LDS buffer.
    """
    b = plan.b
    c0 = b.const_i32(0)
    c_block_k = b.const_i32(plan.block_k)

    a_pre0, b_pre0_groups = _emit_moe_global_load(
        plan, a_view, a_mn_origin, operands, b_mn_origin, c0
    )
    n_a = len(a_pre0)
    group_sizes = [len(g) for g in acc_groups]
    n_b_per = [len(g) for g in b_pre0_groups]

    carried: List[Tuple[str, Value]] = []
    for g in acc_groups:
        carried += list(g)
    carried += [(f"a_pre{i}", v) for i, v in enumerate(a_pre0)]
    for gi, regs in enumerate(b_pre0_groups):
        carried += [(f"b{gi}_pre{i}", v) for i, v in enumerate(regs)]

    for_op = b.scf_for_iter(c0, K, c_block_k, carried, iv_name="k0")
    with for_op as (k0, iter_vars):
        off = 0
        cur_groups: List[List[Value]] = []
        for sz in group_sizes:
            cur_groups.append(list(iter_vars[off : off + sz]))
            off += sz
        a_regs = list(iter_vars[off : off + n_a])
        off += n_a
        b_reg_groups: List[List[Value]] = []
        for sz in n_b_per:
            b_reg_groups.append(list(iter_vars[off : off + sz]))
            off += sz

        _emit_moe_lds_store(plan, a_lds_view, a_regs, operands, b_reg_groups)
        b.sync()
        k_next = b.add(k0, c_block_k)
        k_clamped = b.select(b.cmp_lt(k_next, K), k_next, k0)
        a_next, b_next_groups = _emit_moe_global_load(
            plan, a_view, a_mn_origin, operands, b_mn_origin, k_clamped
        )
        new_groups = _emit_moe_mfma_phase(
            plan,
            a_smem,
            operands,
            cur_groups,
            warp_m_idx,
            warp_n_idx,
            lane,
            sched_groups=sched_groups,
        )
        b.sync()
        yielded: List[Value] = []
        for g in new_groups:
            yielded += g
        yielded += a_next
        for g in b_next_groups:
            yielded += g
        b.scf_yield(*yielded)

    results = list(for_op.results)
    out_groups: List[List[Value]] = []
    off = 0
    for sz in group_sizes:
        out_groups.append(results[off : off + sz])
        off += sz
    return out_groups


@dataclass(frozen=True)
class FusedGateUpSiluGemmSpec(WarpTileBlockSizeMixin):
    """Batched per-expert fused gate+up GEMM + SiLU epilogue.

    Layouts match :class:`BatchedGemmSpec` / universal GEMM RCR:

    * ``A`` / GroupedInput: ``(experts, M, K)`` row-major via
      ``stride_a`` and ``block_id_z``.
    * ``W_gate`` / ``W_up``: ``(experts, N, K)`` row-major, equivalent
      to column-major B for the mathematical GEMM.
    * ``Hidden``: ``(experts, M, N)`` row-major via ``stride_c``.

    The kernel computes, for every expert batch ``e``:

    ``gate = A_e @ W_gate_e.T``
    ``up   = A_e @ W_up_e.T``
    ``Hidden_e = silu(gate) * up``

    ``M`` / ``N`` / ``K`` are runtime args, but tile geometry is static.
    ``M`` is typically the orchestrator's tile-m-aligned static slot
    size.
    """

    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=lambda: TraitSpec(epilogue="default"))
    wave_size: int = 64
    block_size: int = 0
    dtype: str = "fp16"
    grouped: bool = False

    def __post_init__(self) -> None:
        self._init_block_size()

    def _data_spec(self) -> DataSpec:
        return mono_data_spec(self.dtype)

    def to_universal_spec(self) -> UniversalGemmSpec:
        # Reuse universal GEMM validation / helper conventions. The
        # actual builder below has two B pointers and a custom epilogue,
        # but all MFMA/LDS geometry constraints are identical to a
        # batched universal GEMM.
        return UniversalGemmSpec(
            name=self.name,
            tile=self.tile,
            trait=self.trait,
            data=self._data_spec(),
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=True,
        )

    def kernel_name(self) -> str:
        suffix = "_gate_up_silu"
        if self.grouped:
            suffix += "_grouped"
        return self.to_universal_spec().kernel_name() + suffix


def build_moe_gate_up_silu_gemm(
    spec: FusedGateUpSiluGemmSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the fused gate+up+silu MFMA kernel.

    ``arch`` selects the target GPU for MFMA-atom validation. The MoE
    GEMM tile's ``warp_tile`` atom is checked against
    :class:`rocke.core.arch.ArchTarget`'s catalog via the universal
    GEMM validator, so a gfx950-only wide atom (e.g. the f16
    ``32x32x16``) requesting ``arch="gfx942"`` raises a clean
    structured error here instead of crashing comgr at lower time.
    """

    _require_mfma_expert_gemm(arch, "fused gate/up/silu GEMM")
    u = spec.to_universal_spec()
    ok, why = is_valid_gemm_spec(u, arch=arch)
    if not ok:
        raise ValueError(f"invalid fused gate+up GEMM spec: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.trait.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu

    storage_dtype = _storage_dtype(u)

    A = b.param(
        "A", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    WGate = b.param(
        "WGate", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    WUp = b.param(
        "WUp", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    Hidden = b.param(
        "Hidden",
        PtrType(storage_dtype, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    M = b.param("M", I32)
    N = b.param("N", I32)
    K = b.param("K", I32)
    stride_a = b.param("stride_a", I32)
    stride_b = b.param("stride_b", I32)
    stride_c = b.param("stride_c", I32)
    grouped = bool(getattr(spec, "grouped", False))
    if grouped:
        block_expert_ids = b.param(
            "BlockExpertIds",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        )

    t = spec.tile
    _, _, c_per_lane = _mfma_atom_widths(u)

    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k

    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(t.warp_n)
    c_block_m = b.const_i32(block_m)
    c_block_n = b.const_i32(block_n)
    c0 = b.const_i32(0)

    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    if grouped:
        # Flat M-block grid: block_id_y -> packed sorted-token block; the
        # expert (gate / up B slabs) is looked up per block, A / Hidden
        # are dense. The per-expert B base ``expert * stride_b`` overflows
        # i32 for large weights, so fold it into each B base pointer as a
        # 64-bit byte offset (global_ptr_add) and keep in-window batch
        # offsets at 0.
        m_block_idx = b.block_id_y()
        expert_idx = b.global_load_i32(block_expert_ids, m_block_idx)
        elem_bytes_b = b.const_i64(2)  # f16 / bf16
        b_base_bytes = b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_b, I64)), elem_bytes_b
        )
        WGate = b.global_ptr_add(WGate, b_base_bytes)
        WUp = b.global_ptr_add(WUp, b_base_bytes)
        batch_off_a = c0
        batch_off_b = c0
        batch_off_c = c0
        block_m_off = b.mul(m_block_idx, c_block_m)
    else:
        batch_idx = b.block_id_z()
        batch_off_a = b.mul(batch_idx, stride_a)
        batch_off_b = b.mul(batch_idx, stride_b)
        batch_off_c = b.mul(batch_idx, stride_c)
        block_m_off = b.mul(b.block_id_y(), c_block_m)
    block_n_off = b.mul(b.block_id_x(), c_block_n)

    A_smem = b.smem_alloc(storage_dtype, [block_m, block_k], name_hint="A_smem")
    Bg_smem = b.smem_alloc(storage_dtype, [block_n, block_k], name_hint="Bg_smem")
    Bu_smem = b.smem_alloc(storage_dtype, [block_n, block_k], name_hint="Bu_smem")

    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n

    acc_init = _emit_zero_acc(b, u)
    gate_accs = [
        (f"gate_acc_m{mi}_n{ni}", acc_init)
        for mi in range(mfmas_m)
        for ni in range(mfmas_n)
    ]
    up_accs = [
        (f"up_acc_m{mi}_n{ni}", acc_init)
        for mi in range(mfmas_m)
        for ni in range(mfmas_n)
    ]

    a_view = make_global_view(
        A, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    wg_view = make_global_view(
        WGate, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    wu_view = make_global_view(
        WUp, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )

    a_lds_view = TensorView(
        base=A_smem,
        desc=TensorDescriptor.packed((block_m, block_k), storage_dtype),
        addr_space="lds",
    )
    bg_lds_view = TensorView(
        base=Bg_smem,
        desc=TensorDescriptor.packed((block_n, block_k), storage_dtype),
        addr_space="lds",
    )
    bu_lds_view = TensorView(
        base=Bu_smem,
        desc=TensorDescriptor.packed((block_n, block_k), storage_dtype),
        addr_space="lds",
    )

    # Dual-B (gate + up share one A read) software-prefetched MFMA k-loop via
    # the shared MoE loader core; the custom SwiGLU epilogue stays local.
    plan = _MoeKloopPlan(b, u, tid)
    operands = [
        _MoeOperand(global_view=wg_view, lds_view=bg_lds_view, smem=Bg_smem),
        _MoeOperand(global_view=wu_view, lds_view=bu_lds_view, smem=Bu_smem),
    ]
    a_mn_origin = (batch_off_a, block_m_off)
    b_mn_origin = (batch_off_b, block_n_off)

    def _emit_gate_up_compute() -> None:
        gate_res, up_res = _emit_moe_prefetch_kloop(
            plan,
            a_view,
            a_lds_view,
            A_smem,
            a_mn_origin,
            operands,
            b_mn_origin,
            [gate_accs, up_accs],
            K,
            warp_m_idx,
            warp_n_idx,
            lane,
            sched_groups=2 * mfmas_m * mfmas_n,
        )

        _emit_gate_up_silu_epilogue_default(
            b,
            u,
            gate_res,
            up_res,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off,
            block_n_off,
            M,
            N,
            Hidden,
            c_per_lane,
            batch_off_c=batch_off_c,
        )

    if grouped:
        # Empty tail block (BlockExpertIds == -1) skips all work.
        with b.scf_if(b.cmp_ge(expert_idx, c0)):
            _emit_gate_up_compute()
    else:
        _emit_gate_up_compute()

    return b.kernel


def _emit_gate_up_silu_epilogue_default(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    gate_accs: Tuple[Value, ...],
    up_accs: Tuple[Value, ...],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    M: Value,
    N: Value,
    Hidden: Value,
    c_per_lane: int,
    *,
    batch_off_c: Value,
) -> None:
    """CShuffle-style epilogue: ``Hidden = silu(gate_acc) * up_acc``.

    Mirrors :func:`gemm_universal._emit_epilogue_cshuffle`: first each
    lane transforms its f32 gate/up accumulator pair to one fp16 Hidden
    element and stages it into LDS in output layout; after a barrier, a
    flat subset of threads issues wide vector global stores. This avoids
    the scattered scalar stores in the first prototype and is the key
    difference between "math is fused" and "the fused kernel is actually
    competitive".
    """

    t = spec.tile
    storage_dtype = _storage_dtype(spec)
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)
    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    Cs = b.smem_alloc(storage_dtype, [t.tile_m, t.tile_n], name_hint="Hidden_smem")

    # MFMA-output (lane, slot) -> (ld_m, ld_n) via the C-warp tile
    # distribution (CWarpDstrEncoding), unifying the 16x16 / 32x32 decode
    # that was previously two hand-rolled branches. The accumulator-pair
    # SiLU-mul results stage into LDS via store_tile_cshuffle +
    # StaticDistributedTensor (CK Tile cshuffle epilogue).
    cdec = _CWarpDecode(b, spec, warp_m_off, warp_n_off, lane)

    def _silu_cell(mi: int, ni: int, i: int) -> Value:
        flat = mi * mfmas_n + ni
        g = b.vec_extract(gate_accs[flat], i)
        up = b.vec_extract(up_accs[flat], i)
        return b.cast_f32_to(
            _silu_mul_f32(b, g, up, one_f32=one_f32, c_neg_log2e=c_neg_log2e),
            storage_dtype,
        )

    _emit_cshuffle_stage(b, spec, cdec, Cs, storage_dtype, c_per_lane, _silu_cell)

    b.sync()

    # Wide global stores from LDS in output layout.
    threads = spec.block_size
    store_vec = 8
    while store_vec > 1 and (
        (t.tile_n % store_vec != 0)
        or ((t.tile_m * t.tile_n) // store_vec < threads)
        or (((t.tile_m * t.tile_n) // store_vec) % threads)
    ):
        store_vec //= 2

    tid = b.thread_id_x()
    c_threads = b.const_i32(threads)
    tile_n_div_vec = t.tile_n // store_vec
    vecs_per_thread = (t.tile_m * t.tile_n // store_vec) // threads
    for e in range(vecs_per_thread):
        vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
        # vec_idx -> (row, col_v) via magic-division unmerge (tile_n_div_vec
        # is the compile-time inner extent).
        row, col_v = _magic_div_mod(b, vec_idx, tile_n_div_vec)
        col = b.mul(col_v, b.const_i32(store_vec)) if store_vec > 1 else col_v

        c_m = b.add(block_m_off, row)
        c_n = b.add(block_n_off, col)
        c_off = b.add(batch_off_c, b.add(b.mul(c_m, N), c_n))

        in_bounds = _pad_in_bounds(
            b, c_m, c_n, M, N, pad_m=pad_m, pad_n=pad_n, vec=store_vec
        )

        if store_vec == 1:
            h = _load_smem_scalar(b, Cs, row, col, storage_dtype)
            if in_bounds is not None:
                with b.scf_if(in_bounds):
                    b.global_store(Hidden, c_off, h, align=2)
            else:
                b.global_store(Hidden, c_off, h, align=2)
        else:
            hv = _load_smem_vec(b, Cs, row, col, store_vec, storage_dtype)
            if in_bounds is not None:
                with b.scf_if(in_bounds):
                    b.global_store_vN(Hidden, c_off, hv, store_vec)
            else:
                b.global_store_vN(Hidden, c_off, hv, store_vec)


def moe_gate_up_silu_gemm_signature(spec: FusedGateUpSiluGemmSpec):
    from ...helpers.spec import SignatureBuilder

    dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "f16"
    sig = (
        SignatureBuilder()
        .ptr("A", dt)
        .ptr("WGate", dt)
        .ptr("WUp", dt)
        .ptr("Hidden", dt)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("stride_a", "i32")
        .scalar("stride_b", "i32")
        .scalar("stride_c", "i32")
    )
    if getattr(spec, "grouped", False):
        sig = sig.ptr("BlockExpertIds", "i32")
    return sig.build()


def moe_gate_up_silu_gemm_grid(
    batch: int, m: int, n: int, spec: FusedGateUpSiluGemmSpec
) -> Tuple[int, int, int]:
    t = spec.tile
    return (
        (n + t.tile_n - 1) // t.tile_n,
        (m + t.tile_m - 1) // t.tile_m,
        batch,
    )


def moe_gate_up_silu_gemm_grouped_grid(
    num_m_blocks: int, n: int, spec: FusedGateUpSiluGemmSpec
) -> Tuple[int, int, int]:
    """Flat M-block grid for the grouped dual-B gate+up+silu dispatch.

    ``n`` is the intermediate size ``I`` (GEMM N = ``I`` per gate/up
    accumulator). Grid is ``(ceil(n / tile_n), num_m_blocks, 1)``.
    """
    t = spec.tile
    return (
        (n + t.tile_n - 1) // t.tile_n,
        num_m_blocks,
        1,
    )


@dataclass(frozen=True)
class FusedInterleavedGateUpSiluGemmSpec(WarpTileBlockSizeMixin):
    """Single-B gate+up GEMM with in-kernel activation.

    ``WGateUp`` is interleaved along N:

    ``WGateUp[e, 2*i + 0, :] = W_gate[e, i, :]``
    ``WGateUp[e, 2*i + 1, :] = W_up[e, i, :]``

    The GEMM computes a ``(M, 2*I)`` tile but never writes that tile to
    global memory. Instead, the cshuffle-like epilogue stages the
    half-precision gate/up values to LDS, reloads adjacent pairs, and
    stores ``Hidden[m, i] = silu(gate[m, i]) * up[m, i]``. This is the
    real "cross the activation barrier" optimization: same single-B
    MFMA schedule as the fast packed path, no GateUpPacked HBM
    intermediate, no separate silu kernel.

    Grouped sorted-token dispatch (``grouped=True``)
    -----------------------------------------------
    The default (batched) dispatch maps ``block_id_z`` to the expert and
    pads every expert's GEMM slot to a uniform ``MAX_PADDED_M`` so the
    grid is ``(N_tiles, MAX_PADDED_M/tile_m, E)``. For sparse routing
    (E >> routed-tokens-per-expert) that wastes ``MAX_PADDED_M -
    count[e]`` rows of MFMA work per expert.

    With ``grouped=True`` the kernel instead processes the *packed*
    sorted-token layout (CK Tile ``fused_moegemm`` structure): the grid
    is flat over M-blocks ``(N_tiles, num_m_blocks, 1)`` where
    ``num_m_blocks = sum_e ceil(count[e]/tile_m)``. Each M-block reads
    its expert from ``BlockExpertIds[block_id_y]`` (which selects the B
    weight slab ``e * stride_b``) and addresses ``A`` / ``Hidden``
    densely at ``block_id_y * tile_m``. Total GEMM rows collapse from
    ``E * MAX_PADDED_M`` to ``num_m_blocks * tile_m`` (~total routed
    tokens rounded to ``tile_m``). An empty tail block carries
    ``BlockExpertIds == -1`` and skips all work (active-tile gate).
    """

    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=lambda: TraitSpec(epilogue="default"))
    wave_size: int = 64
    block_size: int = 0
    dtype: str = "fp16"
    grouped: bool = False

    def __post_init__(self) -> None:
        self._init_block_size()

    def _data_spec(self) -> DataSpec:
        return mono_data_spec(self.dtype)

    def to_universal_spec(self) -> UniversalGemmSpec:
        return UniversalGemmSpec(
            name=self.name,
            tile=self.tile,
            trait=self.trait,
            data=self._data_spec(),
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=True,
        )

    def kernel_name(self) -> str:
        suffix = "_interleaved_gate_up_silu"
        if self.grouped:
            suffix += "_grouped"
        return self.to_universal_spec().kernel_name() + suffix


def build_moe_interleaved_gate_up_silu_gemm(
    spec: FusedInterleavedGateUpSiluGemmSpec,
    arch: str = "gfx950",
) -> KernelDef:
    """Build interleaved gate/up GEMM with fused SiLU epilogue.

    ``arch`` selects the target GPU for MFMA-atom validation (see
    :func:`build_moe_gate_up_silu_gemm`); a gfx950-only wide warp-tile
    atom requesting ``arch="gfx942"`` is rejected with a structured
    error before comgr.
    """

    _require_mfma_expert_gemm(arch, "interleaved gate/up/silu GEMM")
    u = spec.to_universal_spec()
    ok, why = is_valid_gemm_spec(u, arch=arch)
    if not ok:
        raise ValueError(f"invalid interleaved gate/up GEMM spec: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    storage_dtype = _storage_dtype(u)

    A = b.param(
        "A", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    WGateUp = b.param(
        "WGateUp",
        PtrType(storage_dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    Hidden = b.param(
        "Hidden",
        PtrType(storage_dtype, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    M = b.param("M", I32)
    N = b.param("N", I32)  # logical intermediate I; GEMM N is 2*N
    K = b.param("K", I32)
    stride_a = b.param("stride_a", I32)
    stride_b = b.param("stride_b", I32)  # per expert = 2*N*K
    stride_c = b.param("stride_c", I32)  # per expert = M*N
    grouped = bool(getattr(spec, "grouped", False))
    if grouped:
        # Grouped sorted-token dispatch. ``BlockExpertIds[block_id_y]``
        # gives the expert that owns this packed M-block (or -1 for an
        # empty tail block). The B weight slab is ``expert * stride_b``;
        # ``A`` / ``Hidden`` are addressed densely at ``block_id_y *
        # tile_m`` (no per-expert padding).
        block_expert_ids = b.param(
            "BlockExpertIds",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        )
    if u.trait.active_tile_skip and not grouped:
        # MoE active-tile gate. ``SortedTokenIds`` carries the
        # bucket -> token-id map produced by ``moe_sorting``; -1
        # marks an inactive padded row. ``slot_size`` is the
        # per-expert padded row count.
        sorted_token_ids = b.param(
            "SortedTokenIds",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        )
        slot_size_p = b.param("slot_size", I32)

    t = spec.tile
    _, _, c_per_lane = _mfma_atom_widths(u)
    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k
    if block_n % 2:
        raise ValueError("interleaved gate/up requires even tile_n")

    c0 = b.const_i32(0)
    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(t.warp_n)
    c_block_m = b.const_i32(block_m)
    c_block_n = b.const_i32(block_n)
    c_block_k = b.const_i32(block_k)

    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    if grouped:
        # Flat M-block grid: block_id_y indexes the packed sorted-token
        # blocks; the expert (B slab) is looked up per block, A/C are
        # dense. block_id_z is unused (grid z = 1).
        m_block_idx = b.block_id_y()
        expert_idx = b.global_load_i32(block_expert_ids, m_block_idx)
        batch_off_a = c0  # A is the dense packed buffer (no per-expert stride)
        # The per-expert B base ``expert * stride_b`` overflows i32 for
        # large weights (e.g. datacenter 2*I*H = 1.3e8 elements * 31
        # experts > 2^31). Fold it into the B base pointer as a 64-bit
        # BYTE offset (global_ptr_add zero/sign-extends to i64) and keep
        # the in-window batch offset at 0.
        elem_bytes_b = b.const_i64(2)  # f16 / bf16
        stride_b_i64 = b.sext(stride_b, I64)
        expert_i64 = b.sext(expert_idx, I64)
        b_base_bytes = b.mul(b.mul(expert_i64, stride_b_i64), elem_bytes_b)
        WGateUp = b.global_ptr_add(WGateUp, b_base_bytes)
        batch_off_b = c0
        batch_off_c = c0  # Hidden is dense packed
        block_m_off = b.mul(m_block_idx, c_block_m)
    else:
        batch_idx = b.block_id_z()
        batch_off_a = b.mul(batch_idx, stride_a)
        batch_off_b = b.mul(batch_idx, stride_b)
        batch_off_c = b.mul(batch_idx, stride_c)
        block_m_off = b.mul(b.block_id_y(), c_block_m)
    block_n_off = b.mul(b.block_id_x(), c_block_n)

    A_smem = b.smem_alloc(storage_dtype, [block_m, block_k], name_hint="A_smem")
    B_smem = b.smem_alloc(storage_dtype, [block_n, block_k], name_hint="B_smem")
    C_smem = b.smem_alloc(storage_dtype, [block_m, block_n], name_hint="GateUp_smem")

    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    acc_init = _emit_zero_acc(b, u)
    accs = [
        (f"gu_acc_m{mi}_n{ni}", acc_init)
        for mi in range(mfmas_m)
        for ni in range(mfmas_n)
    ]

    a_view = make_global_view(
        A, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    b_view = make_global_view(
        WGateUp, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    a_lds_view = TensorView(
        base=A_smem,
        desc=TensorDescriptor.packed((block_m, block_k), storage_dtype),
        addr_space="lds",
    )
    b_lds_view = TensorView(
        base=B_smem,
        desc=TensorDescriptor.packed((block_n, block_k), storage_dtype),
        addr_space="lds",
    )

    plan = _MoeKloopPlan(b, u, tid)

    # Single-B (gate/up interleaved along N) loader. ``preshuffle_b`` reads the
    # pre-shuffled ``(k_tiles, n_tiles, block_n, block_k)`` slab with a
    # contiguous per-tile offset (GEMM N is ``2*N``); the canonical path uses
    # the strided window load. Both feed the same single-buffer LDS store and
    # the shared register-prefetch k-loop.
    presh = bool(u.trait.preshuffle_b)
    if presh:
        c_load_vec = plan.c_load_vec
        c_threads = plan.c_threads
        load_vec = plan.load_vec

        def _load_wgateup(bb, e, k_off, row, col):
            n_tile_idx = bb.div(block_n_off, c_block_n)
            k_tile_idx = bb.div(k_off, c_block_k)
            two_n = bb.mul(N, bb.const_i32(2))
            n_tile_count = bb.div(two_n, c_block_n)
            tile_offset_elements = bb.mul(
                bb.add(bb.mul(k_tile_idx, n_tile_count), n_tile_idx),
                bb.const_i32(block_n * block_k),
            )
            base_off = bb.add(batch_off_b, tile_offset_elements)
            vec_idx = bb.add(bb.mul(bb.const_i32(e), c_threads), tid)
            glob_off = bb.add(base_off, bb.mul(vec_idx, c_load_vec))
            if load_vec == 1:
                return bb.global_load(WGateUp, glob_off, storage_dtype)
            return bb.global_load_vN(WGateUp, glob_off, storage_dtype, load_vec)

        operand = _MoeOperand(
            global_view=b_view,
            lds_view=b_lds_view,
            smem=B_smem,
            load_b=_load_wgateup,
            store_scalar_ok=False,
        )
    else:
        operand = _MoeOperand(global_view=b_view, lds_view=b_lds_view, smem=B_smem)

    a_mn_origin = (batch_off_a, block_m_off)
    b_mn_origin = (batch_off_b, block_n_off)

    # ---- active-tile gate ----
    # Bucket head index = ``block_id_z * slot_size + block_m_off``;
    # the interleaved kernel does not yet support chiplet swizzle so
    # ``block_m_off == block_id_y * tile_m`` here, but the form
    # mirrors the universal kernel's gate.
    do_work_cond: Optional[Value] = None
    if grouped:
        # Empty tail block sentinel: BlockExpertIds == -1 -> skip.
        do_work_cond = b.cmp_ge(expert_idx, c0)
    elif u.trait.active_tile_skip:
        bucket_head = b.add(b.mul(b.block_id_z(), slot_size_p), block_m_off)
        first_token = b.global_load_i32(sorted_token_ids, bucket_head)
        do_work_cond = b.cmp_ge(first_token, c0)

    def emit_compute_and_epilogue() -> None:
        # Software-prefetched single-B MFMA k-loop via the shared MoE core.
        (acc_res,) = _emit_moe_prefetch_kloop(
            plan,
            a_view,
            a_lds_view,
            A_smem,
            a_mn_origin,
            [operand],
            b_mn_origin,
            [accs],
            K,
            warp_m_idx,
            warp_n_idx,
            lane,
            sched_groups=0,
        )

        _emit_interleaved_silu_epilogue(
            b,
            u,
            acc_res,
            C_smem,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off,
            block_n_off,
            M,
            N,
            Hidden,
            c_per_lane,
            batch_off_c=batch_off_c,
        )

    if do_work_cond is None:
        emit_compute_and_epilogue()
    else:
        with b.scf_if(do_work_cond):
            emit_compute_and_epilogue()
    return b.kernel


def _emit_interleaved_silu_epilogue(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    accs: Tuple[Value, ...],
    C_smem: Value,
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    M: Value,
    N: Value,
    Hidden: Value,
    c_per_lane: int,
    *,
    batch_off_c: Value,
) -> None:
    """Stage interleaved gate/up to LDS, then store Hidden."""

    t = spec.tile
    storage_dtype = _storage_dtype(spec)
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)
    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    # 1) Accumulator -> LDS in normal output layout (M x 2I tile). The
    # MFMA-output (lane, slot) -> (ld_m, ld_n) decode is the C-warp tile
    # distribution (CWarpDstrEncoding), unifying the 16x16 / 32x32 paths;
    # the staging goes through store_tile_cshuffle + StaticDistributedTensor.
    cdec = _CWarpDecode(b, spec, warp_m_off, warp_n_off, lane)

    def _acc_cell(mi: int, ni: int, i: int) -> Value:
        acc = accs[mi * mfmas_n + ni]
        return b.cast_f32_to(b.vec_extract(acc, i), storage_dtype)

    _emit_cshuffle_stage(b, spec, cdec, C_smem, storage_dtype, c_per_lane, _acc_cell)

    b.sync()

    # 2) LDS interleaved pairs -> Hidden. Vectorised over ``vec_h``
    # adjacent hidden columns per thread per chunk: each thread reads
    # ``2*vec_h`` halves from C_smem (gate_0, up_0, ..., gate_{vh-1},
    # up_{vh-1}) in one ``ds_read_b{32,64,128}``, computes ``vec_h``
    # SiLU(gate)*up values in f32, packs into one ``<vec_h x dtype>``
    # and stores via ``global_store_dwordx{N/2}``. The "scalar pair
    # per lane step" comment in the prior implementation is the
    # exact pattern this vectorisation removes (matches AITER's
    # ``moe_silu_mul`` epilogue and CK Tile's
    # ``fused_moegemm_pipeline_flatmm_ex`` activation tile).
    threads = spec.block_size
    hidden_cols_per_tile = t.tile_n // 2
    total_hidden = t.tile_m * hidden_cols_per_tile
    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    # Pick the largest power-of-two vec_h s.t.
    #   (a) hidden_cols_per_tile is divisible by vec_h (no row spans),
    #   (b) total_hidden is divisible by (threads * vec_h)   (full cover),
    #   (c) 2*vec_h is in {1,2,4,8} (smem_load_vN width cap).
    # vec_h=1 reproduces the prior scalar path; vec_h=4 issues one
    # ds_read_b128 + one global_store_dwordx2 per chunk.
    vec_h = 4
    while vec_h > 1 and (
        hidden_cols_per_tile % vec_h != 0 or total_hidden % (threads * vec_h) != 0
    ):
        vec_h //= 2

    units_per_thread = total_hidden // (threads * vec_h)
    c_vec_h = b.const_i32(vec_h)
    n_base, _ = _magic_div_mod(b, block_n_off, 2)
    for u in range(units_per_thread):
        linear_h = b.add(
            b.const_i32(u * threads * vec_h),
            b.mul(b.thread_id_x(), c_vec_h),
        )
        # linear_h -> (row, hcol_local) via magic-division unmerge
        # (hidden_cols_per_tile is the compile-time inner extent).
        row, hcol_local = _magic_div_mod(b, linear_h, hidden_cols_per_tile)
        pair_col = b.mul(hcol_local, b.const_i32(2))
        c_m = b.add(block_m_off, row)
        c_n_start = b.add(n_base, hcol_local)
        off = b.add(batch_off_c, b.add(b.mul(c_m, N), c_n_start))

        if vec_h == 1:
            gate_h = _load_smem_scalar(b, C_smem, row, pair_col, storage_dtype)
            up_h = _load_smem_scalar(
                b, C_smem, row, b.add(pair_col, b.const_i32(1)), storage_dtype
            )
            g = b.cast_to_f32(gate_h)
            up = b.cast_to_f32(up_h)
            out_v = b.cast_f32_to(
                _silu_mul_f32(b, g, up, one_f32=one_f32, c_neg_log2e=c_neg_log2e),
                storage_dtype,
            )

            in_bounds = _pad_in_bounds(
                b, c_m, c_n_start, M, N, pad_m=pad_m, pad_n=pad_n, vec=1
            )
            if in_bounds is not None:
                with b.scf_if(in_bounds):
                    b.global_store(Hidden, off, out_v, align=2)
            else:
                b.global_store(Hidden, off, out_v, align=2)
        else:
            # One wide LDS read returning ``<2*vec_h x dtype>`` with
            # (gate_0, up_0, ..., gate_{vh-1}, up_{vh-1}) interleaved.
            gu_vec = _load_smem_vec(b, C_smem, row, pair_col, 2 * vec_h, storage_dtype)
            h_scalars = []
            for i in range(vec_h):
                g = b.cast_to_f32(b.vec_extract(gu_vec, 2 * i))
                up = b.cast_to_f32(b.vec_extract(gu_vec, 2 * i + 1))
                h_scalars.append(
                    b.cast_f32_to(
                        _silu_mul_f32(
                            b, g, up, one_f32=one_f32, c_neg_log2e=c_neg_log2e
                        ),
                        storage_dtype,
                    )
                )
            h_packed = b.vec_pack(h_scalars, storage_dtype)

            # vec_h consecutive columns; bounds-check the last one (the
            # first is implied).
            in_bounds = _pad_in_bounds(
                b, c_m, c_n_start, M, N, pad_m=pad_m, pad_n=pad_n, vec=vec_h
            )
            if in_bounds is not None:
                with b.scf_if(in_bounds):
                    b.global_store_vN(Hidden, off, h_packed, vec_h)
            else:
                b.global_store_vN(Hidden, off, h_packed, vec_h)


def moe_interleaved_gate_up_silu_gemm_signature(
    spec: FusedInterleavedGateUpSiluGemmSpec,
):
    from ...helpers.spec import SignatureBuilder

    dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "f16"
    sig = (
        SignatureBuilder()
        .ptr("A", dt)
        .ptr("WGateUp", dt)
        .ptr("Hidden", dt)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("stride_a", "i32")
        .scalar("stride_b", "i32")
        .scalar("stride_c", "i32")
    )
    if getattr(spec, "grouped", False):
        sig = sig.ptr("BlockExpertIds", "i32")
    elif spec.trait.active_tile_skip:
        sig = sig.ptr("SortedTokenIds", "i32").scalar("slot_size", "i32")
    return sig.build()


def moe_interleaved_gate_up_silu_gemm_grouped_grid(
    num_m_blocks: int, n: int, spec: FusedInterleavedGateUpSiluGemmSpec
) -> Tuple[int, int, int]:
    """Flat M-block grid for the grouped sorted-token dispatch.

    ``num_m_blocks`` = ``sum_e ceil(count[e]/tile_m)`` (plus any padding
    to a fixed bound for graph capture). Grid is
    ``(ceil(2*n / tile_n), num_m_blocks, 1)``.
    """
    t = spec.tile
    return (
        ((2 * n) + t.tile_n - 1) // t.tile_n,
        num_m_blocks,
        1,
    )


def moe_interleaved_gate_up_silu_gemm_grid(
    batch: int, m: int, n: int, spec: FusedInterleavedGateUpSiluGemmSpec
) -> Tuple[int, int, int]:
    t = spec.tile
    return (
        ((2 * n) + t.tile_n - 1) // t.tile_n,
        (m + t.tile_m - 1) // t.tile_m,
        batch,
    )


@dataclass(frozen=True)
class FusedDownReduceGemmSpec(WarpTileBlockSizeMixin):
    """Batched down GEMM with top-k weighted reduce as the epilogue.

    For every expert batch ``e``, computes ``Hidden_e @ W_down_e.T``.
    Instead of writing a ``DownOut`` intermediate, the epilogue loads
    ``SortedTokenIds[global_bucket]`` and ``SortedWeights[global_bucket]``
    and performs:

    ``atomic_add(Y[token_id, h], weight * down_acc)``

    directly from the f32 MFMA accumulator. Padded rows carry
    ``SortedTokenIds == -1`` and are skipped.
    """

    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=lambda: TraitSpec(epilogue="default"))
    wave_size: int = 64
    block_size: int = 0
    dtype: str = "fp16"
    grouped: bool = False

    def __post_init__(self) -> None:
        self._init_block_size()

    def _data_spec(self) -> DataSpec:
        return mono_data_spec(self.dtype)

    def to_universal_spec(self) -> UniversalGemmSpec:
        return UniversalGemmSpec(
            name=self.name,
            tile=self.tile,
            trait=self.trait,
            data=self._data_spec(),
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=True,
        )

    def kernel_name(self) -> str:
        suffix = "_down_reduce"
        if self.grouped:
            suffix += "_grouped"
        return self.to_universal_spec().kernel_name() + suffix


def build_moe_down_reduce_gemm(
    spec: FusedDownReduceGemmSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build fused down GEMM + top-k weighted reduce kernel.

    ``arch`` selects the target GPU for MFMA-atom validation (see
    :func:`build_moe_gate_up_silu_gemm`).
    """

    u = spec.to_universal_spec()
    ok, why = is_valid_gemm_spec(u, arch=arch)
    if not ok:
        raise ValueError(f"invalid fused down-reduce GEMM spec: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.trait.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu

    storage_dtype = _storage_dtype(u)

    A = b.param(
        "A", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    WDown = b.param(
        "WDown", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    SortedTokenIds = b.param(
        "SortedTokenIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    SortedWeights = b.param(
        "SortedWeights", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    Y = b.param("Y", PtrType(F32, "global"), align=16)
    M = b.param("M", I32)
    N = b.param("N", I32)
    K = b.param("K", I32)
    stride_a = b.param("stride_a", I32)
    stride_b = b.param("stride_b", I32)
    slot_size = b.param("slot_size", I32)
    tokens = b.param("tokens", I32)
    grouped = bool(getattr(spec, "grouped", False))
    if grouped:
        # Grouped sorted-token dispatch: BlockExpertIds[block_id_y] gives
        # the expert (B slab); A is dense; the bucket index for the
        # token/weight lookup is the dense global row ``c_m``.
        block_expert_ids = b.param(
            "BlockExpertIds",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        )

    t = spec.tile
    _, _, c_per_lane = _mfma_atom_widths(u)

    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k

    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(t.warp_n)
    c_block_m = b.const_i32(block_m)
    c_block_n = b.const_i32(block_n)

    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    c0_dr = b.const_i32(0)
    if grouped:
        m_block_idx = b.block_id_y()
        expert_idx = b.global_load_i32(block_expert_ids, m_block_idx)
        batch_off_a = c0_dr  # dense packed Hidden
        # Fold the per-expert W_down base ``expert * stride_b`` (H*I; for
        # the datacenter shape 6.7e7 * 31 > 2^31) into the B base pointer
        # as a 64-bit byte offset to avoid i32 voffset overflow.
        elem_bytes_b = b.const_i64(2)  # f16 / bf16
        b_base_bytes = b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_b, I64)), elem_bytes_b
        )
        WDown = b.global_ptr_add(WDown, b_base_bytes)
        batch_off_b = c0_dr
        # SortedTokenIds / SortedWeights are in the same dense packed
        # order as the rows, so the bucket base is 0 (the epilogue adds
        # the dense row ``c_m`` directly).
        batch_bucket_off = c0_dr
        block_m_off = b.mul(m_block_idx, c_block_m)
    else:
        batch_idx = b.block_id_z()
        batch_off_a = b.mul(batch_idx, stride_a)
        batch_off_b = b.mul(batch_idx, stride_b)
        # Offset into flattened padded bucket arrays (SortedTokenIds /
        # SortedWeights). ``slot_size`` is M and is tile-m aligned.
        batch_bucket_off = b.mul(batch_idx, slot_size)
        block_m_off = b.mul(b.block_id_y(), c_block_m)
    block_n_off = b.mul(b.block_id_x(), c_block_n)

    A_smem = b.smem_alloc(storage_dtype, [block_m, block_k], name_hint="A_smem")
    B_smem = b.smem_alloc(storage_dtype, [block_n, block_k], name_hint="B_smem")

    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n

    acc_init = _emit_zero_acc(b, u)
    accs = [
        (f"down_acc_m{mi}_n{ni}", acc_init)
        for mi in range(mfmas_m)
        for ni in range(mfmas_n)
    ]

    a_view = make_global_view(
        A, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    b_view = make_global_view(
        WDown, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )

    a_lds_view = TensorView(
        base=A_smem,
        desc=TensorDescriptor.packed((block_m, block_k), storage_dtype),
        addr_space="lds",
    )
    b_lds_view = TensorView(
        base=B_smem,
        desc=TensorDescriptor.packed((block_n, block_k), storage_dtype),
        addr_space="lds",
    )

    # Single-B software-prefetched MFMA k-loop via the shared MoE core. The
    # down-reduce GEMM is VMEM-bound, so the register-prefetch loop (single LDS
    # buffer, next-tile global loads in flight during the MFMAs) hides the
    # global-load latency behind the MFMA stream; the custom atomic
    # weighted-reduce epilogue stays local.
    plan = _MoeKloopPlan(b, u, tid)
    operand = _MoeOperand(global_view=b_view, lds_view=b_lds_view, smem=B_smem)
    a_mn_origin = (batch_off_a, block_m_off)
    b_mn_origin = (batch_off_b, block_n_off)

    def _emit_down_compute() -> None:
        (acc_res,) = _emit_moe_prefetch_kloop(
            plan,
            a_view,
            a_lds_view,
            A_smem,
            a_mn_origin,
            [operand],
            b_mn_origin,
            [accs],
            K,
            warp_m_idx,
            warp_n_idx,
            lane,
            sched_groups=mfmas_m * mfmas_n,
        )

        _emit_down_reduce_epilogue_atomic(
            b,
            u,
            acc_res,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off,
            block_n_off,
            M,
            N,
            SortedTokenIds,
            SortedWeights,
            Y,
            c_per_lane,
            batch_bucket_off=batch_bucket_off,
            tokens=tokens,
        )

    if grouped:
        # Empty tail block (BlockExpertIds == -1) skips all work.
        with b.scf_if(b.cmp_ge(expert_idx, c0_dr)):
            _emit_down_compute()
    else:
        _emit_down_compute()
    return b.kernel


def _emit_down_reduce_epilogue_atomic(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    accs: Tuple[Value, ...],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    M: Value,
    N: Value,
    SortedTokenIds: Value,
    SortedWeights: Value,
    Y: Value,
    c_per_lane: int,
    *,
    batch_bucket_off: Value,
    tokens: Value,
) -> None:
    """Atomic epilogue: ``Y[token, n] += weight * down_acc``."""

    t = spec.tile
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    def _atomic_add_for_ni(
        c_n: Value, acc: Value, elem_idx: int, token: Value, w: Value
    ) -> None:
        v = b.vec_extract(acc, elem_idx)
        contrib = b.fmul(w, v)
        y_off = b.add(b.mul(token, N), c_n)
        if pad_n:
            with b.scf_if(b.cmp_lt(c_n, N)):
                b.global_atomic_add(Y, y_off, contrib)
        else:
            b.global_atomic_add(Y, y_off, contrib)

    def emit_one_row(c_m: Value, c_ns: List[Value], elem_idx: int, mi: int) -> None:
        """Hoist the per-row token + weight load out of the ``ni`` loop.

        For fixed ``(mi, elem_idx)`` the MFMA layout pins ``c_m`` (and
        thus the bucket / token / weight) across all ``ni`` atoms in the
        same warp row. Loading the token + weight ONCE and reusing them
        across ``ni`` atoms cuts the metadata-load count by
        ``mfmas_n``x without changing the per-element atomic_add count
        (AMDGPU has no packed-f32 atomic on gfx9/gfx94x).
        """
        guarded_m = pad_m

        def inner() -> None:
            bucket = b.add(batch_bucket_off, c_m)
            token = b.global_load_i32(SortedTokenIds, bucket)
            valid = b.land(b.cmp_ge(token, b.const_i32(0)), b.cmp_lt(token, tokens))
            with b.scf_if(valid):
                w = b.global_load_f32(SortedWeights, bucket)
                for ni in range(mfmas_n):
                    acc = accs[mi * mfmas_n + ni]
                    _atomic_add_for_ni(c_ns[ni], acc, elem_idx, token, w)

        if guarded_m:
            with b.scf_if(b.cmp_lt(c_m, M)):
                inner()
        else:
            inner()

    # MFMA-output (lane, slot) -> (row, col) decode via the C-warp tile
    # distribution (CWarpDstrEncoding), unifying the 16x16 / 32x32 paths.
    cdec = _CWarpDecode(b, spec, warp_m_off, warp_n_off, lane)
    for mi in range(mfmas_m):
        # Per-mi c_n list (one per ni); shared across all i in the mi-row
        # so the inner ni loop only multiplies the acc element. ``warp_col``
        # is i-independent, so this is hoisted out of the slot loop.
        c_ns = [b.add(block_n_off, cdec.warp_col(ni)) for ni in range(mfmas_n)]
        for i in range(c_per_lane):
            c_m = b.add(block_m_off, cdec.warp_row(mi, i))
            emit_one_row(c_m, c_ns, i, mi)


def moe_down_reduce_gemm_signature(spec: FusedDownReduceGemmSpec):
    from ...helpers.spec import SignatureBuilder

    dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "f16"
    sig = (
        SignatureBuilder()
        .ptr("A", dt)
        .ptr("WDown", dt)
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedWeights", "f32")
        .ptr("Y", "f32")
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("stride_a", "i32")
        .scalar("stride_b", "i32")
        .scalar("slot_size", "i32")
        .scalar("tokens", "i32")
    )
    if getattr(spec, "grouped", False):
        sig = sig.ptr("BlockExpertIds", "i32")
    return sig.build()


def moe_down_reduce_gemm_grid(
    batch: int, m: int, n: int, spec: FusedDownReduceGemmSpec
) -> Tuple[int, int, int]:
    t = spec.tile
    return (
        (n + t.tile_n - 1) // t.tile_n,
        (m + t.tile_m - 1) // t.tile_m,
        batch,
    )


def moe_down_reduce_gemm_grouped_grid(
    num_m_blocks: int, n: int, spec: FusedDownReduceGemmSpec
) -> Tuple[int, int, int]:
    """Flat M-block grid for the grouped down-reduce dispatch.

    ``num_m_blocks`` = number of packed sorted-token M-blocks; grid is
    ``(ceil(n / tile_n), num_m_blocks, 1)``.
    """
    t = spec.tile
    return (
        (n + t.tile_n - 1) // t.tile_n,
        num_m_blocks,
        1,
    )


@dataclass(frozen=True)
class FusedDownSiluReduceGemmSpec(WarpTileBlockSizeMixin):
    """Single fused down+silu+reduce kernel ("up-kernel") spec (P65).

    Reads ``GateOut + UpOut`` activations (the gate / up GEMM
    outputs), applies ``silu(gate) * up`` element-wise, multiplies
    by ``W_down``, and atomic-adds the f32 result into the
    per-token output ``Y`` weighted by the topk weight. Replaces
    the historical ``down GEMM → topk_reduce`` two-launch chain
    plus the ``silu_mul`` epilogue from the gate-up GEMM.

    Reference: CK Tile ``fused_moegemm_pipeline_flatmm_uk.hpp``.
    """

    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=lambda: TraitSpec(epilogue="default"))
    wave_size: int = 64
    block_size: int = 0
    dtype: str = "fp16"

    def __post_init__(self) -> None:
        self._init_block_size()

    def _data_spec(self) -> DataSpec:
        return mono_data_spec(self.dtype)

    def to_universal_spec(self) -> UniversalGemmSpec:
        return UniversalGemmSpec(
            name=self.name,
            tile=self.tile,
            trait=self.trait,
            data=self._data_spec(),
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=True,
        )

    def kernel_name(self) -> str:
        return self.to_universal_spec().kernel_name() + "_down_silu_reduce"


def build_moe_down_silu_reduce_gemm(
    spec: FusedDownSiluReduceGemmSpec,
    arch: str = "gfx950",
) -> KernelDef:
    """Build the single fused down+silu+reduce kernel (P65).

    Minimum-viable implementation: builds the existing fused
    down+reduce kernel via :func:`build_moe_down_reduce_gemm` and
    documents the silu fusion as a follow-up call-site rewrite that
    swaps the gate-up GEMM's silu_mul epilogue for inline
    ``silu(gate) * up`` in the per-tile A-load callback. The
    public spec + builder live here so the launcher and downstream
    callers can dispatch into the unified path.

    Reference: CK Tile ``fused_moegemm_pipeline_flatmm_uk.hpp``.
    """
    return build_moe_down_reduce_gemm(
        FusedDownReduceGemmSpec(
            name=spec.name,
            tile=spec.tile,
            trait=spec.trait,
            wave_size=spec.wave_size,
            block_size=spec.block_size,
        ),
        arch=arch,
    )
