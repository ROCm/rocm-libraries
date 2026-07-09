# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-launch fused-MoE MEGA-kernel for gfx1250 (WMMA, f16/bf16).

This is the WMMA (wave32, gfx1250-class) analog of the gfx950 MFMA mega-kernel
``instances/common/moe_fused_mega.py``. A SINGLE fused kernel computes, per
threadgroup, the full MoE per-expert path with NO HBM intermediates:

    GEMM0 (gate) + GEMM1 (up) sharing one LDS-resident X tile
      -> SiLU(gate) * up in registers
      -> staged through a PERSISTENT LDS ``Hidden_smem`` (never HBM)
      -> GEMM2 (down) reading ``Hidden_smem`` as the LDS-resident A operand
      -> weighted, token-validity-masked atomic reduce into ``Y``.

It mirrors the gfx950 mega's staging / register-resident / software
double-buffering schemes exactly, but every arch-specific primitive is the
gfx1250 WMMA 16x16x32 atom instead of the CDNA MFMA atom:

* Register-resident software double-buffering: the gate+up k-loop pre-loads
  K-tile 0 into registers, then each trip stores the prefetched regs into the
  single LDS buffer, issues the NEXT tile's clamped global loads (in flight
  during the WMMAs), runs the per-K-tile WMMA stream from LDS, and yields the
  prefetch regs + accumulators -- identical structure to
  ``_emit_moe_prefetch_kloop`` on gfx950.
* LDS-resident shared A: W_gate and W_up share one staged A tile.
* LDS-resident Hidden bridge: SiLU(gate)*up is staged into a persistent LDS
  buffer reused as the down-GEMM A operand (the pyisa ``G_reshape`` is
  implicit: the cshuffle write address and the down WMMA A-read address are the
  same logical ``(m, inter)`` element of a row-major LDS tile).

The WMMA wave32 deltas vs the MFMA wave64 mega (same 16x16x32 atom shape):
``a_frag_len``/``b_frag_len`` = 16 (not 8), ``c_frag_len`` = 8 (not 4),
``block_size`` = warp_m*warp_n*32 = 128 (not 256), and the accumulator scatter
is driven by ``op.c_layout()`` (column-distributed) instead of the MFMA
``_CWarpDecode``. The k-loop / cshuffle / atomic-reduce *structure* is a 1:1
mirror of ``moe_gemm_fused.py`` / ``moe_fused_mega.py``; only the per-lane
fragment math and the matmul intrinsic change.

The coalesced global->register loads + the single-buffered LDS stores are
reused verbatim from ``moe_gemm_fused`` (``_emit_moe_global_load`` /
``_emit_moe_lds_store`` are plan-driven and arch-neutral). This file adds the
WMMA plan, the WMMA dual-B k-loop, the SiLU->Hidden LDS stage, the LDS-A down
k-loop, and the WMMA atomic reduce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from ...core.ir import (
    BF16,
    F16,
    F32,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from ...helpers.schedule import SchedulePolicy, WmmaHotLoopInstList
from ...helpers.spec import choose_load_vec
from ...helpers.tensor_view import (
    TensorDescriptor,
    TensorView,
    make_global_view,
    make_tile_window,
)
from ..common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    _emit_smem_load,
    _emit_zero_acc_op,
    _resolve_mma_op,
    _storage_dtype,
    is_valid_spec as is_valid_gemm_spec,
)
from ..common.moe_gemm_fused import (
    _MoeOperand,
    _emit_moe_global_load,
    _emit_moe_lds_store,
    _silu_mul_f32,
    _vec_rowcol,
)


__all__ = [
    "FusedMegaWmmaSpec",
    "build_moe_fused_mega_wmma",
    "moe_fused_mega_wmma_grid",
    "moe_fused_mega_wmma_signature",
]


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedMegaWmmaSpec:
    """Single-launch fused-MoE WMMA mega-kernel spec (gfx1250, f16/bf16).

    The mega-kernel fuses gate+up+silu+down+reduce for one sorted-token
    M-block over one ``tile_n_inter`` slice of the inter dimension. Tiling
    mirrors the gfx950 mega (BUILD_SPEC Section 1.1) with the WMMA wave32
    block size:

    * ``tile_m`` = sorted tokens per m-block, default 16 (one WMMA M atom).
    * ``tile_n_inter`` = inter columns this TG owns; this is the GEMM0/1 N
      extent AND the GEMM2 contraction extent, default 256.
    * ``tile_k_gu`` = K-loop tile along the hidden contraction H for gate/up.
    * WMMA atom = ``16x16x32`` (the gfx1250 fp16/bf16 atom; K=32).

    With ``tile_m=16, tile_n_inter=256, warp_m=1, warp_n=4`` the block has
    ``4 waves x 32 = 128`` threads. Each warp owns ``mfmas_n = 256/(4*16) = 4``
    N-atoms; ``mfmas_m = 1``; ``k_atoms = tile_k_gu/32 = 1``.

    ``M`` / ``N`` (=inter) / ``K`` (=hidden) / ``H_out`` are runtime args; the
    tile geometry is static.
    """

    name: str
    tile_m: int = 16
    tile_n_inter: int = 256
    tile_k_gu: int = 32
    warp_m: int = 1
    warp_n: int = 4
    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 32
    # Down-GEMM tiling: contraction is tile_n_inter (the inter slice this TG
    # owns), tiled along the down output (H_out) in tile_n_down chunks.
    tile_n_down: int = 256
    tile_k_down: int = 32
    # 'mem' = no scheduler hints (default); 'wmma_v1' = WMMA intrawave
    # ds_read/wmma interleave hints (helpers.schedule.WmmaHotLoopInstList).
    pipeline: str = "mem"
    # Occupancy / latency-hiding knobs (the decode-parity levers):
    #   double_buffer -> LDS ping-pong (one barrier/K-tile instead of two) so the
    #     next K-tile's global->LDS copy overlaps the current tile's WMMAs.
    #   waves_per_eu  -> AMDGPU 'amdgpu-waves-per-eu' occupancy hint.
    double_buffer: bool = False
    waves_per_eu: int | None = None
    trait: TraitSpec = field(
        default_factory=lambda: TraitSpec(epilogue="default", pipeline="mem")
    )
    wave_size: int = 32
    block_size: int = 0
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.block_size == 0:
            object.__setattr__(
                self,
                "block_size",
                self.warp_m * self.warp_n * self.wave_size,
            )
        # The WMMA mega only supports the 'default' epilogue; ``pipeline`` is the
        # scheduler knob ('mem' | 'wmma_v1'). Keep the trait in lock-step and
        # carry the occupancy hint.
        if (
            self.trait.pipeline != self.pipeline
            or self.trait.waves_per_eu != self.waves_per_eu
        ):
            object.__setattr__(
                self,
                "trait",
                TraitSpec(
                    epilogue="default",
                    pipeline=self.pipeline,
                    waves_per_eu=self.waves_per_eu,
                ),
            )

    # -- data / tile spec helpers ----------------------------------------

    def _data_spec(self) -> DataSpec:
        dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
        return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)

    def gate_up_tile(self) -> TileSpec:
        """Block/warp tile for the gate+up GEMM (M x I_slice, contract H)."""
        return TileSpec(
            tile_m=self.tile_m,
            tile_n=self.tile_n_inter,
            tile_k=self.tile_k_gu,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_m,
            warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
        )

    def down_tile(self) -> TileSpec:
        """Block/warp tile for the down GEMM (M x H_out_slice, contract I)."""
        return TileSpec(
            tile_m=self.tile_m,
            tile_n=self.tile_n_down,
            tile_k=self.tile_k_down,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_m,
            warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
        )

    def gate_up_universal_spec(self) -> UniversalGemmSpec:
        return UniversalGemmSpec(
            name=self.name + "_gu",
            tile=self.gate_up_tile(),
            trait=self.trait,
            data=self._data_spec(),
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=True,
        )

    def down_universal_spec(self) -> UniversalGemmSpec:
        return UniversalGemmSpec(
            name=self.name + "_down",
            tile=self.down_tile(),
            trait=self.trait,
            data=self._data_spec(),
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=True,
        )

    def kernel_name(self) -> str:
        return self.gate_up_universal_spec().kernel_name() + "_fused_mega_wmma"


# ---------------------------------------------------------------------------
# Grid + signature
# ---------------------------------------------------------------------------


def moe_fused_mega_wmma_grid(
    num_m_blocks: int, inter: int, spec: FusedMegaWmmaSpec
) -> Tuple[int, int, int]:
    """Mega-kernel launch grid.

    ``grid.x`` splits the down-GEMM contraction (inter dim ``I``) across TGs ->
    each TG produces a partial down result + atomic-adds into Y.
    ``grid.y`` selects the sorted m-block (-> expert id + weight base offsets).

    ``grid = (ceil(inter / tile_n_inter), num_m_blocks, 1)``.
    """
    sub_gu = spec.tile_n_inter
    gx = (inter + sub_gu - 1) // sub_gu
    return (gx, num_m_blocks, 1)


def moe_fused_mega_wmma_signature(spec: FusedMegaWmmaSpec):
    from ...helpers.spec import SignatureBuilder

    dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "bf16"
    return (
        SignatureBuilder()
        .ptr("A", dt)
        .ptr("WGate", dt)
        .ptr("WUp", dt)
        .ptr("WDown", dt)
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedWeights", "f32")
        .ptr("BlockExpertIds", "i32")
        .ptr("Y", "f32")
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("H_out", "i32")
        .scalar("stride_a", "i32")
        .scalar("stride_b_gate", "i32")
        .scalar("stride_b_up", "i32")
        .scalar("stride_b_down", "i32")
        .scalar("slot_size", "i32")
        .scalar("tokens", "i32")
        .build()
    )


# ---------------------------------------------------------------------------
# WMMA k-loop plan + fragment-load primitive
# ---------------------------------------------------------------------------


class _WmmaMoePlan:
    """WMMA analog of ``moe_gemm_fused._MoeKloopPlan``.

    Same static-geometry carrier the loader / store / WMMA phases share, but
    the per-lane fragment widths come from the resolved WMMA op contract
    (``op.a_frag_len`` / ``b_frag_len`` / ``c_frag_len`` = 16 / 16 / 8 for the
    gfx1250 16x16x32 atom) instead of the wave64 ``shape // waves`` MFMA
    formula. The coalesced global-load / LDS-store decode (``a_vecs_per_thread``
    / ``b_vecs_per_thread`` / ``_rowcol``) is byte-identical to the MFMA plan,
    so ``_emit_moe_global_load`` / ``_emit_moe_lds_store`` are reused verbatim.
    """

    def __init__(self, b: IRBuilder, u: UniversalGemmSpec, op, tid: Value) -> None:
        t = u.tile
        self.b = b
        self.u = u
        self.t = t
        self.tid = tid
        self.op = op
        self.storage_dtype = _storage_dtype(u)
        self.a_per_lane = op.a_frag_len
        self.b_per_lane = op.b_frag_len
        self.c_per_lane = op.c_frag_len
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


def _emit_frag_smem_load(
    b: IRBuilder,
    src: Value,
    mn_in_atom: Value,
    k_in_atom: Value,
    atom_mn_base: Value,
    k_tile_base: Value,
    frag_len: int,
    storage_dtype: Type,
) -> Value:
    """Load one ``frag_len``-wide WMMA operand fragment from a row-major LDS tile.

    Module-level mirror of the nested ``gemm_universal._emit_frag_smem_load``.
    Both the A LDS tile ``(block_m, block_k)`` and the B LDS tile
    ``(block_n, block_k)`` are row-major with the M/N index as the row and K as
    the column. One lane's fragment occupies a single tile row
    (``atom_mn_base + mn_in_atom``) and ``frag_len`` contiguous K columns from
    ``k_tile_base + k_in_atom``. fp16/bf16 smem loads cap at 8 lanes, so the
    WMMA ``<16 x half>`` fragment is assembled from two 8-wide chunks.
    """
    lds_row = b.add(atom_mn_base, mn_in_atom)
    lds_col = b.add(k_tile_base, k_in_atom)
    max_vec = 8 if storage_dtype in (F16, BF16) else frag_len
    if frag_len <= max_vec:
        return _emit_smem_load(b, src, lds_row, lds_col, frag_len, storage_dtype)
    frag = None
    for off in range(0, frag_len, max_vec):
        chunk = _emit_smem_load(
            b, src, lds_row, b.add(lds_col, b.const_i32(off)), max_vec, storage_dtype
        )
        frag = chunk if frag is None else b.vec_concat(frag, chunk)
    return frag


def _emit_wmma_moe_phase(
    plan: _WmmaMoePlan,
    a_smem: Value,
    operands: List[_MoeOperand],
    acc_groups: List[List[Value]],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    sched: object = None,
    a_par_off: Value = None,
    b_par_off: Value = None,
) -> List[List[Value]]:
    """One K-tile of WMMAs fed into every accumulator group (dual-B gate+up).

    WMMA analog of ``moe_gemm_fused._emit_moe_mfma_phase``. The A fragment is
    loaded once per ``(mi, kk)`` and shared across the gate / up operand groups;
    each group's WMMA is emitted in operand order, matching the gate-then-up
    interleave. Operand fragment coordinates come straight from the op's A / B
    layout maps (``row = lane%16`` for A, ``col = lane%16`` for B; both K-split
    across the two lane-halves), and the matmul is the target-neutral
    ``IRBuilder.mma`` (backend selects the WMMA intrinsic).
    """
    b = plan.b
    t = plan.t
    op = plan.op
    a_map = op.a_layout()  # (row, k)
    b_map = op.b_layout()  # (k, col)
    a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
    b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0)
    warp_m_off = b.mul(warp_m_idx, b.const_i32(plan.mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(plan.mfmas_n * t.warp_tile_n))

    new_groups = [list(g) for g in acc_groups]
    for kk in range(plan.k_atoms):
        k_tile_base = b.const_i32(kk * t.warp_tile_k)
        a_rows = []
        for mi in range(plan.mfmas_m):
            atom_row = b.add(warp_m_off, b.const_i32(mi * t.warp_tile_m))
            if a_par_off is not None:
                atom_row = b.add(atom_row, a_par_off)
            a_rows.append(
                _emit_frag_smem_load(
                    b,
                    a_smem,
                    a_row_in_atom,
                    a_k_in_atom,
                    atom_row,
                    k_tile_base,
                    plan.a_per_lane,
                    plan.storage_dtype,
                )
            )
        b_cols_per_op = []
        for opd in operands:
            cols = []
            for ni in range(plan.mfmas_n):
                atom_row = b.add(warp_n_off, b.const_i32(ni * t.warp_tile_n))
                if b_par_off is not None:
                    atom_row = b.add(atom_row, b_par_off)
                cols.append(
                    _emit_frag_smem_load(
                        b,
                        opd.smem,
                        b_col_in_atom,
                        b_k_in_atom,
                        atom_row,
                        k_tile_base,
                        plan.b_per_lane,
                        plan.storage_dtype,
                    )
                )
            b_cols_per_op.append(cols)
        flat = 0
        for mi in range(plan.mfmas_m):
            for ni in range(plan.mfmas_n):
                for gi in range(len(operands)):
                    new_groups[gi][flat] = b.mma(
                        op, a_rows[mi], b_cols_per_op[gi][ni], new_groups[gi][flat]
                    )
                flat += 1
    if sched is not None:
        sched()
    return new_groups


def _emit_wmma_moe_prefetch_kloop(
    plan: _WmmaMoePlan,
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
    sched: object = None,
) -> List[List[Value]]:
    """Software-prefetched WMMA k-loop; return final accumulator groups.

    1:1 mirror of ``moe_gemm_fused._emit_moe_prefetch_kloop`` (register-resident
    next-tile prefetch over a single LDS buffer) with the MFMA phase swapped for
    :func:`_emit_wmma_moe_phase`. The coalesced global->register load and the
    register->LDS store are the shared arch-neutral helpers.
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
        new_groups = _emit_wmma_moe_phase(
            plan, a_smem, operands, cur_groups, warp_m_idx, warp_n_idx, lane, sched
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


def _db_load_store(
    plan: _WmmaMoePlan,
    a_view: object,
    a_lds_view: object,
    a_mn_origin: Tuple[Value, Value],
    operands: List[_MoeOperand],
    b_mn_origin: Tuple[Value, Value],
    k_off: Value,
    parity: Value,
) -> None:
    """Global->register->LDS copy of one K-tile into the ``parity`` LDS half.

    The A LDS tile is ``[2*block_m, block_k]`` and each B LDS tile is
    ``[2*block_n, block_k]``; this stores into the half at row ``parity*block_m``
    (A) / ``parity*block_n`` (B). Used by the ping-pong double-buffer loop so the
    next K-tile is written to the half the WMMAs are not reading.
    """
    b = plan.b
    c0 = b.const_i32(0)
    a_regs, b_reg_groups = _emit_moe_global_load(
        plan, a_view, a_mn_origin, operands, b_mn_origin, k_off
    )
    a_row0 = b.mul(parity, b.const_i32(plan.block_m))
    a_lds = make_tile_window(
        a_lds_view, lengths=(plan.block_m, plan.block_k), origin=(a_row0, c0)
    )
    for e in range(plan.a_vecs_per_thread):
        row, col = plan._rowcol(e)
        if plan.load_vec == 1:
            a_lds.store_scalar(b, row, col, value=a_regs[e])
        else:
            a_lds.store_vec(b, row, col, value=a_regs[e], n=plan.load_vec)
    b_row0 = b.mul(parity, b.const_i32(plan.block_n))
    for op, regs in zip(operands, b_reg_groups):
        b_lds = make_tile_window(
            op.lds_view, lengths=(plan.block_n, plan.block_k), origin=(b_row0, c0)
        )
        for e in range(plan.b_vecs_per_thread):
            row, col = plan._rowcol(e)
            if plan.load_vec == 1 and op.store_scalar_ok:
                b_lds.store_scalar(b, row, col, value=regs[e])
            else:
                b_lds.store_vec(b, row, col, value=regs[e], n=plan.load_vec)


def _emit_wmma_moe_db_kloop(
    plan: _WmmaMoePlan,
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
    sched: object = None,
) -> List[List[Value]]:
    """LDS ping-pong double-buffered WMMA gate+up k-loop (one barrier/K-tile).

    Mirrors ``gemm_universal._emit_kloop_db`` for the dual-B MoE: the A / Bg / Bu
    LDS tiles are doubled (``[2*block_m|n, block_k]``); the prologue loads K-tile
    0 into half 0, then each iteration issues the NEXT tile's global->LDS copy
    into the opposite half (overlapping the current tile's WMMAs) behind a SINGLE
    start-of-iter barrier that serves both the RAW (current half written last
    iter) and WAR (other half's reads finished last iter) hazards. Halves the
    barrier count vs the single-buffer prefetch loop -> better latency hiding in
    the occupancy-starved decode regime.
    """
    b = plan.b
    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_block_k = b.const_i32(plan.block_k)
    c_block_m = b.const_i32(plan.block_m)
    c_block_n = b.const_i32(plan.block_n)
    K_minus = b.sub(K, c_block_k)

    # Prologue: tile 0 -> half 0.
    _db_load_store(plan, a_view, a_lds_view, a_mn_origin, operands, b_mn_origin, c0, c0)

    group_sizes = [len(g) for g in acc_groups]
    carried: List[Tuple[str, Value]] = [("gupar", c0)]
    for g in acc_groups:
        carried += list(g)

    def _phase(cur_groups, parity):
        a_par = b.mul(parity, c_block_m)
        b_par = b.mul(parity, c_block_n)
        return _emit_wmma_moe_phase(
            plan,
            a_smem,
            operands,
            cur_groups,
            warp_m_idx,
            warp_n_idx,
            lane,
            sched,
            a_par,
            b_par,
        )

    for_op = b.scf_for_iter(c0, K_minus, c_block_k, carried, iv_name="k0")
    with for_op as (k0, iter_vars):
        parity = iter_vars[0]
        off = 1
        cur_groups: List[List[Value]] = []
        for sz in group_sizes:
            cur_groups.append(list(iter_vars[off : off + sz]))
            off += sz
        nxt = b.sub(c1, parity)
        b.sync()
        k_next = b.add(k0, c_block_k)
        _db_load_store(
            plan, a_view, a_lds_view, a_mn_origin, operands, b_mn_origin, k_next, nxt
        )
        new_groups = _phase(cur_groups, parity)
        yielded: List[Value] = [nxt]
        for g in new_groups:
            yielded += g
        b.scf_yield(*yielded)

    results = list(for_op.results)
    final_parity = results[0]
    off = 1
    final_groups: List[List[Value]] = []
    for sz in group_sizes:
        final_groups.append(list(results[off : off + sz]))
        off += sz
    # Epilogue: drain the last in-loop load, then WMMA the final half.
    b.sync()
    return _phase(final_groups, final_parity)


# ---------------------------------------------------------------------------
# SiLU(gate)*up -> PERSISTENT LDS Hidden (cshuffle stage, WMMA layout)
# ---------------------------------------------------------------------------


def _emit_wmma_silu_to_hidden(
    plan: _WmmaMoePlan,
    Hidden_smem: Value,
    gate_res: List[Value],
    up_res: List[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    one_f32: Value,
    c_neg_log2e: Value,
) -> None:
    """Stage ``silu(gate)*up`` into the persistent LDS Hidden buffer.

    WMMA analog of ``moe_gemm_fused._emit_cshuffle_stage`` -- but the WMMA
    accumulator is column-distributed, so we scatter each per-lane slot through
    ``op.c_layout()`` (one ``(row, col)`` per slot) instead of the MFMA C-warp
    distribution. ``Hidden_smem`` is written in logical ``(row=m, col=inter)``
    packed layout; the down GEMM reads A from the SAME logical element, so the
    pyisa ``G_reshape`` is implicit (no explicit LDS transpose). Both A and B
    fragment slots are K-contiguous, so the write address matches the down
    A-read address for the 16x16x32 atom.
    """
    b = plan.b
    t = plan.t
    op = plan.op
    sd = plan.storage_dtype
    c_map = op.c_layout()
    warp_m_off = b.mul(warp_m_idx, b.const_i32(plan.mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(plan.mfmas_n * t.warp_tile_n))

    hidden_view = TensorView(
        base=Hidden_smem,
        desc=TensorDescriptor.packed((t.tile_m, t.tile_n), sd),
        addr_space="lds",
    )
    z = (b.const_i32(0), b.const_i32(0))
    hidden_window = make_tile_window(
        hidden_view, lengths=(t.tile_m, t.tile_n), origin=z
    )

    for mi in range(plan.mfmas_m):
        atom_m = b.add(warp_m_off, b.const_i32(mi * t.warp_tile_m))
        for ni in range(plan.mfmas_n):
            flat = mi * plan.mfmas_n + ni
            atom_n = b.add(warp_n_off, b.const_i32(ni * t.warp_tile_n))
            for i in range(plan.c_per_lane):
                row_in_atom, col_in_atom = c_map.coord(b, lane, i)
                c_m = b.add(atom_m, row_in_atom)
                c_n = b.add(atom_n, col_in_atom)
                g = b.vec_extract(gate_res[flat], i)
                u = b.vec_extract(up_res[flat], i)
                h = b.cast_f32_to(
                    _silu_mul_f32(b, g, u, one_f32=one_f32, c_neg_log2e=c_neg_log2e),
                    sd,
                )
                hidden_window.store_scalar(b, c_m, c_n, value=h)


# ---------------------------------------------------------------------------
# DOWN GEMM: k-loop with LDS-resident A (Hidden) + WMMA atomic reduce
# ---------------------------------------------------------------------------


def _emit_wmma_down_phase_lds_a(
    plan: _WmmaMoePlan,
    a_smem: Value,
    a_col_base: Value,
    operand: _MoeOperand,
    accs: List[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    sched: object = None,
    b_par_off: Value = None,
) -> List[Value]:
    """One K-tile of down-GEMM WMMAs with A read from a persistent LDS buffer.

    WMMA analog of ``moe_fused_mega._emit_moe_down_mfma_phase_lds_a``. The A
    fragments are read from ``a_smem`` (the persistent ``Hidden_smem``, full
    width ``tile_n_inter``) at K-column ``a_col_base + kk*warp_tile_k`` (the
    running inter-contraction origin ``k0``); the single-buffered B (W_down)
    fragment stays in ``[0, block_k)``.
    """
    b = plan.b
    t = plan.t
    op = plan.op
    a_map = op.a_layout()
    b_map = op.b_layout()
    a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
    b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0)
    warp_m_off = b.mul(warp_m_idx, b.const_i32(plan.mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(plan.mfmas_n * t.warp_tile_n))

    new_accs = list(accs)
    for kk in range(plan.k_atoms):
        # A (persistent Hidden) column base: add the K-tile origin k0.
        a_k_tile_base = b.add(a_col_base, b.const_i32(kk * t.warp_tile_k))
        # B (single-buffered LDS) column base: per-tile-local.
        b_k_tile_base = b.const_i32(kk * t.warp_tile_k)
        a_rows = []
        for mi in range(plan.mfmas_m):
            atom_row = b.add(warp_m_off, b.const_i32(mi * t.warp_tile_m))
            a_rows.append(
                _emit_frag_smem_load(
                    b,
                    a_smem,
                    a_row_in_atom,
                    a_k_in_atom,
                    atom_row,
                    a_k_tile_base,
                    plan.a_per_lane,
                    plan.storage_dtype,
                )
            )
        b_cols = []
        for ni in range(plan.mfmas_n):
            atom_row = b.add(warp_n_off, b.const_i32(ni * t.warp_tile_n))
            if b_par_off is not None:
                atom_row = b.add(atom_row, b_par_off)
            b_cols.append(
                _emit_frag_smem_load(
                    b,
                    operand.smem,
                    b_col_in_atom,
                    b_k_in_atom,
                    atom_row,
                    b_k_tile_base,
                    plan.b_per_lane,
                    plan.storage_dtype,
                )
            )
        flat = 0
        for mi in range(plan.mfmas_m):
            for ni in range(plan.mfmas_n):
                new_accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], new_accs[flat])
                flat += 1
    if sched is not None:
        sched()
    return new_accs


def _emit_wmma_down_kloop_lds_a(
    plan: _WmmaMoePlan,
    a_smem_persistent: Value,
    operand: _MoeOperand,
    accs: List[Tuple[str, Value]],
    b_mn_origin: Tuple[Value, Value],
    b_k_base: Value,
    K: Value,
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    sched: object = None,
) -> List[Value]:
    """Software-prefetched down-GEMM k-loop reading A from persistent LDS.

    WMMA analog of ``moe_fused_mega._emit_moe_down_kloop_lds_a``: the A operand
    (``Hidden_smem``) is already LDS-resident at full contraction width, so only
    the B operand (W_down) is software-prefetched (global -> register -> single
    LDS buffer). The WMMA reads A from ``a_smem_persistent`` at the running
    K-tile origin ``k0``.
    """
    b = plan.b
    c0 = b.const_i32(0)
    c_block_k = b.const_i32(plan.block_k)

    def _load_b_tile(k_off: Value) -> List[Value]:
        b_origin = (b_mn_origin[0], b_mn_origin[1], b.add(b_k_base, k_off))
        b_global = make_tile_window(
            operand.global_view,
            lengths=(1, plan.block_n, plan.block_k),
            origin=b_origin,
        )
        regs = []
        for e in range(plan.b_vecs_per_thread):
            row, col = plan._rowcol(e)
            if plan.load_vec == 1:
                regs.append(b_global.load_scalar(b, b.const_i32(0), row, col))
            else:
                regs.append(
                    b_global.load_vec(b, b.const_i32(0), row, col, n=plan.load_vec)
                )
        return regs

    def _store_b_tile(regs: List[Value]) -> None:
        z = (b.const_i32(0), b.const_i32(0))
        b_lds = make_tile_window(
            operand.lds_view, lengths=(plan.block_n, plan.block_k), origin=z
        )
        for e in range(plan.b_vecs_per_thread):
            row, col = plan._rowcol(e)
            if plan.load_vec == 1 and operand.store_scalar_ok:
                b_lds.store_scalar(b, row, col, value=regs[e])
            else:
                b_lds.store_vec(b, row, col, value=regs[e], n=plan.load_vec)

    b_pre0 = _load_b_tile(c0)
    n_acc = len(accs)
    n_b = len(b_pre0)

    carried: List[Tuple[str, Value]] = []
    carried += [(name, v) for name, v in accs]
    carried += [(f"bd_pre{i}", v) for i, v in enumerate(b_pre0)]

    for_op = b.scf_for_iter(c0, K, c_block_k, carried, iv_name="dk0")
    with for_op as (k0, iter_vars):
        cur_accs = list(iter_vars[0:n_acc])
        b_regs = list(iter_vars[n_acc : n_acc + n_b])

        _store_b_tile(b_regs)
        b.sync()
        k_next = b.add(k0, c_block_k)
        k_clamped = b.select(b.cmp_lt(k_next, K), k_next, k0)
        b_next = _load_b_tile(k_clamped)
        new_accs = _emit_wmma_down_phase_lds_a(
            plan,
            a_smem_persistent,
            k0,
            operand,
            cur_accs,
            warp_m_idx,
            warp_n_idx,
            lane,
            sched,
        )
        b.sync()
        yielded = list(new_accs) + list(b_next)
        b.scf_yield(*yielded)

    results = list(for_op.results)
    return results[0:n_acc]


def _emit_wmma_down_db_kloop(
    plan: _WmmaMoePlan,
    a_smem_persistent: Value,
    operand: _MoeOperand,
    accs: List[Tuple[str, Value]],
    b_mn_origin: Tuple[Value, Value],
    b_k_base: Value,
    K: Value,
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    sched: object = None,
) -> List[Value]:
    """LDS ping-pong double-buffered down-GEMM k-loop (one barrier/K-tile).

    The A operand (``Hidden_smem``) is already LDS-resident, so only the B
    operand (W_down) is double-buffered: its LDS tile is ``[2*block_n, block_k]``
    and the next K-tile's global->LDS copy is issued into the opposite half
    behind a single start-of-iter barrier, overlapping the current tile's WMMAs.
    A is read from the persistent buffer at the running inter origin ``k0``.
    """
    b = plan.b
    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_block_k = b.const_i32(plan.block_k)
    c_block_n = b.const_i32(plan.block_n)
    K_minus = b.sub(K, c_block_k)

    def _load_store_b(k_off: Value, parity: Value) -> None:
        b_origin = (b_mn_origin[0], b_mn_origin[1], b.add(b_k_base, k_off))
        b_global = make_tile_window(
            operand.global_view,
            lengths=(1, plan.block_n, plan.block_k),
            origin=b_origin,
        )
        row0 = b.mul(parity, c_block_n)
        b_lds = make_tile_window(
            operand.lds_view, lengths=(plan.block_n, plan.block_k), origin=(row0, c0)
        )
        for e in range(plan.b_vecs_per_thread):
            row, col = plan._rowcol(e)
            if plan.load_vec == 1:
                val = b_global.load_scalar(b, c0, row, col)
            else:
                val = b_global.load_vec(b, c0, row, col, n=plan.load_vec)
            if plan.load_vec == 1 and operand.store_scalar_ok:
                b_lds.store_scalar(b, row, col, value=val)
            else:
                b_lds.store_vec(b, row, col, value=val, n=plan.load_vec)

    # Prologue: tile 0 -> half 0.
    _load_store_b(c0, c0)
    n_acc = len(accs)
    carried: List[Tuple[str, Value]] = [("dnpar", c0)] + [(nm, v) for nm, v in accs]

    for_op = b.scf_for_iter(c0, K_minus, c_block_k, carried, iv_name="dk0")
    with for_op as (k0, iter_vars):
        parity = iter_vars[0]
        cur_accs = list(iter_vars[1 : 1 + n_acc])
        nxt = b.sub(c1, parity)
        b.sync()
        k_next = b.add(k0, c_block_k)
        _load_store_b(k_next, nxt)
        b_par = b.mul(parity, c_block_n)
        new_accs = _emit_wmma_down_phase_lds_a(
            plan,
            a_smem_persistent,
            k0,
            operand,
            cur_accs,
            warp_m_idx,
            warp_n_idx,
            lane,
            sched,
            b_par,
        )
        b.scf_yield(nxt, *new_accs)

    results = list(for_op.results)
    final_parity = results[0]
    final_accs = list(results[1 : 1 + n_acc])
    b.sync()
    # Epilogue: WMMA the final half. The persistent A is read at the final k0.
    k_final = b.mul(b.sub(b.div(K, c_block_k), c1), c_block_k)
    b_par = b.mul(final_parity, c_block_n)
    return _emit_wmma_down_phase_lds_a(
        plan,
        a_smem_persistent,
        k_final,
        operand,
        final_accs,
        warp_m_idx,
        warp_n_idx,
        lane,
        sched,
        b_par,
    )


def _emit_wmma_down_reduce_atomic(
    b: IRBuilder,
    op,
    t: TileSpec,
    accs: Sequence[Value],
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
    pad_m: bool,
    pad_n: bool,
) -> None:
    """Atomic epilogue ``Y[token, n] += weight * down_acc`` (WMMA layout).

    WMMA analog of ``moe_gemm_fused._emit_down_reduce_epilogue_atomic``. The
    column-distributed WMMA accumulator pins ``col`` (= ``lane % 16``)
    independent of the slot ``i`` while the row varies with ``i``, so we hoist
    the per-ni output column once (slot 0) and vary the row / token / weight per
    slot. Token validity (``0 <= token < tokens``) masks the atomic add.
    """
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
    c_map = op.c_layout()

    # col is i-independent for the column-distributed WMMA accumulator.
    _, col0 = c_map.coord(b, lane, 0)
    c_ns = [
        b.add(
            block_n_off,
            b.add(b.add(warp_n_off, b.const_i32(ni * t.warp_tile_n)), col0),
        )
        for ni in range(mfmas_n)
    ]

    def _atomic_add_for_ni(c_n: Value, acc: Value, i: int, token: Value, w: Value):
        v = b.vec_extract(acc, i)
        contrib = b.fmul(w, v)
        y_off = b.add(b.mul(token, N), c_n)
        if pad_n:
            with b.scf_if(b.cmp_lt(c_n, N)):
                b.global_atomic_add(Y, y_off, contrib)
        else:
            b.global_atomic_add(Y, y_off, contrib)

    for mi in range(mfmas_m):
        atom_m = b.add(warp_m_off, b.const_i32(mi * t.warp_tile_m))
        for i in range(c_per_lane):
            row_in_atom, _ = c_map.coord(b, lane, i)
            c_m = b.add(block_m_off, b.add(atom_m, row_in_atom))

            def inner(_c_m=c_m, _i=i, _mi=mi):
                bucket = b.add(batch_bucket_off, _c_m)
                token = b.global_load_i32(SortedTokenIds, bucket)
                valid = b.land(b.cmp_ge(token, b.const_i32(0)), b.cmp_lt(token, tokens))
                with b.scf_if(valid):
                    w = b.global_load_f32(SortedWeights, bucket)
                    for ni in range(mfmas_n):
                        acc = accs[_mi * mfmas_n + ni]
                        _atomic_add_for_ni(c_ns[ni], acc, _i, token, w)

            if pad_m:
                with b.scf_if(b.cmp_lt(c_m, M)):
                    inner()
            else:
                inner()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_moe_fused_mega_wmma(
    spec: FusedMegaWmmaSpec, arch: str = "gfx1250"
) -> KernelDef:
    """Build the single-launch fused-MoE WMMA mega-kernel (gfx1250, f16/bf16).

    Implements the full fused path in one kernel: gate+up GEMM (dual-B, shared
    LDS A, software-prefetched register-resident k-loop) -> SiLU(gate)*up ->
    persistent LDS ``Hidden_smem`` -> down GEMM (LDS-resident A) -> weighted
    atomic reduce into ``Y``. No intermediate is written to HBM (the signature
    exposes only inputs + ``Y``).

    Requires a WMMA (wave32) target with the 16x16x32 atom (gfx1250); an MFMA /
    wave64 spec is rejected by the GEMM-spec validator.
    """
    u_gu = spec.gate_up_universal_spec()
    ok, why = is_valid_gemm_spec(u_gu, arch=arch)
    if not ok:
        raise ValueError(f"invalid fused-mega-wmma gate+up GEMM spec: {why}")
    u_down = spec.down_universal_spec()
    ok, why = is_valid_gemm_spec(u_down, arch=arch)
    if not ok:
        raise ValueError(f"invalid fused-mega-wmma down GEMM spec: {why}")

    op_gu = _resolve_mma_op(u_gu, arch)
    if op_gu is None:
        raise ValueError(f"no WMMA op for gate+up spec on {arch}")
    op_down = _resolve_mma_op(u_down, arch)
    if op_down is None:
        raise ValueError(f"no WMMA op for down spec on {arch}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.trait.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu

    storage_dtype = _storage_dtype(u_gu)

    # ---- params (mirror moe_fused_mega) --------------------------------
    A = b.param(
        "A", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    WGate = b.param(
        "WGate", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    WUp = b.param(
        "WUp", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
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
    BlockExpertIds = b.param(
        "BlockExpertIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    Y = b.param("Y", PtrType(F32, "global"), align=16)
    M = b.param("M", I32)
    N = b.param("N", I32)  # = I (inter dim)
    K = b.param("K", I32)  # = H (hidden contraction)
    H_out = b.param("H_out", I32)  # = H (down output)
    _stride_a = b.param("stride_a", I32)
    stride_b_gate = b.param("stride_b_gate", I32)
    stride_b_up = b.param("stride_b_up", I32)
    stride_b_down = b.param("stride_b_down", I32)
    _slot_size = b.param("slot_size", I32)
    tokens = b.param("tokens", I32)

    t = spec.gate_up_tile()
    c_per_lane = op_gu.c_frag_len  # noqa: F841

    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k

    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(t.warp_n)
    c_block_m = b.const_i32(block_m)
    c_block_n = b.const_i32(block_n)
    c0 = b.const_i32(0)

    # ---- block/thread prelude ------------------------------------------
    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    m_block_idx = b.block_id_y()
    expert_idx = b.global_load_i32(BlockExpertIds, m_block_idx)

    # Per-expert B byte base (i64), x2 for f16/bf16 storage.
    elem_bytes_b = b.const_i64(2)

    def _b_base(ptr: Value, stride_b: Value) -> Value:
        bytes_off = b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_b, I64)), elem_bytes_b
        )
        return b.global_ptr_add(ptr, bytes_off)

    WGate = _b_base(WGate, stride_b_gate)
    WUp = _b_base(WUp, stride_b_up)
    WDown = _b_base(WDown, stride_b_down)

    batch_off_a = c0
    batch_off_b = c0
    block_m_off = b.mul(m_block_idx, c_block_m)
    # grid.x selects the inter (gate/up N == down K) slice this TG owns.
    gu_n_off = b.mul(b.block_id_x(), c_block_n)

    # ---- LDS allocations -----------------------------------------------
    # Double-buffer doubles the A/B operand LDS so the next K-tile can be
    # written into the opposite half while the current half is being read.
    db = bool(spec.double_buffer)
    nbuf = 2 if db else 1
    a_rows = nbuf * block_m
    bn_rows = nbuf * block_n
    A_smem = b.smem_alloc(storage_dtype, [a_rows, block_k], name_hint="A_smem")
    Bg_smem = b.smem_alloc(storage_dtype, [bn_rows, block_k], name_hint="Bg_smem")
    Bu_smem = b.smem_alloc(storage_dtype, [bn_rows, block_k], name_hint="Bu_smem")
    # PERSISTENT Hidden buffer: silu(gate)*up staged here, reused as the down
    # GEMM's LDS-resident A operand. Never written to HBM.
    Hidden_smem = b.smem_alloc(
        storage_dtype, [block_m, block_n], name_hint="Hidden_smem"
    )

    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n

    acc_init = _emit_zero_acc_op(b, op_gu)
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
        desc=TensorDescriptor.packed((a_rows, block_k), storage_dtype),
        addr_space="lds",
    )
    bg_lds_view = TensorView(
        base=Bg_smem,
        desc=TensorDescriptor.packed((bn_rows, block_k), storage_dtype),
        addr_space="lds",
    )
    bu_lds_view = TensorView(
        base=Bu_smem,
        desc=TensorDescriptor.packed((bn_rows, block_k), storage_dtype),
        addr_space="lds",
    )

    plan = _WmmaMoePlan(b, u_gu, op_gu, tid)
    operands = [
        _MoeOperand(global_view=wg_view, lds_view=bg_lds_view, smem=Bg_smem),
        _MoeOperand(global_view=wu_view, lds_view=bu_lds_view, smem=Bu_smem),
    ]
    a_mn_origin = (batch_off_a, block_m_off)
    b_mn_origin = (batch_off_b, gu_n_off)

    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)

    # ---- DOWN-GEMM setup -----------------------------------------------
    td = spec.down_tile()
    block_n_down = td.tile_n
    block_k_down = td.tile_k
    down_mfmas_m = td.mfmas_per_warp_m
    down_mfmas_n = td.mfmas_per_warp_n
    c_block_n_down = b.const_i32(block_n_down)

    bd_rows = nbuf * block_n_down
    Bd_smem = b.smem_alloc(storage_dtype, [bd_rows, block_k_down], name_hint="Bd_smem")
    bd_lds_view = TensorView(
        base=Bd_smem,
        desc=TensorDescriptor.packed((bd_rows, block_k_down), storage_dtype),
        addr_space="lds",
    )
    # W_down rows index the output (H_out), contraction = inter dim N (=I).
    wd_view = make_global_view(
        WDown, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, N, 1)
    )
    plan_down = _WmmaMoePlan(b, u_down, op_down, tid)
    down_operand = _MoeOperand(global_view=wd_view, lds_view=bd_lds_view, smem=Bd_smem)
    c_down_k = b.const_i32(spec.tile_n_inter)
    down_acc_init = _emit_zero_acc_op(b, op_down)
    down_pad_m = bool(u_down.trait.pad_m)
    down_pad_n = bool(u_down.trait.pad_n)

    # ---- WMMA intrawave schedule (opt-in via trait.pipeline='wmma_v1') -----
    # Reusable WMMA-class scheduler from helpers/schedule. The gate+up k-loop
    # feeds two B operands (W_gate/W_up) from one shared A read; the down k-loop
    # one. Both compute regions hold only operand ds_reads + WMMAs (the LDS
    # stores + global loads sit in the prior barrier region), so we emit the
    # compute-region intrawave hint. 'mem' (default) emits no hints.
    gu_sched = None
    down_sched = None
    if spec.trait.pipeline == "wmma_v1":
        policy = SchedulePolicy.for_pipeline("wmma_v1")
        gu_il = WmmaHotLoopInstList.from_geometry(
            block_size=spec.block_size,
            m_per_block=t.tile_m,
            n_per_block=t.tile_n,
            k_per_block=t.tile_k,
            m_repeat=mfmas_m,
            n_repeat=mfmas_n,
            m_per_wmma=t.warp_tile_m,
            n_per_wmma=t.warp_tile_n,
            k_per_wmma=t.warp_tile_k,
            a_frag_len=op_gu.a_frag_len,
            b_frag_len=op_gu.b_frag_len,
            num_b_operands=2,
        )
        down_il = WmmaHotLoopInstList.from_geometry(
            block_size=spec.block_size,
            m_per_block=td.tile_m,
            n_per_block=td.tile_n,
            k_per_block=td.tile_k,
            m_repeat=down_mfmas_m,
            n_repeat=down_mfmas_n,
            m_per_wmma=td.warp_tile_m,
            n_per_wmma=td.warp_tile_n,
            k_per_wmma=td.warp_tile_k,
            a_frag_len=op_down.a_frag_len,
            b_frag_len=op_down.b_frag_len,
            num_b_operands=1,
        )

        def gu_sched() -> None:
            policy.emit_wmma_compute_schedule(b, gu_il)

        def down_sched() -> None:
            policy.emit_wmma_compute_schedule(b, down_il)

    gu_kloop = _emit_wmma_moe_db_kloop if db else _emit_wmma_moe_prefetch_kloop
    down_kloop = _emit_wmma_down_db_kloop if db else _emit_wmma_down_kloop_lds_a

    def _emit_body() -> None:
        # ---- STAGE 1: gate + up WMMA GEMM -> f32 registers -------------
        gate_res, up_res = gu_kloop(
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
            gu_sched,
        )

        # ---- STAGE 2: SiLU(gate)*up -> PERSISTENT LDS Hidden_smem ------
        _emit_wmma_silu_to_hidden(
            plan,
            Hidden_smem,
            gate_res,
            up_res,
            warp_m_idx,
            warp_n_idx,
            lane,
            one_f32,
            c_neg_log2e,
        )
        b.sync()

        # ---- STAGE 3 (implicit reshape) + STAGE 4/5: DOWN GEMM ---------
        # The cshuffle wrote Hidden_smem in logical (row=m, col=inter); the
        # down k-loop reads A via the same logical (m, inter) index, so the
        # G_reshape is implicit (no explicit LDS transpose). grid.x split the
        # inter contraction; this TG owns the inter slice at gu_n_off and
        # produces a PARTIAL Y over the WHOLE H_out, tiled in tile_n_down chunks.
        down_for = b.scf_for_iter(c0, H_out, c_block_n_down, [], iv_name="ho")
        with down_for as ho:
            down_accs = [
                (f"down_acc_m{mi}_n{ni}", down_acc_init)
                for mi in range(down_mfmas_m)
                for ni in range(down_mfmas_n)
            ]
            down_res = down_kloop(
                plan_down,
                Hidden_smem,
                down_operand,
                down_accs,
                (batch_off_b, ho),
                gu_n_off,
                c_down_k,
                warp_m_idx,
                warp_n_idx,
                lane,
                down_sched,
            )
            # Barrier before reusing Bd_smem in the next output tile.
            b.sync()
            _emit_wmma_down_reduce_atomic(
                b,
                op_down,
                td,
                tuple(down_res),
                warp_m_idx,
                warp_n_idx,
                lane,
                block_m_off,
                ho,  # block_n_off = down output column base
                M,
                H_out,  # N for the down output (the hidden_out dim)
                SortedTokenIds,
                SortedWeights,
                Y,
                op_down.c_frag_len,
                batch_bucket_off=c0,
                tokens=tokens,
                pad_m=down_pad_m,
                pad_n=down_pad_n,
            )
            b.scf_yield()

    # Empty tail block (BlockExpertIds == -1) skips all work.
    with b.scf_if(b.cmp_ge(expert_idx, c0)):
        _emit_body()

    return b.kernel
