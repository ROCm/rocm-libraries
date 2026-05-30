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

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.tensor_view import (
    TensorDescriptor,
    TensorView,
    make_global_view,
    make_tile_window,
)
from .gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    _choose_load_vec,
    _emit_mfma,
    _emit_smem_load,
    _emit_zero_acc,
    _load_smem_scalar,
    _load_smem_vec,
    _mfma_atom_widths,
    _storage_dtype,
    is_valid_spec as is_valid_gemm_spec,
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


@dataclass(frozen=True)
class FusedGateUpSiluGemmSpec:
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

    def __post_init__(self) -> None:
        if self.block_size == 0:
            t = self.tile
            object.__setattr__(
                self,
                "block_size",
                t.warp_m * t.warp_n * t.warp_k * self.wave_size,
            )

    def _data_spec(self) -> DataSpec:
        dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
        return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)

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
        return self.to_universal_spec().kernel_name() + "_gate_up_silu"


def build_moe_gate_up_silu_gemm(
    spec: FusedGateUpSiluGemmSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the fused gate+up+silu MFMA kernel.

    ``arch`` selects the target GPU for MFMA-atom validation. The MoE
    GEMM tile's ``warp_tile`` atom is checked against
    :class:`ck_dsl.core.arch.ArchTarget`'s catalog via the universal
    GEMM validator, so a gfx950-only wide atom (e.g. the f16
    ``32x32x16``) requesting ``arch="gfx942"`` raises a clean
    structured error here instead of crashing comgr at lower time.
    """

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

    t = spec.tile
    a_per_lane, b_per_lane, c_per_lane = _mfma_atom_widths(u)

    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k

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
    k_atoms = t.k_atoms_per_tile_k

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

    threads = spec.block_size
    load_vec = _choose_load_vec(u)
    a_total = block_m * block_k
    b_total = block_n * block_k
    a_vec_total = a_total // load_vec
    b_vec_total = b_total // load_vec
    a_vecs_per_thread = a_vec_total // threads
    b_vecs_per_thread = b_vec_total // threads
    c_threads = b.const_i32(threads)
    c_load_vec = b.const_i32(load_vec)
    c_block_k_div_vec = b.const_i32(block_k // load_vec)

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

    # Split global load (long-latency VMEM) from the LDS store so the
    # k-loop can prefetch the *next* tile's global loads while the current
    # tile's MFMAs run (software prefetch / register double-buffer). Same
    # single-buffer LDS footprint, hides global-load latency behind MFMA.
    def emit_global_load(k_off: Value) -> Tuple[List[Value], List[Value], List[Value]]:
        a_global = make_tile_window(
            a_view,
            lengths=(1, block_m, block_k),
            origin=(batch_off_a, block_m_off, k_off),
        )
        wg_global = make_tile_window(
            wg_view,
            lengths=(1, block_n, block_k),
            origin=(batch_off_b, block_n_off, k_off),
        )
        wu_global = make_tile_window(
            wu_view,
            lengths=(1, block_n, block_k),
            origin=(batch_off_b, block_n_off, k_off),
        )
        a_regs: List[Value] = []
        g_regs: List[Value] = []
        u_regs: List[Value] = []
        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                a_regs.append(a_global.load_scalar(b, b.const_i32(0), row, col))
            else:
                a_regs.append(
                    a_global.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                )
        for e in range(b_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                g_regs.append(wg_global.load_scalar(b, b.const_i32(0), row, col))
                u_regs.append(wu_global.load_scalar(b, b.const_i32(0), row, col))
            else:
                g_regs.append(
                    wg_global.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                )
                u_regs.append(
                    wu_global.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                )
        return a_regs, g_regs, u_regs

    def emit_lds_store(
        a_regs: List[Value], g_regs: List[Value], u_regs: List[Value]
    ) -> None:
        a_lds = make_tile_window(
            a_lds_view,
            lengths=(block_m, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )
        bg_lds = make_tile_window(
            bg_lds_view,
            lengths=(block_n, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )
        bu_lds = make_tile_window(
            bu_lds_view,
            lengths=(block_n, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )
        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                a_lds.store_scalar(b, row, col, value=a_regs[e])
            else:
                a_lds.store_vec(b, row, col, value=a_regs[e], n=load_vec)
        for e in range(b_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                bg_lds.store_scalar(b, row, col, value=g_regs[e])
                bu_lds.store_scalar(b, row, col, value=u_regs[e])
            else:
                bg_lds.store_vec(b, row, col, value=g_regs[e], n=load_vec)
                bu_lds.store_vec(b, row, col, value=u_regs[e], n=load_vec)

    def emit_dual_mfma_phase(
        gate_vars: List[Value], up_vars: List[Value]
    ) -> Tuple[List[Value], List[Value]]:
        if (t.warp_tile_m, t.warp_tile_n) == (16, 16):
            m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
            k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
            n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))
        else:
            m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
            k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
            n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))

        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

        new_gate = list(gate_vars)
        new_up = list(up_vars)

        for kk in range(k_atoms):
            col_base = b.add(
                b.mul(k_blk, b.const_i32(a_per_lane)),
                b.const_i32(kk * t.warp_tile_k),
            )
            a_rows = []
            for mi in range(mfmas_m):
                a_row = b.add(
                    warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_in_atom)
                )
                a_rows.append(
                    _emit_smem_load(
                        b, A_smem, a_row, col_base, a_per_lane, storage_dtype
                    )
                )

            gate_cols = []
            up_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom)
                )
                gate_cols.append(
                    _emit_smem_load(
                        b, Bg_smem, b_row, col_base, b_per_lane, storage_dtype
                    )
                )
                up_cols.append(
                    _emit_smem_load(
                        b, Bu_smem, b_row, col_base, b_per_lane, storage_dtype
                    )
                )

            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    new_gate[flat] = _emit_mfma(
                        b, u, a_rows[mi], gate_cols[ni], new_gate[flat]
                    )
                    new_up[flat] = _emit_mfma(
                        b, u, a_rows[mi], up_cols[ni], new_up[flat]
                    )
                    flat += 1

            if spec.trait.pipeline in ("compv3", "compv4"):
                b.sched_group_barrier(0x100, 1, 0)
                # Two MFMA streams: gate + up.
                b.sched_group_barrier(0x008, 2 * mfmas_m * mfmas_n, 0)

        return new_gate, new_up

    # Software-prefetched k-loop: carry the next tile's A/Bg/Bu global-load
    # registers across iterations so VMEM latency overlaps the MFMA stream.
    a_pre0, g_pre0, u_pre0 = emit_global_load(c0)
    n_accs = len(gate_accs)
    n_a = len(a_pre0)
    n_g = len(g_pre0)
    carried0 = (
        list(gate_accs)
        + list(up_accs)
        + [(f"a_pre{i}", v) for i, v in enumerate(a_pre0)]
        + [(f"g_pre{i}", v) for i, v in enumerate(g_pre0)]
        + [(f"u_pre{i}", v) for i, v in enumerate(u_pre0)]
    )
    for_op = b.scf_for_iter(c0, K, c_block_k, carried0, iv_name="k0")
    with for_op as (k0, iter_vars):
        gate_vars = list(iter_vars[:n_accs])
        up_vars = list(iter_vars[n_accs : 2 * n_accs])
        base = 2 * n_accs
        a_regs = list(iter_vars[base : base + n_a])
        g_regs = list(iter_vars[base + n_a : base + n_a + n_g])
        u_regs = list(iter_vars[base + n_a + n_g :])
        emit_lds_store(a_regs, g_regs, u_regs)
        b.sync()
        k_next = b.add(k0, c_block_k)
        k_clamped = b.select(b.cmp_lt(k_next, K), k_next, k0)
        a_next, g_next, u_next = emit_global_load(k_clamped)
        new_gate, new_up = emit_dual_mfma_phase(gate_vars, up_vars)
        b.sync()
        b.scf_yield(*(new_gate + new_up + a_next + g_next + u_next))

    _emit_gate_up_silu_epilogue_default(
        b,
        u,
        for_op.results[:n_accs],
        for_op.results[n_accs : 2 * n_accs],
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
    is_32x32 = (t.warp_tile_m, t.warp_tile_n) == (32, 32)
    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)
    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    Cs = b.smem_alloc(storage_dtype, [t.tile_m, t.tile_n], name_hint="Hidden_smem")

    if is_32x32:
        c_atom_n = b.const_i32(t.warp_tile_n)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                gate_acc = gate_accs[flat]
                up_acc = up_accs[flat]
                flat += 1
                ld_n = b.add(
                    warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom)
                )
                for i in range(c_per_lane):
                    rb = i // 4
                    ri = i % 4
                    m_off = b.add(
                        b.add(
                            b.mul(b.const_i32(8), b.const_i32(rb)),
                            b.mul(b.const_i32(4), m_blk),
                        ),
                        b.const_i32(ri),
                    )
                    ld_m = b.add(
                        warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_off)
                    )
                    g = b.vec_extract(gate_acc, i)
                    up = b.vec_extract(up_acc, i)
                    sig = b.rcp(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
                    silu = b.fmul(g, sig)
                    h = b.cast_f32_to(b.fmul(silu, up), storage_dtype)
                    b.smem_store_vN(Cs, [ld_m, ld_n], h, n=1)
    else:
        c_atom_n = b.const_i32(t.warp_tile_n)
        c_clen = b.const_i32(c_per_lane)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        m_base = b.mul(m_blk, c_clen)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                gate_acc = gate_accs[flat]
                up_acc = up_accs[flat]
                flat += 1
                ld_n = b.add(
                    warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom)
                )
                for i in range(c_per_lane):
                    m_off = b.add(m_base, b.const_i32(i))
                    ld_m = b.add(
                        warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_off)
                    )
                    g = b.vec_extract(gate_acc, i)
                    up = b.vec_extract(up_acc, i)
                    sig = b.rcp(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
                    silu = b.fmul(g, sig)
                    h = b.cast_f32_to(b.fmul(silu, up), storage_dtype)
                    b.smem_store_vN(Cs, [ld_m, ld_n], h, n=1)

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
    c_tile_n_div_vec = b.const_i32(t.tile_n // store_vec)
    vecs_per_thread = (t.tile_m * t.tile_n // store_vec) // threads
    for e in range(vecs_per_thread):
        vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
        row = b.div(vec_idx, c_tile_n_div_vec)
        col_v = b.mod(vec_idx, c_tile_n_div_vec)
        col = b.mul(col_v, b.const_i32(store_vec)) if store_vec > 1 else col_v

        c_m = b.add(block_m_off, row)
        c_n = b.add(block_n_off, col)
        c_off = b.add(batch_off_c, b.add(b.mul(c_m, N), c_n))

        in_bounds = None
        if pad_m or pad_n:
            checks = []
            if pad_m:
                checks.append(b.cmp_lt(c_m, M))
            if pad_n:
                if store_vec == 1:
                    checks.append(b.cmp_lt(c_n, N))
                else:
                    c_n_last = b.add(c_n, b.const_i32(store_vec - 1))
                    checks.append(b.cmp_lt(c_n_last, N))
            in_bounds = checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])

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
    return (
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
        .build()
    )


def moe_gate_up_silu_gemm_grid(
    batch: int, m: int, n: int, spec: FusedGateUpSiluGemmSpec
) -> Tuple[int, int, int]:
    t = spec.tile
    return (
        (n + t.tile_n - 1) // t.tile_n,
        (m + t.tile_m - 1) // t.tile_m,
        batch,
    )


@dataclass(frozen=True)
class FusedInterleavedGateUpSiluGemmSpec:
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
    """

    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=lambda: TraitSpec(epilogue="default"))
    wave_size: int = 64
    block_size: int = 0
    dtype: str = "fp16"

    def __post_init__(self) -> None:
        if self.block_size == 0:
            t = self.tile
            object.__setattr__(
                self,
                "block_size",
                t.warp_m * t.warp_n * t.warp_k * self.wave_size,
            )

    def _data_spec(self) -> DataSpec:
        dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
        return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)

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
        return self.to_universal_spec().kernel_name() + "_interleaved_gate_up_silu"


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
    if u.trait.active_tile_skip:
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
    a_per_lane, b_per_lane, c_per_lane = _mfma_atom_widths(u)
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
    k_atoms = t.k_atoms_per_tile_k
    acc_init = _emit_zero_acc(b, u)
    accs = [
        (f"gu_acc_m{mi}_n{ni}", acc_init)
        for mi in range(mfmas_m)
        for ni in range(mfmas_n)
    ]

    threads = spec.block_size
    load_vec = _choose_load_vec(u)
    a_vec_total = (block_m * block_k) // load_vec
    b_vec_total = (block_n * block_k) // load_vec
    a_vecs_per_thread = a_vec_total // threads
    b_vecs_per_thread = b_vec_total // threads
    c_threads = b.const_i32(threads)
    c_load_vec = b.const_i32(load_vec)
    c_block_k_div_vec = b.const_i32(block_k // load_vec)

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

    # Split global load (long-latency VMEM) from the LDS store so the
    # k-loop can prefetch the *next* tile's global loads while the current
    # tile's MFMAs run (software prefetch / register double-buffer). Same
    # single-buffer LDS footprint, hides global-load latency behind MFMA.
    def emit_global_load(k_off: Value) -> Tuple[List[Value], List[Value]]:
        a_global = make_tile_window(
            a_view,
            lengths=(1, block_m, block_k),
            origin=(batch_off_a, block_m_off, k_off),
        )
        b_global = make_tile_window(
            b_view,
            lengths=(1, block_n, block_k),
            origin=(batch_off_b, block_n_off, k_off),
        )
        a_regs: List[Value] = []
        b_regs: List[Value] = []
        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            a_regs.append(a_global.load_vec(b, b.const_i32(0), row, col, n=load_vec))
        # B-load: identical preshuffle pattern to ``gemm_universal``,
        # except the GEMM N is ``2*N`` (gate+up packed along N).
        if u.trait.preshuffle_b:
            n_tile_idx = b.div(block_n_off, c_block_n)
            k_tile_idx = b.div(k_off, c_block_k)
            two_n = b.mul(N, b.const_i32(2))
            n_tile_count = b.div(two_n, c_block_n)
            tile_offset_elements = b.mul(
                b.add(b.mul(k_tile_idx, n_tile_count), n_tile_idx),
                b.const_i32(block_n * block_k),
            )
            base_off = b.add(batch_off_b, tile_offset_elements)
            for e in range(b_vecs_per_thread):
                vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
                glob_off = b.add(base_off, b.mul(vec_idx, c_load_vec))
                if load_vec == 1:
                    b_regs.append(b.global_load(WGateUp, glob_off, storage_dtype))
                else:
                    b_regs.append(
                        b.global_load_vN(WGateUp, glob_off, storage_dtype, load_vec)
                    )
        else:
            for e in range(b_vecs_per_thread):
                vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
                row = b.div(vec_idx, c_block_k_div_vec)
                col_v = b.mod(vec_idx, c_block_k_div_vec)
                col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
                b_regs.append(
                    b_global.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                )
        return a_regs, b_regs

    def emit_lds_store(a_regs: List[Value], b_regs: List[Value]) -> None:
        a_lds = make_tile_window(
            a_lds_view,
            lengths=(block_m, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )
        b_lds = make_tile_window(
            b_lds_view,
            lengths=(block_n, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )
        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            a_lds.store_vec(b, row, col, value=a_regs[e], n=load_vec)
        for e in range(b_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if u.trait.preshuffle_b and load_vec == 1:
                b_lds.store_scalar(b, row, col, value=b_regs[e])
            else:
                b_lds.store_vec(b, row, col, value=b_regs[e], n=load_vec)

    def emit_mfma_phase(iter_vars: List[Value]) -> List[Value]:
        if (t.warp_tile_m, t.warp_tile_n) == (16, 16):
            m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
            k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
            n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))
        else:
            m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
            k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
            n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))
        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
        new_accs = list(iter_vars)
        for kk in range(k_atoms):
            col_base = b.add(
                b.mul(k_blk, b.const_i32(a_per_lane)), b.const_i32(kk * t.warp_tile_k)
            )
            a_rows = []
            for mi in range(mfmas_m):
                a_row = b.add(
                    warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_in_atom)
                )
                a_rows.append(
                    _emit_smem_load(
                        b, A_smem, a_row, col_base, a_per_lane, storage_dtype
                    )
                )
            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom)
                )
                b_cols.append(
                    _emit_smem_load(
                        b, B_smem, b_row, col_base, b_per_lane, storage_dtype
                    )
                )
            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    new_accs[flat] = _emit_mfma(
                        b, u, a_rows[mi], b_cols[ni], new_accs[flat]
                    )
                    flat += 1
        return new_accs

    # ---- active-tile gate ----
    # Bucket head index = ``block_id_z * slot_size + block_m_off``;
    # the interleaved kernel does not yet support chiplet swizzle so
    # ``block_m_off == block_id_y * tile_m`` here, but the form
    # mirrors the universal kernel's gate.
    do_work_cond: Optional[Value] = None
    if u.trait.active_tile_skip:
        bucket_head = b.add(b.mul(b.block_id_z(), slot_size_p), block_m_off)
        first_token = b.global_load_i32(sorted_token_ids, bucket_head)
        do_work_cond = b.cmp_ge(first_token, c0)

    def emit_compute_and_epilogue() -> None:
        # Software-prefetched k-loop (see emit_global_load / emit_lds_store):
        # carry the next tile's global-load registers across the loop so the
        # VMEM latency overlaps the current tile's MFMA stream.
        a_pre0, b_pre0 = emit_global_load(c0)
        n_a = len(a_pre0)
        n_accs = len(accs)
        carried0 = (
            list(accs)
            + [(f"a_pre{i}", v) for i, v in enumerate(a_pre0)]
            + [(f"b_pre{i}", v) for i, v in enumerate(b_pre0)]
        )
        for_op = b.scf_for_iter(c0, K, c_block_k, carried0, iv_name="k0")
        with for_op as (k0, iter_vars):
            cur_accs = list(iter_vars[:n_accs])
            a_regs = list(iter_vars[n_accs : n_accs + n_a])
            b_regs = list(iter_vars[n_accs + n_a :])
            emit_lds_store(a_regs, b_regs)
            b.sync()
            k_next = b.add(k0, c_block_k)
            k_clamped = b.select(b.cmp_lt(k_next, K), k_next, k0)
            a_next, b_next = emit_global_load(k_clamped)
            new_accs = emit_mfma_phase(cur_accs)
            b.sync()
            b.scf_yield(*(new_accs + a_next + b_next))

        _emit_interleaved_silu_epilogue(
            b,
            u,
            for_op.results[:n_accs],
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
    is_32x32 = (t.warp_tile_m, t.warp_tile_n) == (32, 32)
    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)
    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    # 1) Accumulator -> LDS in normal output layout (M x 2I tile).
    if is_32x32:
        c_atom_n = b.const_i32(t.warp_tile_n)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[flat]
                flat += 1
                ld_n = b.add(
                    warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom)
                )
                for i in range(c_per_lane):
                    rb = i // 4
                    ri = i % 4
                    m_off = b.add(
                        b.add(
                            b.mul(b.const_i32(8), b.const_i32(rb)),
                            b.mul(b.const_i32(4), m_blk),
                        ),
                        b.const_i32(ri),
                    )
                    ld_m = b.add(
                        warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_off)
                    )
                    h = b.cast_f32_to(b.vec_extract(acc, i), storage_dtype)
                    b.smem_store_vN(C_smem, [ld_m, ld_n], h, n=1)
    else:
        c_atom_n = b.const_i32(t.warp_tile_n)
        c_clen = b.const_i32(c_per_lane)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        m_base = b.mul(m_blk, c_clen)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[flat]
                flat += 1
                ld_n = b.add(
                    warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom)
                )
                for i in range(c_per_lane):
                    ld_m = b.add(
                        warp_m_off,
                        b.add(
                            b.const_i32(mi * t.warp_tile_m),
                            b.add(m_base, b.const_i32(i)),
                        ),
                    )
                    h = b.cast_f32_to(b.vec_extract(acc, i), storage_dtype)
                    b.smem_store_vN(C_smem, [ld_m, ld_n], h, n=1)

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
    c_hidden_cols = b.const_i32(hidden_cols_per_tile)
    n_base = b.div(block_n_off, b.const_i32(2))
    for u in range(units_per_thread):
        linear_h = b.add(
            b.const_i32(u * threads * vec_h),
            b.mul(b.thread_id_x(), c_vec_h),
        )
        row = b.div(linear_h, c_hidden_cols)
        hcol_local = b.mod(linear_h, c_hidden_cols)
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
            sig = b.rcp(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
            out_v = b.cast_f32_to(b.fmul(b.fmul(g, sig), up), storage_dtype)

            in_bounds = None
            if pad_m or pad_n:
                checks = []
                if pad_m:
                    checks.append(b.cmp_lt(c_m, M))
                if pad_n:
                    checks.append(b.cmp_lt(c_n_start, N))
                in_bounds = (
                    checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])
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
                sig = b.rcp(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
                h_scalars.append(
                    b.cast_f32_to(b.fmul(b.fmul(g, sig), up), storage_dtype)
                )
            h_packed = b.vec_pack(h_scalars, storage_dtype)

            in_bounds = None
            if pad_m or pad_n:
                checks = []
                if pad_m:
                    checks.append(b.cmp_lt(c_m, M))
                if pad_n:
                    # vec_h consecutive columns; bounds-check the last
                    # one (the first is implied).
                    c_n_last = b.add(c_n_start, b.const_i32(vec_h - 1))
                    checks.append(b.cmp_lt(c_n_last, N))
                in_bounds = (
                    checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])
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
    if spec.trait.active_tile_skip:
        sig = sig.ptr("SortedTokenIds", "i32").scalar("slot_size", "i32")
    return sig.build()


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
class FusedDownReduceGemmSpec:
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

    def __post_init__(self) -> None:
        if self.block_size == 0:
            t = self.tile
            object.__setattr__(
                self,
                "block_size",
                t.warp_m * t.warp_n * t.warp_k * self.wave_size,
            )

    def _data_spec(self) -> DataSpec:
        dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
        return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)

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
        return self.to_universal_spec().kernel_name() + "_down_reduce"


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

    t = spec.tile
    a_per_lane, b_per_lane, c_per_lane = _mfma_atom_widths(u)

    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k

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
    k_atoms = t.k_atoms_per_tile_k

    acc_init = _emit_zero_acc(b, u)
    accs = [
        (f"down_acc_m{mi}_n{ni}", acc_init)
        for mi in range(mfmas_m)
        for ni in range(mfmas_n)
    ]

    threads = spec.block_size
    load_vec = _choose_load_vec(u)
    a_total = block_m * block_k
    b_total = block_n * block_k
    a_vec_total = a_total // load_vec
    b_vec_total = b_total // load_vec
    a_vecs_per_thread = a_vec_total // threads
    b_vecs_per_thread = b_vec_total // threads
    c_threads = b.const_i32(threads)
    c_load_vec = b.const_i32(load_vec)
    c_block_k_div_vec = b.const_i32(block_k // load_vec)

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

    # Split the global load (long-latency VMEM) from the LDS store so the
    # k-loop can issue the *next* tile's global loads while the *current*
    # tile's MFMAs run (software prefetch / register double-buffer). This
    # keeps the single-buffer LDS footprint (no occupancy change on the
    # LDS-limited gfx950 path) but hides global-load latency behind the
    # MFMA stream — the down-reduce GEMM is VMEM-bound (vmem_load >> mfma).
    def emit_global_load(k_off: Value) -> Tuple[List[Value], List[Value]]:
        a_global = make_tile_window(
            a_view,
            lengths=(1, block_m, block_k),
            origin=(batch_off_a, block_m_off, k_off),
        )
        b_global = make_tile_window(
            b_view,
            lengths=(1, block_n, block_k),
            origin=(batch_off_b, block_n_off, k_off),
        )
        a_regs: List[Value] = []
        b_regs: List[Value] = []
        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                a_regs.append(a_global.load_scalar(b, b.const_i32(0), row, col))
            else:
                a_regs.append(
                    a_global.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                )
        for e in range(b_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                b_regs.append(b_global.load_scalar(b, b.const_i32(0), row, col))
            else:
                b_regs.append(
                    b_global.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                )
        return a_regs, b_regs

    def emit_lds_store(a_regs: List[Value], b_regs: List[Value]) -> None:
        a_lds = make_tile_window(
            a_lds_view,
            lengths=(block_m, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )
        b_lds = make_tile_window(
            b_lds_view,
            lengths=(block_n, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )
        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                a_lds.store_scalar(b, row, col, value=a_regs[e])
            else:
                a_lds.store_vec(b, row, col, value=a_regs[e], n=load_vec)
        for e in range(b_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                b_lds.store_scalar(b, row, col, value=b_regs[e])
            else:
                b_lds.store_vec(b, row, col, value=b_regs[e], n=load_vec)

    def emit_mfma_phase(iter_vars: List[Value]) -> List[Value]:
        if (t.warp_tile_m, t.warp_tile_n) == (16, 16):
            m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
            k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
            n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))
        else:
            m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
            k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
            n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))

        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
        new_accs = list(iter_vars)

        for kk in range(k_atoms):
            col_base = b.add(
                b.mul(k_blk, b.const_i32(a_per_lane)),
                b.const_i32(kk * t.warp_tile_k),
            )
            a_rows = []
            for mi in range(mfmas_m):
                a_row = b.add(
                    warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_in_atom)
                )
                a_rows.append(
                    _emit_smem_load(
                        b, A_smem, a_row, col_base, a_per_lane, storage_dtype
                    )
                )

            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom)
                )
                b_cols.append(
                    _emit_smem_load(
                        b, B_smem, b_row, col_base, b_per_lane, storage_dtype
                    )
                )

            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    new_accs[flat] = _emit_mfma(
                        b, u, a_rows[mi], b_cols[ni], new_accs[flat]
                    )
                    flat += 1

            if spec.trait.pipeline in ("compv3", "compv4"):
                b.sched_group_barrier(0x100, 1, 0)
                b.sched_group_barrier(0x008, mfmas_m * mfmas_n, 0)
        return new_accs

    # Software-prefetched k-loop: load tile 0 to registers before the
    # loop, then each iteration stores the prefetched regs to LDS, issues
    # the *next* tile's global loads (in flight during the MFMA), runs the
    # MFMA from LDS, and yields the prefetched regs for the next trip.
    # The loop-carried register values let the next global load overlap
    # the current MFMA without a second LDS buffer.
    a_pre0, b_pre0 = emit_global_load(c0)
    n_a = len(a_pre0)
    carried0 = (
        list(accs)
        + [(f"a_pre{i}", v) for i, v in enumerate(a_pre0)]
        + [(f"b_pre{i}", v) for i, v in enumerate(b_pre0)]
    )
    n_accs = len(accs)
    for_op = b.scf_for_iter(c0, K, c_block_k, carried0, iv_name="k0")
    with for_op as (k0, iter_vars):
        cur_accs = list(iter_vars[:n_accs])
        a_regs = list(iter_vars[n_accs : n_accs + n_a])
        b_regs = list(iter_vars[n_accs + n_a :])
        emit_lds_store(a_regs, b_regs)
        b.sync()
        # Prefetch next tile (clamp k to a valid in-bounds offset so the
        # final iteration's speculative load stays addressable; the regs
        # are simply not consumed after the last trip).
        k_next = b.add(k0, c_block_k)
        k_clamped = b.select(b.cmp_lt(k_next, K), k_next, k0)
        a_next, b_next = emit_global_load(k_clamped)
        new_accs = emit_mfma_phase(cur_accs)
        b.sync()
        b.scf_yield(*(new_accs + a_next + b_next))

    _emit_down_reduce_epilogue_atomic(
        b,
        u,
        for_op.results[:n_accs],
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
    is_32x32 = (t.warp_tile_m, t.warp_tile_n) == (32, 32)
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

    if is_32x32:
        c_atom_n = b.const_i32(t.warp_tile_n)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        # Per-mi c_n list (one per ni); shared across all i in the
        # mi-row so the inner ni loop only multiplies the acc element.
        for mi in range(mfmas_m):
            c_ns = [
                b.add(
                    b.add(block_n_off, warp_n_off),
                    b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom),
                )
                for ni in range(mfmas_n)
            ]
            for i in range(c_per_lane):
                rb = i // 4
                ri = i % 4
                m_off = b.add(
                    b.add(
                        b.mul(b.const_i32(8), b.const_i32(rb)),
                        b.mul(b.const_i32(4), m_blk),
                    ),
                    b.const_i32(ri),
                )
                c_m = b.add(
                    b.add(block_m_off, warp_m_off),
                    b.add(b.const_i32(mi * t.warp_tile_m), m_off),
                )
                emit_one_row(c_m, c_ns, i, mi)
    else:
        c_atom_n = b.const_i32(t.warp_tile_n)
        c_clen = b.const_i32(c_per_lane)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        m_base = b.mul(m_blk, c_clen)
        for mi in range(mfmas_m):
            c_ns = [
                b.add(
                    b.add(block_n_off, warp_n_off),
                    b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom),
                )
                for ni in range(mfmas_n)
            ]
            for i in range(c_per_lane):
                m_off = b.add(m_base, b.const_i32(i))
                c_m = b.add(
                    b.add(block_m_off, warp_m_off),
                    b.add(b.const_i32(mi * t.warp_tile_m), m_off),
                )
                emit_one_row(c_m, c_ns, i, mi)


def moe_down_reduce_gemm_signature(spec: FusedDownReduceGemmSpec):
    from ...helpers.spec import SignatureBuilder

    dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "f16"
    return (
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
        .build()
    )


def moe_down_reduce_gemm_grid(
    batch: int, m: int, n: int, spec: FusedDownReduceGemmSpec
) -> Tuple[int, int, int]:
    t = spec.tile
    return (
        (n + t.tile_n - 1) // t.tile_n,
        (m + t.tile_m - 1) // t.tile_m,
        batch,
    )


@dataclass(frozen=True)
class FusedDownSiluReduceGemmSpec:
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
        if self.block_size == 0:
            t = self.tile
            object.__setattr__(
                self,
                "block_size",
                t.warp_m * t.warp_n * t.warp_k * self.wave_size,
            )

    def _data_spec(self) -> DataSpec:
        dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
        return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)

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
