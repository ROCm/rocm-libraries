# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-launch fused-MoE MEGA-kernel (f16/bf16).

This is the rocke analog of the pyisa ``fmoe_fp8_4wv.py`` mega-kernel, built
in f16/bf16 (no fp8 in this build). A SINGLE fused kernel computes, per
threadgroup, the full MoE per-expert path:

    GEMM0 (gate) + GEMM1 (up) sharing one LDS-resident X tile
      -> SiLU(gate) * up in registers
      -> reshape through LDS (Hidden stays in LDS, never HBM)
      -> GEMM2 (down) -> weighted atomic reduce into Y.

See ``examples/gfx950/fused_mega_moe/docs/BUILD_SPEC.md`` for the authoritative
build specification.

STATUS: COMPLETE (f16 + bf16, single launch).

The full gate+up+silu+down+reduce path is implemented in one kernel:

* Phase 0 + Phase 1: kernel signature, grid function, block/thread prelude, the
  gate+up GEMM via the reused ``_emit_moe_prefetch_kloop`` (dual-B, shared A
  read), the SiLU(gate)*up activation, and staging the f16/bf16 result into a
  PERSISTENT LDS ``Hidden_smem`` buffer via ``_emit_cshuffle_stage`` (NOT HBM).
* Phase 2: the down GEMM (``_emit_moe_down_kloop_lds_a`` reading ``Hidden_smem``
  as the LDS-resident A operand) + the weighted, token-validity-masked atomic
  reduce into ``Y`` via ``_emit_down_reduce_epilogue_atomic``.

The single-launch fusion is structural: the kernel signature exposes only the
inputs (``A``, ``WGate``, ``WUp``, ``WDown``, routing) and the output ``Y`` --
there is NO ``GateOut`` / ``UpOut`` / ``Hidden`` / ``DownOut`` HBM buffer, so
every intermediate stays in registers / LDS. Verified to lower (LLVM-direct) and
assemble (comgr -> HSACO) for gfx950 in both f16 and bf16.

All MFMA / LDS geometry and the gate+up k-loop are 100% reused from
``moe_gemm_fused.py`` (imported, never modified). This file adds the mega-kernel
builder + one trimmed LDS-A down k-loop variant.

Structure + codegen are covered by ``tests/test_moe_fused_mega.py`` (single-
launch signature, f16/bf16 LLVM lowering with MFMA + atomic reduce, wave32
rejection). On-device numeric validation must run on a gfx950 (CDNA4 / MFMA)
device -- the body uses MFMA atoms and cannot run on a wave32 / WMMA target
(gfx1250).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ...core.ir import F32, I32, I64, IRBuilder, KernelDef, PtrType, Value
from ...helpers.tensor_view import (
    TensorDescriptor,
    TensorView,
    make_global_view,
)
from .gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    _emit_zero_acc,
    _mfma_atom_widths,
    _storage_dtype,
    is_valid_spec as is_valid_gemm_spec,
    mono_data_spec,
)
from ...helpers.tensor_view import make_tile_window
from .gemm_universal import _emit_mfma, _emit_smem_load
from .moe_gemm_fused import (
    _CWarpDecode,
    _MoeKloopPlan,
    _MoeOperand,
    _emit_cshuffle_stage,
    _emit_down_reduce_epilogue_atomic,
    _emit_moe_prefetch_kloop,
    _silu_mul_f32,
)


__all__ = [
    "FusedMegaKernelSpec",
    "build_moe_fused_mega_gemm",
    "moe_fused_mega_grid",
    "moe_fused_mega_signature",
]


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedMegaKernelSpec:
    """Single-launch fused-MoE mega-kernel spec (f16/bf16).

    The mega-kernel fuses gate+up+silu+down+reduce for one sorted-token
    M-block over one ``TILE_N_INTER`` slice of the inter dimension. Tiling is
    locked by BUILD_SPEC Section 1.1:

    * ``tile_m`` = sorted tokens per m-block (pyisa ``sub_x``), default 32.
    * ``tile_n_inter`` = inter columns this TG owns (pyisa ``sub_gu``); this is
      the GEMM0/1 N extent AND the GEMM2 contraction extent, default 256.
    * ``tile_k_gu`` = K-loop tile along the hidden contraction H for gate/up.
    * MFMA atom = ``16x16x32`` (the largest-K f16 atom; bf16 analog for bf16).

    The gate+up GEMM uses warp grid ``warp_m x warp_n`` over
    ``(tile_m, tile_n_inter)``; with the 16x16x32 atom and
    ``tile_m=32, tile_n_inter=256, warp_m=1, warp_n=4`` the block has
    4 waves x 64 = 256 threads.

    ``M`` / ``N`` / ``K`` / ``H_out`` are runtime args; tile geometry is static.
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
    tile_k_down: int = 64
    trait: TraitSpec = field(default_factory=lambda: TraitSpec(epilogue="default"))
    wave_size: int = 64
    block_size: int = 0
    dtype: str = "fp16"

    def __post_init__(self) -> None:
        if self.block_size == 0:
            object.__setattr__(
                self,
                "block_size",
                self.warp_m * self.warp_n * self.wave_size,
            )

    # -- data / tile spec helpers ----------------------------------------

    def _data_spec(self) -> DataSpec:
        return mono_data_spec(self.dtype)

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
        return self.gate_up_universal_spec().kernel_name() + "_fused_mega"


# ---------------------------------------------------------------------------
# Grid + signature
# ---------------------------------------------------------------------------


def moe_fused_mega_grid(
    num_m_blocks: int, inter: int, spec: FusedMegaKernelSpec
) -> Tuple[int, int, int]:
    """Mega-kernel launch grid.

    ``grid.x`` splits the down-GEMM contraction (inter dim ``I``) across TGs ->
    each TG produces a partial down result + atomic-adds into Y.
    ``grid.y`` selects the sorted m-block (-> expert id + weight base offsets).

    ``grid = (ceil(inter / tile_n_inter), num_m_blocks, 1)``. For the canonical
    decode (I=7168, tile_n_inter=256) grid.x = 28; grid.y = total_padded/32 = 8.
    """
    sub_gu = spec.tile_n_inter
    gx = (inter + sub_gu - 1) // sub_gu
    return (gx, num_m_blocks, 1)


def moe_fused_mega_signature(spec: FusedMegaKernelSpec):
    from ...helpers.spec import SignatureBuilder

    dt = spec.dtype if spec.dtype in ("f16", "fp16", "bf16") else "f16"
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
# STAGE 4 helper: down-GEMM k-loop with LDS-resident A (Hidden_smem)
# ---------------------------------------------------------------------------


def _emit_moe_down_mfma_phase_lds_a(
    plan: _MoeKloopPlan,
    a_smem: Value,
    a_col_base: Value,
    operand: _MoeOperand,
    accs: list,
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    *,
    sched_groups: int,
) -> list:
    """One K-tile of down-GEMM MFMAs with A read from a persistent LDS buffer.

    Identical to ``_emit_moe_mfma_phase`` (single B operand) EXCEPT the A
    fragments are read from ``a_smem`` (the persistent ``Hidden_smem``, full
    width ``tile_n_inter``) at column ``a_col_base + kk*warp_tile_k`` instead of
    the per-tile-resident ``0 + kk*warp_tile_k`` offset of the prefetch loop.
    ``a_col_base`` is the current K-tile origin (``k0``) into the persistent A.

    The B (W_down) fragment read is unchanged: ``operand.smem`` is the single-
    buffered B LDS tile re-stored every K-tile, so its columns stay in
    ``[0, block_k)``.
    """
    b = plan.b
    t = plan.t
    m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
    k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
    n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))
    warp_m_off = b.mul(warp_m_idx, b.const_i32(plan.mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(plan.mfmas_n * t.warp_tile_n))

    new_accs = list(accs)
    for kk in range(plan.k_atoms):
        # B (single-buffered LDS) column base: per-tile-local.
        b_col_base = b.add(
            b.mul(k_blk, b.const_i32(plan.b_per_lane)),
            b.const_i32(kk * t.warp_tile_k),
        )
        # A (persistent Hidden_smem) column base: add the K-tile origin k0.
        a_col = b.add(
            a_col_base,
            b.add(
                b.mul(k_blk, b.const_i32(plan.a_per_lane)),
                b.const_i32(kk * t.warp_tile_k),
            ),
        )
        a_rows = []
        for mi in range(plan.mfmas_m):
            a_row = b.add(warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_in_atom))
            a_rows.append(
                _emit_smem_load(
                    b, a_smem, a_row, a_col, plan.a_per_lane, plan.storage_dtype
                )
            )
        b_cols = []
        for ni in range(plan.mfmas_n):
            b_row = b.add(warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom))
            b_cols.append(
                _emit_smem_load(
                    b,
                    operand.smem,
                    b_row,
                    b_col_base,
                    plan.b_per_lane,
                    plan.storage_dtype,
                )
            )
        flat = 0
        for mi in range(plan.mfmas_m):
            for ni in range(plan.mfmas_n):
                new_accs[flat] = _emit_mfma(
                    b, plan.u, a_rows[mi], b_cols[ni], new_accs[flat]
                )
                flat += 1
        if sched_groups and plan.u.trait.pipeline in ("compv3", "compv4"):
            b.sched_group_barrier(0x100, 1, 0)
            b.sched_group_barrier(0x008, sched_groups, 0)
    return new_accs


def _emit_moe_down_kloop_lds_a(
    plan: _MoeKloopPlan,
    a_smem_persistent: Value,
    operand: _MoeOperand,
    accs: list,
    b_mn_origin: Tuple[Value, Value],
    b_k_base: Value,
    K: Value,
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    *,
    sched_groups: int,
) -> list:
    """Software-prefetched down-GEMM k-loop reading A from persistent LDS.

    Trimmed copy of ``_emit_moe_prefetch_kloop`` with the A global-load and A
    LDS-store removed: the down-GEMM A operand (``Hidden_smem``) is ALREADY
    resident in LDS at full contraction width (``tile_n_inter``). Only the B
    operand (W_down) is software-prefetched (global-load -> single LDS buffer)
    as in the original loop. The MFMA reads A from ``a_smem_persistent`` at the
    running K-tile origin ``k0`` via :func:`_emit_moe_down_mfma_phase_lds_a`.

    Returns the final accumulator list (one f32 vec per ``(mi, ni)``).
    """
    b = plan.b
    c0 = b.const_i32(0)
    c_block_k = b.const_i32(plan.block_k)

    def _load_b_tile(k_off: Value) -> list:
        """Coalesced global -> register load of one W_down K-tile.

        ``k_off`` is the LDS-local contraction origin; the global W_down read
        offsets it by ``b_k_base`` (this TG's inter slice base).
        """
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

    def _store_b_tile(regs: list) -> None:
        """Store one prefetched W_down K-tile into the single LDS buffer."""
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

    # Prefetch B tile 0 (A is already resident -> no A prefetch).
    b_pre0 = _load_b_tile(c0)
    n_acc = len(accs)
    n_b = len(b_pre0)

    carried: list = []
    carried += [(name, v) for name, v in accs]
    carried += [(f"bd_pre{i}", v) for i, v in enumerate(b_pre0)]

    for_op = b.scf_for_iter(c0, K, c_block_k, carried, iv_name="dk0")
    with for_op as (k0, iter_vars):
        cur_accs = list(iter_vars[0:n_acc])
        b_regs = list(iter_vars[n_acc : n_acc + n_b])

        # Store the prefetched B tile into the single LDS buffer (no A store).
        _store_b_tile(b_regs)
        b.sync()
        # Issue the NEXT B tile's clamped global loads (overlap the MFMAs).
        k_next = b.add(k0, c_block_k)
        k_clamped = b.select(b.cmp_lt(k_next, K), k_next, k0)
        b_next = _load_b_tile(k_clamped)
        # MFMA: A from persistent Hidden_smem at column k0; B from LDS tile.
        new_accs = _emit_moe_down_mfma_phase_lds_a(
            plan,
            a_smem_persistent,
            k0,
            operand,
            cur_accs,
            warp_m_idx,
            warp_n_idx,
            lane,
            sched_groups=sched_groups,
        )
        b.sync()
        yielded = list(new_accs) + list(b_next)
        b.scf_yield(*yielded)

    results = list(for_op.results)
    return results[0:n_acc]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_moe_fused_mega_gemm(
    spec: FusedMegaKernelSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the single-launch fused-MoE mega-kernel (f16/bf16).

    Implements the full fused path in one kernel: gate+up GEMM (dual-B, shared
    LDS A) -> SiLU(gate)*up -> persistent LDS ``Hidden_smem`` -> down GEMM
    (LDS-resident A) -> weighted atomic reduce into ``Y``. No intermediate is
    written to HBM (the signature exposes only inputs + ``Y``).

    ``arch`` selects the target GPU for MFMA-atom validation; an atom not in the
    arch catalog (e.g. a gfx950-only wide atom requested with ``arch="gfx942"``)
    raises a structured error here instead of crashing comgr at lower time.
    Requires an MFMA (CDNA) target; wave32/WMMA targets (gfx1250) are rejected
    by the GEMM-spec validator.
    """

    u_gu = spec.gate_up_universal_spec()
    ok, why = is_valid_gemm_spec(u_gu, arch=arch)
    if not ok:
        raise ValueError(f"invalid fused-mega gate+up GEMM spec: {why}")
    u_down = spec.down_universal_spec()
    ok, why = is_valid_gemm_spec(u_down, arch=arch)
    if not ok:
        raise ValueError(f"invalid fused-mega down GEMM spec: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.trait.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu

    storage_dtype = _storage_dtype(u_gu)

    # ---- params (BUILD_SPEC Section 3.1) -------------------------------
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
    _, _, c_per_lane = _mfma_atom_widths(u_gu)

    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k

    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(t.warp_n)
    c_block_m = b.const_i32(block_m)
    c_block_n = b.const_i32(block_n)
    c0 = b.const_i32(0)

    # ---- block/thread prelude (copy of build_moe_gate_up_silu_gemm) ----
    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    m_block_idx = b.block_id_y()
    expert_idx = b.global_load_i32(BlockExpertIds, m_block_idx)

    # Per-expert B byte base (i64), x2 for f16 storage. Each weight matrix has
    # its own per-expert stride.
    elem_bytes_b = b.const_i64(2)  # f16 / bf16

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

    # ---- LDS allocations ----------------------------------------------
    # gate/up GEMM operands (single-buffered; Phase 3 adds the X double-buffer).
    A_smem = b.smem_alloc(storage_dtype, [block_m, block_k], name_hint="A_smem")
    Bg_smem = b.smem_alloc(storage_dtype, [block_n, block_k], name_hint="Bg_smem")
    Bu_smem = b.smem_alloc(storage_dtype, [block_n, block_k], name_hint="Bu_smem")
    # PERSISTENT Hidden buffer: silu(gate)*up staged here, reused as the down
    # GEMM's LDS-resident A operand. Phase 1 fills it; Phase 2 reads it. Never
    # written to HBM.
    Hidden_smem = b.smem_alloc(
        storage_dtype, [block_m, block_n], name_hint="Hidden_smem"
    )

    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n

    acc_init = _emit_zero_acc(b, u_gu)
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

    plan = _MoeKloopPlan(b, u_gu, tid)
    operands = [
        _MoeOperand(global_view=wg_view, lds_view=bg_lds_view, smem=Bg_smem),
        _MoeOperand(global_view=wu_view, lds_view=bu_lds_view, smem=Bu_smem),
    ]
    a_mn_origin = (batch_off_a, block_m_off)
    b_mn_origin = (batch_off_b, gu_n_off)

    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)

    # ---- DOWN-GEMM setup (Stage 4/5) ----------------------------------
    # A = Hidden_smem (LDS-resident, M x inter-slice); contraction = inter
    # slice (= tile_n_inter, this TG's slice). B = W_down (E, H_out, I): per
    # output tile, rows index H_out (stride = N = I), contraction (inter) has
    # stride 1. Down output H_out is tiled in tile_n_down chunks; grid.x split
    # the inter contraction so each TG atomic-adds a PARTIAL Y over all H_out.
    td = spec.down_tile()
    block_n_down = td.tile_n
    block_k_down = td.tile_k
    down_mfmas_m = td.mfmas_per_warp_m
    down_mfmas_n = td.mfmas_per_warp_n
    c_block_n_down = b.const_i32(block_n_down)

    Bd_smem = b.smem_alloc(
        storage_dtype, [block_n_down, block_k_down], name_hint="Bd_smem"
    )
    bd_lds_view = TensorView(
        base=Bd_smem,
        desc=TensorDescriptor.packed((block_n_down, block_k_down), storage_dtype),
        addr_space="lds",
    )
    # W_down rows index the output (H_out), contraction = inter dim N (=I).
    wd_view = make_global_view(
        WDown, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, N, 1)
    )
    plan_down = _MoeKloopPlan(b, u_down, tid)
    down_operand = _MoeOperand(global_view=wd_view, lds_view=bd_lds_view, smem=Bd_smem)
    # The down contraction extent is the inter slice this TG owns (tile_n_inter).
    c_down_k = b.const_i32(spec.tile_n_inter)
    down_acc_init = _emit_zero_acc(b, u_down)

    def _emit_body() -> None:
        # ---- STAGE 1: gate + up GEMM -> f32 registers (100% reuse) -----
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

        # ---- STAGE 2: SiLU(gate)*up -> PERSISTENT LDS Hidden_smem ------
        # Identical to _emit_gate_up_silu_epilogue_default MINUS the global
        # store loop: the f16 result stays in Hidden_smem (never reaches HBM).
        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
        cdec = _CWarpDecode(b, u_gu, warp_m_off, warp_n_off, lane)

        def _silu_cell(mi: int, ni: int, i: int) -> Value:
            flat = mi * mfmas_n + ni
            g = b.vec_extract(gate_res[flat], i)
            up = b.vec_extract(up_res[flat], i)
            return b.cast_f32_to(
                _silu_mul_f32(b, g, up, one_f32=one_f32, c_neg_log2e=c_neg_log2e),
                storage_dtype,
            )

        _emit_cshuffle_stage(
            b, u_gu, cdec, Hidden_smem, storage_dtype, c_per_lane, _silu_cell
        )
        b.sync()

        # ---- STAGE 3: RESHAPE Hidden_smem -> down-GEMM A layout --------
        # OPTION A (no new IR): _emit_cshuffle_stage stored Hidden_smem in
        # logical (row=m, col=inter) packed layout at cdec.coords. The down
        # k-loop reads A via _emit_smem_load(Hidden_smem, a_row=m,
        # a_col=contraction=inter) -- the SAME logical (m, i) indexing. The
        # C-output -> A-input reshape (pyisa G_reshape) is therefore implicit:
        # the cshuffle write address and the down MFMA A-read address are the
        # same logical element, so no explicit LDS transpose is required for the
        # 16x16x32 atom. (Option B / load_tile_transpose stays the parity
        # fallback if max_abs >= 5e-4.)

        # ---- STAGE 4 + 5: DOWN GEMM (Hidden_LDS @ Wdown^T) -> atomic Y -
        # grid.x split the inter contraction; this TG owns the inter slice at
        # gu_n_off and produces a PARTIAL Y over the WHOLE H_out, tiled in
        # tile_n_down chunks. Loop over the H_out output tiles; per tile run the
        # LDS-A down k-loop (contracting this TG's inter slice) then the
        # weighted, token-validity-masked atomic reduce into Y.
        down_for = b.scf_for_iter(c0, H_out, c_block_n_down, [], iv_name="ho")
        with down_for as ho:
            down_accs = [
                (f"down_acc_m{mi}_n{ni}", down_acc_init)
                for mi in range(down_mfmas_m)
                for ni in range(down_mfmas_n)
            ]
            down_res = _emit_moe_down_kloop_lds_a(
                plan_down,
                Hidden_smem,
                down_operand,
                down_accs,
                # W_down output-row base = ho; contraction base = this TG's
                # inter slice (gu_n_off).
                (batch_off_b, ho),
                gu_n_off,
                c_down_k,
                warp_m_idx,
                warp_n_idx,
                lane,
                sched_groups=down_mfmas_m * down_mfmas_n,
            )
            # Barrier before reusing Bd_smem in the next output tile.
            b.sync()
            _emit_down_reduce_epilogue_atomic(
                b,
                u_down,
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
                c_per_lane,
                batch_bucket_off=c0,
                tokens=tokens,
            )
            b.scf_yield()

    # Empty tail block (BlockExpertIds == -1) skips all work.
    with b.scf_if(b.cmp_ge(expert_idx, c0)):
        _emit_body()

    return b.kernel
