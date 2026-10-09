# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Not vendored: written for this provider. It reuses the WMMA operand layouts, masking
# and online softmax of flash_attn_func_rdna.py, the vendored prefill kernel beside it.

"""Decode attention for gfx11 (RDNA3 / RDNA3.5): split-KV with GQA-packed rows.

A decode step has a handful of query rows and a long key/value history, so the prefill
kernel -- 128 query rows per workgroup, sharing each K/V tile across eight waves -- spends
nearly all of its work on empty rows and reads each KV head once per query head. This
kernel turns that around:

* **Rows.** The query heads that share a KV head (the GQA group) and the query positions
  are packed into one 16-row WMMA tile: row r is query head ``kvh * g + r // Sq`` at
  position ``r % Sq``, valid while ``r < g * Sq``. One KV head is read once for the
  whole group; a group of more than 16 rows takes ``ceil(g * Sq / 16)`` row tiles.
* **Workgroups.** One per (batch, KV head, KV split) and row tile (the grid's second
  dimension); ``num_splits`` is a runtime
  argument the dispatcher picks from the shape and the device's compute-unit count. The
  workgroup's keys are those any row can see -- the window's edge to the diagonal -- cut
  into 32-key tiles and the tiles into splits. Above ``head_dim`` 128 each split has
  ``dv_split`` workgroups, one per slice of the output columns.
* **Waves.** NW waves hold the same rows and take every NW-th tile. Each streams its own
  K straight into the WMMA A operand and stages its own V through a private LDS region
  (WMMA reads V transposed); scheduling fences keep a bounded number of loads in flight.
  The waves combine their running max, sum and O through the same LDS at the end.
* **Output.** With one split the workgroup writes O and the natural-log LSE directly.
  With more it writes a normalized f32 partial and its LSE to workspace, and the merge
  kernel -- compiled into the same code object -- combines the splits by their LSEs.

Masks, scale and strides follow the prefill kernel's runtime arguments; LSE has the same
semantics (O = 0 and LSE = -inf for a row no key reaches). An additive bias
(``has_bias``, a compile-time variant; f32, or the query's dtype with
``bias_query_dtype``, widened to f32 as it is read) is read through its broadcast strides,
one row per lane, and added to the scaled scores; an object without one ignores the bias arguments.
A generic object (``generic_head_dim``) is built for the largest head_dim it serves and
reads the actual one at runtime, skipping the K and V columns past it.

Both WMMA operand layouts are built from this file, chosen by the arch FlyDSL compiles for:
gfx11 (RDNA3/3.5: 16 K-values per lane, accumulator rows 2r + l/16, a trade with the
xor-16 peer for P and O) and gfx12 (RDNA4: 8 K-values per lane at 8*(l/16), accumulator
rows r + 8*(l/16), no trade). The name says gfx11 because the gfx11 objects came first.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import LOG2E as _LOG2E

_NO_WINDOW = -(1 << 30)


def build_flash_attn_decode_module(
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_waves=4,
    waves_per_eu=None,
    daz=True,
    q_resident=None,
    dv_split=None,
    has_bias=False,
    bias_early=None,
    generic_head_dim=False,
    bias_query_dtype=False,
):
    WARP_SIZE = 32
    NW = num_waves
    BLOCK_SIZE = NW * WARP_SIZE
    TILE = 32  # keys per wave per iteration
    D = head_dim
    # The WMMA operand layout: gfx11 passes all 16 K-values per lane for A and B
    # (lanes 16-31 duplicate 0-15) and lays accumulator element r at row
    # 2r + l/16; gfx12 passes 8, at K offset 8*(l/16), and lays element r at row
    # r + 8*(l/16). Chosen by the arch FlyDSL compiles for.
    WMMA_V8 = get_rocm_arch().lower().startswith("gfx12")
    WMMA_AB = 8 if WMMA_V8 else 16
    assert D % 32 == 0 and 64 <= D <= 256
    K_STEPS = D // 16
    D_CHUNKS = D // 16
    CAUSAL = causal
    # An additive f32 bias, broadcast by strides, added to the scaled scores: per-score
    # loads in the inner loop, so a compile-time variant (COVERAGE.md §6).
    HAS_BIAS = bool(has_bias)
    # The bias in the query's dtype rather than f32 (bias_query_dtype): read as such and
    # widened to f32 as it is added.
    BIAS_Q = bool(bias_query_dtype)
    # Row tiles (groups of more than 16 rows) on every object but the bias ones, which
    # already use the whole register file at 128 and up: the row index they would hold
    # through the KV loop spills there. A bias object serves up to 16 rows.
    ROW_TILES = not HAS_BIAS
    # Issue the bias loads before the QK product (hidden behind it) or after it. Ahead
    # pays at 128 and up, where the score epilogue has latency to hide; below, after.
    BIAS_EARLY = (D >= 128) if bias_early is None else bool(bias_early)
    # head_dim is then the largest served; the actual one, a multiple of 8, is a
    # runtime argument. K and V columns past it are fetched out of the buffer's
    # range, so the hardware returns zeros without reading memory, and zeroed
    # again in case the slice is large enough for that offset to land inside it;
    # Q columns past it are zeroed, O columns past it are not stored, and the
    # split workspace's rows are the actual head_dim wide.
    GENERIC_D = bool(generic_head_dim)
    # Above 128 the output columns are split across DV_SPLIT workgroups, as in the
    # prefill kernel: each computes the whole of QK^T but only DV_TILE columns of O,
    # so it holds and stages only those.
    DV_SPLIT = (2 if D > 128 else 1) if dv_split is None else int(dv_split)
    assert D_CHUNKS % DV_SPLIT == 0
    DV_TILE = D // DV_SPLIT
    O_CHUNKS = DV_TILE // 16
    V_STRIDE = DV_TILE + 4
    WAVE_LDS = TILE * V_STRIDE  # elements; also holds the combine record
    elem_cls = {"f16": fx.Float16, "bf16": fx.BFloat16}[dtype_str]
    MERGE_THREADS = 64
    # Q is 16 rows and stays in cache, so above 96 it is re-read per K-step rather than
    # held next to the O accumulators.
    Q_RESIDENT = (D <= 96) if q_resident is None else bool(q_resident)
    # Loads in flight before a scheduling fence: left alone the scheduler issues every
    # K and V load of the tile at once and holds them all in registers.
    V_FENCE = 4
    K_FENCE = 2

    @fx.struct
    class SharedStorage:
        v: fx.Array[elem_cls, NW * WAVE_LDS, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_decode_rdna_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        LSE: fx.Pointer,
        WS_O: fx.Pointer,
        WS_L: fx.Pointer,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
        num_heads: fx.Int32,
        kv_group: fx.Int32,
        right_bound: fx.Int32,
        left_bound: fx.Int32,
        align_br: fx.Int32,
        lse_on: fx.Int32,
        num_splits: fx.Int32,
        scale: fx.Float32,
        q_sb: fx.Int64,
        q_ss: fx.Int64,
        q_sh: fx.Int64,
        k_sb: fx.Int64,
        k_ss: fx.Int64,
        k_sh: fx.Int64,
        v_sb: fx.Int64,
        v_ss: fx.Int64,
        v_sh: fx.Int64,
        o_sb: fx.Int64,
        o_ss: fx.Int64,
        o_sh: fx.Int64,
        lse_sb: fx.Int64,
        lse_ss: fx.Int64,
        lse_sh: fx.Int64,
        BIAS: fx.Pointer,
        bias_sb: fx.Int64,
        bias_sh: fx.Int64,
        bias_sq: fx.Int64,
        bias_skv: fx.Int64,
        head_dim_rt: fx.Int32,
    ):
        elem = elem_cls

        def _fmax(a, b):
            return fx.Float32(a).maximumf(fx.Float32(b))

        def _as_ptr(ptr, ty):
            return fx.recast_iter(
                fx.PointerType.get(ty.ir_type, ptr.address_space), ptr
            )

        q_ptr = _as_ptr(Q, elem)
        k_ptr = _as_ptr(K, elem)
        v_ptr = _as_ptr(V, elem)
        o_ptr = _as_ptr(O, elem)
        wso_ptr = _as_ptr(WS_O, fx.Float32)
        wsl_ptr = _as_ptr(WS_L, fx.Float32)
        lse_ptr = _as_ptr(LSE, fx.Float32)
        bias_ptr = _as_ptr(BIAS, elem if BIAS_Q else fx.Float32)

        def bias_load(p):
            v = fx.ptr_load(p)
            return v.to(fx.Float32) if const_expr(BIAS_Q) else v

        def _buf(ptr, num_records_bytes):
            flags = (7 << 12) | (4 << 15) | (1 << 24) | (3 << 28)
            ty = fx.PointerType.get(
                elem_ty=ptr.element_type.ir_type,
                address_space=fx.rocdl.TargetAddressSpace.BufferDesc,
                alignment=ptr.alignment,
            )
            return fx.make_ptr(
                ty,
                [
                    ptr,
                    fx.Int16(0).ir_value(),
                    fx.Int64(num_records_bytes).ir_value(),
                    fx.Int32(flags).ir_value(),
                ],
            )

        atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, elem, fx.Float32))

        def wmma(a, b, c):
            af = fx.make_rmem_tensor(WMMA_AB, elem)
            bf = fx.make_rmem_tensor(WMMA_AB, elem)
            cf = fx.make_rmem_tensor(8, fx.Float32)
            af.store(Vec(a))
            bf.store(Vec(b))
            cf.store(Vec(c))
            fx.gemm(atom, cf, af, bf, cf)
            return Vec(cf.load())

        def gload(ptr, idx, width):
            return Vec(
                fx.make_view(
                    fx.add_offset(ptr, fx.Int64(idx)), fx.make_layout(width, 1)
                ).load()
            )

        def gstore(ptr, idx, val):
            fx.make_view(
                fx.add_offset(ptr, fx.Int64(idx)), fx.make_layout(val.numel, 1)
            ).store(Vec(val))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_v = lds.v.ptr
        lds_f32 = fx.recast_iter(
            fx.PointerType.get(fx.Float32.ir_type, lds_v.address_space), lds_v
        )

        sq = fx.Int64(seq_len_q)
        skv = fx.Int64(seq_len_kv)
        hq_n = fx.Int64(num_heads)
        g = fx.Int64(kv_group)
        ns = fx.Int64(num_splits)
        hkv_n = hq_n // g

        bid = fx.Int64(gpu.block_idx.x)
        tid = fx.Int64(gpu.thread_idx.x)
        wave = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16
        klane_is_zero = klane == fx.Int64(0)

        dv_idx = bid % DV_SPLIT
        dv_col_base = dv_idx * DV_TILE
        bid_rest = bid // DV_SPLIT
        split = bid_rest % ns
        rest = bid_rest // ns
        kvh = rest % hkv_n
        b = rest // hkv_n

        # Packed row of this lane: query head kvh * g + r // Sq at position r % Sq.
        rows = g * sq
        if const_expr(ROW_TILES):
            # A group of more than 16 rows takes several row tiles, one workgroup each,
            # along the grid's second dimension.
            r_tile = fx.Int64(gpu.block_idx.y) * fx.Int64(16) + lane16
            row_ok = r_tile < rows
            r = fx.Int64(row_ok.select(r_tile, fx.Int64(0)))
        else:
            row_ok = lane16 < rows
            r = fx.Int64(row_ok.select(lane16, fx.Int64(0)))
        hq = kvh * g + r // sq
        qpos = r % sq

        # The rows' mask, in the reference's terms (as the prefill kernel).
        skv_i32 = fx.Int32(seq_len_kv)
        align_i32 = fx.Int32(
            (fx.Int32(align_br) != fx.Int32(0)).select(
                skv_i32 - fx.Int32(seq_len_q), fx.Int32(0)
            )
        )
        coff_i32 = align_i32 + fx.Int32(right_bound)
        wlo_i32 = fx.Int32(
            (fx.Int32(left_bound) >= fx.Int32(0)).select(
                align_i32 - fx.Int32(left_bound), fx.Int32(_NO_WINDOW)
            )
        )
        qpos_i32 = fx.Int32(qpos)
        first_i32 = qpos_i32 + wlo_i32
        if const_expr(CAUSAL):
            _end = qpos_i32 + coff_i32 + fx.Int32(1)
            end_i32 = fx.Int32((_end < skv_i32).select(_end, skv_i32))
        else:
            end_i32 = skv_i32
        span_u32 = fx.Uint32(end_i32 - first_i32)

        # The keys any row may see: from the window's edge for position 0 to the diagonal
        # for position Sq - 1. Split into tiles, the tiles into num_splits chunks.
        _lo = fx.Int64(wlo_i32)
        lo = fx.Int64((_lo > fx.Int64(0)).select(_lo, fx.Int64(0)))
        if const_expr(CAUSAL):
            _hi = sq - fx.Int64(1) + fx.Int64(coff_i32) + fx.Int64(1)
            hi = fx.Int64((_hi < skv).select(_hi, skv))
        else:
            hi = skv
        lo_tile = lo // TILE
        hi_tile = (hi + TILE - 1) // TILE
        n_tiles = fx.Int64((hi_tile > lo_tile).select(hi_tile - lo_tile, fx.Int64(0)))
        chunk = (n_tiles + ns - 1) // ns
        t_begin = lo_tile + split * chunk
        _t_end = t_begin + chunk
        t_end = fx.Int64((_t_end < hi_tile).select(_t_end, hi_tile))
        iters = (chunk + NW - 1) // NW

        # Q (B operand of S^T = K Q^T): 16 head-dim values per lane on gfx11, 8 on
        # gfx12.
        q_base = b * q_sb + hq * q_sh + qpos * q_ss
        zero16 = Vec.filled(16, 0.0, elem)

        if const_expr(GENERIC_D):
            d_rt = fx.Int64(head_dim_rt)
            zero8e = Vec.filled(8, 0.0, elem)
            # An element offset past any slice the dispatcher accepts (< 2 GiB).
            past_slice = fx.Int64(0x3FFFFFF0)

        def q_pack(ks):
            if const_expr(WMMA_V8):
                col = fx.Int64(ks * 16) + klane * 8
                zero8q = Vec.filled(8, 0.0, elem)
                if const_expr(GENERIC_D):
                    ok = col < d_rt
                    raw = gload(q_ptr, q_base + ok.select(col, fx.Int64(0)), 8)
                    return (row_ok & ok).select(raw, zero8q)
                raw = gload(q_ptr, q_base + col, 8)
                return row_ok.select(raw, zero8q)
            if const_expr(GENERIC_D):
                halves = []
                for h in range_constexpr(2):
                    col = fx.Int64(ks * 16 + h * 8)
                    ok = col < d_rt
                    raw = gload(q_ptr, q_base + ok.select(col, fx.Int64(0)), 8)
                    halves.append((row_ok & ok).select(raw, zero8e))
                return Vec.from_elements(
                    [halves[0][i] for i in range(8)] + [halves[1][i] for i in range(8)],
                    elem,
                )
            raw = gload(q_ptr, q_base + fx.Int64(ks * 16), 16)
            return row_ok.select(raw, zero16)

        def kv_load(buf, row_base, col):
            """Eight columns of one K or V row; past head_dim, zeros without a read."""
            if const_expr(GENERIC_D):
                ok = col < d_rt
                raw = gload(buf, ok.select(row_base + col, past_slice), 8)
                return ok.select(raw, zero8e)
            return gload(buf, row_base + col, 8)

        q_packs = []
        if const_expr(Q_RESIDENT):
            for ks in range_constexpr(K_STEPS):
                q_packs.append(q_pack(ks))

        k_slice = b * k_sb + kvh * k_sh
        v_slice = b * v_sb + kvh * v_sh
        if const_expr(GENERIC_D):
            k_buf = _buf(
                fx.add_offset(k_ptr, k_slice),
                ((skv - fx.Int64(1)) * k_ss + d_rt) * fx.Int64(2),
            )
            v_buf = _buf(
                fx.add_offset(v_ptr, v_slice),
                ((skv - fx.Int64(1)) * v_ss + d_rt) * fx.Int64(2),
            )
        else:
            k_buf = _buf(
                fx.add_offset(k_ptr, k_slice),
                ((skv - fx.Int64(1)) * k_ss + fx.Int64(D)) * fx.Int64(2),
            )
            v_buf = _buf(
                fx.add_offset(v_ptr, v_slice),
                ((skv - fx.Int64(1)) * v_ss + fx.Int64(D)) * fx.Int64(2),
            )

        c_ninf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)
        c_one = fx.Float32(1.0)
        if const_expr(HAS_BIAS):
            # The GQA group's slice of the broadcast bias; each lane reads its own
            # (head, position) row. Reads past the slice -- the KV tail -- are zero.
            bias_buf = _buf(
                fx.add_offset(bias_ptr, b * bias_sb + kvh * g * bias_sh),
                (
                    (g - fx.Int64(1)) * bias_sh
                    + (sq - fx.Int64(1)) * bias_sq
                    + (skv - fx.Int64(1)) * bias_skv
                    + fx.Int64(1)
                )
                * fx.Int64(2 if BIAS_Q else 4),
            )
            bias_row_i32 = fx.Int32((hq - kvh * g) * bias_sh + qpos * bias_sq)
            bias_skv_i32 = fx.Int32(bias_skv)
            # Biased scores are scaled before the softmax, so the running max and the
            # exponent are in the scaled domain.
            c_scale = fx.Float32(scale)
            c_sl2e = fx.Float32(_LOG2E)
            c_lse_scale = c_one
        else:
            c_sl2e = fx.Float32(scale) * fx.Float32(_LOG2E)
            c_lse_scale = fx.Float32(scale)
        zero8 = Vec.filled(8, 0.0, fx.Float32)
        w32 = fx.Int32(WARP_SIZE)
        x16 = fx.Int32(16)

        def peer(v):
            return fx.Float32(v).shuffle_xor(x16, w32)

        wave_lds = wave * WAVE_LDS

        init = [c_ninf, c_zero] + [zero8 for _ in range(O_CHUNKS)]
        res = init
        for it, carry in range(fx.Int64(0), iters, fx.Int64(1), init=init):
            m_run = carry[0]
            l_run = carry[1]
            o_acc = [carry[2 + i] for i in range_constexpr(O_CHUNKS)]

            t = t_begin + it * NW + wave
            kv0 = t * TILE
            active = t < t_end

            # V: this wave's 32 rows into its LDS region, one row per lane.
            v_row = kv0 + lane
            for c in range_constexpr(DV_TILE // 8):
                if const_expr(GENERIC_D):
                    vv = kv_load(v_buf, v_row * v_ss, dv_col_base + fx.Int64(c * 8))
                else:
                    vv = gload(v_buf, v_row * v_ss + dv_col_base + fx.Int64(c * 8), 8)
                fx.ptr_store(
                    Vec(vv), lds_v + fx.Int32(wave_lds + lane * V_STRIDE + c * 8)
                )
                if const_expr(c % V_FENCE == V_FENCE - 1):
                    fx.rocdl.sched_barrier(0)

            if const_expr(HAS_BIAS and WMMA_V8):
                # Element i = 8 a + e is key kv0 + 16 a + e + 8 klane.
                b_tile = (
                    bias_row_i32
                    + (fx.Int32(kv0) + fx.Int32(klane) * fx.Int32(8)) * bias_skv_i32
                )
                bias_offs = [
                    b_tile + fx.Int32((i // 8) * 16 + i % 8) * bias_skv_i32
                    for i in range(16)
                ]
                if const_expr(BIAS_EARLY):
                    bias_vals = [bias_load(bias_buf + off) for off in bias_offs]
            elif const_expr(HAS_BIAS):
                # Element i = 8 a + e is key kv0 + 16 a + 2 e + klane.
                b_tile = bias_row_i32 + (fx.Int32(kv0) + fx.Int32(klane)) * bias_skv_i32
                bias_offs = [
                    b_tile + fx.Int32((i // 8) * 16 + 2 * (i % 8)) * bias_skv_i32
                    for i in range(16)
                ]
                if const_expr(BIAS_EARLY):
                    bias_vals = [bias_load(bias_buf + off) for off in bias_offs]

            s_acc = [zero8, zero8]
            if active:
                for ks in range_constexpr(K_STEPS):
                    if const_expr(not Q_RESIDENT):
                        q_cur = q_pack(ks)
                    for a in range_constexpr(2):
                        k_row = kv0 + fx.Int64(a * 16) + lane16
                        if const_expr(WMMA_V8):
                            kp = kv_load(
                                k_buf, k_row * k_ss, fx.Int64(ks * 16) + klane * 8
                            )
                        else:
                            # Buffer loads cap at 128 bits: two 8-element halves.
                            k_lo = kv_load(k_buf, k_row * k_ss, fx.Int64(ks * 16))
                            k_hi = kv_load(k_buf, k_row * k_ss, fx.Int64(ks * 16 + 8))
                            kp = Vec.from_elements(
                                [k_lo[i] for i in range(8)]
                                + [k_hi[i] for i in range(8)],
                                elem,
                            )
                        s_acc[a] = wmma(
                            kp,
                            q_packs[ks] if const_expr(Q_RESIDENT) else q_cur,
                            s_acc[a],
                        )
                    if const_expr(ks % K_FENCE == K_FENCE - 1):
                        fx.rocdl.sched_barrier(0)

            s_vals = [Vec(s_acc[a])[e] for a in range(2) for e in range(8)]
            if const_expr(HAS_BIAS):
                if const_expr(not BIAS_EARLY):
                    bias_vals = [bias_load(bias_buf + off) for off in bias_offs]
                for i in range_constexpr(16):
                    s_vals[i] = fx.math.fma(s_vals[i], c_scale, bias_vals[i])

            if const_expr(WMMA_V8):
                rel_base = fx.Int32(kv0) + fx.Int32(klane) * fx.Int32(8) - first_i32
            else:
                rel_base = fx.Int32(kv0) + fx.Int32(klane) - first_i32
            s_raw = []
            for a in range_constexpr(2):
                for e in range_constexpr(8):
                    sv = s_vals[a * 8 + e]
                    _kv = a * 16 + (e if const_expr(WMMA_V8) else 2 * e)
                    rel = fx.Uint32(rel_base + fx.Int32(_kv))
                    ok = (rel < span_u32) & active
                    s_raw.append(ok.select(sv, c_ninf))

            lmax = s_raw[0]
            for e in range_constexpr(15):
                lmax = _fmax(lmax, s_raw[e + 1])
            rmax = _fmax(lmax, peer(lmax))
            m_new = _fmax(m_run, rmax)
            m_safe = (m_new == c_ninf).select(c_zero, m_new)
            corr = fx.Float32(
                fx.rocdl.exp2(
                    fx.Float32.ir_type, fx.Float32((m_run - m_safe) * c_sl2e).ir_value()
                )
            )
            nsm = c_zero - c_sl2e * m_safe
            p_vals = []
            lsum = c_zero
            for e in range_constexpr(16):
                p = fx.Float32(
                    fx.rocdl.exp2(
                        fx.Float32.ir_type,
                        fx.Float32(fx.math.fma(s_raw[e], c_sl2e, nsm)).ir_value(),
                    )
                )
                p_vals.append(p)
                lsum = lsum + p
            l_new = corr * l_run + (lsum + peer(lsum))
            corr_v = Vec.from_elements([corr], fx.Float32).broadcast_to(8)
            for dc in range_constexpr(O_CHUNKS):
                o_acc[dc] = o_acc[dc] * corr_v

            gpu.barrier()

            p_packs = []
            for pks in range_constexpr(2):
                own = [p_vals[pks * 8 + j] for j in range(8)]
                if const_expr(WMMA_V8):
                    full = own
                else:
                    prr = [peer(own[j]) for j in range(8)]
                    full = []
                    for j in range_constexpr(8):
                        full.append(klane_is_zero.select(own[j], prr[j]))
                        full.append(klane_is_zero.select(prr[j], own[j]))
                p_packs.append(
                    Vec.from_elements([fx.Float32(x).to(elem) for x in full], elem)
                )

            def v_pack(pks, dc):
                dpos = fx.Int64(dc * 16) + lane16
                vals = []
                if const_expr(WMMA_V8):
                    for ksub in range_constexpr(8):
                        krow = fx.Int64(pks * 16 + ksub) + klane * 8
                        idx = wave_lds + krow * V_STRIDE + dpos
                        vals.append(fx.ptr_load(lds_v + fx.Int32(idx)))
                    return Vec.from_elements(vals, elem)
                for ksub in range_constexpr(16):
                    idx = wave_lds + fx.Int64(pks * 16 + ksub) * V_STRIDE + dpos
                    vals.append(fx.ptr_load(lds_v + fx.Int32(idx)))
                return Vec.from_elements(vals, elem)

            if active:
                o_tmp = list(o_acc)
                for pks in range_constexpr(2):
                    for dc in range_constexpr(O_CHUNKS):
                        o_tmp[dc] = wmma(v_pack(pks, dc), p_packs[pks], o_tmp[dc])
                o_acc = o_tmp

            gpu.barrier()
            res = yield [m_new, l_new] + o_acc

        m_w = res[0]
        l_w = res[1]
        o_w = [res[2 + i] for i in range_constexpr(O_CHUNKS)]

        # Combine the waves: each writes (m, l, O fragments) to its LDS region as f32.
        rec = wave * (WAVE_LDS // 2)  # f32 index of this wave's record
        fx.ptr_store(
            Vec.from_elements([m_w, l_w], fx.Float32),
            lds_f32 + fx.Int32(rec + lane * 2),
        )
        for dc in range_constexpr(O_CHUNKS):
            fx.ptr_store(
                Vec(o_w[dc]), lds_f32 + fx.Int32(rec + 64 + (dc * 32 + lane) * 8)
            )
        gpu.barrier()

        if wave == fx.Int64(0):
            ms = []
            ls = []
            for w in range_constexpr(NW):
                wr = w * (WAVE_LDS // 2)
                ml = Vec(
                    fx.make_view(
                        lds_f32 + fx.Int32(wr + lane * 2), fx.make_layout(2, 1)
                    ).load()
                )
                ms.append(ml[0])
                ls.append(ml[1])
            m_all = ms[0]
            for w in range_constexpr(NW - 1):
                m_all = _fmax(m_all, ms[w + 1])
            m_all_safe = (m_all == c_ninf).select(c_zero, m_all)
            fs = []
            l_all = c_zero
            for w in range_constexpr(NW):
                f = fx.Float32(
                    fx.rocdl.exp2(
                        fx.Float32.ir_type,
                        fx.Float32((ms[w] - m_all_safe) * c_sl2e).ir_value(),
                    )
                )
                fs.append(f)
                l_all = l_all + ls[w] * f
            has = l_all > c_zero
            inv_l = has.select(c_one / l_all, c_zero)
            lse_val = has.select(c_lse_scale * m_all + fx.math.log(l_all), c_ninf)
            single = ns == fx.Int64(1)
            o_base = b * o_sb + hq * o_sh + qpos * o_ss
            prow = ((b * hq_n + hq) * sq + qpos) * ns + split
            # Combined and stored one 16-column chunk at a time, so no more than one
            # chunk's output is live.
            for dc in range_constexpr(O_CHUNKS):
                acc = zero8
                for w in range_constexpr(NW):
                    wr = w * (WAVE_LDS // 2)
                    frag = Vec(
                        fx.make_view(
                            lds_f32 + fx.Int32(wr + 64 + (dc * 32 + lane) * 8),
                            fx.make_layout(8, 1),
                        ).load()
                    )
                    acc = acc + frag * Vec.from_elements(
                        [fs[w]], fx.Float32
                    ).broadcast_to(8)
                on = Vec(acc * Vec.from_elements([inv_l], fx.Float32).broadcast_to(8))
                own = [on[e] for e in range(8)]
                if const_expr(WMMA_V8):
                    # gfx12: element e is head-dim row e + 8*klane already.
                    rows_ = own
                else:
                    prr = [peer(own[e]) for e in range(8)]
                    rows_ = []
                    for j in range_constexpr(8):
                        if const_expr(j % 2 == 0):
                            lo_s, hi_s = own, prr
                        else:
                            lo_s, hi_s = prr, own
                        rows_.append(
                            klane_is_zero.select(lo_s[j // 2], hi_s[4 + j // 2])
                        )
                if const_expr(GENERIC_D):
                    o_col = dv_col_base + fx.Int64(dc * 16) + klane * 8
                    if row_ok & (o_col < d_rt):
                        if single:
                            out = Vec.from_elements(rows_, fx.Float32).to(elem)
                            gstore(o_ptr, o_base + o_col, out)
                        else:
                            gstore(
                                wso_ptr,
                                prow * d_rt + o_col,
                                Vec.from_elements(rows_, fx.Float32),
                            )
                elif row_ok:
                    if single:
                        out = Vec.from_elements(rows_, fx.Float32).to(elem)
                        gstore(
                            o_ptr,
                            o_base + dv_col_base + fx.Int64(dc * 16) + klane * 8,
                            out,
                        )
                    else:
                        gstore(
                            wso_ptr,
                            prow * D + dv_col_base + fx.Int64(dc * 16) + klane * 8,
                            Vec.from_elements(rows_, fx.Float32),
                        )
                fx.rocdl.sched_barrier(0)

            # Every column tile of a row computes the same LSE; the first writes it.
            if row_ok & (dv_idx == fx.Int64(0)):
                if single:
                    if fx.Int32(lse_on) != fx.Int32(0):
                        gstore(
                            lse_ptr,
                            b * lse_sb + hq * lse_sh + qpos * lse_ss,
                            Vec.from_elements([lse_val], fx.Float32),
                        )
                else:
                    gstore(wsl_ptr, prow, Vec.from_elements([lse_val], fx.Float32))

    @flyc.kernel(known_block_size=[MERGE_THREADS, 1, 1])
    def flash_attn_decode_merge_rdna_kernel(
        O: fx.Pointer,
        LSE: fx.Pointer,
        WS_O: fx.Pointer,
        WS_L: fx.Pointer,
        seq_len_q: fx.Int32,
        num_heads: fx.Int32,
        lse_on: fx.Int32,
        num_splits: fx.Int32,
        o_sb: fx.Int64,
        o_ss: fx.Int64,
        o_sh: fx.Int64,
        lse_sb: fx.Int64,
        lse_ss: fx.Int64,
        lse_sh: fx.Int64,
        head_dim_rt: fx.Int32,
    ):
        def _as_ptr(ptr, ty):
            return fx.recast_iter(
                fx.PointerType.get(ty.ir_type, ptr.address_space), ptr
            )

        o_ptr = _as_ptr(O, elem_cls)
        lse_ptr = _as_ptr(LSE, fx.Float32)
        wso = _as_ptr(WS_O, fx.Float32)
        wsl = _as_ptr(WS_L, fx.Float32)
        row = fx.Int64(gpu.block_idx.x)  # (b * H + h) * Sq + s
        tid = fx.Int64(gpu.thread_idx.x)
        sq = fx.Int64(seq_len_q)
        hq_n = fx.Int64(num_heads)
        ns = fx.Int64(num_splits)
        s = row % sq
        h = (row // sq) % hq_n
        b = row // (sq * hq_n)
        c_ninf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)

        m = c_ninf
        for i, carry in range(fx.Int64(0), ns, fx.Int64(1), init=[m]):
            li = fx.ptr_load(wsl + fx.Int32(row * ns + i))
            mx = fx.Float32(carry[0]).maximumf(fx.Float32(li))
            m = yield [mx]
        m_all = m
        m_safe = (m_all == c_ninf).select(c_zero, m_all)
        DPT = (D + MERGE_THREADS - 1) // MERGE_THREADS
        # A generic object's workspace rows, and O's, are the actual head_dim wide.
        if const_expr(GENERIC_D):
            ws_d = fx.Int64(head_dim_rt)
        init = [c_zero] + [c_zero for _ in range(DPT)]
        res = init
        for i, carry in range(fx.Int64(0), ns, fx.Int64(1), init=init):
            li = fx.ptr_load(wsl + fx.Int32(row * ns + i))
            w = fx.math.exp(fx.Float32(li) - m_safe)
            w = (fx.Float32(li) == c_ninf).select(c_zero, w)
            acc = []
            for j in range_constexpr(DPT):
                d = tid + fx.Int64(j * MERGE_THREADS)
                if const_expr(GENERIC_D):
                    dd = fx.Int64((d < ws_d).select(d, fx.Int64(0)))
                    ov = fx.ptr_load(wso + fx.Int32((row * ns + i) * ws_d + dd))
                else:
                    dd = fx.Int64((d < fx.Int64(D)).select(d, fx.Int64(0)))
                    ov = fx.ptr_load(wso + fx.Int32((row * ns + i) * D + dd))
                acc.append(fx.Float32(carry[1 + j]) + w * fx.Float32(ov))
            res = yield [fx.Float32(carry[0]) + w] + acc
        l_all = res[0]
        has = l_all > c_zero
        inv = has.select(fx.Float32(1.0) / l_all, c_zero)
        for j in range_constexpr(DPT):
            d = tid + fx.Int64(j * MERGE_THREADS)
            if d < (ws_d if const_expr(GENERIC_D) else fx.Int64(D)):
                val = fx.Float32(res[1 + j]) * inv
                fx.ptr_store(
                    Vec.from_elements([val.to(elem_cls)], elem_cls),
                    o_ptr + fx.Int32(b * o_sb + h * o_sh + s * o_ss + d),
                )
        if tid == fx.Int64(0):
            if fx.Int32(lse_on) != fx.Int32(0):
                lse = has.select(m_all + fx.math.log(l_all), c_ninf)
                fx.ptr_store(
                    Vec.from_elements([lse], fx.Float32),
                    lse_ptr + fx.Int32(b * lse_sb + h * lse_sh + s * lse_ss),
                )

    @flyc.jit
    def launch_flash_attn_decode(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        LSE: fx.Pointer,
        WS_O: fx.Pointer,
        WS_L: fx.Pointer,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_kv: fx.Int32,
        num_heads: fx.Int32,
        kv_group: fx.Int32,
        right_bound: fx.Int32,
        left_bound: fx.Int32,
        align_br: fx.Int32,
        lse_on: fx.Int32,
        num_splits: fx.Int32,
        scale: fx.Float32,
        q_sb: fx.Int64,
        q_ss: fx.Int64,
        q_sh: fx.Int64,
        k_sb: fx.Int64,
        k_ss: fx.Int64,
        k_sh: fx.Int64,
        v_sb: fx.Int64,
        v_ss: fx.Int64,
        v_sh: fx.Int64,
        o_sb: fx.Int64,
        o_ss: fx.Int64,
        o_sh: fx.Int64,
        lse_sb: fx.Int64,
        lse_ss: fx.Int64,
        lse_sh: fx.Int64,
        BIAS: fx.Pointer,
        bias_sb: fx.Int64,
        bias_sh: fx.Int64,
        bias_sq: fx.Int64,
        bias_skv: fx.Int64,
        head_dim_rt: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ctx = CompilationContext.get_current()
        grid_main = (
            fx.Uint64(batch_size)
            * (fx.Uint64(num_heads) // fx.Uint64(kv_group))
            * fx.Uint64(num_splits)
            * DV_SPLIT
        )
        row_tiles = (
            (fx.Uint64(kv_group) * fx.Uint64(seq_len_q) + 15) // 16
            if const_expr(ROW_TILES)
            else fx.Uint64(1)
        )
        grid_merge = fx.Uint64(batch_size) * fx.Uint64(num_heads) * fx.Uint64(seq_len_q)
        main = flash_attn_decode_rdna_kernel(
            Q,
            K,
            V,
            O,
            LSE,
            WS_O,
            WS_L,
            seq_len_q,
            seq_len_kv,
            num_heads,
            kv_group,
            right_bound,
            left_bound,
            align_br,
            lse_on,
            num_splits,
            scale,
            q_sb,
            q_ss,
            q_sh,
            k_sb,
            k_ss,
            k_sh,
            v_sb,
            v_ss,
            v_sh,
            o_sb,
            o_ss,
            o_sh,
            lse_sb,
            lse_ss,
            lse_sh,
            BIAS,
            bias_sb,
            bias_sh,
            bias_sq,
            bias_skv,
            head_dim_rt,
        )
        merge = flash_attn_decode_merge_rdna_kernel(
            O,
            LSE,
            WS_O,
            WS_L,
            seq_len_q,
            num_heads,
            lse_on,
            num_splits,
            o_sb,
            o_ss,
            o_sh,
            lse_sb,
            lse_ss,
            lse_sh,
            head_dim_rt,
        )
        passthrough = []
        if const_expr(daz):
            passthrough.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough)
                if const_expr(waves_per_eu is not None):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(waves_per_eu)
                    )
        main.launch(
            grid=(grid_main, row_tiles, 1), block=(BLOCK_SIZE, 1, 1), stream=stream
        )
        merge.launch(
            grid=(grid_merge, 1, 1), block=(MERGE_THREADS, 1, 1), stream=stream
        )

    launch_flash_attn_decode.compile_hints = {
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }
    launch_flash_attn_decode.block_size = BLOCK_SIZE
    launch_flash_attn_decode.dv_split = DV_SPLIT
    launch_flash_attn_decode.generic_head_dim = GENERIC_D
    launch_flash_attn_decode.row_tiles = ROW_TILES
    launch_flash_attn_decode.merge_threads = MERGE_THREADS
    return launch_flash_attn_decode
