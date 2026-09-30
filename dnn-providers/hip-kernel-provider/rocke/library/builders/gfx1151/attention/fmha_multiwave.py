# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Multi-wave WMMA FMHA-forward kernel for the gfx1151 optimization campaign.

This is the *structural* lever the single-wave campaign (``fmha_singlewave.py``)
identified as the only real path past ~10 TF: put ``n_waves`` wave32s in one
workgroup and **cooperatively stage K and V into LDS once per K-tile**, shared
across all waves. That does three things the single-wave kernel could not:

  * **Amortizes K/V global traffic** over ``n_waves`` query-row tiles -- one CTA
    loads each K/V tile once and feeds ``n_waves`` independent QK/PV pipelines.
  * **Gives the LDS barrier something to hide behind.** Every prior staging
    attempt died because a lone wave32 had no second wave to overlap the
    ``s_barrier`` with; with ``n_waves`` resident the barrier latency is hidden.
  * **Makes both WMMA B-operands contiguous LDS reads.** K is stored ``[kv][d]``
    (QK B-operand reads a contiguous d-slice); V is stored *transposed*
    ``[d][kv]`` (PV B-operand reads a contiguous kv-slice). The V transpose
    scatter is cooperative -- paid once per CTA, not once per wave.

Each wave owns one 16-row Q-tile. CTA owns ``16 * n_waves`` Q rows. Same WMMA
contract / ABI as ``fmha_singlewave`` and the production kernel, and built on the same
CK Tile helper layer: 3D :func:`~rocke.helpers.make_global_view` views drive
both the per-wave Q fragment (via :func:`~rocke.helpers.load_wmma_fragment`) and
the cooperative K/V global reads; the shared K, transposed V, and per-wave P all
live in :func:`~rocke.helpers.make_lds_view` LDS views; the matmuls are
:class:`~rocke.helpers.WmmaAtom`; and the O epilogue uses
:func:`~rocke.helpers.store_wmma_acc`.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import F16, F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers import (
    WmmaAtom,
    WmmaTensor,
    load_wmma_tile,
    make_global_view,
    make_lds_view,
    make_tile_window,
    store_wmma_tile,
    wmma_mma,
)
from rocke.helpers.attention import (
    apply_attention_mask,
    wave_reduce_max,
    wave_reduce_sum,
)

_WMMA_OP_ID = "wmma_f32_16x16x16_f16"
_BLOCK_K = 16


@dataclass(frozen=True)
class MultiWaveCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    n_waves: int = 4  # wave32s per CTA; each owns a 16-row Q-tile
    name: str = "wmma_fmha_multiwave"

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        return 32 * self.n_waves

    @property
    def q_rows_per_cta(self) -> int:
        return 16 * self.n_waves

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            self.mask_mode,
            f"w{self.n_waves}",
        )


def multiwave_grid(cfg: MultiWaveCfg, *, seqlen_q: int, batch: int):
    q_per = cfg.q_rows_per_cta
    if seqlen_q % q_per != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {q_per}")
    return (seqlen_q // q_per, cfg.num_query_heads, batch)


def _declare_params(b: IRBuilder):
    P = {}
    P["Q"] = b.param("Q", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["K"] = b.param("K", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["V"] = b.param("V", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["O"] = b.param(
        "O", PtrType(F16, "global"), noalias=True, writeonly=True, align=16
    )
    P["scale_log2"] = b.param("scale_log2", F32)
    P["seqlen_q"] = b.param("seqlen_q", I32)
    P["seqlen_k"] = b.param("seqlen_k", I32)
    for nm in (
        "stride_q_token",
        "stride_q_head",
        "stride_k_token",
        "stride_k_head",
        "stride_v_token",
        "stride_v_head",
        "stride_o_token",
        "stride_o_head",
    ):
        P[nm] = b.param(nm, I32)
    return P


def build_wmma_fmha_multiwave(cfg: MultiWaveCfg, arch: str = "gfx1151") -> KernelDef:
    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    a_map = atom.a_layout(arch)
    c_map = atom.c_layout(arch)
    a_frag = atom.a_per_lane  # 16  # noqa: F841
    c_frag = atom.c_per_lane  # 8
    n_dk = cfg.head_size // 16
    hs = cfg.head_size
    W = cfg.n_waves
    dtype_ir = F16

    b = IRBuilder(cfg.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = cfg.block_size
    p = _declare_params(b)

    c0 = b.const_i32(0)
    c16 = b.const_i32(16)
    c_wave = b.const_i32(wave)
    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)  # 0..W-1
    lane = b.mod(tid, c_wave)  # 0..wave-1
    a_row = a_map.coord(b, lane, 0)[0]  # lane % 16
    col = b.mod(lane, c16)  # lane % 16

    q_group = b.block_id_x()
    head = b.block_id_y()
    batch = b.block_id_z()

    qh, kvh = cfg.num_query_heads, cfg.kv_heads
    kv_head = head if kvh == qh else b.div(head, b.const_i32(qh // kvh))

    seqlen_q = p["seqlen_q"]
    seqlen_k = p["seqlen_k"]
    sq = p["stride_q_token"]
    sqh = p["stride_q_head"]
    sk = p["stride_k_token"]
    skh = p["stride_k_head"]
    sv = p["stride_v_token"]
    svh = p["stride_v_head"]
    so = p["stride_o_token"]
    soh = p["stride_o_head"]
    scale_log2 = p["scale_log2"]
    Q, K, V, O = p["Q"], p["K"], p["V"], p["O"]  # noqa: E741

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    # ---- CK Tile 3D (head, token, dim) views (see fmha_singlewave for the rationale). ----
    Q_view = make_global_view(
        Q, shape=(qh, 1, hs), dtype=dtype_ir, strides=(sqh, sq, 1)
    )
    K_view = make_global_view(
        K, shape=(kvh, 1, hs), dtype=dtype_ir, strides=(skh, sk, 1)
    )
    V_view = make_global_view(
        V, shape=(kvh, 1, hs), dtype=dtype_ir, strides=(svh, sv, 1)
    )
    O_view = make_global_view(
        O, shape=(qh, 1, hs), dtype=dtype_ir, strides=(soh, so, 1)
    )

    q_rows_per_cta = b.const_i32(cfg.q_rows_per_cta)
    cta_row0 = b.mul(q_group, q_rows_per_cta)  # within-batch first q row of CTA
    wave_row0 = b.add(cta_row0, b.mul(wave_id, c16))  # this wave's first q row
    batch_tok_q = b.mul(batch, seqlen_q)
    batch_tok_k = b.mul(batch, seqlen_k)

    # within-batch q position of this wave's tile (for mask) and global q token base
    q_pos_base = wave_row0
    q_token_base = b.add(wave_row0, batch_tok_q)

    # per-wave Q/O windows (loop-invariant across K-tiles).
    qwin = make_tile_window(Q_view, (1, 16, hs), origin=(head, q_token_base, c0))
    owin = make_tile_window(O_view, (1, 16, hs), origin=(head, q_token_base, c0))

    # ---- LDS views: shared K [kv][d], transposed V [d][kv], per-wave P [W][16][16] ----
    K_lds = make_lds_view(b, dtype=dtype_ir, shape=(_BLOCK_K, hs), name_hint="Ksh")
    V_lds_t = make_lds_view(b, dtype=dtype_ir, shape=(hs, _BLOCK_K), name_hint="VshT")
    P_lds = make_lds_view(b, dtype=dtype_ir, shape=(W, 16, 16), name_hint="Psh")

    # ---- iter-args: m/l (c_frag each) then acc (n_dk vecs) ----
    iter_args = []
    for r in range(c_frag):
        iter_args.append((f"m{r}", neg_inf))
    for r in range(c_frag):
        iter_args.append((f"l{r}", zero_f))
    for d in range(n_dk):
        iter_args.append((f"acc{d}", atom.zero_acc(b)))

    def unpack(state):
        idx = 0
        ms = list(state[idx : idx + c_frag])
        idx += c_frag
        ls = list(state[idx : idx + c_frag])
        idx += c_frag
        accs = [WmmaTensor(atom, "c", v, arch) for v in state[idx : idx + n_dk]]
        idx += n_dk
        return ms, ls, accs

    c_block_k = b.const_i32(_BLOCK_K)
    loop_stop = b.div(seqlen_k, c_block_k)

    # ---- Cooperative loaders. All block_size threads participate. ----
    n_threads = cfg.block_size
    elems = _BLOCK_K * hs  # tile element count (16 x head_size)
    n_chunks = elems // 8  # 8-wide vector chunks
    if n_chunks % n_threads != 0:
        chunks_per_thread = (n_chunks + n_threads - 1) // n_threads
    else:
        chunks_per_thread = n_chunks // n_threads

    def coop_load_k(k_tile_base):
        # K_lds[row][col] = K[k_tile_base+row, col]; contiguous vec8 copy.
        for i in range(chunks_per_thread):
            c = b.add(tid, b.const_i32(i * n_threads))
            base = b.mul(c, b.const_i32(8))  # flat elem index
            row = b.div(base, b.const_i32(hs))
            colc = b.mod(base, b.const_i32(hs))
            tok = b.add(b.add(batch_tok_k, k_tile_base), row)
            v8 = K_view.load_vec(b, [kv_head, tok, colc], n=8)
            K_lds.store_vec(b, [row, colc], v8, 8)

    def coop_load_v_t(k_tile_base):
        # V_lds_t[d][kv] = V[k_tile_base+kv, d]; load contiguous in d, scatter
        # transposed into LDS (the only strided write, paid once per CTA).
        for i in range(chunks_per_thread):
            c = b.add(tid, b.const_i32(i * n_threads))
            base = b.mul(c, b.const_i32(8))
            row = b.div(base, b.const_i32(hs))  # kv row
            colc = b.mod(base, b.const_i32(hs))  # d base
            tok = b.add(b.add(batch_tok_k, k_tile_base), row)
            v8 = V_view.load_vec(b, [kv_head, tok, colc], n=8)
            for u in range(8):
                V_lds_t.store_scalar(
                    b, [b.add(colc, b.const_i32(u)), row], b.vec_extract(v8, u)
                )

    def load_k_b_frag(d):
        # QK B-operand from LDS: K_lds[lane%16][d*16 .. +16] (contiguous in d).
        lo = K_lds.load_vec(b, [col, b.const_i32(d * 16)], n=8)
        hi = K_lds.load_vec(b, [col, b.const_i32(d * 16 + 8)], n=8)
        return b.vec_concat(lo, hi)

    def load_v_b_frag(d):
        # PV B-operand from LDS: V_lds_t[d*16 + lane%16][0 .. 16] (contiguous kv).
        d_col = b.add(b.const_i32(d * 16), col)
        lo = V_lds_t.load_vec(b, [d_col, b.const_i32(0)], n=8)
        hi = V_lds_t.load_vec(b, [d_col, b.const_i32(8)], n=8)
        return b.vec_concat(lo, hi)

    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms, ls, accs = unpack(state)
        k_tile_base = b.mul(kt, c_block_k)

        # ---- cooperative K/V staging (all waves) ----
        coop_load_k(k_tile_base)
        coop_load_v_t(k_tile_base)
        b.sync()

        new_ms = list(ms)
        new_ls = list(ls)
        new_accs = list(accs)
        ps = [None] * c_frag

        # ---- QK: A = this wave's Q rows (global), B = shared K (LDS) ----
        score = WmmaTensor.zero_acc(b, atom, arch=arch)
        for d in range(n_dk):
            q_tile = load_wmma_tile(
                b, qwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
            )
            k_tile = WmmaTensor(atom, "b", load_k_b_frag(d), arch)
            score = wmma_mma(b, q_tile, k_tile, score)

        # ---- online softmax ----
        alpha_vec = b.zero_vec_f32(c_frag)
        for r in range(c_frag):
            row_rel, col_k = score.coord(b, lane, r)
            s_r = b.fmul(score.slot(b, r), scale_log2)
            s_r = apply_attention_mask(
                b,
                s_r,
                mask_mode=cfg.mask_mode,
                k_idx=b.add(k_tile_base, col_k),
                query_pos=b.add(q_pos_base, row_rel),
                sliding_window=0,
            )
            row_max = wave_reduce_max(b, s_r, wave_size=wave, lanes_per_row=16)
            m_new = b.fmax(ms[r], row_max)
            alpha = b.exp2(b.fsub(ms[r], m_new))
            p_r = b.exp2(b.fsub(s_r, m_new))
            row_sum = wave_reduce_sum(b, p_r, wave_size=wave, lanes_per_row=16)
            new_ms[r] = m_new
            new_ls[r] = b.fadd(b.fmul(ls[r], alpha), row_sum)
            ps[r] = p_r
            alpha_vec = b.vec_insert(alpha_vec, alpha, r)
        for d in range(n_dk):
            new_accs[d] = new_accs[d].scale(b, alpha_vec)

        # ---- transpose P (acc layout -> PV A-operand) via this wave's LDS slab ----
        for r in range(c_frag):
            row_rel, col_k = c_map.coord(b, lane, r)
            P_lds.store_scalar(
                b, [wave_id, row_rel, col_k], b.cast_f32_to(ps[r], dtype_ir)
            )
        # P transpose is intra-wave (wave32 is lockstep on its own P_lds slab):
        # an LDS waitcnt suffices, no cross-wave s_barrier needed.
        b.s_waitcnt(lgkmcnt=0)
        lo = P_lds.load_vec(b, [wave_id, a_row, b.const_i32(0)], n=8)
        hi = P_lds.load_vec(b, [wave_id, a_row, b.const_i32(8)], n=8)
        p_a = b.vec_concat(lo, hi)

        # ---- PV: A = P (LDS), B = shared transposed V (LDS) ----
        p_tile = WmmaTensor(atom, "a", p_a, arch)
        for d in range(n_dk):
            v_tile = WmmaTensor(atom, "b", load_v_b_frag(d), arch)
            new_accs[d] = wmma_mma(b, p_tile, v_tile, new_accs[d])

        # barrier before next iteration overwrites the shared K/V tiles
        b.sync()

        yields = []
        for r in range(c_frag):
            yields.append(new_ms[r])
        for r in range(c_frag):
            yields.append(new_ls[r])
        yields.extend(a.value for a in new_accs)
        b.scf_yield(*yields)

    final = kloop.results
    ms_f, ls_f, accs_f = unpack(final)

    # ---- Epilogue (CK Tile store_wmma_acc + TileWindow) ----
    inv_l = []
    for r in range(c_frag):
        l_safe = ls_f[r]
        zmask = b.fcmp("oeq", l_safe, zero_f)
        inv_l.append(b.select(zmask, zero_f, b.rcp(l_safe)))

    def _rescale(bld, val, slot, row, colv, _inv=inv_l):
        return bld.fmul(val, _inv[slot])

    for d in range(n_dk):
        store_wmma_tile(
            b,
            owin,
            accs_f[d],
            lane,
            col_offset=d * 16,
            lead=[c0],
            align=2,
            transform=_rescale,
        )
    b.ret()
    return b.kernel
