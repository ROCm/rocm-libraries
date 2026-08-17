# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""BLOCK_N-widened WMMA FMHA-forward kernel for the gfx1151 campaign.

The single-wave campaign (``fmha_singlewave.py``) is stuck at ~10.5 TF with WMMA only
~1% of the issued instruction stream -- i.e. issue-bound, not FLOP-bound. The one
lever that directly attacks that ratio is **processing more keys per K-loop
iteration**: a wave still owns 16 query rows, but each loop step now consumes
``bn_tiles`` 16-key subtiles (``BLOCK_N = 16 * bn_tiles``). That amortizes the
loop control, the m/l bookkeeping, and the address math over ``bn_tiles`` as many
matmuls, and exposes matmul-level ILP across the unrolled subtiles.

Online softmax is taken over all ``BLOCK_N`` keys per step: each row's running
max/sum is combined across the ``bn_tiles`` subtiles (a register-level max/sum on
top of the per-subtile cross-lane reduce). Q frags are reloaded per subtile (Q is
L1-resident, so this costs no real traffic and avoids pinning ``n_dk`` frags in
VGPR). P is transposed through LDS per subtile; the PV V B-operand is the same
cache-resident column gather the baseline uses.

Same WMMA contract / ABI as ``fmha_singlewave`` and the production kernel, and built on
the same CK Tile helper layer: :func:`~rocke.helpers.make_global_view` 3D
``(head, token, dim)`` views + :func:`~rocke.helpers.make_tile_window` per
subtile, :func:`~rocke.helpers.load_wmma_tile` for QK,
:class:`~rocke.helpers.WmmaTensor` packed score/accumulator tiles with
:func:`~rocke.helpers.wmma_mma` for the matmul,
:func:`~rocke.helpers.make_lds_view` for the per-subtile P-transpose, and
:func:`~rocke.helpers.store_wmma_tile` for the O epilogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class BlockNCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    bn_tiles: int = 2  # 16-key subtiles consumed per K-loop iteration
    fuse_k: Optional[bool] = None  # None = auto (head_size >= 128)
    name: str = "wmma_fmha_blockn"

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        return 32

    @property
    def q_rows_per_cta(self) -> int:
        return 16

    @property
    def block_n(self) -> int:
        return 16 * self.bn_tiles

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            self.mask_mode,
            f"bn{self.bn_tiles}",
        )


def blockn_grid(cfg: BlockNCfg, *, seqlen_q: int, batch: int):
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


def build_wmma_fmha_blockn(cfg: BlockNCfg, arch: str = "gfx1151") -> KernelDef:
    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    a_map = atom.a_layout(arch)
    c_map = atom.c_layout(arch)
    a_frag = atom.a_per_lane  # 16
    c_frag = atom.c_per_lane  # 8
    n_dk = cfg.head_size // 16
    NK = cfg.bn_tiles
    dtype_ir = F16
    fuse_k = cfg.fuse_k if cfg.fuse_k is not None else (cfg.head_size >= 128)

    b = IRBuilder(cfg.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = wave
    p = _declare_params(b)

    c0 = b.const_i32(0)
    c16 = b.const_i32(16)
    lane = b.mod(b.thread_id_x(), b.const_i32(wave))
    a_row = a_map.coord(b, lane, 0)[0]  # lane % 16
    col = b.mod(lane, c16)

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
    hs = cfg.head_size
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
    group_row0 = b.mul(q_group, q_rows_per_cta)
    batch_tok_q = b.mul(batch, seqlen_q)
    batch_tok_k = b.mul(batch, seqlen_k)

    q_pos_base = group_row0
    q_token_base = b.add(group_row0, batch_tok_q)

    # single Q-tile: windows are loop-invariant across K-tiles.
    qwin = make_tile_window(Q_view, (1, 16, hs), origin=(head, q_token_base, c0))
    owin = make_tile_window(O_view, (1, 16, hs), origin=(head, q_token_base, c0))

    def sub_window(view, sub_base):
        return make_tile_window(
            view, (1, 16, hs), origin=(kv_head, b.add(batch_tok_k, sub_base), c0)
        )

    # P staging: one 16x16 slab per key subtile.
    P_lds = make_lds_view(b, dtype=dtype_ir, shape=(NK, 16, 16), name_hint="Pbn")

    # iter-args: m (c_frag) | l (c_frag) | acc (n_dk)
    iter_args = []
    for r in range(c_frag):
        iter_args.append((f"m{r}", neg_inf))
    for r in range(c_frag):
        iter_args.append((f"l{r}", zero_f))
    for d in range(n_dk):
        iter_args.append((f"acc{d}", WmmaTensor.zero_acc(b, atom, arch=arch).value))

    def unpack(state):
        idx = 0
        ms = list(state[idx : idx + c_frag])
        idx += c_frag
        ls = list(state[idx : idx + c_frag])
        idx += c_frag
        accs = [WmmaTensor(atom, "c", v, arch) for v in state[idx : idx + n_dk]]
        idx += n_dk
        return ms, ls, accs

    c_block_n = b.const_i32(cfg.block_n)
    loop_stop = b.div(seqlen_k, c_block_n)

    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms, ls, accs = unpack(state)
        n_base = b.mul(kt, c_block_n)  # first key of this BLOCK_N step

        new_ms = list(ms)
        new_ls = list(ls)
        new_accs = list(accs)

        # ---- QK for every subtile (Q reloaded per subtile: L1-resident) ----
        scores = []
        for s in range(NK):
            sub_base = b.add(n_base, b.const_i32(s * 16))
            kwin = sub_window(K_view, sub_base)
            k_frags = None
            if not fuse_k:
                k_frags = [
                    load_wmma_tile(
                        b, kwin, atom, lane, role="b", k_offset=d * 16, lead=[c0]
                    )
                    for d in range(n_dk)
                ]
            score = WmmaTensor.zero_acc(b, atom, arch=arch)
            for d in range(n_dk):
                q_tile = load_wmma_tile(
                    b, qwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
                )
                if fuse_k:
                    k_tile = load_wmma_tile(
                        b, kwin, atom, lane, role="b", k_offset=d * 16, lead=[c0]
                    )
                else:
                    k_tile = k_frags[d]
                score = wmma_mma(b, q_tile, k_tile, score)
            scores.append(score)

        # ---- online softmax over all BLOCK_N keys ----
        # per (row r): scaled+masked s for each subtile, combined max, then exp/sum.
        s_scaled = [[None] * c_frag for _ in range(NK)]
        alpha_vec = b.zero_vec_f32(c_frag)
        ps = [[None] * c_frag for _ in range(NK)]
        for r in range(c_frag):
            row_rel, col_k = scores[0].coord(b, lane, r)
            local_max = neg_inf
            for s in range(NK):
                sub_base = b.add(n_base, b.const_i32(s * 16))
                s_r = b.fmul(scores[s].slot(b, r), scale_log2)
                s_r = apply_attention_mask(
                    b,
                    s_r,
                    mask_mode=cfg.mask_mode,
                    k_idx=b.add(sub_base, col_k),
                    query_pos=b.add(q_pos_base, row_rel),
                    sliding_window=0,
                )
                s_scaled[s][r] = s_r
                rmax_s = wave_reduce_max(b, s_r, wave_size=wave, lanes_per_row=16)
                local_max = b.fmax(local_max, rmax_s)
            m_new = b.fmax(ms[r], local_max)
            alpha = b.exp2(b.fsub(ms[r], m_new))
            local_sum = zero_f
            for s in range(NK):
                p_r = b.exp2(b.fsub(s_scaled[s][r], m_new))
                rsum_s = wave_reduce_sum(b, p_r, wave_size=wave, lanes_per_row=16)
                local_sum = b.fadd(local_sum, rsum_s)
                ps[s][r] = p_r
            new_ms[r] = m_new
            new_ls[r] = b.fadd(b.fmul(ls[r], alpha), local_sum)
            alpha_vec = b.vec_insert(alpha_vec, alpha, r)
        for d in range(n_dk):
            new_accs[d] = new_accs[d].scale(b, alpha_vec)

        # ---- transpose P (acc layout -> PV A-operand) per subtile via LDS ----
        for s in range(NK):
            cs = b.const_i32(s)
            for r in range(c_frag):
                row_rel, col_k = c_map.coord(b, lane, r)
                P_lds.store_scalar(
                    b, [cs, row_rel, col_k], b.cast_f32_to(ps[s][r], dtype_ir)
                )
        b.sync()
        p_a = []
        for s in range(NK):
            cs = b.const_i32(s)
            lo = P_lds.load_vec(b, [cs, a_row, b.const_i32(0)], n=8)
            hi = P_lds.load_vec(b, [cs, a_row, b.const_i32(8)], n=8)
            p_a.append(b.vec_concat(lo, hi))
        p_tiles = [WmmaTensor(atom, "a", pa, arch) for pa in p_a]

        # ---- PV: acc[d] += sum_s  P_s @ V_s  (V gathered from global) ----
        for d in range(n_dk):
            d_col = b.add(b.const_i32(d * 16), col)
            for s in range(NK):
                sub_base = b.add(n_base, b.const_i32(s * 16))
                vwin = sub_window(V_view, sub_base)
                v_b = b.zero_vec(dtype_ir, a_frag)
                for j in range(a_frag):
                    v_elem = vwin.load_scalar(b, c0, b.const_i32(j), d_col)
                    v_b = b.vec_insert(v_b, v_elem, j)
                v_tile = WmmaTensor(atom, "b", v_b, arch)
                new_accs[d] = wmma_mma(b, p_tiles[s], v_tile, new_accs[d])
        b.sync()  # P_lds reused next iter

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
