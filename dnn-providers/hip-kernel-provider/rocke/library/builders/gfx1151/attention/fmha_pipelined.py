# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Software-pipelined WMMA FMHA-forward kernel for gfx1151.

Ported technique from ck_tile's ``block_fmha_*_pipeline_qr_ks_vs`` family: the
single-wave kernel (``fmha_singlewave.py``) is issue-bound -- WMMA is ~1% of the
issued instructions and the matrix unit stalls behind the softmax VALU because,
within one K-tile, ``QK -> softmax -> PV`` is a hard dependency chain.

The fix is **software pipelining across the K-loop**. ``QK`` of tile ``i+1`` does
NOT depend on the softmax of tile ``i``, so we hoist it: each loop iteration
runs the softmax + PV of the *current* tile while computing the *next* tile's
QK, carrying the next score through the ``scf.for`` iter-args. The scheduler can
then overlap the QK matmuls (matrix unit) with the exp2/reduce VALU (the
production V3 schedule does exactly this), raising matrix-unit utilization.

Optional ``sched_group_barrier`` hints (``sched=True``) deterministically
interleave the WMMA and VALU clusters the way ck_tile's HotLoopScheduler does.

Single wave32 / 16 query rows per CTA, same WMMA contract / ABI as
``fmha_singlewave`` and the production kernel. ``fuse_k`` auto-resolves like
``fmha_singlewave`` (on when ``head_size>=128``).

Like ``fmha_singlewave`` this is built **on the CK Tile helper layer**: the Q/K/V/O
addressing uses :func:`~rocke.helpers.make_global_view` 3D ``(head, token, dim)``
views + :func:`~rocke.helpers.make_tile_window`; the score/accumulator carry as
packed :class:`~rocke.helpers.WmmaTensor` tiles, QK fragments come from
:func:`~rocke.helpers.load_wmma_tile` and the matmul is
:func:`~rocke.helpers.wmma_mma` over :class:`~rocke.helpers.WmmaAtom`; the
online-softmax rescale is one ``tile.scale`` and the per-slot lane decode is
``tile.coord`` off the atom's verified layout map; the lds P-transpose stages
through a :func:`~rocke.helpers.make_lds_view`; and the O epilogue uses
:func:`~rocke.helpers.store_wmma_tile`. The ``p_xpose="shuffle"`` path is the one
exception -- a ds_bpermute register transpose ported from ck_tile's
``PermuteWarpGemmCToA``; it is a **documented dead-end** (wrong numerics, see the
README "Recombining old levers" section) kept as raw IR for the case-study
record, since the register permute has no tile-addressing helper analog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rocke.core.ir import F16, F32, I16, I32, IRBuilder, KernelDef, PtrType, VectorType
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

# AMDGPU sched_group_barrier class masks
_SCHED_MFMA = 0x008  # matrix (WMMA) op
_SCHED_VALU = 0x002  # vector ALU
_SCHED_DS_READ = 0x100
_SCHED_DS_WRITE = 0x200
_SCHED_VMEM_READ = 0x020


@dataclass(frozen=True)
class PipelinedCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    fuse_k: Optional[bool] = None  # None = auto (head_size >= 128)
    sched: bool = False  # emit sched_group_barrier interleave hints
    p_xpose: str = "lds"  # "lds" | "shuffle" (ds_bpermute gfx11 CToA)
    name: str = "wmma_fmha_pipelined"

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        return 32

    @property
    def q_rows_per_cta(self) -> int:
        return 16

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            self.mask_mode,
            "sch" if self.sched else "nsch",
            self.p_xpose,
        )


def pipelined_grid(cfg: PipelinedCfg, *, seqlen_q: int, batch: int):
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


def build_wmma_fmha_pipelined(cfg: PipelinedCfg, arch: str = "gfx1151") -> KernelDef:
    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    a_map = atom.a_layout(arch)
    c_map = atom.c_layout(arch)
    a_frag = atom.a_per_lane  # 16
    c_frag = atom.c_per_lane  # 8
    n_dk = cfg.head_size // 16
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

    # ---- CK Tile 3D (head, token, dim) views: head axis carries the per-head
    # stride (cannot fold into the token stride), token axis folds the batch
    # offset (batch and token share that stride), dim is contiguous. ----
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

    # single Q-tile: build the window once (loop-invariant across K-tiles).
    qwin = make_tile_window(Q_view, (1, 16, hs), origin=(head, q_token_base, c0))
    owin = make_tile_window(O_view, (1, 16, hs), origin=(head, q_token_base, c0))

    P_lds = make_lds_view(b, dtype=dtype_ir, shape=(16, 16), name_hint="Psp")

    c_block_k = b.const_i32(_BLOCK_K)
    loop_stop = b.div(seqlen_k, c_block_k)
    if cfg.mask_mode == "causal":
        # Causal early-exit: this CTA owns q rows [group_row0, group_row0+15].
        # K-tile kt covers keys [kt*16, kt*16+15], needed iff kt*16 <= max q pos,
        # i.e. kt < (group_row0/16)+1. Skipping the fully-masked upper-triangle
        # tiles roughly halves the causal K-loop. loop_stop>=1 (q_group>=0) so the
        # prologue's tile-0 QK is always valid.
        causal_stop = b.add(b.div(group_row0, c_block_k), b.const_i32(1))
        loop_stop = b.select(b.cmp_lt(causal_stop, loop_stop), causal_stop, loop_stop)

    def k_window(k_tile_base):
        return make_tile_window(
            K_view,
            (1, 16, hs),
            origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0),
        )

    def v_window(k_tile_base):
        return make_tile_window(
            V_view,
            (1, 16, hs),
            origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0),
        )

    def compute_qk(k_tile_base):
        """QK^T for one K-tile -> score WmmaTensor (role "c", <8 x f32> C-layout)."""
        kwin = k_window(k_tile_base)
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
        return score

    # ---- gfx11 register P-transpose (ck_tile PermuteWarpGemmCToA) ----
    # Remap the c_frag (8 f32, C-distribution) softmax probabilities into the
    # PV A-operand (16 f16) WITHOUT an LDS round-trip + s_barrier. Emulates
    # permlanex16 (lane<->lane^16 swap) with ds_bpermute, and v_perm_b32 byte
    # interleave with shift/and/or. 8 input f16 -> 4 u32 -> 8 u32 (16 f16).
    c16c = b.const_i32(16)
    mask16 = b.const_i32(0xFFFF)
    maskhi = b.shl(mask16, c16c)  # 0xFFFF0000
    lane_lt16 = b.cmp_lt(lane, c16c)
    bperm_addr = b.shl(b.xor(lane, c16c), b.const_i32(2))  # (lane^16)<<2 byte addr

    def p_transpose_shuffle(ps):
        outs = []
        for i in range(c_frag // 2):
            lo = b.zext(b.bitcast(b.cast_f32_to(ps[2 * i], dtype_ir), I16), I32)
            hi = b.zext(b.bitcast(b.cast_f32_to(ps[2 * i + 1], dtype_ir), I16), I32)
            v = b.lor(lo, b.shl(hi, c16c))  # {f16 slot 2i | f16 slot 2i+1}
            w = b.ds_bpermute(bperm_addr, v)  # value held by lane^16
            A = b.select(lane_lt16, v, w)
            B = b.select(lane_lt16, w, v)
            out0 = b.lor(b.land(A, mask16), b.shl(b.land(B, mask16), c16c))
            out1 = b.lor(b.lshr(A, c16c), b.land(B, maskhi))
            outs.append(out0)
            outs.append(out1)
        packed = b.vec_pack(outs, I32)  # <a_frag/2 x i32>
        return b.vec_bitcast(packed, VectorType(dtype_ir, a_frag))  # <16 x f16>

    # ---- prologue: QK for tile 0 (carried into the loop) ----
    score0 = compute_qk(b.const_i32(0))

    # iter-args: m | l | acc | score_cur (the carried, pre-computed QK).
    # acc and the carried score thread as raw packed .value vectors and are
    # re-wrapped into WmmaTensor (role "c") on unpack.
    iter_args = []
    for r in range(c_frag):
        iter_args.append((f"m{r}", neg_inf))
    for r in range(c_frag):
        iter_args.append((f"l{r}", zero_f))
    for d in range(n_dk):
        iter_args.append((f"acc{d}", WmmaTensor.zero_acc(b, atom, arch=arch).value))
    iter_args.append(("sc", score0.value))

    def unpack(state):
        idx = 0
        ms = list(state[idx : idx + c_frag])
        idx += c_frag
        ls = list(state[idx : idx + c_frag])
        idx += c_frag
        accs = [WmmaTensor(atom, "c", v, arch) for v in state[idx : idx + n_dk]]
        idx += n_dk
        score = WmmaTensor(atom, "c", state[idx], arch)
        idx += 1
        return ms, ls, accs, score

    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms, ls, accs, score = unpack(state)
        k_tile_base = b.mul(kt, c_block_k)

        new_ms = list(ms)
        new_ls = list(ls)
        new_accs = list(accs)
        ps = [None] * c_frag

        # ---- online softmax on the carried current-tile score ----
        # alpha_vec is a throwaway scratch vector (only vec_insert + .scale),
        # NOT an accumulator tile, so it stays a raw zero_acc vector.
        alpha_vec = atom.zero_acc(b)
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

        # ---- PIPELINE: compute NEXT tile's QK now (independent of softmax) ----
        # The matrix-unit work below overlaps the VALU above / the PV below.
        # Clamp the next base to a valid tile on the final iteration (result
        # is yielded but never consumed after the loop).
        kt_next = b.add(kt, b.const_i32(1))
        is_last = b.cmp_ge(kt_next, loop_stop)
        kt_next_safe = b.select(is_last, b.const_i32(0), kt_next)
        score_next = compute_qk(b.mul(kt_next_safe, c_block_k))

        if cfg.sched:
            # nudge the scheduler to interleave the next-QK WMMAs with the
            # transpose/PV that follows (ck_tile HotLoopScheduler style).
            b.sched_group_barrier(_SCHED_MFMA, n_dk)
            b.sched_group_barrier(_SCHED_VALU, 8)

        # ---- transpose P (C-dist -> PV A-operand) ----
        if cfg.p_xpose == "shuffle":
            p_a = p_transpose_shuffle(ps)
        else:
            # scatter each score slot to its (row, col) LDS cell, barrier, then
            # read back row a_row as the contiguous <16 x half> A fragment (two
            # ds_read_b128 halves + concat -- WMMA a-slot j -> k=j).
            for r in range(c_frag):
                row_rel, col_k = c_map.coord(b, lane, r)
                P_lds.store_scalar(b, [row_rel, col_k], b.cast_f32_to(ps[r], dtype_ir))
            b.sync()
            lo = P_lds.load_vec(b, [a_row, b.const_i32(0)], n=8)
            hi = P_lds.load_vec(b, [a_row, b.const_i32(8)], n=8)
            p_a = b.vec_concat(lo, hi)

        # ---- PV: V B-operand gathered from global, accumulate ----
        # p_a / v_b are bare SSA vectors (the P-transpose + V-gather have no tile
        # helper analog), so wrap them inline as WmmaTensor for wmma_mma.
        p_tile = WmmaTensor(atom, "a", p_a, arch)
        vwin = v_window(k_tile_base)
        for d in range(n_dk):
            d_col = b.add(b.const_i32(d * 16), col)
            v_b = b.zero_vec(dtype_ir, a_frag)
            for j in range(a_frag):
                v_elem = vwin.load_scalar(b, c0, b.const_i32(j), d_col)
                v_b = b.vec_insert(v_b, v_elem, j)
            v_tile = WmmaTensor(atom, "b", v_b, arch)
            new_accs[d] = wmma_mma(b, p_tile, v_tile, new_accs[d])
        if cfg.p_xpose != "shuffle":
            b.sync()  # P_lds reused next iter

        yields = []
        for r in range(c_frag):
            yields.append(new_ms[r])
        for r in range(c_frag):
            yields.append(new_ls[r])
        yields.extend(a.value for a in new_accs)
        yields.append(score_next.value)
        b.scf_yield(*yields)

    final = kloop.results
    ms_f, ls_f, accs_f, _ = unpack(final)

    # ---- Epilogue (CK Tile store_wmma_acc + TileWindow) ----
    # inv_l depends only on r; compute once instead of per-d (n_dk reloads).
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
