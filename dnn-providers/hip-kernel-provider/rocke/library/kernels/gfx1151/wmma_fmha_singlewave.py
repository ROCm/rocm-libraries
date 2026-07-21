# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Heavily-parameterized WMMA FMHA-forward kernel for the gfx1151 case study.

This is the single-wave (block_size=32) winner of the optimization campaign in
this directory. It is built **entirely on the CK Tile helper layer**,
demonstrating that the helpers can drive RDNA/WMMA at a minimal instruction
budget (the kernel is issue-bound: WMMA is ~1% of the static instructions, so any
added address-arithmetic op costs TFLOP/s). The helper primitives it reuses:

  * :func:`~rocke.helpers.make_global_view` + :func:`~rocke.helpers.make_tile_window`
    for the Q/K/V/O addressing — 3D ``(head, token, dim)`` views whose head axis
    carries the per-head stride (which cannot fold into the token stride) and
    whose token axis folds the batch offset.
  * :class:`~rocke.helpers.WmmaAtom` (``wmma_f32_16x16x16_f16``) for the WMMA
    contract, with its hardware-verified ``a_layout``/``b_layout``/``c_layout``
    maps driving every lane decode.
  * :class:`~rocke.helpers.WmmaTensor` — a packed distributed tensor carrying
    one lane's fragment/accumulator as a single SSA vector — together with
    :func:`~rocke.helpers.load_wmma_tile` (one packed ``global_load_dwordx8``
    per ``<16 x half>`` fragment, no f32 cast) and :func:`~rocke.helpers.wmma_mma`
    (one ``b.mma``). The score/accumulator carry as ``WmmaTensor`` tiles, so the
    online-softmax rescale is one ``tile.scale`` (a single ``v_mul``) and the
    per-slot lane decode is ``tile.coord`` off the atom's verified layout map.
  * :func:`~rocke.helpers.make_lds_view` for the P-transpose (and the optional
    V-transpose) LDS staging.
  * :func:`~rocke.helpers.store_wmma_tile` for the O epilogue.
  * :mod:`rocke.helpers.attention` (``wave_reduce_max``/``wave_reduce_sum`` /
    ``apply_attention_mask``) for the online softmax.

Swept levers (unchanged from the campaign):

  * ``bm_tiles`` -- number of 16-row Q-tiles a single wave owns. K and V are
    independent of the query rows, so loading them ONCE per K-tile and feeding
    ``bm_tiles`` QK/PV matmuls amortizes the load/gather traffic. (BM amplification.)
  * ``p_mode`` -- ``"lds"`` round-trips P through LDS to transpose the score
    accumulator layout into the PV A-operand layout.
  * ``v_mode`` -- ``"gather"`` reads the PV V B-operand straight from global;
    ``"lds_t"`` stages V *transposed* in LDS so the column gather becomes a
    vectorized contiguous read.
  * ``prefetch_k`` -- hoist the next K-tile's global loads above the compute.

Build with ``compile_kernel(build_wmma_fmha_singlewave(cfg), arch="gfx1151")``.
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
_BLOCK_K = 16
# LDS row padding (in f16 elements) for v_mode="lds_c". The PV B-operand readback
# reads a *column* of the [k, d] V slab (fixed d, k=0..15), so consecutive k are
# `row_stride` elements apart. head_size (128/64) is a multiple of 4 -> a packed
# stride puts all 16 k on the same LDS bank (16-way conflict). Padding to make
# (head_size+pad)/2 odd spreads them across 16 distinct banks (conflict-free).
_V_LDS_PAD = 2


@dataclass(frozen=True)
class SingleWaveCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    bm_tiles: int = 1  # 16-row Q-tiles per wave
    p_mode: str = "lds"  # "lds" | "shuffle"
    # v_mode (Lever 1): PV B-operand (V) staging.
    #   "gather" -- 16 scalar strided global loads per d-column (128/iter, the
    #               dominant memory + address-SALU cost; ISA scan: 128 narrow
    #               b16 loads = 57% of all memory traffic).
    #   "lds_t"  -- stage V transposed [d,k] in LDS; contiguous readback but a
    #               *scalar scatter-store* (128 ds_store) -> regresses (no async).
    #   "lds_c"  -- stage V *contiguous* [k,d] in LDS: 16 WIDE global loads + 16
    #               WIDE contiguous LDS stores, then a padded (conflict-free)
    #               strided LDS readback. Trades the 128 narrow *global* gathers
    #               (+ their buffer-address SALU) for cheap on-chip LDS reads.
    v_mode: str = "gather"  # "gather" | "lds_t" | "lds_c"
    prefetch_k: bool = False
    # v_prefetch (fine-grained sync): software-pipeline the PV d-loop so the
    # next d-column's 16-wide V gather is *issued* before the current column's
    # WMMA. The next loads then fly during the WMMA and the backend can drain
    # only the current column's vmcnt (partial vmcnt) instead of a full
    # vmcnt(0) at every WMMA. gather-mode only (lds_t is already contiguous).
    v_prefetch: bool = False
    q_preload: bool = False  # hoist Q frags out of K-loop (no win: Q is L1-resident)
    # fuse_k: load each K-frag inside the QK matmul (1 live vs n_dk) -- cuts VGPR
    # spills. Net win ONLY when the kernel spills (head_size>=128); at head_size=64
    # there's no spill so the extra reloads just add latency. None = auto.
    fuse_k: Optional[bool] = None
    # waves_per_eu (Lever 1): explicit AMDGPU occupancy target, the equivalent of
    # CK's ``__launch_bounds__(NUM_THREADS, MIN_BLOCKS)``. ``None`` leaves the
    # backend heuristic (which pins this kernel at ~192 VGPR and *spills*);
    # setting a lower occupancy target (e.g. 2) lets the backend spend the top
    # of the RDNA3 wave32 VGPR file (toward 256) instead of spilling. Threaded
    # into ``b.kernel.attrs["waves_per_eu"]`` (lowered at lower_llvm.py:4415).
    waves_per_eu: Optional[int] = None
    # dpp_reduce (fine-grained sync): do the softmax cross-lane max/sum
    # butterflies via DPP (VALU v_max/v_add with row-xor) instead of
    # warp_shuffle_xor (which lowers to ds_swizzle/ds_bpermute on the LDS
    # engine). DPP keeps the reduction in the VALU pipe, eliminating the
    # ds_bpermute chain AND the per-stage s_waitcnt(lgkmcnt(0)) drains the ASM
    # scan found dominate the loop synchronization (~28/iter). Default off
    # until validated (DPP row-xor coverage across the 16-lane row on gfx11).
    dpp_reduce: bool = False
    # lazy_rescale (Lever 3, "lazy-O"): adaptive online-softmax rescale ported
    # from the gfx950 attention_dense reference. Keep the running max as a LAZY
    # max that only re-anchors when a tile's row-max exceeds it by > 8 (log2);
    # when every row is within 8 (wave_all vote) skip the O accumulator rescale
    # entirely (a wave-uniform 0/1-trip scf.for compiles the skip to a scalar
    # branch), cutting the per-tile VALU between the QK and PV WMMA clusters
    # (the measured issue-bound bottleneck). P is then bounded by exp2(8)=256
    # (safe for fp32 accum / fp16 P) rather than <=1 -- numerically APPROXIMATE
    # but within fp16 tolerance. Default off.
    lazy_rescale: bool = False
    # static_shape (coordinate strength-reduction): bake the tensor strides as
    # compile-time constants (standard packed BSHD: stride_head=D,
    # stride_token=H*D) so every address term is a *constant* multiply -- LLVM
    # strength-reduces it to shifts/LEA and hoists the loop-invariant base out of
    # the K-loop -- instead of a runtime `mul` on an i32 stride param (the ISA
    # scan's 348 SALU / address VALU). The runtime stride params stay in the
    # signature (launcher unchanged) but go unused (DCE'd). Fixes the layout.
    static_shape: bool = False
    # static_seqlen (>0): also bake seqlen_q == seqlen_k as a constant (must match
    # the launch args) so the K-loop trip count and the batch token offsets are
    # compile-time (const loop bound -> unrollable; const-mul batch offset).
    static_seqlen: int = 0
    # batch_softmax (VOPD-feeding VALU reduction): emit the per-row online-softmax
    # ops grouped BY OPERATION (all 8 rows' scale-mul, then all 8 fmax, then all 8
    # ls*alpha, then all 8 +row_sum) instead of row-serial dependent chains. The
    # rows are independent, so batching places same-class independent v_mul/v_add
    # /v_max adjacently -> LLVM's GCNCreateVOPD packer fuses them into v_dual_*
    # (2 VALU/issue), and the independent v_exp_f32 (TRANS pipe) overlap. Attacks
    # the issue-bound VALU wall without changing numerics. Default-path only
    # (skip vote / lazy_rescale keeps its dependent form).
    batch_softmax: bool = False
    name: str = "wmma_fmha_singlewave"

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        return 32

    @property
    def q_rows_per_cta(self) -> int:
        return 16 * self.bm_tiles

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            self.mask_mode,
            f"bm{self.bm_tiles}",
            f"p{self.p_mode}",
            f"v{self.v_mode}",
            "pf" if self.prefetch_k else "npf",
            f"vpe{self.waves_per_eu}" if self.waves_per_eu is not None else "vpedef",
            "lazyo" if self.lazy_rescale else "eager",
            "dpp" if self.dpp_reduce else "shfl",
            "vpf" if self.v_prefetch else "nvpf",
            ("stat" if self.static_shape else "dyn")
            + (f"S{self.static_seqlen}" if self.static_seqlen else ""),
            "bsm" if self.batch_softmax else "ssm",
        )


def singlewave_grid(cfg: SingleWaveCfg, *, seqlen_q: int, batch: int):
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


def build_wmma_fmha_singlewave(cfg: SingleWaveCfg, arch: str = "gfx1151") -> KernelDef:
    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    a_map = atom.a_layout(arch)
    c_map = atom.c_layout(arch)
    a_frag = atom.a_per_lane  # 16
    c_frag = atom.c_per_lane  # 8
    n_dk = cfg.head_size // 16
    BM = cfg.bm_tiles
    dtype_ir = F16
    # fuse_k auto: only a win when the PV accumulator spills (head_size>=128);
    # at D64 there's no spill so the extra K reloads just add latency.
    fuse_k = cfg.fuse_k if cfg.fuse_k is not None else (cfg.head_size >= 128)

    b = IRBuilder(cfg.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = wave
    # Lever 1: relax the occupancy target so the backend uses more of the wave32
    # VGPR file instead of spilling at the default heuristic ceiling.
    if cfg.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = cfg.waves_per_eu
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

    if cfg.static_seqlen:
        seqlen_q = b.const_i32(cfg.static_seqlen)
        seqlen_k = b.const_i32(cfg.static_seqlen)
    else:
        seqlen_q = p["seqlen_q"]
        seqlen_k = p["seqlen_k"]
    if cfg.static_shape:
        # Packed BSHD constants -> const-multiply address terms (strength-reduced
        # to shifts, loop-invariant base hoisted). Runtime params go unused.
        D = cfg.head_size
        sq, sqh = qh * D, D
        sk, skh = kvh * D, D
        sv, svh = kvh * D, D
        so, soh = qh * D, D
    else:
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
    # Lazy-O re-anchor threshold (log2 domain): skip the O rescale when every
    # row's (tile_max - running_max) <= this. exp2(8)=256 bounds P safely.
    lazy_thr = b.const_f32(8.0)

    # ---- CK Tile tensor views: 3D (head, token, dim). The head axis carries the
    # per-head stride (which cannot fold into the token stride); the token axis
    # stride is the per-token stride and the batch offset folds into the token
    # ORIGIN (batch and token share that stride). dim is contiguous (stride 1). ----
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
    group_row0 = b.mul(q_group, q_rows_per_cta)  # within-batch first q row
    batch_tok_q = b.mul(batch, seqlen_q)  # batch offset folded into token index
    batch_tok_k = b.mul(batch, seqlen_k)

    # Per-tile within-batch q position base (for mask) and global q-token base.
    def q_pos_base(t):
        return b.add(group_row0, b.const_i32(t * 16))

    def q_token_base(t):
        return b.add(b.add(group_row0, b.const_i32(t * 16)), batch_tok_q)

    def q_window(t):
        return make_tile_window(Q_view, (1, 16, hs), origin=(head, q_token_base(t), c0))

    def o_window(t):
        return make_tile_window(O_view, (1, 16, hs), origin=(head, q_token_base(t), c0))

    # ---- LDS staging (CK Tile LDS views) ----
    P_lds = None
    V_lds_t = None
    V_lds = None
    if cfg.p_mode == "lds":
        P_lds = make_lds_view(b, dtype=dtype_ir, shape=(BM, 16, 16), name_hint="Pwmma")
    if cfg.v_mode == "lds_t":
        # transposed: [d, k] so the B-operand column gather is contiguous in k.
        V_lds_t = make_lds_view(b, dtype=dtype_ir, shape=(hs, 16), name_hint="VwmmaT")
    if cfg.v_mode == "lds_c":
        # contiguous [k, d] with a padded row stride so the strided B-operand
        # readback (column of fixed d across k=0..15) hits 16 distinct banks.
        V_lds = make_lds_view(
            b,
            dtype=dtype_ir,
            shape=(16, hs),
            strides=(hs + _V_LDS_PAD, 1),
            name_hint="Vwmma",
        )

    # ---- iter-args: per-tile m/l (c_frag each) then per-tile acc (n_dk vecs) ----
    iter_args = []
    for t in range(BM):
        for r in range(c_frag):
            iter_args.append((f"m{t}_{r}", neg_inf))
        for r in range(c_frag):
            iter_args.append((f"l{t}_{r}", zero_f))
    for t in range(BM):
        for d in range(n_dk):
            iter_args.append((f"acc{t}_{d}", atom.zero_acc(b)))

    def unpack(state):
        idx = 0
        ms = []
        ls = []
        for _ in range(BM):
            ms.append(list(state[idx : idx + c_frag]))
            idx += c_frag
            ls.append(list(state[idx : idx + c_frag]))
            idx += c_frag
        accs = []
        for _ in range(BM):
            accs.append(
                [WmmaTensor(atom, "c", v, arch) for v in state[idx : idx + n_dk]]
            )
            idx += n_dk
        return ms, ls, accs

    c_block_k = b.const_i32(_BLOCK_K)
    loop_stop = b.div(seqlen_k, c_block_k)
    if cfg.mask_mode == "causal":
        # Causal early-exit: this CTA owns q rows [group_row0, group_row0+BM*16-1].
        # K-tile kt covers keys [kt*16, kt*16+15], needed iff kt*16 <= max query
        # pos, i.e. kt < (group_row0/16)+BM. Skipping the fully-masked upper
        # triangle roughly halves the causal K-loop.
        causal_stop = b.add(b.div(group_row0, c_block_k), b.const_i32(BM))
        loop_stop = b.select(b.cmp_lt(causal_stop, loop_stop), causal_stop, loop_stop)

    # ---- Q is loop-invariant across K-tiles: optionally preload all frags once.
    # Trades VGPR (BM*n_dk live frags) for fewer dynamic loads -- but Q is already
    # L1-resident, so the register pressure can backfire via spills. ----
    q_frags = [[None] * n_dk for _ in range(BM)]
    if cfg.q_preload:
        for t in range(BM):
            qwin = q_window(t)
            for d in range(n_dk):
                q_frags[t][d] = load_wmma_tile(
                    b, qwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
                )

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

    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms, ls, accs = unpack(state)
        k_tile_base = b.mul(kt, c_block_k)
        kwin = k_window(k_tile_base)

        # ---- K frags (shared across all BM Q-tiles). With fuse_k (BM==1) the
        # frags are loaded inside the QK matmul so only one is live at a time. ----
        k_frags = None
        if not fuse_k:
            k_frags = [
                load_wmma_tile(
                    b, kwin, atom, lane, role="b", k_offset=d * 16, lead=[c0]
                )
                for d in range(n_dk)
            ]

        new_ms = [list(ms[t]) for t in range(BM)]
        new_ls = [list(ls[t]) for t in range(BM)]
        new_accs = [list(accs[t]) for t in range(BM)]
        ps = [[None] * c_frag for _ in range(BM)]

        # ---- QK + online softmax for each Q-tile ----
        for t in range(BM):
            qwin = None if cfg.q_preload else q_window(t)
            score = WmmaTensor.zero_acc(b, atom, arch=arch)
            for d in range(n_dk):
                if cfg.q_preload:
                    q_tile = q_frags[t][d]
                else:
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
            # Pass 1: score (scaled + masked) and per-row max. Stored so the
            # lazy-max vote can be computed before m_new (which depends on it).
            s_rows = []
            for r in range(c_frag):
                row_rel, col_k = score.coord(b, lane, r)
                s_r = b.fmul(score.slot(b, r), scale_log2)
                s_r = apply_attention_mask(
                    b,
                    s_r,
                    mask_mode=cfg.mask_mode,
                    k_idx=b.add(k_tile_base, col_k),
                    query_pos=b.add(q_pos_base(t), row_rel),
                    sliding_window=0,
                )
                s_rows.append(s_r)
            # batch_softmax: the 8 scale-muls above are now adjacent + independent
            # (VOPD-packable); reduce afterwards. Row-serial otherwise.
            row_maxs = [
                wave_reduce_max(
                    b,
                    s_rows[r],
                    wave_size=wave,
                    lanes_per_row=16,
                    use_dpp=cfg.dpp_reduce,
                )
                for r in range(c_frag)
            ]

            # Lazy-O vote: skip the O rescale when EVERY row is within the
            # re-anchor threshold of its running max (wave_all across the 32
            # lanes = all 16 output rows). When skipping we keep m_new = m_i so
            # alpha == 1 and the rescale is a mathematical no-op we elide.
            if cfg.lazy_rescale:
                below_i32 = b.const_i32(1)
                for r in range(c_frag):
                    m_diff = b.fsub(row_maxs[r], ms[t][r])
                    below_i32 = b.select(
                        b.fcmp("ole", m_diff, lazy_thr), below_i32, b.const_i32(0)
                    )
                skip = b.cmp_ne(b.wave_all(below_i32), b.const_i32(0))
            else:
                skip = None

            alpha_vec = atom.zero_acc(b)
            if cfg.batch_softmax and skip is None:
                # VOPD-feeding: emit each op across ALL rows adjacently so the
                # independent same-class v_max / v_mul / v_add pack into v_dual_*
                # and the independent v_exp_f32 (TRANS) overlap.
                m_news = [b.fmax(ms[t][r], row_maxs[r]) for r in range(c_frag)]
                alphas = [b.exp2(b.fsub(ms[t][r], m_news[r])) for r in range(c_frag)]
                p_batch = [b.exp2(b.fsub(s_rows[r], m_news[r])) for r in range(c_frag)]
                row_sums = [
                    wave_reduce_sum(
                        b,
                        p_batch[r],
                        wave_size=wave,
                        lanes_per_row=16,
                        use_dpp=cfg.dpp_reduce,
                    )
                    for r in range(c_frag)
                ]
                la = [b.fmul(ls[t][r], alphas[r]) for r in range(c_frag)]
                for r in range(c_frag):
                    new_ms[t][r] = m_news[r]
                    new_ls[t][r] = b.fadd(la[r], row_sums[r])
                    ps[t][r] = p_batch[r]
                    alpha_vec = b.vec_insert(alpha_vec, alphas[r], r)
            else:
                for r in range(c_frag):
                    if skip is not None:
                        m_new = b.select(skip, ms[t][r], b.fmax(ms[t][r], row_maxs[r]))
                    else:
                        m_new = b.fmax(ms[t][r], row_maxs[r])
                    alpha = b.exp2(b.fsub(ms[t][r], m_new))
                    p_r = b.exp2(b.fsub(s_rows[r], m_new))
                    row_sum = wave_reduce_sum(
                        b,
                        p_r,
                        wave_size=wave,
                        lanes_per_row=16,
                        use_dpp=cfg.dpp_reduce,
                    )
                    new_ms[t][r] = m_new
                    new_ls[t][r] = b.fadd(b.fmul(ls[t][r], alpha), row_sum)
                    ps[t][r] = p_r
                    alpha_vec = b.vec_insert(alpha_vec, alpha, r)

            # Online-softmax O rescale: acc[d] *= alpha (one vmul/d). With
            # lazy-O, wrap in a wave-uniform 0/1-trip loop so the rescale (and
            # its n_dk MFMA-shadow VALU) is elided on the skip vote.
            if skip is not None:
                trips = b.select(skip, b.const_i32(0), b.const_i32(1))
                rs_args = [(f"lo{t}_{d}", new_accs[t][d].value) for d in range(n_dk)]
                rs = b.scf_for_iter(
                    b.const_i32(0),
                    trips,
                    b.const_i32(1),
                    rs_args,
                    iv_name=f"lrs{t}",
                )
                with rs as (_iv, rc):
                    outs = [
                        WmmaTensor(atom, "c", rc[d], arch).scale(b, alpha_vec).value
                        for d in range(n_dk)
                    ]
                    b.scf_yield(*outs)
                for d in range(n_dk):
                    new_accs[t][d] = WmmaTensor(atom, "c", rs.results[d], arch)
            else:
                for d in range(n_dk):
                    new_accs[t][d] = new_accs[t][d].scale(b, alpha_vec)

        # ---- transpose P (acc layout -> PV A-operand layout) ----
        p_a = _transpose_p(
            b, cfg, ps, lane, a_row, c_map, a_frag, c_frag, dtype_ir, P_lds
        )
        p_tiles = [WmmaTensor(atom, "a", pa, arch) for pa in p_a]

        # ---- V staging once, shared across tiles ----
        if cfg.v_mode == "lds_t":
            _stage_v_transposed(
                b,
                cfg,
                V_view,
                V_lds_t,
                k_tile_base,
                kv_head,
                a_row,
                batch_tok_k,
                dtype_ir,
            )
            b.sync()
        elif cfg.v_mode == "lds_c":
            _stage_v_contig(
                b, cfg, V_view, V_lds, k_tile_base, kv_head, a_row, batch_tok_k
            )
            # single wave32: intra-wave LDS write->read ordered by an lgkmcnt
            # drain (no full CTA barrier), mirroring the P-transpose.
            b.s_waitcnt(lgkmcnt=0)

        # ---- PV: load V B-operand once per d, reuse across BM tiles ----
        vwin = v_window(k_tile_base)

        def _v_col(dd):
            d_col = b.add(b.const_i32(dd * 16), col)
            return _load_v_b(
                b, cfg, vwin, V_lds_t, dd, d_col, col, a_frag, dtype_ir, V_lds=V_lds
            )

        if cfg.v_mode == "lds_c":
            # LDS transpose readback: hoist ALL columns' reads before any WMMA
            # so the 128 narrow ds_reads pipeline into a single lgkmcnt drain
            # instead of serializing one WMMA-blocking drain per column.
            v_bs = [_v_col(d) for d in range(n_dk)]
            for d in range(n_dk):
                v_tile = WmmaTensor(atom, "b", v_bs[d], arch)
                for t in range(BM):
                    new_accs[t][d] = wmma_mma(b, p_tiles[t], v_tile, new_accs[t][d])
        elif cfg.v_prefetch and cfg.v_mode == "gather":
            # Software-pipeline the gather: issue d+1's 16 loads before d's WMMA
            # so they overlap the matmul; the backend then only needs a partial
            # vmcnt to retire d's loads (not a full drain) at each column.
            v_cur = _v_col(0)
            for d in range(n_dk):
                v_nxt = _v_col(d + 1) if d + 1 < n_dk else None
                v_tile = WmmaTensor(atom, "b", v_cur, arch)
                for t in range(BM):
                    new_accs[t][d] = wmma_mma(b, p_tiles[t], v_tile, new_accs[t][d])
                v_cur = v_nxt
        else:
            for d in range(n_dk):
                v_b = _v_col(d)
                v_tile = WmmaTensor(atom, "b", v_b, arch)
                for t in range(BM):
                    new_accs[t][d] = wmma_mma(b, p_tiles[t], v_tile, new_accs[t][d])

        yields = []
        for t in range(BM):
            for r in range(c_frag):
                yields.append(new_ms[t][r])
            for r in range(c_frag):
                yields.append(new_ls[t][r])
        for t in range(BM):
            yields.extend(a.value for a in new_accs[t])
        b.scf_yield(*yields)

    final = kloop.results
    ms_f, ls_f, accs_f = unpack(final)

    # ---- Epilogue per tile (CK Tile store_wmma_tile + TileWindow) ----
    for t in range(BM):
        owin = o_window(t)
        # inv_l depends only on (t,r); compute once instead of per-d (n_dk reloads).
        inv_l = []
        for r in range(c_frag):
            l_safe = ls_f[t][r]
            zmask = b.fcmp("oeq", l_safe, zero_f)
            inv_l.append(b.select(zmask, zero_f, b.rcp(l_safe)))

        def _rescale(bld, val, slot, row, colv, _inv=inv_l):
            return bld.fmul(val, _inv[slot])

        for d in range(n_dk):
            store_wmma_tile(
                b,
                owin,
                accs_f[t][d],
                lane,
                col_offset=d * 16,
                lead=[c0],
                align=2,
                transform=_rescale,
            )
    b.ret()
    return b.kernel


def _transpose_p(b, cfg, ps, lane, a_row, c_map, a_frag, c_frag, dtype_ir, P_lds):
    """Return a list of BM PV A-operand fragments (one per Q-tile).

    Uses the CK Tile :func:`make_lds_view` ``P_lds`` view: scatter each score
    slot to its ``(t, row, col)`` LDS cell, barrier, then read back row ``a_row``
    as the contiguous ``<16 x half>`` A fragment (two ``ds_read_b128`` halves +
    a concat -- an instruction cut over 16 scalar ds_loads on this issue-bound
    kernel)."""
    BM = cfg.bm_tiles
    if cfg.p_mode == "lds":
        for t in range(BM):
            ct = b.const_i32(t)
            for r in range(c_frag):
                row_rel, col_k = c_map.coord(b, lane, r)
                P_lds.store_scalar(
                    b, [ct, row_rel, col_k], b.cast_f32_to(ps[t][r], dtype_ir)
                )
        # This kernel is a single wave32 (block_size == wave), so the P-transpose
        # scatter-store -> row-read hazard is intra-wave (lockstep): an LDS
        # write-drain (s_waitcnt lgkmcnt=0) orders it correctly, no full CTA
        # s_barrier needed. Removes BM*n_ktiles workgroup barriers from the hot
        # loop vs the prior b.sync(). (Mirrors fmha_multiwave's per-wave P slab.)
        b.s_waitcnt(lgkmcnt=0)
        out = []
        for t in range(BM):
            ct = b.const_i32(t)
            # WMMA a-operand slot j -> k=j, so the 16-wide A fragment is row
            # a_row's contiguous columns 0..15 of P_lds. f16 LDS loads cap at
            # <8 x f16> (ds_read_b128), so read the two halves and concat.
            lo = P_lds.load_vec(b, [ct, a_row, b.const_i32(0)], n=8)
            hi = P_lds.load_vec(b, [ct, a_row, b.const_i32(8)], n=8)
            out.append(b.vec_concat(lo, hi))
        return out
    raise ValueError(f"unknown p_mode {cfg.p_mode!r}")


def _stage_v_transposed(
    b, cfg, V_view, V_lds_t, k_tile_base, kv_head, a_row, batch_tok_k, dtype_ir
):
    """Stage the K-tile's V rows into LDS *transposed* (V_lds_t[d, k]) so the PV
    B-operand column gather becomes a contiguous read. Each lane owns k=a_row and
    scatters its head_size d-slice down the d axis."""
    tok = b.add(b.add(batch_tok_k, k_tile_base), a_row)
    hs = cfg.head_size
    for e in range(hs // 8):
        v_g = V_view.load_vec(b, [kv_head, tok, b.const_i32(e * 8)], n=8)
        for u in range(8):
            V_lds_t.store_scalar(
                b, [b.const_i32(e * 8 + u), a_row], b.vec_extract(v_g, u)
            )


def _stage_v_contig(b, cfg, V_view, V_lds, k_tile_base, kv_head, a_row, batch_tok_k):
    """Stage the K-tile's V rows into LDS in *contiguous* [k, d] layout.

    Each lane owns k=``a_row`` and copies its full ``head_size`` d-slice with
    WIDE loads + WIDE contiguous stores (``hs/8`` ``b128`` each). Unlike
    :func:`_stage_v_transposed` there is **no scalar scatter** -- the transpose
    is deferred to the (cheap, padded, conflict-free) strided LDS *readback* in
    :func:`_load_v_b`. This trades the 128 narrow strided *global* gathers of the
    "gather" path for 16 wide global loads + on-chip LDS reads."""
    tok = b.add(b.add(batch_tok_k, k_tile_base), a_row)
    hs = cfg.head_size
    for e in range(hs // 8):
        off = b.const_i32(e * 8)
        v_g = V_view.load_vec(b, [kv_head, tok, off], n=8)
        V_lds.store_vec(b, [a_row, off], v_g, n=8)


def _load_v_b(b, cfg, vwin, V_lds_t, d, d_col, col, a_frag, dtype_ir, V_lds=None):
    """PV B-operand for this lane's d-column: V[k, d_col] for k=0..15."""
    if cfg.v_mode == "lds_t":
        # contiguous in k: V_lds_t[d_col, 0..15] -- smem vec width caps at 8 for
        # f16, so two 8-wide contiguous reads concatenated into the 16-frag.
        lo = V_lds_t.load_vec(b, [d_col, b.const_i32(0)], n=8)
        hi = V_lds_t.load_vec(b, [d_col, b.const_i32(8)], n=8)
        return b.vec_concat(lo, hi)
    if cfg.v_mode == "lds_c":
        # strided readback: column d_col of the [k, d] slab, k=0..15. Padded row
        # stride makes these 16 ds_read_b16 hit distinct banks (no conflict).
        v_b = b.zero_vec(dtype_ir, a_frag)
        for k in range(a_frag):
            e = V_lds.load_scalar(b, [b.const_i32(k), d_col])
            v_b = b.vec_insert(v_b, e, k)
        return v_b
    # gather from global through the V TileWindow (origin already at the K-tile).
    v_b = b.zero_vec(dtype_ir, a_frag)
    for j in range(a_frag):
        v_elem = vwin.load_scalar(b, b.const_i32(0), b.const_i32(j), d_col)
        v_b = b.vec_insert(v_b, v_elem, j)
    return v_b
