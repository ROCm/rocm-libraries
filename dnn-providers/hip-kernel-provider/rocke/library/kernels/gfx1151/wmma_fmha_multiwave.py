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
# LDS bank-conflict padding (f16 elements) for the shared K / transposed-V tiles.
# Must keep (leading_dim * 2 bytes) a multiple of 16 so ds_read_b128 stays aligned.
_LDS_PAD = 8


@dataclass(frozen=True)
class MultiWaveCfg:
    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    n_waves: int = 4  # wave32s per CTA; each owns a 16-row Q-tile
    # K/V source for the WMMA B-operands:
    #   "lds"    -- cooperatively stage K/V into shared LDS once per CTA (the CK
    #               qr_vr design; costs 2 cross-wave s_barriers per K-tile).
    #   "gather" -- each wave reads its K/V B-operands straight from global (the
    #               single-wave winner's cache-resident gather); NO cross-wave
    #               barrier, so the only sync is the intra-wave P-transpose
    #               waitcnt. On this large-cache APU K/V for one (head,batch) is
    #               L1/L2-resident, so the "redundant" per-wave gather is nearly
    #               free and skips the barrier overhead that dominates "lds".
    # MEASURED (B4 H8 gfx1151): "gather" is ~2x "lds" on every shape (e.g. D128
    # S512 w4: 8.8 vs 4.7 TF) and matches/beats the single-wave winner, because
    # the cooperative LDS staging's 2 s_barriers/K-tile + V-transpose scatter are
    # pure overhead against a cache that already feeds the gather for free. So
    # "gather" is the default; "lds" is kept for the reproducible A/B.
    kv_source: str = "gather"  # "gather" | "lds"
    # waves_per_eu (Lever 1): explicit AMDGPU occupancy target for the 4-warp
    # block, the equivalent of CK's ``__launch_bounds__(128, 2)``. ``None`` leaves
    # the backend heuristic (which spills the moment the per-lane accumulator
    # grows); setting a lower occupancy target lets the distributed 64-VGPR o_acc
    # + fragments fit ~256 VGPR spill-free -- the register headroom Lever 2 needs.
    # Threaded into ``b.kernel.attrs["waves_per_eu"]`` (lowered at
    # lower_llvm.py:4415).
    waves_per_eu: Optional[int] = None
    # sched_mode: inter-wave scheduling of the WMMA clusters.
    #   "none"     -- no priority hints; the arbiter round-robins the resident
    #                 waves (they tend to run in-phase -> WMMA unit idles during
    #                 the shared softmax-VALU phase).
    #   "pingpong" -- bracket each wave's QK and PV WMMA clusters with
    #                 s_setprio(1)/(0). The wave in its matrix cluster wins
    #                 dispatch; the other (in softmax VALU / stalled on a gather)
    #                 yields. Instruction latency drifts the two waves out of
    #                 phase and the priority bias LOCKS the offset, so one wave
    #                 feeds the WMMA unit while the other runs softmax VALU --
    #                 overlapping the two execution units (the issue-bound win).
    #                 Barrier-free: gather mode shares no cross-wave data, so
    #                 there are zero cross-wave hazards to synchronize.
    sched_mode: str = "none"  # "none" | "pingpong"
    # batch_softmax: emit the per-row online-softmax ops grouped by operation
    # (the dense-D128 +4% single-wave win); ported here per wave.
    batch_softmax: bool = False
    # qk_ilp (WMMA-utilization / intra-wave pipelining): the QK matmul sums
    # n_dk d-chunks into ONE `score` accumulator -> the n_dk WMMAs form a serial
    # dependency chain (each reads+writes `score`), so the matrix unit stalls on
    # WMMA latency between issues. Splitting into `qk_ilp` INDEPENDENT partial
    # accumulators (round-robin the d-chunks, then vector-add the partials) lets
    # that many WMMA chains run concurrently -> the unit issues back-to-back and
    # WMMA utilization rises. Cost: (qk_ilp-1) extra <8xf32> live accumulators.
    # 1 = no split (serial); 2 or 4 = that many concurrent chains. Must divide
    # into n_dk sensibly (round-robin handles any value). PV is already
    # ILP-parallel (each output d-chunk is its own accumulator).
    qk_ilp: int = 1
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
            self.kv_source,
            f"vpe{self.waves_per_eu}" if self.waves_per_eu is not None else "vpedef",
            self.sched_mode,
            "bsm" if self.batch_softmax else "ssm",
            f"ilp{self.qk_ilp}",
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
    # Lever 1: explicit occupancy target so the 4-warp block's distributed
    # accumulator fits the VGPR file spill-free (CK's launch_bounds(128,2)).
    if cfg.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = cfg.waves_per_eu
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
    # Bank-pad the K-LDS leading dim (CK's qr_vr stages K bank-padded): consecutive
    # lanes read consecutive K rows (row = lane%16) at a fixed d-slice, so an
    # unpadded stride of ``hs`` (128 f16 = 64 dwords, a multiple of the 32 LDS
    # banks) collides all 16 rows onto the same banks. Pad by 8 f16 (keeps the
    # row-stride*2 a multiple of 16 for ds_read_b128 alignment) to skew the rows
    # across banks. V_lds_t is likewise padded on its leading (d) dim. The shared
    # K/V tiles only exist in the "lds" design; "gather" skips them (saves LDS ->
    # higher occupancy) and reads K/V from global per wave.
    use_lds = cfg.kv_source == "lds"
    K_lds = V_lds_t = None
    if use_lds:
        K_lds = make_lds_view(
            b,
            dtype=dtype_ir,
            shape=(_BLOCK_K, hs),
            strides=(hs + _LDS_PAD, 1),
            name_hint="Ksh",
        )
        V_lds_t = make_lds_view(
            b,
            dtype=dtype_ir,
            shape=(hs, _BLOCK_K),
            strides=(_BLOCK_K + _LDS_PAD, 1),
            name_hint="VshT",
        )
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
    if cfg.mask_mode == "causal":
        # Causal early-exit (same clamp as fmha_singlewave/fmha_pipelined): this
        # CTA owns q rows [cta_row0, cta_row0 + 16*W - 1]. K-tile kt covers keys
        # [kt*16, kt*16+15], needed iff kt*16 <= max query pos, i.e.
        # kt < cta_row0/16 + W. Clamp the K-loop to skip the fully-masked upper
        # triangle -- roughly halves the causal K-loop. Clamp at CTA granularity
        # so all W waves iterate in lockstep (lower waves just compute a few
        # fully-masked tiles, which the mask zeroes -- still correct).
        causal_stop = b.add(b.div(cta_row0, c_block_k), b.const_i32(W))
        loop_stop = b.select(b.cmp_lt(causal_stop, loop_stop), causal_stop, loop_stop)

    # ---- Cooperative loaders. All block_size threads participate. ----
    # HARD GUARD (mirrors fmha_regblocked): the vec8 loaders below index
    # ``tid + i*n_threads`` without masking surplus lanes, so if the tile does
    # not divide evenly into ``n_threads`` vec8 chunks the rounded-up iteration
    # issues OOB global loads and OOB ds_writes (GPU memory fault / 719). Refuse
    # to build such a config rather than emit an out-of-bounds kernel. (Only the
    # "lds" design runs the cooperative loaders.)
    n_threads = cfg.block_size
    elems = _BLOCK_K * hs  # tile element count (16 x head_size)
    if use_lds and elems % (n_threads * 8) != 0:
        raise ValueError(
            f"K/V tile {_BLOCK_K}x{hs} ({elems} elems) not divisible by "
            f"{n_threads}*8 (block_size*vec8); pick head_size/n_waves so the "
            f"cooperative loader divides evenly, or add lane predication"
        )
    n_chunks = elems // 8  # 8-wide vector chunks
    chunks_per_thread = n_chunks // n_threads if elems % (n_threads * 8) == 0 else 0

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

    # ---- "gather" B-operand path: read K/V straight from global (no LDS, no
    # cross-wave barrier). K for one (head,batch) is small and reused across the
    # wave's 16 lanes, so it stays cache-resident; this is the single-wave
    # winner's dataflow, replicated per wave. ----
    def k_window(k_tile_base):
        return make_tile_window(
            K_view, (1, 16, hs), origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0)
        )

    def v_window(k_tile_base):
        return make_tile_window(
            V_view, (1, 16, hs), origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0)
        )

    def gather_v_b_frag(vwin, d):
        # PV B-operand V[k, d_col] for k=0..15 (column gather, cache-resident).
        d_col = b.add(b.const_i32(d * 16), col)
        v_b = b.zero_vec(dtype_ir, atom.a_per_lane)
        for j in range(atom.a_per_lane):
            v_elem = vwin.load_scalar(b, c0, b.const_i32(j), d_col)
            v_b = b.vec_insert(v_b, v_elem, j)
        return v_b

    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        ms, ls, accs = unpack(state)
        k_tile_base = b.mul(kt, c_block_k)

        # ---- cooperative K/V staging (all waves) -- only in the "lds" design ----
        if use_lds:
            coop_load_k(k_tile_base)
            coop_load_v_t(k_tile_base)
            b.sync()
            kwin = vwin = None
        else:
            kwin = k_window(k_tile_base)
            vwin = v_window(k_tile_base)

        new_ms = list(ms)
        new_ls = list(ls)
        new_accs = list(accs)
        ps = [None] * c_frag

        pingpong = cfg.sched_mode == "pingpong"

        # ---- QK: A = this wave's Q rows (global), B = shared K (LDS) or gather.
        # qk_ilp independent partial-score accumulators break the serial WMMA
        # dependency chain so consecutive matrix ops pipeline (WMMA utilization).
        if pingpong:
            b.s_setprio(1)  # enter matrix cluster: win dispatch over the softmax wave
        ilp = max(1, cfg.qk_ilp)
        scores = [WmmaTensor.zero_acc(b, atom, arch=arch) for _ in range(ilp)]
        for d in range(n_dk):
            q_tile = load_wmma_tile(
                b, qwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
            )
            if use_lds:
                k_tile = WmmaTensor(atom, "b", load_k_b_frag(d), arch)
            else:
                k_tile = load_wmma_tile(
                    b, kwin, atom, lane, role="b", k_offset=d * 16, lead=[c0]
                )
            si = d % ilp
            scores[si] = wmma_mma(b, q_tile, k_tile, scores[si])
        # combine the partial score accumulators (matmul over d is a sum, so
        # score = Σ partials is exact): (ilp-1) elementwise <8xf32> vector adds.
        score = scores[0]
        for si in range(1, ilp):
            score = WmmaTensor(
                atom, "c", b.vector_add(score.value, scores[si].value), arch
            )
        if pingpong:
            b.s_setprio(0)  # leave matrix cluster: yield to the other wave's WMMA

        # ---- online softmax (low priority in pingpong: yields to the other
        # wave's WMMA cluster while this wave grinds the VALU) ----
        def _scaled_masked_row(r):
            row_rel, col_k = score.coord(b, lane, r)
            s_r = b.fmul(score.slot(b, r), scale_log2)
            return apply_attention_mask(
                b,
                s_r,
                mask_mode=cfg.mask_mode,
                k_idx=b.add(k_tile_base, col_k),
                query_pos=b.add(q_pos_base, row_rel),
                sliding_window=0,
            )

        alpha_vec = b.zero_vec_f32(c_frag)
        if cfg.batch_softmax:
            # grouped-by-op: independent same-class VALU adjacent (scheduling win)
            s_rows = [_scaled_masked_row(r) for r in range(c_frag)]
            row_maxs = [
                wave_reduce_max(b, s_rows[r], wave_size=wave, lanes_per_row=16)
                for r in range(c_frag)
            ]
            m_news = [b.fmax(ms[r], row_maxs[r]) for r in range(c_frag)]
            alphas = [b.exp2(b.fsub(ms[r], m_news[r])) for r in range(c_frag)]
            p_batch = [b.exp2(b.fsub(s_rows[r], m_news[r])) for r in range(c_frag)]
            row_sums = [
                wave_reduce_sum(b, p_batch[r], wave_size=wave, lanes_per_row=16)
                for r in range(c_frag)
            ]
            for r in range(c_frag):
                new_ms[r] = m_news[r]
                new_ls[r] = b.fadd(b.fmul(ls[r], alphas[r]), row_sums[r])
                ps[r] = p_batch[r]
                alpha_vec = b.vec_insert(alpha_vec, alphas[r], r)
        else:
            for r in range(c_frag):
                s_r = _scaled_masked_row(r)
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

        # ---- PV: A = P (LDS), B = shared transposed V (LDS) or gather ----
        if pingpong:
            b.s_setprio(1)  # enter PV matrix cluster
        p_tile = WmmaTensor(atom, "a", p_a, arch)
        for d in range(n_dk):
            if use_lds:
                v_b = load_v_b_frag(d)
            else:
                v_b = gather_v_b_frag(vwin, d)
            v_tile = WmmaTensor(atom, "b", v_b, arch)
            new_accs[d] = wmma_mma(b, p_tile, v_tile, new_accs[d])
        if pingpong:
            b.s_setprio(0)  # leave PV matrix cluster

        # barrier before next iteration overwrites the shared K/V LDS tiles.
        # In "gather" mode there is no shared LDS K/V, so no cross-wave barrier
        # is needed at all (the P-transpose already used an intra-wave waitcnt).
        if use_lds:
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
