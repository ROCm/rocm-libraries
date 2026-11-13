# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Multi-wave WMMA flash-attention forward body for RDNA (gfx11 and gfx12).

One design for both families, one query row per lane:

* Scores are computed transposed, ``S^T = K @ Q^T`` (K is the WMMA A operand,
  Q the B operand). The accumulator lane owns query row ``lane % 16``; its 8
  slots are 8 key positions. Row max / row sum are an in-lane reduction plus a
  single ``permlanex16`` exchange with the other lane half. The running ``m``
  and ``l`` are plain per-lane scalars (``l`` stays a per-half partial sum until
  the epilogue).
* PV is computed transposed too, ``O^T += V^T @ P^T``: P is the B operand.
  gfx12's accumulator layout (key ``8 * half + i``) *is* the B operand layout,
  so P feeds the WMMA straight from registers. gfx11 holds even/odd keys per
  half and its B operand wants all 16 keys duplicated in both halves: one
  ``permlanex16`` + two ``v_perm_b32`` per packed dword (CK's
  ``PermuteWarpGemmCToA``). No LDS round trip and no barrier for P.
* A CTA is ``block_m / 16`` waves (16 query rows each) that share every K/V
  tile. K is staged row-major and V transposed (``V^T[d][key]``) into padded,
  double-buffered LDS by cooperative 16-byte buffer loads, so both WMMA A
  operands are contiguous ``ds_read_b128`` reads. The next tile is fetched into
  registers while the current one is consumed; one barrier per tile.
* The key loop is split into an unmasked interior region (no per-score
  compares) and a masked region (causal diagonal, sliding-window edge, Sk
  tail), with runtime bounds. All shapes, strides, mask bounds and head widths
  are runtime; dtype, the head-width bucket, the mask class and the tile
  geometry are baked.
* Optional baked features: score transforms (softcap, ALiBi, QQ-bias, additive
  bias; applied on every tile, before the masks), attention sinks (seed the
  softmax state), OCP E4M3FN K/V (decoded and scaled into the LDS tiles) and
  paged K/V (a block-table lookup per key tile, or per row for pages smaller
  than a tile).

The caller (``kernels/gfx1151/wmma_fmha_fwd_v2.py``) declares parameters,
decodes the CTA, and builds per-(batch, head) buffer resources; this module
emits the K loop and the epilogue.

C++ mirror: ``platform/cpp/helpers/rdna_attention.cpp`` (byte-identical
emission, op for op).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rocke.core.ir import BF16, F16, F32, I8, I16, I32, IRBuilder, Value, VectorType

__all__ = ["RdnaAttentionConfig", "rdna_attention_fwd_body", "rdna_attention_lds_bytes"]

_WAVE = 32
# Lazy rescale (log2 domain): skip the O/l rescale when every lane's
# tile_max - m <= threshold. exp2(8) = 256 bounds P before its 16-bit cast.
_LAZY_RESCALE_THRESHOLD = 8.0
# A byte offset >= 2**31 is out of range of every rsrc (num_records < 2**31).
_OOB = -0x80000000
_LN2 = 0.6931471805599453


@dataclass(frozen=True)
class RdnaAttentionConfig:
    """Compile-time shape of one RDNA attention body."""

    dtype: str  # "fp16" | "bf16"
    head_dim: int  # Q/K head-width bucket (multiple of 16)
    head_dim_v: int  # V head-width bucket (multiple of 16)
    block_m: int  # query rows per CTA = 16 * waves
    block_n: int  # keys per K-loop tile (multiple of 16)
    causal: bool  # mask class: runtime causal / window bounds
    gfx12: bool  # gfx12 WMMA operand layout (else gfx11)
    lazy_rescale: bool = True
    k_pad: int = 8  # LDS row padding (elements) of the K tile
    v_pad: int = 8  # LDS row padding (elements) of the V^T tile
    bias: Optional[str] = None  # additive bias: None | "f32" | "q" (Q dtype)
    # Wave-pair head-width split: None -> automatic (either width above 128).
    # gfx12 holds an unsplit 256-wide Q (no operand duplication) as long as V
    # stays <= 128.
    wave_pair: Optional[bool] = None
    softcap: bool = False  # s = cap * tanh(s / cap) (runtime cap)
    sinks: bool = False  # per-head extra logit in the softmax denominator
    alibi: bool = False  # s += slope * (key - context)
    qq_bias: bool = False  # s += qq[query, key - context] inside its bounds
    kv_fp8: bool = False  # K / V stored as OCP E4M3FN bytes (runtime scales)
    page_block_size: int = 0  # paged K / V page (power of two); 0: contiguous

    @property
    def log_domain(self) -> bool:
        """Scores move to the log2 domain up front (needed by any additive or
        nonlinear score term); otherwise the scale folds into the exp2 FMA."""
        return bool(self.bias) or self.softcap or self.alibi or self.qq_bias

    @property
    def pair(self) -> bool:
        """Head widths split across a wave pair (two waves per 16 query rows,
        each owning half of the Q/K and of the V/O columns)."""
        if self.wave_pair is not None:
            return self.wave_pair
        return self.head_dim > 128 or self.head_dim_v > 128

    @property
    def waves(self) -> int:
        return (self.block_m // 16) * (2 if self.pair else 1)

    @property
    def threads(self) -> int:
        return _WAVE * self.waves

    @property
    def buffers(self) -> int:
        """LDS K/V tile buffers: two (one barrier per tile) or, for the wave
        pair, one (two barriers per tile; the S exchange needs one anyway)."""
        return 1 if self.pair else 2

    def validate(self) -> None:
        if self.dtype not in ("fp16", "bf16"):
            raise ValueError(f"dtype must be fp16/bf16, got {self.dtype!r}")
        for name, d in (("head_dim", self.head_dim), ("head_dim_v", self.head_dim_v)):
            if d <= 0 or d % 16 or d > 256:
                raise ValueError(f"{name} must be a multiple of 16 in 16..256, got {d}")
        if self.pair and (self.head_dim % 32 or self.head_dim_v % 32):
            raise ValueError(
                "the wave-pair split needs head widths that are multiples of 32"
            )
        if self.block_m % 16 or not 1 <= self.waves <= 16:
            raise ValueError(f"block_m {self.block_m} gives {self.waves} waves (1..16)")
        if self.block_n % 16 or self.block_n < 16:
            raise ValueError(f"block_n must be a multiple of 16, got {self.block_n}")
        if self.k_pad % 8 or self.v_pad % 8:
            raise ValueError("LDS paddings must keep 16-byte row alignment")
        if self.bias not in (None, "f32", "q"):
            raise ValueError(f"bias must be None, 'f32' or 'q', got {self.bias!r}")
        if not self.pair and (
            self.head_dim_v > 128 or self.head_dim > (256 if self.gfx12 else 128)
        ):
            raise ValueError(
                "unsplit waves hold Q <= 128 (gfx11) / 256 (gfx12) and V <= 128"
            )
        ps = self.page_block_size
        if ps < 0 or ps & (ps - 1):
            raise ValueError(f"page_block_size must be 0 or a power of two, got {ps}")


def _tile_elems(cfg: RdnaAttentionConfig):
    k_tile = cfg.block_n * (cfg.head_dim + cfg.k_pad)
    v_tile = cfg.head_dim_v * (cfg.block_n + cfg.v_pad)
    return k_tile, v_tile


def _xs_floats(cfg: RdnaAttentionConfig) -> int:
    """f32 LDS slots of the wave-pair partial-score exchange."""
    return cfg.waves * _WAVE * (cfg.block_n // 16) * 8 if cfg.pair else 0


def _v_group(cfg: RdnaAttentionConfig):
    """(rows per V unit G, units per thread) for the transposed V store.

    A unit is G consecutive key rows x 8 head-dim columns: G 16-byte loads and
    eight G-element LDS stores into V^T. G minimises the per-thread critical
    path ``units * (G + 8)``; a partial last round is predicated."""
    nt = cfg.threads
    best = None
    for g in (4, 2, 1):
        if cfg.block_n % g:
            continue
        per = -(-(cfg.block_n // g) * (cfg.head_dim_v // 8) // nt)
        cost = per * (g + 8)
        if best is None or cost < best[0]:
            best = (cost, g, per)
    return best[1], best[2]


def _partial_rounds(cfg: RdnaAttentionConfig) -> bool:
    """Whether a cooperative K or V round leaves some threads without work.

    Such threads load out of range (no traffic) and store to a private dummy
    LDS slot instead of branching, which would split the loop body into
    blocks the scheduler cannot overlap."""
    g, _ = _v_group(cfg)
    k_chunks = cfg.block_n * (cfg.head_dim // 8)
    v_units = (cfg.block_n // g) * (cfg.head_dim_v // 8)
    return bool(k_chunks % cfg.threads or v_units % cfg.threads)


def _lds_elems(cfg: RdnaAttentionConfig) -> int:
    k_tile, v_tile = _tile_elems(cfg)
    dummy = cfg.threads * 8 if _partial_rounds(cfg) else 0
    return cfg.buffers * (k_tile + v_tile) + dummy


def rdna_attention_lds_bytes(cfg: RdnaAttentionConfig) -> int:
    """Static LDS bytes of one CTA: the K/V tile buffers (plus the idle-round
    dummy slots) and, for the wave pair, the partial-score exchange."""
    return _lds_elems(cfg) * 2 + _xs_floats(cfg) * 4


def rdna_attention_fwd_body(
    b: IRBuilder,
    cfg: RdnaAttentionConfig,
    *,
    q_rsrc: Value,
    k_rsrc: Value,
    v_rsrc: Value,
    o_rsrc: Value,
    lse_rsrc: Optional[Value],
    stride_q: Value,
    stride_k: Value,
    stride_v: Value,
    stride_o: Value,
    stride_lse: Optional[Value],
    k_bytes: Optional[Value],
    v_bytes: Optional[Value],
    head_dim_q: Value,
    head_dim_v: Value,
    seqlen_q: Value,
    seqlen_k: Value,
    q_start: Value,
    scale_log2: Value,
    diag_offset: Optional[Value] = None,
    window_left: Optional[Value] = None,
    window_right: Optional[Value] = None,
    bias_rsrc: Optional[Value] = None,
    stride_bias_q: Optional[Value] = None,
    bias_lead: Optional[Value] = None,
    bias_bytes: Optional[Value] = None,
    softcap: Optional[Value] = None,
    sink_log2: Optional[Value] = None,
    alibi_slope_log2: Optional[Value] = None,
    context: Optional[Value] = None,
    qq_rsrc: Optional[Value] = None,
    qq_rows: Optional[Value] = None,
    qq_cols: Optional[Value] = None,
    qq_stride: Optional[Value] = None,
    k_scale: Optional[Value] = None,
    v_scale: Optional[Value] = None,
    block_table: Optional[Value] = None,
    table_row: Optional[Value] = None,
    stride_k_block: Optional[Value] = None,
    stride_v_block: Optional[Value] = None,
    num_blocks: Optional[Value] = None,
) -> None:
    """Emit the K loop and epilogue of one CTA.

    Buffer resources address one (batch, head) slice each; offsets are in
    bytes from the slice base and reads past the resource return zero.
    ``stride_*`` are token strides in elements (the head-dim stride is 1).
    ``k_bytes`` / ``v_bytes`` are the K / V resources' ``num_records``, used
    as an always-out-of-range tile offset. ``scale_log2 = softmax_scale *
    log2(e)`` must be positive. Causal class: key ``k`` is visible to query
    ``q`` iff ``q + diag_offset - window_left <= k <= q + diag_offset +
    window_right``; a negative window bound is unbounded on that side.
    The additive bias (``cfg.bias``) is ``bias[q * stride_bias_q + k]`` read
    from ``bias_rsrc`` (a 0 query stride broadcasts over queries). A Q-dtype
    bias resource must start at a dword and span whole dwords
    (``bias_bytes`` is its ``num_records``), with the bias at element
    ``bias_lead`` (0 or 1) of it: the hardware drops every dword of a load
    that ends past ``num_records``. gfx11 reads each row from its
    dword-aligned element and shifts in registers; gfx12 reads rows from their
    exact start and restores the one in-range element such a dword can hold,
    the resource's last.

    Features (log2 domain ``s2 = s * scale_log2``, in this order, every tile):
    softcap ``s2 = cap2 * tanh(s2 / cap2)`` with ``cap2 = softcap * log2(e)``;
    ALiBi ``s2 += alibi_slope_log2 * (k - context)``; QQ-bias ``s2 +=
    qq[q * qq_stride + k - context] * log2(e)`` for ``q < qq_rows`` and ``0 <=
    k - context < qq_cols`` (``qq_rsrc`` over the whole matrix); then the
    additive bias and the masks. Sinks seed the softmax with ``m =
    sink_log2`` (the sink logit times log2(e)) and one unit of denominator
    mass. FP8 K / V (``cfg.kv_fp8``): offsets and ``k_bytes`` / ``v_bytes``
    are in bytes of one-byte elements; values are decoded, multiplied by
    ``k_scale`` / ``v_scale`` and rounded to the Q dtype on the way to LDS.
    Paged K / V (``cfg.page_block_size``): ``k_rsrc`` / ``v_rsrc`` cover the
    whole cache at this head; key ``t`` lives at element ``block_table[
    table_row + t / page] * stride_{k,v}_block + (t % page) * stride_{k,v}``
    (``k_bytes`` / ``v_bytes`` unused). The table's values are caller data: a
    page id outside ``[0, num_blocks)`` (the caches' page count) is not
    addressed -- its keys read as zero K / V rows, like rows past the sequence.
    """
    cfg.validate()
    bf16 = cfg.dtype == "bf16"
    elem = BF16 if bf16 else F16
    gfx12 = cfg.gfx12
    pair = cfg.pair
    BN, DQ, DV = cfg.block_n, cfg.head_dim, cfg.head_dim_v
    NT = cfg.threads
    n_kv = BN // 16
    # Per-wave head-width share (half of each width for the wave pair).
    DQW, DVW = (DQ // 2, DV // 2) if pair else (DQ, DV)
    n_dq, n_dv = DQW // 16, DVW // 16
    op_id = ("wmma_gfx12_f32_16x16x16_" if gfx12 else "wmma_f32_16x16x16_") + (
        "bf16" if bf16 else "f16"
    )
    KS = DQ + cfg.k_pad
    VS = BN + cfg.v_pad
    k_tile, v_tile = _tile_elems(cfg)
    buf_elems = k_tile + v_tile
    causal = cfg.causal
    if causal and (diag_offset is None or window_left is None or window_right is None):
        raise ValueError("the causal class needs diag_offset and both window bounds")
    if cfg.bias and (bias_rsrc is None or stride_bias_q is None):
        raise ValueError("an additive bias needs its resource and query stride")
    if cfg.bias == "q" and (bias_lead is None or bias_bytes is None):
        raise ValueError("a Q-dtype bias needs its lead element and byte size")
    paged = cfg.page_block_size > 0
    missing = [
        name
        for name, need, vals in (
            ("softcap", cfg.softcap, (softcap,)),
            ("sinks", cfg.sinks, (sink_log2,)),
            ("alibi", cfg.alibi, (alibi_slope_log2, context)),
            ("qq_bias", cfg.qq_bias, (qq_rsrc, qq_rows, qq_cols, qq_stride, context)),
            ("kv_fp8", cfg.kv_fp8, (k_scale, v_scale)),
            (
                "paged",
                paged,
                (block_table, table_row, stride_k_block, stride_v_block, num_blocks),
            ),
            ("dense K/V", not paged, (k_bytes, v_bytes)),
        )
        if need and any(v is None for v in vals)
    ]
    if missing:
        raise ValueError(f"missing operands for {', '.join(missing)}")

    ci = b.const_i32
    c0, c1 = ci(0), ci(1)
    oob = ci(_OOB)
    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)
    # With a bias (or any other score feature) the scores move to the log2
    # domain up front (s * c + b * log2e) and the softmax runs at unit scale;
    # otherwise the scale folds into one FMA per score (p = exp2(s * c - m *
    # c)).
    c_in = scale_log2
    c = b.const_f32(1.0) if cfg.log_domain else scale_log2

    tid = b.thread_id_x()
    lane = b.land(tid, ci(31))
    col = b.land(tid, ci(15))  # query row within the wave's 16-row tile
    half = b.land(b.lshr(tid, ci(4)), c1)  # lane half (0: lanes 0-15)
    lo_half = b.cmp_eq(half, c0)
    wave = b.readfirstlane(b.lshr(tid, ci(5)))
    if pair:
        row_wave = b.lshr(wave, c1)
        side = b.land(wave, c1)  # which half of the head widths this wave owns
        dq_base = b.mul(side, ci(DQW))
        dv_base = b.mul(side, ci(DVW))
    else:
        row_wave, side, dq_base, dv_base = wave, None, c0, c0
    q_row = b.add(b.add(q_start, b.mul(row_wave, ci(16))), col)

    def i32x4_to(v, n):
        return b.vec_bitcast(v, VectorType(elem, n))

    def load16(rsrc, byte_off):
        return b.buffer_load_vN(rsrc, byte_off, c0, I32, 4)

    def masked_off(byte_off, ok):
        return b.select(ok, byte_off, oob)

    def pack2(p0, p1):
        """Two f32 -> one dword of two 16-bit values (p0 low)."""
        if bf16:
            # Round-to-nearest-even without NaN handling (P is finite).
            r = []
            for p in (p0, p1):
                x = b.bitcast(p, I32)
                lsb = b.land(b.lshr(x, ci(16)), c1)
                r.append(b.add(b.add(x, ci(0x7FFF)), lsb))
            return b.perm_b32(r[1], r[0], ci(0x07060302))
        lo = b.zext(b.bitcast(b.cast_f32_to(p0, F16), I16), I32)
        hi = b.zext(b.bitcast(b.cast_f32_to(p1, F16), I16), I32)
        return b.lor(lo, b.shl(hi, ci(16)))

    def permx16_f32(v):
        return b.bitcast(b.permlanex16(b.bitcast(v, I32)), F32)

    def tree(vals, op, op3=None):
        while len(vals) > 1:
            nxt = []
            i = 0
            if op3 is not None:
                while i + 3 <= len(vals):
                    nxt.append(op3(vals[i], vals[i + 1], vals[i + 2]))
                    i += 3
            while i + 2 <= len(vals):
                nxt.append(op(vals[i], vals[i + 1]))
                i += 2
            nxt.extend(vals[i:])
            vals = nxt
        return vals[0]

    # ---- Q: B operand of S^T = K Q^T, resident in registers ----
    q_row_off = b.add(b.mul(q_row, stride_q), dq_base)
    q_frags: List[Value] = []
    for dt in range(n_dq):
        if gfx12:
            d0 = b.add(ci(dt * 16), b.mul(half, ci(8)))
            off = b.mul(b.add(q_row_off, d0), ci(2))
            ok = b.cmp_lt(b.add(d0, dq_base), head_dim_q)
            q_frags.append(i32x4_to(load16(q_rsrc, masked_off(off, ok)), 8))
        else:
            parts = []
            for h8 in (0, 8):
                d0 = dt * 16 + h8
                off = b.mul(b.add(q_row_off, ci(d0)), ci(2))
                ok = b.cmp_lt(b.add(ci(d0), dq_base), head_dim_q)
                parts.append(i32x4_to(load16(q_rsrc, masked_off(off, ok)), 8))
            q_frags.append(b.vec_concat(parts[0], parts[1]))

    # ---- LDS: [K tile | V^T tile] x buffers (+ wave-pair score exchange) ----
    lds = b.smem_alloc(elem, [_lds_elems(cfg)], name_hint="attn_kv")
    # Private 16-byte slot per thread for stores of idle cooperative rounds.
    dummy = b.add(ci(cfg.buffers * buf_elems), b.mul(tid, ci(8)))
    xs = b.smem_alloc(F32, [_xs_floats(cfg)], name_hint="attn_xs") if pair else None

    # K / V element bytes, page geometry. Pages that are not a whole number of
    # key tiles take one block-table lookup per cooperative row instead of
    # one per tile.
    ps = cfg.page_block_size
    row_paged = paged and ps % BN != 0

    def kv_byte(x):
        """K / V element offset -> byte offset."""
        return x if cfg.kv_fp8 else b.mul(x, ci(2))

    def load_kv(rsrc, byte_off):
        """One cooperative chunk: 8 K / V elements (16 bytes, or 8 FP8 bytes)."""
        if cfg.kv_fp8:
            return b.buffer_load_vN(rsrc, byte_off, c0, I32, 2)
        return load16(rsrc, byte_off)

    # Cooperative K mapping: 16-byte chunk -> (row, col chunk).
    cpr_k = DQ // 8
    tot_k = BN * cpr_k
    nk = -(-tot_k // NT)
    k_goff, k_loff = [], []
    k_valid, k_rows = [], []
    for i in range(nk):
        cidx = b.add(tid, ci(i * NT))
        valid = b.cmp_lt(cidx, ci(tot_k)) if (i + 1) * NT > tot_k else None
        r = b.div(cidx, ci(cpr_k))
        d0 = b.mul(b.mod(cidx, ci(cpr_k)), ci(8))
        off = kv_byte(d0 if row_paged else b.add(b.mul(r, stride_k), d0))
        ok = b.cmp_lt(d0, head_dim_q)
        k_goff.append(masked_off(off, ok if valid is None else b.land(ok, valid)))
        k_loff.append(b.add(b.mul(r, ci(KS)), d0))
        k_valid.append(valid)
        k_rows.append(r)

    # Cooperative V mapping: G key rows x 8 columns per unit, stored transposed.
    G, nv = _v_group(cfg)
    groups = BN // G
    tot_units = groups * (DV // 8)
    v_goff, v_loff, v_valid, v_rows = [], [], [], []
    for i in range(nv):
        uidx = b.add(tid, ci(i * NT))
        valid = b.cmp_lt(uidx, ci(tot_units)) if (i + 1) * NT > tot_units else None
        g = b.mod(uidx, ci(groups))
        d0 = b.mul(b.div(uidx, ci(groups)), ci(8))
        ok = b.cmp_lt(d0, head_dim_v)
        if valid is not None:
            ok = b.land(ok, valid)
        offs, rows = [], []
        for rr in range(G):
            row = b.add(b.mul(g, ci(G)), ci(rr))
            off = kv_byte(d0 if row_paged else b.add(b.mul(row, stride_v), d0))
            offs.append(masked_off(off, ok))
            rows.append(row)
        v_goff.append(offs)
        v_loff.append(b.add(b.mul(d0, ci(VS)), b.mul(g, ci(G))))
        v_valid.append(valid)
        v_rows.append(rows)
    if not paged:
        kv_eb = 1 if cfg.kv_fp8 else 2
        tile_k_bytes = b.mul(stride_k, ci(BN * kv_eb))
        tile_v_bytes = b.mul(stride_v, ci(BN * kv_eb))

    def issue_loads(t, valid):
        """Global loads of key tile ``t`` (zeros, no traffic, if not valid).

        Dense K/V addressing is the per-thread byte offsets above plus one
        per-tile offset here."""
        if paged:
            return issue_paged_loads(t, valid)
        tk = b.select(valid, b.mul(t, tile_k_bytes), k_bytes)
        tv = b.select(valid, b.mul(t, tile_v_bytes), v_bytes)
        kr = [load_kv(k_rsrc, b.add(o, tk)) for o in k_goff]
        vr = [[load_kv(v_rsrc, b.add(o, tv)) for o in offs] for offs in v_goff]
        return kr, vr

    def page_ok(phys):
        """``phys`` names a page of the cache: ``0 <= phys < num_blocks``."""
        return b.land(b.cmp_ge(phys, c0), b.cmp_lt(phys, num_blocks))

    def issue_paged_loads(t, valid):
        """Paged K / V: rows past the sequence (or of an invalid tile) read
        nothing; their block-table index is clamped to 0."""
        lps = ps.bit_length() - 1
        tok0 = b.mul(t, ci(BN))
        rem = b.select(valid, b.sub(seqlen_k, tok0), c0)  # live rows of the tile
        if not row_paged:
            # The whole tile sits in one page: one lookup per tile.
            blk = b.select(valid, b.lshr(tok0, ci(lps)), c0)
            phys = b.global_load_i32(block_table, b.add(table_row, blk))
            rem = b.select(page_ok(phys), rem, c0)
            in_page = b.land(tok0, ci(ps - 1))
            tk = kv_byte(b.add(b.mul(phys, stride_k_block), b.mul(in_page, stride_k)))
            tv = kv_byte(b.add(b.mul(phys, stride_v_block), b.mul(in_page, stride_v)))

            def at_tile(o, r, tile_off):
                return b.select(b.cmp_lt(r, rem), b.add(o, tile_off), oob)

            kr = [load_kv(k_rsrc, at_tile(o, r, tk)) for o, r in zip(k_goff, k_rows)]
            vr = [
                [load_kv(v_rsrc, at_tile(o, r, tv)) for o, r in zip(offs, rows)]
                for offs, rows in zip(v_goff, v_rows)
            ]
            return kr, vr

        def at_row(o, r, stride_blk, stride_tok):
            tok = b.add(tok0, r)
            ok = b.cmp_lt(r, rem)
            blk = b.select(ok, b.lshr(tok, ci(lps)), c0)
            phys = b.global_load_i32(block_table, b.add(table_row, blk))
            ok = b.land(ok, page_ok(phys))
            in_page = b.land(tok, ci(ps - 1))
            row = b.add(b.mul(phys, stride_blk), b.mul(in_page, stride_tok))
            return b.select(ok, b.add(o, kv_byte(row)), oob)

        kr = [
            load_kv(k_rsrc, at_row(o, r, stride_k_block, stride_k))
            for o, r in zip(k_goff, k_rows)
        ]
        vr = [
            [
                load_kv(v_rsrc, at_row(o, r, stride_v_block, stride_v))
                for o, r in zip(offs, rows)
            ]
            for offs, rows in zip(v_goff, v_rows)
        ]
        return kr, vr

    def widen_fp8(raw, scale):
        """Eight OCP E4M3FN bytes (``<2 x i32>``) -> ``<8 x elem>``: decoded,
        multiplied by ``scale`` and rounded to the Q dtype."""
        from rocke.helpers.attention import decode_fp8e4m3fn_to_f32

        words = [b.vec_extract(raw, 0), b.vec_extract(raw, 1)]
        vals = []
        for k in range(8):
            w = words[k // 4]
            if k % 4:
                w = b.lshr(w, ci(8 * (k % 4)))
            f = decode_fp8e4m3fn_to_f32(b, b.trunc(w, I8))
            vals.append(b.cast_f32_to(b.fmul(f, scale), elem))
        return b.vec_pack(vals, elem)

    def dst_of(addr, valid):
        return addr if valid is None else b.select(valid, addr, dummy)

    def store_k(base, kr):
        for o, v, valid in zip(k_loff, kr, k_valid):
            dst = dst_of(b.add(base, o), valid)
            vec = widen_fp8(v, k_scale) if cfg.kv_fp8 else i32x4_to(v, 8)
            b.smem_store_vN(lds, [dst], vec, 8)

    def store_v(base, vr):
        vbase = b.add(base, ci(k_tile))
        for o, rows, valid in zip(v_loff, vr, v_valid):
            store_v_unit(vbase, o, rows, valid)

    def store_v_unit(vbase, o, rows, valid):
        if cfg.kv_fp8:
            i32x4 = VectorType(I32, 4)
            rows = [b.vec_bitcast(widen_fp8(rv, v_scale), i32x4) for rv in rows]
        words = [[b.vec_extract(rv, j) for j in range(4)] for rv in rows]
        for j in range(4):
            for odd in (0, 1):
                dst = dst_of(b.add(vbase, b.add(o, ci((2 * j + odd) * VS))), valid)
                if G == 1:
                    v8 = i32x4_to(rows[0], 8)
                    b.smem_store_vN(lds, [dst], b.vec_extract(v8, 2 * j + odd), 1)
                    continue
                sel = ci(0x07060302 if odd else 0x05040100)
                pairs = [
                    b.perm_b32(words[rr + 1][j], words[rr][j], sel)
                    for rr in range(0, G, 2)
                ]
                vec = b.vec_pack(pairs, I32)
                b.smem_store_vN(lds, [dst], b.vec_bitcast(vec, VectorType(elem, G)), G)

    # Per-lane LDS operand bases (elements, before the buffer offset).
    hoff = b.mul(half, ci(8)) if gfx12 else c0
    k_lane = b.add(b.add(b.mul(col, ci(KS)), hoff), dq_base)
    v_lane = b.add(b.add(b.mul(col, ci(VS)), hoff), b.mul(dv_base, ci(VS)))

    def lds_frag(off):
        if gfx12:
            return b.smem_load_vN(lds, off, dtype=elem, n=8)
        lo = b.smem_load_vN(lds, off, dtype=elem, n=8)
        hi = b.smem_load_vN(lds, b.add(off, ci(8)), dtype=elem, n=8)
        return b.vec_concat(lo, hi)

    # gfx11 C->B transpose selectors (swapped for the upper half).
    sel_lo = b.select(lo_half, ci(0x05040100), ci(0x01000504))
    sel_hi = b.select(lo_half, ci(0x07060302), ci(0x03020706))

    def p_operand(ps):
        if gfx12:
            words = [pack2(ps[2 * j], ps[2 * j + 1]) for j in range(4)]
            return b.vec_bitcast(b.vec_pack(words, I32), VectorType(elem, 8))
        out = []
        for m in range(4):
            v = pack2(ps[2 * m], ps[2 * m + 1])  # keys {4m+h, 4m+2+h}
            w = b.permlanex16(v)
            out.append(b.perm_b32(w, v, sel_lo))  # keys {4m, 4m+1}
            out.append(b.perm_b32(w, v, sel_hi))  # keys {4m+2, 4m+3}
        return b.vec_bitcast(b.vec_pack(out, I32), VectorType(elem, 16))

    # Key offset of accumulator slot i inside a 16-key sub-tile, minus the
    # lane-half part (folded into the per-lane bounds).
    def slot_key(ns, i):
        return ns * 16 + (i if gfx12 else 2 * i)

    half_key = b.mul(half, ci(8)) if gfx12 else half

    # ---- additive bias: this lane's query row, its slots' keys ----
    log2e = b.const_f32(1.4426950408889634)
    bias_row = b.mul(q_row, stride_bias_q) if cfg.bias else None
    if cfg.bias == "q":
        row = b.add(bias_row, bias_lead)
        if gfx12:
            # Rows load from their exact start, which may sit mid-dword. Of a
            # dword dropped for ending past num_records, only the low half at
            # num_records - 2 can be in range: the resource's last element,
            # key seqlen_k - 1 of the last row. A mid-dword row ending on a
            # dword has an odd seqlen_k, so that key is in a masked tile; the
            # element's dword is fetched once here.
            bias_row = row
            bias_tail_off = b.sub(bias_bytes, ci(2))
            tail = b.buffer_load_vN(bias_rsrc, b.sub(bias_bytes, ci(4)), c0, I32, 1)
            bias_tail = b.lshr(b.vec_extract(tail, 0), ci(16))
        else:
            # Loads start at the dword holding this lane's first key (relative
            # to the dword-aligned resource); ``bias_par`` is the 16-bit half it
            # sits in. Tile and sub-tile key offsets are even, so the parity is
            # per lane.
            row = b.add(row, half)
            bias_par = b.land(row, c1)
            bias_row = b.sub(row, bias_par)
            bias_sh = b.mul(bias_par, ci(16))

    # ---- score features: per-CTA constants, per-lane QQ-bias row ----
    if cfg.softcap:
        cap2 = b.fmul(softcap, log2e)
        cap_in = b.fdiv(c_in, cap2)  # raw score -> tanh argument
    if cfg.qq_bias:
        qq_row = b.mul(q_row, qq_stride)
        qq_q_ok = b.cmp_lt(q_row, qq_rows)

    def rel_keys(t):
        """Key of slot 0 of this lane in tile ``t``, minus ``context``."""
        return b.sub(b.add(b.mul(t, ci(BN)), half_key), context)

    def load_qq(rel0):
        """QQ-bias (raw) of every score slot; 0 outside its bounds."""
        out = []
        for ns in range(n_kv):
            vals = []
            for i in range(8):
                rel = b.add(rel0, ci(slot_key(ns, i)))
                ok = b.land(qq_q_ok, b.land(b.cmp_ge(rel, c0), b.cmp_lt(rel, qq_cols)))
                off = masked_off(b.mul(b.add(qq_row, rel), ci(4)), ok)
                vals.append(
                    b.vec_extract(b.buffer_load_vN(qq_rsrc, off, c0, F32, 1), 0)
                )
            out.append(vals)
        return out

    def load_bias(t, masked):
        """Bias (log2 domain) of every score slot of key tile ``t``."""
        eb = 4 if cfg.bias == "f32" else 2
        out = []
        for ns in range(n_kv):
            # gfx12: the 8 contiguous keys at 8 * half; gfx11: all 16 keys of
            # the sub-tile, of which the lane keeps parity ``half`` (Q dtype:
            # the 16 keys from the lane's aligned start, keeping ``bias_par``).
            k0 = b.add(b.mul(t, ci(BN)), ci(ns * 16))
            if gfx12:
                k0 = b.add(k0, b.mul(half, ci(8)))
            n_keys = 8 if gfx12 else 16
            base = b.mul(b.add(bias_row, k0), ci(eb))
            words = []
            for q in range(n_keys * eb // 16):
                v4 = load16(bias_rsrc, b.add(base, ci(16 * q)))
                words.extend(b.vec_extract(v4, j) for j in range(4))
            if cfg.bias == "q" and gfx12 and masked:
                # Restore the resource's last element if this lane's load lost it.
                words = [
                    b.select(
                        b.cmp_eq(b.add(base, ci(4 * j)), bias_tail_off), bias_tail, w
                    )
                    for j, w in enumerate(words)
                ]
            vals = []
            for i in range(8):
                if cfg.bias == "f32":
                    w = (
                        words[i]
                        if gfx12
                        else b.select(lo_half, words[2 * i], words[2 * i + 1])
                    )
                    x = b.bitcast(w, F32)
                else:
                    w = words[i // 2] if gfx12 else words[i]
                    sh = ci(16 * (i % 2)) if gfx12 else bias_sh
                    h = b.land(b.lshr(w, sh), ci(0xFFFF))
                    if bf16:
                        x = b.bitcast(b.shl(h, ci(16)), F32)
                    else:
                        x = b.cast_to_f32(b.bitcast(b.trunc(h, I16), F16))
                vals.append(b.fmul(x, log2e))
            out.append(vals)
        return out

    # ---- key-tile ranges (all scalar) ----
    def cdiv(x, d):
        return b.div(b.add(x, ci(d - 1)), ci(d))

    zero_rows = b.cmp_le(seqlen_k, c0)
    sk = b.smax(seqlen_k, c0)
    n_tiles = cdiv(sk, BN)
    if not causal:
        lo_t, full_lo, full_hi, hi_t = c0, c0, b.div(sk, ci(BN)), n_tiles
        lane_hi = sk
        lane_lo = None
    else:
        q_last = b.sub(b.smin(b.add(q_start, ci(cfg.block_m)), seqlen_q), c1)
        q_last = b.smax(q_last, q_start)
        wl, wr = window_left, window_right
        has_l = b.cmp_ge(wl, c0)
        has_r = b.cmp_ge(wr, c0)

        def clamp_k(x):
            return b.smin(b.smax(x, c0), sk)

        k_lo = b.select(has_l, clamp_k(b.sub(b.add(q_start, diag_offset), wl)), c0)
        k_hi = b.select(
            has_r, clamp_k(b.add(b.add(b.add(q_last, diag_offset), wr), c1)), sk
        )
        lo_t = b.div(k_lo, ci(BN))
        hi_t = b.smax(cdiv(k_hi, BN), lo_t)
        fl = b.select(
            has_l, cdiv(clamp_k(b.sub(b.add(q_last, diag_offset), wl)), BN), lo_t
        )
        full_lo = b.smin(b.smax(fl, lo_t), hi_t)
        fh = b.select(
            has_r,
            b.div(clamp_k(b.add(b.add(b.add(q_start, diag_offset), wr), c1)), ci(BN)),
            b.div(sk, ci(BN)),
        )
        full_hi = b.smin(b.smax(fh, full_lo), hi_t)
        diag = b.add(q_row, diag_offset)
        lane_hi = b.select(has_r, b.smin(b.add(b.add(diag, wr), c1), sk), sk)
        lane_lo = b.select(has_l, b.sub(diag, wl), ci(-(1 << 30)))
    n_full = b.sub(full_hi, full_lo)
    n_left = b.sub(full_lo, lo_t)
    total = b.add(n_full, b.add(n_left, b.sub(hi_t, full_hi)))
    total = b.select(zero_rows, c0, total)
    n_full = b.select(zero_rows, c0, n_full)

    def tile_of(j):
        if not causal:
            return j
        jl = b.sub(j, n_full)
        return b.select(
            b.cmp_lt(j, n_full),
            b.add(full_lo, j),
            b.select(
                b.cmp_lt(jl, n_left), b.add(lo_t, jl), b.add(full_hi, b.sub(jl, n_left))
            ),
        )

    # ---- prologue: tile 0 -> buffer 0 ----
    kr, vr = issue_loads(tile_of(c0), b.cmp_gt(total, c0))
    store_k(c0, kr)
    store_v(c0, vr)
    b.sync_lds_only()
    n_quads = n_kv * 2

    def exchange_scores(s):
        """Wave pair: publish this wave's partial S^T; returns the partner's base."""
        mine = b.mul(b.add(b.mul(wave, ci(n_quads * _WAVE)), lane), ci(4))
        theirs = b.mul(b.add(b.mul(b.xor(wave, c1), ci(n_quads * _WAVE)), lane), ci(4))
        for ns in range(n_kv):
            for h in range(2):
                v4 = b.vec_pack(
                    [b.vec_extract(s[ns], 4 * h + j) for j in range(4)], F32
                )
                b.smem_store_vN(xs, [b.add(mine, ci((ns * 2 + h) * _WAVE * 4))], v4, 4)
        return theirs

    def step(j, state, masked):
        tag = "msk" if masked else "full"
        m_old, l_old = state[0], state[1]
        accs = list(state[2:])
        t = tile_of(j)
        if cfg.buffers == 2:
            buf = b.mul(b.land(j, c1), ci(buf_elems))
            nbuf = b.sub(ci(buf_elems), buf)
        else:
            buf = nbuf = c0
        j1 = b.add(j, c1)
        has_next = b.cmp_lt(j1, total)
        nkr, nvr = issue_loads(tile_of(b.smin(j1, b.sub(total, c1))), has_next)
        bias = load_bias(t, masked) if cfg.bias else None
        rel0 = rel_keys(t) if cfg.alibi or cfg.qq_bias else None
        qq = load_qq(rel0) if cfg.qq_bias else None

        # S^T = K Q^T, d-outer so the n_kv chains stay independent.
        kb = b.add(buf, k_lane)
        s = [b.zero_vec_f32(8) for _ in range(n_kv)]
        for dt in range(n_dq):
            for ns in range(n_kv):
                kf = lds_frag(b.add(kb, ci(ns * 16 * KS + dt * 16)))
                s[ns] = b.mma(op_id, kf, q_frags[dt], s[ns])

        if pair:
            theirs = exchange_scores(s)
            b.sync_lds_only()  # partials visible; every wave is done reading K
            store_k(nbuf, nkr)
            for ns in range(n_kv):
                parts = [
                    b.smem_load_vN(
                        xs, b.add(theirs, ci((ns * 2 + h) * _WAVE * 4)), dtype=F32, n=4
                    )
                    for h in range(2)
                ]
                s[ns] = b.vector_add(s[ns], b.vec_concat(parts[0], parts[1]))

        # Score transforms run on the per-slot scores here, in order: the
        # move to the log2 domain (through softcap), ALiBi, QQ-bias, the
        # additive bias, then the masks.
        vals = [[b.vec_extract(s[ns], i) for i in range(8)] for ns in range(n_kv)]
        if cfg.softcap or cfg.alibi or cfg.qq_bias:
            fk0 = b.sitofp_f32(rel0) if cfg.alibi else None
            out = []
            for ns in range(n_kv):
                row = []
                for i in range(8):
                    x = vals[ns][i]
                    if cfg.softcap:
                        x = b.fmul(cap2, b.tanh(b.fmul(x, cap_in)))
                    else:
                        x = b.fmul(x, c_in)
                    if cfg.alibi:
                        kf = b.fadd(fk0, b.const_f32(float(slot_key(ns, i))))
                        x = b.fma(alibi_slope_log2, kf, x)
                    if cfg.qq_bias:
                        x = b.fma(qq[ns][i], log2e, x)
                    if cfg.bias:
                        x = b.fadd(x, bias[ns][i])
                    row.append(x)
                out.append(row)
            vals = out
        elif cfg.bias:
            vals = [
                [b.fma(vals[ns][i], c_in, bias[ns][i]) for i in range(8)]
                for ns in range(n_kv)
            ]
        if masked:
            kbase = b.add(b.mul(t, ci(BN)), half_key)
            hi_rel = b.sub(lane_hi, kbase)
            lo_rel = b.sub(lane_lo, kbase) if lane_lo is not None else None
            for ns in range(n_kv):
                for i in range(8):
                    kv = ci(slot_key(ns, i))
                    keep = b.cmp_lt(kv, hi_rel)
                    if lo_rel is not None:
                        keep = b.land(keep, b.cmp_ge(kv, lo_rel))
                    vals[ns][i] = b.select(keep, vals[ns][i], neg_inf)
        flat = [v for row in vals for v in row]
        local_max = tree(list(flat), b.fmax, b.fmax3)
        tile_max = b.fmax(local_max, permx16_f32(local_max))
        if cfg.lazy_rescale:
            within = b.fcmp(
                "ole",
                b.fmul(b.fsub(tile_max, m_old), c),
                b.const_f32(_LAZY_RESCALE_THRESHOLD),
            )
            any_out = b.wave_any(b.select(within, c0, c1))
            skip = b.cmp_eq(any_out, c0)
            m_new = b.select(skip, m_old, b.fmax(m_old, tile_max))
        else:
            skip = None
            m_new = b.fmax(m_old, tile_max)
        neg_mc = b.fneg(b.fmul(m_new, c))
        # A row whose running max is still -inf (every visible key so far is
        # masked, or carries a -inf bias / QQ-bias) must not form -inf - -inf.
        # Only the masked tiles and the additive score terms can produce -inf.
        if masked or cfg.bias or cfg.qq_bias:
            neg_mc = b.select(b.fcmp("oeq", m_new, neg_inf), zero_f, neg_mc)
        ps = [[b.exp2_fast(b.fma(v, c, neg_mc)) for v in row] for row in vals]
        alpha = b.exp2_fast(b.fma(m_old, c, neg_mc))
        lsum = tree([p for row in ps for p in row], b.fadd)
        l_scaled = b.fmul(l_old, alpha)
        if skip is not None:
            l_scaled = b.select(skip, l_old, l_scaled)
        l_new = b.fadd(l_scaled, lsum)
        p_ops = [p_operand(ps[ns]) for ns in range(n_kv)]

        alpha_v = b.vector_splat(alpha, 8)
        if skip is not None:
            rl = b.scf_for_iter(
                c0,
                b.select(skip, c0, c1),
                c1,
                iter_args=[(f"ra{d}_{tag}", accs[d]) for d in range(n_dv)],
                iv_name=f"rsc_{tag}",
            )
            with rl as (_r, rs):
                b.scf_yield(*[b.vector_mul(rs[d], alpha_v) for d in range(n_dv)])
            accs = list(rl.results)
        else:
            accs = [b.vector_mul(a, alpha_v) for a in accs]

        # O^T += V^T P^T, V^T from LDS.
        vb = b.add(b.add(buf, ci(k_tile)), v_lane)
        for ns in range(n_kv):
            for dt in range(n_dv):
                vf = lds_frag(b.add(vb, ci(dt * 16 * VS + ns * 16)))
                accs[dt] = b.mma(op_id, vf, p_ops[ns], accs[dt])

        if pair:
            b.sync_lds_only()  # every wave is done reading V and the partials
            store_v(nbuf, nvr)
        else:
            store_k(nbuf, nkr)
            store_v(nbuf, nvr)
            b.sync_lds_only()
        b.scf_yield(m_new, l_new, *accs)

    if cfg.sinks:
        # The sink logit is the initial running max with one unit of
        # denominator mass (counted once, in the lower lane half's partial l).
        m0 = sink_log2 if cfg.log_domain else b.fdiv(sink_log2, scale_log2)
        l0 = b.select(lo_half, b.const_f32(1.0), zero_f)
    else:
        m0, l0 = neg_inf, zero_f
    state = [m0, l0] + [b.zero_vec_f32(8) for _ in range(n_dv)]
    for masked, lo_j, hi_j in ((False, c0, n_full), (True, n_full, total)):
        tag = "msk" if masked else "full"
        loop = b.scf_for_iter(
            lo_j,
            hi_j,
            c1,
            iter_args=[(f"m_{tag}", state[0]), (f"l_{tag}", state[1])]
            + [(f"o{d}_{tag}", state[2 + d]) for d in range(n_dv)],
            iv_name=f"kt_{tag}",
            elide_trailing_barrier=False,
        )
        with loop as (j, st):
            step(j, st, masked)
        state = list(loop.results)

    # ---- epilogue ----
    m_f, l_f = state[0], state[1]
    accs = state[2:]
    l_tot = b.fadd(l_f, permx16_f32(l_f))
    empty = b.fcmp("oeq", l_tot, zero_f)
    inv_l = b.select(empty, zero_f, b.rcp(l_tot))
    o_row_off = b.add(b.mul(q_row, stride_o), dv_base)
    d_lane = b.mul(half, ci(8))
    for dt in range(n_dv):
        v = [b.fmul(b.vec_extract(accs[dt], i), inv_l) for i in range(8)]
        h16 = [b.zext(b.bitcast(b.cast_f32_to(x, elem), I16), I32) for x in v]
        words = [b.lor(h16[2 * m], b.shl(h16[2 * m + 1], ci(16))) for m in range(4)]
        if not gfx12:
            # Slot i holds d = 2i + half; trade with lane^16 so each half owns
            # 8 contiguous columns (d = 8 * half + 0..7).
            send = [b.select(lo_half, words[2 + k], words[k]) for k in (0, 1)]
            keep = [b.select(lo_half, words[k], words[2 + k]) for k in (0, 1)]
            recv = [b.permlanex16(x) for x in send]
            words = []
            for k in (0, 1):
                words.append(b.perm_b32(recv[k], keep[k], sel_lo))
                words.append(b.perm_b32(recv[k], keep[k], sel_hi))
        d0 = b.add(ci(dt * 16), d_lane)
        off = b.mul(b.add(o_row_off, d0), ci(2))
        ok = b.cmp_lt(b.add(d0, dv_base), head_dim_v)
        out = b.vec_bitcast(b.vec_pack(words, I32), VectorType(F16, 8))
        b.buffer_store_vN_f16(o_rsrc, masked_off(off, ok), c0, out, 4)

    if lse_rsrc is not None:
        lse = b.fmul(b.fadd(b.fmul(m_f, c), b.log2(l_tot)), b.const_f32(_LN2))
        lse = b.select(empty, neg_inf, lse)
        off = b.mul(b.mul(q_row, stride_lse), ci(4))
        writer = lo_half if side is None else b.land(lo_half, b.cmp_eq(side, c0))
        b.buffer_store_f32(lse_rsrc, masked_off(off, writer), c0, lse)
