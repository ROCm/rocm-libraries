# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.
"""D-split operand layers of the native-f32-MFMA backward (gfx950): dK/dV (`DSplitDkDv`) and dQ (`DSplitDq`).

Both split the head dim over the workgroup's W waves: wave w owns d-slice w, `dsu` 16-column panels `[w dsu, (w + 1) dsu)` (`Family.dsu`; dQ at
`block_m` 32 slices in 32-column units for the 32x32x2 accumulation tile). Resident per lane are the wave's slices only, so the resident set is
d / W instead of d. S and dP contract over d, so each wave's accumulators hold a **partial** sum over its slice: the W partials are exchanged
through LDS and added in wave order (fp32, w = 0 .. W - 1: a fixed order, so the result does not depend on scheduling).

Uneven splits (`Family.uneven`, `dsu * W != d / 16`): the last wave(s) own fewer panels or none. Panels past the head dim are *masked*, not
clamped: resident K/V/Q/dO read as zero there, the LDS panel chunks past the head dim are zeroed once (`zero_pads`) and never staged, the
stores skip them, so every d column is owned and stored exactly once and a masked panel adds exactly +0.0 to the partial sums.

Exchange region (after the staged tiles): `nex` slots (S and dP of each 16-row block), each W x 64 lanes x 4 words. Wave w writes slot j at
`part_base + (j W + w) 256 + 4 lane`; a lane reads W partials of one slot at one position and adds them in wave order.

--- dK/dV (`DSplitDkDv`) -----------------------------------------------------

The workgroup owns 16 KV rows; every wave stages and reads only its own slice of the q tile. Phase order per iteration:

    stage; ops.sync(); scores (partials); exchange (write + ops.sync(), then sum)
    (caller: P, dS)  accumulate; ops.barrier()

After `exchange`, lane (g, i) of block s holds tile row 16 s + g + 4 i, column = KV row m: the same layout as the per-wave schedule's S, dP, so
the softmax-recompute math and the dV / dK accumulation are the per-wave schedule's.

--- dQ (`DSplitDq`) ----------------------------------------------------------

The workgroup owns `block_m` q rows (rb 16-row blocks); Q, dO and dQ slices resident; K, V tiles of 16 KV rows staged per slice. The scores are
transposed (S^T = K Q^T, dP^T = V dO^T): A = the K / V chunk read b128, B = the resident Q / dO panels. Phase order per iteration:

    stage_kv; ops.wait_dma()    (a wave reads back only its own slice before the exchange: no barrier)
    scores (partials); write_partials (+ ops.sync()); ds_operands16 | ds_operands32 (sums)
    (caller: dS)  accumulate16 | accumulate32; ops.barrier()

Output rows of dQ^T are d, columns q: lane (g, i) of a 16x16x4 accumulator holds dQ[q m][d 16 t + 4 g + i] (one 128-bit store per panel); lane
l of a 32x32x2 accumulator holds dQ[q l%32][d 32 t + 8 a + 4 (l//32) + e], register 4 a + e (four 128-bit stores per 32-column unit).
dQ is a d-split at every tile of every geometry, so this class serves all 18 (tile, geometry) pairs.
"""


from kernels.attention_bwd_cdna.fmha_f32_common_gfx950 import (  # noqa: E402
    ColMask,
    CHUNK_STRIDE,
    PANEL,
    WAVE,
    accumulate_cols,
    chunk_off,
    chunks_bytes,
    col_read32,
    gaddr,
    load_resident_panels,
    mask_a,
    mfma_dot_panel,
    scores_block,
    stage_panel,
    stage_row_off,
    stg_v4_masked,
    store_resident_panels,
    a_read,
    col_read,
)

EX = WAVE * 4  # words of one wave's partial of one slot


def _own_mask(fam, m):
    """Column mask of a slice: the caller's padded-head-dim mask, else `(tile, True)` when the split is uneven (panels past the head dim),
    else None."""
    if m is not None:
        return m
    return ColMask(fam.tile, True) if fam.uneven else None


class _Exchange:
    """Partial-sum exchange region shared by both D-split layers. Subclasses define `part_base`, `nex`, `W`."""

    def write_partials(self, ops, L, accs):
        """Write this wave's partial accumulators (`nex` V4) and barrier."""
        for j in range(self.nex):
            ops.lds_st(
                self.part_base + (j * self.W + L.wave) * EX + L.lane * 4, accs[j]
            )
        ops.sync()

    def sum_partials(self, ops, slot, pos):
        """Wave-ordered sum of the W partials of `slot` at per-lane word position `pos` (V4)."""
        tot = None
        for w in range(self.W):
            v = ops.lds_ld(self.part_base + (slot * self.W + w) * EX + pos, 4)
            tot = v if tot is None else ops.vadd(tot, v)
        return tot

    @property
    def exchange_words(self):
        return self.nex * self.W * EX


class DSplitDkDv(_Exchange):
    """D-split dK/dV (`Family.kernel == 'dkdv'`, `sched == 'ds'`). Tensor names: 'Q', 'DO' (staged), 'K', 'V' (resident), 'DK', 'DV'.
    `mq` / `mv`: padded-head-dim masks as in `PerWaveDkDv`."""

    def __init__(self, fam):
        if fam.kernel != "dkdv" or fam.sched != "ds":
            raise ValueError(f"not a D-split dK/dV family: {fam}")
        self.fam = fam
        self.W = fam.waves
        self.rb = fam.rb
        self.dsu = fam.dsu
        self.np_lds = fam.panels_lds
        self.tile_f = self.np_lds * self.rb * CHUNK_STRIDE
        self.part_base = 2 * self.tile_f
        self.nex = 2 * self.rb

    @property
    def lds_words(self):
        return self.part_base + self.exchange_words

    @property
    def lds_bytes(self):
        return chunks_bytes(2 * self.np_lds * self.rb) + self.exchange_words * 4

    @property
    def dma_count(self):
        return self.dsu * self.rb * 2

    def kv_row0(self, bx):
        """First KV row of workgroup `bx` (all waves share it)."""
        return bx * PANEL

    def p0(self, wave):
        """First panel (head-dim panel index and LDS panel index) of `wave`'s slice."""
        return wave * self.dsu

    def slot0(self, wave):
        """LDS word offset of the wave's first panel in a staged tensor tile."""
        return self.p0(wave) * self.rb * CHUNK_STRIDE

    def init_acc(self, ops):
        return [ops.zero(4) for _ in range(self.dsu)], [
            ops.zero(4) for _ in range(self.dsu)
        ]

    def zero_pads(self, ops, L):
        """Zero the chunks of this wave's slice that lie past the head dim (both tensors), wave-uniform predicated. Once, before the loop; a wave
        touches only its own slice."""
        if not self.fam.uneven:
            return
        z = ops.zero(4)
        p0 = self.p0(L.wave)
        for pl in range(self.dsu):
            pad = (p0 + pl) >= self.fam.panels_total
            for r in range(self.rb):
                for t in (0, self.tile_f):
                    ops.lds_st(
                        t + chunk_off((p0 + pl) * self.rb + r) + L.lane * 4, z, pad
                    )

    def load_resident(self, ops, L, kv_row_off_k, kv_row_off_v, mq=None, mv=None):
        """K, V resident slices (`dsu` V4 each) of the lane's KV row; `kv_row_off_*` = `gaddr(base, min(kv0 + m, kv_last), stride, 0)`."""
        p0 = self.p0(L.wave)
        K = load_resident_panels(
            ops, L, "K", kv_row_off_k, p0, self.dsu, _own_mask(self.fam, mq)
        )
        V = load_resident_panels(
            ops, L, "V", kv_row_off_v, p0, self.dsu, _own_mask(self.fam, mv)
        )
        return K, V

    def stage(
        self,
        ops,
        L,
        q0,
        q_last,
        q_base,
        do_base,
        stride_q,
        stride_do,
        mq=None,
        mv=None,
        q_span=None,
        do_span=None,
    ):
        """Stage this wave's slice of the q tile starting at row `q0` (rows clamped to `q_last`): global->LDS DMA, skipped for panels past the head
        dim; with a padded-head-dim mask `mq` / `mv` a bounded buffer DMA over the slab (`q_span` / `do_span` = `slab_span`).
        """
        p0 = self.p0(L.wave)
        qro = [
            stage_row_off(ops, q0 + PANEL * r, q_last, stride_q, L.lane)
            for r in range(self.rb)
        ]
        doro = [
            stage_row_off(ops, q0 + PANEL * r, q_last, stride_do, L.lane)
            for r in range(self.rb)
        ]
        for pl in range(self.dsu):
            p = p0 + pl
            owned = (p < self.fam.panels_total) if self.fam.uneven else None
            for r in range(self.rb):
                idx = p * self.rb + r
                stage_panel(
                    ops,
                    "Q",
                    chunk_off(idx),
                    q_base,
                    qro[r],
                    p * PANEL,
                    L.lane,
                    q_span if mq is not None else None,
                    owned,
                )
                stage_panel(
                    ops,
                    "DO",
                    self.tile_f + chunk_off(idx),
                    do_base,
                    doro[r],
                    p * PANEL,
                    L.lane,
                    do_span if mv is not None else None,
                    owned,
                )

    def scores(self, ops, L, K, V, mq=None, mv=None):
        """Partial (S[s], dP[s]) over the wave's slice."""
        S, Pd = [], []
        for s in range(self.rb):
            sa, da = scores_block(
                ops,
                L,
                self.slot0(L.wave),
                s,
                self.rb,
                self.dsu,
                self.tile_f,
                K,
                V,
                self.p0(L.wave),
                mq,
                mv,
            )
            S.append(sa)
            Pd.append(da)
        return S, Pd

    def exchange(self, ops, L, S, Pd):
        """Sum the W partials: returns (S[s], dP[s]) complete, in the per-wave schedule's accumulator layout."""
        self.write_partials(ops, L, S + Pd)
        tot = [self.sum_partials(ops, j, L.lane * 4) for j in range(self.nex)]
        return tot[: self.rb], tot[self.rb :]

    def accumulate(self, ops, L, Pm, dS, dK, dV):
        accumulate_cols(
            ops, L, self.slot0(L.wave), self.rb, self.dsu, self.tile_f, Pm, dS, dK, dV
        )

    def store(
        self,
        ops,
        L,
        kv_row,
        kv_len,
        dk_base,
        dv_base,
        stride_dk,
        stride_dv,
        dK,
        dV,
        mq=None,
        mv=None,
    ):
        """Store the wave's dK, dV slices of the lane's KV row `kv_row + m` (skipped when >= `kv_len`)."""
        p0 = self.p0(L.wave)
        row = kv_row + L.m
        ok = row < kv_len
        store_resident_panels(
            ops,
            L,
            "DK",
            gaddr(ops, dk_base, row, stride_dk, 0),
            p0,
            dK,
            _own_mask(self.fam, mq),
            ok,
        )
        store_resident_panels(
            ops,
            L,
            "DV",
            gaddr(ops, dv_base, row, stride_dv, 0),
            p0,
            dV,
            _own_mask(self.fam, mv),
            ok,
        )


class DSplitDq(_Exchange):
    """D-split dQ (`Family.kernel == 'dq'`). Tensor names: 'Q', 'DO' (resident), 'K', 'V' (staged), 'DQ' (stored), plus 'LSE', 'DELTA' for
    `row_stats`. `mq`: QK-side mask (Q, K, dQ), `mv`: VO-side mask (dO, V)."""

    def __init__(self, fam):
        if fam.kernel != "dq":
            raise ValueError(f"not a dQ family: {fam}")
        self.fam = fam
        self.W = fam.waves
        self.rb = fam.rb
        self.bq = fam.bq
        self.dsu = fam.dsu
        self.np_lds = fam.panels_lds
        self.tile_f = (
            self.np_lds * CHUNK_STRIDE
        )  # one staged tensor: this layer stages 16 KV rows, one chunk per panel
        self.part_base = 2 * self.tile_f
        self.nex = 2 * self.rb

    @property
    def lds_words(self):
        return self.part_base + self.exchange_words

    @property
    def lds_bytes(self):
        return chunks_bytes(2 * self.np_lds) + self.exchange_words * 4

    @property
    def dma_count(self):
        return self.dsu * 2

    def p0(self, wave):
        return wave * self.dsu

    def slot0(self, wave):
        return self.p0(wave) * CHUNK_STRIDE

    def q_row(self, L):
        """Offset from the workgroup's first q row of the q row whose accumulator column / B-operand column the lane holds: m at `block_m` 16,
        `lane % 32` at 32."""
        return L.m if self.bq == 16 else L.lane % 32

    def init_acc(self, ops):
        if self.bq == 16:
            return [ops.zero(4) for _ in range(self.dsu)]
        return [ops.zero(16) for _ in range(self.dsu // 2)]

    def zero_pads(self, ops, L):
        """Zero the K / V chunks of this wave's slice that lie past the head dim, wave-uniform predicated. Once, before the loop."""
        if not self.fam.uneven:
            return
        z = ops.zero(4)
        p0 = self.p0(L.wave)
        for pl in range(self.dsu):
            pad = (p0 + pl) >= self.fam.panels_total
            for t in (0, self.tile_f):
                ops.lds_st(t + chunk_off(p0 + pl) + L.lane * 4, z, pad)

    def load_resident(self, ops, L, q_off, do_off, mq=None, mv=None):
        """Q, dO resident slices: `Qres[s][c]`, `DOres[s][c]` (V4) for the `rb` 16-row blocks and `dsu` panels. `q_off(s)` / `do_off(s)` =
        `gaddr(base, min(q0 + 16 s + m, q_last), stride, 0)`."""
        p0 = self.p0(L.wave)
        Q = [
            load_resident_panels(
                ops, L, "Q", q_off(s), p0, self.dsu, _own_mask(self.fam, mq)
            )
            for s in range(self.rb)
        ]
        DO = [
            load_resident_panels(
                ops, L, "DO", do_off(s), p0, self.dsu, _own_mask(self.fam, mv)
            )
            for s in range(self.rb)
        ]
        return Q, DO

    def row_stats(self, ops, L, q0, q_last, base, pitch):
        """(lse, delta) of the lane's q row `q0 + q_row(L)` (clamped to `q_last`): the B-operand column of the dS operands. `base` / `pitch`:
        `fmha.lse_row_addressing`'s (element offset, row pitch)."""
        row = ops.imin(q0 + self.q_row(L), q_last)
        off = ops.i64(base) + ops.i64(row) * ops.i64(pitch)
        return ops.ldg("LSE", off, 1), ops.ldg("DELTA", off, 1)

    def stage_kv(
        self,
        ops,
        L,
        k0,
        k_last,
        k_base,
        v_base,
        stride_k,
        stride_v,
        mq=None,
        mv=None,
        k_span=None,
        v_span=None,
        k_rsrc=None,
        v_rsrc=None,
    ):
        """Stage this wave's K / V slice of the 16 KV rows starting at `k0` (rows clamped to `k_last`); with `mq` / `mv` a bounded buffer DMA
        over the slab: through a resource built per panel from `k_span` / `v_span` (`slab_span`), or through `k_rsrc` / `v_rsrc` (`ops.buf_rsrc`,
        built once outside the loop; they win over the spans)."""
        p0 = self.p0(L.wave)
        kro = stage_row_off(ops, k0, k_last, stride_k, L.lane)
        vro = stage_row_off(ops, k0, k_last, stride_v, L.lane)
        for pl in range(self.dsu):
            p = p0 + pl
            owned = (p < self.fam.panels_total) if self.fam.uneven else None
            if k_rsrc is not None:
                ops.dma_rsrc(
                    k_rsrc,
                    kro + ops.i64(p * PANEL + (L.lane % 4) * 4),
                    chunk_off(p),
                    owned,
                )
                ops.dma_rsrc(
                    v_rsrc,
                    vro + ops.i64(p * PANEL + (L.lane % 4) * 4),
                    self.tile_f + chunk_off(p),
                    owned,
                )
                continue
            stage_panel(
                ops,
                "K",
                chunk_off(p),
                k_base,
                kro,
                p * PANEL,
                L.lane,
                k_span if mq is not None else None,
                owned,
            )
            stage_panel(
                ops,
                "V",
                self.tile_f + chunk_off(p),
                v_base,
                vro,
                p * PANEL,
                L.lane,
                v_span if mv is not None else None,
                owned,
            )

    def scores(self, ops, L, Q, DO, mq=None, mv=None):
        """Partial (S^T[s], dP^T[s]) per 16-row q block: lane (g, i) of block s holds KV row g + 4 i, q column 16 s + m. Each K / V chunk is read
        once and feeds every block."""
        S = [ops.zero(4) for _ in range(self.rb)]
        Pd = [ops.zero(4) for _ in range(self.rb)]
        a = a_read(L.lane)
        for c in range(self.dsu):
            off = self.slot0(L.wave) + chunk_off(c) + a
            p = self.p0(L.wave)
            ka = mask_a(ops, ops.lds_ld(off, 4), p, c, L.g, mq)
            va = mask_a(ops, ops.lds_ld(off + self.tile_f, 4), p, c, L.g, mv)
            for s in range(self.rb):
                S[s] = mfma_dot_panel(ops, S[s], ka, Q[s][c])
                Pd[s] = mfma_dot_panel(ops, Pd[s], va, DO[s][c])
        return S, Pd

    def ds_operands16(self, ops, L, S, Pd):
        """(`block_m` 16) Exchange, then (S sum, dP sum) V4 at the lane's own position: element i = KV row g + 4 i, q column m, which is
        the B operand of `accumulate16` after the caller's dS = f(S, dP, lse, delta)."""
        self.write_partials(ops, L, S + Pd)
        return self.sum_partials(ops, 0, L.lane * 4), self.sum_partials(
            ops, 1, L.lane * 4
        )

    def ds_operands32(self, ops, L, S, Pd):
        """(`block_m` 32) Exchange, then the 32x32x2 B-operand layout: for h in 0, 1 the lane (q = l%32, g32 = l//32) re-reads the sums at the
        accumulator position of source lane `(2 h + g32) * 16 + (l % 16)` of q block `(l % 32) // 16`; element i of pair h is contraction step
        4 h + i (KV row 2 h + g32 + 4 i). Returns [(S sum, dP sum)] for h = 0, 1."""
        self.write_partials(ops, L, S + Pd)
        m32 = L.lane % 32
        g32 = L.lane // 32
        out = []
        for h in range(2):
            pos = ((g32 + 2 * h) * PANEL + m32 % PANEL) * 4
            out.append(
                (
                    self.sum_partials(ops, m32 // PANEL, pos),
                    self.sum_partials(ops, m32 // PANEL + self.rb, pos),
                )
            )
        return out

    def accumulate16(self, ops, L, acc, dS):
        """acc[t] += K^T dS over the wave's panels: dS[i] = lane's dS for KV row g + 4 i, q column m; A = the K chunk read as b32 columns."""
        for t in range(self.dsu):
            for i in range(4):
                acc[t] = ops.mfma16(
                    ops.lds_ld(
                        self.slot0(L.wave) + chunk_off(t) + col_read(L.lane, i), 1
                    ),
                    dS[i],
                    acc[t],
                )

    def accumulate32(self, ops, L, acc, dS):
        """acc[u] += K^T dS over the wave's 32-column units: eight 32x32x2 steps each; dS[jp] (jp = 4 h + i) per `ds_operands32`."""
        for u in range(self.dsu // 2):
            for jp in range(8):
                ka = ops.lds_ld(
                    self.slot0(L.wave) + chunk_off(2 * u) + col_read32(L.lane, jp), 1
                )
                acc[u] = ops.mfma32(ka, dS[jp], acc[u])

    def store(self, ops, L, q0, q_len, dq_base, stride_dq, acc, mq=None):
        """Store the wave's dQ slice of the lane's q row (skipped when >= `q_len`): one 128-bit store per panel (`block_m` 16) or four per 32-column unit."""
        p0 = self.p0(L.wave)
        m = _own_mask(self.fam, mq)
        row = q0 + self.q_row(L)
        ok = row < q_len
        row_off = gaddr(ops, dq_base, row, stride_dq, 0)
        if self.bq == 16:
            store_resident_panels(ops, L, "DQ", row_off, p0, acc, m, ok)
            return
        g32 = L.lane // 32
        for u, v in enumerate(acc):
            for a in range(4):
                col = (p0 + 2 * u) * PANEL + 8 * a + 4 * g32
                piece = ops.vec([ops.elem(v, 4 * a + e) for e in range(4)])
                if m is None:
                    ops.stg("DQ", row_off + ops.i64(col), piece, ok)
                else:
                    stg_v4_masked(ops, "DQ", row_off, col, m[0], m[1], piece, ok)

    def load_elems(self, ops, name, row_off, cols, col_last):
        """One fp32 per element of the lane's current stage from tensor `name`: element e at column `cols[e]` (clamped to `col_last`, so the access
        stays inside the row) of the row at element offset `row_off`. The element order is the caller's dS order: the bias input of the score
        elements."""
        return [
            ops.ldg(name, row_off + ops.i64(ops.imin(c, col_last)), 1) for c in cols
        ]

    def store_ds(self, ops, L, name, row_off, row_ok, cols, vals, k_len):
        """dB = dS of the lane's current stage: element e (value `vals[e]`, KV column `cols[e]` of the lane's q row at element offset `row_off`) is
        stored where `row_ok` holds and `cols[e] < k_len`. The W waves hold the same sums and compute the same dS, so wave w stores the elements
        with `e % W == w`: every element once."""
        for e, (c, v) in enumerate(zip(cols, vals)):
            ops.stg(
                name,
                row_off + ops.i64(c),
                v,
                row_ok & (c < k_len) & (L.wave == e % self.W),
            )
