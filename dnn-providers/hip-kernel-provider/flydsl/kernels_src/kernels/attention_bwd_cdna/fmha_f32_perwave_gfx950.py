# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.
"""Per-wave dK/dV operand layer of the native-f32-MFMA backward (gfx950).

Schedule 'pw': each wave owns 16 KV rows and the **full** head dim d, so K, V, dK and dV are resident (d fp32 per lane each: `4 * d / 16`
V4 registers... d/4 VGPR per tensor... see `fmha_f32_common_gfx950` for the lane maps); the workgroup of W waves owns 16 W KV rows and stages
one q tile (`block_q` = 16 or 32 rows) of Q and dO that all its waves read. No cross-wave exchange: S = Q K^T and dP = dO V^T are complete in
the wave's accumulators.

Per-iteration phase order (the caller owns the loop, the softmax-recompute math between the GEMMs and the sequence/mask logic):

    stage                       global -> LDS, panels split across the waves
    ops.sync()                  DMA wait + barrier
    scores                      S, dP accumulators (rows g + 4 i of the q tile, column = this lane's KV row m)
    (caller: P, dS in the S accumulator layout)
    accumulate                  dV += dO^T P, dK += Q^T dS
    ops.barrier()               the next stage overwrites the tile

LDS: two tensor tiles (Q, dO), each `panels * rb` chunks, chunk of (panel p, 16-row block s) = `p * rb + s`; `lds_bytes` = 2 tile_f * 4.
Tiles of the family for which this layer is assigned: see `RP/parity/gaps/fp32/sp-n.md` "Schedule table" (12 of 18 (tile, geometry) pairs;
t512 has no per-wave variant that qualifies).
"""


from kernels.attention_bwd_cdna.fmha_f32_common_gfx950 import (  # noqa: E402
    CHUNK_STRIDE,
    PANEL,
    accumulate_cols,
    chunks_bytes,
    chunk_off,
    gaddr,
    load_resident_panels,
    scores_block,
    stage_panel,
    stage_row_off,
    store_resident_panels,
)


class PerWaveDkDv:
    """Operand layer of one per-wave dK/dV variant. All methods are emitters over an `ops` backend (see `fmha_f32_common_gfx950`) except the
    plain-integer properties.

    Tensor names an `ops` carries: 'Q', 'DO' (staged), 'K', 'V' (resident), 'DK', 'DV' (stored). Offsets `*_base` are element offsets of row 0
    of the (batch, head, sequence) tensor; strides are in elements. `mq` / `mv`: None, or `(limit, whole)` = the real column count of the
    QK-side (Q, K, dK) / VO-side (dO, V, dV) tensors for a padded head dim and whether it is a multiple of 4.
    """

    def __init__(self, fam):
        if fam.kernel != "dkdv" or fam.sched != "pw":
            raise ValueError(f"not a per-wave dK/dV family: {fam}")
        self.fam = fam
        self.W = fam.waves
        self.rb = fam.rb
        self.np = fam.panels_total
        self.tile_f = (
            self.np * self.rb * CHUNK_STRIDE
        )  # words of one staged tensor tile

    @property
    def lds_words(self):
        return 2 * self.tile_f

    @property
    def lds_bytes(self):
        return chunks_bytes(2 * self.np * self.rb)

    @property
    def stage_rounds(self):
        """Panels a wave stages per tensor and 16-row block: the panels are split over the W waves, round i of wave w is panel
        `min(w + i W, np - 1)` (the clamped tail restages the last panel with identical bytes).
        """
        return -(-self.np // self.W)

    @property
    def dma_count(self):
        return self.stage_rounds * self.rb * 2

    def kv_row0(self, bx, wave):
        """First KV row of `wave` of workgroup `bx` (before sequence offset)."""
        return (bx * self.W + wave) * PANEL

    def init_acc(self, ops):
        """dK, dV accumulators: np V4 zeros each."""
        return [ops.zero(4) for _ in range(self.np)], [
            ops.zero(4) for _ in range(self.np)
        ]

    def load_resident(
        self,
        ops,
        L,
        kv_row,
        kv_last,
        k_base,
        v_base,
        stride_k,
        stride_v,
        mq=None,
        mv=None,
    ):
        """K, V resident panels of the lane's KV row `kv_row + m` (clamped to `kv_last`): `np` V4 each."""
        row = ops.imin(kv_row + L.m, kv_last)
        K = load_resident_panels(
            ops, L, "K", gaddr(ops, k_base, row, stride_k, 0), 0, self.np, mq
        )
        V = load_resident_panels(
            ops, L, "V", gaddr(ops, v_base, row, stride_v, 0), 0, self.np, mv
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
        """Stage the q tile starting at row `q0` of Q and dO (rows clamped to `q_last`): global->LDS DMA; with a padded-head-dim mask `mq` / `mv`
        a bounded buffer DMA over the slab (`q_span` / `do_span` = `slab_span`)."""
        qro = [
            stage_row_off(ops, q0 + PANEL * r, q_last, stride_q, L.lane)
            for r in range(self.rb)
        ]
        doro = [
            stage_row_off(ops, q0 + PANEL * r, q_last, stride_do, L.lane)
            for r in range(self.rb)
        ]
        for i in range(self.stage_rounds):
            p = ops.imin(L.wave + i * self.W, self.np - 1)
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
                )

    def scores(self, ops, L, K, V, mq=None, mv=None):
        """(S[s], dP[s]) for the `rb` 16-row blocks of the staged tile; lane (g, i) of block s holds tile row 16 s + g + 4 i, column = KV row m."""
        S, Pd = [], []
        for s in range(self.rb):
            sa, da = scores_block(
                ops, L, 0, s, self.rb, self.np, self.tile_f, K, V, 0, mq, mv
            )
            S.append(sa)
            Pd.append(da)
        return S, Pd

    def accumulate(self, ops, L, Pm, dS, dK, dV):
        """dV[t] += dO^T P, dK[t] += Q^T dS in place (see `accumulate_cols`)."""
        accumulate_cols(ops, L, 0, self.rb, self.np, self.tile_f, Pm, dS, dK, dV)

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
        """Store dK, dV of the lane's KV row `kv_row + m` (skipped when >= `kv_len`): one 128-bit store per panel."""
        row = kv_row + L.m
        ok = row < kv_len
        store_resident_panels(
            ops, L, "DK", gaddr(ops, dk_base, row, stride_dk, 0), 0, dK, mq, ok
        )
        store_resident_panels(
            ops, L, "DV", gaddr(ops, dv_base, row, stride_dv, 0), 0, dV, mv, ok
        )
