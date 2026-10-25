# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.
"""Shared core of the native-f32-MFMA SDPA backward family (gfx950): lane maps, 4-byte LDS tile layout, MFMA issue, 4-byte global addressing.

Used by `fmha_f32_perwave_gfx950` and `fmha_f32_dsplit_gfx950` (the two operand layers) and by the fp32 dK/dV and dQ kernels built on them.
Softmax (`ParitySoftmaxHelper`), Philox, varlen decode, window and causal-region helpers are re-exported read-only from the existing modules.

--- MFMA lane maps (wave64, one fp32 register per lane for A and for B) -----

  16x16x4   v_mfma_f32_16x16x4_f32   A[m = l%16][k = l//16]   B[k = l//16][n = l%16]   acc i of lane l = D[4 (l//16) + i][l % 16]
  32x32x2   v_mfma_f32_32x32x2_f32   A[m = l%32][k = l//32]   B[k = l//32][n = l%32]   acc i of lane l = D[8 (i//4) + 4 (l//32) + i%4][l % 32]

GPU-verified (job JL: 0/64 mismatches for both shapes); `tools/fp32_mfma_emulation.py` is the host model of the same maps and
`tools/fp32_core_check.py` checks this module against it.

--- LDS tiles ----------------------------------------------------------------

A staged tile is a sequence of 16 x 16 fp32 *chunks* (1 KiB, one 64-lane x 16 B global->LDS DMA: lane l lands at chunk word 4 l, i.e. row l//4,
columns 4 (l%4) .. +3), `CHUNK_STRIDE` = 272 words apart (a pad of `CHUNK_PAD` = 16 words = 64 B per chunk). Offsets are in fp32 words everywhere.

  - **b128 operand read** (the A operand of `S = Q K^T`, `dP = dO V^T`, `S^T = K Q^T`, ...): lane (g = l//16, m = l%16) reads the four d of
    row `q(m)` at column 4 g, `q(m) = m//4 + 4 (m%4)`. MFMA step j of the four then holds column 4 g + j: the lane's A element for contraction
    index k = g at step j. The row permutation makes the accumulator row of lane (g, i) the tile row `g + 4 i`.
  - **b32 column read** (the A operand of `dV^T += dO^T P`, `dK^T += Q^T dS`, `dQ^T += K^T dS`): step i reads word `64 i + l` of the chunk: row
    `g + 4 i`, column m. The 16 lanes of a group read 16 consecutive words: conflict-free on 32 and on 64 banks.
  - **accumulator reuse**: an S (or dP) accumulator, whose lane holds rows `g + 4 i` of column m, is the *next* GEMM's B operand with no repack:
    step i of the contraction (k = g) is the lane's accumulator register i.
  - **accumulator tile in LDS (transpose-on-write)**: `ACC_PITCH` = 17 words per row (a tile is 272 words = one chunk stride). Lane (g, m)
    writes accumulator register i to row `g + 4 i`, column m (row orientation) or row m, column `g + 4 i` (transposed orientation) with `ds_write_b32`;
    lane l then reads row l//4, columns 4 (l%4) .. +3 with four `ds_read_b32` and stores them as one 128-bit global store. Pitch 17 makes the writes
    of both orientations and the reads conflict-free per 16-lane group on 32 and 64 banks.

--- Global addressing --------------------------------------------------------

Elements are 4 bytes; strides and sizes are in elements with unit column stride. Vector accesses are 4 fp32 = 128 bit and need a 16-byte aligned
row (`stride % 4 == 0`, 16-byte aligned base: `variant.alignment_for`). An unaligned tail (padded head dim not a multiple of 4, or a slice past
the head dim) is described by a `ColMask` and handled as follows:

  - resident registers and stores: masked element-wise forms (`ldg_v4_masked`, `stg_v4_masked`);
  - staged tiles: *bounded buffer DMA* (`stage_panel` with `bound`), the 16-bit family's mechanism: the buffer resource's `num_records` ends the
    (batch, head) slab on its last accessed element, `slab_span` = `(rows - 1) * stride_seq + ceil8(hdim)` elements (`_slab_span_elems` of the
    16-bit modules; the 8xD input contract grants `ceil8(hdim)` contiguous elements per row). The bound only protects the end of the slab (those
    lanes read 0); a row's columns past `hdim` read whatever follows in memory (slack, or the next row), so
  - the A operand of every contraction over d (`scores_block`, the dQ `scores`) is masked on the read side (`mask_a`): panels entirely below
    `ColMask.floor` skip it. Column reads (`col_read`) feed output rows d, so their garbage stays in unstored columns.

--- Backends -----------------------------------------------------------------

Every emitter takes an `ops` object. `DeviceOps` emits FlyDSL; `tools/fp32_core_check.py` supplies a host `ops` that runs the same emitters on numpy
lane arrays over the fp32-MFMA emulation, which is how the emitters are checked without a GPU. `ops` offers: `i64`, `imin`, `sel`, `zero`,
`elem`, `vec`, `vadd`, `lds_ld(off, n)`, `lds_st(off, v, pred)`, `dma(name, goff, lds_off, pred)`, `dma_bounded(name, base, off, lds_off, span, pred)`, `ldg(name, goff, n)`, `stg(name, goff, v, pred)`,
`mfma16`, `mfma32`, `sync`, `wait_dma`, `barrier`. `n` is 1 (fp32), 4 or 16 (vector of fp32).
"""

from dataclasses import dataclass
from typing import NamedTuple

from kernels.attention_bwd_cdna.fmha_common_gfx1201 import (  # noqa: E402,F401  read-only shared helpers
    WINDOW_BOTRIGHT,
    WINDOW_TOPLEFT,
    CausalRegions,
    FastMath,
    MaskedAxis,
    cond_load,
    decode_addressing,
    decompose_causal_regions,
    lse_row_addressing,
    lse_row_step_per_head,
    lse_token_pitch,
    make_addr_pair,
    philox_offset_base,
    philox_report,
    philox_seed_value,
    resolve_window,
    seqinfo_addr,
)
from kernels.attention_bwd_cdna.fmha_dualwave_gfx950 import (
    ParitySoftmaxHelper,
)  # noqa: E402,F401
from kernels.attention_bwd_cdna.gfx950_standalone import (
    buffer_ops,
    dualwave,
)  # noqa: E402
from kernels.attention_bwd_cdna.philox import (  # noqa: E402,F401
    Philox,
    default_width,
    dropout_threshold,
    keep_mask,
    philox_4x,
    philox_u32,
    to_uniform_f32,
)

import flydsl.expr as fx  # noqa: E402
from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.dialects import rocdl as R  # noqa: E402
from flydsl._mlir.dialects import scf as _scf  # noqa: E402
from flydsl.expr.rocdl import _blgp_attr  # noqa: E402
from flydsl.expr.typing import Vector as Vec  # noqa: E402

ELEM_BYTES = 4
VEC = 4  # fp32 per 128-bit access
WAVE = 64
LDS_BYTES_LIMIT = 163840

PANEL = 16  # rows and columns of one chunk
CHUNK = PANEL * PANEL  # words
CHUNK_PAD = 16  # words
CHUNK_STRIDE = CHUNK + CHUNK_PAD
ACC_PITCH = (
    PANEL + 1
)  # words per row of an accumulator tile staged for a transposing store
ACC_TILE = PANEL * ACC_PITCH  # == CHUNK_STRIDE

MFMA16 = "16x16x4"
MFMA32 = "32x32x2"
MFMA_SHAPES = {
    MFMA16: dict(M=16, N=16, K=4, nacc=4),
    MFMA32: dict(M=32, N=32, K=2, nacc=16),
}

TILES = (32, 64, 96, 128, 160, 192, 224, 256, 512)
# (BLOCK_M, BLOCK_N) -> q rows of the staged tile (dK/dV `block_q`) / of the workgroup (dQ `block_m`)
GEOMETRIES = {(32, 16): 32, (16, 16): 16}
SCHEDULES = ("pw", "ds")
WAVE_COUNTS = (2, 4)


# ------------------------------------------------------------------ lane maps (polymorphic: python ints, numpy int arrays, fx.Int32)
def a_pos(shape, lane):
    """(m, k) of the A element held in `lane`'s register."""
    n = MFMA_SHAPES[shape]["M"]
    return lane % n, lane // n


def b_pos(shape, lane):
    """(k, n) of the B element held in `lane`'s register."""
    n = MFMA_SHAPES[shape]["N"]
    return lane // n, lane % n


def acc_pos(shape, lane, i):
    """(row, col) of D held in accumulator register `i` of `lane`."""
    if shape == MFMA16:
        return 4 * (lane // 16) + i, lane % 16
    return 8 * (i // 4) + 4 * (lane // 32) + i % 4, lane % 32


def row_perm(m):
    """Tile row fed to A-operand row m: `m//4 + 4 (m%4)`. Accumulator row 4 g + i then holds tile row `g + 4 i`."""
    return m // 4 + 4 * (m % 4)


def acc_tile_row(lane, i):
    """Tile row (q for S, kv for S^T) of accumulator register i of a lane, under `row_perm` on the A operand: `g + 4 i`."""
    return lane // 16 + 4 * i


# ------------------------------------------------------------------ LDS word offsets
def chunk_off(idx):
    """Word offset of chunk `idx` of a staged tile."""
    return idx * CHUNK_STRIDE


def dma_src(lane):
    """(row, col) inside the chunk of the 16 B the lane fetches; its LDS destination is word `4 lane`."""
    return lane // 4, (lane % 4) * 4


def a_read(lane):
    """Word offset in a chunk of the lane's b128 operand read: row `row_perm(m)`, column 4 g."""
    return row_perm(lane % 16) * PANEL + (lane // 16) * 4


def col_read(lane, i):
    """Word offset in a chunk of the lane's step-i b32 column read: row g + 4 i, column m (= word 64 i + lane)."""
    return 64 * i + (lane // 16) * PANEL + lane % 16


def kv_step32(jp):
    """Row offset (before + lane//32) of 32x32x2 contraction step jp (0..7) of a 16-row K tile: `2 (jp//4) + 4 (jp%4)`."""
    return 2 * (jp // 4) + 4 * (jp % 4)


def col_read32(lane, jp):
    """Word offset from the first chunk of a 32-column (two-chunk) unit of the lane's 32x32x2 step-jp b32 column read:
    row `lane//32 + kv_step32(jp)`, column `lane % 16` of chunk `(lane % 32) // 16`."""
    m32 = lane % 32
    return (m32 // 16) * CHUNK_STRIDE + (lane // 32 + kv_step32(jp)) * PANEL + m32 % 16


def acc_tile_write(lane, i, transposed):
    """Word offset in an accumulator tile of register i of the lane: row `g + 4 i`, column m (row orientation), or the transpose."""
    r, c = acc_tile_row(lane, i), lane % 16
    return c * ACC_PITCH + r if transposed else r * ACC_PITCH + c


def acc_tile_read(lane, e):
    """Word offset in an accumulator tile of element e (0..3) of the lane's 128-bit row segment: row l//4, column 4 (l%4) + e.
    (global row, column) offsets inside the 16 x 16 block are `dma_src(lane)`."""
    r, c = dma_src(lane)
    return r * ACC_PITCH + c + e


def chunks_bytes(n):
    """LDS bytes of `n` staged chunks."""
    return n * CHUNK_STRIDE * ELEM_BYTES


# ------------------------------------------------------------------ bank model
def bank_conflict_degree(words, banks, group=16):
    """Worst-case number of lanes of one `group` (consecutive lanes) that fall on one bank for a 4-byte-per-lane LDS access at `words`.
    1 = conflict-free."""
    worst = 1
    for g0 in range(0, len(words), group):
        seen = {}
        for w in words[g0 : g0 + group]:
            seen[int(w) % banks] = seen.get(int(w) % banks, 0) + 1
        worst = max(worst, max(seen.values()))
    return worst


# ------------------------------------------------------------------ family parameters and the Spec round trip
@dataclass(frozen=True)
class Family:
    """One fp32 variant's layout parameters.

    kernel 'dkdv' | 'dq'; `tile` = head dim (hdim rides the tile; tile % 32 == 0); `bq` = q rows of the staged tile (dK/dV `block_q`) or of the
    workgroup (dQ `block_m`), 16 or 32 (geometry (BLOCK_M, BLOCK_N) = (bq, 16)); `sched` 'pw' (per-wave dK/dV: wave owns 16 KV rows x full d)
    or 'ds' (D-split: workgroup owns 16 KV rows (dK/dV) / `bq` q rows (dQ), wave owns a d-slice; dQ is always 'ds'); `waves` 2 or 4.
    """

    kernel: str
    tile: int
    bq: int
    sched: str
    waves: int

    def __post_init__(self):
        if (
            self.kernel not in ("dkdv", "dq")
            or self.bq not in (16, 32)
            or self.sched not in SCHEDULES
            or self.waves not in WAVE_COUNTS
        ):
            raise ValueError(f"family {self}")
        if self.tile not in TILES:
            raise ValueError(f"tile {self.tile}")
        if self.kernel == "dq" and self.sched != "ds":
            raise ValueError("dQ is a d-split at every tile")

    @property
    def rb(self):
        """16-row blocks of the q tile."""
        return self.bq // 16

    @property
    def unit(self):
        """d granule of a slice: 32 for dQ at `block_m` 32 (the 32x32x2 accumulation tile), else 16."""
        return 32 if (self.kernel == "dq" and self.bq == 32) else 16

    @property
    def panels_total(self):
        """16-column panels of the head dim."""
        return self.tile // PANEL

    @property
    def d_wave(self):
        """d per wave (the busiest wave's slice for an uneven split)."""
        if self.sched == "pw":
            return self.tile
        return self.unit * -(-self.tile // (self.unit * self.waves))

    @property
    def dsu(self):
        """Panels per wave slice (resident panels per lane)."""
        return self.d_wave // PANEL

    @property
    def uneven(self):
        """True when `d_wave * waves != tile`: the last wave(s) own fewer (or no) panels; their other panels are masked."""
        return self.sched == "ds" and self.d_wave * self.waves != self.tile

    @property
    def panels_lds(self):
        """Panels of one staged tensor tile in LDS: the head dim for per-wave, `dsu * waves` for a d-split (an uneven split pads the last slices)."""
        return self.panels_total if self.sched == "pw" else self.dsu * self.waves

    @property
    def block(self):
        return WAVE * self.waves

    @property
    def geometry(self):
        return f"{self.bq}x16"

    def wave_panels(self, wave):
        """(first panel, owned panels) of `wave`."""
        if self.sched == "pw":
            return 0, self.panels_total
        p0 = wave * self.dsu
        return p0, max(0, min(self.dsu, self.panels_total - p0))

    def mfma_counts(self):
        """(16x16x4, 32x32x2) MFMAs one wave issues per loop iteration (the busiest wave's, for an uneven split)."""
        d, rb = self.dsu, self.rb
        if self.kernel == "dkdv":
            return rb * d * 4 * 2 + d * rb * 4 * 2, 0
        scores = rb * d * 4 * 2
        return (scores + d * 4, 0) if self.bq == 16 else (scores, (d // 2) * 8)

    def spec_knobs(self):
        """The canonical fp32 knob set the tuning table pins for this variant (schedule and waves are table entries, not Spec keys)."""
        if self.kernel == "dkdv":
            return {"mfma_rows": 16, "block_q": self.bq, "num_stream_buffers": 1}
        return {"block_m": self.bq, "block_n": 16}

    @staticmethod
    def from_spec(spec, sched, waves):
        """Layout parameters of an fp32 dkdv / dq Spec. The geometry knob (`block_q` / `block_m`) must be pinned; the other fp32 knobs, when
        set, must have their only legal value."""
        if spec.dtype != "fp32" or spec.kernel not in ("dkdv", "dq"):
            raise ValueError(f"not an fp32 dkdv/dq spec: {spec.key()}")
        knob = "block_q" if spec.kernel == "dkdv" else "block_m"
        bq = getattr(spec, knob)
        if bq is None:
            raise ValueError(f"{knob} is not pinned: {spec.key()}")
        legal = {"mfma_rows": 16, "num_stream_buffers": 1, "block_n": 16}
        for k, v in legal.items():
            if getattr(spec, k, None) not in (None, v):
                raise ValueError(f"knob {k} must be {v}: {spec.key()}")
        return Family(spec.kernel, spec.tile, bq, sched, waves)


# ------------------------------------------------------------------ lane coordinates
@dataclass(frozen=True)
class Lane:
    """Per-lane coordinates the emitters index with: `lane` (0..63), `wave`, and the 16x16x4 split `g = lane//16`, `m = lane%16`."""

    lane: object
    wave: object
    g: object
    m: object

    @staticmethod
    def of(lane, wave):
        return Lane(lane, wave, lane // 16, lane % 16)


# ------------------------------------------------------------------ global addressing
def gaddr(ops, base, row, stride_seq, col):
    """Element offset (64-bit) of `(row, col)` of a tensor whose row 0 is at `base` elements: `base + row * stride_seq + col`."""
    return base + ops.i64(row) * stride_seq + ops.i64(col)


def vec_aligned(*strides):
    """A 128-bit access is legal when every row stride is a multiple of VEC elements."""
    return all(s % VEC == 0 for s in strides)


class ColMask(NamedTuple):
    """Real column extent of a padded head dim: columns >= `limit` are not data. `whole`: `limit` is a multiple of 4. `floor`: columns below it are
    always real (a lower bound on the head dim, the rung below the tile), so whole panels under it skip read-side masking.
    """

    limit: object
    whole: bool
    floor: int = 0


def slab_span(ops, rows, stride_seq, hdim):
    """Elements of a (batch, head) slab ending on its last accessed element: `(rows - 1) * stride_seq + ceil8(hdim)`, 0 for no rows. The
    `num_records` of a bounded buffer DMA is `4 * slab_span`. Same bound as the 16-bit modules' `_slab_span_elems`.
    """
    r = ops.i64(rows)
    span = (r - 1) * ops.i64(stride_seq) + (ops.i64(hdim) + 7) // 8 * 8
    return ops.sel(r > 0, span, ops.i64(0))


def mask_a(ops, a_v4, p_first, c, g, mask):
    """Zero the elements of a b128 A-operand read (panel `p_first + c`, lane group g) whose column is >= `mask.limit`. Panels entirely below
    `mask.floor` (static `p_first`) are returned as read."""
    if mask is None:
        return a_v4
    if isinstance(p_first, int) and (p_first + c + 1) * PANEL <= mask.floor:
        return a_v4
    col = (p_first + c) * PANEL + g * 4
    zero = ops.zero(1)
    return ops.vec(
        [ops.sel((col + j) < mask.limit, ops.elem(a_v4, j), zero) for j in range(VEC)]
    )


def ldg_v4_masked(ops, name, row_off, col, limit, whole):
    """4 fp32 at columns `col .. col + 3` of the row at element offset `row_off`, zero where the column is >= `limit`. Never reads past the row's
    first `limit` columns. `whole`: `limit` is a multiple of 4 (and `col` is), so a vector is valid or not as a whole; one 128-bit load.
    """
    zero = ops.zero(1)
    if whole:
        ok = col < limit
        v = ops.ldg(name, row_off + ops.i64(ops.sel(ok, col, 0)), 4)
        return ops.vec([ops.sel(ok, ops.elem(v, e), zero) for e in range(VEC)])
    out = []
    for e in range(VEC):
        ok = (col + e) < limit
        x = ops.ldg(name, row_off + ops.i64(ops.sel(ok, col + e, 0)), 1)
        out.append(ops.sel(ok, x, zero))
    return ops.vec(out)


def stg_v4_masked(ops, name, row_off, col, limit, whole, v, pred=None):
    """Store 4 fp32 at columns `col .. col + 3`, only where the column is < `limit` (and `pred`, a row/validity predicate, holds)."""
    if whole:
        ok = col < limit
        ops.stg(name, row_off + ops.i64(col), v, ok if pred is None else (ok & pred))
        return
    for e in range(VEC):
        ok = (col + e) < limit
        ops.stg(
            name,
            row_off + ops.i64(col + e),
            ops.elem(v, e),
            ok if pred is None else (ok & pred),
        )


def stage_row_off(ops, row, row_last, stride_seq, lane):
    """Element offset, from the slab base, of the row the lane fetches when staging chunks of the 16 rows starting at `row`: row `row + lane//4`
    clamped to `row_last` (a row past the sequence end reads the last valid row, whose data the caller's softmax masks). Loop-invariant across
    the panels of a tile: compute once per (tensor, 16-row block)."""
    return ops.i64(ops.imin(row + lane // 4, row_last)) * ops.i64(stride_seq)


def stage_panel(
    ops, name, lds_off, base, row_off, panel_col, lane, bound=None, pred=None
):
    """Stage one 16 x 16 chunk of tensor `name` at LDS word `lds_off`: the lane's 16 B are columns `panel_col + 4 (lane%4) ..` of the row at
    `row_off` (`stage_row_off`) of the slab at element offset `base`. `bound` None: one global->LDS DMA (the row has at least `panel_col + 16`
    columns in memory). Otherwise a bounded buffer DMA through a resource of `4 * bound` bytes over the slab (`bound` = `slab_span`): lanes
    whose 16 B lie past it read 0 and touch no memory. `pred` (wave-uniform) skips the DMA of a chunk the wave does not own.
    """
    off = row_off + ops.i64(panel_col + (lane % 4) * 4)
    if bound is None:
        ops.dma(name, base + off, lds_off, pred)
    else:
        ops.dma_bounded(name, base, off, lds_off, bound, pred)


# ------------------------------------------------------------------ MFMA issue (shared by both operand layers)
def mfma_dot_panel(ops, acc, a_v4, b_v4):
    """acc += A B over one 16-column panel: four 16x16x4 steps, step j contracts column 4 g + j."""
    for j in range(4):
        acc = ops.mfma16(ops.elem(a_v4, j), ops.elem(b_v4, j), acc)
    return acc


def scores_block(ops, L, tq, s, rb, ncols, tile_f, K, V, p_first=0, mq=None, mv=None):
    """S and dP accumulators of 16-row block s of a staged q tile at LDS word `tq` (this wave's first panel), over `ncols` resident panels.

    A = the tile rows `row_perm` read b128 (Q, dO), B = the resident K, V panels: lane (g, i) of the result holds S[tile row 16 s + g + 4 i][kv m].
    Chunk of (panel c, block s) is `c * rb + s`; the dO tile is `tile_f` words after the Q tile. `p_first` = head-dim panel of the first resident panel;
    `mq` / `mv`: `ColMask` of the Q / dO side (A reads masked past the head dim, see the module docstring).
    """
    sa = ops.zero(4)
    da = ops.zero(4)
    a = a_read(L.lane)
    for c in range(ncols):
        off = tq + chunk_off(c * rb + s) + a
        sa = mfma_dot_panel(
            ops, sa, mask_a(ops, ops.lds_ld(off, 4), p_first, c, L.g, mq), K[c]
        )
        da = mfma_dot_panel(
            ops, da, mask_a(ops, ops.lds_ld(off + tile_f, 4), p_first, c, L.g, mv), V[c]
        )
    return sa, da


def accumulate_cols(ops, L, tq, rb, ncols, tile_f, Pm, dS, dK, dV):
    """dV^T += dO^T P and dK^T += Q^T dS over this wave's `ncols` panels: A = the staged tile read as b32 columns (step i = row g + 4 i),
    B = P / dS in the S accumulator layout (`Pm[s][i]`, `dS[s][i]`: lane value for tile row 16 s + g + 4 i). dV[t], dK[t] are 16x16x4
    accumulators: lane (g, i) = dV[kv m][d 16 t + 4 g + i]."""
    for t in range(ncols):
        for s in range(rb):
            for i in range(4):
                off = tq + chunk_off(t * rb + s) + col_read(L.lane, i)
                dV[t] = ops.mfma16(ops.lds_ld(off + tile_f, 1), Pm[s][i], dV[t])
                dK[t] = ops.mfma16(ops.lds_ld(off, 1), dS[s][i], dK[t])


def load_resident_panels(ops, L, name, row_off, p_first, ncols, masked=None):
    """Resident panels of tensor `name`: `ncols` V4 registers, panel c = columns `16 (p_first + c) + 4 g .. +3` of the row at element offset
    `row_off` (lane's KV or q row). `masked` None: plain 128-bit loads. Else `(limit, whole)`: columns >= `limit` read as zero.
    """
    out = []
    for c in range(ncols):
        col = (p_first + c) * PANEL + L.g * 4
        if masked is None:
            out.append(ops.ldg(name, row_off + ops.i64(col), 4))
        else:
            out.append(ldg_v4_masked(ops, name, row_off, col, masked[0], masked[1]))
    return out


def store_resident_panels(ops, L, name, row_off, p_first, regs, masked=None, pred=None):
    """Store accumulator V4s (lane (g, i) = d `16 (p_first + t) + 4 g + i` of the lane's row) as one 128-bit store per panel; columns >= the
    `masked` limit (and rows where `pred` is false) are skipped."""
    for t, v in enumerate(regs):
        col = (p_first + t) * PANEL + L.g * 4
        if masked is None:
            ops.stg(name, row_off + ops.i64(col), v, pred)
        else:
            stg_v4_masked(ops, name, row_off, col, masked[0], masked[1], v, pred)


def acc_tile_to_rows(ops, L, base, accs, transposed):
    """Transpose-on-write of one 16 x 16 accumulator tile through LDS at word `base` (`ACC_TILE` words): write register i of every lane, then
    return the lane's V4 row segment (row l//4, columns 4 (l%4) ..). The caller separates the two phases with `ops.sync()` when waves share
    the region; a one-wave region needs none. Global store: `stg_v4_masked`/`ops.stg` at `dma_src(lane)` of the block.
    """
    for i in range(4):
        ops.lds_st(base + acc_tile_write(L.lane, i, transposed), ops.elem(accs, i))
    return ops.vec([ops.lds_ld(base + acc_tile_read(L.lane, e), 1) for e in range(4)])


# ------------------------------------------------------------------ device backend
def _ir(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


class DeviceOps:
    """FlyDSL backend of the emitters. `lbuf`: pointer to the workgroup's fp32 LDS buffer; `gptrs`: name -> global fp32 pointer
    (`fx.recast_iter(f32 global ptr type, arg)`), indexed by 64-bit element offsets."""

    def __init__(self, lbuf, gptrs):
        self.lbuf = lbuf
        self.g = gptrs

    # integers and predicates
    @staticmethod
    def i64(x):
        return fx.Int64(x)

    @staticmethod
    def imin(a, b):
        return fx.min(a, b)

    @staticmethod
    def sel(cond, a, b):
        return cond.select(
            a if not isinstance(a, int) else fx.Int32(a),
            b if not isinstance(b, int) else fx.Int32(b),
        )

    # values
    @staticmethod
    def zero(n):
        return fx.Float32(0.0) if n == 1 else Vec.filled(n, 0.0, fx.Float32).ir_value()

    @staticmethod
    def elem(v, j):
        return Vec(v)[j]

    @staticmethod
    def vec(elems):
        return Vec.from_elements([_ir(e) for e in elems], fx.Float32).ir_value()

    @staticmethod
    def vadd(a, b):
        return (Vec(a) + Vec(b)).ir_value()

    # LDS
    def lds_ld(self, off, n):
        ty = fx.Float32 if n == 1 else Vec.make_type(n, fx.Float32)
        return fx.ptr_load(self.lbuf + fx.Int32(off), ty)

    def lds_st(self, off, v, pred=None):
        if pred is None:
            fx.ptr_store(_ir(v), self.lbuf + fx.Int32(off))
            return
        op = _scf.IfOp(fx.as_ir_value(pred), results_=[], has_else=False)
        with ir.InsertionPoint(op.then_block):
            fx.ptr_store(_ir(v), self.lbuf + fx.Int32(off))
            _scf.YieldOp([])

    def dma(self, name, goff, lds_off, pred=None):
        if pred is None:
            R.global_load_lds(
                (self.g[name] + goff).llvm_ptr,
                (self.lbuf + fx.Int32(lds_off)).llvm_ptr,
                16,
                0,
            )
            return
        op = _scf.IfOp(fx.as_ir_value(pred), results_=[], has_else=False)
        with ir.InsertionPoint(op.then_block):
            R.global_load_lds(
                (self.g[name] + goff).llvm_ptr,
                (self.lbuf + fx.Int32(lds_off)).llvm_ptr,
                16,
                0,
            )
            _scf.YieldOp([])

    def buf_rsrc(self, name, base, span):
        """Buffer resource over the slab of tensor `name` at element `base`, `num_records` = 4 `span` bytes (built once, outside a loop, for `dma_rsrc`)."""
        addr = fx.Int64(fx.ptrtoint(self.g[name])) + fx.Int64(base) * 4
        return fx.rocdl.get_buffer_rsrc(
            buffer_ops.create_buffer_resource_from_addr(
                _ir(addr), num_records_bytes=_ir(fx.Int64(span) * 4)
            )
        )

    def dma_bounded(self, name, base, off, lds_off, span, pred=None):
        """Buffer DMA of 16 B per lane through a resource over the slab at element `base`, `num_records` = 4 `span` bytes, lane byte offset `4 off`."""
        self.dma_rsrc(self.buf_rsrc(name, base, span), off, lds_off, pred)

    def dma_rsrc(self, rsrc, off, lds_off, pred=None):
        """`dma_bounded` through a resource built by `buf_rsrc`."""
        lds_ptr = (self.lbuf + fx.Int32(lds_off)).llvm_ptr

        def issue():
            fx.rocdl.raw_ptr_buffer_load_lds(
                rsrc,
                lds_ptr,
                fx.Int32(16),
                fx.Int32(off * 4),
                fx.Int32(0),
                fx.Int32(0),
                fx.Int32(1),
            )

        if pred is None:
            issue()
            return
        op = _scf.IfOp(fx.as_ir_value(pred), results_=[], has_else=False)
        with ir.InsertionPoint(op.then_block):
            issue()
            _scf.YieldOp([])

    # global
    def ldg(self, name, goff, n):
        ty = fx.Float32 if n == 1 else Vec.make_type(n, fx.Float32)
        return fx.ptr_load(self.g[name] + goff, ty)

    def stg(self, name, goff, v, pred=None):
        p = self.g[name] + goff
        if pred is None:
            fx.ptr_store(_ir(v), p)
            return
        # an explicit `scf.if` (this module's helpers are not rewritten from Python `if`)
        op = _scf.IfOp(fx.as_ir_value(pred), results_=[], has_else=False)
        with ir.InsertionPoint(op.then_block):
            fx.ptr_store(_ir(v), p)
            _scf.YieldOp([])

    # MFMA
    @staticmethod
    def mfma16(a, b, c):
        return R.mfma_f32_16x16x4f32(
            Vec.make_type(4, fx.Float32), _ir(a), _ir(b), _ir(c), 0, 0, _blgp_attr(0)
        ).result

    @staticmethod
    def mfma32(a, b, c):
        return R.mfma_f32_32x32x2f32(
            Vec.make_type(16, fx.Float32), _ir(a), _ir(b), _ir(c), 0, 0, _blgp_attr(0)
        ).result

    # synchronisation
    @staticmethod
    def sync():
        """Wait for this wave's outstanding global->LDS DMA, then a workgroup barrier."""
        dualwave._waitcnt_vm_n(0)
        fx.gpu.barrier()

    @staticmethod
    def wait_dma():
        """Wait for this wave's outstanding global->LDS DMA (no barrier)."""
        dualwave._waitcnt_vm_n(0)

    @staticmethod
    def barrier():
        fx.gpu.barrier()
