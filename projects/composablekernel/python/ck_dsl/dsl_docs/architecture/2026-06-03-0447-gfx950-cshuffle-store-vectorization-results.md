# gfx950 Deep Fusion: C-Shuffle Store Vectorization Investigation

Status snapshot from 2026-06-03 04:47.

This note records the investigation of vectorizing the C-shuffle LDS
accumulator *stores* (`_stage_accumulators_to_cshuffle_lds`), the follow-up
identified in `2026-06-03-0439-gfx950-conv1-lds-vectorization-results.md`.
The result is a **correct negative**: the MFMA C-fragment layout makes
per-lane stores fundamentally non-coalescable under the row-major LDS
layout that the downstream conv1 vectorized reads require.

## Change

No code change. The investigation determined that the proposed optimization
is blocked by a layout conflict between the store pattern and the read
pattern that would require regressing the conv1 read vectorization (the
prior 28% speedup) to resolve.

## Analysis: MFMA 32x32 C-Fragment Lane Layout

The kernel uses the `f16_32x32x16` MFMA atom (`c_per_lane=16`). From
`helpers/atoms.py` lines 538-548, the C-fragment output mapping is:

```
m_blk = lane / 32    (0 or 1, since wave64 has 64 lanes)
col   = lane % 32    (0..31, CONSTANT across all 16 acc slots)

For accumulator slot i (0..15):
  row = (i // 4) * 8 + m_blk * 4 + (i % 4)
```

Each lane's 16 accumulator elements map to 16 different ROWS at the SAME
COLUMN. Within each group of 4 (i%4 = 0,1,2,3), rows are consecutive:

```
Group 0: i=0..3  -> rows  0+m_blk*4,  1+m_blk*4,  2+m_blk*4,  3+m_blk*4
Group 1: i=4..7  -> rows  8+m_blk*4,  9+m_blk*4, 10+m_blk*4, 11+m_blk*4
Group 2: i=8..11 -> rows 16+m_blk*4, 17+m_blk*4, 18+m_blk*4, 19+m_blk*4
Group 3: i=12..15-> rows 24+m_blk*4, 25+m_blk*4, 26+m_blk*4, 27+m_blk*4
```

## Why Stores Cannot Be Vectorized

The C-shuffle LDS tile is `[tile_m, tile_n]` = `[128, 32]` row-major
(`LdsLayout.cshuffle`, `k_pad=0`, no swizzle). In row-major layout:

```
address(row, col) = row * tile_n + col = row * 32 + col
```

For a group of 4 consecutive-row same-column elements (e.g. i=0..3):

```
addr(row,   col) = row * 32 + col
addr(row+1, col) = row * 32 + col + 32
addr(row+2, col) = row * 32 + col + 64
addr(row+3, col) = row * 32 + col + 96
```

These are **stride-32** in element terms (stride-64 bytes). The GFX9
`ds_write_b{32,64,128}` instructions require CONTIGUOUS bytes. There is
no strided vector LDS write on AMDGPU. Therefore, each accumulator
element requires a separate `ds_write_b16`, which is the current behavior.

## Column-Major Layout Would Fix Stores But Break Reads

A transposed `[tile_n, tile_m]` layout would make same-column
consecutive-row elements contiguous:

```
addr_transposed(row, col) = col * tile_m + row
```

For i=0..3: `col*128+row, col*128+row+1, col*128+row+2, col*128+row+3`
-- contiguous! Could use `ds_write_b64` (4 x f16).

However, the conv1 MFMA operand reads use
`load_smem_frag_contiguous_f16(c0_smem, row, col_base, frag_len=8)` which
reads `smem[row, col_base:col_base+8]` -- a contiguous run of 8 columns
in the same row. In row-major this is contiguous (the entire prior 28%
speedup). In column-major these 8 columns would be at
`col_base*128+row, (col_base+1)*128+row, ...` -- stride-128 apart,
completely destroying the read vectorization.

The store pattern (same col, different rows) is **orthogonal** to the read
pattern (same row, different cols) in the physical LDS address space. No
single flat layout can make both contiguous simultaneously.

## Other Approaches Considered and Rejected

1. **ds_write2_b32 (strided 2-dword write)**: not exposed in the IRBuilder
   and would only save 2x (8 instead of 16 instructions per MFMA) at the
   cost of complex addressing. Marginal benefit vs implementation risk.

2. **Two-phase LDS transpose** (write column-major to temp, barrier, copy
   to row-major): doubles LDS traffic and adds a barrier. Strictly worse
   than 16 scalar stores.

3. **Flat 1D LDS with per-lane packing** (each lane's 16 elements packed
   contiguously): makes stores 4x vectorizable but conv1 reads across
   lanes (8 contiguous columns) become stride-16, destroying read
   vectorization.

4. **Eliminating one cshuffle pass**: both passes are structurally required
   -- conv0 accumulators must be staged for conv1 operand reads, and conv1
   accumulators must be staged for maxpool reads.

## Counters (Baseline, Unchanged)

```
metric                  value       notes
----------------------  ----------  -----------------------------------
MfmaUtil                  8.40 %    (from prior vectorization)
SQ_INSTS_LDS_STORE     11,664,000   unchanged, still scalar
SQ_INSTS_LDS_LOAD       5,832,000   (vectorized in prior change)
LdsBankConflict           0.60 %    negligible
LdsLatency               57.0 cyc
VALUBusy                 63.58 %    VALU is now the gating unit
```

## Interpretation

The C-shuffle store vectorization is blocked by a fundamental layout
conflict: the MFMA 32x32 atom places all per-lane accumulator elements in
a single column (col = lane % 32), spread across different rows. In the
row-major LDS layout required by the downstream vectorized conv1 reads,
same-column different-row elements are strided by `tile_n` -- never
contiguous. No layout change can satisfy both the store-side and read-side
contiguity requirements simultaneously.

This is an inherent property of the MFMA 32x32 C-fragment layout combined
with row-major LDS and a GEMM that consumes the staged tile along the
column dimension.

## Next Bottleneck

With cshuffle store vectorization ruled out, the remaining levers from the
counter baseline are, in priority order:

1. **Cut conv0 im2col address-math VALU.** `VALUBusy = 63.58%` is now the
   gating unit. The conv0 implicit-GEMM generates per-element coordinate
   arithmetic (div/mod for H,W,R,S,C from the GEMM M and K indices) that
   dominates the VALU instruction count (`SQ_INSTS_VALU = 109,900,800`,
   VALU:MFMA ratio = 60.5:1). Reducing this via:
   - Precomputing and caching coordinate offsets in registers
   - Strength-reducing div/mod to shifts/masks where dimensions are
     powers of 2 (C=8 is a power of 2)
   - Combining address terms across loop iterations

2. **Reduce total LDS instruction count** through algorithmic changes to
   the fusion schedule (e.g. tighter M tiles that reduce per-CTA work,
   or fusing the two cshuffle stages when possible).

Success signal: `SQ_INSTS_VALU` falls, `VALUBusy` falls, `MfmaUtil` rises
while correctness stays `bad=0`.

## Reproduce

```text
# verify + bench (unchanged from prior)
<venv>/python -m ck_dsl.examples.gfx950.deep_fused_conv_pool_verify \
  --verify --bench --h 2160 --w 3840 --c 8 --k0 32 --k1 24 \
  --warmup 100 --iters 200

# The investigation was purely analytical; no code change to test.
# The MFMA C-fragment layout analysis is in helpers/atoms.py lines 538-548.
```
