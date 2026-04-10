# LDS Swizzle for Bank Conflict Elimination

Column-level pair-swap + circular rotation to eliminate 4x LDS bank
conflict serialization during Local Read (`ds_read_b128`). Only column
indices are permuted; row assignments are unchanged.

## Hardware Background (GFX950)

LDS has 64 banks x 4 bytes = 256 bytes per bank row, holding exactly
16 dwordx4 columns per bank row. When a tile row's K dimension spans
fewer than 16 columns, multiple tile rows pack into the same bank row
and reads from different rows at the same column offset hit the same
banks, causing serialization.

Other architectures have 32 banks (128-byte bank rows, 8 dwordx4 columns).

## Parameters

In the code (`LDSSwizzleParams` struct in `LowerTile.cpp`):

```
numColumns      = tileK / (128 / elementBits)     -- dwordx4 chunks per tile row
rowsPerBankRow  = columnsPerBankRow / numColumns   -- tile rows per bank row
elementsPerChunk = 128 / elementBits               -- elements per dwordx4 chunk
columnsPerBankRow = numBanks / 4              -- arch-dependent (16 for GFX950, 8 for 32-bank)
```

Per-lane terms:

```
bankRowIdx = row / rowsPerBankRow    -- which bank row a tile row belongs to
```

Examples (tile rows x columns = 16 dwordx4 per bank row):

```
FP4  macK=128:  4 cols/row x 4 rows/bank-row
FP4  macK=256:  8 cols/row x 2 rows/bank-row
FP8  macK=128:  8 cols/row x 2 rows/bank-row
FP16 macK=128: 16 cols/row x 1 row/bank-row  (no conflicts, swizzle skipped)
FP16 macK=64:   8 cols/row x 2 rows/bank-row
FP16 macK=32:   4 cols/row x 4 rows/bank-row
```

When `numColumns >= columnsPerBankRow`, each tile row fills an entire bank
row and there are no conflicts. The code skips swizzle in this case
(`LDSSwizzleParams::noConflicts()`).

## `ds_read_b128` Thread Phases

The LDS unit processes a `ds_read_b128` from a 64-lane wave in 4 phases,
each executing 16 threads simultaneously. The 16 threads in a phase access
LDS in parallel, so bank conflicts only occur between threads within the
same phase. The phase assignment follows the hardware SIMD layout:

| Phase | Threads                                   |
|-------|-------------------------------------------|
| 0     | T0-3, T12-15, T20-23, T24-27              |
| 1     | T32-35, T44-47, T52-55, T56-59            |
| 2     | T4-7, T8-11, T16-19, T28-31               |
| 3     | T36-39, T40-43, T48-51, T60-63            |

Each `ds_read_b128` reads 16 bytes (4 banks) per thread. With 16 threads
per phase and 64 banks total (16 bank groups of 4), zero bank conflicts
means each thread in a phase accesses a distinct bank group.

Without swizzling, threads within a phase that read the same K-column chunk
hit the same bank group, causing 4x serialization. The swizzle permutes
column assignments so that the 16 threads in each phase spread across all
16 bank groups.

## Permutation: PairSwap + Rotate

The swizzle is decomposed into two composable coordinate graph edges:

**PairSwap**: XORs the column with 1 on even bank rows (self-inverse).

```
bankRowIdx    = row / rowsPerBankRow
swapOnEvenRow = (bankRowIdx ^ 1) & 1
swappedCol    = col ^ swapOnEvenRow
```

**Rotate**: Circular rotation of the column based on bank row index.

```
-- Forward (LoadTiled / write to LDS):
rotation    = numColumns - (bankRowIdx / 2) * 2
rotatedCol  = (col + rotation) & (numColumns - 1)

-- Inverse (LoadLDSTile / read from LDS):
invRotation = (bankRowIdx / 2) * 2
rotatedCol  = (col + invRotation) & (numColumns - 1)
```

Together, PairSwap then Rotate produces `numColumns` distinct column
permutations, enough to differentiate all tile rows sharing a bank row.

## Implementation in rocRoller

Gated by `LDSBankSwizzleMode` kernel option (None, Swizzle). The LoadTiled and
LoadLDSTile sides each insert `PairSwap` and `Rotate` coordinate graph edges. No
new instruction emission or cross-lane operations are needed.

### LoadTiled (global memory to LDS)

During LoadTiled, the column coordinate is
permuted before being used in the Tile decomposition, so each lane
fetches from a different global column, producing a swizzled layout
in LDS.

Graph topology (MATRIX_B, swizzle enabled):

```
  Workitem --Flatten--> {nThrX, nThrY}
  {nThrX, nThrY} --PairSwap--> swappedCol
  {swappedCol, nThrY} --Rotate(fwd)--> grSwizzleNThrX
  grSwizzleNThrX, iThrX --Tile--> iMacX
```

For MATRIX_A, the roles of X/Y are swapped (K is dim 1).

### LoadLDSTile (LDS to VGPRs)

During LoadLDSTile, the element-granularity K-column
index is decomposed into a dwordx4 chunk index and sub-element offset.
The chunk index is un-permuted (inverse Rotate, then PairSwap) so the
wave reads the correct logical element from the swizzled LDS layout.

Graph topology (swizzle enabled):

```
  colCoord --Flatten--> {colChunk, elemInChunk}
  {colChunk, rowCoord} --Rotate(inv)--> rotatedCol
  {rotatedCol, rowCoord} --PairSwap--> rawColChunk
  {rawColChunk, elemInChunk} --Tile--> rawColElem
  ldsTag --Tile--> {rawColElem, rowDim}
```

### Multi-wave workgroups

The row coordinate used by PairSwap and Rotate (`nThrY` for MATRIX_B,
`nThrX` for MATRIX_A) is derived from `Workitem(0)`, which spans the
entire workgroup. In a multi-wave workgroup, lanes from different waves
naturally get different row values, producing different `bankRowIdx`
values and therefore different column permutations. No separate
inter-wave rotation step is needed -- the swizzle handles multiple
waves by construction.

### Verification tables

LoadTiled verification (numColumns=8, rowsPerBankRow=2):

| lane | base_col | bankRowIdx | swap_col | gr_rotation | gr_col | expected   |
|------|----------|------------|----------|-------------|--------|------------|
| 0    | 0        | 0 (even)   | 1        | 8           | 1      | K1 at L0   |
| 1    | 1        | 0 (even)   | 0        | 8           | 0      | K0 at L1   |
| 8    | 0        | 0 (even)   | 1        | 8           | 1      | K1 at L8   |
| 9    | 1        | 0 (even)   | 0        | 8           | 0      | K0 at L9   |
| 16   | 0        | 1 (odd)    | 0        | 8           | 0      | K0 at L16  |
| 17   | 1        | 1 (odd)    | 1        | 8           | 1      | K1 at L17  |
| 32   | 0        | 2 (even)   | 1        | 6           | 7      | K7 at L32  |
| 33   | 1        | 2 (even)   | 0        | 6           | 6      | K6 at L33  |
| 35   | 3        | 2 (even)   | 2        | 6           | 0      | K0 at L35  |
| 48   | 0        | 3 (odd)    | 0        | 6           | 6      | K6 at L48  |
| 50   | 2        | 3 (odd)    | 2        | 6           | 0      | K0 at L50  |
| 56   | 0        | 3 (odd)    | 0        | 6           | 6      | K6 at L56  |
| 63   | 7        | 3 (odd)    | 7        | 6           | 5      | K5 at L63  |

### Non-FP4 data types

The swizzle logic is data-type agnostic -- `numColumns = tileK / (128 / elementBits)`
generalizes across FP4, FP8, FP16, etc. The `noConflicts()` check automatically
skips swizzle when `numColumns >= 16` (e.g., FP16 macK=128).
