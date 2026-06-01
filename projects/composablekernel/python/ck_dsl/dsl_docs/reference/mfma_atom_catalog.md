# MFMA Atom Catalog

The shipped f16 atoms are exported as `helpers.atoms.MFMA_F16_ATOMS`. Lookup helper: `mfma_atom("f16", m, n, k)`. The `IRBuilder` also exposes bf16 MFMA methods that are not yet in the catalog.

## Atom Table

| Atom             | (m, n, k) | A per lane  | B per lane  | C per lane   | Output tile | Target  | Notes                              |
|------------------|-----------|------------:|------------:|-------------:|------------:|---------|------------------------------------|
| `f16_4x4x4`      | 4 x 4 x 4  | `<4xhalf>`  | `<4xhalf>`  | `<4xfloat>`  | 4x4 x 16 batches | gfx940+ | 16 independent 4x4 matmuls per wave (small-channel direct conv) |
| `f16_16x16x16`   | 16x16x16 | `<4xhalf>`  | `<4xhalf>`  | `<4xfloat>`  | 16x16        | gfx940+ | Legacy CDNA atom                   |
| `f16_16x16x32`   | 16x16x32 | `<8xhalf>`  | `<8xhalf>`  | `<4xfloat>`  | 16x16        | gfx950+ | K-packed: halves K-loop trip count |
| `f16_32x32x8`    | 32x32x8  | `<4xhalf>`  | `<4xhalf>`  | `<16xfloat>` | 32x32        | gfx940+ | Canonical hero atom                |
| `f16_32x32x16`   | 32x32x16 | `<8xhalf>`  | `<8xhalf>`  | `<16xfloat>` | 32x32        | gfx950+ | K-packed                           |

bf16 MFMA methods exposed by `IRBuilder` (not in `MFMA_F16_ATOMS`):

```text
mfma_f32_16x16x16_bf16   # gfx950 lowers via *_1k variant with <4 x i16> operands
mfma_f32_16x16x32_bf16   # operands <8 x bfloat>
```

The bf16 16x16x16 path uses the `_1k` variant introduced for CDNA2; the plain `mfma.f32.16x16x16.bf16` intrinsic does not exist on this LLVM target. Attempting to declare it produces an undefined symbol at link time. This is why `helpers/attention.py` provides `mfma_16x16x16_for_dtype` and `mfma_16x16x32_for_dtype` that dispatch by dtype.

## Lane Layouts (Output)

The kernel epilogue must agree with `MfmaAtom.lane_to_output(b, lane, i)`. The layouts below come from `helpers/atoms.py`.

### `16x16x16`, `16x16x32`

```text
c_per_lane = 4
m_blk      = lane / 16        # 0..3
n_in_atom  = lane % 16        # 0..15

For i in 0..3:
  row = m_blk * 4 + i
  col = n_in_atom
```

### `32x32x8`, `32x32x16`

```text
c_per_lane = 16
m_blk      = lane / 32        # 0 or 1
n_in_atom  = lane % 32        # 0..31

For i in 0..15:
  row = (i // 4) * 8 + m_blk * 4 + (i % 4)
  col = n_in_atom
```

### `4x4x4`

```text
c_per_lane     = 4
batch_idx      = lane / 4     # 0..15  (one of 16 independent 4x4 batches)
lane_in_batch  = lane % 4     # 0..3

For i in 0..3:
  row = i
  col = lane_in_batch
```

The batch index is composed by the caller. `lane_to_output` returns only the in-atom `(row, col)`.

## Per-Lane K-Slice Layouts (Input A)

For each f16 atom on wave64, the per-lane A operand covers the following K slice:

### `16x16x16`

```text
k_blk = lane / 16  in {0,1,2,3}
A lane holds K = [k_blk*4 : k_blk*4 + 4]    # 4 contiguous halves
```

### `16x16x32`

```text
c4 = lane / 16    in {0,1,2,3}
A lane holds K = [c4*8 : c4*8 + 8]          # 8 contiguous halves
```

**Important**: the K-packed layout is contiguous, not flat-concat. Packing as `[c4*4 : c4*4 + 4] + [c4*4 + 16 : c4*4 + 20]` compiles, runs, validates within ~1e-2, and fails at 1e-3.

### `32x32x8`

```text
k_blk = lane / 32  in {0, 1}
A lane holds K = [k_blk*4 : k_blk*4 + 4]
```

### `32x32x16`

```text
c2 = lane / 32  in {0, 1}
A lane holds K = [c2*8 : c2*8 + 8]
```

### `4x4x4`

```text
batch_idx     = lane / 4
lane_in_batch = lane % 4
A holds the 4 K-elements of row `lane_in_batch` of A in batch `batch_idx`.
B holds the 4 K-elements of column `lane_in_batch` of B in batch `batch_idx`.
```

## VGPR Budget Implications

The accumulator vector width `c_per_lane` is the primary VGPR-pressure knob:

| Atom         | acc per lane | mfmas_per_warp = M x N | acc VGPRs per warp tile  |
|--------------|-------------:|------------------------|--------------------------|
| 16x16x*      |  4 floats    | (block_m/warp_m)/16 x (block_n/warp_n)/16 | `4 * mfmas` |
| 32x32x*      | 16 floats    | (block_m/warp_m)/32 x (block_n/warp_n)/32 | `16 * mfmas` |
| 4x4x4        |  4 floats    | per-batch, 1 per group  | `4 * (groups / wave)` |

A `tile_m=128, tile_n=128, warp_m=2, warp_n=2, warp_tile_m=32, warp_tile_n=32` GEMM with 32x32x16 atom has `mfmas_per_warp = (128/2)/32 x (128/2)/32 = 2x2 = 4` per warp tile, so 64 floats per lane. That's 64 VGPRs just for the accumulator before A/B fragment registers and address arithmetic.

## When To Switch Atoms

- `16x16x16 -> 16x16x32`: halves K-loop trips at the cost of 2x wider A/B per-lane fragment. Same accumulator footprint. Best when memory was already hidden and the K loop was MFMA-throughput-limited.
- `32x32x8 -> 32x32x16`: same trade for the big atom. Already-large accumulator; worth checking VGPR count after the change.
- `16x16x* -> 32x32x*`: 4x more output per atom; 4x larger accumulator per lane. Best when occupancy was already low and a single warp's output tile is small; can push VGPR over the limit if not paired with `waves_per_eu` re-tuning.
- `32x32x* -> 4x4x4`: only for small-channel grouped patterns (direct grouped conv 4c).

## Failure Modes

- Switching atoms without updating `lane_to_output` in the epilogue: silent correctness bug.
- K-packed lane order packed as flat-concat: numerically close, not bit-correct.
- Using `f16_16x16x32` or `f16_32x32x16` on gfx940/gfx942: link-time failure (intrinsic absent).
- Using bf16 atoms without going through `mfma_16x16x16_for_dtype`: declaration mismatch (plain `bf16.16x16x16` doesn't exist; must use the `_1k` variant).
- `4x4x4` epilogue stores not compositing `batch_idx` into the per-group offset: 16 batches collide on the same outputs.
