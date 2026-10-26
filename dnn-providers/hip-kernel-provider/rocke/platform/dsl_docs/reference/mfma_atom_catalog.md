# MFMA Atom Catalog

The shipped MFMA atoms live in `helpers.atoms`, grouped into `MFMA_F16_ATOMS`, `MFMA_BF16_ATOMS`, `MFMA_FP8_ATOMS` (fp8 / bf8), and `MFMA_MX_ATOMS` (fp4 / fp6), with the unified `MFMA_ATOMS` catalog covering all of them. Lookup helper: `mfma_atom("f16", m, n, k)` (also accepts `bf16`, `fp8` / `fp8e4m3`, `bf8` / `bf8e5m2`). The bf16, fp8/bf8, and MX atoms below are first-class catalog entries — not just bare `IRBuilder` methods. The RDNA wave32 / WMMA targets use the separate `WmmaAtom` catalog (`WMMA_ATOMS`, looked up via `wmma_atom(...)`); the layouts in this doc are the wave64 MFMA layouts.

These MFMA atoms are the wave64 CDNA path (gfx942 / gfx950). The Python and C++ engines emit byte-identical IR for them.

## Atom Table

| Atom             | (m, n, k) | A per lane  | B per lane  | C per lane   | Output tile | Target  | Notes                              |
|------------------|-----------|------------:|------------:|-------------:|------------:|---------|------------------------------------|
| `f16_4x4x4`      | 4 x 4 x 4  | `<4xhalf>`  | `<4xhalf>`  | `<4xfloat>`  | 4x4 x 16 batches | gfx940+ | 16 independent 4x4 matmuls per wave (small-channel direct conv) |
| `f16_16x16x16`   | 16x16x16 | `<4xhalf>`  | `<4xhalf>`  | `<4xfloat>`  | 16x16        | gfx940+ | Legacy CDNA atom                   |
| `f16_16x16x32`   | 16x16x32 | `<8xhalf>`  | `<8xhalf>`  | `<4xfloat>`  | 16x16        | gfx950+ | K-packed: halves K-loop trip count |
| `f16_32x32x8`    | 32x32x8  | `<4xhalf>`  | `<4xhalf>`  | `<16xfloat>` | 32x32        | gfx940+ | Canonical hero atom                |
| `f16_32x32x16`   | 32x32x16 | `<8xhalf>`  | `<8xhalf>`  | `<16xfloat>` | 32x32        | gfx950+ | K-packed                           |
| `bf16_16x16x16`  | 16x16x16 | `<4xbfloat>` | `<4xbfloat>` | `<4xfloat>` | 16x16       | gfx942+ | Lowers via the `_1k` variant (`<4 x i16>` operands) |
| `bf16_16x16x32`  | 16x16x32 | `<8xbfloat>` | `<8xbfloat>` | `<4xfloat>` | 16x16       | gfx950+ | K-packed bf16                      |
| `bf16_32x32x16`  | 32x32x16 | `<8xbfloat>` | `<8xbfloat>` | `<16xfloat>` | 32x32      | gfx950+ | K-packed bf16 hero                 |
| `fp8_16x16x32` / `bf8_16x16x32` | 16x16x32 | `<8xfp8e4m3>` / `<8xbf8e5m2>` | same | `<4xfloat>`  | 16x16 | gfx942+ | 8 f8 bytes/lane = `<2 x i32>` at the intrinsic boundary |
| `fp8_32x32x16` / `bf8_32x32x16` | 32x32x16 | `<8xfp8e4m3>` / `<8xbf8e5m2>` | same | `<16xfloat>` | 32x32 | gfx942+ | f8/bf8 hero shape                  |
| `fp8_16x16x128`  | 16x16x128 | `<32xfp8e4m3>` | `<32xfp8e4m3>` | `<4xfloat>` | 16x16   | gfx950  | Wide-K f8 throughput ceiling; lowers via the scaled `f8f6f4` intrinsic with the E8M0 scales pinned to 1.0 (numerically equal to a plain unscaled fp8 MFMA) |
| `fp4_16x16x128`  | 16x16x128 | 16 fp4 (i64) | 16 fp4 (i64) | `<4xfloat>` | 16x16    | gfx950  | MX fp4 atom; densest gfx950 MFMA shape |
| `fp6_16x16x96`   | 16x16x96 | 12 fp6      | 12 fp6      | `<4xfloat>`  | 16x16        | gfx950  | MX fp6 atom                        |

The bf16 16x16x16 path uses the `_1k` variant (`mfma.f32.16x16x16bf16.1k` with `<4 x i16>` operands); the plain `mfma.f32.16x16x16.bf16` MFMA intrinsic does not exist on this LLVM target. Attempting to declare it produces an undefined symbol at link time. This is why `helpers/attention.py` provides `mfma_16x16x16_for_dtype` and `mfma_16x16x32_for_dtype` that dispatch by dtype.

The wide-K / MX atoms (`fp8_16x16x128`, `fp4_16x16x128`, `fp6_16x16x96`) are gfx950-only — they map onto the `f8f6f4` / MX-scaled MFMA instructions that only the CDNA4 target exposes.

## Native gfx950 32x32x64 Scaled MFMA

The `helpers.atoms.MfmaAtom` table above is separate from the target catalog's
`mfma_scaled` family. On gfx950, that family exposes 32x32x64 operations for
all 25 ordered FP8/BF8/FP6/BF6/FP4 pairs, with independent A/B E8M0 scales
over K32 and 16 FP32 accumulator values per lane. Select the complete contract
with `ArchTarget.mma.op_for_shape(...)`, then emit it with `IRBuilder.mma(...)`.

`MfmaAtom.emit_scaled()` remains a fixed 16x16x128 wrapper; it does not select
the 32x32x64 operation. The existing `fp8_16x16x128` atom uses `emit()` and
the scaled intrinsic with constant in-instruction scales. The gfx950 FP8
fused-MoE dispatcher uses that atom and applies its FP32 block scales outside
the MFMA. The 32x32x64 family currently has direct GEMM examples.

See [MMA selection, emission and dispatcher paths](mma_metadata.md#selection-ir-emission-and-dispatch),
the [gfx950 GEMM contract](../instances/gfx950_block_scaled_gemm.md), and
[runnable scaled-GEMM examples](../../python/rocke/examples/gfx950/gemm/README.md).

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

The bf16 twin is the bare `IRBuilder.mfma_f32_4x4x4_bf16` op (not an `MfmaAtom`
catalog entry). It uses the same lane layout and lowers through the `_1k`
intrinsic `llvm.amdgcn.mfma.f32.4x4x4bf16.1k` with `<4 x i16>` operands
(`v_mfma_f32_4x4x4_16b_bf16` on gfx950).

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

Logical `tf32` on gfx942 has two native atoms: `mfma_f32_16x16x8_xf32` and
`mfma_f32_32x32x4_xf32`. Both use two TF32 operands per lane for each input,
stored as two I32 words; accumulator sizes are four and sixteen FP32 values.
The [numerical example](../../python/rocke/examples/gfx942/tf32_numerics/README.md)
demonstrates raw input transport and explicit RNE preparation. These atoms do
not alter ordinary FP32 selection or enable TF32 production GEMM dispatch.

On gfx950, which has no native XF32 instruction, logical `tf32` selects two emulated
atoms, `mfma_f32_16x16x32_xf32` and `mfma_f32_32x32x16_xf32`. They take the same TF32
operands (`vec<tf32x8>`, one I32 word per element) and FP32 accumulators (four and
sixteen values), and lower to three BF16 MFMAs on `mfma_f32_16x16x32_bf16` and
`mfma_f32_32x32x16_bf16`; see the
[emulation description](../architecture/xf32_bf16x3_emulation.md).

gfx942 declares two f64 atoms in the arch catalog. Both hold one f64 value per
lane for A and B.

`mfma_f64_16x16x4_f64` uses the fp32 16x16x4 A/B lane maps
(`row`/`col = lane % 16`, `k = lane / 16`). C/D hold four f64 values per lane
in a different output layout from the f32 16x16 atoms:

```text
For i in 0..3:
  row = i * 4 + lane / 16
  col = lane % 16
```

An epilogue written for the f32 16x16 layout stores f64 results to the wrong
rows. Take the layout from the atom's `c_layout()`.

`mfma_f64_4x4x4_f64` computes four independent 4x4x4 products per wave. The
catalog describes the blocks stacked along rows, so A, B, and C/D are each
16x4 and block `blk` occupies rows `blk*4 .. blk*4 + 3`. C/D hold one f64
value per lane (`IRBuilder.mma` returns `<1 x f64>`):

```text
A:   lane = 16*k + 4*blk + m  ->  (blk*4 + m, k)
B:   lane = 16*k + 4*blk + n  ->  (blk*4 + k, n)
C/D: lane = 16*m + 4*blk + n  ->  (blk*4 + m, n)
```
