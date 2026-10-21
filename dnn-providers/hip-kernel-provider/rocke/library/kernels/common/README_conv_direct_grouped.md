# Direct Grouped Convolution

This file covers two directions, both built from `conv_direct_grouped.py`:

| Direction | Entry point | Section |
|-----------|-------------|---------|
| Forward (`D = A ⊛ B`) | `DirectConvSpec` / `DirectConv{4,8,16,32}cSpec` / `DirectDepthwiseSpec` | [cpg Variants](#cpg-variants) |
| Backward-weights (`dW = dY ⊛ X`) | `DirectConvWgradSpec` | [Backward Weights](#backward-weights--directconvwgradspec) |

## Algorithm

The direct grouped convolution kernel computes:

```
D[n, ho, wo, k] = sum_{y, x, c} A[n, ho - pH + y, wo - pW + x, c_group(k) + c] * B[k, y, x, c]
```

where `c_group(k) = (k // kpg) * cpg` selects the input-channel slab for group
`k // kpg`, and `cpg = kpg = C / groups = K / groups`.

Unlike the implicit-GEMM kernel, which maps the whole convolution to a tiled
matrix multiplication, the direct kernel iterates over the spatial filter
window `(y, x)` in the outer loop and accumulates contributions from all input
channels `c` within the group using MFMA or scalar FMA.  This avoids the
coordinate-transform descriptor overhead but constrains the number of supported
channels-per-group.

---

## Operands

| Role | Tensor | Layout | Constraint |
|------|--------|--------|------------|
| A    | Input activations  | NHWC   | C = groups × cpg |
| B    | Weights            | KHWC/g | K = groups × kpg, cpg = kpg |
| D    | Output activations | NHWK   | stride=1 for depthwise |

The output spatial dimensions equal the input spatial dimensions
(`Ho = H`, `Wo = W`) because all current variants require `stride = 1` with
symmetric padding `PAD = (KH - 1) / 2`.

---

## cpg Variants

The variant is selected automatically by `DirectConvSpec` (the generic
dispatcher) from `cpg = C / groups`:

| cpg | Spec class           | MFMA atom                    | Requirement       |
|-----|----------------------|------------------------------|-------------------|
| 1   | `DirectDepthwiseSpec`| scalar FMA (no MFMA)         | stride = 1        |
| 4   | `DirectConv4cSpec`   | `mfma_f32_4x4x4_f16`        | cpg = kpg = 4     |
| 8   | `DirectConv8cSpec`   | `mfma_f32_16x16x16_f16`     | cpg = kpg = 8     |
| 16  | `DirectConv16cSpec`  | `mfma_f32_16x16x16_f16`/`32`| cpg = kpg = 16    |
| 32  | `DirectConv32cSpec`  | `mfma_f32_32x32x8_f16`      | cpg = kpg = 32    |

The fixed-`cpg` variants in the table require `kpg = cpg`.  The generic
`DirectConvSpec` does not: it accepts any `kpg >= 1` against any `cpg` that is a
positive multiple of 4.  If neither fits, use the implicit-GEMM kernel
([`conv_implicit_gemm.py`](conv_implicit_gemm.py)).

For `groups == 1` these variants degenerate to one wave per output row; use the
non-grouped direct conv instead ([`README_conv_direct_nongrouped.md`](README_conv_direct_nongrouped.md)).

### cpg = 1 — Depthwise (`DirectDepthwiseSpec`)

Each output channel is computed independently using scalar FMA.  The kernel
iterates over `(y, x)` and accumulates one channel at a time.  No MFMA
instruction is used.  Only `stride = 1` is supported.

Tunable parameters:
- `block_w` — output W positions per block (default 16; swept: 4, 8, 16, 32).
- `block_waves` — waves per workgroup (default 2; swept: 1, 2, 4).

Launch grid: `(ceil(W / block_w), ceil(groups / block_ch), N)`.

Large filters (13x13 .. 31x31) go to the output-stationary
`DirectDepthwiseTiledSpec` (`build_direct_depthwise_tiled`): a runtime loop over
the filter rows with one weight row and a sliding window of input rows in
registers, an output tile of `block_h x block_w`, and a grid split over output
rows too. Its footprint is linear in the filter width, so it avoids the
register spills and the steep compile cost the row-streaming kernel has at these
sizes. See `platform/dsl_docs/instances/convolution.md`.

### cpg = 1 — Column-streamed depthwise (`DirectDepthwiseColSpec`)

Same work as `DirectDepthwiseSpec` with the loop order flipped: the `KW` filter
columns are a runtime loop, and the input rows and `KH` filter rows of one
output-row tile are unrolled.  Only one filter column is live at a time, so the
per-lane register cost is `block_h * block_w + KH` rather than
`KH * KW + KH * block_w`.  That makes it the variant for large filters, and it
takes any `stride >= 1`, any `PAD < min(KH, KW)` (with `Ho <= H`), and
fp16/bf16 through its own `dtype` field.

Tunable parameters:
- `block_h` — output rows per block (default 16; swept: 8, 16, 32).  The row
  loop is unrolled at build time, so this is the AOT capability; the image
  height is a kernarg.
- `block_w` — output W positions per block (default 1; swept: 1, 2, 4).
- `block_waves` — waves per workgroup (default 1; swept: 1, 2, 4).

Pairs whose band exceeds the arch's live-f32 budget (3/8 of the VGPR file) are
rejected by `is_valid_depthwise_col_spec`.

Launch grid: `(ceil(Wo / block_w), ceil(groups / block_ch), N * ceil(Ho / block_h))`.

### cpg = 4 — `DirectConv4cSpec`

Uses sixteen independent `mfma_f32_4x4x4_f16` calls per `(y, x)` step to
cover all 4 input channels in one atom.  Each wave computes 4 output channels
for 4 output positions simultaneously.

Tunable parameters:
- `block_q` — output W positions per block (must be a multiple of 4; default 4).
- `block_groups` — groups per workgroup (must be a multiple of 16; default 16).

Launch grid: `(ceil(W / block_q), groups // block_groups, N)`.

### cpg = 8 — `DirectConv8cSpec`

Uses `mfma_f32_16x16x16_f16` to fold two consecutive filter positions
(`s = 0` and `s = 1`) into a single K=16 accumulation, reducing MFMA
instruction count by 2×.  Position `s = 2` is handled as a zero-padded residual.

Tunable parameters:
- `block_q` — output W positions per block (must be a multiple of 16; default 16).
- `block_groups` — groups per workgroup (default 8).
- `double_buffer` — double-buffer the B tile in LDS (default True).

### cpg = 16 — `DirectConv16cSpec`

Uses `mfma_f32_16x16x16_f16` (K=16 per atom, two atoms per `(y, x)` step) or,
on **gfx950** only, `mfma_f32_16x16x32_f16` (K=32, one atom per `(y, x)` step,
`fold_k32=True`).  The `fold_k32` path halves the number of MFMA instructions.

Tunable parameters:
- `block_q` — output W positions per block (default 16).
- `block_groups` — groups per workgroup (default 8).
- `fold_k32` — use the 16×16×32 atom; **gfx950 only** (default True on gfx950).
- `double_buffer` — double-buffer B tile (default True).

### cpg = 32 — `DirectConv32cSpec`

Uses four `mfma_f32_32x32x8_f16` calls per `(y, x)` step (K chunks of 8,
four to cover 32 channels): supported on **gfx942 and gfx950**.

Tunable parameters:
- `block_q` — output W positions per block (must be a multiple of 32; default 32).
- `block_groups` — groups per workgroup (default 4).
- `double_buffer` — double-buffer B tile (default True).

---

## `DirectConvProblem`

```python
@dataclass(frozen=True)
class DirectConvProblem:
    N:      int          # batch size
    H:      int          # input height  (= output height for stride=1)
    W:      int          # input width   (= output width  for stride=1)
    groups: int          # number of conv groups
    cpg:    int          # channels per group (input)
    kpg:    int          # channels per group (output); = cpg for the fixed-cpg
                         # variants, free for DirectConvSpec and the wgrad spec
    KH:     int = 3      # filter height
    KW:     int = 3      # filter width
    PAD:    int = 1      # symmetric padding applied to H and W
    stride: int = 1      # spatial stride (depthwise: must be 1)
```

Key derived properties:

| Property   | Formula              |
|------------|----------------------|
| `total_c`  | `groups * cpg`       |
| `total_k`  | `groups * kpg`       |
| `Ho`       | `(H + 2*PAD - KH) // stride + 1` |
| `Wo`       | `(W + 2*PAD - KW) // stride + 1` |
| `flops`    | `2 * N * H * W * groups * kpg * KH * KW * cpg` |
| `short()`  | `"N{N}H{H}W{W}_g{groups}_c{cpg}k{kpg}"`         |

---

## Generic Dispatcher — `DirectConvSpec`

`DirectConvSpec` is a convenient entry point when writing new benchmarks or dispatch
code. It accepts any `cpg` that is a positive multiple of 4 and builds a parametric
kernel using `mfma_f32_16x16x16_f16` with a runtime K-atom loop.
For the specialised fixed-`cpg` kernels (4/8/16/32), use the corresponding `DirectConv*cSpec` classes.
```python
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConvSpec,
    build_direct_conv,
    is_valid_spec,
)

p = DirectConvProblem(N=8, H=56, W=56, groups=4, cpg=16, kpg=16)
spec = DirectConvSpec(problem=p, name="my_kernel", block_q=16, block_groups=8)

ok, reason = is_valid_spec(spec, arch="gfx950")
if ok:
    kernel = build_direct_conv(spec, arch="gfx950")
```

Tunable parameters common to all grouped specs:

| Parameter       | Default | Notes |
|-----------------|---------|-------|
| `block_q`       | 16      | Output W-positions per block; must be ≥ 16 and a multiple of 16 |
| `block_groups`  | 8       | Groups per workgroup; `groups % block_groups == 0` required |
| `double_buffer` | True    | Double-buffer the B (weight) tile in LDS |

For **depthwise** (`cpg = 1`) use `DirectDepthwiseSpec` and `build_direct_depthwise`:

```python
from kernels.common.conv_direct_grouped import (
    DirectDepthwiseSpec,
    build_direct_depthwise,
    is_valid_depthwise_spec,
)

p = DirectConvProblem(N=8, H=56, W=56, groups=64, cpg=1, kpg=1)
spec = DirectDepthwiseSpec(problem=p, name="my_dw", block_w=16, block_waves=2)
kernel = build_direct_depthwise(spec, arch="gfx950")
```

---

## Backward Weights — `DirectConvWgradSpec`

The wgrad kernel computes the weight gradient

```
dW[k, r, s, c] = sum_{n, ho, wo} dY[n, ho, wo, k] * X[n, hi, wi, c]
```

with `hi = ho * stride + r - PAD` and `wi = wo * stride + s - PAD`.

### Algorithm — delta register ring + S-row strip

One block owns **all** `KH × KW` filter taps, so each loaded operand is reused
across the whole window instead of being re-read per tap.  The outer loop walks
**input** rows `hi` (not output rows), which at `stride = 1` pairs row `hi` with
output row `hi + PAD - r`.  Per input row:

1. **Delta ring** — one `dY` row is staged in LDS and read back into a
   `KH`-slot register ring.  Because consecutive `hi` iterations reuse the same
   row through different `r` taps, each `dY` row is loaded once and consumed
   `KH` times.
2. **S-row strip** — one `X` strip of `STRIP_COLS = WO_BLOCK + KW - 1` columns
   is staged in LDS.  All `KW` s-taps read that single strip at a one-row
   shift, so the strip costs one load per input row instead of `KW`.
3. **Compute** — `KH × KW` MFMAs accumulate into `KH × KW` independent
   `<4 x float>` accumulators.

The block owns exactly one `wo` tile, so there is no inner `wo` loop.  The row
loop is software-pipelined by one iteration: iteration `i` commits the fragments
issued at `i - 1` and issues row `i + 1`'s, which keeps every `s_waitcnt vmcnt`
a full compute phase away from its load.

Both LDS tiles are stored **spatial-major**, exactly as NHWC delivers them
(`dy_lds[sp][k_ch]`, `s_strip_lds[col][c_ch]`), so a lane's `VEC_CH` channels
land in one contiguous run and go back with a single `ds_write_b{64,128}`.  The
transposed per-lane operand the MFMA wants is recovered on the read side by
`ds_read_b64_tr_b16`, which is free — it is the same LDS traffic an untransposed
read would do.  Storing channel-major instead would cost `VEC_CH` scalar
`ds_write_b16` per lane per tile and make the kernel LDS-instruction bound.

Each LDS tile is keyed on exactly the wave axes its contents depend on
(`dy_lds` on `(wave_k, wave_q)`, `s_strip_lds` on `(wave_c, wave_q)`) and every
wave writes precisely the bytes it later reads.  That is what lets the row loop
run on **one barrier per iteration**.

### Spec

```python
@dataclass(frozen=True)
class DirectConvWgradSpec:
    problem:      DirectConvProblem
    name:         str = "direct_conv_wgrad"
    wave_tile_k:  int = 16   # K output channels per wave (MFMA M-dim)
    wave_tile_c:  int = 16   # C input  channels per wave (MFMA N-dim)
    waves_k:      int = 1    # waves along K
    waves_c:      int = 1    # waves along C
    waves_q:      int = 1    # waves along Q; each owns one wo_tile
    wave_size:    int = 64
    ho_per_block: int = 4    # input rows per block; tunes grid occupancy
    mfma_k:       int = 32   # MFMA K-inner: 32 (gfx950 default) or 16
```

| Property            | Formula                                  |
|---------------------|------------------------------------------|
| `block_k`           | `waves_k * wave_tile_k`                  |
| `block_c`           | `waves_c * wave_tile_c`                  |
| `threads_per_block` | `waves_k * waves_c * waves_q * wave_size`|
| `wo_block`          | `mfma_k`                                 |
| `n_wo_tiles()`      | `ceil(Wo / wo_block)`                    |
| `n_q_blocks()`      | `ceil(n_wo_tiles / waves_q)`             |
| `n_ho_blocks()`     | `ceil(H / ho_per_block)`                 |

`mfma_k` picks the atom and everything derived from it:

| `mfma_k` | Atom | `WO_BLOCK` | `VEC_CH` | DRAM load | `ds_read_tr` per fragment |
|----------|------|-----------|----------|-----------|---------------------------|
| 32 | `mfma_f32_16x16x32_f16` | 32 | 8 | vec8 | 2 (+ `vec_concat`) |
| 16 | `mfma_f32_16x16x16_f16` | 16 | 4 | vec4 | 1 |

`mfma_k = 32` covers twice the spatial positions per atom and issues vec8 DRAM
loads, so it halves the loop-iteration count and doubles cache-line utilisation.

### Operands

| Role | Tensor | Layout | dtype |
|------|--------|--------|-------|
| A | `dY` output gradient | `[N, Ho, Wo, groups*kpg]` | f16 |
| B | `X` input activations | `[N, H, W, groups*cpg]` | f16 |
| D | `dW` weight gradient | `[groups*kpg, KH, KW, cpg]` | **f32** |

`dW` is fp32 and is accumulated with `global_atomic_add` — **the caller must
zero it before launch.**  Note that the `c` axis of `dW` is per-group while its
`k` axis is global; the kernel indexes it with the in-group channel.

Because `D` is `ptr<f32, global>` (the forward variants take `ptr<f16, global>`)
this kernel does **not** share their manifest signature, even though the six
argument names are the same.

### Constraints

| Check | Rule |
|-------|------|
| `kpg` | `>= wave_tile_k` (16) |
| `cpg` | `>= wave_tile_c` (16); **need not equal `kpg`** |
| `waves_k * waves_c` | `<= 16` |
| `ho_per_block` | `> 0` |
| `mfma_k` | `16` or `32` |
| `stride` | must be `1` |
| arch | `mfma_f32_16x16x16_f16`, plus `mfma_f32_16x16x32_f16` when `mfma_k = 32`, plus `ds_read_tr16_b64` |

The `ds_read_tr16_b64` requirement makes this a **gfx950-only** kernel: the LDS
staging is built around the transpose read, and gfx942 has no equivalent.

Stride is a hard `1`: the row loop's `hi ↔ ho` pairing and the one-column-shift
strip read are both stride-1 identities.  At stride 2 the taps would have to
step the strip by `stride` columns and the row pairing would skip rows.

### Usage

```python
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConvWgradSpec,
    build_direct_conv_wgrad,
    is_valid_wgrad_spec,
)

p = DirectConvProblem(N=8, H=56, W=56, groups=4, cpg=16, kpg=16)
spec = DirectConvWgradSpec(problem=p, name="my_wgrad", mfma_k=32)

ok, reason = is_valid_wgrad_spec(spec, arch="gfx950")
if ok:
    kernel = build_direct_conv_wgrad(spec, arch="gfx950")
```

---

## fp32 storage and TF32 — MFMA forward and MFMA dgrad

`DirectConvProblem.dtype` takes two fp32-storage values besides `fp16` / `bf16`:

| `dtype` | Tensors | MFMA | gfx942 | gfx950 |
|---------|---------|------|--------|--------|
| `"fp32"` | fp32 | exact fp32 atoms | `16x16x4` / `32x32x2` f32 | same |
| `"tf32"` | fp32 | catalog TF32 atoms | native `16x16x8` / `32x32x4` xf32 | emulated `16x16x32` / `32x32x16` xf32 (3 bf16 MFMAs each) |

This covers `DirectConv16cSpec` (with and without `fold_k32`), `DirectConv8cSpec`,
`DirectConv32cSpec`, the generic `DirectConvSpec`, and the MFMA dgrad pipeline
(`make_dgrad_fprop_spec`, plus the transpose / reorganize / coalesced
weight-copy kernels).

- **Generic modes.** Every mode builds: plain loops, `fold_k32`,
  `runtime_k_loop`, `waves_k`, `waves_q`, single buffering, `persistent_grid`
  and `runtime_channels`. All of them except `persistent_grid` are checked on
  the GPU.
- **Coalesced weight copy.** `build_direct_coalesced_weights_dgrad` builds but
  has no caller and no GPU check, for any dtype.

**`DirectConv4cSpec` rejects both.** No 4x4 TF32 or fp32 atom exists on any
arch. Use the generic `DirectConvSpec` for `cpg = 4`: it runs the group as a
partial 16x16 K-atom.

### How it works

Nothing in the addressing changes. Every kernel builds the same K-contiguous
per-lane fragments as its fp16 path: 4 or 8 elements, read at element offsets.
Only the widths change:

- Global loads become `buffer_load_dwordx4`, one per 4 elements.
- LDS rows hold 4-byte elements, so the LDS footprint doubles.
- LDS reads are `ds_read_b128`, at most 4 elements and 16 bytes each.
- The epilogue stores fp32 without truncating.

Each logical `m x m` MFMA goes through `_mma_f32_logical`. This is exact
because every 16x16 / 32x32 layout groups lanes the same way: `lane // m` is the
K group, and the slot indexes within it. The helper does one of two things:

- **Split** the fragment into consecutive atom-wide slices on one accumulator.
- **Zero-pad** the fragment, and pad A and B alike, up to a wider atom.

For `"tf32"`, each operand is then bitcast to `vec<tf32 x N>`. There is no
rounding step: gfx942 hardware truncates, and gfx950 splits the full fp32
payload into BF16x3.

Some kernels chain several 4-wide atoms on one accumulator: the 16c kernel
without `fold_k32`, the 8c kernel, and the 32c kernel. For these,
`_mma_f32_chain` concatenates consecutive pairs into one 8-wide gfx950 TF32
atom instead of padding each one.

Physical MFMAs per filter row (3x3):

| Variant | gfx950 tf32 (bf16 MFMAs) | gfx942 tf32 (xf32) | fp32 |
|---------|--------------------------|--------------------|------|
| 16c `fold_k32` | 6 | 8 | 16 |
| 16c non-fold | 6 | 6 | 12 |
| 8c | 3 | 4 | 8 |
| 32c | 18 | 24 | 48 |

What this means for each configuration:

- **16c without `fold_k32`** is never worse than with it. The fold
  zero-pads the s=2 residual.
- **Generic `DirectConvSpec`**: its K-atoms are a runtime loop, so nothing
  can be paired.
  - On gfx950 TF32, prefer `fold_k32` (`cpg % 32 == 0`). Without it, each
    4-wide fragment is padded to the 8-wide emulated atom.
  - On gfx942 TF32, prefer the non-fold path: two native atoms per K-atom.

The fixed kernels preload the weights before the row loop. In ISA, LLVM hoists
the BF16x3 split (`v_cvt_pk_bf16_f32`) of those loop-invariant weights out of
the loop. The input fragment's split is common-subexpression-eliminated across
the KH taps that share it.

### Names, validation and LDS

- **Kernel names.** fp32 kernels end in `_f32` and TF32 kernels in `_tf32`;
  fp16 / bf16 names are unchanged.
  - The dgrad weight-copy kernels move elements without arithmetic, so fp32 and
    tf32 share one binary, tagged `_f32`.
  - `direct_dgrad_workspace_bytes` and `direct_dgrad_coalesced_workspace_bytes`
    size the workspace at 4 bytes per element.
- **Validation.** The validators replace the f16 atom checks with the TF32 /
  fp32 plan. They also check the doubled LDS row ring:
  - `is_valid_spec_{8c,16c,32c}` reject a fp32 ring over the arch's LDS (e.g.
    a large 32c tile on gfx942).
  - `is_valid_spec`'s LDS budget counts 4-byte elements.
- **ABI.** fp32 and tf32 kernels take `ptr<f32, global>` operands:
  `conv_direct_args_signature("fp32")`.

### Tests

`library/tests/test_conv_direct_tf32_mfma.py` has two parts.

- **CPU.** For every variant, on gfx950 and gfx942, for both dtypes:
  - the variant builds and lowers;
  - the MFMA intrinsics and their counts are right;
  - the rejections fire;
  - fp16 / bf16 names are unchanged;
  - one comgr ISA check per arch, including the hoisting above.
- **gfx950 GPU.** Every variant against a float64 reference, with the error
  taken relative to the peak.

---

## fp32 storage and TF32 — scalar-FMA kernels and direct wgrad

### Scalar-FMA kernels: one fp32 binary for both dtypes

These kernels accept `"fp32"` and `"tf32"`:

- `DirectConvDgradSpec`;
- the depthwise family: `DirectDepthwiseSpec`, `DirectDepthwiseSpatialSpec`,
  `DirectDepthwiseTiledSpec`, `DirectDepthwiseColSpec` (its own `dtype`
  field), `DirectDepthwiseDgradSpec` and `DirectDepthwiseDgradStreamSpec`.

All of them widen each element to f32 on load, reduce with scalar `fma`, and
narrow once on store. fp32 storage only changes three things:

- the load becomes a one-dword `buffer_load_vN`;
- the store becomes `buffer_store_f32`, with no truncation;
- the byte strides, which are now `_elem_bytes(dtype)`.

**`"tf32"` runs the same exact fp32 math.** TF32 permits reduced precision; it
does not require it. These kernels have no channel reduction for an MMA to
take: each output reduces over its own KH x KW taps, or over `kpg` for the
scalar dgrad. So fp32 and tf32 problems build the same IR under the same name,
tagged `_f32` (the column kernel spells its tag `f32`). One binary serves both.

The cache identity still records the problem's dtype, so an AOT cache stores
two entries with identical code. The sweep could map `tf32` to `fp32` for these
variants to keep one.

The shared helpers are `_scalar_load`, `_scalar_store`, `_scalar_name_flags`
and `_scalar_dtype_reason`. A dtype outside `fp16` / `bf16` / `fp32` / `tf32` is
rejected with a reason that names it.

### Direct wgrad (`DirectConvWgradSpec`)

The MFMA K axis is spatial (`wo`). The 16-bit kernel reads both operands with
`ds_read_b64_tr_b16`, and there is no 32-bit transpose read. The fp32 path
keeps the same spatial-major LDS tiles and the same software pipeline, and
changes three things:

- **Gather reads.** Each lane reads its fragment
  `tile[(lane/16)*VEC_CH + j][lane%16]`, `j < VEC_CH`, with `VEC_CH`
  `ds_read_b32`. LLVM pairs these into `ds_read2_b32`. The KW strip fragments
  are row shifts of one another, so their overlapping rows are loaded once.
- **Padded row stride.** The four lane groups of one read start
  `VEC_CH * row` dwords apart. That offset is 16 or 48 mod 64 (and 16 mod 32),
  so by address analysis the reads are bank-conflict free with 64 or 32 banks.
  This needs a row of 18 floats at `mfma_k=32` and 20 floats at `mfma_k=16`.
  An 18-float row is only 8-byte aligned, so the commit writes it as
  `ds_write_b64` pieces. LLVM pairs these into `ds_write2_b64`.
- **The MMA.** Each logical 16x16 MMA goes through `_mma_f32_logical`. The
  gathered fragment is K-contiguous like the 16-bit one.
  - **`"fp32"`:** `16x16x4` f32 atoms, 8 per logical atom at `mfma_k=32`.
  - **`"tf32"` on gfx950:** one emulated `16x16x32` xf32, which is 3
    `v_mfma_f32_16x16x32_bf16`. At `mfma_k=16` the fragment is zero-padded to
    8 elements, so prefer `mfma_k=32` there.
  - **`"tf32"` on gfx942:** native `v_mfma_f32_16x16x8_xf32`, 4 per logical
    atom at `mfma_k=32` and 2 at `mfma_k=16`.

Because the fp32 path needs no `ds_read_tr`, fp32 / tf32 wgrad is also valid
on **gfx942**. The 16-bit wgrad stays gfx950-only. dW stays the fp32 atomic
accumulator.

**Names.** `_f32` for fp32 and `_tf32` for tf32 (the code differs). The fp16 /
bf16 names and IR are unchanged.

ISA at 3x3, `ho_per_block=4`, `mfma_k=32`: 36 logical MMAs per kernel.

| | gfx950 tf32 | gfx942 tf32 | fp32 (both) | gfx950 fp16 |
|---|---|---|---|---|
| MFMAs | 108 `16x16x32_bf16` | 144 `16x16x8_xf32` | 288 `16x16x4_f32` | 36 `16x16x32_f16` |
| LDS reads | 44 `ds_read2_b32` | 44 `ds_read2_b32` | 44 `ds_read2_b32` | 36 `ds_read_b64_tr_b16` |
| LDS writes | 28 `ds_write2_b64` | 28 `ds_write2_b64` | 28 `ds_write2_b64` | 14 `ds_write_b128` |
| LDS bytes | 6912 | 6912 | 6912 | 3072 |
| VGPR + AGPR | 164 + 60 | 124 + 36 | 124 + 36 | 88 + 36 |

None of these spill. On gfx950 tf32, the BF16x3 split costs 148
`v_cvt_pk_bf16_f32`.

The implicit-GEMM wgrad, which also supports TF32, remains the choice for
strided problems: direct wgrad is stride-1 only, in every dtype.

### Tests

`library/tests/test_conv_direct_tf32_scalar.py` has two parts.

- **CPU.** Every scalar variant builds and lowers on gfx950 and gfx942, and
  fp32 and tf32 give the same name and the same IR. The test also checks that
  fp16 / bf16 names are unchanged and that bad dtypes are rejected. For wgrad,
  it builds every variant for fp32 and tf32 on both arches and checks the
  names and rejections. A comgr ISA check counts the MFMAs above.
- **GPU (gfx950).** The test compares against an IEEE fp32 CPU torch
  reference, with the error taken relative to the peak. The tolerance is 1e-5
  for the scalar kernels and fp32 wgrad, and 1e-4 for tf32 wgrad. Run it as
  `python3 library/tests/test_conv_direct_tf32_scalar.py`.

---

## Launch Grid

Every direct kernel is AOT: the batch, the spatial extents, the group count and
the activation strides are kernargs (`conv_direct_arg_names(direction=...)`),
so one binary serves any image. Take the grid from
`direct_launch_geometry(spec)` and the kernarg values from
`ConvArgs.from_problem(problem, direction=...).to_launch_values(...)` rather
than re-deriving them at the call site; the formulas below document what those
helpers compute.

### Grouped variants

```
grid  = (q_tiles, g_tiles, N)
block = (spec.threads_per_block, 1, 1)

q_tiles = ceil(W / block_q)
g_tiles = groups // block_groups
```

### Depthwise

```
grid  = (ceil(W / block_w), ceil(groups / block_ch), N)
block = (spec.threads_per_block, 1, 1)
```

The column-streamed variant tiles output rows too; the kernel decodes
`block_id_z = n * ceil(Ho / block_h) + h_tile`, deriving the tile count from
the `p_Ho` kernarg:

```
grid  = (ceil(Wo / block_w), ceil(groups / block_ch), N * ceil(Ho / block_h))
block = (spec.threads_per_block, 1, 1)
```

### Wgrad

```
grid  = (groups * n_k_tiles * n_c_tiles, ceil(H / ho_per_block), N * n_q_blocks)
block = (spec.threads_per_block, 1, 1)

n_k_tiles = ceil(kpg / spec.block_k)
n_c_tiles = ceil(cpg / spec.block_c)
n_q_blocks = spec.n_q_blocks()
```

The kernel decodes the three axes as:

```
c_tile  =  bx %  n_c_tiles          # x: flattened (group, k_tile, c_tile)
k_tile  = (bx // n_c_tiles) %  n_k_tiles
group   = (bx // n_c_tiles) // n_k_tiles

hi_block = by                       # y: input-row block

n        = bz // n_q_blocks         # z: flattened (batch, q_block)
q_block  = bz %  n_q_blocks
```

The `y` extent is `spec.n_ho_blocks()`, which is sized on the **input** height
— the row loop walks `hi`, and `Ho == H` only when `2 * PAD == KH - 1`.  A wave
whose `wo_tile` lands past `n_wo_tiles` runs the loop but has its epilogue
atomics suppressed, so an over-provisioned `z` extent is safe.

`n_wo_tiles = ceil(Wo / mfma_k)` and `n_q_blocks = ceil(n_wo_tiles / waves_q)`
are computed in-kernel from the `p_Wo` kernarg, with the same ceilings
`spec.n_q_blocks()` uses for the host-side `z` extent.

---

## Architecture Support

| Variant | gfx942 | gfx950 | gfx1250 |
|---------|--------|--------|---------|
| cpg=1 (depthwise) | ✓ | ✓ | — |
| cpg=4  | ✓ | ✓ | — |
| cpg=8  | ✓ | ✓ | — |
| cpg=16 (`fold_k32=False`) | ✓ | ✓ | — |
| cpg=16 (`fold_k32=True`)  | — | ✓ | — |
| cpg=32 | ✓ | ✓ | — |
| wgrad (`DirectConvWgradSpec`) | — | ✓ | — |

gfx1250 (WMMA/RDNA) is not supported by any direct-conv variant.  Use the
implicit-GEMM kernel with `pipeline="wavelet"` on that target.

The wgrad kernel is gfx950-only in both `mfma_k` modes: even `mfma_k = 16`,
whose MFMA atom exists on gfx942, needs `ds_read_tr16_b64` for the LDS staging.
For a backward-weights pass on gfx942 use the implicit-GEMM wgrad kernel
([`conv_implicit_gemm_wgrad.py`](conv_implicit_gemm_wgrad.py)).

---

## When to Use Direct Conv vs. Implicit-GEMM

| Scenario | Recommendation |
|----------|----------------|
| `cpg` ∈ {4, 8, 16, 32} and small K loop (`KH×KW×cpg ≲ 128`) | Try direct-conv first; lower setup cost |
| `cpg` ∈ {4, 8, 16, 32} and large K loop | Both; compare with `benchmark_conv_compare.py` |
| `cpg` not in {1, 4, 8, 16, 32} | Implicit-GEMM only |
| `stride > 1` or `dilation > 1` | Implicit-GEMM only |
| Grouped with arbitrary `groups` | Implicit-GEMM |
| gfx1250 target | Implicit-GEMM (`pipeline="wavelet"`) |
| Depthwise (`cpg = 1`, `stride = 1`) | Direct-conv depthwise |
| Backward-weights, gfx950, `cpg` and `kpg` ≥ 16, `stride = 1` | Direct-conv wgrad |
| Backward-weights, anything else | Implicit-GEMM wgrad |
| Non-grouped (`groups = 1`), `stride ≤ 2` | Sweep `conv_direct_nongrouped` and implicit-GEMM, keep the faster ([`README_conv_direct_nongrouped.md`](README_conv_direct_nongrouped.md)) |

---

## Changelog

### Initial implementation

- `DirectConv16cSpec` and `build_direct_conv_16c`: cpg=16 grouped conv using
  `mfma_f32_16x16x16_f16`.  Outer loop over `(y, x)`; inner accumulation over
  `c ∈ [0, 16)` via two K=8 atom calls.  Supported on gfx942 and gfx950.

### cpg=4 variant

- `DirectConv4cSpec` and `build_direct_conv_4c`: sixteen independent
  `mfma_f32_4x4x4_f16` calls per `(y, x)` step.

### cpg=8 variant

- `DirectConv8cSpec` and `build_direct_conv_8c`: folds positions `s=0` and
  `s=1` into one `mfma_f32_16x16x16_f16` with K=16.

### fold_k32 (cpg=16, gfx950)

- `DirectConv16cSpec.fold_k32=True` (default on gfx950): uses
  `mfma_f32_16x16x32_f16` to process all 16 channels in one K=32 atom per
  `(y, x)` step, halving instruction count.  Rejected by `is_valid_spec_16c`
  when the 16×16×32 atom is absent (gfx942).

### Depthwise variant

- `DirectDepthwiseSpec` and `build_direct_depthwise`: scalar FMA path for
  cpg = kpg = 1.  No MFMA; each wave processes one output channel.

### cpg=32 variant

- `DirectConv32cSpec` and `build_direct_conv_32c`: four
  `mfma_f32_32x32x8_f16` calls per `(y, x)` step (K chunks of 8).

### Generic dispatcher

- `DirectConvSpec` and `build_direct_conv`: selects the appropriate cpg-specific
  kernel at build time from `cpg ∈ {4, 8, 16, 32}`.

### MIOpenDriver input for benchmarks

- `benchmark_direct_conv.py` now accepts `--miopen-cmd` and `--miopen-file` to
  load conv shapes from MIOpenDriver command strings.

### `kpg != cpg` for the generic dispatcher

- `DirectConvSpec` / `is_valid_spec` no longer require `kpg == cpg`; any
  `kpg >= 1` is accepted against a `cpg` that is a positive multiple of 4.  The
  fixed-`cpg` variants (4/8/16/32) are unchanged and still require `kpg == cpg`.
- `DirectConvProblem` gained the `Ho` / `Wo` output-geometry properties.

### Backward-weights variant

- `DirectConvWgradSpec`, `is_valid_wgrad_spec` and `build_direct_conv_wgrad`:
  direct wgrad via a `KH`-slot delta register ring plus a shared S-row strip in
  LDS, with all `KH × KW` taps in one block and `ds_read_b64_tr_b16` recovering
  the MFMA operands from spatial-major tiles.  `dW` is fp32 and accumulated with
  `global_atomic_add`, so the caller must zero it first.  gfx950 only;
  `stride = 1` only; `cpg` and `kpg` are independent.

---

## Dual-Engine Parity

Every kernel in this file exists in both the Python engine (this module) and the
C++ engine (`cpp/instances/common/conv_direct_grouped_*.cpp`), and the two
**must emit the same LLVM-IR bytes**.  A change to any builder here has to be
mirrored into its C++ peer in the same change, and proven with:

```bash
cd platform && export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python tools/check_byte_identity.py --only conv_direct
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py --only conv_direct
```

The gate drives the sampled spec configs in
`tests/instances/parity/conv_direct_grouped_emit.{py,c}` — configs 0-24 are the
existing direct-conv variants, 25-31 the wgrad variant (25 `mfma_k=32`, 26
`mfma_k=16`, 27 multi-wave K/C/Q, 28-29 the two gfx942 rejection paths, 30-31
bf16 at `mfma_k=32` / `16`).  Add a config to **both** emitters when you add a
variant.

| Python | C++ |
|--------|-----|
| `DirectConvWgradSpec` | `rocke_direct_conv_wgrad_spec_t` |
| `spec.validate()` | `rocke_direct_conv_wgrad_validate` |
| `is_valid_wgrad_spec` | `rocke_direct_conv_wgrad_is_valid_spec` |
| `build_direct_conv_wgrad` | `rocke_build_direct_conv_wgrad` / `_new` |

One C++-specific hazard worth knowing when mirroring a builder: C++ leaves the
evaluation order of sibling call arguments **unsequenced**, so a Python
expression such as `b.land(b.cmp_ge(...), b.cmp_lt(...))` must be hoisted into
explicitly sequenced locals on the C++ side.  Written inline it will emit the
two comparisons in whichever order the compiler picks, and the byte-identity
gate will fail on the SSA numbering.

---
