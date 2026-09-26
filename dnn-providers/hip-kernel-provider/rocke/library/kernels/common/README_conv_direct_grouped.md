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

### cpg = 1 — Depthwise (`DirectDepthwiseSpec`)

Each output channel is computed independently using scalar FMA.  The kernel
iterates over `(y, x)` and accumulates one channel at a time.  No MFMA
instruction is used.  Only `stride = 1` is supported.

Tunable parameters:
- `block_w` — output W positions per block (default 16; swept: 4, 8, 16, 32).
- `block_waves` — waves per workgroup (default 2; swept: 1, 2, 4).

Launch grid: `(ceil(W / block_w), ceil(groups / block_ch), N)`.

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
| `n_ho_blocks()`     | `ceil(Ho / ho_per_block)`                |

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
from rocke.instances.common.conv_direct_grouped import (
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

## Launch Grid

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

At the supported stride-1 geometry `Ho == H`, so the `y` extent is equally
`spec.n_ho_blocks()`.  A wave whose `wo_tile` lands past `n_wo_tiles` runs the
loop but has its epilogue atomics suppressed, so an over-provisioned `z` extent
is safe.

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
`tests/instances/parity/conv_direct_grouped_emit.{py,c}` — configs 0-8 are the
forward variants, 9-13 the wgrad variant (9 `mfma_k=32`, 10 `mfma_k=16`, 11
multi-wave K/C/Q, 12-13 the two gfx942 rejection paths).  Add a config to
**both** emitters when you add a variant.

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
