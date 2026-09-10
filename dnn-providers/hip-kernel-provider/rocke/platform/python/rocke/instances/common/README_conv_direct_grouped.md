# Direct Grouped Convolution

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

All grouped variants require `kpg = cpg`.  If `cpg` is not in the table above,
use the implicit-GEMM kernel ([`conv_implicit_gemm.py`](conv_implicit_gemm.py)).

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
    kpg:    int          # channels per group (output);  must equal cpg
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
| `flops`    | `2 * N * H * W * groups * kpg * KH * KW * cpg` |
| `short()`  | `"N{N}H{H}W{W}_g{groups}_c{cpg}k{kpg}"`         |

---

## Generic Dispatcher — `DirectConvSpec`

`DirectConvSpec` is the recommended entry point when writing new benchmarks or
dispatch code.  It accepts any `cpg` that is a positive multiple of 4 and
selects the matching specialised spec internally.

```python
from rocke.instances.common.conv_direct_grouped import (
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
from rocke.instances.common.conv_direct_grouped import (
    DirectDepthwiseSpec,
    build_direct_depthwise,
    is_valid_depthwise_spec,
)

p = DirectConvProblem(N=8, H=56, W=56, groups=64, cpg=1, kpg=1)
spec = DirectDepthwiseSpec(problem=p, name="my_dw", block_w=16, block_waves=2)
kernel = build_direct_depthwise(spec, arch="gfx950")
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

gfx1250 (WMMA/RDNA) is not supported by any direct-conv variant.  Use the
implicit-GEMM kernel with `pipeline="wavelet"` on that target.

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

---
