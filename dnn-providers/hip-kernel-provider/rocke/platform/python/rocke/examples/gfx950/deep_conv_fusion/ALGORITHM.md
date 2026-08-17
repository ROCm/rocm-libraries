# The deep-fused conv→pool kernel, from the math up

This is the algorithm the kernel implements and *why* it is **one kernel**. The
[README](README.md) is the optimization history; this file is the specification,
the data layout, and the precise per-threadgroup steps.

The kernel fuses a five-stage convolution block into a single launch with **no
HBM intermediates**:

```text
conv0 (3x3, K0 channels)  ->  ReLU  ->  conv1 (1x1, K1 channels)  ->  ReLU
                          ->  2x2 stride-2 maxpool  ->  final NHWK store
```

A naive realization is a pipeline of separate kernels (conv0, relu, conv1, relu,
pool) that writes `conv0` and `conv1` to HBM and reads them back. **The fused
kernel does the whole block per output tile in one launch and never spills an
intermediate feature map to HBM** — the only HBM traffic is the input `A`, the
two weight tensors `B0`/`W1`, and the pooled output `Y`.

## 0. Notation

| symbol | meaning |
|---|---|
| `N, Hi, Wi, C` | input batch / height / width / channels (`A`, NHWC) |
| `K0` | conv0 output channels (`conv.K`) |
| `R, S` | conv0 filter height / width (3×3 here) |
| `Ho, Wo` | conv0/conv1 output spatial dims (`= Hi, Wi` with pad 1, stride 1) |
| `K1` | conv1 (1×1) output channels (`conv1_channels`) |
| `B0` | conv0 weights, `[K0, R, S, C]` |
| `W1` | conv1 1×1 weights, `[K1, K0]` |
| `Y` | pooled output, `[N, pool_ho, pool_wo, K1]` |
| `pool_y, pool_x` | pool window (2×2) |
| `pool_stride_h/w` | pool stride (2) |
| `tile_m` | conv rows a threadgroup computes (= one pooled-output tile's source patch) |
| `tile_n` | conv output channels a threadgroup owns (≥ `K1`) |
| `tile_k` | contraction-tile width along the implicit-GEMM K axis |

The conv is fixed to `R=S=3`, `stride=1`, `pad=1`, `dilation=1`, so
`Ho=Hi`, `Wo=Wi`; the pool is fixed to `2×2 stride-2`. Both convs accumulate in
f32; operands and the final store are f16.

## 1. What the block computes (the specification)

For one batch element, the dense definition is three composed operators:

```text
C0[n,h,w,k]  = relu( Σ_{r,s,c} A[n, h+r-1, w+s-1, c] · B0[k,r,s,c] )      # conv0 3x3 + ReLU,  k in [0,K0)
C1[n,h,w,o]  = relu( Σ_k C0[n,h,w,k] · W1[o,k] )                          # conv1 1x1 + ReLU,  o in [0,K1)
Y[n,ho,wo,o] = max_{i<2, j<2} C1[n, 2*ho+i, 2*wo+j, o]                    # 2x2 stride-2 maxpool
```

`relu(x) = max(x, 0)`. This is exactly the NumPy reference in
`deep_fused_conv_pool_verify.py` (`_reference_conv1x1_relu_pool`), with the f16
round-trips inserted at each ReLU boundary to match the kernel's storage
precision.

## 2. Both convolutions are GEMMs (the implicit-GEMM lemma)

A convolution over NHWC data is a matrix multiply once you flatten the
output-pixel and filter-tap axes:

- **conv0** is a GEMM with `M = N·Ho·Wo` output pixels, `N_gemm = K0` output
  channels, and `K_gemm = R·S·C` contraction taps. The "A matrix" rows are the
  `im2col` patches of the input; the "B matrix" is `B0` reshaped to
  `[R·S·C, K0]`. This is the standard implicit-GEMM formulation — the patch
  matrix is never materialized; addresses are computed on the fly.
- **conv1** is a 1×1 convolution, i.e. a *pure* GEMM with `K_gemm = K0`,
  `N_gemm = K1`, contracting over the conv0 output channels with no spatial
  taps. Its "A matrix" is the conv0 output tile `C0`; its "B matrix" is `W1`.

So the whole block is **two chained GEMMs with two pointwise ReLUs and a final
spatial reduction**. The fusion opportunity is the chaining: conv1's A operand
**is** conv0's output, and the pool's input **is** conv1's output. If the
threadgroup keeps both on chip, the two intermediate feature maps never touch
HBM.

## 3. The fusion idea (the heart)

Tile by the **final pooled output**, and pull the whole dependency cone on chip.

One threadgroup (CTA) owns a rectangular tile of the pooled output:
`pool_tile_h × pool_tile_w` pooled pixels, for all `K1` channels. To produce that
tile it needs:

- the `(pool_tile_h·pool_stride_h) × (pool_tile_w·pool_stride_w)` window of
  conv1 outputs that the pool reduces, and
- the conv0 outputs feeding that conv1 window, and
- the (haloed) input patch feeding that conv0 window.

The pooled tile's source conv patch is exactly `tile_m` rows of the implicit
GEMM:

```text
tile_m = (pool_tile_h · pool_stride_h) · (pool_tile_w · pool_stride_w)
```

(the validator enforces this equality — see §8.2). So the threadgroup:

1. runs the **conv0 implicit GEMM** for its `tile_m × K0` accumulator tile,
   applying ReLU as a static accumulator epilogue;
2. stages that ReLU'd conv0 tile through **C-shuffle LDS**;
3. runs the **conv1 1×1 GEMM** (`tile_m × K1`) consuming the on-chip conv0 tile
   as its A operand, applying the second ReLU;
4. **inline-pools** the conv1 tile (2×2 stride-2 max) and writes only the pooled
   result to HBM.

Different `(pooled-tile)` CTAs share nothing, so the grid is embarrassingly
parallel over the pooled-output spatial extent.

## 4. From spec to kernel: who computes what

**Grid.** The launch grid is

```text
grid = (1, pool_ho // pool_tile_h, pool_wo // pool_tile_w)
```

(`deep_fused_conv_pool_grid`). A CTA at `(·, by, bz)` owns the pooled tile at
`(by·pool_tile_h, bz·pool_tile_w)`; the conv source window it computes starts at
`by·pool_tile_h·pool_stride_h`, `bz·pool_tile_w·pool_stride_w`. The validator
requires `pool_ho`/`pool_wo` to divide the pool-tile dims, so the grid covers the
output exactly with no partial tiles.

**Block.** `block_size = warp_m · warp_n · wave_size`. On gfx950 that is
`2 · 1 · 64 = 128` threads (two wave64s) for the winning geometry.

**What lives where.**

| state | location |
|---|---|
| input patch `A` (haloed) | HBM, gathered per K-tile (implicit-GEMM A loads) |
| conv0 / conv1 weights `B0`, `W1` | HBM, streamed per K-tile |
| conv0 accumulator `[tile_m, K0]` | f32 registers (MFMA C-fragment) |
| ReLU'd conv0 tile | **C-shuffle LDS** (the conv0→conv1 hand-off) |
| conv1 accumulator `[tile_m, K1]` | f32 registers |
| pooled `[pool_tile_h, pool_tile_w, K1]` | registers / LDS, then HBM |
| pooled output `Y` | HBM (the only intermediate-free store) |

## 5. One threadgroup, step by step

Let the CTA own pooled tile `(by, bz)`, source conv window
`tile_m = conv_tile_h × conv_tile_w`.

### 5.1 — conv0 implicit GEMM + ReLU
Walk the conv0 contraction `K_gemm = R·S·C` in `tile_k` chunks. Each step loads
the implicit-GEMM A patch (the haloed input region for these `tile_m` output
pixels) and the `B0` slice, and issues `32×32×16` MFMAs (gfx950) into the f32
conv0 accumulator. After the last K-tile, the **accumulator epilogue applies
ReLU** (`ConvAccumulatorEpilogue(relu=True)`) — `relu(C0)` lives only in
registers, never in HBM.

The A operand uses a `decompose_m=False` descriptor path: the
`m → (n, ho, wo)` mapping is passed directly rather than recovered by
magic-division inside the kernel (see §8.3).

### 5.2 — Stage conv0 to C-shuffle LDS (the hard hand-off)
The conv0 MFMA leaves its result in the C-output lane distribution
(same-column / different-row per lane), but conv1 needs it as an MFMA A operand
(row-major, contiguous along K0). The kernel publishes the ReLU'd conv0 tile to a
row-major `[tile_m, tile_n]` C-shuffle LDS buffer, with a barrier, then conv1
reads its A rows back as wide vector loads. This M→K0 transpose is intrinsic to
chaining two GEMMs and is the kernel's remaining structural hand-off (§10).

### 5.3 — conv1 1×1 GEMM + ReLU
The 1×1 conv is a pure GEMM contracting the on-chip conv0 channels `K0`. Read the
conv0 A operand from C-shuffle LDS and the `W1` B operand (loaded from HBM,
sharing the conv0-cshuffle barrier — see the README's barrier-merge lever),
issue the conv1 MFMAs into a `[tile_m, K1]` f32 accumulator, and apply the
**second ReLU** as its accumulator epilogue. Because `K1 ≤ tile_n`, the conv1
output fits one tile-n's worth of channels.

### 5.4 — Inline 2×2 stride-2 maxpool
The `tile_m` conv1 rows are exactly the `conv_tile_h × conv_tile_w` spatial patch
of one pooled tile. The pool reduces each `2×2` window to one output pixel:

```text
Y_tile[ho, wo, o] = max_{i<2, j<2} C1_tile[2*ho+i, 2*wo+j, o]
```

For the winning geometry the four window elements that one lane needs are already
**register-resident in that lane**, so the 2×2 max is done in registers with no
conv1→pool LDS hand-off (the largest of the later levers — digest lever C). For
other geometries the kernel falls back to a vectorized LDS gather of the pool
window.

### 5.5 — Write the pooled tile to HBM
Store `Y_tile` to the global `[N, pool_ho, pool_wo, K1]` output. This is the only
intermediate-free write — neither `C0` nor `C1` was ever materialized in HBM.

## 6. The epilogue — fuse the second ReLU past the pool

A correctness-preserving algebraic identity lets the conv1 ReLU move to **after**
the pool:

```text
relu(max(x))  ==  max(relu(x))        (relu is monotone non-decreasing)
```

So the kernel may defer the conv1 ReLU until after the maxpool, applying one ReLU
to the pooled result instead of to every pre-pool element (digest lever A). The
result is identical; the work is smaller.

## 7. Why this is not the bottleneck you'd expect

The useful-FLOP accounting for the full target shape
(`[1, 2160, 3840, 8] → K0=32 → K1=24 → pool`):

```text
conv0 useful work:  38.22 GFLOP
conv1 useful work:  12.74 GFLOP
total useful conv:  50.96 GFLOP
```

Despite the large feature maps, hardware counters show the kernel is **not**
HBM-bound and **not** MFMA-bound — early captures put `MfmaUtil` at ~6% and
`MemUnitStalled` at ~0.06%, with **VALU and LDS-wait dominating**. The fusion
already removed the HBM round-trips, so the residual cost is the *operand-delivery
path*: coordinate arithmetic, LDS staging/reads, and synchronization. That is why
every kept optimization in the README targets VALU / LDS / barrier overhead, not
more matmul throughput or input caching. (See the README for the full counter
read and the per-lever ledger.)

## 8. Implementation details worth the math

### 8.1 The C-shuffle layout constraint
The MFMA C-fragment is laid out same-column / different-row per lane, but conv1's
A read wants row-major same-row / contiguous-column. A store layout that
vectorizes the conv0→LDS publish breaks the conv1 read vectorization, and vice
versa — they want opposite layouts. This is why C-shuffle store vectorization is
a closed path (README "rejected") and why the kept win was to vectorize the
*conv1 read* (`ds_read_b128`) instead.

### 8.2 The `tile_m` ↔ pool-tile equality (a validity requirement)
`tile_m` is **not** free: it must equal the rectangular conv patch backing one
pooled tile,

```text
tile_m == (pool_tile_h · pool_stride_h) · (pool_tile_w · pool_stride_w).
```

`make_deep_fused_conv_pool_spec` derives `tile_m` from `pool_tile_*` so they can
never disagree; `is_valid_spec` rejects any spec where they do. The validator
also requires `N = 1` (single-batch tiled schedule), both `K0 ≤ tile_n` and
`K1 ≤ tile_n` (one CTA owns all channels), `K0` divisible by `8` (the `W1`
loader), `pool_ho`/`pool_wo` divisible by the pool tile, the fixed `2×2 stride-2`
pool, and `tile_m` divisible by `warp_m · warp_tile_m`.

### 8.3 The `decompose_m=False` A-descriptor
The implicit-GEMM A descriptor normally recovers `(n, ho, wo)` from the flat
m-index via magic division inside the kernel. Because the fused carrier already
knows the pooled-tile coordinates, it passes `(n, ho, wo)` directly and bypasses
the redundant decode — removing VALU on the operand path (digest lever D).

### 8.4 Arch-parametric body
The kernel body lives in `instances/common/deep_fused_conv_pool.py` and is driven
by the resolved MMA op, so the same code emits the gfx950 MFMA path (wave64,
`32×32×16`) and, via a separate sibling example, the gfx1201 WMMA path (wave32,
`16×16×16`). This example pins the gfx950 geometry and kernel name.

## 9. The whole kernel in pseudo-code

```text
for each CTA (by, bz):                       # grid = (1, pool_ho/pool_tile_h, pool_wo/pool_tile_w)
    # source conv window for this pooled tile
    h0 = by * pool_tile_h * pool_stride_h
    w0 = bz * pool_tile_w * pool_stride_w

    conv0_acc = 0                            # f32 [tile_m, K0]
    for kt in range(0, R*S*C, tile_k):       # conv0 implicit GEMM
        A_patch = load_implicit_gemm_A(h0, w0, kt)        # haloed input, decompose_m=False
        B0_tile = load(B0, kt)
        conv0_acc += MFMA_32x32x16(A_patch, B0_tile)
    conv0_acc = relu(conv0_acc)              # accumulator epilogue
    C0_lds = cshuffle_store(conv0_acc)       # row-major [tile_m, K0] in LDS; barrier

    conv1_acc = 0                            # f32 [tile_m, K1]
    for kt in range(0, K0, tile_k):          # conv1 1x1 GEMM
        A1 = cshuffle_read(C0_lds, kt)       # vectorized ds_read_b128
        W1_tile = load(W1, kt)               # shares the conv0-cshuffle barrier
        conv1_acc += MFMA_32x32x16(A1, W1_tile)
    # ReLU may defer past the pool: relu(max(x)) == max(relu(x))

    Y_tile = inline_2x2_maxpool(relu(conv1_acc))   # register-resident for the best geometry
    store(Y[n, by*pool_tile_h .., bz*pool_tile_w .., :], Y_tile)
```

## 10. Where the algorithm ends and tuning begins

The math above is fixed. The README's levers — conv1 LDS-read vectorization, the
barrier merge, the `4×4/tile_m=64/tile_k=32` geometry, deferring the conv1 ReLU
past the pool, register-resident pooling, and the `decompose_m=False` A-descriptor
— change *only how* these steps are scheduled and how operands are delivered,
never *what* is computed. The one structural hand-off that survives is the
conv0→conv1 M→K0 transpose of §5.2 (the conv1→pool hand-off was eliminated by
register-resident pooling); cutting it further is the open work.

This is an experimental prototype: it is a gfx950-only fp16 / f32-accumulation
proof. It does **not** yet implement int8/int4 packed paths, true two-pointer
virtual concat, production autotuning, or generic graph fusion (see the digest's
"Remaining Work").
