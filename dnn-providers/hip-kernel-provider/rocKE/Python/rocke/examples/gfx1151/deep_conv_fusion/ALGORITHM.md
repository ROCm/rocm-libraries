# The fused conv+maxpool kernel, from the math up

This is the algorithm the kernel computes and *why* it is one kernel. The README
is the optimization history; this file is the specification, the requant algebra,
the data layout, and the precise per-CTA steps.

The kernel fuses the whole `encoder_0` image-encoder block into a single launch
on **gfx1151** (Strix Halo / Radeon 8060S, RDNA3.5, wave32, WMMA). It is the
wave32/WMMA sibling of the gfx950 (CDNA/MFMA/wave64) prototype in
`../../gfx950/deep_conv_fusion/`. Two things make it worth a from-the-math-up
write-up: it is **genuine low-bit** (real int8/int4 in HBM, not fake-quant), and
every requant step is **bit-exact** to an integer reference — the kernel matches
a numpy integer model with **zero** mismatches, not "close in f16."

> If you just want to *run* it, see [`README.md`](README.md). The optimization
> campaign — every lever, what worked, what did not, and why — is in
> [`CASE-STUDY-optimizations.md`](CASE-STUDY-optimizations.md). This file explains
> *what* the kernel computes before either of those explains *how* it was tuned.

## 0. Notation

| symbol | meaning |
|---|---|
| `N, Hi, Wi, C` | input batch, height, width, channels (NHWC) |
| `K0` | conv0 output channels (32 in the target; padded from the spec `--k0`) |
| `R, S` | conv0 kernel height/width (3×3, pad 1) |
| `Ho, Wo` | conv0 output spatial dims (= `Hi, Wi` at pad 1, stride 1) |
| `K1` | conv1 output channels (24 in the target, `--k1`) |
| `X` | activations, int8, `[N, Hi, Wi, C]` |
| `W0` | conv0 weights, int8, `[K0, R, S, C]` |
| `W1` | conv1 weights, packed signed int4, `[K1, K0/2]` bytes |
| `Y` | pooled output, packed signed int4, `[pool_ho, pool_wo, ceil(K1/8)]` i32 words |
| `m0, m0b, m1, mf` | the four compile-time requant inverse-scale multipliers |
| `B_k` | WMMA contraction width, `B_k = 16` |

All conv accumulation is over integer codes; the WMMA atom contracts 16 at a
time. The four requant multipliers default to exact powers of two:
`m0 = 0.0625` (= 1/16), `m0b = 0.5`, `m1 = 0.25`, `mf = 1.0`.

## 1. What `encoder_0` computes (the specification)

`encoder_0` is a low-bit image-encoder block. In math, over the spatial grid:

```
P0 = conv0(X, W0)               # 3x3 pad1, int8*int8 -> int32     [Ho, Wo, K0]
q0 = relu( Q_i8( P0 * m0 ) )    # requant int32->int8, then ReLU
C0 = Q_i4( q0 * m0b )           # requant int8->int4               (int4 codes)
P1 = conv1(C0, W1)              # 1x1, int4*int4 -> int32          [Ho, Wo, K1]
C1 = relu( Q_i4( P1 * m1 ) )    # requant int32->int4, then ReLU   (int4 codes)
M  = maxpool_2x2_s2( C1 )       # 2x2 stride-2 max over int4 codes [pool_ho, pool_wo, K1]
Y  = Q_i4( M * mf )             # final requant int4->int4 (mf=1 -> identity pack)
```

`Q_i8(v) = clamp(round_half_even(v), -127, 127)` and
`Q_i4(v) = clamp(round_half_even(v), -8, 7)`. `relu(x) = max(x, 0)`. The full
target shape (`N=1, H=2160, W=3840, C=8, K0=32, K1=24`) is ~51.0 GOP; the
op-by-op breakdown is in [`inputs.md`](inputs.md).

The naïve realization is a pipeline of separate kernels — conv0, two requants, a
ReLU, conv1, a requant, a ReLU, a maxpool, a final requant — each writing its
intermediate to HBM and reading it back. **The fused kernel does the entire chain
in one launch and never writes the conv0 or conv1 intermediate to HBM.** Only the
final packed-int4 `Y` reaches global memory.

## 2. The requant lemma (why the codes can stay integer)

Every `Quant` node is a per-tensor symmetric quantizer. For a true value `v`
(here the exact integer conv accumulator) and an output scale, the stored code is

```
code = clamp( round_half_even( v * inv_scale ), qmin, qmax )
```

The three physical scales at each node — input (activation) scale, weight scale,
output scale — fold into a **single** compile-time multiplier
`inv_scale = act_scale * weight_scale / out_scale`. Because all defaults are
exact powers of two, that multiply is exact in f32, and the conv operands
themselves carry an effective dequant scale of `1.0` — i.e. the matmul runs on
**raw integer codes** and the requant multiplier absorbs all the scaling *after*
the contraction. This is the whole reason the pipeline can be integer-exact: no
scale is ever applied per element inside the dot product, only once per requant
node.

The four multipliers map onto the four `Quant` nodes:

| node | multiplier | from → to | range |
|---|---|---|---|
| conv0 requant | `m0 = 1/16` | int32 → int8 | `[-127, 127]` |
| post-ReLU requant | `m0b = 1/2` | int8 → int4 | `[-8, 7]` |
| conv1 requant | `m1 = 1/4` | int32 → int4 | `[-8, 7]` |
| final requant | `mf = 1` | int4 → int4 | `[-8, 7]` (identity pack) |

In the native-int path these power-of-two multipliers become integer
round-half-even **shifts** (`_quant_i8_shift` / `_quant_i4_shift`: divide by
`2^k` with the standard `(x + half - 1 + (q&1)) >> k` round-to-nearest-even
bias), so there is no float in the requant at all.

## 3. Genuine low-bit storage (the data layout)

The codes are real, in HBM, at their natural bit width:

- **`X`, `W0`** — int8, one byte per code, plain NHWC / `[K0,R,S,C]` layout.
- **`W1`, `Y`** — packed signed int4: two nibbles per byte,
  `byte = (hi << 4) | (lo & 0xF)`, with sign-extension on unpack
  (`nib >= 8 → nib - 16`). `W1` is `[K1, K0/2]` bytes; `Y` is assembled as i32
  words of 8 nibbles each (`pool_ho × pool_wo × ceil(K1/8)`), so a thread that
  owns one pooled pixel writes the channels as whole i32 words rather than
  per-byte stores.

The host driver packs the four base pointers into the kernel ABI as
`struct.pack("<QQQQ", X, W0, Y, W1)` — four pointers, no scalar size arguments
(every dimension is baked into the compiled spec). The numpy reference packs
`W1` and unpacks `Y` with the *identical* nibble layout, so a correct kernel
reproduces the integer reference bit-for-bit.

## 4. The fusion idea (the heart): backward-planned per-CTA tiles

Each CTA owns a rectangular tile of the **final pooled output** and is planned
**backward** from there:

```
pooled tile  ->  conv1 patch  ->  conv0 region  ->  input halo
[pth, ptw]       [2*pth, 2*ptw]    (+ no spatial      (conv0 region + a 1-px
 of Y            (the 2x2/s2        growth: conv1       border for the 3x3
                 pool pre-image)    is 1x1)             conv0, padded at edges)
```

The default pool tile is `2×64` (`--pool-tile-h/-w`), giving a `4×128` conv tile,
i.e. `tile_m = 4*128 = 512` conv-output pixels per CTA. The CTA runs the **entire
encoder_0 chain for its tile on chip**: conv0 over the staged input halo, the two
requants + ReLU, conv1, its requant + ReLU, the 2×2 maxpool, the final pack — and
only then writes the packed-int4 pooled tile to `Y`. Nothing between `X`/`W0`/`W1`
and `Y` ever touches HBM.

The grid is one tile per CTA: `grid = (1, h_tiles, w_tiles)` by default (the
`w_fast` layout transposes the two to `(1, w_tiles, h_tiles)`), where
`h_tiles = pool_ho / pool_tile_h`, `w_tiles = pool_wo / pool_tile_w`. At the full
target shape this is `(1, 540, 30)` = 16,200 CTAs — a
deep queue whose inter-CTA overlap is itself a free latency-hiding mechanism on
the small CU slice (see §9 and the case study's persistent-kernel discussion).

## 5. The matrix unit and the two convolutions as GEMMs

gfx1151 exposes three 16×16×16 (K=16) WMMA atoms in the catalog: the f16 atom
`wmma_f32_16x16x16_f16` (f32 accumulator) used by the fp16-emulation fallback,
and the native integer atoms `wmma_i32_16x16x16_iu8` / `wmma_i32_16x16x16_iu4`
(i32 accumulator) used by the default native-int regime. The f16 atom is also
the shape source for the shared warp grid in both regimes. Both convolutions are
mapped to implicit GEMM:

- **conv0** is a 3×3 contraction over `K_gemm = R*S*C = 72` channels — five
  16-wide K-atoms (last partial). The A operand is the input footprint, the B
  operand is `W0`.
- **conv1** is a 1×1 contraction over `K0 = 32` channels — **two** 16-wide
  K-atoms. The A operand is the conv0 output `C0`, the B operand is `W1`.

The contraction depth `K` here is the **channel count**, and it **cannot be
deepened** — it is fixed by the problem (72 for conv0, 32 for conv1). Two K-atoms
for conv1 is the structural fact that makes the whole kernel latency/overhead-
bound rather than matrix-throughput-bound (§9).

### Two execution regimes for the same math

- **Native-int (DEFAULT).** conv0 uses the integer atom
  `wmma_i32_16x16x16_iu8` (raw int8 → i8 LDS → exact i32 accumulation); conv1
  uses native packed-int4 / iu8 WMMA. The requant chain is integer shifts. This
  deletes all fp16 dequant overhead and shrinks LDS staging density (i8/i4 vs
  f16). Enabled with `--native-int` (default), `--no-native-int` selects the
  fallback.
- **fp16-emulation (FALLBACK, `--no-native-int`).** Integer codes are
  dequantized to fp16 and fed to the f16 WMMA with f32 accumulation. This is
  **bit-exact** to native integer MMA for these ranges, because the accumulator
  sums fit exactly in the f32 mantissa:
  - conv0: `|sum| <= 72 * 127^2 ≈ 1.16M < 2^24` — exact.
  - conv1: `|sum| <= 32 * 8^2 = 2048` — exact.

  The int4 codes `[-8,7]` and int8 codes `[-127,127]` are themselves exactly
  representable in fp16, so only the on-chip *storage dtype* is fp16; the numbers
  are integer-exact. (This is the same trick the shipped `matmul_nbits` uses:
  dequant int4 → fp16, then f16 WMMA.)

### The RDNA3.5 WMMA sub-ULP snap (a correctness requirement)

`wmma_f32_16x16x16_f16` is **not** bit-exact for exact-integer operands: it
carries ~`7.6e-6` (~`2^-17`) sub-ULP accumulator noise that can flip
round-half-even at an exact `.5` quant tie. The true accumulator value is a known
exact integer and `|noise| << 0.5`, so the kernel snaps each accumulator with
`rint_f32` before the requant chain. After the snap, the round-half-even result
matches the integer reference exactly. (The native-int path accumulates in i32
and never sees this noise; the snap is the fp16-path safeguard.)

## 6. One CTA, step by step

Fix a CTA owning a `pool_tile_h × pool_tile_w` pooled tile. The thread block is
`warp_m × warp_n` waves of 32 lanes (default `16×1`, `block_size = 512`).

### 6.1 — Stage the input halo (direct footprint cache, DEFAULT)
Stream the CTA's input region — the conv0 pre-image of the conv1 patch, plus a
1-pixel border for the 3×3 conv0, edge-padded — once into a small LDS footprint
cache. This is the `--direct` path: each input pixel is loaded **once**, in
contrast to im2col, which re-stages the overlapping 3×3 `R*S` footprint
redundantly. `--no-direct` selects the im2col A operand. The conv0 input global
loads are issued to distinct VGPRs *before* any `ds_store` (`--batch-loads`,
default on) so they coalesce under one `vmcnt(0)` and overlap.

### 6.2 — conv0 GEMM → requant → ReLU → requant
For each 16-wide K-atom along `K_gemm = 72`:

```
conv0_acc += WMMA( footprint[:, k-atom], W0[:, k-atom] )    # i32 (native) or f32
```

Then per output element: snap (fp16 path), `q0 = relu(Q_i8(conv0_acc * m0))`,
`C0 = Q_i4(q0 * m0b)` — the int4 conv0 output codes.

### 6.3 — Hand `C0` to conv1 in registers (`fused_c0a1`, DEFAULT)
conv1's A operand is `C0`. The naïve handoff scatters `C0` to an LDS buffer
(`c0_smem`), a full-WG barrier, then reads it back as the conv1 A-fragment. The
fused path eliminates that round-trip:

1. **Reorient conv0** so `W0` is the WMMA-A operand. Then conv0's accumulator
   lands with `lane = m`, `slot = k0` — i.e. the conv0 output channel index `k0`
   sits in the WMMA C slot.
2. conv1 needs `C0` as an **A** operand, which wants `k0` on the lane axis. So
   transpose `k0 ↔ m` **in-register** with `permlanex16` + `v_perm_b32` (the
   gfx11 FMHA C→A pattern), packing nibbles in the same pass.

This deletes `c0_smem`, its scatter, and the handoff barrier. It is bit-exact by
construction and requires `--native-int --direct` with `K0 % 16 == 0`. Enabled by
default (`--no-fused-c0a1` to disable). It is mutually exclusive with the
`--repack-c0` / `--packed-c0` handoff variants.

### 6.4 — conv1 GEMM → requant → ReLU
For each of the **two** 16-wide K-atoms along `K0 = 32`:

```
conv1_acc += WMMA( C0_A[k-atom], W1_B[k-atom] )
```

Then `C1 = relu(Q_i4(conv1_acc * m1))` — the int4 conv1 output codes. `W1` is
staged from HBM as the WMMA B operand (a free bitcast, since `W1` is static and
pre-packable). With `--conv1-int8` the contraction runs the **iu8** atom over the
int4-range codes (the 16 contiguous `k0` byte codes pass straight through as a
`<4 x i32>` iu8 A-fragment); the codes stay in int4 range so dot products are
**bit-identical** to the iu4 path, and the nibble-pack bitwork is deleted.

### 6.5 — 2×2 stride-2 maxpool over int4 codes
Reduce each 2×2 window of `C1` to its per-channel maximum. Under the WMMA
accumulator layout the four corners of a pool window land in **four different
lanes**, so the reduction is an LDS-gather maxpool (gfx1151 has no intra-lane
register fast path — see §8). With `--pk-maxpool` the codes are widened to i16
and the 2×2 max uses `vector.smax` (`llvm.smax.v<N>i16`) instead of per-channel
cmp/select.

### 6.6 — Final requant and packed-int4 store
`Y = Q_i4(M * mf)`; with `mf = 1` this is an identity clamp/pack. One thread
gathers a pooled pixel's `K1` channels from LDS and assembles `ceil(K1/8)` i32
words (8 signed nibbles each) with i32 bit-ops, then stores whole words to `Y`.

## 7. The whole kernel in pseudo-code

```
for each CTA (owns pooled tile [pth, ptw]):              # grid = (1, h_tiles, w_tiles)
    halo = load_input_region(X) -> LDS                   # direct footprint cache, once
    conv0_acc = 0
    for ka in range(0, K_gemm=72, 16):                   # conv0 3x3 implicit GEMM
        conv0_acc += WMMA(halo[:,ka], W0[:,ka])
    q0  = relu(Q_i8(snap(conv0_acc) * m0))               # int32->int8 requant + ReLU
    C0  = Q_i4(q0 * m0b)                                  # int8->int4 requant (codes)
    C0_A = permute_k0_to_m(C0)                            # in-register C0 -> conv1 A
    conv1_acc = 0
    for ka in range(0, K0=32, 16):                        # conv1 1x1, 2 K-atoms
        conv1_acc += WMMA(C0_A[ka], W1_B[ka])
    C1  = relu(Q_i4(snap(conv1_acc) * m1))               # int32->int4 requant + ReLU
    M   = maxpool_2x2_s2(C1)                              # LDS-gather max over codes
    Y[tile] = Q_i4(M * mf)                                # final requant -> packed int4
```

Everything between `X`/`W0`/`W1` and `Y` lives in registers and LDS. The only HBM
writes are the packed-int4 pooled tiles.

## 8. WMMA-vs-MFMA porting differences

The gfx950 sibling targets CDNA (MFMA, wave64). The fusion *math* is identical,
but the wave32/WMMA layout forces three structural differences:

| axis | gfx950 (CDNA, MFMA, wave64) | gfx1151 (RDNA3.5, WMMA, wave32) |
|---|---|---|
| C-fragment store | cshuffle-store vectorizable | **cannot vectorize.** WMMA C is `<8 x float>`, slot `i → row = 2i + lane//16, col = lane%16`: a lane owns a fixed column and stride-2 rows, non-contiguous in the row-major LDS tile, so the gfx950 cshuffle vectorization lever does not exist |
| maxpool | intra-lane register fast path | **LDS-gather only.** The 2×2 window's four corners land in four different lanes under the WMMA acc layout, so the gfx950 register fast path cannot port |
| occupancy | LDS/VGPR occupancy trade | **wave32, ~64 KB LDS/CU, one WG resident per CU** — warp count is a *free* latency-hiding lever, not an occupancy trade |
| C→A handoff | (MFMA layout) | `permlanex16` + `v_perm_b32` (the gfx11 FMHA C→A transpose) for `fused_c0a1` |

## 9. Where the algorithm ends and tuning begins

The algorithm above is fixed and exact. What makes the *engineering* interesting
is that this is a **tiny-GEMM, latency/overhead-bound** workload, not bandwidth-
bound and not matrix-throughput-bound:

- Arithmetic intensity ≈ **560 OP/byte** (~91 MB HBM for 51 GOP) — memory never
  binds.
- conv1's `K = 32` is **two** 16-wide K-atoms; `K` is the channel count and
  cannot be deepened, so the MMA/LDS operand pipeline cannot be amortized. An
  ablation pins ~**89.7%** of wall-clock in the conv1 GEMM operand pipeline; the
  rest (conv0 GEMM, both requant epilogues, maxpool, store) is ~10%.

Consequently the levers that win are **latency-hiding and redundant-LDS-traffic
removal**, not occupancy or VALU reduction:

- backward-planned tiling so no intermediate hits HBM (the core fusion, §4),
- the native-int pipeline and the direct footprint cache (§5–6.1),
- the in-register `fused_c0a1` handoff (§6.3),
- the conv1 ILP pair `conv1_prefetch_k` + `conv1_sched_fuse` — hoist all conv1
  B-fragment `ds_read`s before any MMA and fuse the per-k-step schedule barriers
  into one group, so the scheduler interleaves the k0/k1 MMAs and hides the
  exposed LDS-read latency at no VGPR cost (they must ship as a pair),
- max-waves warp geometry (`16×1`), wide-short pool tiles (`2×64`), and the
  `conv1_int8` / `pk_maxpool` operand-formation simplifications.

These change *only* how the steps above are scheduled onto the hardware, never
what is computed — every variant is gated bit-exact (`max_abs_diff = 0`) against
the integer numpy reference before any speed quote. The full lever-by-lever
campaign, including the dead ends (waves-per-eu=2, mask-maxpool, butterfly /
repack-c0, persistent grid-stride) and *why* each lost, is in
[`CASE-STUDY-optimizations.md`](CASE-STUDY-optimizations.md).
