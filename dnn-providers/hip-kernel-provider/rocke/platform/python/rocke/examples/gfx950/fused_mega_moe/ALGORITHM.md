# The fused-MoE mega-kernel, from the math up

This is the algorithm the kernel implements and *why* it is one kernel. The
README is the optimization history; this file is the specification, the data
layout, and the precise per-threadgroup steps.

## 0. Notation

| symbol | meaning |
|---|---|
| `T` | tokens in the batch |
| `E` | number of experts |
| `K` | top-k (experts chosen per token) |
| `H` | hidden (model) dim — the gate/up contraction and the down output |
| `I` | intermediate dim — the gate/up output and the down contraction |
| `X` | activations, `[T, H]` |
| `Wg, Wu` | gate / up weights, `[E, I, H]` (row = output `I`, contracted over `H`) |
| `Wd` | down weights, `[E, H, I]` (row = output `H`, contracted over `I`) |
| `Y` | output, `[T, H]` |
| `tile_m` | tokens per m-block a threadgroup processes (16) |
| `tile_n` | intermediate columns a threadgroup owns (`tile_n_inter`, 256) |
| `GROUP_K` | block-scale group size, 128 |
| `e4m3` | fp8 format; `FP8_MAX = 448` |

All matmuls accumulate in f32. fp8 operands carry **per-128-block scales**.

## 1. What a fused MoE computes (the specification)

A router picks, per token `t`, a set of `K` experts with weights `w_{t,e}`. For
each chosen `(t, e)` the token runs that expert's FFN, and the results are
combined by the routing weights:

```
Hidden_{t}   = silu( X_t · Wg_e^T ) ⊙ ( X_t · Wu_e^T )      # [I]
FFN_{t,e}    = Hidden_{t} · Wd_e^T                           # [H]
Y_t          = Σ_{e ∈ topk(t)}  w_{t,e} · FFN_{t,e}          # [H]
```

`silu(x) = x · σ(x)`, `σ(x) = 1/(1+e^{-x})`. The `⊙` is element-wise (SwiGLU).

The naïve realization is a pipeline of separate kernels (gate GEMM, up GEMM,
silu, down GEMM, reduce) that writes `Hidden` to HBM and reads it back. **The
mega-kernel does the whole per-token FFN in one launch and never spills `Hidden`
to HBM.**

## 2. Block-scale fp8 (the quantization lemma)

Weights and activations are fp8 e4m3 with one f32 scale per 128-element block.
For a block `b`, a stored fp8 value `q` represents `q · s_b`. A dot product over
a 128-block therefore dequantizes once, at the block boundary:

```
Σ_k (a_k · s_a) (b_k · s_b)  =  s_a · s_b · Σ_k a_k b_k
```

so the MFMA runs on raw fp8 and the f32 accumulator is multiplied by
`s_a · s_b` **after** each 128-wide K-block — never per element. This is why the
K=128 MFMA atom (§8.1) aligns perfectly with the scale granularity: one MFMA =
one block = one dequant.

The intermediate `Hidden` is produced in f32 and must be fp8 to feed the down
GEMM, so it is **dynamically quantized** (§5.4): per 128-inter-block take the
`amax` over **all `tile_m` rows of the block** (one scale per block, row-uniform,
NOT per row), set `s = max(amax, ε)/448`, store `q = round(h / s)`, and keep `s`
for the down dequant. The block (row-uniform) granularity is mandatory: the
down-GEMM dequant fold applies a single per-lane scalar to output slots that span
several different rows, so only a row-uniform block scale stays correct under that
fold.

## 3. The fusion idea (the heart)

Keep `Hidden` on chip. One threadgroup owns:

- `tile_m` sorted tokens of **one** expert (an *m-block*), and
- a `tile_n` slice of the intermediate dim `I`.

It computes that slice of `Hidden` for those tokens entirely in registers/LDS,
then immediately consumes it in the down GEMM. Because each threadgroup owns only
a `tile_n` slice of `I`, its down GEMM contracts only that slice and therefore
produces a **partial** sum over the full output `H`. Those partials — across the
`I`-slices (grid.x) and across the `K` experts a token was routed to — are
combined by **atomic add** into `Y`. Atomics make the contribution order
irrelevant, which is what lets the kernel be embarrassingly parallel over
`(I-slice, m-block)`.

## 4. From spec to kernel: who computes what

**Sorted / de-padded layout.** The router output is sorted so each expert's
tokens are contiguous, then grouped into `tile_m`-row *m-blocks*. Only **active**
blocks are emitted: `num_m_blocks = Σ_e ceil(count_e / tile_m)`. `BlockExpertIds[b]`
gives the expert for block `b`; `SortedTokenIds[b·tile_m + r]` gives the real
output-row token id for slot `r` (or `-1` for padding tail rows, which are
masked out of the atomic write); `SortedWeights` carries `w_{t,e}`.

**Grid.** `grid = (I / tile_n, num_m_blocks)`. Block = 256 threads (4 waves).
A threadgroup `(bx, by)` handles intermediate slice `bx` and m-block `by`.

**What lives where.**

| state | location |
|---|---|
| X tile `[tile_m, H]` | global→VGPR load (cheap, reused across the up/gate of a K-group; **not** LDS-staged by default) |
| gate / up weight tiles `Wg/Wu` | staged global→LDS→MFMA (direct-to-LDS, the dominant weight stream; default `use_dtla=True`) |
| gate / up accumulators | f32 registers (AGPR/VGPR) |
| `Hidden` slice `[tile_m, tile_n]` | LDS (fp8, after dynamic quant) |
| down accumulator | f32 registers |
| `Y` | HBM, written by atomic add (f32) |

## 5. One threadgroup, step by step

Let `e = BlockExpertIds[by]`, the expert this block serves. The contraction `H`
is walked in `GROUP_K=128` chunks.

### 5.1 — Load the X fragment
The `[tile_m, H]` activation is a small, repeatedly-reused operand, so by default
it is a cheap global→VGPR load: per K-group each lane loads its `a_per_lane` fp8
bytes once and reuses that fragment across **both** the gate and the up MFMAs (and
across every intermediate-column cell `ni` of the row). The dominant traffic — the
gate/up **weight** tiles (§4) — is what goes through the direct-to-LDS path
(`use_dtla`, default on): staged HBM→LDS with `global_load_lds`, ping-pong
double-buffered over `ni`, then read back with `ds_read` to feed the MFMA. (An
*X*-via-LDS variant
exists behind the `ROCKE_FP8_X_DTLA` flag but is default-off — it measured slower
because the extra LDS round-trip adds a drain the scheduler cannot hide.)

### 5.2 — Gate and up GEMMs (shared A)
For each 128-block `c` along `H`:

```
gate_acc += MFMA_16x16x128( X[:, c],  Wg_e[bx-slice, c] )   # f32
up_acc   += MFMA_16x16x128( X[:, c],  Wu_e[bx-slice, c] )
```

both consuming the same X block. After each block, dequant the new contribution
by `s_X[c] · s_Wg[e, slice, c]` (and `s_Wu`) per the §2 lemma.

### 5.3 — SiLU·up (the activation)
In f32 registers, element-wise:

```
h = silu(gate_acc) · up_acc          # [tile_m, tile_n], f32
```

### 5.4 — Dynamic-quantize `Hidden` to fp8
Per 128-inter-block (one scale per block, reduced over **all `tile_m` rows** of
the block — row-uniform, NOT per-row; §2): `amax = max|h|` is folded into the
SiLU·up pass — each lane returns the abs-max over the cells it owns, a 64-lane
butterfly collapses that to the warp amax, and the warps sharing a block are
combined (no separate LDS re-read sweep for the amax itself). Then
`s_h = max(amax, ε)/448`, `q = round(h / s_h)`, broadcast to every row of the
scale scratch. The quantize itself re-reads the f32 `Hidden` from an LDS scratch
(`HiddenF32_smem`) in a separate packed pass (`cvt_pk_fp8_f32x4`, 4 columns/iter)
and stores `q` (fp8) into the `Hidden` LDS buffer, keeping `s_h` for the down
dequant. (Folding the amax into the SiLU·up pass — rather than a separate sweep
over LDS — is one of the kept levers.)

### 5.5 — `Hidden` LDS hand-off (implicit reshape, no transpose)
There is **no explicit LDS transpose or swizzle**. The dynamic-quant pass writes
each fp8 `Hidden` value at the logical `(m, inter)` cell, and the down GEMM reads
its A operand from that *same* `(m, inter)` cell of the LDS buffer — the quant
write address equals the down-MFMA A-read address. The reshape from the gate/up
MFMA C-output layout into the down-GEMM A-input layout is therefore *implicit* in
the addressing (BUILD_SPEC_FP8 §3.5), accomplished by the write/read addressing
itself rather than by a separate swizzled write/read pair.

### 5.6 — Down GEMM (contract the I-slice this block owns)
For each 128-block `c` along this threadgroup's `tile_n` slice of `I`:

```
down_acc += MFMA_16x16x128( Hidden[:, c],  Wd_e[:, slice·tile_n + c] )
```

dequantized by `s_h · s_Wd[e, …, c]`. Because the block owns only `tile_n` of
`I`, `down_acc` is a **partial** sum over the full output `H`.

### 5.7 — Weight and atomic-accumulate
For each output row `r` of the block with token id `t = SortedTokenIds[by·tile_m+r]`
(skip if `t = -1`):

```
Y[t, :]  +=  w_{t,e} · down_acc[r, :]        # atomic add, f32
```

The atomic add merges (a) the partial sums from the other `I`-slices (other `bx`)
and (b) the contributions of the other experts the token was routed to.

## 6. The output / epilogue

There is no separate reduction kernel. `Y` is an **f32** buffer (the atomic
epilogue is reused unchanged from the f16 kernel), zeroed once before the launch;
every threadgroup atomic-adds its weighted partial. After the launch `Y` holds
`Σ_{e∈topk} w_{t,e} · FFN_{t,e}` exactly. The mega-kernel itself emits f32 — it
does **not** contain an in-kernel bf16 cast; producing a bf16 result is a
downstream cast outside this kernel.

## 7. The sorted, active-block grid (the structural win)

A fixed grid over *all* experts would launch `tile_m`-row blocks for inactive
experts (pure padding) — at decode that is the dominant waste. Emitting only the
`num_m_blocks` *active* blocks (and skipping `-1` padding rows in §5.7) makes the
launched work proportional to real tokens. At `T=1` this shrinks the grid from
`(I/tile_n, E)` to `(I/tile_n, 2)` — the single biggest decode lever.

## 8. Implementation details worth the math

### 8.1 K=128 hero atom
The MFMA is `f32 = 16×16×128` over fp8 (`mfma_scale_f32_16x16x128_f8f6f4`, scale
exponents pinned to 0 = unscaled). K=128 per instruction means **4× fewer K-loop
trips** than a 16×16×32 atom, and — by §2 — one MFMA covers exactly one 128-wide
scale block, so dequant is one f32 multiply per MFMA.

### 8.2 The `m_tile_base` condition (a correctness requirement)
When a block spans more than one MFMA m-tile (an expert with `> tile_m` tokens,
or `tile_m > 16`), the down GEMM's LDS A-read must offset by the real m-tile
index `mi·16`, not a constant 0. Reading row 0–15 for every m-tile silently
corrupts tokens 16+. The parity gate therefore includes a skewed expert with
`> tile_m` tokens.

### 8.3 Persistent dispatch
Optionally launch a fixed resident grid and loop each threadgroup over several
`(bx, by)` work-items, re-initializing the accumulators / quant scales / barriers
per item. The atomic-add in §5.7 makes work-item order irrelevant, so this is a
pure scheduling change that amortizes per-launch overhead.

## 9. The whole kernel in pseudo-code

```
Y = 0
for each threadgroup (bx, by):                  # grid = (I/tile_n, num_active_m_blocks)
    e   = BlockExpertIds[by]
    gate_acc = up_acc = 0
    for c in range(0, H, 128):                    # gate + up, shared A
        Xc = load X[block by rows, c] -> VGPR     # cheap global->VGPR, reused by gate+up (Wg/Wu go via LDS)
        gate_acc += MFMA(Xc, Wg[e, bx-slice, c]);  dequant by s_X·s_Wg
        up_acc   += MFMA(Xc, Wu[e, bx-slice, c]);  dequant by s_X·s_Wu
    h   = silu(gate_acc) * up_acc                 # f32, [tile_m, tile_n]
    s_h = max(amax_per_block(h), eps)/448         # in-register amax, one scale/128-block over ALL rows
    Hs  = quantize(h, s_h) -> LDS (fp8)           # written at logical (m,inter); down reads same cell (no transpose)
    down_acc = 0
    for c in range(0, tile_n, 128):               # down: contract this I-slice
        down_acc += MFMA(Hs[:,c], Wd[e, :, bx·tile_n + c]); dequant by s_h·s_Wd
    for r in range(tile_m):                        # weight + scatter
        t = SortedTokenIds[by*tile_m + r]
        if t != -1:
            atomic_add(Y[t, :], SortedWeights[...] * down_acc[r, :])   # Y is f32
# (a bf16 result, if needed, is a downstream cast outside this kernel)
```

## 10. Where the algorithm ends and tuning begins

The algorithm above is fixed; the README's levers (tiling, prefetch depth,
scheduling cadence, active grid, persistent loop) change *only* how these steps
are scheduled onto the hardware, never what is computed. Correctness is pinned by
the hardened parity gate (§8.2); performance is the per-threadgroup schedule.
