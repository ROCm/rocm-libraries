# The activation gather/rescale prologue, from the math up

This is the algorithm the kernel implements and *why* it has to exist. The README
is what it produces and how to run it; this file is the specification, the data
layout, and the precise per-threadgroup steps.

Source of truth: `platform/python/rocke/instances/common/moe_gather_rescale_a.py`.

## 0. Notation

| symbol | meaning |
|---|---|
| `T` | tokens in the batch |
| `E` | number of experts |
| `K` | top-k (experts chosen per token) |
| `H` | hidden (model) dim; the gate/up contraction |
| `GROUP_K` | block-scale group size along `H`, **128**, fixed by the MoE kernels |
| `n_hb` | `H / GROUP_K`, the number of scale groups per row |
| `tile_m` | rows per expert block (16) — must match the consuming GEMM |
| `n_flat` | `T · K`, the number of real `(token, slot)` pairs |
| `n_blocks` | number of `tile_m`-row blocks the caller's alignment produced |
| `e4m3` | fp8 format; `FP8_MAX = 448` |

A quantized value is a pair: an fp8 byte `q` and an f32 scale `s`, representing
`q · s`. The scale's *granularity* — which elements share one `s` — is the whole
subject of this document.

## 1. What the prologue computes (the specification)

The caller arrives with activations already quantized **per token**: `Aq [T, H]`
in fp8, `AqScale [T, n_hb]` in f32, one scale per `(token, 128-group)`. It also
has a routing plan, block-aligned by an upstream sort: `SortedIds
[n_blocks·tile_m]` holding flattened `(token, slot)` ids — `id = token·K + slot` —
with any value `≥ n_flat` meaning "pad row".

The prologue emits the same activations in the layout the mega-kernel reads:

```
for each block  m ∈ [0, n_blocks),  group  g ∈ [0, n_hb):

    rows(m)      = { r ∈ [0, tile_m) : SortedIds[m·tile_m + r] < n_flat }
    s_block[m,g] = max( max_{r ∈ rows(m)} AqScale[tok(r), g],  1e-30 )

    A[m·tile_m + r, :]  =  round_fp8( Aq[tok(r), :] · ( s_row[r,g] / s_block[m,g] ) )
    AScale[m·tile_m + r, g]  =  s_block[m,g]            # every row, pads included
```

with `tok(r) = SortedIds[m·tile_m + r] / K` and pad rows producing exact zeros.

The identity that makes this legal is one line:

```
q · s_row  ==  ( q · s_row/s_block ) · s_block
```

The dequantized value is unchanged. What changes is only *which* fp8 grid the
value is snapped to — and, as a consequence, the layout constraint below is
satisfied. Because `s_row ≤ s_block` by construction, the ratio lies in `[0, 1]`
and the re-round can never overflow e4m3's range; it can only lose low-order
precision.

## 2. Why block-uniform (the constraint that forces this kernel)

The mega-kernel's gate/up GEMM dequantizes with a fold: after each 128-wide
K-group it multiplies the f32 accumulator by `s_a · s_b`, once, instead of
touching every element (see `../fused_mega_moe/ALGORITHM.md` §2). The activation
scalar `s_a` is fetched per lane with the row index of that lane's **A fragment**:

```
m_row        = m_tile_base + lane_decode.m_in_atom
a_scale_off  = m_row · stride_a_scale + kg
```

For the 16×16 MFMA output, the row a lane reads for its A fragment and the rows a
lane *owns in the accumulator* are two different things: lane `l` supplies A row
`l % 16`, but holds accumulator slots for rows `4·(l/16) + 0..3`. So that one
scalar, chosen by the A-fragment row, is applied to four output rows that are not
it.

If the scale is uniform across the rows of a block, this is harmless — every
candidate row would have given the same number. If it is per-token, the fold
multiplies each output row by some *other* token's scale. The addresses are all
legal and no bound is violated, so there is no error: the result is just quietly
wrong, by a factor that depends on how much the tokens in a block differ in
magnitude. That is a bug class you find in an accuracy regression weeks later, not
at the call site.

The same constraint governs the mega-kernel's own intermediate: its dynamic-quant
epilogue reduces `amax` over all `tile_m` rows of a block and broadcasts one scale
to every row, for exactly this reason. This kernel is that rule applied to the
*input* side, which the mega-kernel cannot do for itself because it never sees a
token's rows together in one place before it needs the scale.

Hence the two-sided contract, and why `tile_m` is a spec field rather than an
implementation detail: **the prologue's `tile_m` must equal the consuming GEMM's
`tile_m`**, because that is the row set the scale is uniform over. A mismatch
reintroduces the identical silent failure.

## 3. Data layout

Inputs (all `readonly`, `noalias`):

| name | shape | type |
|---|---|---|
| `Aq` | `[T, H]` row-major | fp8 e4m3 |
| `AqScale` | `[T, n_hb]` row-major | f32 |
| `SortedIds` | `[n_blocks·tile_m]` | i32, `≥ n_flat` = pad |
| `TopkWeights` | `[n_flat]` indexed by flat id | f32 |

Outputs:

| name | shape | type |
|---|---|---|
| `A` | `[n_blocks·tile_m, H]` row-major | fp8 e4m3 |
| `AScale` | `[n_blocks·tile_m, n_hb]` row-major | f32 |
| `SortedTokenIds` | `[n_blocks·tile_m]` | i32, `-1` = pad |
| `SortedWeights` | `[n_blocks·tile_m]` | f32, `0` = pad |

Runtime scalars: `n_flat`, `topk`, `hidden`, `n_hb`.

`A` and `AScale` are the GEMM's A operand and its scale array. `SortedTokenIds`
and `SortedWeights` are the scatter metadata the mega-kernel's epilogue uses to
weight and atomically accumulate each row into `Y`; the prologue converts the
caller's flat-id-and-sentinel convention into the token-id-and-`-1` convention the
epilogue expects, since it is already reading `SortedIds` anyway.

The grid is **one workgroup per expert block**: `moe_gather_rescale_a_grid`
returns `(n_blocks, 1, 1)`. A block is `block_size` = 256 threads = 4 waves. No
workgroup reads another's rows and no output cell is written twice, so the whole
kernel needs no atomics and no global synchronization.

### LDS

Three arrays, at the default spec (`tile_m=16`, `max_n_hb=32`):

| array | shape | bytes | lifetime |
|---|---|--:|---|
| `tok` | `i32 [tile_m]` | 64 | phase 0 → 4 |
| `rowscale` | `f32 [tile_m, max_n_hb]` | 2048 | per-token scale (1→2), then ratio (3→4) |
| `blkscale` | `f32 [max_n_hb]` | 128 | phase 2 → 3 |

2240 B total — small enough to be irrelevant to occupancy, which is the point:
this kernel is bandwidth-bound and wants as many waves resident as the memory
system will feed.

`rowscale` is deliberately reused for two different quantities rather than
allocated twice. `max_n_hb` exists only to size it at build time; `hidden` stays a
runtime argument so one compiled kernel serves every model width up to the bound.

## 4. One threadgroup, step by step

Four phases, four barriers. Every phase is a grid-stride loop over the workgroup,
so `block_size` never has to divide the work.

### 4.1 — Phase 0: row metadata (`tid < tile_m`)

One thread per row of the block reads `SortedIds[row]`, classifies it against
`n_flat`, and derives `tok = id / topk`. Validity is then carried **as a negative
`tok`** in LDS, which is why no later phase needs a separate predicate array.

The pad-row read is index-guarded rather than branch-guarded: `sid_safe =
select(valid, sid, 0)` keeps the `TopkWeights` load in bounds unconditionally, so
all `tile_m` lanes issue the same load with no divergence, and the *value* is
masked to zero afterwards. Both scatter arrays are written here, and never
touched again.

### 4.2 — Phase 1: gather per-token scales

Grid-stride over the `tile_m · n_hb` scale cells. Cell `(r, g)` loads
`AqScale[tok(r), g]` into `rowscale[r, g]`, or **zero** if row `r` is a pad. Zero
is the identity for the max in the next phase, so pad rows cannot inflate the
block scale — which they otherwise would, since they carry whatever token 0's
scale happens to be.

This is a scattered gather of at most `tile_m · n_hb` f32 (2 KB), which is why the
scales are staged in LDS once and read `tile_m`-times from there rather than
re-read from HBM.

### 4.3 — Phase 2: reduce to the block scale

One thread per group `g` takes the max down the `tile_m` rows of `rowscale[:, g]`.
The row loop is unrolled at build time (`tile_m` is a spec constant), so this is a
straight-line chain of `ds_read` + `v_max_f32` with no loop overhead and no
cross-lane traffic — each group is reduced entirely within one thread, and groups
are independent.

The result is clamped: `max(acc, 1e-30)`. A block whose rows are **all** pads —
which the caller's block alignment can produce at the tail — would otherwise
publish a scale of exactly zero, and the consumer divides by the scale.

### 4.4 — Phase 3: publish `AScale`, fold the rescale into a ratio

Grid-stride over the same `tile_m · n_hb` cells. Each cell writes the block scale
to `AScale[row_base + r, g]` — for **every** row, pads included, because the fold
indexes that array by whatever row a lane happens to hold and must find the block
scale there — and then overwrites `rowscale[r, g]` in place with the ratio
`s_row / s_block`.

Overwriting is safe: phase 2's readers are behind a barrier, and each cell is
rewritten by the single thread that owns it.

Folding the division into a ratio *here* rather than in the inner loop is the
whole reason the rescale is nearly free. Phase 4 touches `GROUP_K = 128` elements
per `(r, g)` cell; computing the ratio once per cell makes it a 1-in-128 cost.

For a pad row the ratio is `0 / s_block = 0`, which is what drives `A` to exact
zeros there without a branch in the inner loop.

### 4.5 — Phase 4: the gather

Rows are unrolled at build time. For each row `r`:

```
col0   = tid · vec
stride = block_size · vec
for col in range(col0, hidden, stride):
    ratio = rowscale[r, col / GROUP_K]              # LDS scalar
    v     = load  vec fp8 from Aq[tok(r)·H + col]   # one dwordx2 per lane
    out   = round_fp8( f32(v) · ratio )             # vec-wide, in registers
    store out to A[(row_base + r)·H + col]
```

Three properties make this the whole loop body, with no tail and no predication:

* `vec` **divides `GROUP_K`** (enforced in `__post_init__`), so a vector can never
  straddle two scale groups and one scalar covers all `vec` elements. This is the
  only reason the LDS read is hoistable out of the element math.
* `hidden` is a multiple of `GROUP_K`, so a full vector is always in bounds.
* `tok` is clamped with `smax(tok, 0)`, so a pad row reads token 0's real bytes —
  in bounds, and multiplied by a ratio of zero. Cheaper than skipping the row, and
  branch-free.

The 256 threads of a wave-group cover `256 · 8 = 2048` contiguous bytes per
iteration on both the load and the store, so both sides are fully coalesced. The
widening to f32 and the re-round back to fp8 live entirely in registers between
them; nothing intermediate is ever written to memory. That is the structural
advantage over doing this in a framework, which must materialize the f32.

The only division in the inner loop is by `GROUP_K`, a compile-time power of two —
the runtime `n_hb` division was confined to phases 1 and 3 precisely so it would
not appear here.

## 5. The exact divide (a numerical requirement, not a preference)

The ratio in phase 3 is emitted as an IEEE `fdiv`, **not** the fast hardware
reciprocal (`v_rcp_f32`, ~1 ULP) that a reciprocal-and-multiply would give.

The reason is the re-round in phase 4. `round_fp8` is round-to-nearest-even on a
grid with 3 mantissa bits, so its decision boundaries are dense in relative terms:
plenty of products land close enough to a tie that a 1-ULP perturbation of the
*ratio* moves them across a boundary and changes the stored byte. The effect is
tiny and non-catastrophic — a handful of differing bytes out of millions — which
is exactly what makes it worth pinning down. It passes any tolerance-based
comparison, so it does not show up as a failure; it shows up much later as this
kernel and the framework fallback disagreeing on results that are each
individually plausible, with no obvious suspect.

The usual argument for `rcp_fast` does not apply, because the ratio is computed
**once per `(row, group)` cell and reused across all 128 elements of the group**.
Whatever a full-precision divide costs, it is amortized over 128 multiply-convert
pairs and a 128-byte load/store pair; it is not measurable against the memory
traffic. Paying it buys byte-exact agreement with a reference implementation, which
is the only kind of agreement worth having when the failure mode is silent.

`verify_gather_rescale.py` compares `A` byte for byte for this reason. A
tolerance-based check on this kernel is close to worthless.

## 6. What the restatement costs

Restating a row under a scale larger than its own loses precision: the row's
values now sit lower in e4m3's relative grid. The loss is bounded by the spread of
per-token magnitudes within a block, and is zero when a block holds a single real
row (its scale *is* the block scale, so the ratio is exactly 1 — a case the
verifier exercises).

This is not overhead the kernel adds; it is the price of the block-uniform layout
the mega-kernel's fold requires, and the mega-kernel already pays the same price
on its intermediate. The alternative is not "more precision" — it is a per-token
scale that the fold mis-applies.

## 7. The whole kernel in pseudo-code

```
grid  = (n_blocks, 1, 1)          # one workgroup per expert block
block = (block_size, 1, 1)        # 256 threads, 4 waves

row_base = blockIdx.x * tile_m

# ---- phase 0: row metadata + scatter arrays
if tid < tile_m:
    sid   = SortedIds[row_base + tid]
    valid = sid < n_flat
    tok   = valid ? sid / topk : -1                  # negative encodes "pad"
    lds.tok[tid] = tok
    SortedTokenIds[row_base + tid] = tok
    SortedWeights [row_base + tid] = valid ? TopkWeights[sid] : 0
barrier()

# ---- phase 1: per-token scales into LDS
for cell in grid_stride(tile_m * n_hb):
    r, g = divmod(cell, n_hb)
    lds.rowscale[r, g] = (lds.tok[r] >= 0) ? AqScale[lds.tok[r], g] : 0
barrier()

# ---- phase 2: one scale per group, over the block's rows
for g in grid_stride(n_hb):
    lds.blkscale[g] = max( max_{r < tile_m} lds.rowscale[r, g], 1e-30 )
barrier()

# ---- phase 3: publish AScale, fold rescale into a ratio (in place)
for cell in grid_stride(tile_m * n_hb):
    r, g = divmod(cell, n_hb)
    AScale[(row_base + r) * n_hb + g] = lds.blkscale[g]     # every row
    lds.rowscale[r, g] = fdiv(lds.rowscale[r, g], lds.blkscale[g])   # exact
barrier()

# ---- phase 4: gather + rescale + re-round
for r in unrolled(tile_m):
    src = max(lds.tok[r], 0) * hidden
    dst = (row_base + r) * hidden
    for col in range(tid * vec, hidden, block_size * vec):
        ratio = lds.rowscale[r, col / GROUP_K]
        A[dst + col : +vec] = to_fp8( to_f32(Aq[src + col : +vec]) * ratio )
```

## 8. Spec knobs, and where the algorithm ends

`MoeGatherRescaleSpec` fields, and what each one actually constrains:

| field | default | constraint |
|---|--:|---|
| `tile_m` | 16 | **Correctness, not tuning.** The row set the scale is uniform over; must equal the consuming GEMM's `tile_m` (§2). Also sets the phase-2 unroll depth and the phase-4 row unroll. |
| `max_n_hb` | 32 | Static LDS sizing only. Caller must ensure `hidden/128 ≤ max_n_hb`. |
| `block_size` | 256 | Waves per workgroup. Trades parallelism within a block against the number of resident blocks. |
| `vec` | 8 | Bytes per thread per access. Must divide `GROUP_K` (§4.5), enforced in `__post_init__`. |

Every knob is a spec field frozen into the kernel name
(`rocke_moe_gather_rescale_a_tm16_b256_v8_h32`), so a build is reproducible from
its name alone and two configurations cannot collide in a cache. There are no
environment variables: nothing about the emitted code depends on the state of the
process that built it.

Two of the four are correctness contracts rather than tuning levers, which is most
of what there is to say about tuning this kernel. It moves `2 · tile_m · hidden`
bytes per workgroup with perfectly coalesced access on both sides and ~2 KB of
LDS; the arithmetic between the load and the store is a convert, a multiply, and a
convert. There is no reuse to exploit and no tiling decision to make. If it is not
running at memory bandwidth, the reason will be occupancy or launch overhead, not
the algorithm.
