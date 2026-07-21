# The square GEMM, from the math up

This is the algorithm the kernel computes and *why* its operand-streaming
schedule is shaped the way it is on **MI355X / gfx950 (CDNA4)**. The
[`README.md`](README.md) is the runnable field guide and the optimization ladder;
this file is the specification, the data layout, and the precise reasoning behind
each streaming lever — especially the LDS bank-conflict arithmetic, which is the
heart of the study.

> **A note on the directory name.** The directory is named `warpspec`, but the
> shipped kernel is the production universal-GEMM body driven at one fixed
> geometry with a sequence of `TraitSpec` levers — *not* a producer/consumer
> warp-specialized pipeline. No warp-specialization is implemented or claimed. The
> "ladder" in this study varies only *how operands are streamed past the matrix
> unit*, never the accumulation math.

## 0. Notation

| symbol | meaning |
|---|---|
| `A` | left operand, row-major `[M][K]` (the "R" of RCR) |
| `B` | right operand, column-major `[N][K]` — the kernel computes `A·Bᵀ` (the "C") |
| `C` | output, row-major `[M][N]` (the "R") |
| `M, N, K` | the three GEMM dimensions; here all equal `S = 8192` |
| `tile_m, tile_n, tile_k` | macro-tile a workgroup owns: `256 × 256`, walking K in `64`-half steps |
| `warp_m × warp_n` | the warp grid inside the macro-tile: `4 × 4` (warp_k = 1) |
| atom | the matrix instruction `v_mfma_f32_16x16x32` (wave64, K = 32 halves) |
| half | one `fp16` element (2 bytes); the LDS unit of accounting |
| bank | one of the 64 LDS banks on gfx950 (CDNA4); a bank-word is 4 bytes = 2 halves. The wide `ds_read_b128` the MFMA operand reads use has a 64-dword conflict period on gfx950 (vs 32 on gfx942/gfx90a) |
| WG/CU | workgroups resident per compute unit (occupancy) |

The accumulator is `fp32` throughout; inputs and output are `fp16`. Layout is
**RCR**: A row-major, B column-major, C row-major.

## 1. What the kernel computes (the specification)

A dense matrix product, one workgroup per `256 × 256` output macro-tile:

```
C = A · Bᵀ        A: [M][K] (row-major)   B: [N][K] (col-major)   C: [M][N]
C[m][n] = Σ_k  A[m][k] · B[n][k]            accumulate in f32, store f16
```

This is the entire mathematical object. Every dimension is `8192`, so it is a
square problem; the accumulation order is fixed for the whole study. Nothing in
the ladder changes what is computed — each rung is bit-exact (`bad=0`) against the
same reference. The study is **entirely about the schedule that feeds operands to
the matrix unit**.

The matrix unit on CDNA4 is `v_mfma_f32_16x16x32`: it multiplies two `16×32`
`fp16` fragments and accumulates into a `16×16` `fp32` tile, with `K = 32` halves
contracted per instruction. The macro-tile's `64`-half K-step is therefore two
atom K-passes. The `4×4` warp grid splits the `256×256` macro-tile into `64×64`
per-warp regions, so each warp owns a `4×4` block of `16×16` atoms
(`mfmas_per_warp_m × mfmas_per_warp_n = 256/(4·16) × 256/(4·16) = 4 × 4`) and
issues those `16` output-tile MFMAs per K-pass. Pad_m / pad_n / pad_k are all on,
so the kernel is correct for any `S`; the fixed `8192³` shape divides evenly.

## 2. Why this geometry, and why one workgroup per CU

The macro-tile is `256 × 256 × 64`, `4 × 4` warps, which puts **one workgroup per
CU**. That choice is settled by measurement, not assumed — and it is the fact that
makes the rest of the schedule (especially §4's single barrier) the right design:

- A `256 × 256` macro-tile maximizes operand **reuse**: each loaded A-row and
  B-column feeds a full `256`-wide strip of MFMAs before it is evicted. Smaller
  tiles re-read the same operands more often.
- A `64`-half (`tile_k`) K-step gives the K-loop enough MFMA work per trip
  (two atom K-passes × each warp's `4×4` atom block = `32` MFMAs/warp/trip) that
  per-trip overhead — waitcnts, barriers, loop control — is amortized. A `32`-half
  step halves the MFMAs per trip and the fixed per-trip cost dominates.
- The two together fill the register file and LDS so completely that only **one**
  workgroup fits per CU. Higher-occupancy alternatives (smaller tiles → 2+ WG/CU)
  all measure *slower* (README's measured-negatives table), because the reuse loss
  and the barrier-per-FLOP increase outweigh any latency the second workgroup
  could hide.

The single-workgroup occupancy is not just a consequence of the tile size — it is
the premise of §4. With no second workgroup to overlap, every barrier bubble is
fully exposed, so the number of barriers in the mainloop is a first-order cost.

## 3. Direct-to-LDS: removing the staging round-trip

The stock tile pipeline stages every operand tile **global → VGPR → LDS**: a
`buffer_load` into registers, then a `ds_write` from those registers into LDS,
then the MFMA reads operands back from LDS. That is three hops, and the staging
VGPRs are live for the whole transfer.

CDNA4 can stream global memory **straight into LDS** in one instruction
(`buffer_load_dwordx4 ... lds`). `TraitSpec.direct_to_lds` switches the loaders to
this path:

```
global ──buffer_load … lds──▶ LDS ──ds_read──▶ MFMA
```

The `ds_write` and the staging VGPRs both vanish — the emitted mainloop has
**0 `ds_write`** (ISA-confirmed). This is a small win on its own, but it is the
foundation the prefetch (§4) and the swizzle (§5) both build on: both operate on
the LDS buffers that direct-to-LDS fills.

`direct_to_lds` requires `block_k` to be a multiple of the direct-to-LDS half
granularity (the spec raises otherwise); `tile_k = 64` satisfies it.

## 4. Depth-2 prefetch with a single mainloop barrier

This is the largest single rung in the ladder, and the reason is entirely the
single-workgroup occupancy of §2.

**Depth-2 ping-pong.** Allocate two LDS half-buffers. While the MFMAs consume
half `p`, the next K-tile streams asynchronously into half `p^1`. The MFMA work
of this iteration overlaps the HBM→LDS transfer of the next.
`TraitSpec.dtl_prefetch` enables this (and requires `direct_to_lds=True` — the
spec raises `ValueError` otherwise, because there is nothing to ping-pong without
the fused load).

**One barrier per K-tile, not two.** A double buffer has two hazards:

- a **read-after-write** hazard — this iteration must not MFMA-read half `p`
  until the load that filled it has *landed*;
- a **write-after-read** hazard — this iteration must not overwrite half `p^1`
  until the *previous* iteration's reads of `p^1` have *drained*.

The naïve structure spends a separate `s_barrier` on each. At **1 WG/CU there is
no second workgroup to hide a barrier bubble**, so a second barrier directly
subtracts from the time the MFMA unit is running. The fix is *ordering*: issue the
next-tile async write **after** the single drain, so one waitcnt plus one bare
barrier cover both hazards:

```
for each K-tile (parity p, other half p^1):
    s_waitcnt(vmcnt=0, lgkmcnt=0)   # vmcnt0:  the load into half(p) has LANDED   (RAW)
                                    # lgkmcnt0: prior reads of half(p^1) DRAINED  (WAR)
    s_barrier                       # one workgroup rendezvous: half(p) now visible
    load  -> half(p^1)             # async; issued AFTER the drain, so it cannot
                                    #   race the just-finished reads — no 2nd barrier needed
    mma   <- half(p)               # matrix work overlaps the async HBM transfer
```

The single `s_waitcnt` retires both counters, the single `s_barrier` makes the
just-landed half visible to all waves, and issuing the next async write *after*
the drain means it provably cannot collide with the reads it would otherwise race.
Collapsing two barriers to one is the biggest jump in the ladder precisely because
each barrier is fully exposed at 1 WG/CU.

## 5. Killing LDS bank conflicts (the arithmetic)

With the mainloop tight, the matrix unit was still starving: the counter read
**52% `LDSBankConflict`** while `MfmaUtil` sat at ~35%. The MFMAs were waiting on
conflicted LDS reads. Removing those conflicts is the second-largest part of the
climb, and it is pure arithmetic — no layout change, bit-exact.

### 5.1 Why the conflict exists (a stride alias)

The A tile is row-major `[M][K]` with `K = 64` halves = `128` bytes = `32`
`4`-byte bank-words. gfx950 has `64` LDS banks, and the wide `ds_read_b128` the
MFMA operand reads use has a `64`-dword (bank-word) conflict period on CDNA4. So
a row spans exactly **half** the conflict period, and the bank a dword hits is

```
bank = (row·K + col)·2 / 4  mod 64          (·2 bytes/half, /4 bytes/bank-word)
     = (row·64 + col)/2      mod 64
     = (row·32 + col/2)      mod 64          ← row·32 alternates 0 / 32 (mod 64)
```

Because the row stride (`32` bank-words) divides the conflict period, the `row`
term collapses to a two-state value — `0` for even rows, `32` for odd rows — so
all even M-rows of an atom land on one bank half and all odd M-rows on the other,
each determined only by `col`. The `16` M-rows an atom reads in parallel therefore
collapse onto just two bank sets — a deterministic, heavily-conflicting *stride
alias* baked into the row stride, not a random layout accident. The measured cost
is a **52% `LDSBankConflict`** counter (§5 intro).

### 5.2 The tool is a swizzle, not padding

XOR a function of the row into the column address before computing the bank:

```
bank = (col ^ f(row))/2  mod 64
```

Now `f(row)` re-introduces a row-dependent term that the modulo cannot erase, so
different rows scatter across different banks. Two properties make this *free* and
*exact*:

- **It is a bijection (bit-exact).** On the direct-to-LDS path the swizzle is
  applied to the *global element address* the loader fetches into each
  lane-linear LDS slot (`col ^ f(row)`, see `gemm_universal.py` `_swz_col` at the
  A/B `async_buffer_load_lds_addr` sites), and the MFMA `ds_read` then asks LDS for
  the same `col ^ f(row)` — the identical XOR on both sides cancels (XOR is its own
  inverse). The data that comes back is identical to what would have come back
  without the swizzle. No layout reinterpretation, no correction pass: every rung
  in the ladder is `bad=0`.
- **It costs zero LDS.** Unlike padding the row stride (§5.4), the swizzle only
  permutes *which* element occupies each existing LDS slot; it does not widen the
  buffer.

Two parameters decide how far rows spread:

- **Granularity.** Swizzle whole `ds_read_b128` slots (`8` halves = bit 3), so
  every target stays 16-byte aligned and the wide LDS read still issues as one
  `ds_read_b128`.
- **Which row bits, how many slots.** The aliasing dimension is `m_in_atom`
  (`0…15`) — the **low** row bits — and there are `block_k / 8 = 8` slots to
  spread into.

### 5.3 Rung D vs rung E — wrong bits vs right bits

**Rung D — a coarse swizzle on the wrong bits.** Toggling a *high* row bit into a
*high* column bit (`CK_SWZ_R/W/L = 3,1,4`, i.e. `col ^= ((row>>3)&1)<<4`)
distinguishes warp groups, not the `16` rows inside an atom. With only 2 slots
(one toggle) against a `16`-row stride alias it can only partially decorrelate
the rows: it cuts the measured `LDSBankConflict` `52% → 25%` and lifts `MfmaUtil`,
but cannot go further.

**Rung E — element-granular, low-bit, all-slots → 0%.** Derive the parameters
from the geometry instead of guessing: granularity `L = log2(8) = 3`, use all
slots `W = log2(block_k / 8) = 3`, key on the low row bits `R = 0`:

```
col ^= (row & 7) << 3        # permute all 8 b128 slots by the low row bits
```

This spreads the low `3` row bits across all `8` `ds_read_b128` slots, which is
exactly the permutation that fully decorrelates the 16 aliasing M-rows. The
measured `LDSBankConflict` drops to **0.0%** and `MfmaUtil` rises to ~54%, still
bit-exact (verified race-free across
a K-sweep of 1…13 tiles, rectangular shapes, and large shapes). This is the
auto-derived default `lds_swizzle` form: with `CK_SWZ_*` unset the spec computes
`L/W/R` from the atom and `block_k` and produces exactly this; the env vars exist
only to force the coarse rung-D form for the A/B comparison.

### 5.4 Why not just pad the LDS row?

Widening the row stride (`lds_k_pad`) also breaks the alias — a padded row stride
of `(64+pad)` halves no longer divides the 64-bank-word conflict period. But it
spends LDS that the 1-WG/CU tile cannot give up
(the macro-tile already approaches the 160 KB/CU limit), and it measured
net-negative: the padding either pushes occupancy below 1 WG/CU or simply costs
more than it saves. The swizzle achieves the same de-aliasing for **zero** LDS, so
it wins. `lds_k_pad` remains in the spec as an opt-in knob for shapes/dtypes where
the swizzle path does not apply.

## 6. The L2 cache-coherency hint (square reuse)

The direct-to-LDS global loads carry a per-operand cache-coherency hint
(`dtl_cache_a`, `dtl_cache_b`): integer `0 = CACHE_ALL` (keep resident), integer
`2 = CACHE_STREAM` (one-shot, SLC set, "do not pollute L2"). The `TraitSpec`
defaults are `dtl_cache_a = 0` (A resident) and `dtl_cache_b = 2` (B streamed) —
the sensible default for the common case where B is a weight matrix read once.

But a **square** GEMM reuses the *whole* B across every M-tile of the grid:
streaming B and re-fetching it from HBM for each row-band of output is exactly
wrong. Setting `dtl_cache_b = CACHE_ALL` (`0`) keeps B L2-resident and turns those
re-fetches into L2 hits. This is rung F. The lever is workload-specific — it is a
win *because* the problem is square (full B reuse); on a tall-skinny or one-shot-B
problem the default stream hint would be correct.

## 7. Where the algorithm ends and the schedule begins

Everything above keeps the §1 accumulation fixed. The ladder is a sequence of
*streaming-schedule* changes:

| rung | lever | what it changes about the schedule |
|---|---|---|
| A | stock tile pipeline | global → VGPR → LDS, per-tile barrier |
| B | `direct_to_lds` | global → LDS in one instruction (0 `ds_write`) |
| C | `dtl_prefetch` | depth-2 ping-pong, **one** mainloop barrier (§4) |
| D | `lds_swizzle` (coarse, `CK_SWZ=3,1,4`) | high-bit XOR, conflicts 52% → 25% |
| E | `lds_swizzle` (auto, element-granular) | low-bit all-slot XOR, conflicts → 0% |
| F | `dtl_cache_b=CACHE_ALL` | keep the reused B L2-resident |

Two further additive `TraitSpec` knobs each add a small amount on top of rung F:
`chiplet_swizzle` (an XCD grid remap; measured best with `chiplet_chunk_size=32`,
versus the spec default of `64`) and `waves_per_eu` (an AMDGPU occupancy hint).
They are not ladder rungs and `scripts/ladder.py` does not enable them — to try
them, set the corresponding `TraitSpec` fields in the script's per-rung `dict`s.

## 8. Why the IR-lowered kernel stops short of hand-tuned assembly

rocBLAS for this shape is Tensile-generated, hand-scheduled GCN assembly with
register-double-buffered operands. The rocke kernel is authored in Python and
lowered through LLVM IR (`libamd_comgr`). Independently of rocBLAS this study
reaches its **exact geometry** (`DepthU = 64`, 256-wide macro-tile,
double-buffered — confirmed by mining the Tensile solution), drives bank conflicts
to **0%**, and emits the same direct-to-LDS + double-buffer skeleton. The residual
gap is **instruction scheduling**, and it traces to one structural limit:

- **direct-to-LDS** gives 0 `ds_write`, and the shipped kernel does double-buffer
  it (rung C's `dtl_prefetch` ping-pongs two LDS half-buffers, §4) — but that is an
  *LDS-buffer* depth-2 prefetch; the fused `buffer_load … lds` cannot also be
  *register*-staged, so operands never sit in VGPRs ahead of the MFMA the way
  rocBLAS stages them.
- **register-staged depth-2 prefetch** (the rocBLAS PGR2 lever) *requires* an
  explicit `ds_write` to LDS, and so cannot coexist with the 0-`ds_write`
  direct-to-LDS path. An offline experiment (not shipped in this example, and not
  built by `scripts/ladder.py`) implemented it numerically-exact but measured it
  *slower* — roughly half the direct-to-LDS path — because `comgr` does not place
  the `ds_write` into an MFMA issue slot, so its cost is fully exposed and competes
  with the operand `ds_read`s. (The README records the figure; it is an offline
  measurement, not reproducible from this example's code.)

rocBLAS gets **both** depth-2 prefetch *and* hidden `ds_write`s only because
hand-written assembly places each `ds_write` into a specific MFMA issue slot. That
instruction-granularity placement is the irreducible missing capability: a
DSL → IR → comgr pipeline emits IR and lets the backend schedule it; it cannot
place individual instructions at that granularity. The sanctioned IR-level
scheduling hint (`iglp_opt`) is *exactly neutral* on the fast direct-to-LDS path
(it has no `ds_write`s to interleave and is barrier-bound, not schedule-bound), so
even it cannot move the ceiling. The net is a Python-authored kernel at ~0.81× of
hand-tuned assembly on the same geometry at boost clock (the gap narrows when
sustained, since rocBLAS scales with clock better than this kernel's fixed
one-barrier-per-K-tile cost), with the remaining gap characterized down to that
single capability — *proven by building the alternative and measuring it lose*,
not assumed. See the README for the full ledger.
