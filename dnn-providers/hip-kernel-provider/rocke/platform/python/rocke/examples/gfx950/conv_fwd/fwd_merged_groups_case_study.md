<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Forward depthwise convolution: merged groups — gfx950 case study

This document records the methodology, levers and **relative** measured results
for adding merged-group support to the forward implicit-GEMM convolution path on
`gfx950`. Measured scope: 35 forward **depthwise** (`C/groups == 1`) convolution
configurations in **NHWC**, at bf16 and fp16, swept over the kernel's own config
space; the comparison arms are rocke's own unmerged and merged forward
implicit-GEMM kernels, with MIOpen 3.5.1 (ROCm 7.2.0) as the external reference
point. Per [`platform/AGENTS.md`](../../../../../AGENTS.md) §Compliance, only
ratios appear here — no latency, throughput, achieved-FLOP or bandwidth figure
is recorded in this file or in any commit message. Absolute values live in the
approved access-controlled record.

## Table of contents

- [What was investigated](#what-was-investigated)
- [The arrangement](#the-arrangement)
- [Finding 1: the win is on the A load, not the C store](#finding-1-the-win-is-on-the-a-load-not-the-c-store)
- [Finding 2: the best degree is shape-dependent, and a flat cap is harmful](#finding-2-the-best-degree-is-shape-dependent-and-a-flat-cap-is-harmful)
- [Finding 3: the diagonal belongs on the B load](#finding-3-the-diagonal-belongs-on-the-b-load)
- [Finding 4: the sweep can be pruned without losing the optimum](#finding-4-the-sweep-can-be-pruned-without-losing-the-optimum)
- [Correctness gate](#correctness-gate)
- [Results](#results)
- [Caveats](#caveats)
- [Replay](#replay)
- [Keep / revert / defer decisions](#keep--revert--defer-decisions)

## What was investigated

Depthwise convolution is the worst case for the forward implicit-GEMM path. In
NHWC the memory order is `N, H, W, G, C`; with `C/groups == 1` there is nothing
contiguous under the GEMM-K axis, and the GEMM-N extent (`kpg`) is a single
element. Input loads and output stores both degenerate to one element per
instruction, and every MFMA wastes all but one of its N lanes.

CK Tile addresses this with `NumGroupsToMerge`: fold `Gm` consecutive
convolution groups into one GEMM tile so the group index becomes the
fastest-varying factor of a GEMM dimension, restoring `Gm`-wide vector access at
the cost of `Gm`x redundant multiply-accumulates that land in otherwise idle
MFMA lanes. rocke already had this for backward-weight
([`conv_implicit_gemm_wgrad.py`](../../../../../../library/kernels/common/conv_implicit_gemm_wgrad.py));
the question was what the right arrangement is for forward, and whether the
redundant work is genuinely free.

The public model architectures that contributed depthwise shapes to the measured
set are ConvNeXt / ConvNeXt-V2, EfficientNetV2, MobileNetV4, RepLKNet, SLaK,
UniRepLKNet and YOLO (v11/v12). Shapes span `G` from 3 to 1536, filters from
3x3 to 31x31, batch 1 to 128, and both unit and strided cases.

## The arrangement

Forward's stride-1 axis (`c`) already sits *inside* the reduction axis, which is
the opposite of wgrad. So rather than copying the wgrad/CK "`Gm` on M and N"
mapping, forward puts `Gm` on **GemmN and GemmK**:

| | unmerged | merged |
| --- | --- | --- |
| `M` | `N*Ho*Wo` | `N*Ho*Wo` — unchanged; the group does **not** go on M |
| `N_gemm` | `kpg` (= 1) | `Gm` — the GEMM-N index *is* the merged group `g_n` |
| `K_gemm` | `Y*X*cpg` (= `Y*X`) | `Y*X*Gm`, decoding to `(y, x, g_k)` with `g_k` innermost |

Everything load-side binds to `spec.merged_problem`
(`dc_replace(problem, groups=groups // group_merge)`), which is the identity at
`group_merge == 1`. That is what keeps the default path byte-identical.

Total MFMA issue is **unchanged** against `Gm == 1`: `grid.z` shrinks by exactly
the factor `K_gemm` grows by. The redundant FLOPs are absorbed by the previously
idle N lanes, so they cost no additional MFMA issue slots.

## Finding 1: the win is on the A load, not the C store

The intuitive reading — that merging widens the *output* store — is wrong about
where the leverage is, and it matters because it leads to the wrong gate.

Vector widths are derived in `ImplicitGemmConvSpec.default_vector_sizes(C, K)`,
which the builder calls with `p_load.cpg, p_load.kpg`, **not** the true `C`/`K`.
On a depthwise problem `cpg == kpg == 1`, so unmerged this reads `(1, 1)` and
merged it reads `(Gm, Gm)`. Counting instructions in the emitted IR for one
shape:

| | A (input) loads | C stores | B (weight) loads |
| --- | --- | --- | --- |
| `Gm = 1` | 33 scalar `i16` | 17 scalar `i16` | 17 scalar `i16` |
| `Gm = 8` | 3x `v4i32` (`dwordx4`) | 3x `v4i32` | 17 scalar `i16` (unchanged) |

Both A and C widen, but A is the one on the inner loop and the one whose
latency was exposed; the store is amortised across the whole K loop. The
practical consequence is that the A-load vector **saturates at 8** (the widest
`_vec` degree), so any benefit at `Gm = 16` or `32` cannot be coming from
vectorisation. It comes from MFMA N-lane utilisation plus the `grid.z` shrink
cutting redundant re-reads of A. That distinction is what Finding 2 rests on.

## Finding 2: the best degree is shape-dependent, and a flat cap is harmful

All 31 distinct shapes (35 configurations) were swept over the config space at
every admissible degree. Normalising each row to its own `Gm = 1` best, so the
numbers are self-relative. `s` is the stride; `—` means the degree was not
admissible for that shape.

Large-batch shapes, where `M = N*Ho*Wo` is many tiles wide:

| shape | s | dtype | `Gm=2` | `Gm=4` | `Gm=8` | `Gm=16` | `Gm=32` | `Gm=64` | best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `N64 H14 W14 C512 K512 Y7 X7 G512` | 1 | bf16 | 1.87x | 3.31x | 5.36x | **6.66x** | 6.25x | 5.10x | `Gm=16` |
| `N64 H56 W56 C128 K128 Y7 X7 G128` | 1 | bf16 | 1.78x | 3.10x | 5.78x | 7.54x | **8.67x** | 6.19x | `Gm=32` |
| `N128 H14 W14 C512 K512 Y7 X7 G512` | 1 | bf16 | 1.91x | 3.42x | 6.08x | **7.81x** | 6.42x | 5.14x | `Gm=16` |
| `N128 H56 W56 C128 K128 Y7 X7 G128` | 1 | bf16 | 1.95x | 3.36x | 5.74x | 7.75x | **8.70x** | 6.96x | `Gm=32` |
| `N64 H12 W12 C1536 K1536 Y3 X3 G1536` | 1 | fp16 | 2.08x | 3.61x | 5.80x | **8.70x** | 8.55x | 7.04x | `Gm=16` |
| `N64 H24 W24 C960 K960 Y3 X3 G960` | 1 | fp16 | 1.94x | 3.83x | 6.66x | 10.90x | **10.93x** | 8.17x | `Gm=32` |
| `N64 H48 W48 C256 K256 Y3 X3 G256` | 2 | fp16 | 2.24x | 4.03x | 5.68x | 8.78x | **9.80x** | 8.38x | `Gm=32` |
| `N64 H24 W24 C960 K960 Y3 X3 G960` | 2 | fp16 | 1.92x | 3.61x | 5.18x | 6.97x | **8.26x** | 6.24x | `Gm=32` |
| `N128 H12 W12 C1536 K1536 Y3 X3 G1536` | 1 | fp16 | 2.02x | 3.59x | 6.61x | 9.44x | **9.47x** | 8.26x | `Gm=32` |
| `N128 H24 W24 C960 K960 Y3 X3 G960` | 1 | fp16 | 1.71x | 3.20x | 5.48x | 9.91x | **10.23x** | 8.48x | `Gm=32` |
| `N128 H48 W48 C256 K256 Y3 X3 G256` | 2 | fp16 | 1.84x | 3.45x | 6.51x | 9.79x | 13.35x | **13.75x** | `Gm=64` |
| `N42 H120 W160 C192 K192 Y3 X3 G192` | 2 | bf16 | 2.01x | 3.92x | 7.09x | 12.35x | 21.73x | **25.03x** | `Gm=64` |
| `N42 H60 W80 C256 K256 Y3 X3 G256` | 1 | bf16 | 1.61x | 3.66x | 5.57x | 11.33x | **16.20x** | 13.40x | `Gm=32` |
| `N42 H60 W80 C256 K256 Y3 X3 G256` | 2 | bf16 | 1.94x | 4.00x | 7.12x | 12.94x | 19.28x | **22.57x** | `Gm=64` |
| `N42 H30 W40 C512 K512 Y3 X3 G512` | 1 | bf16 | 1.91x | 3.99x | 6.44x | 10.82x | **13.94x** | 10.78x | `Gm=32` |
| `N42 H480 W640 C3 K3 Y11 X11 G3` | 1 | bf16 | — | — | — | — | — | — | `Gm=1` |

`N = 1` shapes, where `M` is a handful of tiles or fewer:

| shape | s | dtype | `Gm=2` | `Gm=4` | `Gm=8` | `Gm=16` | `Gm=32` | `Gm=64` | best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `N1 H56 W56 C144 K144 Y3 X3 G144` | 1 | bf16 | 1.10x | 1.13x | **1.14x** | 1.14x | — | — | `Gm=8` |
| `N1 H56 W56 C144 K144 Y5 X5 G144` | 2 | bf16 | 1.03x | **1.06x** | 1.05x | 1.02x | — | — | `Gm=4` |
| `N1 H24 W24 C256 K256 Y3 X3 G256` | 1 | bf16 | 1.00x | 1.02x | **1.02x** | 1.01x | 1.00x | 0.96x | `Gm=8` |
| `N1 H56 W56 C192 K192 Y7 X7 G192` | 1 | bf16 | 1.65x | 2.64x | **3.92x** | 3.72x | 2.40x | 1.16x | `Gm=8` |
| `N1 H14 W14 C768 K768 Y7 X7 G768` | 1 | bf16 | 1.35x | 1.39x | **1.39x** | 1.36x | 0.92x | 0.37x | `Gm=8` |
| `N1 H7 W7 C1536 K1536 Y7 X7 G1536` | 1 | bf16 | 1.02x | **1.04x** | 1.03x | 1.00x | 0.67x | 0.29x | `Gm=4` |
| `N1 H14 W14 C768 K768 Y7 X7 G768` | 1 | fp16 | 1.30x | **1.35x** | 1.35x | 1.30x | 0.92x | 0.37x | `Gm=4` |
| `N1 H28 W28 C256 K256 Y31 X31 G256` | 1 | bf16 | 1.58x | **2.36x** | 2.15x | 1.29x | 0.61x | 0.25x | `Gm=4` |
| `N1 H14 W14 C512 K512 Y31 X31 G512` | 1 | bf16 | 1.50x | **1.63x** | 1.04x | 0.49x | 0.23x | 0.09x | `Gm=4` |
| `N1 H14 W14 C512 K512 Y51 X5 G512` | 1 | bf16 | 1.48x | **1.51x** | 1.19x | 0.62x | 0.32x | 0.12x | `Gm=4` |
| `N1 H14 W14 C512 K512 Y5 X51 G512` | 1 | bf16 | 1.38x | **1.53x** | 1.15x | 0.57x | 0.30x | 0.11x | `Gm=4` |
| `N1 H28 W28 C384 K384 Y13 X13 G384` | 1 | bf16 | 1.66x | 2.75x | **3.03x** | 2.33x | 1.20x | 0.53x | `Gm=8` |
| `N1 H80 W80 C256 K256 Y3 X3 G256` | 1 | bf16 | 1.72x | 2.61x | 2.81x | **2.87x** | 2.85x | 2.74x | `Gm=16` |
| `N1 H14 W14 C1024 K1024 Y7 X7 G1024` | 1 | bf16 | 1.44x | 1.64x | **1.65x** | 1.59x | 1.11x | 0.45x | `Gm=8` |
| `N1 H96 W96 C192 K192 Y7 X7 G192` | 1 | bf16 | 1.77x | 3.20x | 5.25x | **6.75x** | 5.76x | 3.15x | `Gm=16` |
| `N1 H56 W56 C192 K192 Y7 X7 G192` | 1 | fp16 | 1.65x | 2.67x | **4.01x** | 3.70x | 2.45x | 1.25x | `Gm=8` |
| `N1 H56 W56 C192 K192 Y31 X31 G192` | 1 | bf16 | 1.70x | 2.82x | **4.13x** | 3.64x | 1.95x | 0.89x | `Gm=8` |
| `N1 H96 W96 C144 K144 Y3 X3 G144` | 1 | bf16 | 1.63x | 2.16x | **2.23x** | 2.22x | — | — | `Gm=8` |
| `N1 H40 W40 C256 K256 Y7 X7 G256` | 1 | bf16 | 1.59x | 2.63x | **3.12x** | 2.80x | 1.89x | 0.89x | `Gm=8` |

Winning-degree histogram across the 35 configurations:
`Gm=1`: 1, `Gm=4`: 7, `Gm=8`: 10, `Gm=16`: 5, `Gm=32`: 9, `Gm=64`: 3.

Two things fall out, and the second is the load-bearing one.

First, where `M` is large, merging is a large win at these measured
configurations — up to 25.0x over the best unmerged configuration of the same
shape. The winning tile family was consistently `256x32x64 / w2x2 / a16x16x32`.

Second, **the optimum is shape-dependent and a flat cap actively regresses some
shapes.** No single degree is even close to right everywhere: five different
merge degrees win on some shape, and every one of them loses on another. On
several `N=1` rows, `Gm = 32` is materially *slower* than
not merging at all and `Gm = 64` is catastrophic — as low as 0.09x. The
mechanism follows directly from Finding 1: merging divides
`grid.z` by `Gm`, and when `M` is only a tile or two wide, `grid.z` *is* the
parallelism. Past the point where the part runs out of blocks, the `grid.z`
shrink stops buying re-read savings and starts starving the machine.

`_pick_group_merge` in
[`grouped_convolution.py`](../../../../../../library/dispatch/grouped_convolution.py)
therefore applies three independently binding caps rather than a constant:

- **`tile_n`** — `Gm` must fit the GEMM-N tile, or the block-N offset can exceed
  `Gm`.
- **Degree ceiling** (`_FWD_MERGE_MAX = 32`) — 64 remains reachable from the
  sweep harness, but dispatch will not select it. See below for why.
- **Occupancy** — `ceil(M / tile_m) * groups / _FWD_MERGE_MIN_CTAS`, targeting
  two CTAs per CU on a 256-CU `gfx950` part. This is the cap that stops the
  function from being a constant, and it is the one that saves the `N=1`
  rows. The CU count is hardcoded deliberately: a device query inside dispatch
  would make selection non-deterministic and unusable for cross-compile.

Divisibility (`groups % Gm == 0`) is a hard admissibility condition, not a cap.
Over the 35 measured configurations, 34 admit `Gm >= 16` and 31 admit up to 64;
the single exclusion is `G = 3`, for which no power-of-two degree divides.

### Why the ceiling is 32 and not 64

Three of the 35 configurations do peak at `Gm = 64` — all large-batch, all
stride-2. Raising the ceiling to capture them was measured rather than argued:
the shipped `_pick_group_merge` was run against every configuration's measured
optimum at both ceilings, scoring each pick by the fraction of that shape's own
achievable gain it realises.

| ceiling | picks the measured optimum | geomean fraction of achievable gain | worst case |
| --- | --- | --- | --- |
| `32` | 19 / 35 | 0.970 | 0.82 |
| `64` | 13 / 35 | 0.907 | 0.66 |

Raising the ceiling makes the policy **worse on both axes**. The three shapes it
rescues gain 3–17% each; the six it breaks lose 23–34% each, because at the
shipped 64x64 dispatch tile the occupancy cap does not bind early enough to
protect shapes whose true optimum is 16 or 32. The ceiling is therefore a
deliberate trade, not an observation that nothing prefers 64: dispatch has no
cheap request-side signal that separates the three winners from the six losers,
so it declines the bet. A shape that genuinely wants 64 can still be reached
through the sweep harness.

## Finding 3: the diagonal belongs on the B load

Correctness requires `g_k == g_n`: the accumulator lane for merged group `g_n`
must only receive contributions from that same group's weights. There are two
places to enforce it, and the choice determines whether the store stays dense.

CK Tile masks late and consequently ships a narrow `VectorSizeC`. In this
arrangement the mask goes on the **B (weight) load** instead:

```
g_n  = block_n_off_v + row
g_k  = kg & (Gm - 1)
yx   = kg >> log2(Gm)
off, valid = B_desc.offset(k_out=gm_group*Gm + g_n, k_gemm=yx)
return off, land(valid, cmp_eq(g_k, g_n))
```

Off-diagonal B elements get `valid = False`, which the sync loader turns into
the out-of-bounds sentinel and the buffer resource turns into a hardware zero —
no memory traffic, no scratch buffer. Because the zeros enter through B, **every
output lane is a real output at a consecutive stride-1 position**, so the C
store needs no mask and no gather and stays dense. The price is `vector_size_b`
forced to 1 and a B tile that is `1/Gm` dense; for depthwise, weights are
negligible against activations, which is why this trade is the right one here
and would not be on a dense convolution.

Two emitter details that are easy to get wrong:

- The whole merged `b_descriptor` wrapper is built **inside** `if
  spec.group_merge > 1:`. `IRBuilder` performs no constant folding and no CSE,
  so an unconditionally materialised `const_i32` renumbers every downstream SSA
  value and silently breaks byte-identity even though the kernel is unchanged.
  For the same reason the `gm_group*Gm` term is elided when it is provably zero.
- The validator's effective store-vector computation and the cshuffle epilogue's
  `max_store_vec` both read `spec.merged_problem`. They are deliberately
  identical expressions: wgrad's history records that two copies of one gate is
  exactly how a dispatcher came to hand the builder specs the builder rejected.

## Finding 4: the sweep can be pruned without losing the optimum

Adding merged combinations grew the forward depthwise sweep from 8,880
combinations to 34,530, which is what users feel as "merged group convolutions
take a long time to compile". The cost is sweep *volume*, not per-kernel cost: a
merged kernel is cheaper through comgr than an unmerged one at the same tile,
because the diagonal collapses the B tile. Nothing is to be gained by making
individual kernels cheaper; the question is how many to build.

This section is qualitative by design. The measured evidence behind every claim
in it lives in the approved access-controlled record, not here.

Two observations make most of that volume redundant.

**The degree axis is already answered.** `_pick_group_merge` (Finding 2) exists
precisely to deduce the right `Gm` from shape geometry, and it recovers most of
each shape's measured optimum on the depthwise corpus. Sweeping all six degrees
to rediscover a number the dispatcher will compute analytically anyway is the
single largest redundancy. But pruning to exactly the picked degree would make
the sweep unable to contradict the policy it is supposed to validate — so the
sweep keeps a **±1 window** over the *admissible* degrees, bracketing the pick
rather than assuming it. `--group-merge-window RADIUS` controls the radius.

The window must be over admissible degrees, not over `_FWD_MERGE_DEGREES` slots.
`groups=72` caps at 8 and admits only `(8, 4, 2)`; a raw-tuple window around 8
would name 32 and 16, which the sweep then drops, collapsing the bracket to one
entry — the opposite of what the window is for.

**The unmerged leg is a control, not a search.** It is swept to establish the
baseline each merged configuration is measured against, and a baseline does not
need every tile. `--unmerged-frac FRAC` samples it; the default keeps 10%.

Both flags restore today's exhaustive behaviour exactly — `--group-merge-window
-1 --unmerged-frac 1.0` reproduces the 34,530-combination pool with all six
degrees present.

### Validating the window out of sample

Scoring the window on the corpus it was tuned against would be circular, so the
window was scored on **12 synthesised held-out depthwise shapes** that appear
nowhere in the corpus, chosen to populate each of the three regimes the policy
can land in — ceiling-bound (`_FWD_MERGE_MAX`), divisor-bound (group counts with
poor 2-adic valuation), and occupancy-bound (small `M`). Each was swept across
the **full** degree axis and the **full** unmerged leg, and both the full and the
pruned answer were then computed offline from those same measured rows, so no
kernel is timed twice and the comparison isolates the pruning.

The metric is `pruned / full`: how much slower is the kernel the pruned sweep
ships. The obvious alternative — realised gain fraction,
`(pruned - base) / (full - base)` — is not sufficient on its own, and the gap is
not academic. On a shape whose exhaustive winner is the *unmerged* leg,
`full == base`, so realised gain is 1.0 by construction no matter how badly the
unmerged sampling hurt: the metric is blind to the only loss that shape can
suffer. Exactly one of the 12 held-out shapes is that shape, which is why both
are computed and `pruned / full` is the one that decides.

Scored across the held-out shapes and 64 sampling seeds each, at the shipped
settings (radius 1, 10% unmerged), the conclusions are:

- The pruned sweep's winner is **indistinguishable from the exhaustive sweep's**
  on these shapes — the shortfall is far inside the tolerance that would make it
  worth sweeping the extra degrees.
- The ±1 bracket is **load-bearing, not decoration**: at radius 0 the worst
  held-out shape degrades enough to be visible to a user. Trusting the policy's
  pick outright is the one version of this change that would not be safe.
- Radius 2 is indistinguishable from radius 1, so the extra degrees buy nothing.
- Sampling the unmerged leg is nearly free, which is what "it is a control, not a
  search" predicts.

### Cost

Compared back to back on one device at `--sample 1.0`, the pruned defaults cut
the forward depthwise sweep from 34,530 combinations to 14,208, and cut sweep
wall time by substantially more than that cardinality ratio alone — the
combinations the pruning keeps skew merged, and merged kernels are the cheap
ones.

The pruned pool is a strict subset of the exhaustive pool — verified directly,
10,464 compiled configurations of 25,376, every one present in both arms.

### Why that A/B does not measure performance

The back-to-back comparison above is a **compile-time** measurement only; its two
`Best:` lines are not comparable and no perf claim rests on them. Across the
10,464 configurations common to both arms, the same kernel timed in two different
runs varies enough that a meaningful share of identical configurations disagree
between arms. These depthwise shapes are short-running, and at the default timed
iteration count the max-order-statistic over tens of thousands of candidates is
dominated by that noise. Reading a winner out of it would be reading noise as
signal — which is why the question above is answered offline from one set of
measured rows rather than from two runs.

### Keeping the copy honest

The benchmark deliberately does **not** import the dispatcher — the sweep is the
ground truth the policy is tuned against, so importing the policy into the sweep
would make the measurement depend on the thing being measured. The cost is a
hand-copied `_fwd_merge_pick`, and the mirror is what makes that copy safe:
`fwd_group_merge_for_geometry` was extracted from `_pick_group_merge` as a pure
refactor (no emission change), and
[`library/tests/test_fwd_merge_window.py`](../../../../../../library/tests/test_fwd_merge_window.py)
asserts the two agree across 10,725 `(groups, M, tile_m, tile_n)` points plus the
three shared constants. A change to `_FWD_MERGE_MAX` / `_FWD_MERGE_MIN_CTAS` /
`_FWD_MERGE_DEGREES` that is not mirrored fails there, rather than silently
producing a sweep that brackets the wrong degree.

## Correctness gate

Every measured arm in the results below passed an on-silicon numeric check in
the same run that produced its timing — the sweep was executed with `--verify`,
which compares each timed configuration's output against an independent torch
reference on device before the timing loop is accepted, and records the verdict
in the `passed` column of the CSV. The report filters on that column, so an arm
that did not verify contributes no ratio.

Declared gate: input dtypes bf16 and fp16 as listed per configuration; metric is
peak-normalised relative error, `max|out - ref| / max(max|ref|, 1)`, against a
torch convolution reference computed once per shape in fp32; bound `5e-2`, keyed
to the **compute** dtype rather than the storage dtype. That bound is loose, and
the metric's global-max denominator can mask a large relative error on a
small-magnitude element — it is a real gate, not a tight one, and it is the same
gate applied identically to both arms.

Separately, and not a substitute for the above:

- Byte-identity is GREEN for all four convolution families at both LLVM flavors.
  Merged groups is implemented in **both** engines, and the forward parity
  emitters build five merged/depthwise configs (indices 17–21), so the gate
  byte-compares merged emission directly rather than only proving the default
  path is undisturbed. No existing config's output changed, so no golden
  re-bless was required; the five new configs register as new-and-unblessed,
  which the golden check reports as informational.
- `TestConvFwdGroupMergeNumerics` (on-GPU, 27 subtests) covers merged numerics
  across degrees, including an `_assert_case_ran` guard — wgrad's merged cases
  silently skipped their entire `group_merge > 1` axis until an equivalent guard
  existed, and a sweep whose subtests all skip still reports `passed`.
- `test_conv_fwd_group_merge_gate.py` (10 tests) asserts the predicate and
  `validate()` agree, that each degree gets a distinct kernel name, and that the
  scalar case emits **zero** vector buffer loads while `gm8` emits some.
- `test_grouped_conv_fwd_merge_dispatch.py` (10 tests, 26 subtests) asserts the
  host grid equals the kernel's own grid under merge. That is the highest
  severity failure mode on this path: a grid taken off the *true* problem
  launches `Gm`x redundant CTAs in z while covering 1 of `Gm` columns in x — it
  does not crash, it just produces a plausible-looking wrong answer in most
  channels.

## Results

All 35 forward depthwise configurations, NHWC, `gfx950`. **Before** is the best
`group_merge == 1` configuration and **after** is the best configuration at any
degree, both taken from the same sweep process under the same sampling seed, so
the ratio is a like-for-like comparison of rocke against rocke. `Gm` is the
degree the winning configuration used. `vs MIOpen` is rocke against MIOpen 3.5.1
on the same shape; values below 1.00x mean MIOpen is ahead. Every arm below
passed the `--verify` gate described above.

Large-batch configurations:

| shape | s | dtype | `Gm` | merged gain | vs MIOpen before | after |
| --- | --- | --- | --- | --- | --- | --- |
| `N64 H14 W14 C512 K512 Y7 X7 G512` | 1 | bf16 | 16 | 6.66x | 0.45x | 2.98x |
| `N64 H56 W56 C128 K128 Y7 X7 G128` | 1 | bf16 | 32 | 8.67x | 0.41x | 3.56x |
| `N128 H14 W14 C512 K512 Y7 X7 G512` | 1 | bf16 | 16 | 7.81x | 0.47x | 3.66x |
| `N128 H56 W56 C128 K128 Y7 X7 G128` | 1 | bf16 | 32 | 8.70x | 0.37x | 3.24x |
| `N64 H12 W12 C1536 K1536 Y3 X3 G1536` | 1 | fp16 | 16 | 8.70x | 0.90x | 7.85x |
| `N64 H24 W24 C960 K960 Y3 X3 G960` | 1 | fp16 | 32 | 10.93x | 1.04x | 11.36x |
| `N64 H48 W48 C256 K256 Y3 X3 G256` | 2 | fp16 | 32 | 9.80x | 1.33x | 13.07x |
| `N64 H24 W24 C960 K960 Y3 X3 G960` | 2 | fp16 | 32 | 8.26x | 0.94x | 7.73x |
| `N128 H12 W12 C1536 K1536 Y3 X3 G1536` | 1 | fp16 | 32 | 9.47x | 0.98x | 9.27x |
| `N128 H24 W24 C960 K960 Y3 X3 G960` | 1 | fp16 | 32 | 10.23x | 0.91x | 9.28x |
| `N128 H48 W48 C256 K256 Y3 X3 G256` | 2 | fp16 | 64 | 13.75x | 0.96x | 13.22x |
| `N42 H120 W160 C192 K192 Y3 X3 G192` | 2 | bf16 | 64 | 25.03x | 0.19x | 4.74x |
| `N42 H60 W80 C256 K256 Y3 X3 G256` | 1 | bf16 | 32 | 16.20x | 0.34x | 5.46x |
| `N42 H60 W80 C256 K256 Y3 X3 G256` | 2 | bf16 | 64 | 22.57x | 0.18x | 4.04x |
| `N42 H30 W40 C512 K512 Y3 X3 G512` | 1 | bf16 | 32 | 13.94x | 0.41x | 5.65x |
| `N42 H480 W640 C3 K3 Y11 X11 G3` | 1 | bf16 | 1 | 1.00x | 0.94x | 0.94x |

`N = 1` configurations:

| shape | s | dtype | `Gm` | merged gain | vs MIOpen before | after |
| --- | --- | --- | --- | --- | --- | --- |
| `N1 H56 W56 C144 K144 Y3 X3 G144` | 1 | bf16 | 8 | 1.14x | 1.07x | 1.23x |
| `N1 H56 W56 C144 K144 Y5 X5 G144` | 2 | bf16 | 4 | 1.06x | 0.64x | 0.67x |
| `N1 H24 W24 C256 K256 Y3 X3 G256` | 1 | bf16 | 8 | 1.02x | 0.71x | 0.73x |
| `N1 H56 W56 C192 K192 Y7 X7 G192` | 1 | bf16 | 8 | 3.92x | 0.46x | 1.80x |
| `N1 H14 W14 C768 K768 Y7 X7 G768` | 1 | bf16 | 8 | 1.39x | 0.71x | 0.99x |
| `N1 H7 W7 C1536 K1536 Y7 X7 G1536` | 1 | bf16 | 4 | 1.04x | 0.80x | 0.83x |
| `N1 H14 W14 C768 K768 Y7 X7 G768` | 1 | fp16 | 4 | 1.35x | 0.70x | 0.94x |
| `N1 H28 W28 C256 K256 Y31 X31 G256` | 1 | bf16 | 4 | 2.36x | 0.54x | 1.29x |
| `N1 H14 W14 C512 K512 Y31 X31 G512` | 1 | bf16 | 4 | 1.63x | 0.78x | 1.28x |
| `N1 H14 W14 C512 K512 Y51 X5 G512` | 1 | bf16 | 4 | 1.51x | 0.77x | 1.17x |
| `N1 H14 W14 C512 K512 Y5 X51 G512` | 1 | bf16 | 4 | 1.53x | 0.89x | 1.35x |
| `N1 H28 W28 C384 K384 Y13 X13 G384` | 1 | bf16 | 8 | 3.03x | 0.41x | 1.23x |
| `N1 H80 W80 C256 K256 Y3 X3 G256` | 1 | bf16 | 16 | 2.87x | 0.95x | 2.73x |
| `N1 H14 W14 C1024 K1024 Y7 X7 G1024` | 1 | bf16 | 8 | 1.65x | 0.68x | 1.11x |
| `N1 H96 W96 C192 K192 Y7 X7 G192` | 1 | bf16 | 16 | 6.75x | 0.40x | 2.67x |
| `N1 H56 W56 C192 K192 Y7 X7 G192` | 1 | fp16 | 8 | 4.01x | 0.46x | 1.85x |
| `N1 H56 W56 C192 K192 Y31 X31 G192` | 1 | bf16 | 8 | 4.13x | 0.42x | 1.74x |
| `N1 H96 W96 C144 K144 Y3 X3 G144` | 1 | bf16 | 8 | 2.23x | 1.15x | 2.55x |
| `N1 H40 W40 C256 K256 Y7 X7 G256` | 1 | bf16 | 8 | 3.12x | 0.50x | 1.55x |

Geometric means, and the count of configurations where rocke is at or ahead of
MIOpen:

| set | n | merged gain | vs MIOpen before | after | ahead before | ahead after |
| --- | --- | --- | --- | --- | --- | --- |
| large-batch | 16 | 9.59x | 0.58x | 5.51x | 2 / 16 | 15 / 16 |
| `N = 1` | 19 | 2.06x | 0.65x | 1.34x | 2 / 19 | 14 / 19 |
| **all** | **35** | **4.16x** | **0.62x** | **2.56x** | **4 / 35** | **29 / 35** |

The split is exactly what the mechanism predicts and is not an artifact of which
models the shapes came from. Merging converts idle MFMA N lanes into useful work
and shrinks `grid.z`; where `M` is wide enough that `grid.z` was not the scarce
resource, that is close to free and the gain is large. Where `M` is a handful of
tiles, the `grid.z` it spends is the parallelism the shape had, and the gain
compresses toward 1.00x — the occupancy cap's job is to keep it from going
below.

One configuration is unchanged at 1.00x: `G = 3` admits no power-of-two merge
degree, so `_pick_group_merge` returns 1 and the shape takes the unmerged path
untouched. That is the designed fallback, not a failure to improve.

## Caveats

- **The measurement vehicle is the sweep, not dispatch.** The before/after
  ratios are best-config against best-config over the same config space in the
  same process. Library dispatch ships one fixed tile
  (`64x64x64 / w2x2 / a32x32x16`) for every forward grouped shape, while the
  depthwise optimum measured here is `256x32x64 / w2x2 / a16x16x32`. The merged
  candidate inherits that fixed tile, so the dispatch path will realise less
  than the sweep shows. Widening dispatch's tile selection is pre-existing work,
  out of scope for this change, and the single largest remaining lever.
- **This compares rocke against rocke.** The MIOpen column is a reference point
  for whether the remaining gap is closed, not a like-for-like comparison: it is
  a different implementation with its own tuning database, and it selects its
  own solver per shape.
- Merged implicit-GEMM is faster than unmerged implicit-GEMM at these measured
  configurations, but on the
  shapes where a direct-convolution kernel is already the better rocke choice,
  implicit-GEMM remains behind it. The end-to-end effect is therefore
  concentrated on the configurations that implicit-GEMM actually serves.
- Results are from a `--sample 0.10` sweep with a fixed seed, not an exhaustive
  one. A sampled sweep can miss a shape's true optimum in either arm; the
  sampling is identical for both arms, so the ratio is the defensible quantity
  and the per-arm bests are not.
- **Pruning the sweep by the dispatch policy is mildly circular**, and the ±1
  window mitigates that without removing it. A degree the policy is wrong about
  by more than one admissible step is outside the bracket and the pruned sweep
  cannot report it. The held-out validation in Finding 4 is what bounds the
  residual risk — it was designed for this specific question and found no loss at
  radius 1 — but it bounds it on 12 shapes, not on all shapes. `--group-merge-window -1`
  exists so that any shape suspected of being that case can be re-swept exhaustively.
- Finding 4's compile-time A/B is a **wall-time** measurement. Its per-arm `Best:`
  configurations are not comparable: run-to-run timing noise on these
  short-running kernels moves a meaningful share of identical configurations.
  The conclusion there comes from scoring both policies offline against one set
  of measured rows, never from comparing two runs.
- `async_dma`, pointwise (`Y == X == 1`), and `wave_size != 64` are explicitly
  **not** supported under merge and are rejected by the gate. Pointwise is the
  notable one — it is where merging should be most attractive, and it is
  deferred only because the flat pointwise fast path builds its `valid`
  predicate from scratch and has no slot for the diagonal.

## Replay

Provenance of the measured run: rocke at `develop` commit `8f77b587ff4` plus the
merged-groups change this document describes; ROCm 7.2.0, MIOpen 3.5.1; a single
`gfx950` device; swept 2026-09.

From the rocke root, with both packages importable:

```bash
export PYTHONPATH=$(pwd)/platform/python:$(pwd)/library
```

Sweep one shape across every admissible degree, verifying each timed
configuration. Merged combinations are generated alongside unmerged ones, so a
single run produces both arms; group the CSV by the `group_merge` column:

```bash
python3 library/benchmarks/common/benchmark_implicit_gemm_conv.py \
    --direction fwd --arch gfx950 \
    --miopen-cmd "MIOpenDriver convbfp16 -n 128 -c 512 -H 14 -W 14 -k 512 \
        -y 7 -x 7 -p 3 -q 3 -u 1 -v 1 -l 1 -j 1 -g 512 -F 1 -t 1 \
        -in_layout NHWC -out_layout NHWC -fil_layout NHWC" \
    --jobs 0 --sample 0.10 --seed 0 --top 5 --warmup 3 --iters 10 \
    --verify --csv sweep.csv --csv-top 999999
```

`--verify` gates **every** timed configuration, not just the first; the
`--help` string is misleading on this point, the call site in the per-config
loop is authoritative.

Since Finding 4 the command above sweeps the **pruned** space. To reproduce the
pre-pruning pool — 34,530 combinations, all six degrees, full unmerged leg — pin
both axes explicitly:

```bash
    --group-merge-window -1 --unmerged-frac 1.0
```

Pin them for any measurement whose purpose is to *evaluate* the degree policy;
leaving them at their defaults would score the policy against itself. Everything
in Findings 1–3 predates the flags and corresponds to `-1 / 1.0`.

The mirror test is CPU-only and needs no device:

```bash
python3 -m pytest library/tests/test_fwd_merge_window.py -q
```

Correctness and gate:

```bash
python3 -m pytest library/tests/test_conv_fwd_correctness.py -q -rs \
    -k GroupMerge
python3 -m pytest library/tests/test_conv_fwd_group_merge_gate.py \
    library/tests/dispatch/test_grouped_conv_fwd_merge_dispatch.py -q
python3 tools/run_checks.py --op conv
```

Read the `-rs` skip list. `_assert_case_ran` makes an all-skipped merged sweep
a hard failure, but only for those cases.

Byte-identity, both flavors — must be GREEN. `conv_implicit_gemm` reports 22
configs; 17–21 are the merged/depthwise ones:

```bash
cd platform && export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python3 tools/check_byte_identity.py --only conv_implicit_gemm
ROCKE_LLVM_FLAVOR=llvm22 python3 tools/check_byte_identity.py
```

## Keep / revert / defer decisions

| Change | Decision | Basis |
| --- | --- | --- |
| `Gm` on GemmN and GemmK (rather than wgrad/CK's M and N) | **Keep** | Leaves `M` untouched, keeps the C store dense and unmasked, and needs no transposed A window |
| Diagonal mask on the B load | **Keep** | Off-diagonal elements become hardware zeros with no memory traffic; this is what lets the store stay wide |
| `vector_size_b = 1` under merge | **Keep** | The B tile is `1/Gm` dense by construction; for depthwise, weights are negligible against activations |
| Shape-dependent `_pick_group_merge` with an occupancy cap | **Keep** | The winning degree spreads across five values over 35 shapes, and every flat choice regresses some shape below unmerged; the cap is the reason the function is not a constant |
| `_FWD_MERGE_MAX = 32` ceiling in dispatch | **Keep** | Scored against every shape's measured optimum, a ceiling of 64 realises less of the achievable gain and has a worse floor. Three shapes do prefer 64; dispatch cannot identify them cheaply, so 64 stays reachable only from the sweep harness |
| `group_merge` default of 1 | **Keep** | Additive by construction: parity config 17 is the unmerged depthwise control and byte-compares identical, so the knob is provably inert where it is not asked for |
| Merge + `async_dma` | **Defer** | `AsyncTileLoader`'s predicate is chunk-granular and cannot express a per-element diagonal |
| Merge + pointwise (`Y == X == 1`) | **Defer** | The flat fast path builds `valid` from scratch with no slot for the diagonal. Highest-value follow-up: this is where merging should pay most |
| Merge + WMMA / `wave_size != 64` | **Defer** | Separate fragment mapping; gated off rather than guessed |
| Widening dispatch's forward tile selection | **Defer** | Pre-existing and independent of merging, but it is what stands between the dispatch path and the sweep result |
| `--group-merge-window` default of 1 | **Keep** | Substantially cheaper sweeps for an out-of-sample shortfall small enough not to matter. Radius 0 is visibly worse on one held-out shape; radius 2 is indistinguishable from radius 1 |
| `--unmerged-frac` default of 0.10 | **Keep** | The unmerged leg is the baseline control, not a search; sampling it is nearly free out of sample |
| Both pruning axes flag-gated rather than hardcoded | **Keep** | A sweep pruned by the dispatch policy cannot be used to evaluate that policy. `-1 / 1.0` restores the exhaustive pool exactly, and is the required setting for any future degree-policy work |
