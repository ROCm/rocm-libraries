<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Split-K auto-deduction for conv wgrad — gfx950 case study

How the split-K degree and reduction for implicit-GEMM conv wgrad/dgrad came to
be chosen by a fitted cost model (`rocke.helpers.split_k_planner`, resolved for
conv by `kernels.common.conv_split_k.resolve_conv_split_k`) instead of the CK
"fill the device" formula (`rocke.helpers.split_k.select_split_k_wgrad`).

Per [`platform/AGENTS.md`](../../../../../AGENTS.md) §Compliance this document
records **methodology, levers and relative ratios only**: every number below is
a ratio between configurations measured in the same process on the same
silicon; no absolute latency, throughput or bandwidth appears here, in the
model's constants (which are dimensionless, see [The model](#the-model)) or in
any commit message. Measured scope: gfx950, bf16 and fp16, `--direction wgrad`,
the cached AOT wgrad kernels (`--run-from-cache`), 99 shape/dtype cases from
the conv shape corpus (dense, grouped and depthwise), every power-of-two split
degree from 2 to 1024. Baseline: the CK formula as shipped before this change,
on the same binaries.

## Table of contents

- [What was investigated](#what-was-investigated)
- [The instrument](#the-instrument)
- [Finding 1: "fill the device" is the wrong question](#finding-1-fill-the-device-is-the-wrong-question)
- [Finding 2: latency plus a shared throughput term, not a max](#finding-2-latency-plus-a-shared-throughput-term-not-a-max)
- [Finding 3: a per-CTA cost decides deep splits of wide grids](#finding-3-a-per-cta-cost-decides-deep-splits-of-wide-grids)
- [Finding 4: the reduction must be chosen, not assumed](#finding-4-the-reduction-must-be-chosen-not-assumed)
- [Finding 5: a wave bound is as good as exact occupancy](#finding-5-a-wave-bound-is-as-good-as-exact-occupancy)
- [Finding 6: degrees off the power-of-two ladder](#finding-6-degrees-off-the-power-of-two-ladder)
- [The model](#the-model)
- [Rejected levers](#rejected-levers)
- [Correctness gate](#correctness-gate)
- [Results](#results)
- [Caveats](#caveats)
- [Replay](#replay)
- [Keep / revert / defer decisions](#keep--revert--defer-decisions)

## What was investigated

A wgrad GEMM has a filter-sized output (often a handful of tiles) and a
reduction of `N*Ho*Wo`, which can be millions long. Split-K cuts that reduction
into `s` slices, one CTA per (tile, slice), and folds the partials with packed
16-bit atomics into dW or with f32 atomics into a workspace that a Stage-2 pass
folds. The CK formula sizes `s` so the grid reaches two blocks per CU and
always prefers the packed atomic. The question was whether a model of *why*
split-K pays could choose better, for every shape, before any kernel is
compiled.

## The instrument

`library/benchmarks/common/split_k_cost_fit.py`:

- `sweep` runs `benchmark_implicit_gemm_conv.py --run-from-cache` per shape
  with `--split-k-degrees 2,4,...,1024` and `--verify`, timing every cached
  split-K binary at every degree (the degree is a kernarg, so one binary covers
  them all) next to the unsplit kernels.
- `fit` turns the logs into one degree curve per (shape, binary) and fits the
  model in two stages (degree, then reduction), cross-validated by shape.

Supplementary runs used the same harness: full unpruned curves for the depthwise
shapes, atomic-vs-workspace-vs-unsplit runs on the same main loop
(`--two-stage always --all-vector-sizes`), and re-timing of every degree the
sweep had pruned or early-stopped but a candidate rule would pick.

The scoring metric is **percent of best**: the best measured degree of a binary
over the time at the chosen degree (100% = the optimum). Choices are made as
production makes them -- over every feasible degree (within the K iterations,
the gridDim.z budget and the reduction's cap) -- and a held-out shape is always
scored by a model fitted without it.

## Finding 1: "fill the device" is the wrong question

At the best degree, the median kernel uses about three quarters of the resident
slots, and the best degree varies by orders of magnitude with the shape. A
fill rule recomputed with the *exact* occupancy of each binary was no better
than the CK formula. Small, padded tiles over long reductions keep gaining from
shorter slices well past "full"; big tiles over short reductions lose from any
split because the CUs are already busy and every slice adds reduction traffic.

## Finding 2: latency plus a shared throughput term, not a max

The first model charged each K iteration `max(latency, compute, bandwidth)`.
That saturates abruptly: once a CU holds a few workgroups the bandwidth branch
wins and time becomes constant in the degree. The measured curves do not
saturate abruptly -- more concurrency keeps hiding latency. Charging
`latency + max(compute, bandwidth)` (each K iteration waits for its loads, then
shares the CU's throughput) improved held-out percent-of-best on every shape
class, most on depthwise and on very long reductions.

## Finding 3: a per-CTA cost decides deep splits of wide grids

Merged-group depthwise kernels at high degrees launch tens of thousands of CTAs
of a few K iterations each, and get slower although the K loop keeps
shrinking. A fixed cost per CTA (dispatch, prologue, epilogue) captures that; a
per-round cost does not (fitted to effectively zero and removed: dropping it
changed none of the 95k decisions).

## Finding 4: the reduction must be chosen, not assumed

The benchmark used to time the packed-atomic and the two-stage kernels on
disjoint sets of shapes, so the choice between them was never measured. Timed
side by side (each with its required clear), the workspace path won about half
of the decisions, and when it won the margin was large: long reductions, where
the 16-bit atomic is capped for accuracy and fp16 packed atomics are much slower
than bf16 ones. The model charges each reduction its atomic traffic and its
setup (zeroing dW, or clearing and folding the workspace).

The packed 16-bit atomic is capped at 128: `--verify` fails broadly from 256
up, while the workspace path passes at every degree.

## Finding 5: a wave bound is as good as exact occupancy

The resident workgroups per CU are taken as `floor(32 / waves_per_block)`, the
hardware wave limit. Register-heavy kernels reach less, but they are the big
tiles that are throughput-bound anyway: held-out regret with the bound matched
the regret with each binary's exact occupancy. The bound needs no compiled
binary, so the builders and the benchmark resolve one deterministic answer.

## Finding 6: degrees off the power-of-two ladder

The degree is a kernarg, so nothing ties it to powers of two. The planner also
scores the divisors and multiples of the CU count, the smallest degree for each
K-iteration count per CTA (any larger degree with the same count only adds
CTAs) and the degrees that end a dispatch round. On 25 shapes (dense, grouped
and depthwise, bf16 and fp16), each competitive split-K binary was timed at
those degrees and the ladder around its pick, verified at every degree:

- The best measured degree beats the best power of two by 3.7% (geomean) and
  up to 24% per binary: there is something to gain.
- **Grouped launches must keep the group stride XCD-aligned.** With
  `z = group * split_k + slice` and workgroups dealt to the 8 XCDs by flat id,
  `tiles_per_group * split_k` off a multiple of 8 scatters the groups' shared
  NHWC lines over every die's L2: misaligned degrees ran 1.04-1.8x slower on
  grouped shapes and showed no effect on dense ones. Every power of two >= 8 is
  aligned, which is why the ladder never hit it. On a grouped launch the planner
  offers only aligned degrees off the ladder and keeps the ladder itself whole
  -- a small misaligned degree (2, 4) can still beat the nearest aligned one
  (the CK formula's grouped snap encodes the same rule).
- **Off the ladder the model resolves round filling more finely than its
  occupancy bound supports** (a 2.3% predicted gain for degree 3 over 2 on a
  wide dense grid measured 4% slower). A degree off the ladder is therefore
  chosen only when it beats the best ladder degree by `ladder_margin` (5%,
  from the fit tool's `margin` subcommand on the sweep above).

Against the previous planner (powers of two only), on 30 held-out shapes that
played no part in choosing the rules or the margin, each competitive binary
timed at both planners' degrees in one run:

| | previous planner | wider pool |
|---|---|---|
| shapes, best binary (geomean) | 1.000 | 0.989 (1.1% faster) |
| shapes more than 1% faster | -- | 6 of 30 (up to 19%) |
| shapes more than 1% slower | -- | 1 of 30 (5.2%, fp16 stride 2) |
| worst single binary | -- | 1.07x slower |

The gain is modest and concentrated in a few shapes; most shapes keep the
degree they had.

## The model

Costs are in units of `L`, the load latency of one K iteration, so every fitted
constant is a dimensionless ratio to it. For a GEMM `M x N x K` with `G`
independent groups, tile `tm x tn x tk` and `w` waves per block:

```
T = ceil(M/tm) * ceil(N/tn) * G      kt = ceil(K/tk)
C(s) = T*s      i(s) = ceil(kt/s)      P = num_CUs * floor(32/w)
F_it = 2*tm*tn*tk                     B_it = (min(tm,M) + min(tn,N)) * tk * bytes

tau(a)  = 1 + max(a*F_it/F, a*B_it/B)                 a = resident CTAs per CU
t(s,r)  = floor(C/P) * i * tau(P/num_CUs) + [C mod P > 0] * i * tau(ceil((C mod P)/num_CUs))
        + c*C + C*min(tm,M)*min(tn,N)*4/A_r + setup_r
setup_atomic    = dW_bytes / Fill
setup_workspace = (2 * R * G*M*N*4 + dW_bytes) / Fill        (R workspace copies)
choice  = argmin over s in {1} U {2, 4, ..., cap} and the buildable reductions r
cap     = min(1024, kt, 65535 // G_on_z, 128 for the 16-bit atomic)
```

`F_it` counts the whole tile (the MFMA computes padded lanes); `B_it` only the
real extent (out-of-bounds lanes load nothing), which is what makes a tiny
output in a large tile latency-bound. `F`, `B`, `c`, `A_r` and `Fill` are fitted
per arch; arches without a fitted model keep the CK formula.

## Rejected levers

- **Exact occupancy from the binary's register count:** no better than the wave
  bound on the power-of-two ladder (Finding 5), and it needs the binary, which
  dispatch does not have when it chooses. Retried with the wider pool, where
  the cliffs it places matter more (dispatch compiling the binary first, the
  constants refitted with it): held-out degree regret improved (geomean 1.036
  to 1.028), but on the 30 held-out shapes it bought about 1% more than the
  wave bound while 4 of 30 shapes instead of 1 got more than 1% slower and
  single binaries up to 1.34-1.48x slower. One more round is not a whole extra
  K loop for every kernel, so exact round boundaries make the model confidently
  wrong. Not shipped.
- **Per-address atomic contention term** (slices of one tile serialising on the
  same addresses): no held-out gain.
- **Bandwidth that degrades with the number of concurrent K streams:** a small
  held-out gain, but it did not fix the cases it targeted and pushed the other
  constants away from plausible values. Hardware counters on one such case
  showed DRAM traffic flat across degrees.
- **A fixed cost per extra launch** for the reduction passes: not identifiable
  from back-to-back launches (it fitted to the search floor); the setup cost is
  carried by the bytes those passes stream.

## Correctness gate

Every timed point in every run was preceded by `--verify` of that kernel at
that degree against the torch reference, at the benchmark's declared dtype
tolerance; failing points were never timed or ranked. The resolver itself is
covered by CPU tests (`library/tests/test_conv_split_k.py`,
`platform/tests/dispatch/dispatch_tests/gemm/test_split_k_planner.py`) and the
GPU wgrad correctness test, which builds the packed-atomic kernel at the
resolver's degree. The GPU dgrad correctness test does the same for a dgrad
shape the model splits about a hundred ways, and at the 16-bit atomic cap a
forced degree is held to.

## Results

Held-out (4-fold by shape), kernels within 25% of their shape's best, choices
over every feasible degree:

| | CK formula | cost model |
|---|---|---|
| mean percent of best | 89.5% | 96.8% |
| median | 94.2% | 100% |
| 10th percentile | 70.9% | 88.5% |
| choices within 3% of best | 43% | 72% |
| depthwise mean | 93.4% | 96.2% |
| fp16 mean | 87.6% | 97.7% |

Reduction and split-vs-unsplit decisions (same main loop, 17 shapes, held out
with the degree and setup constants both fitted without the held-out shapes;
the previous rule scored on the unsplit kernel where the CK degree is 1): mean
percent of best 95.6% (previous rule) vs 97.5%; within 3% of best 67% vs 70%.
The bootstrap intervals overlap, so on this set the reduction choice is no
worse than the previous rule rather than clearly better; the gain is in the
degree.

End to end, the kernel the dispatcher selects, timed at the previous rule's
choice and at the new one on 72 corpus shapes (median of two guarded passes):
geometric-mean speedup 1.11x overall and 1.16x on the 51 shapes whose choice
changed (dense 1.21x, grouped 1.32x, depthwise neutral). 25 of those 51 are
faster by more than 2%, 13 are within 2%, 13 are slower by more than 2% (the
slowest at 0.79x).

## Caveats

- The oracle is the best *measured* degree on the power-of-two grid up to 1024;
  some long reductions still improve at 1024, so the cap is the limit of
  validation, not of benefit.
- 13 dispatch shapes are slower than before, mostly by a few percent; on the
  ones inspected the optimum is a sharp point the smooth model misses (flat
  DRAM traffic across degrees on the profiled one).
- The fit covers 16-bit operands on gfx950 only; fp32 reuses the constants.
  Every other arch, gfx942 included, keeps the CK formula: gfx950 constants
  scaled by hardware ratios would be unvalidated, and gfx942 ships in more
  than one CU/XCD geometry.
- The ladder margin was measured on a 25-shape subset, not the full corpus.
- Run-to-run noise of the end-to-end measurement: median 0.4%, 90th percentile
  1.9% between two full passes.

## Replay

From `library/`:

```bash
export PYTHONPATH=.:../platform/python
python benchmarks/common/benchmark_implicit_gemm_conv.py --compile-all \
    --arch gfx950 --direction wgrad --cache-dir kernel_cache
python -m benchmarks.common.split_k_cost_fit sweep \
    --shapes wgrad_shapes.txt --cache kernel_cache --out sweep_logs
python -m benchmarks.common.split_k_cost_fit fit --logs sweep_logs --arch gfx950 --cv
```

`fit` prints the held-out comparison against the CK formula and the fitted
`SplitKCostModel(...)` literal for `_COST_MODELS`. For the ladder margin, sweep
a shape subset off the ladder too and score it:

```bash
python -m benchmarks.common.split_k_cost_fit sweep --shapes subset.txt \
    --cache kernel_cache --out offladder_logs --degrees 2-160,192,256,384,512,768,1024
python -m benchmarks.common.split_k_cost_fit margin --logs offladder_logs
```

## Keep / revert / defer decisions

- **Keep:** the cost model for gfx950 wgrad (degree and reduction; the
  reduction choice needs more side-by-side shapes to show a clear gain), one resolver
  for the builders and the benchmark, the 16-bit atomic cap, the benchmark
  timing the resolver's degree and both reductions.
- **Keep, unchanged:** the CK formula on arches without a fitted model.
- **Defer:** a gfx942 model, fitted on gfx942 measurements across the device geometries it ships in; a dgrad sweep before dgrad dispatch uses the
  planner (it keeps its pinned degree until then); modelling the sharp optima behind the remaining
  regressions; stream-K as a third reduction choice.
