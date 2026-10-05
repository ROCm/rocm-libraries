<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Grouped dgrad on the direct-MFMA pipeline — gfx950 dispatch case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
This records the mechanism, the levers swept, the keep/revert decisions, the
correctness bugs the work exposed, and the replay path.

Toolchain: every measurement and byte-identity run behind this study used the
llvm20 backend (`ROCKE_LLVM_FLAVOR` unset). The production llvm22 backend was
not available on the machine used, so neither timings nor the llvm22 golden
flavor were checked here; re-run the replay and the llvm22 gate below on a
machine that has it.

## What changed

Before this change the grouped-convolution dispatcher had exactly one dgrad
candidate on gfx950: the 64x64x64 implicit-GEMM (`implicit_gemm_conv_dgrad`).
The direct-MFMA dgrad pipeline in `conv_direct_grouped.py` (weight pre-pass +
the streaming fprop kernel run on dY and the flipped, channel-swapped weights)
was reachable only from the benchmark sweep.

Now:

* `kernels/common/conv_direct_grouped.py` owns the pipeline description:
  `plan_direct_mfma_dgrad(problem, fprop_spec)` returns every stage (role, spec,
  grid, block, buffer bindings) and every buffer size, and
  `direct_mfma_dgrad_stage_kernel(stage, arch)` builds a stage. Harnesses no
  longer re-derive grids.
* `dispatch/grouped_convolution.py` registers `direct_mfma_conv_dgrad`
  (priority 5, ahead of the igemm candidate at 10) for the measured win region
  (originally a filter-size cap, a problem-size gate and a narrow-reduction
  rule, below; since refitted, see the integration section),
  with a table-driven knob policy (`GFX950_DIRECT_DGRAD_RULES` for
  `block_groups`, `_direct_dgrad_block_h` and `_direct_dgrad_block_q` for the
  spatial tiling). The selected `ConvGroupedDirectDgradSpec` exposes
  `launch_plan(req)`; the candidate's `grid`/`block` are the main kernel's.
  The igemm candidate stays registered and still serves everything else.
* `benchmarks/common/benchmark_direct_conv.py` sweeps `block_h = 0` and the
  single-wave `fold_k32` combo, labels every distinguishing knob (including
  `block_groups`), builds the pipeline from the plan, and verifies with dX and
  scratch poisoned.
* The weight transpose pre-pass is indexed flat over `W_T` (one lane per
  element, 256-lane blocks); it used to launch a 64-lane block per
  `(group, r, s, c)` with one lane per `k`, so with `kpg = 4` sixty of every
  sixty-four lanes idled and the launch count scaled with `cpg * KH * KW`.

## Integration with the single-kernel forms (current state)

The sections after this one record how the candidate was built and reviewed
while every direct dgrad still paid a weight pre-pass. Two later pieces of
work changed both sides of the comparison, so the candidate was re-wired and
its eligibility refitted:

* the fused weight transform (`dgrad_fused_weights`, see
  `dgrad_fused_weights_case_study.md`) lets the generic main kernel read the
  original `W` with flipped, channel-swapped addressing in its prologue, and the
  batched 4x4x4 kernel (`dgrad_4c_bf16_case_study.md`) covers
  `cpg == kpg == 4` in fp16 and bf16 with the same fused prologue;
* the stride-1 implicit-GEMM dgrad gained a constant-folded sub-GEMM record and
  a tap-outer K loop (`stride1_igemm_dgrad_case_study.md`), so the igemm
  fallback that every decline lands on got faster on grouped shapes too.

What the candidate selects now (`_select_direct_dgrad_spec`):

| Shape class | Main kernel | Launches |
| --- | --- | --- |
| `cpg == kpg == 4`, 1x1/3x3, `groups % 16 == 0`, 4c grid above its floor | `DirectConv4cSpec`, default `block_q`/`block_groups`, fused weights staged through LDS (`ds_read_b64_tr_b16`) | one |
| other admitted shapes whose preloaded weight fragments fit the register budget | generic `DirectConvSpec`, the knob table below, fused weights (LDS-staged where the slice fits), `waves_per_eu = 4` for a small preload footprint | one |
| admitted shapes past that budget (large filters with 32-wide channel groups on both sides) | the pre-pass pipeline described below | two |

The 4c row has a grid floor because one wave is one workgroup and the kernel
does not tile H: with one or two images its grid is a few dozen workgroups,
and the generic kernel (whose H tiling fills the device) wins there.

The `waves_per_eu = 4` hint for small preload footprints and the kernel-level helper
(`direct_dgrad_spec_for_problem`) were both compared against "A's knob table
plus fused weights": the hint was kept (neutral on aggregate, it removes the
serialised next-row loads where it fires), the hook's own knob choice was not
(it lost on aggregate to the table, mostly on `block_h`/`block_q`).

### Refitting the eligibility policy

The measurement-round gates below were fitted against the baseline igemm with the
pre-pass cost in every direct pipeline. Re-measured with the fused kernels
against the faster igemm, they both forgo many wins (small problems, few
groups, short images, 1x1 narrow reductions all win once the pre-pass is gone)
and admit some losses (large filters on the pre-pass fallback, tiny images
with wide channel groups, which no earlier cohort covered).

The policy was refitted from scratch rather than patched:

1. A random draw of shapes over the structural region (channel pairs from
   {4, 8, 12, 16, 24, 32}, 1x1..7x7, H and W from 3 to 64, 2..64 groups, batch
   sized for a wide range of igemm workgroup counts, fp16/bf16) was measured
   in one session: the igemm candidate against the direct candidate forced
   past every policy decline, each verified first.
2. Per-class decision trees on that data (fused vs pre-pass form, tiny widths,
   1x1 wide reductions, small problems) suggested the split features; each
   tree was rewritten as one or two readable conditions and scored on the
   whole training draw by the geometric mean of the chosen kernel's time
   against an oracle that always picks the faster one, plus the list of
   admitted losses and forgone wins.
3. A disjoint hold-out draw, measured the same way and not used for fitting,
   checked the final rules: the chosen kernel stays close to the oracle and
   no admitted shape loses by a large margin. The remaining admitted losses
   are small, scattered and not separable by the features available at
   dispatch time.

The resulting terms (`_direct_dgrad_policy_errors`, every reason prefixed with
`_DIRECT_DGRAD_POLICY_PREFIX`):

* pre-pass form: with more than 20 output channels per group only full
  16-wide K atoms (`kpg % 16 == 0`), filled column strips and a size floor;
  otherwise a size floor (and half-filled strips for 16+ channel groups whose
  `kpg` is not a multiple of 16); then the cost model of the next section;
* tiny images (`W <= 3`, `H*W <= 16`, and `W <= 5` with wide output
  channels): the 16-wide output strips stay mostly empty;
* wide output with a narrow reduction needs filled column strips;
* 1x1 with a wide reduction is igemm's best case except below 32 channels on
  strips wider than 6 columns above a size floor;
* small problems with a narrow reduction or four channels need a per-class
  size floor.

The four shape classes the last measurement round flagged (tiny non-narrow images,
`cpg = 32` / `kpg = 24` at few groups, 7x7 `kpg = 4` on `W = 24`, 13-row
`cpg = 24` / `kpg = 8`) are covered by the refit: the first three route to
igemm through the tiny-image and pre-pass terms, and the fourth now runs the
fused kernel, which no longer carries the pre-pass that made it lose.
`test_policy_declines_where_igemm_measured_faster` and
`test_policy_admits_where_direct_measured_faster` pin representative shapes
of every term.

### The pre-pass pipeline: a faster transpose and a cost model

A final review found the pre-pass pipeline admitted where it lost to the
unmodified tree: many groups (a few hundred), `cpg` 24..32 with `kpg` 16 or
32, 5x5/7x7 filters on 10x10..12x12 images, one to four images. The kernel
split showed the weight transpose alone costing about as much as the whole
igemm kernel there. These shapes take the pre-pass form because their fused
weight fragments exceed the main kernel's register budget (the preload needs
well over the 128-VGPR budget), so the fused single-kernel form cannot serve
them; the cause was attacked on both sides of the comparison.

**The transpose kernel.** It mapped one lane to one `W_T` element, so its
stores were coalesced but every lane read a 2-byte element with a
`KH * KW * cpg` stride: far from the bandwidth bound on large weight tensors.
`build_direct_transpose_weights_dgrad` now runs one workgroup per (group,
filter row, chunk of source `k` rows). The workgroup stages its slice -- `k`
rows of `KW * cpg` contiguous elements -- into LDS with 8- or 16-byte loads,
then writes the destination runs of `KW * kpg` contiguous elements with 8- or
16-byte stores whose lanes gather their `k` run from LDS. A chunk is the
largest `k` range that fits the LDS budget and still leaves one workgroup per
CU, else the smallest.

Two details decide whether it reaches the bandwidth bound:

* **LDS banks.** The write-back lanes of a wave step through `k` blocks of
  `vout` rows; for common channel counts (`cpg` 32 at 5x5/7x7) that block
  stride is a multiple of the bank count, so most of a wave gathered from a
  few banks. `_transpose_block_pad` shifts each block by the smallest whole
  number of staging vectors (alignment kept) that spreads the first wave over
  the banks. With it, the transpose runs at the speed of a plain vector copy
  of the same bytes, both alone and inside the pipeline (where the main
  kernel between calls leaves the caches colder, for the copy as much as for
  the transpose) -- so the pre-pass cannot get cheaper short of removing it.
* **Small tensors.** Below `_TRANSPOSE_STAGED_MIN_ELEMENTS` weight elements
  the copy is latency-bound and the staged kernel's extra LDS round trip and
  barrier made the pipeline slower than the flat kernel on small problems;
  those keep the flat kernel. So do slices too large to stage.

Checked bit-exact against numpy, in the form the size rule picks and with
staging forced, on a set covering both vector widths, tail passes, `k`
chunking, the block shift and the flat forms
(`test_weight_transpose_is_bit_exact`).

**The decline: a cost model, not another threshold.** Even at the copy
bound, the pipeline loses wherever its main kernel's advantage over igemm is
smaller than the transpose it pays on every call -- and a fresh random sweep
of pre-pass-form problems that pass every other gate showed a second loss
class the earlier refits had not sampled: the main kernel itself losing to
igemm, mostly with `cpg` not a multiple of 16 (the 16-wide output tiles run
partly empty), a partial second K atom (`kpg` 20..28) or poorly filled
column strips. Simple work-ratio rules (a minimum `N * H * W`, a bound on
weight elements per activation element, with or without those classes) were
tried first on the same data: they either kept losses below the unmodified
tree or gave up most of the pipeline's wins, because the transpose's share
and the main kernel's efficiency vary independently. So the decline is one
predicted-cost comparison (`_direct_dgrad_prepass_cost_ratio`), unitless --
every cost is relative to igemm's fixed per-call cost:

* transpose: linear in the weight elements `K * Y * X * cpg`;
* main kernel: a fixed cost plus GFLOPs scaled by its waste terms --
  `block_q`-wide column-strip padding over `W`, 16-wide output-tile padding
  over `cpg`, K-atom padding of `kpg` (32-deep atoms under `fold_k32`), a
  partial second K atom -- and by the problem size `N * H * W`;
* igemm: its fixed cost plus GFLOPs scaled by its 64-wide N-tile padding over
  `cpg`.

The term set is the smallest one found by backward elimination from a larger
model (halo rows, filter size, group count, channel counts, H tiling, igemm
M-tile padding): dropping those terms kept the same safety on the hold-out
draw at nearly the same distance to the oracle. Elimination also dropped the
K-atom padding term; a second fresh draw, measured on the production
dispatch, then found one admitted loss of exactly that kind (a narrow
reduction, `kpg` 8 under `cpg` 20, on a short, narrow image), and the term
was put back -- it separates that problem without moving the rest. Each stage was
least-squares fitted (log-linear in the waste terms) on a same-session random
sweep of pre-pass-form problems over the structural region together with the
review neighbourhood (`G` 64..300, `cpg` 24..32, `kpg` 16/32, 3x3..7x7,
H, W 8..16, N 1..8, fp16/bf16). The pipeline is kept while its predicted cost
stays under a fixed fraction of igemm's (`_DIRECT_DGRAD_PREPASS_MAX_COST_RATIO`):
the largest fraction that kept every fitted problem at or above the
unmodified tree, less one step for the main-kernel model's error band. A
disjoint hold-out draw of both sweeps, and a third fresh draw measured on the
production dispatch, confirmed it: no admitted problem below the unmodified
tree. The review's 10x10 problems run igemm; with the faster transpose the
same family on larger images (16x16 with several images) and the
large-image, few-group and narrow-channel pre-pass wins stay on the
pipeline.

The cost is the margin: some problems where the pipeline would have been
faster than igemm are declined to igemm (still faster than the unmodified
tree), because the model cannot tell them apart from the losses it must
avoid. Pinned by `test_prepass_decline_names_the_cost_model`, the
`prepass_unamortized_*` / `prepass_main_loses_*` rows of
`test_policy_declines_where_igemm_measured_faster` and the `prepass_*` rows of
`test_policy_admits_where_direct_measured_faster`.

## Mechanism: why the igemm candidate is the wrong family here

The igemm dgrad tiles the per-group GEMM `M = N*H*W`, `N = cpg`,
`K = Y*X*kpg` with a 64-wide N tile. Grouped shapes with small channel groups
fill only `cpg / 64` of that tile, the K loop is a handful of 64-deep steps,
and each loop body issues a few MFMAs against hundreds of VALU instructions of
im2col address arithmetic (static ISA counts of the dispatched kernels show a
single-digit MFMA count against several hundred VALU in the body). The grid is
`(M tiles x 1, groups)`, so the device runs a very large number of tiny,
address-bound workgroups.

The direct pipeline streams dY rows through a double-buffered LDS row ring
(each dY row is loaded once per workgroup when `block_h = 0`), keeps a ring of
`KH` accumulator slots per output row, and drives 16x16x16 (or 16x16x32 under
`fold_k32`) MFMAs straight from that ring. The weight flip and channel swap
cost one small pre-pass kernel instead of per-element address math in the hot
loop. None of the dispatched main kernels spill; register use is modest on the
3x3 shapes and grows with the filter (7x7 + `fold_k32` is the heaviest and is
why large filters get H tiles).

## Levers swept (Step 0) and decisions

The sweep ran every legal combination of `block_q {16, 32}`,
`block_groups {1, 2, 4, 8, 16}` (dividing `groups`), `block_h {0, 8, 16}` and
the wave combos `(waves_k, runtime_k_loop, fold_k32)` over the three target
shapes and a cohort of grouped stride-1 shapes: group widths 4/8/16/24/32/64,
batch 1..128, spatial 7..112, 3x3/5x5/7x7, and `cpg != kpg`. Every point was
verified before it was timed (conv rule, NaN-poisoned dX and scratch). Timing is
`rocprofv3 --kernel-trace`: the sum of the median duration of each pipeline
kernel, so host launch overhead does not enter the ranking. The pipeline's
weight pre-pass is included.

| Lever | Finding | Decision |
| --- | --- | --- |
| `block_h = 0` (no H tiling) | Best whenever the untiled grid already fills the device: each dY row is loaded once instead of once per tile plus a `KH-1` halo | **Keep**: chosen when `ceil(Wo/16) * groups * N` reaches the wave target |
| `block_h = 8` | Wins on mid-size grids, and is forced above 64 rows to bound the Python-unrolled row loop | **Keep**: next tile tried when the untiled grid is short of the target |
| `block_h = 4` | Not in the first sweep. Best on small grids (small batch, wide channel groups) and on every 5x5/7x7 shape: twice the H tiles of `block_h = 8`, and the serial row chain per workgroup halves | **Keep**: fallback when 8-row tiles still miss the target; always for 5x5/7x7 |
| `block_h = 16` | Occasionally best on mid-size grids; not consistently | Revert (not selected) |
| wave target for H tiling | Grid-searched over the second cohort; tall images (H > 16) take the largest tile that reaches it | **Keep** for tall images |
| short-image tile cost model (H <= 16) | Third measurement round: a single wave target tiled 9- and 10-row images into 8+1 or 4+4+1 rows on low-group, large-batch grids that already filled the device, and lost to igemm. A cost model (serial row chain incl. halo times rounds of waves over a `block_groups`-scaled slot budget) picks the sweep-best tile on the low-group cohort and is neutral on the second cohort | **Keep**: replaces the short-image wave target |
| `block_groups` | The strongest per-shape knob. Best keeps `block_groups * max(cpg, kpg)` near a 32..64-channel slice. The first table keyed it on `kpg` alone, which gave `cpg = 32, kpg = 4..8` four groups per workgroup and lost | **Keep**: table on `max(cpg, kpg)`: 1 at >= 32, 2 at 16..31, 4 below |
| `block_groups` halved for 5x5/7x7 | First cohort: no aggregate gain. Second cohort (with `block_h = 4` and the wider channel axis): consistent gain; the `KH`-deep accumulator ring makes each wave heavy, so fewer groups per workgroup keep the grid large | **Keep** |
| `block_q = 32` | First cohort: a loss on aggregate. Second cohort: a clear win with H tiling once the wider channel count is >= 16 (or >= 8 with 5x5/7x7): the per-strip halo (`KW - 1` columns) and per-strip weight traffic halve. Loses on narrow channels (cpg <= 8, 3x3), on untiled grids, and when halving the strip count drops the grid below a wave floor (e.g. `groups = 3`) | **Keep** under those conditions (`_direct_dgrad_block_q`) |
| `fold_k32` (16x16x32) | Wins whenever `kpg % 32 == 0`, with a single wave (`waves_k = 1`); the old sweep never tried that pairing | **Keep**: on iff `kpg % 32 == 0` |
| `runtime_k_loop` | Same or slightly better main-kernel time on 7x7, plus a reorganize pre-pass that eats the gain | Revert (never selected) |
| `waves_k > 1` | Only competitive on the 7x7 shape, and below the tiled single-wave point there | Revert (never selected) |
| Group width 64 | igemm wins (the per-wave M-tile count and accumulator ring outgrow the direct kernel) | **Excluded**: eligibility stops at `cpg, kpg <= 32` |
| Filters above 7x7 | Correct, but the main kernel spills registers (accumulator ring of `KH` slots) and is at best level with igemm | **Excluded**: eligibility stops at 7x7, the measured region. 1x1 (`pad = 0`) is inside the region and was checked correct and faster in the adversarial rounds |
| Problem-size gate | See below | **Keep**: `ceil(N*H*W/64) * groups >= 768` |
| Narrow-reduction rule | See below | **Keep** |
| Transpose pre-pass: flat lane mapping | Every lane works; the launch count follows the element count | **Keep** |
| Transpose pre-pass: 4x4 (c, k) register tile per lane, 8-byte loads and stores | Within run-to-run noise of the flat scalar mapping; the pre-pass cost on these sizes is launch and latency, not bytes | Revert (simpler kernel kept) |

The next five sections are the history of the pre-pass-era gates, kept for
the reasoning only: the integration refit replaced those gates, their named
constants and the tests that pinned them, so none of the rules below exists
in the tree as written. The mechanisms they describe (pre-pass fixed cost,
under-filled K atoms and output strips, halo share on short images) are what
the refit's terms encode.

### The problem-size gate (found by adversarial measurement)

The first version admitted every in-region shape regardless of size. A
small-batch probe set (late-stage ResNeXt layers at batch 1..8, 7x7 and
14x14 images) all lost to igemm. Per-kernel traces showed why: the direct main
kernel alone was level with or faster than igemm, but the weight transpose is a
separate launch with a roughly fixed cost, and on these problems igemm runs one
partial wave of workgroups at its launch floor, so there is nothing for the
main kernel to win back.

The work measure is the igemm candidate's own grid at stride 1,
`ceil(N*H*W / 64) * groups` (one N tile per group since `cpg <= 64`). A
dX-element count is not enough: for the same dX size, smaller channel groups
give more igemm tiles, each wasting more of its 64-wide N tile, which is
exactly where direct wins. A small-shape cohort (batch 1..16, 7x7/14x14/28x28,
groups 2..32, group widths 4..32, 3x3/5x5/7x7, `cpg != kpg`), each shape timed
as forced-igemm against the forced direct policy pick in the same process, put
every shape below about two igemm workgroups per CU on the igemm side and every
shape at or above about three per CU on the direct side, independent of group
width and filter size. A boundary cohort between the two placed the crossover;
the threshold sits at its upper edge, which forgoes a few marginal direct wins
just below it rather than admit marginal losses.

The gate is a performance decision only: the knob policy is still tested for
numerical correctness on small adversarial shapes below the gate
(`test_policy_knobs_below_the_size_gate`). Removing the pre-pass launch (fusing
the weight transform into the main kernel) would remove the reason for the gate
and should re-measure it.

### The narrow-reduction class (found in the second adversarial measurement round)

A second adversarial measurement round measured regressions above the size gate, all with
`cpg > kpg`: `cpg = 32` with `kpg = 4` or `8` at 3x3, and 5x5/7x7 at
`cpg = 32`, up to large batch. The per-kernel split showed the main kernel
alone slower than igemm, so neither the gate nor the pre-pass explained it.

Mechanism. The transposed fprop reduces over `kpg`. With `kpg = 4` or `8` the
16-wide MFMA K atom is a quarter or half full, while each wave still carries
`ceil(cpg / 16) = 2` output M-tiles times a `KH`-slot accumulator ring.
igemm's N width is `cpg`, so `cpg = 32` is its best case. On top of that, the
first knob table keyed `block_groups` on `kpg` alone, so these shapes ran four
groups per workgroup (a 128-channel output slice, a quarter of the grid) with
8-row H tiles and 16-wide strips; the ISA of that pick for the 7x7 case has
twice the VGPR + AGPR footprint of the new pick and one wave per SIMD.

What was done, one lever at a time, each verified before it was timed:

1. Transpose pre-pass made flat (above). Removes most of the pre-pass cost on
   these shapes; does not change the main kernel.
2. A second cohort (one hundred shapes: the first cohort, the target shapes,
   an adversarial probe set, and a grid over `cpg/kpg` in
   {32/4, 32/8, 32/16, 16/4, 16/8, 8/4}, 3x3/5x5/7x7 at `cpg = 32`, batch 4..64,
   spatial 7..56 including an odd 33x19), each timed as igemm plus the full
   direct sweep with `block_h = 4` added, in one locked trace per shape.
3. Knob policy re-fit on that cohort (rows above): `block_groups` keyed on
   `max(cpg, kpg)` and halved for 5x5/7x7, `block_h = 4`, conditional
   `block_q = 32`. With it most of the previously losing probe shapes win
   (including the 3x3 `cpg32/kpg8` and the 7x7 `cpg32/kpg8` cases); the ISA of
   the new pick shows half the register footprint per wave and two to four
   times the workgroups.
4. What still loses with the best direct config in the sweep becomes the
   narrow-reduction rule (`cpg > kpg`, `kpg <= 8`): decline 5x5/7x7 below
   a minimum igemm workgroup count, and decline
   widths that leave the 16-wide column strips less than three-quarters full
   (W = 19: two strips for 19 columns). Single-strip images (W <= 16) are not
   affected by the strip rule; they were measured winning. Shapes on the
   decline side that were marginal wins (a few small 5x5/7x7 ones) are
   forgone rather than risk the marginal losses beside them.

### Low groups, short images, large batch (found in the third adversarial measurement round)

A third adversarial measurement round measured regressions on shapes the second cohort barely
covered: few groups (2..8), short images (7..16 rows) and a large batch. Two
causes, separated by the per-kernel split and a direct sweep per shape:

1. **Spatial tiling of short images.** The short-image wave target cut 9- and
   10-row images into 8-row or 4-row tiles (8+1, 4+4+1): a near-empty last
   tile and a `KH - 1` halo re-loaded per tile, on grids whose untiled form
   already filled the device. `block_h = 0` was the sweep-best direct config
   there and beats igemm. Fix: for `H <= 16`, `_direct_dgrad_block_h` picks
   the whole image, 8-row or 4-row tiles by the smallest
   `(tile rows + KH - 1) * max(slots, waves)`, with `slots` one wave per SIMD
   for the 32-channel rule row, scaled by the rule's `block_groups` (less work
   per wave, more resident waves). Tall images keep the wave-target rule.
2. **Narrow reduction on small images.** With `cpg > kpg`, `kpg <= 8` and an
   image that is one mostly empty 16-wide strip (`W <= 9`), both the MFMA K
   atom and its M width are mostly empty; the main kernel alone is at best
   level with igemm, so the weight pre-pass (whose duration is bimodal on a
   shared GPU) decides. A first version admitted the 7x7 case at 32 groups,
   where some batch sizes won; a batch sweep at 32 groups then showed wins
   and losses alternating with batch size, so the class is declined as a
   whole. Fix: the narrow rule also declines a single strip below
   a minimum fill.

Data: a third cohort on that axis (groups 2/4/8, images 5..16 rows plus a
30x30 one, batch 14..600, `cpg/kpg` in {32/4, 32/8, 32/16, 32/32, 16/4, 16/8,
16/16, 16/32, 8/32, 24/4}, 3x3 plus some 5x5/7x7, bf16 and fp16, the
adversarial probe shapes included), each timed as igemm plus the direct sweep in one
locked trace. The model and the new decline were fitted on it, then checked
on a fresh hold-out set from the same region (including shapes on both sides
of the new decline boundary) that was not used for fitting, and on the second
cohort, where the pick changes only on a few `G = 32`, 14-row shapes, at
parity or better.

Two further declines were measured and not taken. Declining few-group
narrow shapes whose direct grid has few wave columns (an alternative
rule considered) also declines shapes of the same class that win clearly (wider
images, `cpg = 16`, 24 or 28). Declining the `cpg = 32`, few-group, 4-row-tile
part of the narrow corner removes a handful of losses that are within a few
percent of parity, but forgoes as many wins of a similar or larger size on
neighbouring shapes; the available features do not separate the two. Those
near-parity shapes stay on the direct candidate; a narrow-K main kernel or a
fused weight transform is the way to move them.

Hold-out shapes (not in the fitting cohort) and the adversarial probe shapes were
then timed same-session against the unmodified checkout. Raw numbers and the
per-shape table live outside the tree.

Measurement note: within one process, the same pipeline config repeated a
few calls apart showed bimodal pre-pass durations and a several-percent
main-kernel spread (clock state; the GPU is shared). Single sweep points
choose between neighbouring configs well, but every keep/revert at the margin
was re-measured as interleaved repetitions of baseline and candidate.

Across the second cohort the policy lands on, or close to, each shape's
sweep-best direct config, and beats the igemm pick on every shape it admits.

### The narrow class next to the size gate (found in the fourth adversarial measurement round)

A fourth adversarial measurement round hunted the narrow-reduction class (`cpg > kpg`, `kpg <= 8`)
right at the size gate with few groups, which earlier cohorts had barely
sampled, and found reproducible losses: `cpg = 32`, `kpg = 8`, 3x3, 16x16,
`G = 2` and `G = 6`, a single full strip, at the gate. The per-kernel split
showed the main kernel alone faster than igemm and the separate transpose
pre-pass turning that into a loss: igemm reduces over only `9 * kpg` there and
is already cheap.

What was done:

1. **A few-groups gate for the narrow class.** For `G <= 6`, `H <= 16` and
   `cpg >= 24`, direct needs a minimum number of igemm
   workgroups in total and a minimum number
   per group (the per-group term is what binds at `G = 5, 6`). An alternative
   rule (decline `G <= 6`, `H <= 16`, or `cpg >= 24` below roughly
   1.5x the gate at any group count) was checked against the data and not
   taken as is: its second clause also declines `G = 8` shapes that win
   (`cpg = 28/32`, `kpg = 4`, 15..16 rows), and its first clause declines
   larger few-group shapes that win clearly (16x16 and 12x12 at `G = 2/4`
   past the new gate). `cpg = 16` and tall images keep the general gate:
   measured wins there.
2. **A hold-out found two more pockets**, each turned into a rule and then
   re-checked on a further fresh hold-out:
   - `cpg = 24`, `kpg = 8`, `W <= 12` lost or tied even past the few-groups
     gate (two partly empty 16-wide output tiles per wave and a partly empty
     column strip), while `cpg = 24` at `W = 14/16` and `kpg = 4` won. Rule:
     at `kpg = 8` on a short image, decline when (`cpg` over its 16-wide
     tiles) times (strip fill) is below a fitted floor.
   - `G = 6` at 13 rows lost past the total gate; hence the per-group term.
     A last hold-out then put a 13-row `G = 4` shape a little below parity,
     with the other 13/14-row `cpg = 32`, `kpg = 8` shapes only just above
     it: these images do not split into 4-row tiles, so they stream whole on
     a longer serial row chain (a `block_h` sweep confirmed whole-image is
     still the best tile). The few-groups gate doubles there,
     giving up those
     small wins rather than risk the loss.
3. All new declines only move shapes to the igemm candidate with the
   baseline spec; no admitted shape changed its knobs. Across every shape of
   the earlier cohorts, the picks changed only for the shapes the new rules
   decline.

What remains near parity: a few `G = 8`, `cpg = 32` shapes at the gate
(13..14 rows) sit within a few percent of igemm, either side, next to
same-feature shapes that win (e.g. the same image at a smaller batch). No
feature separates them, so they stay on direct. In the trace, the pre-pass
duration depends on whether the queue is backed up when it is issued. It is
short when the host is still ahead and longer when kernels run
back-to-back, and the main kernel's length decides which mode a shape runs
in. Shapes whose margin is smaller than that swing are the ones that flip.

### Big filters on short images, `G = 7`, and 1x1 (found in the fifth adversarial measurement round)

A fifth adversarial measurement round timed regions earlier cohorts had not sampled and found three
losing pockets in the narrow class, all newly routed to direct:

1. **5x5/7x7 on short images.** Every earlier 5x5/7x7 narrow sample had at
   least 28 rows. On 12..16-row images the main kernel alone is at best level
   with igemm (the `KH`-row halo re-read per tile is a large share of the
   image) and the 5x5/7x7 transpose pre-pass costs clearly more than the 3x3 one,
   so the pipeline lost at `G = 2, 4, 32`, `kpg = 4` and `8`, bf16 and fp16.
   A sweep of 17..27-row images (forced direct, same session against the
   baseline) then showed `kpg = 8` losing or tying wherever the second
   column strip was 0.75..0.81 full (`W = 24..26`) and winning or tying from
   `W = 27` on, while `kpg = 4` won. Rules: decline the 5x5/7x7 narrow class
   on `H <= _DIRECT_DGRAD_SHORT_H`, and at `kpg = 8` below
   a minimum strip fill. A few wins on
   either side (a 16-row `G = 32` 5x5 shape, a 26-wide `G = 32` 7x7 one) are
   forgone.
2. **`G = 7`.** The few-groups gate stopped at `G = 6`; `G = 7` fell back to
   the general gate and lost on 12- and 14-row images. A `G = 7` sweep showed
   the 12..14-row images losing well past the general gate (in the
   per-group regime), while 16-row images won from about the total
   few-groups gate on, at `G = 7` and also at `G = 5/6`. An alternative
   rule with a per-group term at `G = 7` would have declined those
   16-row wins. Rule: the few-groups gate covers `G <= 7`, and its per-group
   term binds only on images shorter than 16 rows (16-row images split into
   two full 8-row tiles). This admits some 16-row `G = 5/6` shapes that the
   previous gate declined; those were checked on hold-out shapes and win.
3. **1x1 at `G = 2`.** `cpg = 32`, `kpg = 8` 1x1 at the general gate lost
   narrowly on 16x16 (bf16 and fp16), and won from a fixed
   igemm workgroup count on; `cpg = 16/24`,
   `kpg = 4` and `G >= 3` won at the general gate. Rule: for 1x1, `G <= 2`,
   `cpg = 32`, `kpg = 8`, require that many workgroups. A few small wins
   below it on other image sizes are forgone.

Each rule was fitted on the adversarial probe shapes plus a fitting sweep, then
checked on two fresh hold-out sets (shapes on both sides of every new
boundary). Admitted hold-out shapes won or tied; declined ones lost, tied or
were small forgone wins. Across all shapes of the earlier cohorts the picks
did not change; only the newly measured shapes moved, all of them from direct
to the baseline igemm spec, or (16-row `G = 5/6`) from igemm to direct.

The transpose pre-pass cost is close to a fixed cost per filter size (the
5x5/7x7 one costlier than the 3x3 one) and bimodal in the trace. It is what
turns most of these near-parity shapes into losses, so a fused weight
transform (no separate launch) is the lever that would reclaim them.
A fused weight transform (no separate pre-pass launch) removes the swing and
should take this class back. That is the integration follow-up.

Pinned in `test_narrow_reduction_few_groups_near_gate_keeps_igemm`,
`test_narrow_reduction_few_groups_admitted_where_direct_wins` and
`test_narrow_reduction_partial_output_tiles_keep_igemm`.

### Slotting a specialised kernel (cpg == 4 batched MFMA)

`ConvGroupedDirectDgradSpec` carries a `variant`. The 4x4x4 batched-MFMA
kernel for `cpg == kpg == 4` went in this way (`variant = "4c"`, selected ahead
of the rule table by `_select_direct_dgrad_4c_spec`); the spatial policy was
untouched, and `plan_direct_mfma_dgrad` learned the 4c and fused stage lists
in one place, so the dispatcher, the benchmark and the tests share them.

## Correctness bugs found and fixed

All were silent: each produced a wrong dX with no error, and each was missed
because the paths were exercised only with `groups == 1`, zero-filled scratch,
or 'same'-padded 3x3.

1. **Reorganize pre-pass applied the group offset twice.**
   `build_direct_reorganize_weights` indexed `W_T[k_new, r, s, c_new]` with
   `c_new = g*kpg + atom*K + c4*E`, but `W_T`'s last dim is the per-group `kpg`
   (the transpose writes `W_T[c_abs, r', s', k_in_g]`). Every group after the
   first read the wrong taps. This broke every main-kernel path that reads the
   coalesced layout (`runtime_k_loop`, `waves_k > 1`) on every grouped problem,
   not just `cpg = 4` where it was first reported.
2. **Unwritten workspace lanes.** The same pass skipped the store for lanes
   whose source is out of range (partial K-atom such as `kpg = 4`, partial
   M-tile). The main kernel zero-masks only the dY operand of a partial atom, so
   it multiplied stale scratch by zero: NaN in, NaN out. The pass now stores
   every lane, zero where the source is invalid.
3. **`fold_k32` + `waves_k` validated against the wrong atom count.**
   `validate()` checked `ceil(cpg/16) % waves_k`; under `fold_k32` the builder
   slices `cpg/32` atoms, so `cpg = 32, waves_k = 2` gave each wave zero atoms
   and the kernel wrote zeros. `validate()` now counts atoms at the width in use,
   and `is_valid_spec` runs `validate()` and checks the LDS footprint.
4. **Transposed `kpg` not a multiple of 4.** The flush stores 4 channels per
   lane and guards only `c4*4 < rows_in_tile`, so a group width of 6 wrote into
   the next group. Rejected in `validate()`; dispatch requires `cpg, kpg` to be
   multiples of 4.
5. **Non-'same' padding.** The row stream flushes output rows `0..H-1` of its
   input height; with pad 0 on a 3x3 the transposed problem pads by 2 and its
   last two output rows were never written. Rejected in `validate()`
   (`Ho == H`, `Wo == W` at stride 1); dispatch requires `2*pad == Y-1`.

The regression tests poison dX and every scratch buffer with 0xFF bytes (NaN in
fp16 and bf16). Re-introducing bug 1 makes every coalesced-path regression case
fail; with the fix they pass.

There is no C++ engine mirror of the generic `DirectConvSpec` builder or the
two pre-pass kernels (the byte-identity surface of `conv_direct_grouped` covers
the 4c/8c/16c/32c, depthwise, scalar-dgrad and wgrad builders), so these fixes
change no golden. The `conv` byte-identity gate stays green.

## Harness fixes

* The dgrad sweep never tried `block_h = 0` or `fold_k32` with one wave, which
  are the best points on the target shapes.
* Its label omitted `block_groups`, the strongest knob.
* The reorganize grid was sized with the 16-wide K-atom count even under
  `fold_k32` (twice the blocks; the extra ones only hit the buffer-bounds guard).
* Verification used a max-relative-error threshold on a zeroed output; it now
  uses the manifest-runner conv rule on poisoned buffers.

## Replay

From `platform/`, with `PYTHONPATH=$(pwd)/python:$(pwd)/../library` and
`ROCKE_CPP_STRICT=1`.

Dispatch policy and plan (CPU only):

```bash
python3 -m pytest ../library/tests/dispatch/test_grouped_conv_wgrad_dispatch.py -q -k Dgrad
python3 -m pytest ../library/tests/test_direct_mfma_dgrad_correctness.py -q \
    -k PlanAndValidation
```

Numerics of what dispatch ships, plus the coalesced-path regressions (gfx950):

```bash
python3 -m pytest ../library/tests/test_direct_mfma_dgrad_correctness.py -q
```

One request end to end (prints the selection, the rule row and every stage;
requests outside the region print the igemm pick). The driver lives in
`library/` (it drives the library dispatcher); run it from `library/` with
`PYTHONPATH=$(pwd)/../platform/python:$(pwd)`:

```bash
python3 benchmarks/common/grouped_conv/run_direct_dgrad_dispatch.py \
    --N 128 --C 512 --K 512 --H 14 --W 14 --G 16 --dtype bf16 --verify
rocprofv3 --kernel-trace --stats -d prof -o run -- python3 \
    benchmarks/common/grouped_conv/run_direct_dgrad_dispatch.py \
    --N 128 --C 512 --K 512 --H 14 --W 14 --G 16 --dtype bf16 --loop 50
```

Run the same two commands from a checkout without this change to A/B the igemm
pick (the driver runs whichever candidate dispatch selects, and imports the
direct-pipeline builders only when dispatch picks them). Use per-kernel
durations from the trace, not host-timed means: for a pipeline this short
the Python launch path, not the GPU, sets the host-timed period.

Step 0 sweep (all knobs, verified; `block_h` includes 4):

```bash
python3 ../library/benchmarks/common/benchmark_direct_conv.py --direction dgrad \
    --N 128 --C 512 --K 512 --Hi 14 --Wi 14 --Y 3 --X 3 --pH 1 --pW 1 \
    --groups 16 --dtype bf16 --verify --jobs 24
```

Byte identity:

```bash
export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python3 tools/check_byte_identity.py --only conv
ROCKE_LLVM_FLAVOR=llvm22 python3 tools/check_byte_identity.py --only conv
```

## Open items

* No C++ engine mirror, parity pair or golden exists for the generic
  `DirectConvSpec` main kernel, the transpose pre-pass or the reorganize
  pre-pass, so the reorganize fix and the new `validate()` rules have no
  byte-identity surface. This predates the change, but the change makes that
  builder the default grouped-dgrad dispatch on gfx950. Follow-up: port the
  transpose stage and the main kernel at the dispatched knobs to `platform/cpp/`
  with parity configs.
* `ConvGroupedRequest.vec_size_c` is an igemm epilogue hint; the direct
  candidate ignores it and says so in the dispatch explanation.

* The pre-pass cost model's safety margin declines some problems where the
  pipeline would beat igemm; a tighter main-kernel model would take some of
  them back. The root-cause lever is a main kernel that streams large weight
  slices from LDS instead of preloading them into registers: it would serve
  these shapes in the fused single-kernel form and retire the pre-pass (the
  transpose is already at the copy bound, so it cannot get cheaper).
* Done since: the weight pre-pass is fused into the main kernel's prologue
  wherever the preloaded fragments fit the register budget, and the batched
  4x4x4 kernel serves `cpg = kpg = 4` (see the integration section). The fused
  form's register budget still sends large filters with 32-wide channel groups
  to the pre-pass pipeline; staging their weights through LDS instead of
  registers is the lever there.
* The under-filled K atom is still why wide-output narrow reductions
  (`kpg <= 8`) lose on poorly filled strips; a narrow-K main kernel is the way
  to take them back.
* `runtime_k_loop` with `block_q = 32` is marginally better than the chosen
  point on 7x7; it needs the reorganize pass folded into the transpose to pay.
