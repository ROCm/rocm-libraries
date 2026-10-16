<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Grouped dgrad on the direct-MFMA kernels — gfx950 dispatch case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.

This records why grouped and small-channel stride-1 dgrad runs on the direct
MFMA kernels instead of the implicit GEMM, how the eligibility and knob policy
were fitted, how that policy was mapped onto the AOT cache, the correctness
bugs the work exposed, and the replay path. The row-stream knobs the policy
adds to the fused generic kernel are covered in
[`grouped_generic_row_stream_case_study.md`](grouped_generic_row_stream_case_study.md).

## Where it lives

| Piece | File |
| --- | --- |
| Policy (pure functions of a `DirectConvProblem` and the arch) | [`library/dispatch/grouped_conv_dgrad_direct.py`](../../../../../../library/dispatch/grouped_conv_dgrad_direct.py): `select_generic`, `select_4c`, `select_fused`, `_fused_block_h`, `_fused_block_q`, `_fused_stream_knobs`, `generic_shape_errors`, `generic_policy_errors`, `launch_errors` |
| Dispatch candidate | [`library/dispatch/grouped_convolution.py`](../../../../../../library/dispatch/grouped_convolution.py): `direct_mfma_conv_dgrad` (priority 5, ahead of `implicit_gemm_conv_dgrad` at 10), selection type `ConvDirectDgradSpec` |
| Cached knob image | [`library/benchmarks/common/direct_kernel_sweep.py`](../../../../../../library/benchmarks/common/direct_kernel_sweep.py): `DGRAD_4C_*`, `DGRAD_FUSED_*`, `_dgrad_fused_policy_reason` |
| Kernels | [`library/kernels/common/conv_direct_grouped.py`](../../../../../../library/kernels/common/conv_direct_grouped.py): `DirectConv4cSpec`, the generic `DirectConvSpec` with fused weights, `make_direct_dgrad_spec`, `dgrad_fused_block_h` |
| Selection -> cache identity | [`library/benchmarks/common/grouped_conv/dgrad_dispatch_cache.py`](../../../../../../library/benchmarks/common/grouped_conv/dgrad_dispatch_cache.py) |

The candidate is gfx950 only, fp16/bf16, NHWC, 2D. Everything it declines
goes to the implicit-GEMM candidate.

## What the candidate selects

`select_generic` first applies the capability checks of
`generic_shape_errors` (stride 1, dilation 1, square "same"-padded filters in
`GENERIC_FILTERS`, `cpg` and `kpg` multiples of 4 up to
`GENERIC_MAX_CHANNELS`, fused weight fragments within
`PRELOAD_WEIGHT_VGPR_BUDGET`), then picks one of two single-kernel variants:

| Shape class | Variant | Kernel |
| --- | --- | --- |
| `cpg == kpg == 4`, 1x1/3x3, groups a multiple of `_4C_BLOCK_GROUPS`, above its grid floors (`select_4c`) | `direct_grouped_dgrad_4c` | `DirectConv4cSpec`: one wave carries `_4C_BLOCK_GROUPS` groups at full 4x4x4 MFMA width, weights read in the prologue (LDS-staged with transpose reads on gfx950), row-staged input (`stage_rows`) where `_4c_takes_stage_rows` admits it |
| every other admitted shape (`groups == 1` included) | `direct_grouped_dgrad_fused` | generic `DirectConvSpec` row stream over the transposed problem with the weight transform fused into its prologue (`select_fused`) |

Finally it applies the measured-loss corners (`generic_policy_errors`) and the
launch limits (`launch_errors`). Both variants are one launch with no
workspace.

**No pre-pass pipeline is dispatched.** The PR also routed shapes whose fused
weight fragments exceed the register budget (large filters with wide channel
groups on both sides) to a two-launch pipeline: a weight transpose pre-pass
plus the main kernel, guarded by a predicted-cost model. In the port those
shapes fail `generic_shape_errors` (a capability: the fused variant has no
cache row for them, see `_dgrad_fused_weights_fit`) and run implicit GEMM. The
pre-pass variant `direct_grouped_dgrad_mfma` is still built by the direct cache
and swept by the benchmark, but no dispatch candidate picks it. The PR's cost
model, its staged transpose kernel and its tests are therefore not part of
the port.

## Mechanism: why implicit GEMM is the wrong family here

The implicit-GEMM dgrad tiles the per-group GEMM `M = N*H*W`, `N = cpg`,
`K = Y*X*kpg` with a 64-wide N tile. Small channel groups fill only a small
fraction of that tile, the K loop is a handful of steps, and each loop body
issues a few MFMAs against hundreds of VALU instructions of im2col address
arithmetic (static ISA of the dispatched kernels: a single-digit MFMA count
against several hundred VALU instructions in the body). The grid is one tile
column per group, so the device runs very many small, address-bound
workgroups.

The direct kernels compute dgrad as a forward convolution over the transposed
problem (`cpg' = kpg`, `kpg' = cpg`, flipped taps). The generic kernel streams
dY rows through a double-buffered LDS row, keeps a ring of `KH` accumulator
slots per output row and drives 16x16x16 (16x16x32 under `fold_k32`) MFMAs
straight from the staged row. With the fused weight transform the flip and
channel swap are prologue addressing, not a pre-pass launch and not per-element
address math in the hot loop.

## Levers swept and decisions

Every point was verified (NaN-poisoned dX) before it was timed; timing used
per-kernel trace durations, so host launch overhead did not enter the ranking.

| Lever | Finding | Decision (port name) |
| --- | --- | --- |
| Whole image (no H tiling) | Best when the untiled grid already fills the device: each dY row is loaded once per workgroup, no per-tile `KH - 1` halo | **Keep** (`_fused_block_h` returns 0; mapped onto a cached tile, see below) |
| 8-row tiles | Best on mid-size grids; forced on images taller than `_MAX_UNTILED_H` to bound the unrolled row loop | **Keep** (`_TILE_H`) |
| 4-row tiles | Best on small grids and on every 5x5/7x7 shape: twice the H tiles, half the serial row chain per workgroup | **Keep** (`_SMALL_TILE_H`; always for 5x5/7x7) |
| 16-row tiles as a policy choice | Not consistently best | Revert (the cache holds them only as whole-image targets) |
| Wave target for tall images | Images above `_SHORT_H` rows take the largest tile whose grid reaches `_TARGET_WAVES` | **Keep** |
| Short-image tile cost model | A single wave target cut slightly-taller-than-a-tile images into a full tile plus a near-empty one and lost; a row-chain-times-launch-rounds model (`(tile rows + KH - 1) * max(slots, waves)`, slots `_SHORT_WAVE_SLOTS` scaled by the rule's `block_groups`) picks the sweep-best tile | **Keep** for images up to `_SHORT_H` rows |
| `block_groups` | The strongest per-shape knob: keep `block_groups * max(cpg, kpg)` near a 32..64-channel slice. Keying it on `kpg` alone over-packed wide-`cpg` narrow-`kpg` shapes and lost | **Keep** (`_rule_block_groups` on `max(cpg, kpg)`) |
| `block_groups` halved for 5x5/7x7 | The `KH`-deep accumulator ring makes each wave heavy; fewer groups per workgroup keep the grid large | **Keep** |
| 32-column strips | Win on H-tiled images with wide channel groups (halo and per-strip weight traffic halve); lose on narrow channels, untiled grids, extra padding, and when the halved strip count drops the grid below a wave floor | **Keep** under those conditions (`_fused_block_q`, `_WIDE_MIN_WAVES`) |
| `fold_k32` (16x16x32) | Wins whenever `kpg % 32 == 0`, with a single wave | **Keep** where valid |
| `runtime_k_loop`, `waves_k > 1` | Needed a reorganize pre-pass that eats the gain; competitive only on large filters, below the tiled single-wave point | Revert (not dispatched; still built for the pre-pass variant) |
| Group width above the channel cap | Implicit GEMM wins (per-wave M tiles and accumulator ring outgrow the direct kernel) | **Excluded** (`GENERIC_MAX_CHANNELS`) |
| Filters above 7x7 | Correct, but the accumulator ring spills registers; at best level with implicit GEMM | **Excluded** (`GENERIC_FILTERS`) |
| `waves_per_eu` hint on small preload footprints | Neutral on aggregate; removes serialised next-row loads where it fires | **Keep** (`_FUSED_WPE` up to `_FUSED_WPE_MAX_VGPRS` preloaded VGPRs, 16-wide atoms only) |
| Kernel-level helper's own knob choice (`direct_dgrad_spec_for_problem`) | Lost on aggregate to the dispatch table, mostly on tile height and strip width | Not used by dispatch |

## Fitting the eligibility policy

The PR first fitted a sequence of gates (a problem-size gate, a
narrow-reduction rule, few-group and short-image gates, large-filter and 1x1
corners) while every direct dgrad still paid a separate weight pre-pass. Each
came from an adversarial measurement round: a per-kernel split showed whether
the main kernel or the pre-pass launch made a shape lose, and the mechanisms
recurred: a fixed pre-pass cost that small problems cannot amortise,
under-filled 16-wide MFMA K atoms when `kpg` is narrow, under-filled 16-wide
column strips, and the halo share on short images.

Once the weight transform was fused into the main kernel and the stride-1
implicit GEMM had become faster too, those gates both forgave wins and
admitted losses, so the policy was refitted from scratch rather than patched:

1. **Training draw.** Random shapes over the structural region (the channel
   pairs, filters, image sizes, group counts and batch sizes the kernels
   serve, both dtypes), measured in one session: implicit GEMM against the
   direct candidate forced past every policy decline, each verified first.
2. **Per-class trees, rewritten by hand.** Decision trees per class (tiny
   widths, wide 1x1 reductions, small problems, wide output on narrow
   reductions) suggested the split features. Each tree was rewritten as one or
   two readable conditions and scored on the whole draw against an oracle
   that always picks the faster kernel, together with the list of admitted
   losses and forgone wins.
3. **Hold-out.** A disjoint draw, measured the same way and not used for
   fitting, checked the final rules: the chosen kernel stays close to the
   oracle and no admitted shape loses by a large margin. The remaining
   admitted losses are small, scattered and not separable by the features
   available at dispatch time.

The resulting terms, in `generic_policy_errors` (every reason starts with
`POLICY_PREFIX`, so a caller can tell a performance decline from a
capability):

* tiny images (`_TINY_W`, `_TINY_HW`): the 16-wide output strips stay mostly
  empty;
* wide output (`cpg > _WIDE_CPG`, two 16-wide output tiles per wave) on very
  narrow images (`_WIDE_TINY_W`), or with a narrow side on strips filled below
  `_MIN_STRIP_FILL`;
* 1x1 with a wide reduction, implicit GEMM's best case, except on narrower
  channels, wide enough strips and enough work (`_PW_MAX_CHANNELS`,
  `_PW_MIN_W`, `_PW_MIN_TILES`);
* small problems: per-class floors for narrow reductions and four-channel
  groups (`_SMALL_TILES`, `_SMALL_NARROW_WIDE_MIN_TILES`,
  `_SMALL_NARROW_MIN_TILES`, `_SMALL_4CH_MIN_TILES`).

Work is counted as the implicit-GEMM candidate's own grid (`igemm_tiles`:
`_IGEMM_TILE_M`-row tiles times groups), not as a dX element count: for the
same dX size, smaller channel groups give more implicit-GEMM tiles, each
wasting more of its N tile, which is exactly where the direct kernel wins.
The 4c row has no corner terms; its own grid floors (`_4C_MIN_GRID`, and the
row-staged bounds `_4C_STAGED_*`) decide it, because one wave is one
workgroup and the kernel does not tile H, so with few images its grid is too
small and the H-tiled fused kernel wins.

In the port the thresholds were re-checked on the AOT binaries where they map
onto them. One changed: the row-staged 4c kernel takes 8-column tiles for 3x3
filters (`_4C_STAGED_BLOCK_Q_3X3`, two waves sharing the staged row and its
halo columns), which was the fastest cached tile there; 1x1 keeps 4-column
tiles.

## What the AOT port changed

**Capabilities vs kernargs.** A cached direct binary is a `KernelIdentity`:
variant, dtype, capabilities (`DirectCaps`: `KH`, `PAD`, `stride`, `cpg`,
`kpg`) and a knob dict. `N`, `H`, `W` and `groups` are kernel arguments. A
selection is a `ConvDirectDgradSpec` (variant, capabilities, sorted knob
pairs); `make_direct_dgrad_spec` is the one constructor shared by dispatch and
`direct_kernel_sweep.make_spec`, so a dispatched `(variant, knobs)` is exactly
the spec the cache compiled.

**ABI.** The 4c and fused kernels launch with the forward direct ABI
(`conv_direct_args_signature(dtype, direction="fwd")`) over the transposed
problem (`ConvDirectDgradSpec.launch_problem`), buffers `(dy, w, dx)`, no
workspace and no `sub_gemm_buf`; only the implicit-GEMM pick takes the
implicit-GEMM argument signature.

**Whole-image picks.** `block_h = 0` does not exist in the cache: the image
height is a kernarg, so `block_h > 0` is mandatory and baked. A whole-image
decision of `_fused_block_h` maps to `dgrad_fused_block_h(H)`, the smallest
tile of `DGRAD_FUSED_BLOCK_H` at least as tall as the image (the tallest when
none is, so the tallest images stream in several bands of it). Two knobs make
the dead rows of a too-tall tile cheap:

* `row_guard` (set on every fused binary by `make_direct_dgrad_spec`): the
  MFMAs of each row past `p_Hi` run in a block-uniform one-trip loop, so they
  are skipped on short images;
* `row_exit` (on the tile heights in `DGRAD_FUSED_EXIT_BLOCK_H`, which the
  policy only reaches through whole-image picks): the workgroup leaves once
  the next row to flush is past `p_Hi`, skipping the loads, staging, barriers
  and masked flushes of those rows too.

The spatial policy itself (`_fused_block_h`, `_fused_block_q`) still reasons
about whole images, so the cost model and the row-stream rule see the image
the PR saw; only the binary it names changed.

**Every knob limited to the cached image.** The fused variant does not cache
a product of knobs but the policy's image (`_dgrad_fused_policy_reason` filters
`_knob_grid`): `DGRAD_FUSED_BLOCK_Q`, `block_groups` at or below the
per-channel rule (`DGRAD_FUSED_BLOCK_GROUPS`), tile heights
`DGRAD_FUSED_BLOCK_H` (`DGRAD_FUSED_BLOCK_H_LARGE` for 5x5/7x7), `fold_k32`
exactly where `kpg` allows it and the atom is valid, LDS-staged weights where
valid and gathered otherwise, the waves-per-EU hint where the policy adds it,
and the row-stream stack (`DGRAD_FUSED_STREAM`). Correspondingly `select_fused`
halves `block_groups` until it divides `groups` (a kernarg; `plan_for`
re-checks it on the real problem), tries `fold_k32` and LDS-staged weights and
keeps them only if the cached form is valid, and returns `None` (decline)
where no cached form is valid. The 4c row is limited the same way
(`DGRAD_4C_BLOCK_Q`, `DGRAD_4C_BLOCK_GROUPS`, `DGRAD_4C_FORMS`).

**Launch limits.** Because sizes are kernargs, `launch_errors` declines a pick
whose tensors exceed the i32 byte sizes of the direct ABI or whose grid
exceeds the y/z launch extent.

**Two guards keep dispatch and cache in step.**
`test_dgrad_fused_policy_image_is_enumerated` replays the policy over a
random spread of problems and requires every pick to be a cached, valid
identity; `test_dgrad_dispatch_outputs_are_cached_identities` does the same
through the dispatcher for every gfx950 dgrad selection over a synthetic grid
that reaches every candidate.

## Correctness bugs found and fixed

All were silent wrong-answer bugs, missed because the paths had only been
exercised with one group, zero-filled scratch, or "same"-padded 3x3.

1. **Reorganize pre-pass applied the group offset twice**, and
2. **left out-of-range workspace lanes unwritten** (partial K atom or M tile),
   so stale scratch reached the MFMA as `0 * NaN`. Both live in
   `build_direct_reorganize_weights`, which the coalesced forms
   (`waves_k > 1`, `runtime_k_loop`) of the cached pre-pass variant launch;
   both are fixed in the Python and C++ emitters
   (`conv_direct_grouped_build_dgrad_transforms.cpp`), and
   `test_mfma_dgrad_baked` in `tests/test_direct_conv_correctness.py` runs
   every case grouped, with partial K atoms and NaN-prefilled dX. Caches built
   before the fix keep the old binary and must be rebuilt.
3. **Transposed `kpg` not a multiple of 4.** The flush stores four channels
   per lane and would write into the next group. Rejected by `is_valid_spec`
   and by `generic_shape_errors`
   (`test_transposed_kpg_must_be_a_store_vector_multiple`).
4. **Non-"same" padding.** The row stream flushes output rows of its input
   height, so the last rows of a non-"same" problem were never written.
   Rejected by the validator and by dispatch
   (`test_non_same_padding_is_rejected`).

The PR also fixed the generic validator's atom count for `fold_k32` with
`waves_k > 1` (it counted 16-wide atoms). That fix is not in the port: the
port's `DirectConvSpec.validate()` still checks `ceil(cpg / 16) % waves_k`.
The fused variant never sets `waves_k > 1`, so nothing dispatched is
affected; check the pre-pass variant's `fold_k32` + `waves_k > 1` entries
before dispatching it.

The generic `DirectConvSpec` builder has a C++ mirror
(`platform/cpp/instances/common/conv_direct_grouped_build_generic.cpp`, spec
and validator in `conv_direct_grouped_spec_and_validate.cpp`) including the
fused-weight and row-stream knobs, byte-identity gated by
`conv_direct_dgrad_ext` parity configs 30..44.

## Tests

From `library/` (GPU tests need gfx950):

* `tests/dispatch/test_grouped_conv_wgrad_dispatch.py`,
  `TestGroupedDirectDgradDispatch`: routing with cached knobs, out-of-region
  requests on implicit GEMM, the `POLICY_PREFIX` declines, the 4c group rule,
  grid/block equal to the kernel's, other arches never see the candidate.
* `tests/dispatch/test_grouped_conv_dgrad_dispatch_aot.py`: every selection
  over a synthetic grid is a cached identity offered by `plan_for`, and the
  direct signatures equal the cached plan's step signature.
* `tests/test_direct_kernel_cache.py`: `test_dgrad_fused_cached_scope`,
  `test_dgrad_fused_policy_image_is_enumerated`,
  `test_dgrad_fused_rows_fit_the_weight_register_budget`.
* `tests/test_direct_mfma_dgrad_correctness.py`,
  `tests/test_conv_dgrad_fused_weights.py`, `tests/test_conv_dgrad_4c.py`:
  validation on CPU, numerics of the cached binaries on adversarial shapes
  with NaN-prefilled dX.

## Replay

From `library/`, with `PYTHONPATH=$(pwd)/../platform/python:$(pwd)`.

Build the caches (no GPU needed):

```bash
python3 -m benchmarks.common.benchmark_direct_conv --compile-all \
    --direction dgrad --dtype bf16 --cache-dir <direct cache> --jobs 0
python3 -m benchmarks.common.benchmark_implicit_gemm_conv --compile-all \
    --direction dgrad --dtype bf16 --cache-dir <igemm cache> --jobs 0
```

Replay exactly what dispatch ships (prints the candidate, variant and knobs;
loads the cached binary, compiles nothing). Verify and time in separate
processes:

```bash
python3 -m benchmarks.common.grouped_conv.run_direct_dgrad_dispatch \
    --N 8 --H 12 --W 12 --C 96 --K 96 --G 24 --Y 3 --dtype bf16 \
    --cache-dir <igemm cache> --direct-cache-dir <direct cache> --verify
python3 -m benchmarks.common.grouped_conv.run_direct_dgrad_dispatch \
    --N 8 --H 12 --W 12 --C 96 --K 96 --G 24 --Y 3 --dtype bf16 \
    --cache-dir <igemm cache> --direct-cache-dir <direct cache> --time
```

`--loop <n>` issues back-to-back launches for `rocprofv3 --kernel-trace`;
use per-kernel durations from the trace rather than host-timed means.

Sweep every cached single-kernel grouped dgrad form of one shape (verified):

```bash
python3 -m benchmarks.common.benchmark_direct_conv --direction dgrad \
    --run-from-cache <direct cache> \
    --variants direct_grouped_dgrad_4c,direct_grouped_dgrad_fused \
    --N 8 --Hi 12 --Wi 12 --C 96 --K 96 --Y 3 --X 3 --pH 1 --pW 1 \
    --groups 24 --dtype bf16 --verify
```

Tests and byte identity:

```bash
python3 -m pytest tests/dispatch/test_grouped_conv_wgrad_dispatch.py \
    tests/dispatch/test_grouped_conv_dgrad_dispatch_aot.py -q
python3 -m pytest tests/test_direct_kernel_cache.py -q -k dgrad_fused
cd ../platform && python3 tools/check_byte_identity.py --only conv_direct
```

## Open items

* Shapes past the fused register budget run implicit GEMM. A main kernel that
  streams large weight slices from LDS instead of preloading them into
  registers would serve them in one launch.
* The under-filled K atom is still why wide-output narrow reductions decline
  on poorly filled strips; a narrow-K main kernel is the lever there.
* `ConvGroupedRequest.vec_size_c` is an implicit-GEMM epilogue hint; the
  direct candidate ignores it and says so in the dispatch explanation.
