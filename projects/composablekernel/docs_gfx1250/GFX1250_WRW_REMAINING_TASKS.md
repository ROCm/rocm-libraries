# gfx1250 Grouped-Conv Backward-Weight (wrw) — Remaining Tasks

This document hands off the WRW (backward-weight convolution) optimization
work remaining after `GFX1250_CONV_OPTIMIZATION_ROADMAP.md` items T0-01
through T2-04 and the follow-on "large_tiles instance family" item (commit
`b84a22d479c`, done). Each task below is scoped to be **independently
implementable** by an agent with no other context: it states the problem,
cites exact evidence (file:line), gives a concrete implementation plan, and
gives an exact validation recipe using tools already proven to work in this
environment.

**Status of T2-01 (block-diagonal WMMA packing)**: implemented and verified
for `grouped_conv2d_fwd` only (commit `932dce7b168`, ~97x speedup at the
target shape). **Not yet ported to `grouped_conv2d_bwd_weight`** — see Task 4.

## Environment / tooling recap (so an agent can move immediately)

- Toolchain: `~/rocm-10.1` (hipcc, ROCm 7.16). Build with:
  ```
  cmake -D CMAKE_PREFIX_PATH=/home/sgundabo/rocm-10.1 \
        -D CMAKE_CXX_COMPILER=/home/sgundabo/rocm-10.1/bin/hipcc \
        -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS=gfx1250 \
        -D BUILD_CK_PROFILER=ON ..
  make -j100 ckProfiler   # -j100, NOT higher — higher risks OOM on this box
  ```
- Profiler op for this whole document: `grouped_conv_bwd_weight`. Full arg
  order (2D, NHWGC/GKYXC/NHWGK layout, bf16):
  ```
  ckProfiler grouped_conv_bwd_weight 5 2 <verify:0|1> <init:1> 0 1 2 \
    <G> <N> <K_per_group> <C_per_group> <Y> <X> <Hi> <Wi> \
    <Sy> <Sx> <Dy> <Dx> <LPy> <LPx> <RPy> <RPx> <SplitK|-1|all>
  ```
  `dtype=5` is bf16/bf16/bf16, `layout=2` is NHWGC/GKYXC/NHWGK. **`K`/`C` are
  per-group** (MIOpen's totals divided by `G` — this bit a real bug earlier in
  this investigation, see `script/convert_miopen_driver_to_profiler.py`
  `run_ck_profiler()`, which does exactly this division).
- 29 representative MIOpen-driver-derived shapes (converted with the same
  script) live at `projects/composablekernel/miopen_wrw_shapes.txt`
  (`name|full ckProfiler arg string`, one per line); baseline timings at
  `projects/composablekernel/miopen_baseline_results.csv`. Both files are
  currently **untracked** — `git add` them if you rely on them long-term.
  Regenerate a full run with:
  ```
  bash /tmp/run_miopen_shapes.sh <ckProfiler_binary> <output.csv>
  ```
  if that script no longer exists, it is trivial to recreate: it loops the
  shapes file, runs each command with `timeout 90`, greps the
  `Best configuration parameters:` block (`avg_time:`, `tflops:`, `GB/s:`,
  `name:`) into a CSV row.
- **Rigor lesson learned this session**: this machine has visible
  run-to-run noise (confirmed by the user — likely concurrent users). A
  single before/after reading is not sufficient evidence of a regression or a
  win. Any performance claim in this document's validation steps should be
  confirmed with **interleaved A/B on two separately-saved binaries**
  (`cp build/bin/ckProfiler /tmp/ckProfiler_A`, rebuild the other variant, `cp`
  again, then alternate calls `/tmp/ckProfiler_A ARGS` / `/tmp/ckProfiler_B
  ARGS` in a loop), not two sequential blocks of runs separated by a rebuild.
  A one-off "big" delta (e.g. 3x, 30%) that doesn't reproduce on retest is
  noise, not a real effect — this happened twice in the current session
  (see Task 1's write-up).

---

## Task 1 — Split-K auto-heuristic for `TwoStage` WMMA bwd-weight is measurably suboptimal for small-grid shapes

**Priority**: high (proven real, bounded effort, no kernel changes needed).
**Risk**: low if scoped correctly (see the failure mode below).

### Problem

`get_best_occupancy_k_batch_value()`
(`include/ck/tensor_operation/gpu/device/impl/split_k_utils.hpp:37-62`) picks
`k_batch = floor(max_occupancy * num_cu / grid_size)` — it purely maximizes
GPU-wide occupancy with no regard for how much reduction work (K-iterations)
each split-K workgroup is left with. For problems where `grid_size` (number
of output-tile workgroups, before splitting K) is small — which is common
for wrw's `TwoStage` path since its M/N tiles are small
(`DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<32,16,16,32,...>` is
currently the *only* registered TwoStage bf16 instance, see Task 2) — this
picks a k_batch so large that each workgroup does only ~3 main-loop
iterations, at which point fixed per-workgroup overhead (descriptor setup,
LDS init, epilogue write, plus TwoStage's own stage-2 reduce pass) dominates
and measured performance *regresses* relative to a smaller k_batch, despite
"better occupancy."

**Concrete measured evidence** (shape: `G=1,N=42,K=24,C=3,Y=X=3,Hi=480,Wi=640,
stride=2,pad=1` — a real RGB-stem-style conv, ckProfiler args
`1 42 24 3 3 3 480 640 2 2 1 1 1 1 1 1`):

| SplitK | avg_time (ms) | TFlops |
|---|---|---|
| 128 | 1.063 | 3.93 |
| 256 | 0.656 | 6.37 |
| 512 | 0.416 | 10.05 |
| **1024** | **0.346** | **12.09 (peak)** |
| 2048 (auto-picked) | 0.407 | 10.28 |
| 4096 | 0.658 | 6.35 |

Auto-heuristic picks 2048; true optimum for this shape is ~900-1200
(measured plateau 11.8-12.2 TFlops across 900/1024/1200), a consistent
**~15-18% win** available.

### Why the obvious fix is wrong — learn from this before attempting

A naive fix (already attempted and reverted this session): cap
`k_batch_` so `gemmK / k_batch_ >= MinIters * KPerBlock * ABK1` for some
constant `MinIters` (e.g. 12), applied inside
`device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`'s
`Argument` constructor (around the `if(split_k < 0) { ... k_batch_ =
get_best_occupancy_k_batch_value(...); ... }` block). This fixed the shape
above (SplitK auto-picked 1050, 11.83 TFlops — matches the true optimum) but
**broke a different shape** (`G=1,N=42,K=10,C=128,Y=X=1,Hi=120,Wi=160`,
ckProfiler args `1 42 10 128 1 1 120 160 1 1 1 1 0 0 0 0`): its true optimum
is `SplitK=1024` (11.56 TFlops) which the *unmodified* occupancy-only
heuristic already finds correctly, but the "MinIters" cap over-restricted it
to `SplitK=262` (only 8.82 TFlops available among instances at that cap) — a
**26.6% regression**.

Root cause of the discrepancy: the two shapes need genuinely different
"sweet spot" iteration counts (~12 iterations/split for the first shape,
~3 for the second), which correlates with `grid_size` (how parallelism-
starved the GPU already is from the base M×N tile count), not with
`gemmK` alone. The existing occupancy-only formula already correctly
solves the second shape; the fix needs to identify *when* it is wrong, not
replace it universally.

### Concrete implementation plan

1. In `device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`
   (and consider the sibling `device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp`
   one-stage device op, and `device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp`
   / `device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp` XDL siblings — all five
   call the same shared `get_best_occupancy_k_batch_value`), build a **joint**
   correction that only tightens the split when the base problem is
   padding-waste-dominated: e.g. only reduce k_batch below the occupancy
   heuristic's pick when `NPerBlock` (or `MPerBlock`) padding waste for the
   *actual* problem `N`/`M` exceeds some threshold (concretely: `N_actual <
   NPerBlock / 2` or similar) AND `grid_size` is already small. This targets
   exactly the narrow-C stem-conv case (Task's example 1) without touching
   the well-behaved case (Task's example 2, where `N=128` fills `NPerBlock=16`
   exactly, no padding waste).
2. Alternatively/additionally: since CK already computes `max_occupancy_`
   via `hipOccupancyMaxActiveBlocksPerMultiprocessor` for the *specific*
   kernel template instantiation, consider deriving the iteration floor from
   that kernel's own measured occupancy rather than a global constant — e.g.
   scale the floor by `1 / max_occupancy_` so kernels with low intrinsic
   occupancy (implying more register/LDS pressure, likely more prologue
   overhead relative to steady-state) get a higher floor.
3. Whatever formula is chosen, it must be validated on **at least 15-20
   shapes spanning the full parameter space** (small M large N, large M
   small N, both large, both small, various `Y,X,stride`), not 2 — this is
   exactly what caught the regression above. Use
   `projects/composablekernel/miopen_wrw_shapes.txt` as the base corpus, and
   supplement with a few synthetic extreme-aspect-ratio shapes.

### Validation recipe

```bash
# Full 29-shape sweep, verify=1, compare avg_time_ms columns before/after
bash /tmp/run_miopen_shapes.sh <binary> <out.csv>
# For any shape whose SplitK changes: interleaved A/B per the rigor note above,
# ≥5 rounds each direction, before declaring win or regression.
```

---

## Task 2 — Validate and selectively re-enable disabled `TwoStage` bf16/f16 tile configs

**Priority**: medium (real potential upside, but demonstrated hardware risk
— must be done config-by-config, not in bulk).
**Risk**: **high if done carelessly** — see the crash below.

### Problem

`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp`
registers, for both `bf16` and `f16`, **exactly one** active
`DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3` tile config
(`<32,16,16,32,...>`, i.e. `BlockSize=32, MPerBlock=16, NPerBlock=16,
KPerBlock=32, ABK1=8`). 7 other, larger tile configs sit commented out per
dtype (2 explicitly marked `// Incorrect results for at least GemmDefault`;
5 have no comment explaining why they're disabled). Since the `TwoStage`
family is the *only* instance family that supports narrow-channel shapes at
all (see Task 3), and it currently has only this one tiny tile, every
narrow-channel wrw shape is stuck on a config with heavy compute waste
regardless of its actual M/N size.

### Concrete, demonstrated hazard — read before touching this file

Earlier this session, all 7 disabled configs (except the 2 explicitly
marked incorrect) were uncommented in bulk and rebuilt. Result: **real
hardware crash**, `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, on two 1×1-conv
shapes (`G=1,N=42,K=256,C=16,Y=X=1,Hi=Wi=1` and `G=1,N=42,K=512,C=32,Y=X=1,
Hi=Wi=1`) via the `<128,128,128,32,...>` config specifically. The GPU
recovered on its own (subsequent ckProfiler invocations worked normally),
but this confirms these configs are commented out because **they are
broken**, not merely "unvalidated but probably fine." The change was
reverted in full.

### Concrete implementation plan

1. Re-read the 7 candidate configs at
   `device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp` (currently
   lines ~76-85 for bf16, ~49-58 for f16 as of this writing — grep for
   `DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3` inside the
   `_two_stage_wmma_instance.hpp` file to get current line numbers, they may
   have shifted).
2. Uncomment **exactly one** config at a time (start with the smallest,
   e.g. `<64,64,64,64,...>`, before the `<128,128,128,32,...>` config known
   to crash — that one specifically needs root-causing, not just retrying).
3. Rebuild `ckProfiler` only (not the whole tree).
4. Run `verify=1` across a **broad shape matrix** before trusting it:
   - 1×1 conv, `Hi=Wi=1` (the exact shape class that crashed)
   - 1×1 conv, `Hi=Wi>1` (e.g. 28×28, 56×56)
   - 3×3 conv, stride 1 and stride 2, various `Hi/Wi`
   - channel counts both divisible and not divisible by 4/8/16 (this
     instance file's vectorized `BlockTransfer*ScalarPerVector` fields are
     4-8; a config with e.g. `ScalarPerVector=8` will silently be rejected by
     `IsSupportedArgument()` for non-divisible channel counts rather than
     crash — the *crash* risk is specifically for channel counts that pass
     the support check but still hit a real memory-addressing bug).
5. Keep only configs that pass `verify=1` cleanly on the *entire* matrix
   with **zero** crashes, zero incorrect-result reports. Discard (leave
   commented, with a note of what was tried and what failed) anything that
   doesn't.
6. For configs that pass: benchmark against the existing single tile
   (interleaved A/B, per the rigor note) on shapes where the new config
   actually gets selected as "best" (check via `--list-instances` or the
   printed `Best configuration parameters:` block) to confirm real upside
   before committing.

### Validation recipe

```bash
# Sanity: does the new config even get selected for any of the 29 shapes?
./bin/ckProfiler grouped_conv_bwd_weight 5 2 1 1 0 1 2 <shape args> -1 2>&1 | grep "Best configuration"
# Full crash-and-correctness sweep across the matrix in step 4, verify=1 throughout.
# GPU health check after any suspected crash:
rocm-smi --showhw   # confirm device still enumerates
./bin/ckProfiler grouped_conv_bwd_weight 5 2 0 1 0 1 2 <any known-good shape> -1  # confirm still functional
```

---

## Task 3 — Narrow-channel (C<16) non-grouped conv backward-weight needs a real kernel, not instance tuning

**Priority**: medium-low (real, large potential upside for a common shape
class — RGB stem convs — but requires new kernel engineering, not a config
tweak).
**Risk**: low to attempt (additive — a new kernel path, existing paths
untouched) but nontrivial engineering effort.

### Problem

For a non-grouped (G=1) conv with small input-channel count (e.g. the RGB
stem conv `C=3`), the backward-weight GEMM has `N = C = 3` — far below any
WMMA tile's native width (16 for `NPerWmma`). The *only* instance family
that even supports such a shape is `TwoStage` (all one-stage
`Wmma_CShuffleV3` instances require `BBlockTransferSrcScalarPerVector` of
4-8, which rejects `C=3` outright via `IsSupportedArgument()`), and even
`TwoStage`'s tile wastes ~81% of every WMMA op's N-lanes on zero-padding.
Measured: `10.1-10.3 TFlops` for `G=1,N=42,K=24,C=3,Y=X=3,Hi=480,Wi=640,
stride=2` vs. `560-577 TFlops` for a "normal" `C=128,K=128` shape at similar
overall problem size — roughly **55x** lower utilization, and no amount of
split-K or instance-selection tuning can close this gap (confirmed: split-K
sweep finds no config above ~12.2 TFlops for this shape; instance-widening
is blocked by vectorization requirements per Task 2).

### Concrete implementation plan

Two independent approaches, either is viable:

**(a) VALU/scalar fallback kernel for small C.** For `C` below some
threshold (e.g. `C < NPerWmma`), a plain vector-load + FMA reduction kernel
(no WMMA at all) processing the full `N=C` width per thread has zero
padding waste and could plausibly beat the WMMA path outright for this
regime, since WMMA's advantage (throughput per matrix op) evaporates when
81%+ of every op is wasted. This is new kernel work: a new
`DeviceGroupedConvBwdWeight` implementation (or a scalar specialization
inside the existing `TwoStage` gridwise gemm, gated on `N < threshold`).
Reference: `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`
for the overall two-stage structure (stage 1: split-K partial-sum GEMM to an
fp32 workspace; stage 2: `kernel_batched_elementwise` reduce+cast) to reuse
as much scaffolding as possible; only stage 1's inner GEMM loop would need a
non-WMMA variant.

**(b) N-padding-aware block-diagonal-style packing**, conceptually similar
to Task 4's block-diagonal WMMA packing but for the *non-grouped* case:
since there's only one group here, there's nothing to pack across groups,
but a similar idea — pack multiple *spatial* or *batch* output tiles'
worth of the tiny-N problem into one WMMA op's N-lanes — could reclaim the
wasted lanes. This is more novel (no existing CK or hipConv/MISA precedent
found in this investigation for this specific case) and higher risk/effort
than (a).

**Recommendation**: attempt (a) first — it is bounded, well-precedented
(every GEMM library has a scalar fallback path for degenerate tile shapes),
and directly comparable in effort/reward to Task 1.

### Validation recipe

```bash
# Target repro shape:
./bin/ckProfiler grouped_conv_bwd_weight 5 2 1 1 0 1 2 1 42 24 3 3 3 480 640 2 2 1 1 1 1 1 1 -1
# Baseline to beat: ~10.1-10.3 TFlops (SplitK 2048, TwoStage<32,16,16,32>).
# Also re-run the same shape's neighbors (C=1,2,4,8 at various K,H,W) to
# confirm the new path activates and helps across the whole narrow-C regime,
# not just this one point.
```

---

## Task 4 — Port T2-01 block-diagonal WMMA packing from `grouped_conv2d_fwd` to `grouped_conv2d_bwd_weight`

**Priority**: high (largest proven upside of any item on this list — ~97x
at the fwd target shape — and now has a working, hardware-verified template
to follow instead of designing from scratch).
**Risk**: medium (nontrivial port across a materially different device-op
family, but the design is proven and the fwd commit is a literal template).

### Problem

`docs_gfx1250/T2-01_BLOCK_DIAGONAL_WMMA_PACKING_DESIGN.md` describes packing
`GroupsPerWmma` consecutive small-channel groups block-diagonally into one
WMMA instruction, instead of padding each group's tiny K/C up to a full
WMMA tile and wasting most of it on zeros — this is exactly the same
underlying inefficiency as Task 3, but for **grouped** (not non-grouped)
small-channel convs. **This has been implemented and hardware-verified for
`grouped_conv2d_fwd` only** (commit `932dce7b168`,
`gfx1250: implement T2-01 block-diagonal WMMA packing for grouped_conv2d_fwd`):
at the target shape `G=32,N=64,C=128,K=128,Y=X=3,Hi=Wi=28`
(`C_per_group=K_per_group=4, GroupsPerWmma=4`), measured **4.56ms/0.10 TFlops
→ 0.047ms/9.86 TFlops, ~97x**, verified correct against a CPU reference.

**Not yet ported to `grouped_conv2d_bwd_weight`.** The depthwise/small-
channels-per-group wrw shapes this session profiled and explicitly
deprioritized as "not tractable via instance tuning" (`M06`: `g=192,
C_per_group=K_per_group=1`, 0.335 TFlops; `M14`/`M15`: `g=256`, ~0.40 TFlops;
`M19`: `g=3`, 2.26 TFlops; `M25`: `g=512`, 0.36 TFlops — all in
`miopen_baseline_results.csv`) are exactly this shape class, and are exactly
what the fwd implementation's ~97x precedent suggests is achievable here
too.

### Concrete implementation plan (mirror the fwd commit's structure)

The fwd commit touched exactly these files/mechanisms — port each
analogously to the bwd_weight device-op family:

1. **Transform layer**: fwd added a `GroupsPerWmma` template param to
   `include/ck/tensor_operation/gpu/device/impl/transform_conv_fwd_to_gemm.hpp`,
   restructuring the A/B/E grid-descriptor construction to merge
   `GroupsPerWmma` consecutive groups' channels into the GEMM's K (for A) or
   a packed-buffer view (for B), and unmerging them back out of N (for E)
   via the tensor's real per-group stride (no xor/pad trick, unlike
   `NumGroupsToMerge`). For bwd_weight, find the analogous transform file
   (`transform_conv_bwd_weight_to_gemm.hpp` or equivalent — check
   `include/ck/tensor_operation/gpu/device/impl/` for the exact name used by
   `device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp`) and apply the same
   restructuring, accounting for wrw's different A/B/E roles: **A = output
   gradient** (fwd's role for the input activation), **B = input activation**
   (fwd's role for weight), **E = weight** (fwd's role for output) — the
   specific merge/unmerge assignment per operand will differ from fwd's
   because wrw's GEMM-M/N/K mapping is `M = K_out*Y*X`, `N = C_in`, `K =
   N_batch*Ho*Wo` (see `get_bwd_weight_gemm_sizes` in
   `include/ck/tensor_operation/gpu/device/impl/split_k_utils.hpp:65-92` for
   the exact existing mapping), unlike fwd's `M = N_batch*Ho*Wo`, `N =
   K_out`, `K = C_in*Y*X`.
2. **Device-op layer**: fwd added the same `GroupsPerWmma` param to
   `device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp`, statically
   scoped to plain `grouped_conv2d_fwd` (no multi-A/B/D, no NGCHW transpose,
   default conv specialization) so existing `GroupsPerWmma=1` instances are
   provably unaffected. Port the same param, same scoping discipline, into
   `device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp` (the one-stage
   variant — start here, not `TwoStage`, since one-stage is simpler and is
   what "normal" wrw shapes already use; extend to `TwoStage` only after the
   one-stage port is validated, since `TwoStage` has its own prepass/reduce
   structure that interacts with packing differently).
3. **Prepass kernel**: fwd added `kernel_pack_block_diagonal_wmma_weight`,
   one thread per packed-buffer element, gated by `(row's group-in-cluster
   == col's group-in-cluster)` to realize the block-diagonal zero pattern
   once at pack time (avoiding a per-WMMA-lane predicate). For bwd_weight,
   the operand that needs packing is likely the **input activation** (B,
   analogous to fwd's weight-packing role) — write an analogous prepass
   kernel; reuse fwd's exact gating logic and workspace-sizing pattern
   (`GetWorkspaceATensorSizeBytes`/`GetWorkspaceBTensorSizeBytes`-style
   methods already exist on the `TwoStage` `Argument` struct at
   `device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp:715-735`
   as a pattern reference even though this port targets one-stage first).
4. **Validity checks**: fwd fixed a real bug in this work — the B vector-
   access validity check tested against the *original* per-group C instead
   of the packed buffer's real contiguous run (`GroupsPerWmma*Y*X*C`),
   incorrectly rejecting valid wide-vector configs. Watch for the exact
   analogous check in whichever operand gets packed for bwd_weight.
5. **New example**: fwd added a standalone validation example at
   `example/70_grouped_conv2d_fwd_wmma_block_diagonal/`. Add an analogous
   `example/7x_grouped_conv2d_bwd_weight_wmma_block_diagonal/` validating
   against `ReferenceConvBwdWeight` (CPU reference) at a wrw-appropriate
   target shape — use the roadmap's own depthwise shapes as targets, e.g.
   something in the shape family of `M06` (`g=192, C_per_group=K_per_group=1`)
   or a slightly less extreme `C_per_group=K_per_group=4` analogous to fwd's
   validated shape, tuning `GroupsPerWmma` and instance parameters (vector-
   transfer widths capped by each operand's real contiguous run, `KPerBlock`
   swept) the same way fwd's example did.

### Validation recipe

```bash
# Correctness: build and run the new example directly (mirrors fwd's own validation)
./bin/<new_example_binary>   # should report PASS against CPU reference

# Perf: before/after on the exact depthwise shapes already profiled this session
./bin/ckProfiler grouped_conv_bwd_weight 5 2 1 1 0 1 2 192 42 1 1 3 3 120 160 2 2 1 1 1 1 1 1 -1  # M06-equivalent, current: 0.335 TFlops
./bin/ckProfiler grouped_conv_bwd_weight 5 2 1 1 0 1 2 256 42 1 1 3 3 60 80 1 1 1 1 1 1 1 1 -1    # M14-equivalent, current: ~0.40 TFlops
# Full 29-shape sweep to confirm zero regression on non-target (non-depthwise) shapes,
# since GroupsPerWmma=1 (untouched path) must remain bit-identical in behavior.
```

---

## Task 5 — `LargeTensors` grouped-conv backward-weight disabled on gfx1250 — investigate rocKE's `ds_load_tr16_b128` hazard as root cause

**Priority**: high (concrete, well-evidenced, single actionable lead;
correctness fix, not perf — but unblocks a whole shape class that currently
silently falls back to a slower path).
**Risk**: medium — requires care with LLVM-level codegen; low blast radius
if scoped to the exact guarded code path.

### Problem

`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp:1442-1448`
(`IsSupportedArgument()`):
```cpp
// Memory access runtime error on gfx1250 (inconsistent across runs)
// TODO: need fix
if constexpr(LargeTensors)
{
    if(is_gfx125_supported())
    {
        return false;
    }
}
```
unconditionally rejects the `LargeTensors=true` template instantiation on
gfx1250 — confirmed still present, current lines as of this writing.
Matches `CHANGELOG.md` 1.3.0's entry ("Disabled the large tensor XDL grouped
convolution backward weight instances on gfx1250, where they could produce
intermittent memory access faults"), and the identical pattern also exists
for **bwd_data**
(`device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v3.hpp:992-1000`) and
**fwd**
(`device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp:1456-1464`) — so a
fix here is potentially reusable across all three directions
(`GFX950_VS_GFX1250_COVERAGE.md` §5.5).

**"Inconsistent across runs" is the key symptom** — it is the exact
signature of a data race / uninitialized-read hazard, not a deterministic
tile-size limitation, which is why this is a strong candidate for the
following independently-confirmed compiler hazard rather than a genuine
hardware limit:

### The candidate root cause (rocKE's finding, `ROCKE_GFX1250_PORTING_NOTES.md:41-116`)

The AMDGPU LLVM backend, when it sees a plain sequential `load <8 x half>`
(or bf16) from LDS (`addrspace(3)`) feeding a WMMA operand, **silently
substitutes** the `ds_load_tr16_b128` hardware transpose-read instruction —
which assumes the LDS tile was stored **column-major**. If the kernel
actually stored the tile **row-major** (common for coalesced global→LDS
writes), the substitution silently corrupts the WMMA input. rocKE measured
this producing wrong results by a factor of ~136x in isolated tests, and
fixed it by marking every ordinary (non-transpose-intended) 8-vector LDS
load destined for WMMA as LLVM `volatile` — `volatile` is opaque to the
backend's pattern-matching substitution pass, forcing the plain
`ds_read_b128` instead. rocKE documents this as **zero-cost** (LDS is
sequentially consistent within a wave, so `volatile` adds no real
synchronization overhead here).

rocKE's own explicit recommendation for CK (`ROCKE_GFX1250_PORTING_NOTES.md:108-116`):
> audit every gfx1250-disabled WMMA GEMM/MX instance config ("tile too
> small", "illegal memory access", "numerical issues") for whether it is
> *actually* hitting this exact backend auto-substitution hazard on a
> row-major LDS tile, rather than a genuine hardware/tile-size limitation.
> If so, the fix is either (a) mark the offending LDS loads `volatile` at
> the point CK emits them, or (b) verify CK's inline-asm/intrinsic path
> already forces the non-transpose opcode. **This is the single most
> actionable, concrete lead in this investigation.**

**Caveat to verify first**: this hazard is stated as WMMA-specific (it's
about an instruction, `ds_load_tr16_b128`, that only exists to feed WMMA
ops). `device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp` is nominally an
"XDL" (MFMA) device op. Before assuming this hazard applies: confirm
whether, on gfx1250, this "Xdl"-named C++ template actually lowers to WMMA
builtins under the hood (gfx1250 has no real XDL/MFMA hardware — it is a
WMMA/wave32-only part per `HIPCONV_GFX1250_CONV_LEARNINGS.md`'s "CDNA5"
classification discussion) or genuinely emits MFMA-equivalent codegen via
some other compatibility path. If it does lower to WMMA, this hazard is a
strong candidate; if not, this specific guarded code path needs a different
root-cause investigation (dump the generated ISA for the `LargeTensors=true`
instantiation and grep for `ds_load_tr16` either way — this is the first
concrete step regardless).

### Concrete implementation plan

1. Build a minimal repro: temporarily remove the `if constexpr(LargeTensors)`
   early-return, force-instantiate a `LargeTensors=true` config, and run it
   with `verify=1` at a shape large enough to trigger `LargeTensors` (check
   the sizing threshold that selects this template parameter — search
   call sites instantiating this device op with `LargeTensors=true` in
   `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_weight/nhwgc/xdl/nhwgc_gkyxc_nhwgk/device_grouped_conv2d_bwd_weight_two_stage_xdl_nhwgc_gkyxc_nhwgk_bf16_large_tensors_instance.cpp`
   and its `_default_large_tensors_instance.cpp` sibling for the exact
   tile/size parameters already registered for this path).
2. Dump the compiled kernel's disassembly (`llvm-objdump -d` /
   `roc-obj-ls` + `roc-obj-extract`, or `hipcc -S -mllvm --amdgpu-verify-ir`
   style, whichever this toolchain supports) and confirm/deny the presence
   of `ds_load_tr16_b128` at LDS→WMMA (or LDS→MFMA) load sites.
3. If confirmed: apply rocKE's fix — mark the relevant LDS load(s) as
   `volatile` at CK's emission point. In CK this likely means an inline-asm
   or explicit-load-instruction change in the gridwise-gemm's
   thread/blockwise LDS-read path used by this device op (find where the
   A/B operand blockwise-copy reads from LDS immediately before the MMA
   call — likely in a shared `blockwise_gemm_*` or `threadwise_gemm_*`
   header this device op includes). Confirm the fix by re-running the same
   disassembly check (substitution gone) and `verify=1` at scale
   (specifically: run it **multiple times** since "inconsistent across
   runs" was the original symptom — a single passing run is not sufficient
   evidence).
4. If NOT confirmed (no `ds_load_tr16_b128` present, or the hazard doesn't
   apply to this XDL path): document the negative result precisely (what
   was checked, what the disassembly showed) so the next person doesn't
   repeat the investigation, and this remains open as a genuine hardware/
   design-limitation question rather than a known compiler bug.
5. Once fixed, remove the `if constexpr(LargeTensors) { if(is_gfx125_supported())
   return false; }` guard (or narrow it if the fix only covers some
   sub-case), and consider applying the identical fix to the bwd_data and
   fwd siblings referenced above, since they share the exact same disabled
   pattern.

### Validation recipe

```bash
# Force-enable LargeTensors path temporarily, rebuild, verify=1, run MANY times
# (the original symptom was non-deterministic across runs):
for i in $(seq 1 20); do
  ./bin/ckProfiler grouped_conv_bwd_weight 5 2 1 1 0 1 2 <large-tensor-triggering shape> -1
done | grep -c "Error\|Incorrect"
# Expect 0 across all 20 runs before considering this fixed.
```

---

## Task 6 — DEVICE vs SYSTEM atomic scope for wrw split-K — correctness stress test

**Priority**: high (correctness, not perf — per repo convention this comes
before further perf work); low implementation effort (a test, not a code
change, unless it fails).
**Risk**: low to test; if it fails, the fix (widen atomic scope) is small
but touches a hot, shared path (`raw_buffer_atomic_add`).

### Problem

`include/ck/utility/amd_buffer_addressing.hpp:600-605` (current lines) —
CK already gates gfx1250 atomics to **DEVICE** (16) scope, per a comment
crediting this as gfx1250's specific cross-CU-atomics requirement:
```cpp
#if defined(__gfx125__)
    // gfx1250 requires DEVICE scope for cross-CU buffer atomics; CU scope is sufficient elsewhere.
    constexpr int coherence_flag = static_cast<int>(AmdBufferCoherenceEnum::DEVICE);
#else
    constexpr int coherence_flag = static_cast<int>(AmdBufferCoherenceEnum::DefaultCoherence);
#endif
```
(scope encodings: `CU=0, SE=8, DEVICE=16, SYSTEM=24`, per
`include/ck/utility/amd_buffer_coherence.hpp:24-27`). This is exactly the
code path wrw's split-K epilogue depends on:
`device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp` dispatches to
`InMemoryDataOperationEnum::AtomicAdd` whenever `KBatch > 1` (confirmed:
`gemm_arg.KBatch > 1` branch around line 925 as of this writing).

**MISA independently found a different atomic instruction class
(`global_atomic_add_f32`, a flat/global atomic, not CK's buffer-resource-
based `raw_buffer_atomic_add`) needs the wider SYSTEM (24) scope on the same
hardware** ("bare atomics silently drop cross-CU updates on this HW",
MISA's `gfx1250_streamk_design.md`, found/fixed in MISA's Phase 17;
cross-referenced in `MISA_GFX1250_CONV_LEARNINGS.md:83-123`). This is
third-party, independently-derived confirmation that gfx1250 cross-CU
atomics need wider-than-CU scope in general — the open question is whether
buffer atomics (CK's instruction class) genuinely only need DEVICE, or
whether they too need SYSTEM and CK's current DEVICE choice is silently
dropping updates at scale (buffer atomics route through the buffer
descriptor's cache-coherence path, global atomics through L2 directly — they
may have different requirements, or CK's choice may simply be
under-tested).

### Concrete implementation plan (this is fundamentally a test, not a code change, unless it fails)

1. Construct a wrw shape that is maximally atomic-heavy: small M/N (few
   output weight elements, so few distinct atomic-add destinations, so high
   *contention* per destination), large K (deep reduction, so many
   partial-sum atomic adds accumulate into each destination), high split-K
   (many concurrent writers per destination), run across as many CUs as
   possible. Good candidate: reuse Task 1's narrow-C example shape
   (`G=1,N=42,K=24,C=3,...`) at its natural high auto-picked split-K
   (`SplitK≈2048`), or force an even higher explicit `SplitK` value via the
   profiler's positional split-K arg.
2. Run with `verify=1` **many times** (atomic races are non-deterministic;
   a single pass proves little) — e.g. 20-50 repeated invocations — and
   check for *any* verification failure. Absence of failure across many
   runs at DEVICE scope is reassuring but not conclusive (races can be rare).
3. For a stronger test: temporarily force `coherence_flag` to `SYSTEM` (24)
   at `amd_buffer_addressing.hpp:600-605` behind a quick local `#define` or
   build flag, rebuild, and run the **same** repeated-verification sweep.
   If SYSTEM scope shows zero difference in pass rate vs DEVICE across many
   runs (both already 100% for the whole `verify=1` sweep), that's decent
   evidence DEVICE is sufficient for CK's buffer-atomic instruction class
   specifically, and MISA's SYSTEM requirement is instruction-class-specific
   as suspected, not universal. If DEVICE shows *any* verification failures
   that disappear at SYSTEM scope, that's a real, confirmed correctness bug
   in CK today — widen the scope for this instruction class and this
   becomes a real, well-evidenced follow-on fix commit.
4. Given atomics are notoriously hard to trigger deterministically, also
   consider sweeping CU-count-adjacent variables if this environment
   exposes any (e.g. via `HSA_VISIBLE_DEVICES` restricting the number of
   partitions/XCDs if this is a multi-die part, or simply varying shape
   sizes to change how many CUs the grid actually spans) to maximize the
   chance of surfacing a real race if one exists.

### Validation recipe

```bash
for i in $(seq 1 30); do
  ./bin/ckProfiler grouped_conv_bwd_weight 5 2 1 1 0 1 2 1 42 24 3 3 3 480 640 2 2 1 1 1 1 1 1 2048
done 2>&1 | grep -c "Error\|Incorrect"
# Expect 0. Repeat with coherence_flag forced to SYSTEM and compare.
```

---

## Task 7 (lower priority, speculative) — TwoStage's extra global read/write pass

**Priority**: low (architectural, not a bug; this session's profiling
suggests limited upside, but wasn't checked at the specific split-K regime
where it would matter most).

### Context

`TwoStage` exists specifically because directly atomic-accumulating into
bf16/fp16 output isn't numerically safe at scale, so it always pays an
extra full pass: stage 1 writes SplitK-many fp32 partial sums to a
workspace buffer, stage 2 (`kernel_batched_elementwise`) reads them all back
and reduces+casts to the final dtype. This session's PMC profiling
(rocprofv3) on the shapes actually measured found stage 2 costs only
~1-2% of total kernel time (e.g. `4.5μs` of a `235μs` total for one profiled
shape) — i.e. **not** the bottleneck for the shapes checked, which used
moderate split-K.

**Open question, not yet checked**: at very high split-K (the regime Task 1
identifies as suboptimal, e.g. `SplitK=2048+`), the workspace stage 1
writes and stage 2 reads scale **linearly with split-K** (more partial
sums = more workspace I/O), while the useful compute per split shrinks. It
is plausible that at very high split-K, stage 2's relative cost is much
higher than the 1-2% measured at moderate split-K, and could compound with
Task 1's finding (i.e., fixing Task 1's over-splitting could recover more
than the compute-side numbers alone suggest, because it would also shrink
stage 2's I/O volume proportionally). Re-profile stage 2's relative cost
specifically at `SplitK≈2048-4096` (the range Task 1 identifies as
over-split) before concluding this is not worth optimizing further —
if it turns out to be a meaningful fraction of total time at high split-K,
Task 1's fix indirectly addresses this too and no separate work is needed;
if stage 2 remains a small fraction even at high split-K, this item can be
closed as "not worth pursuing" with that evidence recorded.
