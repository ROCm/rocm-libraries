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
for `grouped_conv2d_fwd` (commit `932dce7b168`, ~97x speedup at the target
shape). **Task 4 (`grouped_conv2d_bwd_weight`): resolved, but not via a
literal port** — bwd_weight's GEMM-K axis carries no channel content
(see Task 4's Resolution), so T2-01's K-axis packing mechanism cannot
apply; instead extended and validated CK's existing, previously-dormant
`NumGroupsToMerge` block-diagonal M/N packing for the depthwise shape
family T2-01's fwd precedent targets, 1.2x-2.2x measured on gfx1250.

**Task 6 (atomic scope stress test)**: run per its own recipe; the
DEVICE-vs-SYSTEM hypothesis it set out to test was ruled out, but the test
surfaced a real, reproducible, previously-undocumented correctness bug in
`DeviceGroupedConvBwdWeight_Wmma_CShuffleV3`'s split-K atomic-add path under
high per-destination contention — see **Task 8** (new).

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

**Status: RESOLVED** (commit `980e8001ae0`, see "Resolution" below). Landed
approach 2 from the implementation plan below: derive the cap from this
kernel's own measured `max_occupancy_` rather than a per-dimension
padding-waste gate. `k_batch_ = min(k_batch_, max_occupancy_per_cu^2)`,
applied unconditionally right after the existing occupancy-driven pick -
the `min()` self-gates (a no-op whenever `grid_size >= num_cu /
max_occupancy_per_cu`, which is exactly the regime where the unmodified
formula is already correct). No per-dimension (`gemmM`/`gemmN` vs.
`MPerBlock`/`NPerBlock`) condition needed at all - see "Resolution" for why
that axis doesn't actually distinguish the two named example shapes.

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

### Resolution

Landed in `device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp`'s
`Argument` constructor, immediately after the existing
`k_batch_ = std::min(k_batch_, k_batch_max)` gemmK-derived cap:

```cpp
const ck::index_t occupancy_k_batch_ceiling =
    active_workgroups_per_cu.max_occupancy_ * active_workgroups_per_cu.max_occupancy_;
k_batch_ = clamp_gemm_k_batch(std::min(k_batch_, occupancy_k_batch_ceiling));
```

**Why the per-dimension gate (plan step 1) doesn't work, confirmed on the
doc's own two named shapes**: neither `gemmM` nor `gemmN` is narrower than
its `MPerBlock`/`NPerBlock=16` tile for the *first* example shape
(`M18`/G1,N42,K24,C3,Y=X=3,480x640,s2,p1: `gemmM=24`, `gemmN=27`, both
>16) - a "narrower than one tile" gate never fires for it at all, leaving
its real ~15-19% loss completely unaddressed. Conversely the *second*
example shape (`M00`/G1,N42,K10,C128,Y=X=1,120x160: `gemmM=10<16`) *does*
trip such a gate, and an earlier attempt at exactly this gate (tightening
to `2 * max_occupancy_per_cu` when triggered) cut `M00`'s k_batch from the
already-correct auto pick of 1024 down to 64 - a **~3.2x slowdown**, far
worse than the `MinIters` attempt's already-reverted -26.6%. Per-dimension
tile narrowness is not the axis that separates these two shapes.

**What actually separates them is `grid_size`** (4 for `M18`, 8 for
`M00`) relative to `num_cu / max_occupancy_per_cu` (`256 / 32 = 8` for
this tile/GPU): both shapes' true optimum sits at essentially the same
*absolute* `k_batch` (~1024 for both, empirically), which is exactly
`max_occupancy_per_cu^2` - the value the unmodified occupancy formula
itself would produce right at `grid_size == num_cu /
max_occupancy_per_cu`. Capping there via `min()` therefore self-gates: it
is a no-op for any `grid_size` at or above that threshold (where the
unmodified formula already picks `<= max_occupancy_per_cu^2`), and only
ever pulls `k_batch_` down when `grid_size` is smaller.

**Validated** (gfx1250, bf16, the exact - and currently only - registered
`<32,16,16,32,...>` `TwoStage` tile, via a standalone probe instantiating
the device op directly against `ReferenceConvBwdWeight`):

| Shape (grid_size) | k_batch: before → after | avg_time: before → after | Δ |
|---|---|---|---|
| `M00` (8) | 1024 → 1024 (no-op) | 0.177 → 0.168 ms | ~+5% (noise, no regression) |
| `M18` (4) | 2048 → 1024 | 0.411 → 0.330 ms | **-19.6%** |
| synthetic grid=2 (K16,C3,Y=X=3, else = M18) | 4096 → 1024 | 0.562 → 0.286 ms | **-49%** |
| synthetic grid=6 (K24,C40,Y=X=1, else = M00) | 1365 → 1024 | 0.222 → 0.196 ms | **-12%** |
| synthetic grid=16 (K10,C256,Y=X=1, else = M00) | 512 → 512 (no-op) | 0.285 → 0.285 ms | 0% |

All 29 `miopen_wrw_shapes.txt` shapes re-verified (`verify=1`) directly
against the `TwoStage` device op post-fix: **29/29 pass**. The 5 shapes
where `TwoStage` is ckProfiler's selected-best instance in
`miopen_baseline_results.csv` (`M00,M06,M14,M15,M19,M25`, excluding
`M18` which is also `TwoStage`-best) show no regression (all within
±5%, consistent with this machine's documented run-to-run noise); `M19`
specifically confirmed via `CK_LOGGING=1` to pick an unchanged `k_batch`
(`grid_size=24 > 8`, cap correctly a no-op).

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

### Resolution

**T2-01's literal mechanism does not port to `bwd_weight` — this is a
mathematical fact, not an engineering gap.** fwd's GEMM-K axis is
`C_per_group*Y*X` (channel-bearing, hence the small-channel K-padding
waste T2-01 closes by packing groups block-diagonally into K).
`bwd_weight`'s GEMM-K axis is `N*Ho*Wo` (batch/spatial) —
`get_bwd_weight_gemm_sizes()` (`split_k_utils.hpp:65-92`) and
`TransformConvBwdWeightToGemmV2::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N`
confirm `GemmM = K_per_group`, `GemmN = C_per_group*Y*X`, `GemmK =
N*Ho*Wo` — channels appear only in M and N, **never** in K, because
weight-gradient reduction is over batch and space, not channels. There is
no small-channel K content to pack block-diagonally for this GEMM, at any
implementation choice; a literal port (new K-axis prepass kernel, per the
design doc's §3) would be solving a problem this GEMM shape does not have.

**The structurally-analogous inefficiency, and its existing fix**: since
both channel axes live in M/N which share one group-independent K, the
real padding waste for small-channels-per-group `bwd_weight` is
plain MPerBlock/NPerBlock tile underutilization — and CK already has a
proven, descriptor-only block-diagonal packing mechanism for exactly that:
`NumGroupsToMerge` (xor+pad trick,
`TransformConvBwdWeightToGemmV2::make_wei_grid_desc`), already wired into
`DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3`'s template parameter
list. It was, however, gated at `IsSupportedArgument()`
(`device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp:1355`) to
`Conv_C_==1 && Conv_K_==1` (pure depthwise) and **never exercised with
`NumGroupsToMerge > 1` by any registered instance** — every instance in
both the active and commented-out tables in
`device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp` uses
`NumGroupsToMerge=1`. This dormant capability is exactly what the
roadmap's cited depthwise shapes (M06/M14/M15/M19/M25 — all
`C_per_group=K_per_group=1`) need, so the actual "port" of T2-01's intent
(dense packing instead of instance tuning around a fixed waste factor) is
to exercise and register it, not to invent a new K-axis prepass.

**What was done**: added
`example/71_grouped_conv_bwd_weight_wmma_group_merge/` (mirrors fwd's own
standalone-example validation pattern), exercising
`DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3` with
`NumGroupsToMerge=16` for the 3x3 depthwise shape family
(`C_per_group=K_per_group=1`): `GemmM = 16*K_per_group = 16` exactly fills
`MPerBlock=16` (was `K_per_group=1` padded 16x), `GemmN =
16*C_per_group*Y*X = 144` exactly fills `NPerBlock=144 =
NRepeat(9)*NPerWmma(16)` (was `C_per_group*Y*X=9` padded ~1.8x) — all
other tile parameters are the existing `NumGroupsToMerge=1` instance's,
unchanged (the epilogue loops `NRepeat` times over the same 16x16 pass
shape, so only B's N-extent needed retuning). Verified correct against
`ReferenceConvBwdWeight` (CPU reference, bf16) at G=192/256/512. Registered
the validated config as a second bf16 instance in
`device_grouped_conv_bwd_weight_two_stage_wmma_instance.hpp` (existing
`NumGroupsToMerge=1` instance left completely unchanged, zero risk to it).

**Validated on gfx1250 hardware** (`ckProfiler ... verify=1`, standard
auto split_k `-1`, no manual tuning — real end-user path):

| Shape | Before (TFlops) | After (TFlops) | Speedup |
|---|---|---|---|
| M06 (g=192, 120x160, s2) | 0.335 | 0.591 | 1.76x |
| M14 (g=256, 60x80, s1) | 0.400 | 0.836 | 2.09x |
| M15 (g=256, 60x80, s2) | 0.413 | 0.493 | 1.19x |
| M25 (g=512, 30x40, s1) | 0.361 | 0.783 | 2.17x |

Manual split_k tuning (via the new example) finds still more headroom
(e.g. M06 up to 1.08 TFlops at split_k=256, ~3.2x) — left as future
auto-heuristic work (same class of gap as Task 1, not re-solved here).

**Full 29-shape regression sweep** (`miopen_wrw_shapes.txt`, `verify=1`,
`-1` auto split_k, avoiding `all`-splitK mode — see hazard note below):
27/29 shapes within noise of baseline or better; the 4 `NumGroupsToMerge=1`-
eligible depthwise shapes this instance targets all improved as above with
zero incorrect-result reports across the whole corpus. Two shapes
(`M00`, `M01`) measured 7-9% below `miopen_baseline_results.csv`'s
recorded numbers; both select instances this change never touches
(`DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<32,16,16,32,...,1>`
and `DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<512,128,256,256,...>`
respectively) and reproduced consistently at the lower figure across 3
repeats each post-reboot — i.e. environment drift between the original
baseline capture and now (this session hit a real GPU hang mid-validation,
requiring a machine reboot; post-reboot clock/thermal state differs), not
a regression from this change.

**Hazard note for future work on this file**: `ckProfiler ... all` (full
splitK sweep, all instances) reproducibly crashes with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on the M06 shape on this
machine, **independent of this change** (reproduces against the
unmodified pre-existing instance set) — consistent with Task 2's
documented hardware-crash hazard for this instance family. Always use
explicit or `-1` (auto) split_k values against this op, never `all`.

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

### Resolution

**Status: tested, hypothesis ruled out — but the stress test surfaced a
different, real, previously-undocumented correctness bug (see Task 8).**

Step 1's suggested repro shape (`G=1,N=42,K=24,C=3,...`, Task 1's narrow-C
example) turned out **not** to exercise the atomic path at all: ckProfiler's
automatic instance selection picks
`DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3` for this shape, which
**never uses atomics** for split-K (it writes each split's partial sum to a
distinct workspace slice and reduces with an ordinary read in stage 2 -
that's the entire reason `TwoStage` exists, per Task 7's context). 30/30
`verify=1` runs at `SplitK=2048` passed, but this data point says nothing
about atomic-scope correctness.

To actually test the atomic path, built a standalone example
(`example/72_grouped_conv_bwd_weight_wmma_atomic_stress/`) directly
instantiating `DeviceGroupedConvBwdWeight_Wmma_CShuffleV3` (confirmed via
`grep AtomicAdd` to dispatch `InMemoryDataOperationEnum::AtomicAdd` whenever
`KBatch > 1`) at a deliberately adversarial shape: `G=1, K_per_group =
C_per_group = 32, Y=X=1` — `GemmM = GemmN = 32`, exactly one `MPerBlock x
NPerBlock` tile (the smallest registered bf16 instance), so every split-K
workgroup atomic-adds into the *same* destination range.

**Found a real, reproducible, split_k-dependent correctness bug** (10 reps
per `split_k`, same shape, DEVICE scope - CK's current default):

| split_k | pass/10 | | split_k | pass/10 |
|---|---|---|---|---|
| 2 | 20/20 (extra reps) | | 10 | 4/10 |
| 4 | 20/20 (extra reps) | | 12 | 0/10 |
| 6 | 9/10 | | 16 | 0/10 |
| 8 | 6/10 | | 32-1024 | 0/10 |

Below `split_k≈6` always passes; `6-10` is genuinely non-deterministic
(the classic race signature); `≥12` fails every time. Error magnitude scales
with `split_k` (a few `~1%`-magnitude wrong elements at `split_k=16`, up to
53% of all elements wrong by up to `~30%` at `split_k=1024`) - consistent
with atomic adds being dropped under contention, not a one-off logic bug.

**Tested this doc's exact hypothesis (SYSTEM vs DEVICE scope) directly**:
forced `coherence_flag` to `SYSTEM` (24) at `amd_buffer_addressing.hpp:602`,
rebuilt, re-ran the same `split_k` sweep. **No improvement** - `split_k=16`
through `1024` still fail 10/10 at SYSTEM scope, and `split_k=8` was
similarly non-deterministic (9/10 pass) rather than fixed. **This rules out
DEVICE-vs-SYSTEM buffer-atomic coherence scope as the root cause** of what
was found; MISA's SYSTEM-scope requirement for `global_atomic_add_f32`
(a different instruction class) does not appear to transfer to CK's
`raw_buffer_atomic_add` on this hardware, at least not as the fix for this
bug. Reverted the SYSTEM-scope change in full (confirmed clean `git diff`).

**Also checked**: is this specific to the single-tile (`GemmM=GemmN=
MPerBlock=NPerBlock`) edge case? No - reproduces identically with
`K_per_group=64` (2 M-tiles) and `128` (4 M-tiles), all still 0/3 pass at
`split_k=16`. **Also checked**: does a "normal" wider shape (M01-equivalent,
`C=K=128`, `GemmM=128, GemmN=1152`) show any failures at high split_k? No -
`ckProfiler ... verify=1` at `split_k=16/32/64` for that shape reports
`valids: 20` (all candidate instances, including the atomic-based XDL/WMMA
one, pass) every time. **The bug is specific to high per-destination atomic
contention (few, or fully saturated, M/N destinations)**, not split-K or
atomics in general - i.e. it's a real latent risk specifically for
narrow-channel/depthwise shapes (the exact regime Task 3/4 target) if a
caller or future auto-heuristic ever pushes `split_k` past roughly 10-12 for
such a shape on the one-stage atomic device op.

This is a new, more concrete, better-evidenced correctness lead than the
original DEVICE-vs-SYSTEM hypothesis. See **Task 8** for the writeup and
next steps (full root-cause requires disassembly-level investigation,
scoped out of this task per its own "this is fundamentally a test" framing).

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

---

## Task 8 (new, high priority) — Real split-K atomic-add correctness bug under high per-destination contention, `DeviceGroupedConvBwdWeight_Wmma_CShuffleV3`

**Priority**: high (confirmed, reproducible correctness bug in shipping
code, not a perf item; discovered while executing Task 6's stress test).
**Risk to fix**: unknown until root-caused - could be anywhere from a small
targeted fix (if it's the rocKE `ds_load_tr16_b128` LDS-load hazard Task 5
describes, manifesting here instead of via `LargeTensors`) to a deeper
split-K/atomic design issue. **Root-causing this is a genuinely large task
(disassembly-level investigation) - not attempted here; this section is the
bounded, well-evidenced repro + diagnosis handoff**, consistent with Task
5's own "document the negative/positive result precisely" convention.

### Problem

`DeviceGroupedConvBwdWeight_Wmma_CShuffleV3` (one-stage WMMA bwd-weight,
`device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp`), when split-K forces
its `InMemoryDataOperationEnum::AtomicAdd` epilogue (`KBatch > 1`), produces
**wrong results** once per-destination atomic contention gets high enough -
concretely, once the number of concurrent split-K workgroups per M/N tile
(`split_k` for a shape whose `GemmM<=MPerBlock` and `GemmN<=NPerBlock`, i.e.
everything lands in one tile) exceeds roughly 10-12. Below that it's
correct; `6-10` is flakily wrong (non-deterministic - the classic race
signature); at and above `~12` it is wrong on every single run, with error
magnitude scaling with `split_k` (up to 53% of all elements wrong, up to
~30% relative error, at `split_k=1024`). Full evidence, exact repro
commands, and what was ruled out (SYSTEM vs DEVICE atomic scope; the
single-tile-grid edge case) are in Task 6's Resolution above - this task
exists to make the finding trackable independent of Task 6's original
(now-superseded) hypothesis.

**Reproducer**: `example/72_grouped_conv_bwd_weight_wmma_atomic_stress/`
(new, added alongside this finding). Usage:
`./bin/example_grouped_conv_bwd_weight_wmma_atomic_stress_bf16
<split_k> [K_per_group]` (defaults: `split_k=1024`, `K_per_group=32`).
Prints `PASS`/`FAIL split_k=N` and returns a matching exit code; loop it to
see the non-deterministic band.

**Blast radius, not yet checked**: only `DeviceGroupedConvBwdWeight_Wmma_
CShuffleV3` (plain WMMA one-stage) was tested. `device_grouped_conv_bwd_
weight_xdl_cshuffle_v3.hpp` (confirmed, via its own `GetTypeString()`
appending `_WmmaPorted` whenever `get_warp_size() != 64`, to *also* lower to
WMMA on gfx1250 - directly validating Task 5's caveat that this nominally-
"Xdl" template is WMMA-backed here) uses the identical `AtomicAdd`-on-
`KBatch>1` dispatch pattern and was **not** observed to fail in a quick check
at `split_k` up to 64 - but that check used a "normal" wide shape (`GemmM=
128, GemmN=1152`), not a saturated single-tile shape; it has not been tested
at matching per-destination contention (`GemmM<=MPerBlock, GemmN<=NPerBlock`)
the way the one-stage WMMA op was. Do that check before assuming XDL/
WmmaPorted is unaffected - if it is affected too, this is very likely the
same underlying hazard as Task 5 (both device ops sharing gfx1250 WMMA
codegen for their LDS→MMA feed), which would make Task 5's disassembly
investigation directly applicable here as well and vice versa.

### Concrete implementation plan

1. Confirm/deny the same failure on `device_grouped_conv_bwd_weight_xdl_
   cshuffle_v3.hpp` at matching contention (small `GemmM`/`GemmN`, `split_k`
   swept 4-64) - determines whether this is one device op's bug or a shared
   hazard.
2. Follow Task 5's disassembly playbook: dump the compiled kernel for the
   failing `AtomicAdd`-dispatched instantiation and check whether the
   atomic-add source/destination registers are fed by a `ds_load_tr16_b128`
   auto-substitution (same rocKE hazard) rather than investigating atomics
   in isolation - a corrupted *operand* to the atomic add would produce
   exactly this "wrong sum, worse under more concurrent workgroups" pattern
   just as plausibly as a genuine dropped-atomic race would.
3. If the LDS-load hazard is confirmed: apply rocKE's `volatile`-marking fix
   at the relevant load site(s) in the shared blockwise/threadwise LDS-read
   path this device op includes; re-run the exact repro sweep above
   (`split_k` 2 through 1024, 10+ reps each) and confirm the pass rate goes
   to 100% at every level, not just the previously-failing ones.
4. If not confirmed: this remains a genuine atomic/split-K correctness bug
   independent of Task 5's hazard - escalate as its own investigation
   (possibly CU-vs-DEVICE-vs-SYSTEM was the wrong axis entirely; consider
   whether the epilogue's read-modify-write ordering, or the zero-
   initialization of the destination buffer before the first atomic add,
   has a visibility gap under high contention).

### Validation recipe

```bash
# Reproduce (expect a mix of PASS/FAIL around split_k=6-10, all-FAIL at >=12):
for sk in 2 4 6 8 10 12 16 32 1024; do
  echo "split_k=$sk:"
  for i in $(seq 1 10); do ./bin/example_grouped_conv_bwd_weight_wmma_atomic_stress_bf16 $sk; done | sort | uniq -c
done
```

---

## Task 9 — Port ck_tile's 8-wave async-load GEMM pipeline to convolution (fwd first, then bwd-weight), pairing `global_load_async` for A with TDM for B

**Priority**: medium-high (large potential upside per the ck_tile GEMM
precedent this is modeled on; scoped as new pipeline engineering, not a
tuning knob).
**Risk**: medium — new pipeline code, but follows an existing, working
ck_tile GEMM pattern closely rather than inventing from scratch.

### Problem / opportunity

ck_tile already has a compute-optimized **8-wave** GEMM pipeline,
`GemmPipelineAgBgCrCompAsyncEightWaves`
(`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp`,
policy at `gemm_pipeline_ag_bg_cr_comp_async_eight_waves_policy.hpp`, shared
base at `gemm_pipeline_ag_bg_cr_eight_waves_base.hpp`). Its defining feature
(per its own doc comment) is *"asynchronous load from global memory to LDS,
skipping the intermediate loading into pipeline registers"* — i.e. it uses
`async_load_tile`/`async_load_tile_raw`
(`include/ck_tile/core/tensor/load_tile.hpp:165-215`, which lower to a
direct global→LDS copy instruction, fenced with `s_waitcnt vmcnt`) for
**both** A and B today
(`MakeAsyncLoadADramWindow`/`MakeAsyncLoadBDramWindow`, referenced at
`gemm_pipeline_ag_bg_cr_eight_waves_base.hpp:125,159`). This pipeline is
MFMA-based (`BlockGemm`/`WarpGemm` selection via
`Policy::GetBlockGemm<Problem, IsScaledGemm>()`) and its 8-wave occupancy
profile (large `BlockGemmShape::BlockWarps` product) is the kind of
configuration that benefits most from MI355/gfx950's larger per-CU register
file and LDS capacity — this needs to be re-confirmed for gfx1250
specifically (gfx1250 is WMMA/wave32, not MFMA/wave64 — see the caveat
below) rather than assumed to transfer directly.

ck_tile also already has **separate** TDM-based GEMM pipelines
(`gemm_pipeline_ag_bg_cr_comp_tdm_v1.hpp`, `_v2.hpp`,
`gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp`, and
`wp_pipeline_agmem_bgmem_creg_tdm.hpp` + its policy), demonstrating the TDM
load API (`load_tile_tdm` /
`tile_window.tdm_load_to_lds(tdm_config, lds_tile, ...)`,
`include/ck_tile/core/tensor/load_tile.hpp:179-198`) as a working,
in-tree alternative bulk-transfer mechanism to plain async-copy. TDM is
already used for gfx1250 elsewhere in ck_tile (the FMHA TDM pipeline,
per recent commits touching `qr_tdm`/`fp8 quantization scales on gfx1250
FMHA TDM pipeline` — confirms TDM is a real, exercised gfx1250 hardware
feature, not just a gfx950 one).

**The specific idea to scope**: today's 8-wave pipeline uses the *same*
load mechanism (async-copy) for both A and B. The suggestion is a **hybrid**
pipeline — `global_load_async` (direct-to-LDS async copy) for the A operand,
TDM for the B operand — applied to **convolution** (fwd first, per the
request; this doc's own scope is wrd/bwd-weight, so bwd-weight is the
natural second target once fwd is proven, mirroring how Task 4 ported
block-diagonal packing from a proven fwd implementation). No such hybrid
pipeline currently exists in ck_tile for GEMM or for convolution — this is
new engineering informed by two existing, separately-proven building
blocks, not a port of a single existing pipeline.

### Concrete implementation plan

1. **Establish the gfx1250 applicability of "8 waves" first**, independent
   of the async/TDM question: `GemmPipelineAgBgCrCompAsyncEightWaves` is
   MFMA-typed (`BlockGemm`/`WarpGemm` selection implies MFMA warp-tiles).
   gfx1250 is WMMA/wave32 hardware (confirmed repeatedly across this
   investigation series — see Task 5's caveat and
   `HIPCONV_GFX1250_CONV_LEARNINGS.md`'s "CDNA5" classification). Before
   porting anything: determine whether "8 waves" as a *concept* (high wave
   occupancy per block to hide async-load latency behind compute) is
   separable from the MFMA-specific implementation, and what the
   WMMA/wave32 equivalent occupancy target should be (likely not literally
   "8" — wave32 vs wave64 changes the wave-count-to-thread-count and
   register-pressure-per-wave arithmetic; consult
   `GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` for any documented gfx1250
   occupancy/wave-count guidance before picking a number). Do not assume
   "8" is the right constant for gfx1250 without this check.
2. **Build the conv-fwd GEMM-transform pipeline** using ck_tile's tile-API
   conventions (this is `ck_tile`, not old `ck::` — confirm which
   namespace/layer conv-fwd's WMMA path actually lives in for gfx1250
   today; if it's still old `ck::` only, this task additionally requires
   either porting the relevant conv-fwd device op into `ck_tile`, or
   building the hybrid pipeline in old `ck::`'s pipeline/policy idiom
   instead — check which is true before starting, since the two codebases'
   pipeline APIs are not interchangeable).
3. **A-operand loading**: reuse `async_load_tile`/`async_load_tile_raw`
   exactly as `GemmPipelineAgBgCrCompAsyncEightWaves` does today — this half
   is a direct port, not new design.
4. **B-operand loading**: adapt `load_tile_tdm` from
   `gemm_pipeline_ag_bg_cr_comp_tdm_v1.hpp`/`_v2.hpp` (read both — v1/v2
   likely differ in prefetch depth or TDM config shape; pick whichever
   matches this hybrid pipeline's intended prefetch structure) into the new
   pipeline's B-loading path. Pay attention to `TDMConfig_` construction
   (`load_tile_tdm`'s first argument) — this encodes the hardware
   descriptor for the bulk transfer and will need a policy method analogous
   to `GetBlockGemm`/`GetVectorSizeA`/`GetVectorSizeB` to compute correctly
   for whatever B-operand shape (weight tensor, in fwd's A/B/E convention)
   the conv-to-GEMM transform produces.
5. **New pipeline scheduling**: since A and B now use two different load
   mechanisms with likely different latency profiles, the hot-loop
   scheduler (`__builtin_amdgcn_sched_group_barrier`/`sched_barrier` hints,
   see the existing pipeline's `hot_loop_scheduler` lambda,
   `gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp:198-209`, and this
   investigation's own T1-03 finding that CK's `HotLoopScheduler()` is dead
   code on every arch today per `include/ck_tile/core/arch/arch.hpp`'s
   `s_wave_barrier`) will need new, gfx1250-specific tuning — don't assume
   the existing async-eight-waves scheduler hints transfer unchanged to a
   mixed async+TDM load pattern.
6. **New instance/example scaffolding**: mirror how Task 4's fwd
   block-diagonal port added `example/70_grouped_conv2d_fwd_wmma_block_diagonal/`
   as a standalone correctness+perf validation harness before touching any
   production instance file — do the same here (a new numbered example
   under `example/ck_tile/` or `example/`, whichever matches where the
   target conv-fwd WMMA pipeline actually lives per step 2's finding).

### Validation recipe

```bash
# Correctness: new standalone example against CPU reference (ReferenceConvFwd),
# mirroring Task 4's example/70_grouped_conv2d_fwd_wmma_block_diagonal/ harness.

# Perf: before/after against the current best conv-fwd WMMA pipeline at a range
# of shapes (small and large M/N/K - the whole point of "8 waves" is hiding
# load latency behind compute, so it should show most benefit at large K
# where there's plenty of hot-loop iterations to overlap into).

# Once fwd is validated: port to grouped_conv_bwd_weight's own GEMM-transform
# pipeline/device-op (device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp or
# its ck_tile equivalent if one exists), re-run the full 29-shape sweep
# (miopen_wrw_shapes.txt) plus the depthwise shapes Task 4 targets, since
# those are exactly the shapes with the most K-reduction depth to hide load
# latency behind.
```

---

## Task 10 — Extend the old-`ck::` wavelet-model conv pipeline with `global_load_async` direct-to-LDS loads and double LDS buffering

**Priority**: medium (real, well-scoped extension of an existing, already-
wrw-applicable pipeline; bounded to one file family).
**Risk**: medium — touches a working, shipping pipeline
(`device_grouped_conv_bwd_weight_xdl_waveletmodel_cshuffle_v3.hpp` is
registered and presumably in active use); must not regress the existing
`wavelet_default`/`wavelet_pad0`/`wavelet_4w2_default`/`wavelet_4w2_pad0`
instances.

### Problem

`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_waveletmodel_cshuffle_conv_v3.hpp`
implements a wave-specialized ("wavelet model") gridwise GEMM already used
by **both** `device_grouped_conv_bwd_weight_xdl_waveletmodel_cshuffle_v3.hpp`
(wrw — directly relevant to this doc) and
`device_grouped_conv_fwd_multiple_abd_xdl_waveletmodel_cshuffle_v3.hpp`
(fwd). Its own doc comment (lines 25-33) describes the design: dedicated
**load waves** run the conv-to-GEMM descriptor transforms (`RunRead` +
`MoveSrcSliceWindow` + `RunWrite`) and write to LDS, while separate **math
waves** read LDS and do MFMA + CShuffle epilogue — splitting VALU-heavy
descriptor work off of the MFMA-issuing waves specifically to avoid
MFMA/VALU issue-slot conflicts. **Confirmed current limitations**, straight
from the source:
- `DirectLoadEnabled = false;  // DirectLoad is not supported, wavelet model
  requires LDS as the sync boundary` (line 184-185) — loads today are
  ordinary two-phase buffer loads (`RunRead` reads global→registers, then
  `RunWrite` writes registers→LDS;
  `include/ck/tensor_operation/gpu/grid/gridwise_gemm_waveletmodel.hpp:59-98`),
  not a direct-to-LDS async copy.
- Only one LDS tile per operand is referenced throughout
  `gridwise_gemm_waveletmodel.hpp`'s `RunLoadWavePipeline` (`a_block_buf`,
  `b_block_buf`, singular, no `[0]`/`[1]` alternation) — confirmed **single**
  LDS buffer, and the load-wave struct is explicitly named for **1-stage
  prefetch** (`template <typename TileLoadThreadGroup> struct
  GridwiseGemmLoadWave<TileLoadThreadGroup, 1>`) — i.e. no double-buffered
  overlap between "write this iteration's LDS tile" and "read last
  iteration's LDS tile" exists today; math waves must wait for load waves to
  finish writing before consuming, once per iteration, serializing what
  double-buffering would otherwise overlap.

### Concrete implementation plan

1. **Direct-to-LDS async load for the load-wave's global read.** Replace
   the load wave's `RunRead`-then-`RunWrite` two-phase copy
   (`gridwise_gemm_waveletmodel.hpp:59-98`) with a direct global→LDS async
   copy analogous to ck_tile's `async_load_tile`/`async_load_tile_raw`
   (Task 9 above uses the same primitive family, `ck_tile`-side — for this
   old-`ck::` pipeline, find or add the equivalent low-level intrinsic
   wrapper; check `include/ck/tensor_operation/gpu/thread/` and
   `include/ck/tensor_operation/gpu/block/` for any existing `ck::`-side
   direct-to-LDS copy primitive before writing a new one — CK's `XDL`
   direct-load device ops (e.g. the `_direct_load_instance.cpp` files
   already registered for bwd_weight, `xdl/nhwgc_gkyxc_nhwgk/
   device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_{bf16,f16}_direct_load.cpp`)
   likely already wrap whatever primitive this needs — reuse it rather than
   reinventing). This changes the wave-specialization boundary's *cost*
   (load waves no longer hold data in registers before writing LDS) but not
   its *structure* (load waves vs math waves stays the same).
2. **Double LDS buffer.** Extend `GridwiseGemmLoadWave`/`GridwiseGemmMathWave`
   (`gridwise_gemm_waveletmodel.hpp`) to alternate between two LDS tiles per
   operand (the conventional `NumGemmKPrefetchStage=2` ping-pong pattern
   already used elsewhere in CK's non-wavelet pipelines — e.g.
   `BlkGemmPipelinePrefetchStages` in the WMMA `CShuffleV3` pipelines this
   whole investigation has repeatedly referenced). This is the change that
   actually lets load waves work on iteration `i+1` while math waves consume
   iteration `i`'s already-written LDS tile — today's 1-stage design cannot
   do this even with faster (async) loads, since there is nowhere to put the
   next tile until the current one is fully consumed. **This is the
   higher-value half of this task** — the async-load change alone reduces
   the load wave's own latency, but without double buffering, math waves
   still stall waiting for load waves each iteration; only the combination
   unlocks true overlap.
3. **Synchronization**: identify and update whatever mechanism currently
   signals "LDS tile ready" between load and math waves (likely an
   `__syncthreads()`-equivalent or explicit LDS-based semaphore, given the
   two thread groups are disjoint subsets of one block, not separate
   blocks) to handle two independent buffer slots' readiness instead of one
   — this needs its own pair of ready/consumed flags (or a single
   flip-flopping flag scheme) rather than one shared barrier.
4. **Do not touch `DirectLoadEnabled` semantics elsewhere** — this flag
   likely gates other pipelines' behavior too (grep all call sites before
   assuming it's safe to flip for this device op alone); prefer a
   wavelet-model-local mechanism if the existing flag is shared broader
   than this file.

### Validation recipe

```bash
# Correctness: the wavelet model already has registered instances for wrw -
# rebuild and verify=1 across the miopen_wrw_shapes.txt corpus plus the
# existing wavelet-specific instance files' own target shapes:
#   xdl/nhwgc_gkyxc_nhwgk/device_grouped_conv2d_bwd_weight_xdl_nhwgc_gkyxc_nhwgk_{bf16,f16}_wavelet_default_instance.cpp
#   ..._wavelet_pad0_instance.cpp, ..._wavelet_4w2_default_instance.cpp, ..._wavelet_4w2_pad0_instance.cpp

# Perf: interleaved A/B (per this doc's rigor note) old-single-buffer vs new
# double-buffer+async binaries, across a range of K-depths - benefit should
# scale with how many main-loop iterations there are to overlap (shallow-K
# shapes won't show much; deep-K shapes should show the most).

# Also validate the fwd wavelet-model sibling
# (device_grouped_conv_fwd_multiple_abd_xdl_waveletmodel_cshuffle_v3.hpp)
# is unaffected or also benefits, since it shares the same gridwise kernel.
```
