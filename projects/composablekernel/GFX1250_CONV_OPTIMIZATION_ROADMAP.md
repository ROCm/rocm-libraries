# gfx1250 Convolution Optimization Roadmap for Composable Kernel

This document is a **prioritized, implementable task list** derived from the
six investigation documents already in this folder. It is meant to be
handed to an implementing agent **one item at a time**. Each item links to
the specific section(s) of the source documents that justify and detail
it — **do not duplicate that content here**; read the linked section before
implementing.

Source documents (all in `projects/composablekernel/`):
- `GFX950_VS_GFX1250_COVERAGE.md` — gap analysis, CK's own gfx1250 code audit
- `ROCKE_GFX1250_PORTING_NOTES.md` — rocKE project learnings
- `MISA_GFX1250_CONV_LEARNINGS.md` — MISA project learnings (hand-assembly conv generator)
- `HIPCONV_GFX1250_CONV_LEARNINGS.md` — hipConv project learnings (HIP C++ conv library)
- `FLYDSL_GFX1250_CONV_LEARNINGS.md` — FlyDSL project learnings (MLIR GEMM/MoE compiler)
- `GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` — AMD internal GEMM training material, distilled

## How to Use This Document

1. Pick **one** item, starting from Tier 0, then Tier 1 in order. Do not
   batch multiple items into one branch/commit — each item needs its own
   isolated before/after measurement.
2. Read the item's linked source section(s) in full before touching code —
   they contain the file:line evidence, the exact mechanism, and known
   caveats/negative-results this document intentionally omits.
3. Implement the change against the exact CK target file(s) named in the
   item. If the linked source gives a reference implementation from another
   project (rocKE/MISA/hipConv/FlyDSL), treat it as a design reference, not
   code to copy verbatim — CK's types/templates/naming must be followed.
4. Benchmark **before and after** per the protocol in
   [Benchmarking Methodology](#benchmarking-methodology) below, on real
   gfx1250 hardware.
5. Commit on a new branch per [Branch and Commit Workflow](#branch-and-commit-workflow).
6. If the change regresses any shape/precision by more than noise (see
   tolerance below), do not merge it — record the regression in the PR
   description and either narrow the change's applicability (e.g. gate it
   to the shapes that improved) or abandon it.

## Benchmarking Methodology

CK's convolution profiler binary is `ckProfiler`, built via
`BUILD_CK_PROFILER=ON` (default). Usage reference:
`profiler/README.md`, or run any conv op with no args for the full
positional-argument help (`arg1: tensor operation`, `arg2: data type`,
`arg3: tensor layout`, ..., plus `ck::utils::conv::get_conv_param_parser_helper_msg()`
for the trailing `-n/-c/-hi/-wi/-k/-y/-x/...` shape flags). Relevant ops:
`grouped_conv_fwd`, `grouped_conv_bwd_data`, `grouped_conv_bwd_weight` (see
`profiler/src/profile_grouped_conv_{fwd,bwd_data,bwd_weight}.cpp`).

**Build for gfx1250 only** (fast iteration, per `AGENTS.md`
"Development Commands"):
```
../script/cmake-ck-dev.sh --preset=dev-gfx1250   # if no such preset exists yet, configure manually:
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS=gfx1250 -D BUILD_CK_PROFILER=ON ..
make -j$(nproc) ckProfiler
```

**Representative shape set** — every item below must be benchmarked against
at least this set (add shapes named in the specific item):

| Shape class | Example ckProfiler args (fp16, layout 1 = NHWGC/GKYXC/NHWGK) | Why it's in the set |
|---|---|---|
| Large, well-fit fwd | `-n 128 -c 256 -hi 56 -wi 56 -k 256 -y 3 -x 3 -sy 1 -sx 1 -py 1 -px 1` | Compute-bound baseline; should not regress |
| Small-output-channel wrw | `-n 128 -c 64 -hi 56 -wi 56 -k 8 -y 3 -x 3` (direction=wrw, `-F 4`) | Targets the documented wrw small-M gap |
| Small-channels-per-group grouped conv | `-g 32 -n 64 -c 128 -hi 28 -wi 28 -k 128 -y 3 -x 3` (channels-per-group = 4) | Targets block-diagonal-packing item |
| 1×1 / shallow-K | `-n 128 -c 512 -hi 14 -wi 14 -k 512 -y 1 -x 1` | Targets epilogue/store-bound items |
| Large-tensor grouped conv | Shape from `GFX950_VS_GFX1250_COVERAGE.md` §5.5 (LargeTensors-gated) | Targets the disabled large-tensor path, once/if re-enabled |

Run each shape with `-v 1` (verify) at least once after the change to
confirm numerical correctness is unchanged, then with `-v 0 --timekernel 1`
(or the profiler's built-in timing) for the performance measurement.

**Protocol** (per the measurement-noise lesson independently found in
`MISA_GFX1250_CONV_LEARNINGS.md` §4.2 and echoed in
`GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` §6.3 — a shared GPU or one under
other load produces artifact swings up to 20×):
1. Confirm the gfx1250 device is **uncontended** (no other process using it —
   check `rocm-smi`/equivalent).
2. Run each shape **5 times**; record median TFLOP/s (or GB/s for
   bandwidth-bound shapes), not a single sample.
3. Record: kernel/instance name selected, TFLOP/s or GB/s, and whether
   verification passed (`max_rel` if applicable).
4. A change only counts as an improvement if the median moves by more than
   noise (use the spread across the 5 runs as the noise floor) on at least
   one shape in the set, **with no shape regressing beyond its own noise
   floor**.

## Branch and Commit Workflow

- Work directly on the current integration branch
  (`users/SreecharanGundaboluAMD/ck_improvements_gfx1250`) — do not create a
  new branch per item. Items accumulate here as the work proceeds.
- **One item per commit, no exceptions**, even though items share a branch:
  each commit's diff must be self-contained (touch only the files/lines that
  item's task names) so it can be cleanly `git cherry-pick`ed onto its own
  `gfx1250-opt/<item-id>-<short-slug>` branch later for a focused, reviewable
  PR. Do not squash multiple items into one commit and do not let one item's
  commit depend on another's uncommitted state.
- Commit message format (unchanged — this is what gets cherry-picked, so it
  must stand on its own):
  ```
  gfx1250: <one-line summary of the change>

  Item: <item ID from this roadmap>
  Source: <linked doc>#<section>

  Before: <shape> -> <median TFLOP/s or GB/s>
  After:  <shape> -> <median TFLOP/s or GB/s>  (+X%)
  No regression observed on: <other benchmarked shapes>
  Verified: ckProfiler -v 1 pass on all benchmarked shapes
  ```
  (Tier 0 correctness-only items: replace the Before/After/regression lines
  with the pass/fail verification actually run, per that item's Verify step.)
- Do not rebase/merge this branch into `develop` directly — the eventual
  per-item cherry-picked branches are what get opened for review; this
  roadmap does not grant standing authorization to merge to main lines.

---

## Tier 0 — Correctness Gate (verify before/alongside any Tier 1+ item)

### T0-01: Audit gfx1250 narrow-accumulate WMMA overloads for the LLVM C-operand WAR hazard
**Not a performance task** — a silent-wrong-answer risk that any scheduling
change (including several Tier 1 items below) could newly expose or mask.
Run once, and re-run after any change that touches WMMA issue scheduling in
the files this touches.

- **Source**: `HIPCONV_GFX1250_CONV_LEARNINGS.md` §1.1 (full mechanism,
  hipConv's exact fix).
- **Target**: `include/ck_tile/core/arch/mma/wmma/wmma_gfx12.hpp:417` (bf16→bf16)
  and `:1067` (fp16→fp16) `amdgcn_mma` specializations, and every call site
  in `include/ck_tile/ops/gemm/{pipeline,block}/*.hpp` that instantiates
  them for gfx1250.
- **Task**: confirm whether a `__builtin_amdgcn_sched_barrier(0)` (or
  equivalent) already follows every issue of these two specializations. If
  not, add one at each site per hipConv's exact fix pattern (cited in the
  source section).
- **Verify**: run `wmma`-family CK unit tests
  (`test/ck_tile/warp_gemm/`, any gfx1250 WMMA test) at high occupancy
  (large grid) repeatedly (≥20 runs) looking for intermittent `valid:n` —
  this hazard is occupancy-dependent and may not reproduce on small test
  grids. No ckProfiler TFLOP/s comparison needed for this item; record
  pass/fail only.

---

## Tier 1 — Highest-Confidence Performance Wins

### T1-01: Add LDS row/bank-conflict padding to CK's gfx1250 WMMA epilogue and transposed-operand staging
**Highest priority in this roadmap.** Independently measured as the single
largest gfx1250 WMMA performance lever found across this investigation
series (+41–98% relative, up to +70% specifically on wrw-shaped problems).

- **Source**: `MISA_GFX1250_CONV_LEARNINGS.md` §1.3 (root cause: CK's
  epilogue has zero padding, confirmed by direct grep) and the underlying
  MISA measurement it cites (`docs/gfx1250_w3_transposed_pitch_sweep.md`,
  `docs/gfx1250_w4_tile_size_comparison.md` in `/home/sgundabo/MISA`).
- **Target**: `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma_base.hpp`
  (confirmed zero pad/bank-conflict logic present) and any WMMA main-loop
  LDS staging for transposed operands (bwd-weight's B operand, wrw's A/B
  operands) in `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`.
- **Task**: add a row-padding parameter to the LDS tile-linear address
  computation in the epilogue (and transposed-operand staging) such that
  the row stride is no longer an exact multiple of the LDS bank period (64
  banks × 4 bytes on gfx1250). Follow MISA's finding: valid conflict-free
  pads satisfy `gcd(stride_dwords, 64) == 4` (e.g. `+16` byte pad); `+0`
  (current CK state) and other non-conflict-free pads must be avoided.
  Gate the new LDS budget against the existing 64KB/workgroup ceiling
  (`GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` §6.1 correctness checklist:
  "verify LDS padding and bank behavior").
- **Verify**: benchmark the full representative shape set, expect the
  largest gains on the small-output-channel wrw and small-channels-per-group
  shapes (most LDS-staging-heavy). Confirm `-v 1` passes unchanged.

### T1-02: Port `s_setprio(1)`/`s_setprio(0)` WMMA-issue bracketing from CK's v1 pipeline into the v3 pipeline actually used by grouped-conv
Confirmed present in `blockwise_gemm_pipeline_wmmaops_v1.hpp` (verified
directly: lines 934–936, 940–942) but **absent** from
`blockwise_gemm_pipeline_wmmaops_v3.hpp`'s `HotLoopScheduler()` (confirmed
directly: no `s_setprio` occurrences in that file). Grouped-conv WMMA v3
device ops (`device_grouped_conv_*_wmma_cshuffle_v3.hpp`) default to
`BlockGemmPipelineVersion::v1` but also ship `v3` instances — confirm which
is actually selected for the benchmarked shapes before/after.

- **Source**: independently confirmed by direct `grep` in this session
  (see also the cross-project convergence noted in
  `HIPCONV_GFX1250_CONV_LEARNINGS.md` §3 — hipConv, rocKE, MISA, and
  FlyDSL's `sched_dsrd`/`sched_mfma` primitives all independently arrived
  at bracketing the matrix-instruction issue with a priority/scheduling
  hint).
- **Target**: `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`,
  `HotLoopScheduler()` (lines 182–288) and the WMMA issue call sites inside
  the main K-loop body (lines ~495, 569, 826, 908 per
  `MISA_GFX1250_CONV_LEARNINGS.md` §1.5).
- **Task**: bracket each WMMA-issue burst in the v3 hot loop with
  `__builtin_amdgcn_s_setprio(1)` immediately before the first WMMA issue of
  the burst and `__builtin_amdgcn_s_setprio(0)` immediately after the last,
  matching v1's existing pattern exactly (each edge wrapped in
  `sched_barrier(0)`).
- **Verify**: benchmark the full shape set. This is a cheap, low-risk
  experiment — MISA's own docs note CK's comments don't quantify its
  benefit, so a neutral or small-negative result is a valid, reportable
  outcome; do not force-merge if it regresses.

### T1-03: Re-test CK's v3 `HotLoopScheduler()` on/off on real gfx1250 hardware
The scheduler is confirmed **active** (not dead code, contrary to an
initial hypothesis — see `MISA_GFX1250_CONV_LEARNINGS.md` §1.5), but four
independent projects (CK's own now-superseded belief, MISA, rocKE, FlyDSL)
each found that automatic or hand-templated instruction interleaving tends
**not** to beat a simpler baseline on gfx1250 WMMA specifically (see
`FLYDSL_GFX1250_CONV_LEARNINGS.md` §8 for the fourth independent
confirmation).

- **Source**: `MISA_GFX1250_CONV_LEARNINGS.md` §1.5 and §4.2 (MISA's own
  measured 4–7% regression from a structurally similar interleave scheme);
  `FLYDSL_GFX1250_CONV_LEARNINGS.md` §8.
- **Target**: `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`,
  `HotLoopScheduler()`.
- **Task**: add a compile-time toggle (template parameter or macro) to
  disable the `__builtin_amdgcn_sched_group_barrier` calls inside
  `HotLoopScheduler()` for gfx1250 specifically, leaving the compiler's
  default instruction scheduling in place. This is an **investigation**,
  not a guaranteed-direction change — do not assume disabling it wins.
- **Verify**: benchmark both configurations (scheduler on vs. off) across
  the full shape set. If "off" wins on net, propose disabling it for
  gfx1250 only (leave gfx950/other archs untouched); if "on" wins, close
  this item with no code change and record the confirmation.

### T1-04: Condition TDM/global-load prefetch issue timing on tile width, if CK's gfx1250 pipeline issues an explicit next-tile prefetch
- **Source**: `FLYDSL_GFX1250_CONV_LEARNINGS.md` §5 (the exact rule: wide
  tiles issue prefetch *before* the first K-step's WMMA; narrow tiles issue
  it *after*, to avoid the prefetch's issue-cost sitting on the critical
  path when there isn't enough independent MMA work to hide it).
- **Target**: wherever CK's gfx1250 WMMA main loop issues its next-K-tile
  global/TDM load relative to the current tile's WMMA compute — locate in
  `blockwise_gemm_pipeline_wmmaops_v3.hpp`'s prefetch-stage logic (per
  `GFX950_VS_GFX1250_COVERAGE.md` §"CoreSrc" findings: `PrefetchStages = 2`,
  `PrefillStages = 1`).
- **Task**: determine CK's current fixed prefetch-issue point in the K-step
  unroll; if it does not already condition on tile width (number of WMMA
  M-repeats per wave), add that conditioning per FlyDSL's rule.
- **Verify**: benchmark the shape set, paying particular attention to wide
  vs. narrow tile configurations within the same precision (compare a
  128×128 instance against a 64×64 instance on the same shape).

---

## Tier 2 — Medium-Confidence Wins / Coverage-Unlocking Fixes

### T2-01: Implement block-diagonal WMMA packing for small-channels-per-group grouped convolution
Independently confirmed as the correct approach for small-group WMMA
convolution by **three separate projects** (hipConv, and two more
cross-referenced within the FlyDSL/MISA investigation) — the strongest
convergent-evidence item in this roadmap, and the most directly
convolution-specific (not a generic-GEMM technique retrofitted to conv).

- **Source**: `HIPCONV_GFX1250_CONV_LEARNINGS.md` §4 (full mechanism:
  pack `GPW = 16/G` groups diagonally into one 16×16×32 WMMA instruction for
  channels-per-group `G ∈ {4,8,16}`, structurally zeroing off-diagonal
  cross-group products in the A-operand construction; separate K-tap
  pairing compounds a second packing axis).
- **Target**: CK's gfx1250 grouped-conv WMMA instance library
  (`library/src/tensor_operation_instance/gpu/grouped_conv{2d,3d}_fwd/.../wmma/`)
  and the underlying device-op/gridwise-gemm A-operand construction for
  small-`G` grouped conv.
- **Task**: this is new kernel engineering, not a parameter tweak — design
  and implement a new WMMA instance variant (or generalize an existing one)
  that packs multiple small groups into one WMMA instruction's M-rows,
  following hipConv's diagonal-masking A-operand construction as the design
  reference (do not copy hipConv code verbatim — CK's device-op/gridwise
  abstraction is structured differently).
- **Verify**: benchmark the small-channels-per-group shape in the
  representative set (and add a `G=8`, `G=16` variant); compare against
  CK's current (padded-to-full-tile) small-group handling.

### T2-02: Use gfx1250 TDM's native per-dimension OOB-clip field for wrw/small non-tile-aligned shapes; branch the output-store path on split-K
- **Source**: `FLYDSL_GFX1250_CONV_LEARNINGS.md` §7 (full mechanism +
  FlyDSL's own measured 15–82% regression from applying a generic-safe
  store path unconditionally — the store-path branch on split-K is not
  optional, it's load-bearing for this to be a net win).
- **Target**: CK's `cluster_load`/TDM usage (if any) in the gfx1250
  grouped-conv load path, and the output-store selection logic in
  `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma_base.hpp`
  / the relevant `device_grouped_conv_*_wmma_cshuffle_v3.hpp` split-K branch.
- **Task**: (a) if CK pads or masks non-tile-aligned M on the host side for
  gfx1250 wrw/small shapes today, replace it with the TDM engine's native
  per-dimension OOB-clip descriptor field, matching the mechanism cited in
  the source section; (b) branch the epilogue store path so that
  `split_k == 1` (or non-split-K) uses the fast/aligned store path
  unconditionally, and only `split_k > 1` (which needs atomic accumulation
  TDM cannot do) falls back to the predicated/masked path.
- **Verify**: benchmark the small-output-channel wrw shape and at least one
  additional ragged-M shape (M not a multiple of the tile size); confirm no
  regression on the large well-fit fwd shape (which should hit the
  fast/aligned path both before and after).

### T2-03: Re-investigate CK's "only `warp_tile_k=128` is correct on gfx1250" finding as a probable CK-side bug
This is a **correctness investigation that could unlock a coverage/perf
win** (K=64 tiles may allow smaller, better-fitting configurations for
certain shapes) — treat it as diagnostic first, optimization second.

- **Source**: `FLYDSL_GFX1250_CONV_LEARNINGS.md` §3 (FlyDSL's own compiler
  treats K=64 fp8/bf8 WMMA as fully hardware-correct and uses it in
  production; CK's `warp_tile_k=32`/`16` wrong-result finding is likely a
  CK-side codegen/scheduling bug, not a gfx1250 hardware limitation).
- **Target**: CK's gfx1250 fp8/bf8 WMMA K=64/K=32 code generation —
  compare CK's register-layout formula for these K depths against FlyDSL's
  `getThrValLayoutAB` formula (`K = block*16 + (lane/16)*8 + within_block`,
  cited in the source section) to find the discrepancy.
- **Task**: locate the exact CK code path producing wrong results at
  `warp_tile_k=32`/`16` (per `dispatcher/python/grouped_gemm_abquant_utils.py`
  in the dispatcher subtree, or CK's WMMA instance library if the bug is
  reachable there), and either fix it or conclusively confirm it is a real
  gfx1250 hardware limitation (in which case, no action — but document the
  confirmation).
- **Verify**: if fixed, benchmark whether a K=64 instance outperforms the
  current K=128-only path on any shape in the representative set
  (particularly shapes where K=128 forces a larger, worse-fitting tile).

### T2-04: Audit CK's disabled gfx1250 MX-GEMM instance configs for two specific addressing bugs
- **Source**: `FLYDSL_GFX1250_CONV_LEARNINGS.md` §4 (two exact bug patterns
  FlyDSL found and fixed: tile-shape-coupled B-scale preshuffle addressing,
  and ragged-M-tail A-scale VGPR loads assuming full 32-row block
  granularity — both produce the same "illegal memory access"/"numerical
  issues" signature CK's disabled configs show).
- **Target**: `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_*`
  instance headers with `#if !defined(__gfx125__)`-excluded tile configs
  (see `GFX950_VS_GFX1250_COVERAGE.md` §5.4 for the exact excluded-config
  list and file names).
- **Task**: for each disabled config, check whether its E8M0 scale
  index/stride math (a) depends on the specific tile_n/tile_k chosen at
  compile time in a way that should be factored out, or (b) assumes a full
  32-row scale "super-row" without handling a shorter M tail. Fix and
  re-enable configs where either root cause is confirmed.
- **Verify**: re-enable each fixed config and run `ckProfiler` with `-v 1`
  on the shape that previously triggered the failure (per the CHANGELOG/code
  comment citing the original bug report), then benchmark it against the
  currently-shipping fallback config for the same shape.

---

## Tier 3 — Exploratory (lower confidence, worth a time-boxed spike)

### T3-01: Investigate cache coherence/temporal hints (SCOPE/TH) on CK's gfx1250 WMMA memory operations
- **Source**: `GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` §1 (full hint
  taxonomy and suggested per-buffer-role usage: `RT_NT` for cooperative
  A/B loads, `NT` for C stores, `LU` for last-use reads). Entirely
  unexplored axis — no prior item in this roadmap touches it.
- **Target**: CK's buffer-descriptor construction for gfx1250 A/B tile
  loads and C-tile stores (`include/ck/utility/amd_buffer_addressing.hpp`,
  the same file already patched for the DEVICE-scope atomic fix).
- **Task**: time-boxed spike — add the suggested temporal hints to the
  buffer descriptors used by CK's gfx1250 WMMA A/B loads and C stores, one
  buffer role at a time (do not change all three at once, to isolate which
  hint helps/hurts).
- **Verify**: benchmark the full shape set per hint change; heed the
  source's explicit warning to classify actual reuse before applying a
  hint mechanically — a wrong hint can *regress* performance by evicting
  useful cache lines.

### T3-02: Add "claused stores" as a third output-store strategy option
- **Source**: `GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` §2.
- **Target**: `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma_base.hpp`.
- **Task**: as an alternative to LDS-reshuffle staging, compute all output
  addresses first and issue the stores as a back-to-back instruction clause
  (no interleaved unrelated instructions between them), letting the
  hardware write-combiner merge adjacent partial-cache-line writes.
- **Verify**: benchmark the 1×1/shallow-K shape in the representative set
  (this technique is explicitly framed as most relevant when store cost is
  a large fraction of runtime).

### T3-03: Implement TF32 support on gfx1250 via bf16-pair emulation
This is a **coverage addition**, not a pure optimization of an existing
path — include it in this roadmap because it makes an entire precision
class performant on gfx1250 where CK currently has none, using a technique
already proven correct and tested by another project.

- **Source**: `HIPCONV_GFX1250_CONV_LEARNINGS.md` §5 (full mechanism:
  store as fp32, split into a big/small bf16 pair on the way into the MMA,
  same emulation technique CK's own XDL path already uses for TF32 on other
  architectures — this is porting an existing CK pattern to a new arch, not
  inventing one).
- **Target**: CK's TF32 gating (`CK_ENABLE_TF32` currently excludes gfx1250
  per `GFX950_VS_GFX1250_COVERAGE.md` §3) and the relevant WMMA instance
  files.
- **Task**: extend CK's existing bf16-pair TF32 emulation (used elsewhere in
  the XDL path) to the gfx1250 WMMA path; extend the `CK_ENABLE_TF32` gate
  to include gfx1250.
- **Verify**: benchmark a TF32 conv shape (data type arg `8` in
  `ckProfiler grouped_conv_fwd`'s `--help` table) with `-v 1`; there is no
  "before" number to compare against since this is new coverage — record
  the new absolute TFLOP/s instead.

### T3-04: Cluster-launch/TDM-multicast for grouped-conv weight/activation reuse across neighboring workgroups
**Explicitly time-boxed and lower-priority** — the underlying hardware
mechanism is documented as safe (graceful, correctness-preserving fallback
on peer-drift timeout), but every project that has attempted it
(FlyDSL) has hit real, currently-unresolved toolchain/compiler issues.

- **Source**: `GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` §4 (hardware
  mechanics and the risk re-calibration versus `FLYDSL_GFX1250_CONV_LEARNINGS.md`
  §9's caution).
- **Target**: n/a — this is a research spike, not a specific file change.
- **Task**: assess whether CK's build/compiler toolchain for gfx1250
  supports cluster-launch (`hipLaunchAttributeClusterDimension`) and TDM
  `workgroup_mask` descriptors at all before attempting any kernel change.
  Convolution's natural multicast opportunity (per
  `GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` §5, item 8: neighboring
  output-tile workgroups very commonly share overlapping input-activation
  or weight footprints) makes this worth a spike, but do not commit
  production kernel changes until the toolchain-level feasibility is
  confirmed.
- **Verify**: n/a for this spike phase — report findings only.

---

## Appendix: Full Cross-Reference

| Item | Primary source doc | Section |
|---|---|---|
| T0-01 | `HIPCONV_GFX1250_CONV_LEARNINGS.md` | §1.1 |
| T1-01 | `MISA_GFX1250_CONV_LEARNINGS.md` | §1.3 |
| T1-02 | `HIPCONV_GFX1250_CONV_LEARNINGS.md` / `MISA_GFX1250_CONV_LEARNINGS.md` | §3 / §1.5 |
| T1-03 | `MISA_GFX1250_CONV_LEARNINGS.md` / `FLYDSL_GFX1250_CONV_LEARNINGS.md` | §1.5, §4.2 / §8 |
| T1-04 | `FLYDSL_GFX1250_CONV_LEARNINGS.md` | §5 |
| T2-01 | `HIPCONV_GFX1250_CONV_LEARNINGS.md` | §4 |
| T2-02 | `FLYDSL_GFX1250_CONV_LEARNINGS.md` | §7 |
| T2-03 | `FLYDSL_GFX1250_CONV_LEARNINGS.md` | §3 |
| T2-04 | `FLYDSL_GFX1250_CONV_LEARNINGS.md` | §4 |
| T3-01 | `GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` | §1 |
| T3-02 | `GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` | §2 |
| T3-03 | `HIPCONV_GFX1250_CONV_LEARNINGS.md` | §5 |
| T3-04 | `GFX1250_GEMM_OPTIMIZATION_PRINCIPLES.md` / `FLYDSL_GFX1250_CONV_LEARNINGS.md` | §4 / §9 |

Items deliberately **excluded** from this roadmap (present in the source
docs but not performance-actionable or not conv-specific enough to include
here): gfx1250 real-hardware CI setup, arch-classification refactors
(`is_xdl_supported()` cleanup), multi-target CMake shard-convergence audit,
static compile-time WMMA/TDM legality tests, PR-checklist adoption, and
documentation-naming items — these remain valid engineering work but are
process/infrastructure, not conv performance optimizations with a
before/after ckProfiler signal.
