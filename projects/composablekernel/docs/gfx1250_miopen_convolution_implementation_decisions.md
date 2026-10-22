# gfx1250 MIOpen-facing CK convolution: implementation decisions and trial evidence

This records the implementation work following [the performance scope](gfx1250_miopen_convolution_performance_scope.md). The scope report describes the earlier investigation; this file distinguishes changes **retained in this branch**, temporary candidates **removed after measurement**, and ideas **not yet tested as convolutions**.

**Answer to the shape-selection question:** not every initial rejection covered the strongest case for its proposed mechanism. In particular, the first forward-v3 comparisons used short 1x1 reductions. We subsequently tested FP16 and BF16 on 3x3, C=128: **36 K32 reduction blocks**, a sustained two-prefetch case. The v3 row was correct but still lost to its v1 counterpart or the best existing family, so it remains removed. Other conclusions below are explicitly *candidate- and workload-specific*. In particular, the removed async prototype does not disprove async hardware transfers, and LDS-transpose never had a convolution performance result.

## Measurement boundary

- GPU: gfx1250. New builds and executions used `/home/sgundabo/rocm-10.2` (`hipcc` AMD clang 24). Only one build or GPU benchmark ran at a time; builds used at most 100 jobs.
- `ckProfiler` was built with `GPU_TARGETS=gfx1250`, `DTYPES=fp16;bf16;fp32`, `MIOPEN_REQ_LIBS_ONLY=OFF`, and the new `CK_PROFILER_MIOPEN_LAYOUTS_ONLY=ON`. The last option restricts profiling to 2D NHWGC and 3D NDHWGC grouped forward/backward-data/backward-weight, including the explicit-GEMM WRW competitors. `MIOPEN_REQ_LIBS_ONLY=ON` itself disables the profiler target. See `profiler/README.md` for the build recipe.
- Latencies below are **milliseconds for CK invocations**, not MIOpen end-to-end latency. The timed CK launcher includes its preprocessing, so split-K clears and two-stage WRW conversion are not omitted. Timed comparisons alternate candidates; the retained-path table gives medians and observed ranges from three independent runs per candidate. Candidate IDs are shape-dependent; enumerate with `--list-instances` before using `--instance`.
- These are synthetic boundary probes, not a frequency-weighted MIOpen workload or held-out ranking set. GPU clock/power/cache policy was not independently fixed or measured. Do not extrapolate a shape-specific ratio into a production speedup. No new MIOpen plugin or driver was built or timed, per the requested CK-profiler-first boundary.

## Retained changes

| Package | Change and safety limit | Evidence |
|---|---|---|
| W01 backward-data | Skip the output clear only for homogeneous FP16/BF16, no D tensor, split 1, packed channels-last 2D/3D `Filter1x1Stride1Pad0` with positive matching extents and an actual 1x1/stride-1/pad-0 problem. Aliased-D behavior is preserved; general filters, stride holes and split atomics still clear. | Poisoned-output CPU-reference tests exercise 2D and 3D, partial tiles, odd C/K, multiple groups, and stride-2 holes. In a final ROCm 10.2 trace, the selected 1x1 WMMA-v3 kernel has **no `hipMemsetAsync`**, while the stride-2 control dispatches a fill immediately before its kernel. The traces prove dispatch removal/retention, **not a measured W01 latency gain**. |
| W01 WRW | Omit direct-output or FP32-workspace clearing only for packed split-1 `Set` kernels with complete logical weight coverage. Two-stage merged groups, transpose/irregular strides and split >1 keep initialization; workspace alignment padding is not promised to be written. | FP16/BF16 dirty-output and NaN-workspace tests pass for 2D **and 3D**, including odd C/K, repeated invocation and split-2 atomic controls. |
| W02 build parity | Use the normalized `SUPPORTED_GPU_TARGETS STREQUAL "gfx1250"` guard for the existing BF16 large-tile compile options in the three required 2D NHWGC CMake lists. Mixed-architecture sources do not receive target-specific flags. | Compile commands from the ROCm 10.2 `GPU_ARCHS=gfx1250`, MIOpen-only CK configuration contain `-enable-post-misched=1`, `--amdgpu-mfma-vgpr-form` and gfx1250 on all three named sources. No flag-induced speedup or MIOpen candidate-selection gain was measured. |
| W05 narrow WRW | Register FP16/BF16 M16/N16/K32 scalar direct-output WMMA rows with FP32 accumulation and one final low-precision `Set` store. The wrapper rejects an effective split >1; scalar atomic instantiations are excluded. | Odd per-group C=3, K=5 with G=3, N=2, 3x3, input 5x6, stride 1/pad 1: correct against CPU, with only two eligible ordinary candidates. Timed comparisons below include the competing two-stage FP32 workspace and cast. 3D odd-channel dirty-output tests pass as well. |
| W11 backward-data split-K | Admit only gfx1250 2D NHWGK/GKYXC/NHWGC, no-D, PassThrough FP16/BF16 vector-2/4/8 split 2 or 4 with packed E strides and aligned paired output. Other gfx12 devices, odd tails and 3D remain excluded. Atomic accumulation remains **nondeterministic**; a deterministic caller must choose split 1. | CPU-reference selected-candidate runs and dirty-output tests cover split 2/4, paired boundaries, odd-C rejection, repeated calls, large cancelling partials and subnormal-scale results. With `CK_TIME_KERNEL=OFF`, the three `launch_and_time_kernel_with_preprocess` overloads now execute preprocessing instead of silently skipping the required clear; a separate concrete split-4 FP16/BF16 test compiles in that mode and passes. |
| Profiler subset | `CK_PROFILER_MIOPEN_LAYOUTS_ONLY` filters the three grouped-convolution commands and links their six required layout libraries plus generic explicit WRW. Full profiler behavior remains unchanged when it is OFF. | Filtered `ckProfiler` builds and runs with FP32/FP16/BF16 on ROCm 10.2. A group-major request is rejected instead of invoking an unlinked factory; selected FP32 2D and 3D forward instances pass CPU verification. |

### Measured retained-path comparisons

| CK shape and precision | Comparator median (range), ms | Retained candidate median (range), ms | Ratio |
|---|---:|---:|---:|
| WRW FP16, G=3 N=2 C=3 K=5, 3x3, 5x6, split 1 | two-stage: 0.04609 (0.04593–0.04920) | direct scalar: 0.03279 (0.03055–0.03870) | 1.41x |
| Same WRW, BF16 | two-stage: 0.04580 (0.04311–0.05103) | direct scalar: 0.03568 (0.03218–0.03708) | 1.28x |
| Backward-data FP16, G=N=1 C=64 K=512, 1x1, 8x8 | best eligible split 1: 0.01133 (0.01126–0.01138) | split 4: 0.00928 (0.00921–0.00960) | 1.22x |
| Backward-data BF16, G=N=1 C=16 K=1024, 1x1, 8x8 | best eligible split 1: 0.01110 (0.01104–0.01120) | split 4: 0.00876 (0.00854–0.00891) | 1.27x |

These are **best eligible nonsplit** comparisons for the stated backward-data shapes, not evidence that split 4 wins generally: at BF16 K=512/C=64, the best split 1 was faster. The direct WRW advantage also shrinks on longer reductions; search should retain existing competitors. FP32 CK forward execution was checked, but no new FP32 numerical contract or FP32 speedup is claimed.

## Trial decisions: was the workload favorable?

| Package | Intended mechanism and workloads actually exercised | Decision, limits of evidence |
|---|---|---|
| W03 forward v3 | Two-prefetch pipeline should amortize its extra buffer over a long reduction. Initial FP16 scalar 1x1 C=32/64/96 produced just 1/2/3 K32 blocks at N=1, 8x8: **not** a sustained-hot-loop test. Aligned 1x1 C=32/128, Kout=64/128, N=8, 28x28/56x56 did provide many spatial CTAs but only 1/4 K32 blocks. Matched v3 could beat v1, yet existing old-WMMA/XDL families were much faster. **Follow-up in response to the shape concern:** 2D Default FP16/BF16 G=1 N=8 C=Kout=128, 3x3/s1/p1, 28x28 gives GEMM M=6272, output N=128, reduction R=1152 = **36 K32 blocks**; the aligned M128/N64/K32 v3 row was supported, executed and CPU-correct. | Retired the experimental rows **after the long-K check**. FP16 matched-tile medians: v3 **0.05396** vs v1 **0.05129** ms; all-candidate best **0.01601** ms (205 eligible). BF16 selected v3 **0.05129** vs v1 **0.05203/0.05096** ms; the all-candidate run gave v3 **0.04048** vs best **0.01590** ms (188 eligible). Thus the tested v3 tile is not competitive even in this favorable sustained-prefetch case. This does not disprove different tiles or a redesigned pipeline; no actual VGPR/spill/occupancy inspection was performed. The temporary 2D registrations were removed again. |
| W04 N16/N32 small-channel tiles | Forward GEMM N is **per-group output K**; backward-data GEMM N is **per-group input C**. New FP16/BF16 width-1/2/4 N16/N32 rows directly targeted the old N64 padding. Tests included G=1/4/32, symmetric C/K=1/2/4/12 and asymmetric C/K=3/5 or 7/5, 1x1 and 3x3, short and long 2D spatial extents, plus odd-channel 3D 3x3x3. Examples: forward G32 C=K=1 1x1 new **0.01443** vs best **0.00548** ms; G32 C=K=4 3x3 new **0.01351** vs best **0.00726**; 3D G4 C=3/K=5 new **0.02033** vs best **0.00892**. Backward-data 3D G4 C=3/K=5 new **0.01755** vs best **0.01594**. | Removed the tested portfolio because existing old-WMMA/ported/merged competitors won on these genuinely narrow-N cases. It is **not** a proof against every narrow tile: symmetric C/K=3/6/7/8/16, additional asymmetric C/K, more stride/dilation combinations, and actual vector-instruction/resource counts remain gaps. Odd channels already had scalar fallbacks; no functionality was added by these rows. |
| W06 FP16 depthwise merge16 | The borrowed BF16 geometry merges 16 depthwise groups into M16/N144/K32, cutting the group grid by 16 while computing off-diagonal/padded work. Eligible FP16 C=K=1, 3x3 split-1 cases were run at G16/N1/8x8 and G32 or G64/N8/28x28. G16/N1 merged **0.13975** vs ordinary two-stage **0.03142** ms; G64/N8 merged **2.18881** vs direct scalar **0.32624**. | Removed the FP16 merged row: it lost badly on the *specified first-experiment group counts*, including the larger 28x28 case. **Qualification:** G/16 is only 1/2/4 group clusters at those G values; fixed split 2/auto and substantially larger G were not benchmarked. A future high-group occupancy experiment may change this verdict, but the measured row is not a current improvement. Existing BF16 merged support was not removed. |
| W07 factor-4 forward packing | Tested the exact example sweet spot: FP16 G=32 N=64, per-group C=K=4, 3x3/s1/p1, 28x28. Packing merges four groups and reserves four times the source weight elements; the example allocates scratch and times **pack plus GEMM each invocation**. Four repeats: median **0.04527** ms versus best eligible un-packed CK factory **0.05858** ms. | **Positive measurement, not a dropped performance idea.** No candidate was registered: MIOpen's current 2D forward workspace-size callback reports zero and its non-WRW invokers do not set CK's workspace pointer. A packing row could appear eligible and dereference null scratch. That integration contract must be implemented and tested before registration; this CK-only pass did not test MIOpen integration. BF16/factor-2/other shapes remain unmeasured. For this temporary Clang-24 example build, an unused local variable was deleted to satisfy `-Werror`; the source line was restored afterward, with kernel parameters unchanged. |
| W08 gfx1250 wave transfer/direct store | Aligned FP16 K32 native-fragment/direct-store geometry was exercised: block128/M64/N256/K32, output NRepeat4, C=32/Kout=256, G=N=1, 1x1, 8x8; also C=128/Kout=256, N=8, 56x56. Full-output, partial tiles/G>1 and padded 3x3 passed CPU verification after fixing a wave-to-WMMA K-chunk descriptor mismatch in the trial. The short shape targets launch/CShuffle overhead; the larger shape offers enough work to amortize it. | The **tested port** lost to existing families: **0.01479** vs best **0.00393** ms on the short shape, about **0.02736** vs **0.00889** on the larger shape. Removed the opt-in guard, helper changes and tests. These are appropriate aligned/direct-store shapes, but no ISA store-width, LDS transaction, barrier, VGPR or occupancy counters were captured; the measurement does not prove direct stores are inherently slow on gfx1250. |
| W09 async global-to-LDS | An isolated FP16/BF16 identity 1x1, vector-8, K32, single-LDS-buffer async trial reused the gfx1250 direct-load wrapper with ASYNCcnt and DS-read drains. CPU-correct probes included grouped partial tiles and C=32/40/64/96/128/256, Kout=64/72/128, N=1/4/8, and spatial 7x5/8x8/28x28/56x56. These are aligned supported cases, and C=256 supplies eight K32 blocks; the trial did **not** accept 3x3/im2col. Matched FP16 v1 vs async: C32/Kout64,N4,28x28 **0.01107/0.01397** ms; C64/Kout128 **0.01393/0.01888**; C128/Kout128,N8,56x56 **0.01517/0.01801**. | Removed **this single-buffer prototype**, which lost on the tested supported cases. This is **not** evidence that async transfers cannot help a different convolution pipeline: no actual staging-register/VGPR bottleneck, emitted async opcode, spill/occupancy delta or transfer stalls were measured; the prototype did not overlap a producer with useful compute via double buffering. A 3x3 C128 long-K async candidate was never implemented or eligible, so cannot be called a failed 3x3 experiment. |
| W10 LDS-transpose compute loads | A standalone gfx1250 FP16/BF16 tagged 16x16 LDS mapping test (also padded 13x11 logical tile) passed for the existing `ds_load_tr16_b128` wrapper. The legacy convolution A/B compute-copy descriptor was **not** ported or executed through the wrapper. | **No convolution latency result and no performance rejection.** The standalone test was removed rather than leaving an unused CI target. First identify an operand with measured software-permutation/LDS cost, then prove its physical descriptor and every lane address valid: the LDS-transpose ISA ignores EXEC even on padded tiles. Compare actual convolution codegen and bank behavior before adding a candidate. |

## Earlier-stage gates and reproducibility

1. **MIOpen selection/FP32:** No grouped-solver ranking was changed without held-out MIOpen shapes. The earlier phase did not test the installed gfx1250 CK plugin, layout-conversion cost, chosen solver, deterministic policy or FP32 solver launch. The follow-on below checks selected FP16 packing with a *locally built matching* plugin, not installed-plugin coverage or end-to-end selection.
2. **Packing:** W07's CK-only win was not registered in the earlier phase because MIOpen supplied no forward scratch. F02 below resolves that workspace contract before registering one bounded FP16 candidate. Solver search/ranking and the end-to-end cost of layout conversions still require separate measurement.
3. **Shape claims:** W03's previously missing 36-block 3x3 case is now checked. W04/W06/W08 decisions are limited to the tabulated rows and shapes; W09 and W10 remain open architecture hypotheses with unestablished bottlenecks. Any renewed trial must record supported **and executed** candidates, exact type strings, CPU correctness, per-group dimensions, whole-invocation median/spread and actual code-object resources.

The retained source and tests are in `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_{data,weight}*wmma_cshuffle_v3.hpp`, `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_wmma_instance.hpp`, the three 2D NHWGC CMake lists, `profiler/src/`, and `test/grouped_convnd_bwd_{data,weight}/`. The three focused GPU test binaries passed **13 cases** total: dirty 2D/3D backward-data and WRW outputs/workspaces, paired split boundaries/numerical cases, and untimed preprocessing. The `CK_TIME_KERNEL=OFF` test explicitly checks all three preprocess-launcher overloads. Final single-target compile commands and a CPU-correct FP32 2D/3D forward smoke run establish build coverage, not FP32 speedup.

## Follow-on: F01-F09 measured decisions

These experiments implement the [follow-on scope](gfx1250_miopen_convolution_follow_on_scope.md) against gfx1250 with ROCm 10.2/Clang 24. The starting branch commit was `ea1b3bab97a`; candidate measurements used the subsequent working-tree changes, not an unmodified commit. The filtered CK profiler used `GPU_TARGETS=gfx1250`, FP16/BF16/FP32, `CK_PROFILER_MIOPEN_LAYOUTS_ONLY=ON`, and `MIOPEN_REQ_LIBS_ONLY=OFF`. A detached clean-HEAD build with only F01 added supplied the matched pre-trial metric. All new CK numbers below are **raw complete `Invoker::Run` calls**, each the mean of 50 separately synchronized same-stream HIP-event intervals after warm-up, with hot-reused buffers, no artificial cache flush and no fixed subtraction. Cross-process medians are identified explicitly; clocks/power were not pinned. These numbers cannot be substituted for the earlier adjusted `Perf:` measurements or for MIOpen end-to-end latency.

### Retained

- **F01 raw metric and candidate identity.** The three grouped-convolution profilers accept `--raw-invocation` without changing the legacy positional timing mode. An untimed raw call includes split-atomic initialization, two-stage conversion and forward weight packing when required. The forward factory no longer invokes or lists its first operation twice: the BF16 G1/N42/C=K128, 3x3/30x40 request lists 186 supported configurations instead of 187 with the duplicate. Selected records carry supported ID, exact type and requested/effective split. The gfx1250 `test_grouped_convnd_fwd_profiler_raw` checks list numbering and finite raw-only output; a traced BF16 split-4 backward-data invocation showed the clear and WMMA kernel in each call.
- **F02 safe factor-4 forward packing.** Register one gfx1250 FP16 2D NHWGC Default row with packed GKYXC weights, C=K=4 per group and divisible G. MIOpen's problem callback reserves the maximum supported CK scratch; selected NHWC/NCHW invokers query the actual argument, reject null/undersized scratch, and put a disjoint aligned fourth region after NCHW conversions. Internal CK plugin/loader API version is 6, requiring a matching rebuild. For G32/N64, 3x3/s1/p1, input 28x28, the selected candidate requests **36,864 bytes** (source weights 9,216 bytes). Five alternating process runs on the final CK binary gave best eligible unpacked **0.052956 ms** versus packed **0.049173 ms** median complete-call time, about **7.1% lower**; the packed row passed CK CPU verification. The locally rebuilt MIOpen core/plugin's CPU workspace-partition and GPU selected-solution tests both passed, including zero-scratch no-pack control, repeated changed weights, poisoned scratch/output, native NHWC and externally transformed NCHW. No MIOpen solver-ranking or end-to-end conversion-cost win is claimed.
- **F05 bounded BF16 short-R depthwise WRW (re-evaluation).** The first group-local trial lost on real corpus long reductions, but that did not exercise its short-R advantage. A BF16-only 2D NHWGC/GKYXC/NHWGK split-1 row now admits packed 3x3 groups only when `G >= 256` and `R = N*Ho*Wo <= min(G, 512)`; no FP16 registration or split atomics. Repeated CPU/GPU group-tail and pad/stride tests, null-pointer dry support, and negative long-R eligibility checks passed. Against the best other eligible CK split, independent raw-call medians were **0.014918 vs 0.017254 ms** at G256/R256, **0.016207 vs 0.019551 ms** at G450/R256, **0.025031 vs 0.028149 ms** at G512/stride2/R512, and **0.023473 vs 0.024067 ms** at G512/R512 (borderline; ranges overlap). All are synthetic; the inspected real BF16 WRW corpus has no high-G short-R command. See [the shape-strength audit](gfx1250_miopen_convolution_follow_on_shape_theories.md) for comparator split IDs, negative boundaries and unmeasured default MIOpen ranking.

A locally rebuilt matching MIOpen plugin enumerated the BF16 G450/R256 candidate; its split-1 selected solution passed two invocations with changing signed input on the same pointers and poisoned weight outputs. The selected MIOpen WRW solution requested **16,200 bytes of generic alpha/beta workspace**, which the test supplied, although the CK direct kernel has zero native scratch. This proves selected-path correctness, not automatic solver selection or MIOpen latency.

### Rejected or stopped trial regimes

- **F03 exact backward-data residue windows.** Old/new descriptor address and contribution tuples matched brute force, and FP16/BF16 repeated poisoned-output CPU tests passed. For BF16 G1/N42/K96/C24, 3x3/s2/p1, input 240x320, tightening reduced modeled M64 CTAs by about 1.45%; a slow matched WMMA-v3 candidate improved **0.976549 -> 0.936462 ms** (three-process medians), but the fast WMMA candidate was **0.243100 -> 0.241163 ms** (within the observed small-run variation) against an existing XDL family near **0.220 ms**. On both modeled N4/8x8 2x2/s2/p0 and 3x3/s2/p1 cases, matched fastest WMMA candidates regressed about 2-4% despite halved modeled CTA counts. No demonstrated best-family utility; the transform specialization and its tests were removed.
- **F04 general-convolution clear elision.** The conservative split-1 2D packed/no-D axis-coverage proof passed positive/negative dirty-output controls, but on the same real BF16 shape the faster native candidate worsened from **0.240788 to 0.275671 ms** with only F04 enabled (three-process medians), while a much slower native row gained about 1.5%. The cause of the faster row's regression was not isolated with traffic counters. Retain the earlier W01 1x1 guard, not the broad new predicate.
- **F05 unrestricted long-R trial.** The original FP32-reduction/group8-by-reduction32 scalar CTA passed FP16/BF16 group-tail and stride/pad correctness, but BF16 corpus M06/M14/M15/M25 raw complete-call single-run estimates were **5.189/4.913/1.309/1.529 ms**, versus best other eligible **1.096/1.034/0.407/0.513 ms**. Those real workloads have 1,575 or 6,300 serial terms per reduction lane, compared with at most 16 for the bounded BF16 row above. The costs of serial address work, transactions and stores were not isolated. The unbounded eligibility and FP16 registration were removed; this is not a rejection of short-R group-local reduction. These four comparisons are not independent-run medians.
- **F06 scalar WRW auto split.** Source tracing found no production auto-only gfx1250 consumer: MIOpen initializes fixed split 1 and enumerates it before auto, while its AI path excludes gfx1250. The existing split-1 scalar candidate remains available; no adapter was retained.
- **F07 CTA order.** The real BF16 3x3/30x40 winner was XDL-v3 with N0=1 and grouped M01=4, not the proposed old-map default 8. A real high-spatial 1x1/K512 winner also used a different native-v3 grouped map. The old-map 8-to-1 trial failed its actual-winner entry gate; no map or registration changed.
- **F08 forward address progression.** The traced BF16 3x3 winner's gfx1250 ISA retained repeated quotient/mask work. An isolated A-only carry-check row passed a grouped odd/tail CPU control but regressed on its matched G1/N42/C=K128, 3x3/30x40 case: five-process medians **0.026409 -> 0.030722 ms**. VGPRs fell 128 -> 120, with both at 32,768-byte LDS and zero scratch. The lower register count did not recover full-call latency; variant removed.
- **F09 backward-data one-slice lookup.** The actual BF16 K1024/C16 1x1/8x8 split-4 winner's gfx1250 ISA contained two serialized block-range scalar loads and a binary-search branch despite a compile-time one-slice capacity. A constexpr bypass removed 232 bytes of kernel text, but left 64 VGPRs, 16,896-byte LDS and zero scratch unchanged. Fifteen alternating full-call process runs measured **0.014090 -> 0.014639 ms** median with no speedup. Bypass and its trial tests removed; mixed-residue lookup remains unchanged.

The final retained follow-on changes are F01, F02 and bounded BF16 F05. Focused CK raw-reporting, prior backward-data overwrite/split-K, BF16 short-R WRW and selected CK packing tests passed; the locally built matching MIOpen forward and BF16 backward-weight selected-solution tests passed. The local MIOpen validation build used FP16/BF16/BF8/FP8 CK instance definitions and disabled inherited warning diagnostics for pre-existing CK header warnings; this is a build-only setting, not a new numerical contract. The installed plugin, held-out MIOpen default ranking, conversion-inclusive latency, deterministic admission and FP32 solver launch remain unmeasured.

## WM1 scheduling and WM2 reuse feasibility on the actual forward winner

**Decision: retain no scheduler or reuse change yet.** The BF16 2D NHWGC G1/N42/C=K128, 3x3/s1/p1, input30x40 S1 winner is the XDL-named, native-WMMA Interwave-v1 block256/M128/N128/K64 row at `device_grouped_conv_fwd_xdl_comp_instance.hpp:98` (supported ID 44 **only in these builds**). A complete-call factory pass again placed that row first of 186 supported configurations (0.027350 ms versus the next row at 0.031688 ms); this single sequential pass is a selection check, not a comparison against the independently timed experimental binary. The unchanged executable SHA-256 was `66b2203d7596f327a2660039fb229d2635fa4acf802e35f2ac10fbb70339c10b`, ROCm 10.2/Clang 24, gfx1250, with the filtered CK profiler's FP16/BF16/FP32 and MIOpen-layout-only configuration. Raw results below are each 50 hot-reuse, independently synchronized complete invocations, with legacy timing disabled.

**G0 provenance and limits.** rocprofv3 1.4.1 ATT saved loaded HSACO snapshots; dispatch `Kernel_Id` joined to `kernel_symbols.code_object_id=2` and the `s1_gfx1250_code_object_id_2.out` snapshot. The unchanged loaded object SHA-256 was `f8bb98f1dabd292ab893a428271b536dfbef02e836c36918f2fd2b6ee821ed52`. Its exact kernel metadata gives wave32, 394 CTAs of 256 threads, 32,768 bytes LDS, zero private scratch, 576-byte kernarg, 72 SGPRs, 244 VGPRs and zero recorded spills. rocprofv3's dispatch field reads **128 VGPRs**, not 244; those values have different provenance/units and must not be conflated into an occupancy assertion. A separately instrumented HIP/kernel timeline replaces the queue's device-memory ring and changes raw timing; none of its complete-call times is a latency baseline. `rocprofv3 --list-avail` exposes an emulated `SQ_VALU_WMMA_FLOP_BF16` and cache/activity counters, but not the proposed `SQ_INST_CYCLES_VALU_WMMA`, `SQ_INSTS_VEC32_VALU_WMMA`, `SQ_WAIT_INST_ANY`, or `SQ_INSTS_LDS_LOAD` on this installed configuration. No achieved WMMA issue or stall fraction was established.

| Independent one-row ablation | Loaded HSACO SHA-256; emitted changes | S1 matched raw complete calls (45 independent processes each) | Real S2 holdout: BF16 G1/N42/C48/K192, 3x3/input120x160 (nine processes each) |
|---|---|---|---|
| **WM1a:** omit only `s_setprio 1/0`, retaining all compiler scheduling and workgroup/LDS barriers | `c99ff2b3e530fa56961bac88d9e484d2f5537b9fbe8f0c502497c6cc1054fadf`; selected kernel has 0 versus 4 static `s_setprio`, the same 32 WMMA, 244 VGPRs, 32,768-byte LDS and zero scratch | Baseline 0.040012 versus WM1a 0.039950 ms median; 30-process held-out paired median **+0.000037 ms** for WM1a. No reliable improvement. | Matched baseline 0.281613 versus WM1a 0.280842 ms median; small relative to the unrelated winning family. **Reject priority removal.** |
| **WM1b:** independently use `NumMacClusters=2` for this tile, leaving priority enabled | `ee925ccb8fbe0f4ecbe5d56747e39e307a6f3c325a9c2d43aff212bc823006b9`; `KPerInnerLoop` 32→16, `KRepeat` 1→2, static WMMA count stays 32, priority writes 4→8, VGPR metadata 244→196, LDS/scratch unchanged. Extra cluster synchronization/scheduling also changes. | Baseline 0.040012 versus WM1b 0.039409 ms median; 30-process paired median **-0.000496 ms** (27/30 faster), but runs were bimodal around 0.026 and 0.040 ms and independently traced kernel durations did not show a reproducible stage gain. | Matched baseline 0.281613 versus WM1b 0.274164 ms median, but a separate all-eligible factory pass found row 153 at **0.218227 ms**; this row is not S2's best family. **Do not retain** a broad cluster policy or claim a production speedup from its lower VGPR count. |

The two temporary `/tmp/ck-wm1-{a,b}-overlay/` headers were each compiled into **only** the BF16 2D NHWGC comp registrar's replacement object and linked against identical baseline objects; neither changed `ck.hpp`'s global cluster macro, the registered tuple, the accumulation order or the source worktree. The matched row listed as ID44/186 in all three binaries. Both variants passed signed-integer CPU-reference checks with G3, partial K=136, C120, asymmetric padding, and 1-/2-/4-K-block controls (filter2x2/C8, filter3x3/C8 or C24); S1/S2 performance runs did not use CPU verification. Independent-process samples, not trace-perturbed timings, are saved at `/tmp/ck-wm1-s1-raw.csv`, `/tmp/ck-wm1-s1-holdout-raw.csv`, and `/tmp/ck-wm1-s2-holdout-raw.csv`. No clocks were pinned, and the bimodality was not explained. The S1 kernel-stage and mode-stability gate remains open for WM1b; there is no candidate-specific production parameter or MIOpen selection/latency result.

**WM2 feasibility stop.** CK's gfx1250 FP16/BF16 FP32 WMMA builtins still pass `false,false` for matrix reuse. Isolated compiler/assembler probes in the [scope](gfx1250_miopen_convolution_follow_on_scope.md#wm2--matrix-a-reuse-supported-encoding-unresolved-runtime-contract) establish modifier encoding, not correct runtime reuse. The supplied CDNA5 ISA §7.12 says RA/RB cache for the **next** instruction yet also says enabling reuse when the **current** instruction is not the same as the **previous** instruction has undefined results. It does not define sameness (opcode, physical A/B register sets, C/D, or modifiers), interposed instruction behavior, or first/last/loop boundaries. S1 repeats physical A while B and C/D change and waits can interpose. Do not enable A hints or execute a potentially undefined tagged sequence until an authoritative hardware/runtime contract resolves those conditions. WM3's affected native row and WM4's async resource/address gates remain separate deferred work, not implied by these scheduling results.

### Measurement-state follow-up: per-event modes, not inferred stage throughput

**Uninstrumented complete calls.** The unchanged baseline executable above and the isolated WM1b executable (`54906982a4502c9e00cc8c259846e4b74d0325c60598c20735b33c609812a847`) ran the same S1 row44 with `--instance 44 --raw-invocation`, positional time/verify both zero, 50 hot-reuse complete invocations per process. Sixty adjacent baseline/WM1b process pairs alternated AB and BA by round. Each pair has subprocess start/end monotonic timestamps, PID, position and the raw GPU-event complete-call mean in `/tmp/ck-wm1-state-paired-raw.csv`. A separate observer queried `amdsmi_get_gpu_metrics_info` about every 20 ms and timestamped the beginning and end of each query in `/tmp/ck-wm1-measurement-state-telemetry.csv`; it did **not** enqueue GPU work. Process windows contain 13–22 readings. Sensor refresh/averaging windows remain unknown, so these are **process-window context**, never clock or power for a 26–40 µs kernel invocation.

| Comparison from that AB/BA campaign | Baseline | WM1b | Interpretation |
|---|---:|---:|---|
| All 60 pairs | Median 0.039967 ms; 10 process means below 0.035 ms | Median 0.039614 ms; 4 below 0.035 ms | WM1b faster in 41/60 pairs; paired median difference **-0.000399 ms**, but the *mean* difference is +0.000728 ms because mode occupancy differs. Pooled medians/means cannot settle the mode-dependent benefit. |
| 46 pairs with **both** raw means at least 0.035 ms (post-hoc descriptive stratum) | Median 0.039993 ms | Median 0.039629 ms | WM1b faster in 37/46; paired median **-0.000425 ms**. AB: 17/22 faster, median -0.000455 ms; BA: 20/24 faster, median -0.000408 ms. The advantage survives both orders **within this observed high-mode stratum**; selecting the stratum after observing the outcome is not a prospective production speedup guarantee. |

Across all 120 process windows the eight reported gfx clocks were 2336–2370 MHz and socket power 1096–1121 W. In an independent 100-process, **baseline-only** state control (original binary, `/tmp/ck-wm1-baseline-state-control.csv`), 16 low-mode process means had 2359.18 MHz mean of their eight-clock process-window readings and 1103.2 W mean power; 84 high-mode means had 2356.73 MHz and 1103.3 W. A roughly 0.1% clock difference cannot by itself explain a roughly 50% event-time difference. Median child voluntary and involuntary context switches were 21 and 13 in **both** groups; cumulative child scheduling counts likewise do not identify the cause. The reported GPU activity was 100% even when `amd-smi process` detected no client; it cannot identify a contender. No clocks were locked. Neither these coarse samples nor the AB/BA balance excludes sub-process clock changes, cache/workload interference, host enqueue timing, or firmware state.

**Separate diagnostic, not a faster-kernel claim.** A temporary rebuild of **only the profiler's raw-event measurement translation unit**, linked to the unchanged baseline CK objects, stored each of the 50 already-measured HIP event intervals and printed them *after* the timing loop. It is a different executable, not the measured production binary or an ATT performance trace. Across 50 diagnostic processes (`/tmp/ck-wm1-baseline-event-diagnostic.csv`), 457/2500 intervals were below 33 µs and 2040/2500 were above 35 µs; 33 processes never crossed 33 µs, 16 crossed once, and one twice. Most mixed windows switched **from approximately 40 to 26 µs** after varying repeat indices, within the same warmed process; one slow 0.7 ms outlier divided by 50 does **not** explain the mean modes. In a *further diagnostic-only* 200-repeat build, 21/30 processes ended with all last 50 events in the fast mode, while seven ended with none (`/tmp/ck-wm1-long-event-diagnostic.json`). The profiler's positional `repeats=50` text is hard-coded, so do not use that 200-repeat build's mislabeled aggregate line as production latency; only its separately printed per-event values were analyzed. This establishes sustained, sometimes transitional event-level timing states, **not their hardware cause**. The earlier ATT loaded-object/codegen finding remains valid, but its instrumented durations neither prove nor refute the roughly 0.4 µs uninstrumented within-high-mode difference.

**Deferred WM1b reconstruction:** In an *overlay copy only* of `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp`, replace its single `NumMacClusters` definition with the following lines, and compile the overlay ahead of the standard headers for **only** `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/nhwgc/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwk_bf16_comp_instance.cpp`. Reuse baseline objects for every other translation unit; link this one replacement object into a separate profiler executable. The geometry test alone is **not** a safe global production policy or dtype guard; a whole-project rebuild with this overlay would change other rows. Keep the isolated `/tmp/ck-wm1-b-overlay/` header, `/tmp/ck-wm1-b-run/ckProfiler`, raw CSVs and loaded-object SHA-256 above as local trial artifacts; these paths are transient, while this reconstruction and findings survive in the repository.

```cpp
static constexpr bool WM1Target = BlockSize == 256 && MPerBlock == 128 &&
                                  NPerBlock == 128 && KPerBlock == 64;
static constexpr index_t NumMacClusters =
    WM1Target ? 2 : CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING_MAC_CLUSTERS;
```

S2 still bars generalization: WM1b's matched 0.274164 ms is approximately **25.6% slower** than a separately timed best existing 0.218227 ms family. A stable high-mode S1 directional signal preserves the *experiment*, not a registered row or a global cluster default. A prospective same-state complete-call and end-to-end best-family test is required before reconsideration; the fast-state cause and production MIOpen selection remain unestablished.

### WM2 authoritative-specification gate

The local AMD CDNA5 ISA §7.12 [official PDF](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf) says RA/RB **hint** that the current matrix is likely reused by the **next** instruction and should be cached, then warns that setting reuse when the **current** instruction is not the same as the **previous** instruction gives undefined results. Scale-WMMA §7.12.6 also describes forward-looking reuse; the §7.12.1 VGPR `Same` definition applies to a *different data-hazard table*, not this cache. An LLVM intrinsic boolean, assembler acceptance, and the installed CK Tile's convention of requesting A reuse after the first N iteration are implementation examples, not an authoritative dense-WMMA runtime contract. Four answers are still required from hardware documentation/AMD before enabling this hint:

1. Is the marked instruction the producer for a subsequent instruction, the consumer of a preceding cached operand, or both—and how does that reconcile the next/previous text?
2. Which exact WMMA opcode, A/B/C/D physical register ranges, values, encodings and modifiers must match; does only the hinted operand matter?
3. Which intervening waits, LDS/VALU instructions, branches or other WMMA operations preserve or invalidate the cached matrix?
4. What are the legal first/last hints and loop-backedge/tail-exit behaviors when no matching predecessor or successor exists?

None of the four has a verified answer in the available ISA wording. Do not extrapolate from the encoding probes or enable a possibly undefined hint; WM2 remains specification-blocked independently of the completed WM1 measurement-state investigation.

## TC1 and TC2: targeted BF16 convolution work from the 86-command NCHW inventory

**Retain two bounded changes, not a new tile portfolio.** The user's `tracelense_workload_conv_only.txt` omits layout overrides, so the driver requests **NCHW**. CK profiler rows below describe the CK plugin's *internal* channels-last layout; they are not end-to-end measurements of the original driver commands. The prior section-9 list-only census established eligibility only. These follow-on runs used ROCm 10.2 on gfx1250; user-provided scope and unrelated untracked files were not edited.

**TC1 — BF16 backward-data input line 2, C128/K10.** The added 2D NHWGK/GKYXC/NHWGC `Filter1x1Stride1Pad0` WMMA-v3 block128/M128/N128/K32 row changes **only the A/dY source width from 1 to 2**. It preserves B4, E8, FP32 accumulation, the K32 padded reduction, and fixed split1; its candidate-local support guard rejects split2/4 without altering the scalar parent. The factory's supported split1 IDs in the new build were 3 scalar and 4 paired (IDs are not stable across builds). On the exact G1/N42/C128/K10, input120x160 request, `ckProfiler grouped_conv_bwd_data 2 1 0 2 0 0 2 1 42 10 128 1 1 120 160 1 1 1 1 0 0 0 0 1 --instance <id> --raw-invocation` measures 50 hot complete CK invocations per process. The unchanged parent was already the best of the four preexisting split1 rows and of the eight preexisting auto-split supported configurations in the initial full-factory pass. Thirty AB/BA parent/new pairs had **0.050350/0.049665 ms** separate medians and a **-0.000686 ms paired median** (30/30 paired wins); 16 independent held-out pairs measured **0.050397/0.049698 ms**, paired median **-0.000668 ms** (16/16 wins). These are approximately 1.3% lower *internal CK complete-call* latencies, not conversion-inclusive NCHW or default-MIOpen speedup claims. Clocks were not pinned. The exact full-size paired row returned success with CPU reference enabled; focused direct tests passed signed and cancelling contributions from only K channels 8/9, changed data and poisoned dX over repeated calls, grouped K10, partial spatial tiles, K8/K16 and K11 scalar-only plus split2/4 candidate rejection. ATT tied both dispatches to distinct symbols in the same loaded HSACO: the selected A load instruction mix changed from **32 `buffer_load_u16` to 16 `buffer_load_b32`**, while eight B `buffer_load_b64`, 16 BF16 WMMA, 56 reported VGPR, 16,896-byte LDS and zero scratch were unchanged. ATT was used for codegen/resources only, never a latency comparator. Raw independent-process evidence is transient at `/tmp/ck-tc1-K10-{paired,heldout}-raw.csv`.

The new row's support predicate also requires `ck::is_gfx125_supported()` at runtime: `CK_USE_GFX1250` alone is build-wide in mixed gfx12 target lists and would otherwise expose the experimental row on gfx1200/gfx1201.

The new row is also explicitly reachable through the locally rebuilt MIOpen CK plugin with the **original NCHW layout**: a focused small K10 selected-solution test passed two signed channel8/9 reductions, poisoned dX, and the retained three staging regions. For exact line-2 N42/input120x160, a separate, temporary selected-plugin harness held the **scalar or vector2 CK type, split1, workspace, problem and caller buffers** fixed, alternated their invokers, poisoned dX *outside* each measurement, and bracketed complete MIOpen invocations in same-stream HIP events. Twelve independent processes each timed 12 alternating pairs of 50 hot calls; process-median scalar/vector times were **0.134021/0.132857 ms** and the median paired process delta was **-0.001097 ms** (12/12 vector2 faster). Modes differed between processes and clocks were not pinned. The full-size vector2 output was spot-checked at beginning, middle and end; exhaustive CPU reference and channel-tail cancellation were covered by the focused CK test instead. This is a **selected-CK, conversion-inclusive** result, not default MIOpen speedup: the original line-2 NCHW shape, with verification disabled for timing, still selected `ConvHipConv` at **0.169549 ms** on a separate driver pass. Selected-plugin artifacts are transient at `/tmp/miopen-tc1-exact-selected-process-pairs.csv` and `/tmp/miopen-tc1-selected-bwd-timing.cpp`.

**TC2 — packed BF16 G1, N42, unit-spatial NCHW pointwise.** For a default-layout, exactly 1x1/s1/p0 invocation with alpha1/beta0, the grouped CK wrapper now borrows current x/w/y caller buffers when the existing transpose solutions are skippable, the invocation descriptors remain packed and identical to the problem's NCHW descriptors, pointers meet 16-byte alignment, all three live spans are pairwise disjoint, and the **entire** conservatively reserved workspace is disjoint. Otherwise it retains all four original staging conversions. Workspace size, CK native scratch, output clear semantics, selected CK kernel and backward direction tags are unchanged. A fixed-ID BF16 forward test chose the existing `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_WmmaPorted<64, 64, 32, 32, Default, 32, 32, 2, 1, 8, 8, 8, 1, 1, 1>` CK row. It checked disjoint, overlapping x/y, changed inputs/allocations, poisoned outputs, missing workspace and nondegenerate spatial fallback; selected split1 backward-data/weight controls checked direction tags, overlap, workspace and output values. A separate rocprofv3 **structural** trace of six forward calls found the **same CK kernel symbol** in every call, zero transpose launches for the two safe calls and four for each overlapping or nondegenerate call (16 total). Tracing is not latency evidence.

The same fixed forward CK type also passed disjoint and overlapping NCHW tests for the **C256/K16, C32/K512 and C512/K32** unit-spatial controls, with per-case workspace and output checks. Split1 BWD/WRW checks cover the primary C16/K256 pair. These controls establish the bounded copy-elision contract across channel direction/widths; only the primary forward shape was timed.

A temporary, separately linked copy of the focused test (not a repository test modification) bracketed each complete selected CK invoker call in same-stream HIP events; each of 30 alternating process pairs took a median of 50 hot calls after resetting x **outside** each timed interval. The disjoint borrowed path had a **0.008733 ms** median; the overlapping staged fallback had **0.020550 ms**, with a **-0.011877 ms paired median** and 30/30 borrowed wins (AB and BA both lower). That comparison keeps the selected CK symbol and reserved workspace fixed but deliberately changes caller aliasing to force the legacy fallback; it does **not** isolate cache effects of aliasing or prove the same ratio for every nonoverlapping NCHW caller under an older library. Evidence is transient at `/tmp/miopen-tc2-paired-complete-call.csv` and `/tmp/miopen-tc2-selected-trace/selected_kernel_trace.csv`. An explicit grouped CK forward MIOpenDriver run of the original line-16 shape after the change verified against the GPU reference at **0.009014 ms** conversion-inclusive profiled kernel time; it fixes the solver, not a verified CK instance. The unmodified original **default Find** still selected `GemmFwd1x1_0_1` (observed **0.026218 ms** on a separate pass), so this is not a demonstrated default-driver speedup. A *restricted* fresh Find search comparing that solver with grouped CK, in a temporary database, selected grouped CK at **0.012299 ms**; this does not establish full default ranking or installed-plugin behavior. No layout workspace was removed.

The exact original unit-spatial backward-data and backward-weight NCHW commands also verified with the rebuilt local libraries: backward-data selected `ConvHipConv` at **0.012147 ms**; backward-weight selected grouped CK `ConvHipImplicitGemmGroupWrwXdlops` at **0.012793 ms**. Thus the WRW no-copy branch can be used by a default selected solver in this local build. Those are single default-driver observations, not a matched old/new WRW latency comparison or an installed-plugin ranking claim.

**Verification and limits.** Focused CK vector2 reference/factory tests and MIOpen's pinned K10 vector2 backward-data, four-pair forward, split1 backward-data/weight, original FP16 packed-forward and retained BF16 depthwise WRW tests passed locally. The original NCHW MIOpen driver and the CK internal-layout profiler are intentionally separate metrics. The small TC1 selected-CK gain merits retaining exactly one even-K BF16 pointwise row; the TC2 no-copy benefit applies only if grouped CK is actually chosen and its runtime pointers pass the borrow proof. Original line-2 BWD and line-16 FWD defaults selected other solvers; line-18 WRW selected grouped CK locally. The existing untracked scope document remains an investigation snapshot, not this outcome.

## Family-broadening follow-up: retained, gated, and rejected

**Retained CK domains.** The gfx1250 BF16 2D packed depthwise WRW direct candidate now admits `G>=128` rather than `G>=256`, but still requires `R=N*Ho*Wo<=min(G,512)` and effective split one. The 11x11 Fy2 row-strip admits G1-7 when its checked group-filter span is at most 847 FP32 elements per strip; its 11x11/s1/d1/p5 algorithm and scratch/index checks are unchanged. The grouped 3x3 Group16 row-strip keeps R=513..201600, S=N*ceil(Ho/8)*ceil(Wo/80)<=336, `S*ceil(G/16)<=10752` CTAs and at most 6,193,152 logical scratch bytes. Newly admitted G<192 or G>512 also needs at least 512 CTAs; the prior G192-512 support remains unchanged. These are algorithm/work budgets, not equality checks for N42 or any image dimensions.

**CK measurement boundary.** ROCm 10.2 gfx1250 filtered `ckProfiler`, `--raw-invocation`, time flag zero: each process reports 50 hot-reuse complete `Invoker::Run` GPU-event intervals, including CK clearing and finalization. For each new shape, `all` enumerated eligible requested auto and fixed splits; the comparator below is the fastest other supported family in that pass. Directional controls then alternated candidate/competitor processes in AB/BA order. GPU clocks were not pinned. These are CK internal-layout comparisons, not full NCHW MIOpen operation times.

| BF16 WRW family and held-out case | Candidate / best other CK (ms) | Evidence |
|---|---:|---|
| Group16 3x3 G24, N64, 28x28, stride 1 (512 CTAs) | 0.048634 / 0.060241 | 5/5 paired wins; synthetic work-level control |
| Group16 3x3 G58, N32, 28x28, stride 1 (512 CTAs) | 0.048261 / 0.056680 | 5/5 paired wins; independent 4k-command cohort |
| Group16 3x3 G88, N64, 55x55, stride 2 (1,536 CTAs) | 0.080532 / 0.158519 | 5/5 paired wins; independent cohort |
| Group16 3x3 G122, N32, 28x28, stride 1 (1,024 CTAs) | 0.059039 / 0.085064 | 5/5 paired wins; group tail |
| Group16 3x3 G576, N32, 14x14, stride 2 (1,152 CTAs) | 0.041192 / 0.051550 | 5/5 paired wins; above the former group limit |
| Group16 G48 (336 CTAs), G176 (154 CTAs), G576 (396 CTAs) | 0.095067 / 0.091730; 0.125689 / 0.063783; 0.035456 / 0.032844 | 0/5 paired wins each; motivate the new-domain 512-CTA gate |
| Fy2 11x11 G1/G2/G4, N32, 480x640 | 0.462042 / 1.176520; 0.903182 / 2.327340; 2.026360 / 4.653850 | One full all-split pass per shape; comparator auto was best other |
| Fy2 11x11 G7, N32, 480x640 | 6.766770 / 8.227750 | 5/5 paired wins; all-split pass found no faster other row |
| Direct short-R G128/R121, G176/R169, G240/R196 | 0.011472 / 0.013330; 0.013419 / 0.016998; 0.013953 / 0.018262 | 5/5, 7/7, 7/7 paired wins; retain the old R>G exclusions |

**MIOpen selection, not just CK eligibility.** A locally relinked gfx1250 ABI-6 plugin used the rebuilt current CK convolution archive; each original NCHW command ran a fresh full Find in an isolated performance DB and a subsequent cached/default invocation. GPU-reference verification passed. Group16 G58 selected `DeviceGroupedConvBwdWeightDepthwiseGroupedRowStripBf16<16, 8, 9, Split1>+-1` (cached full call 0.062997 ms); the excluded low-CTA G176 case selected the existing two-stage row at fixed split 32. Fy2 G2 selected `DeviceGroupedConvBwdWeightDepthwiseRowStripBf16<12, 128, 11, 16, Fy2, Split1>+-1` (cached 0.992172 ms). Short-R G176/R169 selected `DeviceGroupedConvBwdWeightDepthwiseBf16<8, 32, Filter1, Split1>+-1` (cached 0.024913 ms). These are selected-path demonstrations, not a newly measured 112-command sum, model wall time or installed-plugin claim. The independent `miopen_benchmarker/jonathan_4k_shapes.txt` supplies G58/G88/G122/G176/G576 and short-R controls; its shape geometry alone never established a speedup.

The above-range G576/N32/14x14 stride-two WRW command also passed fresh full Find and cached GPU-reference verification with the same Group16 type at requested split 1 (cached 0.058363 ms). This is a second selected family point, not an aggregate model claim.

**3D default-layout backward-data staging.** MIOpen now recognizes only selected no-D 3D XDL-v1 BF16/FP16 backward-data bare IDs with implicit CK split one, default alpha/beta, packed NCDHW descriptors and zero transposed-output pads as defining all packed E. Its CK invoker still clears E before residue GEMMs. Forward, weight-gradient, fused, explicit-split and unrecognized 3D IDs keep old-output staging. A selected-plugin GPU test covers stride-two holes, poisoned output/workspace, changed signed inputs and alias fallback. In a fresh full Find, an independent BF16 N1/C32/K64 3x3/s2 64-cube and FP16 N4/C128/K256 3x3/s2 16-cube both chose solver `ConvHipImplicitGemm3DGroupBwdXdlops` with an XDL-v1 default CK type; cached invocations verified against GPU reference. The tiny 1x1 3D test problem instead chose the naive solver by default, so its selected-CK correctness is not a default speedup claim.

The runtime copy-elision gate is gfx1250-only. The selected-output predicate describes a structural full-E proof, but other devices keep their original 3D staging until their complete-call benefit is measured; existing 2D behavior is unchanged.
Focused BF16 and FP16 selected-plugin GPU regressions both passed with poisoned output/scratch, stride-two holes, changed signed inputs and alias fallback.

A temporary **baseline-only** plugin linked the same CK archive and current plugin objects but disabled the selected-output definition proof; no source-tree or installed plugin was changed. rocprofv3 cached-driver traces recorded four NCDHW staging transposes per call with that baseline versus three with the retained proof, with the same CK dispatch and required clears (ten repeated calls per trace). Uninstrumented cached-driver AB/BA process pairs measured BF16 old/new medians **0.068609 / 0.0585655 ms** and paired median delta **-0.010292 ms** (12/12 new wins); FP16 **0.072397 / 0.0654555 ms**, paired delta **-0.006705 ms** (6/6 new wins). The original NCDHW driver and selected XDL-v1 kernel were held fixed; clocks were not pinned. This is a measured selected-family benefit, not a universal 3D shortcut.

**Rejected extension.** An FP16 packed-forward `GroupsPerWmma=2`, C=K=8 row passed CPU-reference CK execution and factory/scratch tests but lost to the best eligible CK family on G32/N16/56x56 (0.096461 vs 0.056142 ms, 0/5 paired wins) and G32/N32/225x225/s2 (1.302460 vs 0.965956 ms, 0/5). It also lost on a 7x7 small control; the added row, admission branch and tests were removed. The existing C=K=4/GPW4 candidate is unchanged. No BF16 packed row or speculative A-vector4 backward-data row was registered; vector2 already admits an even-K family but lacked default-solver evidence for a wider portfolio.

## Production-hardware corpus investigation (2026-10-21)

**Scope and provenance.** The user identifies the current device as production hardware,
unlike the hardware used for earlier trials. New GPU work was serialized with
`ROCR_VISIBLE_DEVICES=2`: gfx1250, UUID `GPU-6c0381e65d76b84b`, reported ASIC revision 1.
ROCm/compiler prefix was `/home/sgundabo/rocm-10.2`. The starting branch was
`5a3e570c85f450bc541c153123886bd27ef78991`. No clocks, power limits or other devices were
changed. Only MIOpen-consumable BF16, FP16 and FP32 convolutions were investigated.
Earlier measurements remain evidence for their original hardware, not permanent
performance rejections on this device.

### Corpus priorities

The supplied `brad_corpus_9471_5a3e570.xlsx` and matching JSON contain 6,295 FP16,
1,709 BF16 and 1,467 FP32 unique convolutions. Of 9,468 timed rows, 218 selected Naive;
their summed latency is 8.004384 ms out of 1,280.190425 ms, only 0.625%.
Naive selection alone is therefore a poor prioritization metric. The largest Naive
family is FP16 NHWC RGB backward-data, C3/K24, 3x3/stride2, N16/128 at 384/480 square:
four rows total 3.731732 ms. A neighboring N16/256-square probe verified, selected
Naive at 0.077491 ms, and had a best CK raw invocation of 0.111423 ms. Do not ban
Naive or claim missing eligibility when an eligible CK candidate is simply slower.

FP32 dense 3D 3x3x3/stride1 contributes 204.880 ms across 180 direction/shape rows.
The two rows above 10 ms are FP32 160-cube forward and weight-gradient convolutions.
FP16 pointwise and dilated families contribute 145.175 and 127.972 ms; BF16 dense
2D 3x3/stride1 contributes 181.060 ms across all three directions. These are
latency-family targets, not shape-equality admission rules.

Against `brad_corpus_9471_10202026_8a29d2e.json`, successful paired rows with successful
tuning and measurement give aggregate baseline/branch speedups of 1.0244x BF16
(1,698 pairs), 1.2582x FP16 (6,237), and 0.9923x FP32 (1,458). These are historical
two-pass corpus measurements, not controlled new kernel comparisons. Eight branch
measurement rows exceed their recorded Find time by more than 4x; original tuning
logs show independently faster verified execution. For example FP16 G32/C256/K256,
N128/64-square/stride2 is 0.169769 ms in the tuning log but 6.198481 ms in JSON's
measurement pass. Timing state, cached configuration and selection need investigation;
no kernel root cause or corrected corpus speedup is inferred. Six BF16 records have
failed or mismatched tuning/measurement; original logs include hipConv or hipBLASLt
errors before downstream launch failures. They are not evidence that CK caused them.

### Retained: FP32 3D backward-data staging

MIOpen's existing selected no-D XDL-v1 full-E proof now accepts FP32 3D default-layout
problems. Runtime borrowing/copy elision remains gfx1250-only, packed NCDHW, default
alpha1/beta0, aligned/disjoint caller spans and workspace, zero transposed-output pads,
and a recognized bare XDL-v1 kernel ID with implicit split one. Other directions,
explicit splits, fused/unrecognized kernels and aliases retain staging. CK still
clears all packed E before residue GEMMs. No arithmetic, precision, kernel, solver
ranking, scratch allocation or public API changed.

An isolated plugin reused the baseline CK archive and all other plugin objects.
Because shared inline/template symbols can coalesce across replacement and baseline
objects, the temporary predicate and invoker template were uniquely named; a separate
diagnostic established `selected=1, borrowable=1, safe=1, skip=1` before removing its
logging and measuring. Earlier mixed-object samples that did not exercise the modified
branch are not speedup evidence.

| FP32 N1, 3x3x3/pad1, original NCDHW backward-data | Baseline / changed median, ms | Paired process runs |
|---|---:|---:|
| C32/K64, input64-cube, stride1 | 0.3366845 / 0.2595145 | 6 AB/BA pairs, 6/6 wins |
| C32/K64, input48-cube, stride2 | 0.086303 / 0.064384 | 4 AB/BA pairs, 4/4 wins |
| C64/K32, input96-cube, stride1 | 0.935457 / 0.786554 | 4 AB/BA pairs, 4/4 wins |

These are conversion-inclusive driver GPU kernel-time averages (50 iterations on
the first case, 30 on holdouts), not host wall time. Each invocation passed GPU
reference verification. The selected CK type was unchanged across comparisons:
`DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<256, 128, 32, 32, 8, 8, Default, 32, 32, 1, 1, 8, 8, 1, 1>`.
The rebuilt production plugin subsequently verified the same three paths at
0.261119, 0.064402 and 0.784488 ms. Four focused tests passed: the family/split/scalar
predicate and FP16/BF16/FP32 selected-invoker regressions with stride-two holes,
poisoned output/workspace, changed signed data and alias fallback. No full-corpus
post-change speedup, cross-device benefit or all-kernel numerical sweep is claimed.

### FP32 matrix coverage trial: not retained

gfx1250 already has native float-operand/float-accumulator
`v_wmma_f32_16x16x4_f32`; FP32 does not require BF16/FP16 conversion. The ordinary
forward remapping predicate excludes float, rejecting existing public-32x32 dense
rows. A temporary, no-D/unmerged float remap of only the 3D FP32 compute registrar
made four additional configurations executable (15 to 19 supported candidates).
Complete CK hot-reuse invocations included all invoker work, with no cache correction:
new best 0.707165 versus existing 0.289176 ms at C64/K32/input64-cube; new best
0.182415 versus existing 0.160725 ms at C128/K128/input32-cube. No new row beat the
best existing family. The remap was not retained; wider eligibility is not uplift.
Performance runs did not CPU-verify these rejected rows, so no new correctness
contract is asserted.

### Reconsidered earlier trials on production hardware

**FP16 C8/GPW2 packed forward remains rejected.** A temporary registrar added the
previously rejected two-group candidate to the current portfolio, including today's
GPW4/GPW8 alternatives. Raw whole-invocation timing includes weight packing.
At G32/N16/input56-square/3x3/stride1, three AB/BA process pairs measured GPW2
0.0957624 versus best-other 0.0545729 ms. At G32/N32/input225-square/stride2,
GPW2 was 1.2951 versus current GPW4 0.633808 ms. No new correctness claim or
production registration follows.

**BF16 WM1b remains promising, not a global change.** Reconstructed the documented
`NumMacClusters=2` ablation only for block256/M128/N128/K64 in the BF16 2D compute
registrar, preserving priorities, matrix-reuse settings and the rest of the archive.
At G1/N42/C=K128, 3x3/input30x40, five AB/BA pairs gave matched baseline/WM1b
0.0230531/0.0207936 ms (5/5 wins). At C48/K192/input120x160, they gave
0.238616/0.225823 ms (5/5). But current best-other wavelet rows in the full passes
were 0.0188738 and 0.146519 ms respectively. The matched-row improvement is stronger
on this hardware than the old evidence, yet still does not improve the winning
portfolio. Reconsider this scheduling mechanism only on a current winning family
with relevant resource/stall evidence; no blanket macro or matrix-reuse change.
These performance runs did not establish new CPU correctness.

### Artifacts and next targets

The external investigation workbook is
`/home/sgundabo/miopen_benchmarker/ck_convolution_production_hw_investigation.xlsx`.
Raw logs, commands, UUID/commit provenance, family/anomaly CSVs and paired samples
are in `/tmp/ck-corpus-gpu2-hxh4a5_4/`; original benchmark inputs were not modified.
Temporary overlay sources were removed after recording the experiments.

Prioritize BF16 channels-last dense forward/WRW, FP16 NCHW pointwise/dilated
conversion-inclusive paths, and FP32 3D forward/WRW native instruction utilization.
FP16 depthwise partitioned-reduction coverage and FP32 scalar/DL competitors remain
unmeasured candidates, not promised gains. Reconstruct promising prior trials with
one independent mechanism per overlay, compare current best eligible families,
include held-out work-level/stride/channel controls, and reject regressions before
registration. Hardware change removes a historical performance veto, not correctness,
precision, alias, determinism or specification requirements.


## Continued production-hardware exploration after FP32 backward-data commit

The measured FP32 backward-data change, tests and earlier evidence were committed
as `02f67b3cce2` (`miopen: elide redundant gfx1250 FP32 3D backward-data staging`).
Continued work used the same production gfx1250 device and serialized
`ROCR_VISIBLE_DEVICES=2` runs. No unrelated dtype/operator or device was optimized.

### Retained: 3D forward XDL-v3 output-copy elision

The existing selected-output proof now accepts only ordinary unmerged no-D
`DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3` and its `_WmmaPorted` variant
for default-layout 3D forward. Exact type prefixes exclude merged groups,
DirectLoad, Large_Tensor and unknown variants. Only the default PassThrough
forward factory opts into the proof; Scale/Bilinear/fusion retain old-output
staging. The runtime gate remains gfx1250, packed descriptors, alpha1/beta0,
aligned disjoint operands and whole reserved workspace, and zero transposed
output pads. CK arithmetic, selected kernel, precision, input/weight transposes,
final output transpose and workspace size are unchanged.

An isolated replacement plugin with uniquely named trial templates/predicate
was measured before retention. Five AB/BA process pairs at FP32 N1/C64/K32,
3x3x3/pad1/input64-cube/stride1 measured baseline/new 0.366672/0.353648 ms.
Four pairs each measured FP16 N1/C32/K64/input48-cube/stride2 at
0.033694/0.027823 ms and BF16 N1/C64/K64/input40-cube/stride1 at
0.0526295/0.0412185 ms. All selected the same XDL-v3 kernel before/after and
passed GPU-reference verification. Times are conversion-inclusive driver GPU
kernel-time averages, not host wall time.

**Prospective cohort, including unchanged controls.** Nine shapes were fixed
before measurement: each precision FP32/FP16/BF16 at (N,C,K,input cube,stride)
of (1,32,64,32,1), (2,64,32,48,2), and (2,64,64,24,1), all 3x3x3/pad1.
Each baseline did fresh unrestricted Find, followed by three AB/BA cached
process pairs with 30 driver iterations. All nine selected grouped 3D CK;
the two unchanged FP32 controls selected legacy XDL, not XDL-v3.

| Precision, three-shape cohort | Sum of baseline / changed shape medians, ms | Aggregate ratio |
|---|---:|---:|
| FP32 | 0.234190 / 0.229657 | 1.0197x |
| FP16 | 0.121414 / 0.103940 | 1.1681x |
| BF16 | 0.119890 / 0.100613 | 1.1916x |
| All nine | 0.475494 / 0.434210 | 1.0951x |

The unchanged FP32 controls varied -0.9% and approximately 0%; no shape-specific
gate was added to fit them. These results are this prospective cohort's GPU
latency ratio, not a 9,471-shape workload speedup. Every sample verified on GPU.
Seven focused tests passed after retention: existing backward-data invariants,
the extended selected-family/split/scalar predicate, and FP32/FP16/BF16 forward
selected-invoker regressions. Forward tests cover G2 partial 5x7x9 volumes,
3x3x3/pad1, stride1/2, changing signed weights, poisoned output/scratch,
unchanged inputs/weights, repeated invocations and x/y alias fallback against
an independent host convolution. No cross-device or exhaustive numerical
coverage is claimed.

### Actual-wavelet winner trials: not retained

The BF16 DirectLoad wavelet winners use `loop_knm_v1`, not the Interwave
`NumMacClusters` implementation. Importing Interwave workgroup barriers into
math-only waves would violate load/math synchronization; no such change was
attempted. Two independent actual-winner ablations were evaluated instead.

**Redundant-tail loading.** A temporary native load-pipeline clone skipped only
unused odd final A1/B1 loads and terminal coordinate precomputation, retaining
the role barriers and math order. CPU-reference selected-row checks passed
for C8/24/40/48/128 with 3x3, G3, K136 and partial 5x7 outputs (one/two/three/
four/nine reduction blocks). Five AB/BA pairs on the existing S1 BF16
N42/C=K128/input30x40 winner regressed 0.0188585 -> 0.0196863 ms; S2
C48/K192/input120x160 regressed 0.148696 -> 0.153657 ms, 0/5 wins each.
The variant was removed; modeled removed loads are not a latency benefit.

**Transient operand compaction.** A uniquely named clone retained `(k,n,m)`
accumulation order, source offsets, waits and WMMA modifiers while compacting
the A/B staging descriptors to their live current-K/current-N regions.
A selected CPU-reference G3/C40/K136/5x7 control passed. The matched BF16
registrar's 155 kernel metadata records were identical before/after, with zero
resource-field changes. The actual native S1 metadata retained 206/208 VGPRs
depending on specialization, not the source buffer-size reduction. No
performance promotion or register-pressure improvement is claimed; removed.

### Additional scoped findings and stop gates

FP16 NCHW pointwise/dilated probes selected already-copy-elided WMMA-v3 rows;
a larger N8/C512/K1024/input96-square pointwise probe selected hipConv at
0.125972 ms. A forward XDL-v3 copy proof would not improve those winners.

BF16 dense WRW N2/C=K128/3x3/input256-square all-split raw CK enumeration
found 208 configurations; the best was the native block512/M128/N128/K64 row
at requested auto/effective split56, 0.0729589 ms. This is an existing-row
single factory pass, not a new speedup. NCHW old-dW staging proof, FP16
partitioned depthwise reductions, the XDL-v3 wave32 LDS query referencing
GridwiseGemm64, and source-vector narrowing intended for DirectLoad remain
source-backed next experiments. Do not relax Large_Tensor or DirectLoad
hardware safety guards without a supported implementation and measured winner.

New paired logs, commands, cohort samples and compiled-resource comparisons
are in `/tmp/ck-uplift2-gpu2-ca53v5ij/`. The external investigation workbook
records the continued cohort and trial decisions. Trial sources/objects were
removed; original benchmark data and shared ROCm installation were unchanged.


## Dense WRW, FP16 partitioned reduction, longest cases and Naive census

All new GPU work remains serialized on `ROCR_VISIBLE_DEVICES=2`; evidence is in
`/tmp/ck-wrw-gpu2-4nigx8s0/`. The independent read-only static-analysis agent
returned four experiments: actual-wavelet CShuffle batching, one-step LDS
lookahead, reduction-aware WRW split neighborhoods, and mixed-residue BWD
range metadata. These are unmeasured hypotheses, not new speedups. It also
identified process-static current-device occupancy/CU caching without device
identity; that is a separate multi-device defect, not a measured device2 gain.

### Retained: explicit split-one WMMA-v3 old-dW staging

The existing selected-output proof now admits only ordinary packed 2D
FP16/BF16 `DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<...>` at explicit split1.
Runtime copy elision is gfx1250-only, alpha1/beta0, aligned/disjoint packed
operands and entire reserved workspace. Auto, split>1, 3D, FP32, two-stage,
fused and unrecognized families keep staging. CK's own Set/clear behavior,
math, precision, scratch size and final dW conversion are unchanged.

Six AB/BA process pairs on FP16 N2/C=K512, 3x3/dilation4/pad4/input16-square
gave baseline/new 0.036507/0.0288535 ms. Four pairs on N1/C256/K512,
3x3/pad1/input8-square gave 0.025420/0.0208535 ms. Both unrestricted Find
passes selected ordinary WMMA-v3 at explicit split1, with the same kernel
and stored config in each comparison. Auto XDL-v3 and split8 WMMA controls
were unchanged by the proof and showed only small timing variation.
The rebuilt plugin verified the retained paths at 0.028553 and 0.020566 ms.
FP16 and BF16 selected-invoker tests passed against an independent host WRW
reference with G2, partial 5x7 outputs, signed changing dY, dirty dW/scratch,
repeated calls, input/dW alias fallback, and unchanged source tensors.
No BF16 default-selection or full-corpus performance gain is asserted.

### FP16 partitioned depthwise trial: not retained

A temporary dtype-correct FP16 Group16 producer/finalizer clone retained
FP32 partial accumulation and one final FP16 conversion. All-split CK
comparisons on three depthwise families gave:

| G/N/input/stride, 3x3/pad1 | FP16 partitioned / best existing CK, ms |
|---|---:|
| G240/N128/28-square/stride2 | 0.045039 / 0.044502 |
| G672/N128/14-square/stride1 | 0.086018 / 0.093603 |
| G64/N128/128-square/stride2 | 0.306224 / 0.220800 |

These are single all-family raw hot-reuse passes, not paired speedups.
A selected G35/N2/7x9/stride2 CPU-reference control passed. A matching
MIOpen trial plugin selected the partitioned CK type for G672, but unrestricted
Find still chose hipConv (0.118937-ms driver execution). The marginal CK-only
win did not establish default-solver uplift; other work regimes tied/lost.
The trial was not registered in production.

### XDL-v3 LDS/source-vector admission: no uplift demonstrated

A throwaway resource census instantiated 30 existing 2D/3D FP16/BF16/FP32
v2/v5 tuples. Twenty had legal wave32 geometry; every one had equal wave64
and wave32 LDS byte counts. Invalid wave32 tuples were not instantiated.
The wave32 admission query naming GridwiseGemm64 is a source mismatch, but
this census found no newly eligible row or performance benefit. No admission
change was retained. Ordinary source vectors already use the original width:
GridwiseGemmBase passes `DirectLoad ? aligned_width : original_width` for
both operands. Broadening the aligned-width helper would not alter ordinary
convolution code generation. DirectLoad/Large_Tensor safety guards remain.

### Longest-running cases

The top 25 timed rows total 134.957008 ms; two are >4x Find/measurement
timing anomalies rather than established kernel regressions. Original tuning
logs and selected IDs were recorded for every case. The longest genuine
family is FP32 N2/C64/K32/input160-cube/3x3x3: forward, WRW and BWD.
Using the original tuned IDs, baseline/current driver measurements verified:
forward 10.369306/10.063589 ms; WRW 10.356744/10.397013 ms; BWD
6.934646/6.922801 ms. Both comparators already contain the committed BWD
copy elision; these are single executions, not a new paired family claim.

Current CK forward enumeration on that exact largest case found 15 supported
rows; the existing XDL-v3 v2 M128/N16 row was best at 8.739070-ms complete
internal-layout invocation. It was already the original selected type.
WRW full enumeration timed out after 600 seconds; it is not reported as
complete. Selected legal controls on the original small native-float legacy
type gave auto/effective128 9.176040 ms, fixed32 13.298400, fixed64 11.173900,
and fixed128 9.172280. An existing M256/N128/fixed128 tile was 50.472400 ms.
Neither lower split nor larger tile improved this longest family. New work
needs an actual addressing/operand-reuse bottleneck, not another registration
sweep or a split increase beyond the existing numerical cap.

### Complete Naive-winning classification

All 218 Naive winners match between tuning and measurement records, and all
218 original exhaustive tuning logs contain successfully timed non-Naive
competitors. There are **zero Naive-only applicability cases**, **zero
competitor-search-failure-only wins**, and no unresolved cases in this census.
Each row records dtype, direction, family, successful competitor names/times,
measurement and Find latency, errors and timing-anomaly status. Workspace
logs saying `Skipped (no workspace required)` were not treated as
inapplicability: that shortcut does not call IsApplicable.

The largest Naive family remains FP16 NHWC C3/K24 3x3/stride2 BWD at
N16/128 and 384/480-square: four rows sum 3.731732 ms. Naive-winning counts
are 147 forward, 62 backward-data and 9 WRW. The workbook contains the full
classification and longest25 inventory; raw classifications and selected
measurements are JSON/CSV under the evidence directory above. Original
benchmark inputs were unchanged. No blanket Naive exclusion was added.

## Why Naive wins, NCHW layout tax and a rejected FP32 transpose kernel

All runs used `ROCR_VISIBLE_DEVICES=2`; traces used `rocprofv3 --kernel-trace`
with immediate-mode `-S <solver id>` so Find did not pollute steady state.
Evidence is under `/tmp/ck-naive-trace-gpu2-sn_m96vj/`.

### Naive winners

The 218 Naive wins total 8 ms (0.63% of corpus latency). 157 are depthwise
(`C == G`), median work is 5.4 MFLOP and median Naive latency is 11.8 us; the
best CK competitor is a median 3.2x slower than Naive (1.01x-11.45x). Three
mechanisms explain them:

- **Depthwise maps to degenerate GEMMs.** FP16 NCDHW depthwise forward
  (G128, 3x3x3/s2) runs the old WMMA grouped kernel at 117 us versus Naive's
  46 us, before 4-5 transposes. Per-group GEMM N is 1, so a 16-wide WMMA tile
  does one useful column. BF16 1x31 depthwise backward-data: CK kernel 60 us
  plus two transposes and a fill versus Naive 42 us.
- **Fixed NCHW conversion cost on tiny problems.** FP32 C128/K256 3x3 on a
  3x3 image WRW: CK's kernel is 7.6 us, faster than Naive's 25 us, but one
  fill and four transposes add about 21 us. 134 of 218 competitors include
  transposes.
- **Phase decomposition with tiny channel counts.** FP16 NHWC C3/K24 3x3/s2
  backward-data launches four stride-phase kernels of about 83 us each with
  N=C=3; one Naive kernel takes 258 us.

All depthwise convolutions total 21.5 ms (1.68%), so even an ideal
depthwise kernel is bounded near 1-2% corpus uplift.

### NCHW layout tax

NCHW/NCDHW grouped-CK winners are 830.5 ms (65% of corpus). Six selected
FP16 NCHW CK configurations spent 21-54% (median 24%) of measured GPU
kernel time in MIOpen transposes and fills. Extrapolating that share to every
dtype and family is not a measured corpus estimate. The standalone
70-transpose sample gave 3% savings from ideal existing-variant selection;
it does not prove the heuristic is optimal for every consumer invocation.

**Rejected: gfx1250 FP32 float4 64x64 transpose.** An offline microbenchmark
suggested 1.66x on FP32 transposes, but its baseline kernels were compiled
offline and ran 2x slower than MIOpen's runtime-compiled 32x32 kernel. In
the real library both reach ~6.7 TB/s on large tensors. The 18-shape FP32
NCHW/NCDHW FWD/BWD/WRW A/B, with pinned system DB, separate caches,
alternating runs and GPU verification, gave 1.003x aggregate (0.954x-1.061x).
The kernel, heuristic and test change were reverted. Comparisons must use a
relocated baseline library with `MIOPEN_SYSTEM_DB_PATH` set; otherwise MIOpen
loses the system perf DB and reports 7-225x false gains.

Memset is not a lever: `hipMemsetAsync`, `hipMemsetD32Async` and a 16-byte
zero kernel all write at ~4.6 TB/s on this device.

### CK backward-data XDL-v1 clear

`device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp` computes a
coverage condition (split 1, no dilation/stride gaps, stride <= filter) and
then overrides it with `bwd_needs_zero_out = !if_d_is_output_mem` as a
"temporary workaround", so every call memsets E, even at stride 1. XDL-v1
wins 427 backward-data rows (138 ms); rows meeting the coverage condition
would avoid about 5.9 ms (0.46%), mostly FP32. Lifting it requires proving
slice coverage at image edges and updating MIOpen's selected-output proof,
which currently assumes CK's full clear. Not attempted.

### NCHW 1x1 as GEMM

NCHW 1x1/s1/p0 needs no transposes: forward is batched `W x X[n]`.
MIOpen's BLAS 1x1 solvers already do this and are within 1.1-1.4x of
CK-with-transposes for FP16 forward/backward-data (they win 477 FP16
forward rows), but 4-7x slower for WRW. CK-won FP16 NCHW pointwise is
91 ms; a transpose-free CK batched-GEMM path is the remaining structural
option and needs a gating CK batched-GEMM measurement first.

### CDNA5 ISA features not used by the convolution winners

From the CDNA5 ISA (section line numbers in the source document), mapped to
CK source; all unmeasured:

1. Cluster multicast (ISA 5806-5848) of shared operands in the DirectLoad
   wavelet family; wrappers exist in `amd_cluster_load.hpp` and a cluster
   launch overload in `kernel_launch.hpp`, with no callers.
2. Tensor Data Mover (6004-6361) in the wavelet load wave, starting with
   weights and 1x1/s1/p0 activations; the ck_tile TDM path assumes affine
   views and does not express general im2col.
3. Named/split barriers (2101-2243, 6532-6609) so load waves stop waiting on
   `ASYNCcnt(0)` plus full workgroup barriers.
4. L2 prefetch (5753-5772) two K tiles ahead; `UseDataCachePrefetch` is
   hard-coded false in conv_v3.
5. Asynchronous store-from-LDS epilogue and non-temporal output hints
   (1621-1717); all conv buffers use default coherence.

The ISA does not resolve the WMMA matrix-reuse sameness question
(4287-4309, 4760-4763, 8171-8175), so WM2 stays blocked.

## Transpose-free NCHW 1x1 batched GEMM and ISA feature trials (2026-10-21)

Evidence is archived under
`~/miopen_benchmarker/ck_investigation_artifacts_20261021/`.

### NCHW data path

MIOpen hands CK channels-last operands (`NHWGC`/`GKYXC`/`NHWGK`) for NCHW
problems. Its batched transposes perform the global layout conversion;
those CK instances do not repeat it. CK's NGCHW variants can instead own
the conversion internally, but MIOpen does not register those variants.
gfx125 LDS transpose loads rearrange an already-loaded matrix tile for
WMMA, including selected WRW `LdsTranspose` instances. They are not a
second whole-tensor layout conversion. Global transpose loads require full
EXEC and fixed matrix tiles (ISA 5924-5951); CK's wave-tile transfer path is
currently gated to gfx120, while LDS-transpose paths are gfx125-enabled.
A native NCHW path can avoid the conversion read/write, but must preserve
im2col indexing, padding validity and output-store contiguity. Its input
loads may become scalar or strided; performance requires measurement.

### Excluded NCHW 1x1 batched-GEMM experiment

The previous session mistakenly implemented this despite the user's explicit
instruction to avoid it. After reboot the implementation, registrations,
heuristic overrides and tests were removed; sources remain only in the
persistent investigation archive. CK's batched-GEMM LDS-transpose template
extension was also removed. No end-to-end uplift or retained change is claimed
for this excluded experiment.

The separate 1x1 WRW proxy timed FP16 per-image GEMMs only. Its FP32
reduction cost was estimated, not executed; those combined numbers are not
measured convolution speedups. No WRW implementation was retained.

### CDNA5 feature trials

- **Cluster multicast** of weight tiles in the DirectLoad wavelet forward
  family: GPU-verified; all timed shapes within +/-1% (cluster launch alone
  costs 0-1.4%; small shapes -1 to -2%). Rejected.
- **L2 prefetch** one tile ahead (`global_prefetch_b8`, one lane per 128 B
  line, speculative SE scope): wavelet forward -5.5% FP32/FP16 and -13.4%
  BF16 aggregate; XDL-v3-ported forward about -20%. A builtin-based first
  version caused VGPR spills. Rejected; native WMMA-v3 not measured.
- **Tensor Data Mover** weight loads and **split/named barriers** in the
  wavelet load wave: implemented (barriers GPU-verified), not yet timed.
- **Async store-from-LDS / non-temporal output hints**: interrupted.

A `rocprofv3 --pmc` run wedged the amdgpu driver on this node during its
startup power-profile switch, before any kernel launched; do not collect
hardware counters on shared gfx1250 nodes.

## Post-reboot non-pointwise native-layout and ISA measurements

Evidence: `~/miopen_benchmarker/ck_resume_20261021/`. GPU allocation started
with 1/2, then changed to GPU 3 only; every A/B pair stays on one device and
each table records it. No hardware-counter collection was used.

### Native NCHW/NCDHW prototype (not retained)

Extended the existing input descriptor to map original NCHW/NCDHW through
padding and filter/stride/dilation embeds directly into GEMM, with original
KC[Z]YX weights and native output stores. Used native WMMA-v3 for FP16/BF16
and the WMMA-ported XDL path for FP32. The prototype reserves zero CK
workspace and performs no whole-tensor conversion. Grouped, padded,
dilated, strided, odd-tail 2D/3D cases passed reference checks.

Spatial vector loads require a contiguous stride-one, unpadded window and
output spatial extent divisible by the vector width; output vector stores
also require each channel's spatial product divisible by their width.
Otherwise scalar transfers are required. Wider tiles reduce the penalty,
but the prototype loses most measured padded corpus families to the tuned
MIOpen path with conversions.

Five-pair GPU-3 comparisons on four no-padding 3x3/3x3x3 family points per
dtype, including complete MIOpen invocation versus native invocation:

| Precision | 2D sum-of-medians speedup | 3D sum-of-medians speedup |
|---|---:|---:|
| FP32 | 1.363x | 0.934x |
| FP16 | 1.008x | 0.595x |
| BF16 | 1.040x | 0.579x |

Small 2D points individually improved 1.48-1.65x, but larger low-precision
and 3D points regressed. No global cutover, shape-specific registration or
NCHW 1x1 batched-GEMM path is retained. Native candidates would need layout-
aware transfer/tile selection alongside existing kernels, not a forced
replacement merely because they remove transposes.

### Completed rejected ISA mechanisms

- Split/named barriers: all 24 forward winner-cohort shapes verified and
  timed with five alternating pairs on GPU 1. Latency changed +0.30% FP32,
  +31.20% FP16, +18.41% BF16. Epilogue-only early load-wave exit was neutral
  (within 0.03%). Neither retained.
- Native WMMA-v3 L2 prefetch: all 16 FP16/BF16 winner-cohort points verified
  and timed on GPU 3. Aggregate speedups 0.992x FP16, 0.881x BF16. Rejected,
  consistent with the previously rejected wavelet/XDL-v3 experiments.

### Timing and failure attribution

The original profiler's cache-flush helper later faulted in an unmodified
baseline. New experiments use `time_kernel=0 --raw-invocation`, which
brackets complete consumer invocations without cache flushing or timing
subtraction; old cold-cache and new hot-reuse tables are not combined.

A custom WRW assembly-atomic experiment faulted independently: a signed
`readfirstlane` pointer word sign-extended when reconstructing a 64-bit
address with low bit 31 set. Casting each word to uint32 fixes the address
reconstruction, but does not establish a performance gain. The corrected
variant and unmodified builtin both failed reference verification on one
automatic-split shape. No assembly-store optimization is retained.

### Retained TDM weight loads: consumer verification

The retained change uses gfx1250 `TENSOR_LOAD_TO_LDS` for the B-weight tile
in the existing double-buffered DirectLoad wavelet forward pipeline.
Only unmerged packed GK[Z][Y]XC weight views qualify; unsupported LDS
layouts and other architectures keep the previous transfer. One issue wave
loads the tile with TDM padding matching the math-wave LDS descriptor;
`TENSORcnt` is drained before each buffer-publish barrier and on retirement.
Kernel arithmetic, type strings, global tensor layouts and workspace
contracts are unchanged. This does not eliminate MIOpen's transposes.

The compile-time LDS eligibility check validates every element offset with
cached per-row constants, avoiding compiler constexpr-step-limit failures.
The first simplified production transfer failed the new regression and
consumer verification; restoring the independently verified transfer fixed
all six regression cases. The failing-before/passing-after test is retained.

Normal CK flags also exposed a numerical miscompilation with post-RA
scheduling disabled. Controlled registration builds isolated
`-enable-post-misched=0`: restoring post-RA scheduling fixed all seven
wavelet tests; changing the other backend flags did not. The six affected
registration sources receive `-mllvm -enable-post-misched=1` when the build
contains gfx1250. A frontend-only `-Xarch_gfx1250` option routed correctly
in `-###` output but did not fix final AMDGPU code generation; it was removed.
In fat builds, this source-specific scheduling setting also affects the
other architectures compiled into these six registration files. TDM itself
remains gfx1250-gated; no other source files receive the setting.

Final normal-codegen MIOpenDriver A/B, exact selected IDs, five alternating
pairs on GPU 3, baseline/variant verification for all 24 cohort shapes.
Seventeen shape comparisons completed before the second reboot; seven
incomplete comparisons restarted afterward on the same reserved GPU:

| Precision | Baseline sum ms | TDM sum ms | Speedup |
|---|---:|---:|---:|
| FP32 | 19.872103 | 19.725447 | 1.0074x |
| FP16 | 6.267286 | 6.217948 | 1.0079x |
| BF16 | 6.131975 | 5.796328 | 1.0579x |

Every cohort point improved. These are selected-family results, not a
post-change full-corpus or cross-device uplift claim. Larger BF16 cases
improved 5.3-6.7%. The experimental CK-only GPU-1 table separately showed
1.0167x FP32, 1.0158x FP16, 1.1022x BF16; it is not mixed with GPU-3 data.

Evidence: `tdm-miopen/results-final-stock-gpu3.json`,
`tdm-miopen/aggregate-final.json`, `final-ck-regression-source-option.log`,
and `ResumeTdmBarriers/` under the persistent evidence directory.

### Store-hint / async-store decisions

All 24 wavelet-forward cohort points were verified/timed on GPU 3:
non-temporal hints gave 0.9933x FP32, 0.9870x FP16, 0.9979x BF16;
async-store variants were neutral/slower (0.9928-0.9986x). The FP32
backward-data CK-only NT table improved 1.0173x, but a scoped MIOpen
comparison gave 25.779941 -> 25.912318 ms (0.9949x). The scoped template
extension was removed: no output-hint, async-store, or atomic-store patch
is retained.

## Native-output forward integration (2026-10-22)

The retained integration removes only the forward output conversion. CK consumes
packed `NHWGC/GKYXC` or `NDHWGC/GKZYXC` operands and writes packed `NGKHW` or
`NGKDHW` directly through the existing `CTranspose` path. Input and weight
conversions remain in MIOpen. The earlier fully native input prototype and
global-transpose-load experiment are not integrated.

Each dimension/precision factory has six candidates: 64x64x32, 128x128x32, and
128x128x64 tiles, with output vector widths one and four. Their kernel IDs start
with `NativeOutput:`; existing IDs and ranked defaults are unchanged. Find can
select either native-output or converted-output candidates.

Admission requires gfx1250, packed non-pointwise forward convolutions, FP16,
BF16, or exact FP32, per-group C divisible by eight and K divisible by four,
int32-fitting tensor extents, alpha exactly one, and beta exactly zero. Vector
four additionally requires an aligned output spatial extent. TF32, channels-last
caller layouts, fused operations, scale/bilinear cases, and backward operations
do not use these candidates.

The selected solution reserves only the input/weight conversion buffers, using
the existing workspace alignment scheme; the solver-wide workspace query remains
a conservative upper bound for all candidates. Before any writes, invocation
checks exact descriptors, scalar values, pointer alignment, disjoint operand and
workspace spans, and sufficient workspace. Invalid selected invocations fail
rather than silently applying native strides to an incompatible buffer. The
existing converted path remains available for ineligible problems.

Six precision/dimension regression cases exercise every supported native ID,
grouping, padding, stride/dilation, odd spatial tails, poisoned/reused outputs,
changing operands on stable pointers, independent CPU convolution references,
and rejection without writes for invalid descriptors, aliases, scalars, and
workspace. FP32 reference data distinguishes full-precision from TF32 products.
All six passed, as did nine adjacent existing invocation regressions.

Production MIOpenDriver comparisons used actual Find searches of the existing
CK solver with baseline/current plugins, two neighboring shapes per dimension
and precision, GPU-reference verification of both arms, and five alternating
pairs of complete invocations on reserved GPU 3:

| Precision | 2D sum-of-medians speedup | 3D sum-of-medians speedup |
|---|---:|---:|
| FP32 | 1.373x | 1.246x |
| FP16 | 1.449x | 1.149x |
| BF16 | 1.300x | 1.125x |

All twelve current-plugin winners were native-output candidates. This is
selected-family, CK-solver evidence, not a full-corpus or fastest-all-solvers
uplift claim. TDM and other existing candidates are unchanged.

Reproducible source/build commands, selected IDs, raw logs, and paired results
are under `~/miopen_benchmarker/ck_native_output_production_20261022/`;
`paired-results.json` and `paired-summary.json` contain the consumer comparison.

