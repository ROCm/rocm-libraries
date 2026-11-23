# GPU Reference Test Coverage Rules

`hipdnn_gpu_ref_tests` (`tests/gpu-ref/`) checks the GPU reference executor
against the CPU reference (`hipdnn_test_sdk` `CpuFpReference*`). Each case fills
random inputs, runs both references and compares. This page explains which
cases add coverage, which ones only add run time, and why the suite was cut
from 8066 to 3470 cases in the quick and standard tiers (3446 after the cut, plus
24 grouped conv cases with K != C added afterwards).

## What a case costs

Measured on gfx90a (MI210), quick + standard tiers, one process:

| Cost | Size | When |
|---|---|---|
| HipRTC kernel compile | ~0.45 s per kernel (~0.95 s for layernorm bwd, which compiles two) | First use of each unique (source, `-D` defines, function) in a process |
| Kernel launch + CPU reference + compare | 2-10 ms for most cases | Every case |

About 610 compiles accounted for ~365 s of the 407 s baseline. A case that
reuses a compiled kernel is almost free. A case whose parameters change a `-D`
define pays a full compile. ctest runs each tier as a separate process, so the
standard tier recompiles every kernel that the quick tier already compiled.

Parameters that are compile-time defines (each value is a new compile):

| Op | Defines that change per case |
|---|---|
| Layernorm fwd/bwd | dtypes, `OUTER_SIZE`, `INNER_SIZE`, `STRIDE`: every shape is a compile |
| Batchnorm | dtypes, launch geometry (`LOCAL_SIZE_X/Y`), channel-last flag |
| Reduction | dtypes, `MODE`, rank |
| Resample | dtypes, `RESAMPLE_MODE`, `PADDING_MODE`, `HAS_INDEX`, spatial rank |
| Pointwise | dtypes, op |
| Matmul | dtypes, rank |
| Conv | dtypes, spatial rank |

## Rules

A case adds coverage only if it reaches GPU-kernel, launcher or CPU-reference
code that no other case reaches: a different compiled kernel, a different
branch, or a different boundary of a loop or grid (one block vs many,
multiple vs non-multiple of the block size, rank, group count, padding, stride,
broadcast). A case that only changes the values inside a loop bound does not.

### P1: datatype is orthogonal to shape

In every op here the datatype enters only through per-element load and store
conversion (`static_cast` / `toAccum` / `fromAccum`) into a compute type that the
test fixes. Index math, grid sizing and loop bounds do not depend on the
datatype. So:

- Each reduced-precision type (fp16, bf16, fp8, int8, TF32, mixed, upcast,
  downcast) needs to run each compiled kernel variant once, on small shapes.
- Larger shapes and grid boundaries are covered by the fp32 medium-shape run.
  Running the medium shapes again in fp16 and bf16 reaches no new code.

Applied: medium shapes are fp32-only for conv (fwd, dgrad, wgrad),
batchnorm (fwd inference, fwd inference with variance, fwd training),
layernorm, RMSNorm, matmul, pointwise, reduction and resample. Batchnorm
backward was already fp32-only.

For layernorm, where every shape is a compile, the fp16 and bf16 suites use
`getLayernormSmall{4,5}DDtypeTestCases()`. They contain one channel-first case
with mean/rstd outputs and one channel-last case without them, so both
layouts and the mean/rstd output conversion still run in every datatype.
fp32 keeps the full small catalog. Other ops keep the full small catalog for
fp16 and bf16: there the extra cases reuse compiled kernels and cost a few
milliseconds.

### P2: channel-last is stride-only where the code indexes by stride

The conv, reduction and resample kernels, launchers and CPU references index
only through tensor strides. They have no layout branch (grep `gpu-ref/src` and
`gpu-ref/kernels`). A channel-last case reaches the same code as the
channel-first case with different stride values.

Applied:

- Conv: channel-last runs fp32 only, in every tier. Packed fp16 and bf16 cover
  the datatype, and channel-last fp32 covers the layout.
- Reduction and resample parameterized suites: channel-first only. The explicit
  `TestGpuReductionRefChannelLast.*` and
  `TestGpuResample{Fwd,Bwd}RefChannelLast.*` tests stay as the channel-last
  representatives.

P2 does not apply to batchnorm, which has `IS_CHANNEL_LAST_LAYOUT` and a
channel-last launch geometry, to RMSNorm, whose launcher branches on the layout,
or to layernorm, whose channel-last cases change the `STRIDE`, `INNER_SIZE` and
`OUTER_SIZE` defines. All three keep both layouts.

### P3: the same compiled code and the same CPU path

| Cut | Why it is a duplicate |
|---|---|
| Resample `AVGPOOL_*` x `NEG_INF_PAD` (`makeResampleTestCases`) | `PADDING_MODE` is read only for `MAXPOOL` (`GpuRefResampleFwd.cpp:114`; CPU `CpuFpReferenceResampleFwd.hpp:81-84`; the bwd kernel and the bwd CPU reference never read it). It is still a `-D` define, so the average-pool, negative-infinity case compiled a byte-identical second kernel. `MAXPOOL` keeps both modes, and backward keeps both modes for the index-generating forward pass. |
| Resample `OptionalArgs.AverageExcludePadding` / `AvgPoolIncludePadding` (fwd and bwd) | Same mode, padding and dtype as parameterized cases. The parameterized shapes already include asymmetric padding. |
| Layernorm `getLayernormMedium5DTestCases`: 11 shapes cut to 3 | These 5D cases normalize only the last dimension, so `INNER_SIZE` is that dimension (at most 32 here) and the loop `for(i = lid; i < INNER_SIZE; i += LOCAL_SIZE)` (`LOCAL_SIZE=1024`) runs once for every shape; the shapes differ only in `OUTER_SIZE` and, for channel-last with C > 1, `STRIDE = C`. Every distinct shape is a separate compile. Kept: `{32,1,14,14,14}` (C=1, where channel-last is memory-identical to channel-first) and `{32,32,6,6,6}` and `{32,32,6,4,12}` (C=32, channel-last `STRIDE` 32). Each of the 8 dropped shapes was 6 to 12 compiles. |
| Conv medium catalogs | Group count, depthwise, pointwise kernels, kernel size and padding are runtime loop bounds and index math (`cPerGroup = C/groups`, `kPerGroup = K/groups`). The small (smoke) shapes cover G>1 with K == C and with K != C (`GroupedChanMult1d`, `GroupedChanMult2d`, `Grouped2x3d`), depthwise, 1x1 and padding. K != C matters: a kernel that mixes up K and C in the per-group index passes every K == C case. The kept medium shapes are multi-block with stride+pad (`ResNetStem7x7`), dilation+pad (`Dilation2MedScale`), non-cube 3D and the 1D stride, dilation and depthwise cases. |
| SDPA `LargerShape` [2,8,512,128] x 3 dtypes | About 6 s of CPU reference. Its multi-block launch is covered by `RealisticHeadDim`, which has 65536 outputs (256 blocks), and its long softmax reduction by `LongSequence` (Skv=256). The kernel has no branch on Skv or on the grid size. |
| SDPA `MaskDtype` float-mask instances | A float mask is the `Plain` additive-mask test again. `MatchesFloatMaskBitwise<float>` compared a run against itself. The fp16 and bf16 masks, which take a separate `MASK_TYPE` cast, stay. |
| SDPA `ExplicitScaleCausal`, `MixedPrecision.HalfInputsFloatOutputCausal` / `HalfInputsFloatOutputGqa` | The scale is a runtime scalar, and causal masking and GQA do not depend on the datatype. `ExplicitScaleAdditiveMask`, `HalfInputsFloatOutput`, the causal and GQA typed tests, and `MaskCartesian` cover each part. |

## Tier layout

The tiers are in `test_categories_gpu_ref.yaml`, which applies only to this
binary.

| Tier | Contents |
|---|---|
| quick | fp32 smoke: every op and kernel variant (mode, rank, direction) on small shapes in fp32, plus validation, channel-last, optional-argument and broadcast tests. Layernorm 5D fp32 small shapes are in standard (prefix `StandardSmall`) |
| standard | fp32 medium shapes, plus every reduced-precision and mixed-type case on small shapes: instantiation prefix `Standard*`, plus the name patterns in `test_categories_gpu_ref.yaml` (the single list of them). Quick still holds the typed validator suites (`TestGpuFpValidation`, `TestGpuIntValidation`, `TestGpuFpReferenceRmsValidation`, which are cheap) and the two bf16 `TestGpuSdpaFwdProviderContract` tests |
| comprehensive | `Comprehensive*` instantiations: large edge-case shapes, fp32 only (plus batchnorm bwd `ComprehensiveDouble`). Nightly runs `ctest -L comprehensive` |
| full | `Full*` instantiations: large stress shapes, fp32 only. Prerelease runs `ctest -L full` |

A new reduced-precision or mixed-type case belongs in standard. Give a
parameterized suite a `Standard` prefix. A plain `TEST` is placed by its suite
name, so name it with the datatype token (`...Fp16`, `...MixedType`).

## What was kept, and why

- Every fp32 small-shape case that compiles a distinct kernel: each reduction
  mode, each pointwise op, each resample mode, each rank, each conv direction.
  These compiles are most of the quick-tier time, and each one is real
  coverage.
- All mixed-type, upcast and downcast `TEST`s. Each is a distinct conversion
  tuple.
- All `ThrowsOn*` validation tests. They cost no compile and each reaches a
  distinct host-side check.
- Batchnorm fwd inference and fwd inference with variance. They use different
  kernel entry points (`BatchnormFwdInfRef` vs `BatchnormFwdInfWithVarRef`).
- Batchnorm bwd `ComprehensiveDouble`: the only double-precision batchnorm case.
- Conv and batchnorm `Full` blocks: they use their own large "stress" catalogs
  (`getLargeStress*`, `getBatchnormLargeStress*`), not copies of other tiers.

## Comprehensive and Full

Before this change no ctest category selected the `Comprehensive*` and `Full*`
instantiations: the `comprehensive` and `full` categories of
`test_categories.yaml` matched only the lowercase golden-data names, so the
nightly `ctest -L comprehensive` ran a gpu-ref comprehensive suite of zero
cases. `test_categories_gpu_ref.yaml` now selects them in the `comprehensive`
(nightly) and `full` (prerelease) categories. Before enabling them, the same
rules cut them down:

| Cut | Why |
|---|---|
| Every fp16, bf16, mixed and upcast instantiation in both tiers | P1. The quick/standard small-shape runs cover each datatype's conversion; large shapes add nothing datatype-specific. This was about two thirds of the nightly time. |
| `Full` blocks of layernorm, RMSNorm and pointwise | Each was a lambda that concatenated the small, medium and large catalogs, so it re-ran every quick, standard and comprehensive case. |
| `getLayernormLarge5DTestCases`: 18 shapes cut to 3 | As in the medium catalog, only the last dimension is normalized, so every shape is the same single-iteration loop and differs only in `OUTER_SIZE` and `STRIDE` (P3); each is a separate compile. Kept: `{32,2,32,57,125}`, `{1,3,16,240,320}` and `{256,1,32,32,32}` (C=1). |

Remaining candidates, not cut yet: the matmul large catalog (8 entries; only
K=1, non-multiple multi-tile and rank-5-with-tails look distinct) and the
reduction large catalog (10 entries; only the exact-multiple multi-iteration
domain `{4,512,4,8}->{4,1,1,1}` looks distinct). Together they cost about 15 s
per nightly run in fp32.

## Accepted risk

P1 removes fp16 and bf16 runs at medium sizes. Those runs also showed whether
the per-op tolerance still holds after long reductions in reduced precision.
That is a property of the tolerance, not of the reference code, because both
references accumulate in the same compute type. If a tolerance needs checking
at scale, add one fp16 case for that purpose and say so in a comment. Do not
restore the whole matrix.

## Measured effect

Measured on gfx90a (MI210) with `ctest -R "hipdnn_gpu_ref_tests_(quick|standard)_suite"`.
Each pair of columns comes from one node and one build tree.

Cut (MI210 node A):

| Tier | Cases before | Cases after | ctest time before | ctest time after |
|---|---|---|---|---|
| quick | 4412 | 2847 | 278.6 s | 239.7 s |
| standard | 3654 | 599 | 258.4 s | 57.2 s |
| **total** | **8066** | **3446** | **537.0 s** | **297.0 s** |

Tier move (MI210 node B). The same 3446 cases, with
reduced-precision and mixed-type cases moved to standard:

| Tier | Cases before | Cases after | ctest time before | ctest time after |
|---|---|---|---|---|
| quick | 2847 | 1330 | 254.7 s | 90.8 s |
| standard | 599 | 2116 | 68.1 s | 230.7 s |
| **total** | **3446** | **3446** | **322.8 s** | **321.5 s** |

Comprehensive and Full (MI210 node B,
`hipdnn_gpu_ref_tests --gtest_filter="Comprehensive*:Full*"`, one process). The
"before" column already excludes the conv channel-last fp16/bf16 classes
removed in the first cut:

| | Cases | Wall time |
|---|---|---|
| before | 4224 | 17 min 29 s |
| after | 762 | 3 min 25 s |

These tables predate the `GroupedChanMult1d` and `GroupedChanMult2d` cases (12 quick,
12 standard). At the current head on gfx90a: quick 1342 (89.7 s), standard 2128
(230.0 s), comprehensive 570 (152.6 s), full 192 (55.3 s).

All runs passed. The cut saves time mostly in standard, because most of the
remaining time is distinct compiles. The reduction and resample layout cuts
halve their case counts but save almost no time, because those cases reused
compiled kernels. The tier move changes no total: it moves ~160 s of compiles
from quick to standard.

## Further savings that need code changes, not cuts

- **Layernorm shape defines.** `OUTER_SIZE`, `INNER_SIZE` and `STRIDE` are
  passed as `-D` defines. Passing them as kernel arguments would let one compile
  serve every shape. Layernorm is still the largest compile cost in both tiers.
- **Cross-process compile cache.** The standard tier recompiles every kernel
  that the quick tier already compiled. Setting `AMD_COMGR_CACHE_DIR` for these
  ctest entries would share the compiled kernels between the tiers and the
  `ffm-quick` lane.
