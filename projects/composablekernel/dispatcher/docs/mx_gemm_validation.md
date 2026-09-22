# Native MxGemmKernel validation on gfx1250

All five `MxGemmKernel` pipelines passed the September 21, 2026 validation through
the dispatcher bridge and old Tile Engine. These results cover RCR, FP4 E2M1
and FP8 OCP E4M3 inputs, E8M0 scales per 32 K elements, FP32 accumulation and
FP16 output. Each pipeline uses its own native implementation.

## Optional gfx1250 warp tiles

The shared Tile Engine validator now exposes 32x32x128 for FP4/FP8 and
32x16x128 for FP4 with TDM V1/V2. The bridge no longer imposes a second,
fixed-16x16 gate. Defaults remain 16x16x128: all 128 gfx1250 default headers,
344 gfx950 default headers and 16 gfx1250 CI headers are unchanged.

The 32x32 matrix uses 2x2x1 warps and block tiles 64x64x128, 128x128x128,
128x256x256, 256x128x256, 192x256x128 and 256x256x256, with both input types:

| Pipeline | Configurations | Bridge builds | Tile Engine builds | Bridge reference comparisons | Tile Engine reference comparisons |
| --- | ---: | ---: | ---: | ---: | ---: |
| `comp_tdm` | 12 | 12/12 | 12/12 | 480/480 | 144/144 |
| `comp_tdm_v2` | 12 | 12/12 | 12/12 | 480/480 | 144/144 |
| **Total** | **24** | **24/24** | **24/24** | **960/960** | **288/288** |

All **48 bridge rejection checks** passed. The matrix uses the independent
CPU decoding, varied E8M0 scales, raw input codes, partial tiles, K-loop cases
and repeated launches described below. Maximum normalized bridge error was
**0.000949956244**, below the unchanged 0.05 threshold. All **5,662 recorded
C++ source/header hashes** and **24 generated-header pairs** match the final
source. These additions bring the three matrices to **216 configurations**
and **11,136 numerical comparisons**.

The updated checked-in GPU suite passed **492 comparisons across 30
configurations**, covering all five pipelines and both input types. Four
configurations exercise the optional 32x32 TDM warp tile.

The latest focused CPU regression passed **225 tests and 965 subtests**,
including **49 MX bridge tests**, with five local compiler-dependent skips.
The new tests verify Tile Engine enumeration and exact bridge/Tile Engine
header parity for both optional shapes, FP4-only selection for 32x16, and
architecture/pipeline rejection. Ruff, full-tree clang-format **18.1.3**, ASCII,
CRLF and whitespace checks passed. No C++ kernel source changed for this update.

FP4 32x16 is exposed without Python A0/B0 revision checks, but it is not a
working MX kernel in the current native stack. Both TDM V1/V2 build probes,
through both host paths, fail with `no matching function for call to
'wmma_intrinsic'`: the current 32x16 FP4 trait implements only the unscaled
operation. Instruction support is left to native compilation as requested;
this shape is excluded from default/CI sweeps and the GPU numerical suite.
This compile failure does not establish A0/B0 hardware instruction support.

## LDS capacity review correction

PR #12173 corrected the dispatcher's `ArchFilter` capacity table. MX GEMM uses
Tile Engine's separate `gemm_validation_utils.py`, which still lacked gfx1250.
An override covered the three CShuffle pipelines, but TDM V1/V2 remained on the
64 KiB fallback. The shared Tile Engine table now records gfx1250's 320 KiB
capacity, and both staging and CShuffle epilogue checks use that table.

The TDM staging calculation includes descriptor padding and both buffers.
For example, an FP8 256x352x256 tile needs 311296 raw staging bytes, but 330688
bytes with padding, so it is rejected. The neighboring 256x320x256 tile needs
313280 bytes including padding and is accepted. Native device compilation
verified **136 allocation assertions**: all 128 default pipeline/datatype/tile
combinations and eight boundary cases match `GetSmemSize()` exactly.

The corrected default enumeration contains **128 TDM configurations** instead
of 58. The 70 additional configurations passed the same bridge and Tile Engine
matrix described below:

| Pipeline | Additional configurations | Bridge builds | Tile Engine builds | Bridge reference comparisons | Tile Engine reference comparisons |
| --- | ---: | ---: | ---: | ---: | ---: |
| `comp_tdm` | 35 | 35/35 | 35/35 | 1400/1400 | 420/420 |
| `comp_tdm_v2` | 35 | 35/35 | 35/35 | 1400/1400 | 420/420 |
| **Total** | **70** | **70/70** | **70/70** | **2800/2800** | **840/840** |

All **140 additional bridge rejection checks** passed. The local CPU regression
passed **293 tests and 641 subtests**; the five compiler-dependent skips passed
in the remote **21/21 LDS suite**. The MX CPU suite passed **46/46** remotely.
Regression coverage checks capacity consistency between Tile Engine and the
dispatcher, all five MX pipelines, architecture suffixes, unknown-target
fallbacks, and padding at the capacity boundary.

The updated checked-in GPU suite passed **428 comparisons across 26
configurations**, including four 256x256x256 TDM configurations whose LDS
allocation exceeds 64 KiB. It covers all five native pipelines with both input
types, varied scales, partial tiles, K-loop cases, and repeated eight-wave launches.

The 16 gfx1250 CI kernels and 344 gfx950 default kernels are unchanged. All
122 previously validated gfx1250 generated headers and 16 sampled non-MX
headers are unchanged, and this correction changes no C++ kernel source.
Together, the two matrices cover **192 configurations** and **9888 numerical
comparisons** through both host paths.

## Original native pipeline matrix

This matrix was run at `d67e7e80e7`, before the LDS capacity correction above.

| Pipeline | Configurations | Bridge builds | Tile Engine builds | Bridge reference comparisons | Tile Engine reference comparisons |
| --- | ---: | ---: | ---: | ---: | ---: |
| `comp_tdm` | 29 | 29/29 | 29/29 | 1160/1160 | 348/348 |
| `comp_tdm_v2` | 29 | 29/29 | 29/29 | 1160/1160 | 348/348 |
| `comp_async` | 36 | 36/36 | 36/36 | 1440/1440 | 432/432 |
| `comp_async_eight_waves` | 16 | 16/16 | 16/16 | 640/640 | 192/192 |
| `weight_preshuffle` | 12 | 12/12 | 12/12 | 384/384 | 144/144 |
| **Total** | **122** | **122/122** | **122/122** | **4784/4784** | **1464/1464** |

The matrix contains the then-enumerated 58 gfx1250 default configurations plus
64 explicit CShuffle configurations:

- TDM V1/V2: the `default_config_gfx1250.json` enumeration before the LDS correction.
- Async: block M/N in {64, 128, 256}, K in {128, 256}, for both input types.
- Eight-wave async: block M/N in {128, 256}, K in {128, 256}, for both types.
- Weight preshuffle: M in {32, 64, 128}, K=256; FP4 N=512 and
  FP8 N in {128, 256, 512}.

All use a 16x16x128 warp tile. Async/TDM use 2x2x1 warps, eight-wave async
uses 4x2x1, and weight preshuffle uses 1x4x1.

The bridge comparisons independently decode the actual input bytes on the CPU,
apply independently varied per-row/per-K-block scales, and compare against a
NumPy FP32 GEMM rounded to FP16. They cover partial tiles, raw input codes, two
seeds and K-loop lengths 1, 2, 3, 4, 5, 7 and 8. Weight preshuffle uses N sizes
aligned to 16 and K-loop lengths 1, 2, 3, 4, 5 and 8. Selected shapes are repeated
five times to check deterministic buffer reuse. Tile Engine runs three
block-relative shapes and a 512x512x2048 allocation-boundary case, each with
three initialization modes.

The initial sweep exposed speculative async reads past the physical input
buffer in 13 FP8 configurations at 512x512x2048. The gfx1250 global-to-LDS
path now checks buffer bounds explicitly. The complete rerun reported here
includes that fix; failed cases from the initial run are retained separately.

The maximum bridge normalized error was **0.000956772943**, below the unchanged
0.05 threshold. The denominator is `abs(reference) + max(max_abs(reference)*0.01,
1e-6)`. All **256 bridge** and
**256 Tile Engine rejection checks passed**, covering
unsupported K, split-K, and weight-preshuffle N alignment.

## Original implementation regression checks

- The checked-in GPU regression suite passed **364 numerical comparisons** across
  22 gfx1250 configurations: 16 TDM CI configurations and six native CShuffle
  pipeline/datatype configurations. It also checks input rejection and repeats
  every eight-wave case five times.
- The new native C++ split-K argument-validation target passed **2/2 tests**.
  Scalar FP16 CShuffle accepts ordinary GEMM and rejects split-K; paired output
  vectors retain native split-K argument support.
- An additional FP4 check at 512x512x4096 passed **12/12 comparisons** across
  the three CShuffle pipelines. Both packed inputs occupy 1 MiB, checking the
  physical buffer boundary with FP4's two-elements-per-byte storage as well.
- CPU regression: **254 tests and 913 subtests passed** locally. The five local
  compiler-dependent skips were exercised remotely in the **21/21 passing LDS
  suite**.
- gfx950: all six pipeline/datatype builds passed, with **48/48** independent
  reference comparisons and **12/12** rejection checks using the final native
  source changes.
- All 344 gfx950 default generated headers, 58 gfx1250 default headers and
  16 gfx1250 CI headers match their previously validated versions. All 16 sampled
  non-MX generated headers are unchanged (ordinary, batched, grouped and
  preshuffle GEMM; FP16/BF16; gfx950/gfx1250).
- All 5,662 C++ source/header hashes captured
  by the final sweep match the final source. Bridge and Tile Engine generated
  headers match for all 122 configurations. Ruff, clang-format 14 on changed
  C++ regions, and whitespace checks passed.

## Environments and scope

The gfx1250 tests used ROCm 10.0.0a20260729 / HIP 7.15.26306. gfx950 used
ROCm 7.2.1. The gfx1250 runtime emitted rocjitsu translation warnings; these
results make no performance claim.

The 216 configurations across the three matrices are a finite set, not every possible
native configuration. MXFlatMM, other layouts/output types, cluster launch,
and a full rocm-libraries build are outside this validation. The gfx1250 bridge
rejects split-K, K padding and persistent execution. Weight preshuffle requires
problem N divisible by 16.

The architecture default JSON now enumerates 128 TDM configurations;
all five pipelines are available through explicit selection. See the
[MX bridge guide](mx_gemm.md) for pipeline helpers, input formats and commands
to run the checked-in CPU, GPU and native regression tests.
