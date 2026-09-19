# Changelog for hipBLASLt

Full documentation for hipBLASLt is available at [rocm.docs.amd.com/projects/hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html).

## hipBLASLt 1.5.0 for ROCm 10.1.0

### Added

* `FusedGemmA2A` TensileLite problem-type parameter (default `0`, off) that fuses an all-to-all redistribution into the GEMM store path using SDMA, avoiding a separate collective kernel and staging buffer; currently limited to gfx950 and bf16.
* Tensor swizzling (pre-swizzled/pre-tiled A/B tensors) support for gfx11 (WMMA) architectures.
* Batch-offset support for General Batched GEMM on gfx1250.
* gfx1250 v0 ASIC revision support, modeled as a separate `gfx1250v0` architecture identity (selectable via `--architecture=gfx1250v0`, `--gpu-targets gfx1250v0`, or `GlobalParameters.Architecture: gfx1250v0`) that shares ISA `{12,5,0}` and compiler target `gfx1250` but disables TDM multicast and the FP4 32x16 WMMA path, with a dedicated tuned GEMM library subtree selected at runtime by device `asicRevision`; use `install.sh --asic-revision <v0|v1>` or the `HIPBLASLT_ASIC_REVISION` CMake cache variable to build a specific revision, or leave it unset to build both.
* `gbps-bandwidth` (`GbpsBW`) result key in `tensilelite-client` for logging approximate R/W bandwidth in GB/s during performance reporting.
* Complex CGEMM/ZGEMM support for gfx1250.
* `TENSILE_FIXED_WGMXCCSPLITK` environment variable to override the split-K work-group XCC mapping factor for StreamK GEMMs.
* `HIPBLASLT_MATRIX_LAYOUT_OFFSET` matrix-layout attribute for 64-bit element offsets into sub-matrices in General Batched GEMM (`batch_mode=1`), along with `hipblaslt-bench` `batch_offset_a/b/c/d` arguments; nonzero offsets require `HIPBLASLT_BATCH_MODE_POINTER_ARRAY` and are rejected for sub-byte MX types (`HIP_R_6F_E2M3`, `HIP_R_6F_E3M2`, `HIP_R_4F_E2M1`) with `HIPBLAS_STATUS_NOT_SUPPORTED`.

### Changed

* `--global-parameters` and `--benchmark-parameters` values are now parsed as Python literals via `ast.literal_eval` instead of `eval`, correctly handling values containing `=` and rejecting non-literal expressions with an `argparse.ArgumentTypeError`.
* `HIPBLASLT_TENSILE_LIBPATH` and `HIPBLASLT_EXT_OP_LIBRARY_PATH` are now ignored when the process runs in a secure execution context (set-uid/set-gid or other credential-changing exec), falling back to the default library location with a diagnostic; behavior is unchanged for non-privileged processes.
* Enabled gfx1250 cluster-launch kernels for GEMM sizes whose work-group count is not a multiple of `ClusterDim` by padding the launch grid up to a `ClusterDim` multiple and early-exiting the padded work-groups, removing the `ClusterDimCheck` predicate that previously rejected these sizes.
* Stream-K flags are now per-stream: a handle reserves an extra fixed 8 MiB at creation and serves at most 64 distinct streams for Stream-K matmuls (claimed on a stream's first use, held until the handle is destroyed); beyond that the matmul returns `HIPBLAS_STATUS_INTERNAL_ERROR`.
* Stream-K workspace size reported by the heuristic APIs is now smaller, and the SK grid is bounded, so `TENSILE_STREAMK_GRID_MULTIPLIER` values past that bound no longer take effect.
* Solution cache key now includes `HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET` and the StreamK tile scheduling mode, so the same problem can select a different kernel than before.

### Removed

* Removed the OpenCL runtime backend from TensileLite: `RuntimeLanguage: OCL` (`--runtime-language OCL`), the `-p`/`--platform` option and `Platform` global parameter, and the `platform-idx` client option are no longer accepted, leaving only `HIP` and `HSA`.

### Optimized

* Improved gfx950 GEMM performance by updating Origami solution libraries with work-stealing support and additional tuned TF32 and MX kernels.
* Improved split-K GEMM performance with a K-first work-group reordering (K-Coherent) approach that increases L2 cache reuse across K-slices.

### Resolved issues

* Fixed incorrect results (`beta` applied twice) for `AdaptiveGemmGSUA` GEMMs that resolve to MultipleBuffer accumulation with a non-zero `beta`.
* Fixed out-of-bounds tensor loads in the single-wave TDM kernel for edge (non-tile-aligned) `M`/`N` sizes on gfx1250, which could produce incorrect results.
* Fixed a Stream-K flag-region overrun on dynamic-queue paths (`StreamK=4` and the SK4 sub-path of `StreamK=5`) where a grid scaled via `TENSILE_STREAMK_GRID_MULTIPLIER` could write past its own region.
* Fixed a deadlock in the Stream-K remainder path when concurrent GEMMs ran on multiple streams sharing one handle, where the process hung with the GPU at 100% and no HIP error reported.
* Fixed a race condition in StreamK (SK4/SK5) kernels where untokened mailbox `ds_store`/`ds_load` operations could overlap with LDS0 traffic, and incorrect results in SK5 hybrid StreamK caused by AND-masking the live `MagicShiftItersPerTile` register that corrupted the aliased `SKTiles` overlay.
* Fixed intermittent nonfinite (`NaN`/`Inf`) values in `hipblasLtMatmul` output on gfx950 caused by the first MFMA instruction executing before pre-loop local-data-share reads completed, affecting kernels using custom main-loop scheduling with forced unroll subiterations.
* Fixed a missing wait for VALU-to-global-atomic/store read-after-write dependencies under expert scheduling mode 2 on gfx1250, which could leave the StreamK dynamic work-queue counter dirty and produce nondeterministic incorrect results.
* Restored the StinkyTofu ESM2 scheduling path for sparse SpMM problem types on gfx1250 after resolving the intermittent correctness failures that previously required disabling it.
* Fixed a cross-wave read-after-write race in triple LDS buffering (`TDMPlusLdsBuf`) that could produce incorrect results; triple buffering now falls back to double buffering.
* Fixed incorrect results for general-batched (pointer-array) GEMM on gfx1250 in the TDM path, where A/B batch pointers were not dereferenced before tile-offset arithmetic.
* Fixed out-of-bounds GSU synchronizer pointer arithmetic for grouped GEMMs with more than 16 problems, which handed later problems a pointer past the end of the synchronizer allocation on gfx942 and gfx90a.
* Fixed nondeterministic wrong results on gfx1250 caused by read-token races in the `TDMSplit` load path, which is now disabled (rejected during solution selection) until a complete fix lands.
* Fixed out-of-bounds global memory reads in TDM iterate mode (`iterate_enable`) when a workgroup processed a partial tile.
* Fixed incorrect kernel selection for very large `K` problems where Dot2 kernels were mistakenly parameterized as Stream-K kernels.
* Fixed a `branch size exceeds simm16` build failure caused by replay hazard protection enlarging the loop body, by emitting a 32-bit branch sequence for the backward branch.
* Fixed an SGPR budget overflow in gfx1250 StreamK GEMM kernels using wave-separated TDM stagger (for example `PrefetchGL2=1` with `StreamKForceDPOnly=0`) that caused affected kernels to write nothing and produce incorrect results for MX-FP8 and MX-FP4 problems.
* Fixed an `hipErrorIllegalAddress` crash on gfx1201 (RDNA4) for GEMMs using DirectToVgpr transpose loads (`global_load_tr`) when the free dimension is not a multiple of the macro-tile, caused by an out-of-bounds read past the operand.
* Fixed an intermittent `SIGBUS` crash in the `tensilelite-client` hardware-monitoring thread on gfx1151 caused by out-of-bounds indexing of `amdsmi_frequencies_t::frequency[]` when a clock domain is power-gated.
* Fixed silent build failures when the assembler binary is missing or not executable; `rocisa` now raises a `RuntimeError` naming the assembler path instead of returning an empty capability map.
* Fixed incorrect results for XFP32 (`F32XdlMathOp: X`) GEMM on gfx1250 where the transpose could overwrite local-read data before it landed, along with an SGPR double-checkout error affecting XFP32 emulation kernels.
* Fixed a possible page fault or silent data corruption on gfx1250 caused by an xnack replay hazard in multi-dword SMEM loads (`SLoadB64`) where the destination and base address registers overlapped.
* Fixed a hang and stale partial reads on gfx1250 in the StreamK dynamic and hybrid fixup paths caused by a missing acquire fence on the partial-tile flag read.
* Corrected `PhysicalMaxVgprCU` for gfx1102 and gfx1103, where gfx1103 was reported with a 1536-VGPR per-SIMD file instead of 1024 and gfx1102 omitted the two-SIMDs-per-CU factor, both affecting occupancy and kernel selection.
* Fixed a StreamK per-XCD work-queue counter that failed to reset between launches when the StreamK grid size was not a multiple of the XCD count, causing progressively slower execution on repeated GEMM launches (`WorkGroupMappingXCC == -1` and `StreamKXCCMapping` chiplet-remap paths).
* Fixed out-of-bounds stores in subtile GEMM kernels on gfx950 and gfx1250 when the `M` dimension does not evenly fill the macro tile (for example `M=8` with a 32-row tile).

## hipBLASLt 1.4.1 for ROCm 7.14

### Added

* Introduced a new API: hipBLASLt-ext::isSolutionSupported(). This API is used by new hipBLASLt integration from rocBLAS to check if a given solution is supported for a certain GPU and Problem Type. 
* `HIPBLASLT_CHECK_SYNCHRONIZER` environment variable: opt-in post-launch
  dirty-buffer check for the handle's inter-workgroup flag buffers, which their
  kernels must leave at zero (`1`, `on` or `true` to enable; anything else
  disables). It scans the whole `Synchronizer` buffer used by GSU
  MultipleBufferSingleKernel and amaxD, and the Stream-K flag block bound to the
  launch's stream. Covers `rocblaslt_matmul_impl` only. Unlike
  `HIPBLASLT_CHECK_NUMERICS`, each covered call pays a stream sync and a
  device-to-host copy, so this is single-threaded debugging use only, not for
  concurrent-stream workloads.
* `HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT`,
  `hipblaslt_ext::GemmPreference::setUniformSummationOrder`, and
  `hipblasLtSetUniformSummationOrder` / `hipblasLtGetUniformSummationOrder`
  opt into a uniform summation order across `M` (not run-to-run determinism).
  See `hipblaslt.h`. `hipblaslt-bench --uniform_summation_order` forwards
  the descriptor attribute.

## hipBLASLt 1.4.0

### Added

* Complex datatype support for gfx942 and gfx950.
* `hipblasLtSetSmCountTarget()` / `hipblasLtGetSmCountTarget()` handle-level
  helpers (the analogue of cuBLAS's `cublasSetSmCountTarget` /
  `cublasGetSmCountTarget`). `int32_t`, default `0` meaning "use all
  compute units"; negative values are rejected with
  `HIPBLAS_STATUS_INVALID_VALUE`.
* `HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET` matmul-descriptor attribute and
  `HIPBLASLT_MATMUL_PREF_SM_COUNT_TARGET` preference attribute (same
  semantics). When a per-matmul / per-preference value is non-zero it
  takes precedence over the handle-level value. Lets callers convey an
  estimate of how many CUs hipBLASLt should target for kernel selection
  and persistent-grid sizing — useful when another kernel (e.g. RCCL)
  is co-running on the device or when a persistent grid should be sized
  for a known CU budget. (This is a hint, not a CU reservation.)
* `HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT` and matching ext API
  (`setStreamKTileSchedulingMode()` / `getStreamKTileSchedulingMode()`) accept
  the tri-state `hipblasLtStreamKTileSchedulingMode_t` enum (OFF=0 static SK3,
  ON=1 dynamic SK4, AUTO=2 origami heuristic). Invalid values return
  `HIPBLAS_STATUS_INVALID_VALUE`. `hipblaslt-bench` `--streamk_tile_scheduling`
  and `--sm_count_target` forward into the matmul descriptor
  (see `clients/bench/README.md`).

### Changed

* StreamK=5 tile scheduling default is now `OFF` (`0`, static SK3 sub-path)
  instead of `AUTO` (`2`). Set `HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET` (or
  `hipblasLtSetSmCountTarget`) to a positive value to engage the origami
  hybrid-mode heuristic per launch even when the mode is `OFF`; use
  `HIPBLASLT_STREAMK_TILE_SCHEDULING_AUTO` (`2`) to always delegate to the
  heuristic regardless of `sm_count_target`.

## hipBLASLt 1.3.0

### Added

* General Batched GEMM support.
* `HIPBLASLT_CHECK_NUMERICS` environment variable: opt-in post-GEMM NaN
  scanner for `hipblasLtMatmul` output (D matrix). Accepts numeric
  (`1`, `2`) or word values (`info`, `warn`, `none`/`off`). Output goes
  to the standard hipBLASLt log sink (or `stderr` if no log sink is
  configured). Per-call cost is one scanner kernel launch only -- no
  `hipStreamSynchronize`, no per-call alloc/free. A persistent 4-byte
  device flag is allocated once in the handle constructor and drained
  once at handle destruction with a single `hipDeviceSynchronize`;
  a teardown log line reports the first matmul call_id at which NaN
  was observed (or that none was, with a sampling caveat when the
  scanner only ran on a subset of calls). Companion env vars allow
  sampling (`HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY`) and a bisect window
  (`HIPBLASLT_CHECK_NUMERICS_SCAN_FROM` / `_SCAN_UNTIL`). A C API
  `hipblasLtCheckNumericsDrain(handle, &first_nan_call_id)` lets
  frameworks drive a drain on demand.

### Changed

* Replaced `install.sh` with an invoke-based task runner (`tasks.py`) to support cross-platform builds including Windows (ROCm 7.0+).
* gtest and msgpack-cxx are now fetched automatically via CMake FetchContent if not found on the system.
* Greatly improved MXFP4 GEMM performance when using HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT

## hipBLASLt 1.2.2 for ROCm 7.2.1

### Added

* Support for AMD SMI.

### Changed

* Migrate `HIPBLASLT_ENABLE_LLVM` to `HIPBLASLT_ENABLE_YAML` and synchronize with tensilelite's build library format.

### Deprecation

* ROCm SMI is deprecated and dependencies are removed.

## hipBLASLt 1.2.1 for ROCm 7.2.1

### Resolved issues

* Fix issue where users might encounter a `HIPBLAS_STATUS_INTERNAL_ERROR` with various sizes in CPX mode.

## hipBLASLt 1.2.0 for ROCm 7.2.0

### Added

* Support for 'BF16' input with 'FP32' output data type for gfx90a.
* Support for hipBLASLtExt operation APIs on gfx11XX and gfx12XX.
* Added ``HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT`` to support pre-swizzled block scaling data.

## hipBLASLt 1.1.0 for ROCm 7.1.0

### Added

* Fused Clamp GEMM for ``HIPBLASLT_EPILOGUE_CLAMP_EXT`` and ``HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT``. This feature requires the minimum (``HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT``) and maximum (``HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT``) to be set.
* Support for ReLU/Clamp activation functions with auxiliary output for the `f16` and `bf16` data types for gfx942 to capture intermediate results. This feature is enabled for ``HIPBLASLT_EPILOGUE_RELU_AUX``, ``HIPBLASLT_EPILOGUE_RELU_AUX_BIAS``, ``HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT``, and ``HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT``.
* Support for `HIPBLAS_COMPUTE_32F_FAST_16BF` for FP32 data type for gfx950 only.
* Added the CPP extension APIs ``setMaxWorkspaceBytes`` and ``getMaxWorkspaceBytes``.
* Added the ability to print logs (using ``HIPBLASLT_LOG_MASK=32``) for Grouped GEMM.
* Support for swizzleA by using the hipblaslt-ext cpp API.
* Support for hipBLASLt extop for gfx11xx and gfx12xx.

### Changed

* ``hipblasLtMatmul()`` now returns an error when the workspace size is insufficient, rather than causing a segmentation fault.

### Resolved issues

* Fix incorrect results when using ldd and ldc with some solutions

## hipBLASLt 1.0.0 for ROCm 7.0.0

### Added

* Stream-K GEMM support has been enabled for the `FP32`, `FP16`, `BF16`, `FP8`, and `BF8` data types on the MI300A APU. To activate this feature, set the `TENSILE_SOLUTION_SELECTION_METHOD` environment variable to `2`, for example, `export TENSILE_SOLUTION_SELECTION_METHOD=2`.
* Fused Swish/SiLU GEMM in hipBLASLt (enabled by ``HIPBLASLT_EPILOGUE_SWISH_EXT`` and ``HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT``)
* Added support for ``HIPBLASLT_EPILOGUE_GELU_AUX_BIAS`` for gfx942
* Added `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` to constrain max workspace size for user offline tuning
* Added ``HIPBLASLT_ORDER_COL16_4R16`` and ``HIPBLASLT_ORDER_COL16_4R8`` to ``hipblasLtOrder_t`` to support FP16/BF16 swizzle GEMM and FP8/BF8 swizzle GEMM respectively.
* Added TF32 emulation on gfx950

### Changed

* ``HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT`` and ``HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT`` are removed. Use the ``HIPBLASLT_MATMUL_DESC_A_SCALE_MODE`` and ``HIPBLASLT_MATMUL_DESC_B_SCALE_MODE`` attributes to set scalar (``HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F``) or vector (``HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F``).
* The non-V2 APIs (``GemmPreference``, ``GemmProblemType``, ``GemmEpilogue``, ``GemmTuning``, ``GemmInputs``) in the Cpp header are now the same as the V2 APIs (``GemmPreferenceV2``, ``GemmProblemTypeV2``, ``GemmEpilogueV2``, ``GemmTuningV2``, ``GemmInputsV2``). The original non-V2 APIs are removed.
* `hipblasltExtAMaxWithScale` API is removed.

### Optimized

* Improved performance for 8-bit (FP8/BF8/I8) NN/NT cases by adding ``s_delay_alu`` to reduce stalls from dependent ALU operations on gfx12+.
* Improved performance for 8-bit and 16-bit (FP16/BF16) TN cases by enabling software dependency check (Expert Scheduling Mode) under certain restrictions to reduce redundant hardware dependency checks on gfx12+.
* Improved performance for 8-bit, 16-bit, and 32-bit batched GEMM with a better heuristic search algorithm for gfx942.

### Upcoming changes

* V2 APIs (``GemmPreferenceV2``, ``GemmProblemTypeV2``, ``GemmEpilogueV2``, ``GemmTuningV2``, ``GemmInputsV2``) are deprecated.

## hipBLASLt 0.12.1 for ROCm 6.4.2

### Added

* Support for gfx1151

## hipBLASLt 0.12.1 for ROCm 6.4.1

### Resolved issues

* Fixed an accuracy issue that occurred for some solutions using an `FP32` or `TF32` data type with a TT transpose.

## hipBLASLt 0.12.0 for ROCm 6.4.0

### Added

* Support roctx if `HIPBLASLT_ENABLE_MARKER=1` is set
* Output the profile logging if `HIPBLASLT_LOG_MASK=64` is set
* Support FP16 compute type
* Add memory bandwidth information in hipblaslt-bench output
* Support user offline tuning mechanism
* Add more samples

### Changed

* Output the bench command along with solution index if `HIPBLASLT_LOG_MASK=32` is set

### Optimized

* Improve the overall performance of XF32/FP16/BF16/FP8/BF8 data type
* Reduce library size

### Resolved issues

* Fix multi-threads bug
* Fix multi-streams bug

## hipBLASLt 0.10.0 for ROCm 6.3.0

### Added

* Support the V2 CPP extension API for backward compatibility
* Support for data type Int8 in with Int8 out
* Support for data type FP32/FP64 for gfx110x
* Add the Extension API `hipblaslt_ext::matmulIsTuned`
* Output atol and rtol for hipblaslt-bench validation
* Output the bench command for hipblaslt CPP ext API path if `HIPBLASLT_LOG_MASK=32` is set
* Support odd sizes for FP8/BF8 GEMM

### Changed

* Reorganize and add more sample code
* Add a dependency with the hipblas-common package and remove the dependency with the hipblas package

### Optimized

* Support fused kernel for HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER for FP8/BF8 data type
* Improve the library loading time
* Improve the overall performance of first returned solution

### Upcoming changes

*  The V1 CPP extension API will be deprecated in a future release of hipBLASLt

## hipBLASLt 0.8.0

### Added

* Extension APIs:
  * `hipblasltExtAMaxWithScale`
* `GemmTuning` extension parameter to set wgm by user
* Support HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER for the FP8/BF8 data types
* Support for FP8/BF8 input, FP32/FP16/BF16/F8/BF8 output (only for the gfx94x architectures)
* Support HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT and HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT for FP16 input data type to use FP8/BF8 mfma
* Support for the gfx110x architecture

### Optimized

* Improve the library loading time

## hipBLASLt 0.7.0

### Additions

* Extension APIs:
  * `hipblasltExtSoftmax`
  * `hipblasltExtLayerNorm`
  * `hipblasltExtAMax`
* `GemmTuning` extension parameter to set split-k by user
* Support for mixed-precision datatype: FP16/FP8 in with FP16 out
* Add CMake support for documentation

### Deprecations

* algoGetHeuristic() ext API for GroupGemm will be deprecated in a future release of hipBLASLt

## hipBLASLt 0.6.0

### Additions

* New `UserArguments` variable for `GroupedGemm`
* Support for datatype: FP16 in with FP32 out
* Support for datatype: Int8 in Int32 out
* Support for gfx94x platform
* Support for FP8/BF8 datatype (only for gfx94x platform)
* Support scalar A,B,C,D for FP8/BF8 datatype
* Added samples

### Changes

* Replaced `hipblasDatatype_t` with `hipDataType`
* Replaced `hipblasLtComputeType_t` with `hipblasComputeType_t`

### Removals

* Deprecated `HIPBLASLT_MATMUL_DESC_D_SCALE_VECTOR_POINTER`

## hipBLASLt 0.3.0

### Additions

* Added `getAllAlgos` extension APIs
* TensileLite support for new epilogues: gradient gelu, gradient D, gradient A/B, aux
* Added a sample package that includes three sample apps
* Added a new C++ GEMM class in the hipBLASLt extension

### Changes

* Refactored GroupGemm APIs as C++ class in the hipBLASLt extension
* Changed the scaleD vector enum to `HIPBLASLT_MATMUL_DESC_D_SCALE_VECTOR_POINTER`

### Fixes

* Enabled norm check validation for CI

### Optimizations

* GSU kernel: wider memory, PGR N
* Updated logic yaml to improve some FP16 NN sizes
* GroupGemm support for GSU kernel
* Added grouped GEMM tuning for aldebaran

## hipBLASLt 0.2.0

### Additions

* Added CI tests for TensileLite
* Initialized extension group GEMM APIs (FP16 only)
* Added a group GEMM sample app: `example_hipblaslt_groupedgemm`

### Fixes

* Fixed incorrect results for the ScaleD kernel

### Optimizations

* Tuned equality sizes for the HHS data type
* Reduced host-side overhead for `hipblasLtMatmul()`
* Removed unused kernel arguments
* Schedule values setup before first `s_waitcnt`
* Refactored TensileLite host codes
* Optimized build time

## hipBLASLt 0.1.0

### Additions

* Enabled hipBLASLt APIs
* Support for gfx90a
* Support for problem type: FP32, FP16, BF16
* Support activation: relu, gelu
* Support for bias vectors
* Integrated with TensileLite kernel generator
* Added Gtest: `hipblaslt-test`
* Added the full function tool `hipblaslt-bench`
* Added the sample app `example_hipblaslt_preference`

### Optimizations

* gridBase solution search algorithm for untuned size
* Tuned 10k sizes for each problem type
