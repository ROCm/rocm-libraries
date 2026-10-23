.. meta::
   :description: Optional CK Tile grouped convolution integration for gfx1250
   :keywords: MIOpen, CK Tile, gfx1250, convolution

CK Tile grouped convolution on gfx1250
======================================

MIOpen normally consumes legacy ``ck::`` operation instances through per-architecture
``MIOpenCKGroupedConv_<arch>`` plugins. The host library loads the plugin at runtime;
it does not link CK device kernels directly. The optional ``MIOPEN_USE_CK_TILE``
build switch adds native ``ck_tile::`` candidates to the existing gfx1250 plugin
without changing MIOpen's public API, plugin API version, or solver identifiers.
Legacy candidates remain available. The option defaults to ``OFF``.

Build and run
-------------

Configure MIOpen with a matching installed CK package and ROCm compiler, or use
``MIOPEN_BUILD_CK=ON`` for the inline CK build. CK Tile grouped-convolution headers
must be present. ``MIOPEN_CK_TILE_INCLUDE_DIR`` can select a matching CK include root
when they are separate from the legacy instance package::

   cmake -S projects/miopen -B build-miopen \
     -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
     -DCMAKE_PREFIX_PATH="/opt/rocm;<CK-install-prefix>" \
     -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1250 \
     -DMIOPEN_USE_COMPOSABLEKERNEL=ON -DMIOPEN_USE_CK_TILE=ON \
     -DMIOPEN_TEST_DISCRETE=ON
   cmake --build build-miopen --target test_unit_ck_tile_grouped_conv -j 8
   LD_LIBRARY_PATH="$PWD/build-miopen/lib:/opt/rocm/lib" \
     MIOPEN_CK_LIB_PATH="$PWD/build-miopen/lib" \
     HIP_VISIBLE_DEVICES=3 \
     build-miopen/bin/test_unit_ck_tile_grouped_conv --gtest_filter='*CkTileGroupedConv*'

These commands start at the monorepo root. Use the intended ROCm installation and
choose build parallelism for available memory. Set ``MIOPEN_CK_LIB_PATH`` before
creating a handle: the loader caches successful and failed loads by architecture.
The loaded MIOpen core and plugin must have matching internal C++ ABI/API versions.
A build-tree executable can otherwise resolve an installed MIOpen library instead
of the newly built one; select its library directory explicitly during validation.

Supported candidates
--------------------

* Forward, backward-data, and backward-weights; 2D and 3D; homogeneous FP16, BF16,
  and exact FP32 with FP32 accumulation.
* Packed NHWC/NDHWC. Packed NCHW/NCDHW uses MIOpen's existing operand conversions
  and workspace partitioning. General filters, padding, strides, dilation, odd
  per-group channels, and multiple groups are checked by the actual Tile kernel.
* An explicit native WMMA tuning portfolio in ``ck_tile_grouped_conv_configs.hpp``:
  CompV3, BasicV1, and Memory pipelines; scalar and vector2/4/8 access; narrow,
  balanced, and larger block tiles. FP32 omits vector8. Warp tiles remain
  ``16x16x32`` for FP16/BF16 and ``16x16x4`` for FP32. All recipes use fully padded
  intrawave kernels, single LDS buffering, and CShuffle. General TDM and
  asynchronous-load convolution are not part of this portfolio.
* ``alpha=1``, ``beta=0``, no cast/mixed-type or fused epilogue. Other contracts
  remain on legacy candidates. Kernel IDs begin with ``CKTileGroupedConv`` and
  encode direction, rank, dtype, and kernel configuration; tuning can select them
  through the existing grouped CK solver/performance configuration.
* Forward uses split one. Scalar FP16/BF16 backward-data also requires split one.
  Backward-weights accepts supported positive splits, including split two; negative
  auto-split requests are not normalized. Existing deterministic policy excludes
  atomic split reductions.
* Tensor/descriptor products and buffer byte offsets must fit conservative signed
  32-bit bounds. Oversized strides, nonpacked storage, incompatible output geometry,
  and excessive backward residue counts are rejected, not silently narrowed.

Workspace and streams
---------------------

FP16/BF16 backward-weights writes FP32 partials into caller-owned scratch, then
converts once to the requested output dtype. Native scratch is
``G * K_per_group * C_per_group * product(filter_lengths) * sizeof(float)``, including
split one. The solution's workspace size includes any MIOpen layout conversions
and existing scalar-processing reserve. Query it rather than inferring it from
native scratch alone. Missing/undersized workspace is rejected before invocation.
Scratch must be aligned and disjoint from live operands.

Supported vectorized FP16/BF16 backward-data split reductions also accumulate into
caller-owned FP32 scratch, then convert once. Their native scratch is the logical
input-gradient element count times ``sizeof(float)``. Split-one backward-data
requires no native scratch; scalar low-precision split reductions remain rejected.

Backward-data clears the whole computed tensor to cover stride/dilation holes;
backward-weights clears its accumulation buffer for split reductions. Clears,
convolution, and conversion all use the caller's stream. There is no device scratch
allocation, benchmark repetition, or completion synchronization in the Tile engine.
Keep tensors and workspace alive until that stream completes. Direct computed-output
aliases are rejected; existing channels-first staging remains intact.

Validation and performance
--------------------------

``unit_ck_tile_grouped_conv.cpp`` runs every eligible Tile candidate against CPU
references, including odd and aligned grouped channels, asymmetric padding,
backward holes, split reductions, poisoned output/scratch, changed contents and
addresses, and independent streams.
Enabled gfx1250 runs fail if Tile candidates are absent rather than treating that as
a skip. Other architectures and disabled-feature builds do not exercise this path.

Enabling the catalog does not imply a performance win or change a trained heuristic.
Measure selected Tile and legacy candidates with the same shape, dtype, layout,
stream, and cache policy, including layout-conversion/clear/FP32-conversion costs.
Do not transfer standalone GEMM/TDM speedups to convolution claims.

Selecting kernels through MIOpenDriver
----------------------------------------

``MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4`` on the normal Find path fully tunes
applicable solver configurations, including Tile and legacy CK candidates. No
Tile-only mode or solver override is needed. ``-S <solver>`` instead selects an
immediate-mode solver and does not request full tuning.

For a controlled unrestricted forward run, start at the monorepo root and use a
fresh directory for user/system databases and compiled cache::

   work=$(mktemp -d /tmp/miopen-tile-search.XXXXXX)
   mkdir -p "$work/user" "$work/system" "$work/cache"
   export HIP_VISIBLE_DEVICES=3
   export LD_LIBRARY_PATH="$PWD/build-miopen/lib:/opt/rocm/lib"
   export MIOPEN_CK_LIB_PATH="$PWD/build-miopen/lib"
   export MIOPEN_USER_DB_PATH="$work/user"
   export MIOPEN_SYSTEM_DB_PATH="$work/system"
   export MIOPEN_CUSTOM_CACHE_DIR="$work/cache"
   export MIOPEN_DEBUG_DISABLE_SYSTEM_DB=1
   export MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4
   export MIOPEN_SEARCH_CUTOFF=0
   unset MIOPEN_DEBUG_TUNING_ITERATIONS_MAX MIOPEN_TUNING_PATIENCE
   export MIOPEN_LOG_LEVEL=7 MIOPEN_PERFORMANCE_LOGS=4
   unset MIOPEN_DEBUG_FIND_ONLY_SOLVER
   unset MIOPEN_DEBUG_CONV_GEMM MIOPEN_DEBUG_CONV_FFT
   unset MIOPEN_DEBUG_CONV_DIRECT MIOPEN_DEBUG_CONV_WINOGRAD
   build-miopen/bin/MIOpenDriver convfp16 -F 1 -n 2 -c 128 -k 128 -g 2 \
     -H 8 -W 8 -y 3 -x 3 -p 1 -q 1 -I NHWC -f NHWC -O NHWC -M 0 \
     -S -1 -i 2 -t 1 -V 1

Use ``-F 2``/``-F 4`` for backward-data/backward-weights; 3D uses the spatial-dimension
and depth/filter-depth flags with NDHWC/NCDHW. Exact FP32 uses ``conv -M 0``;
BF16 uses ``convbfp16``. For an optional CK-only diagnostic, filter the corresponding
``ConvHipImplicitGemm[3D]Group{Fwd,Bwd,Wrw}Xdlops`` solver; do not leave that filter
or backend-disable settings enabled for ordinary full tuning.

Check actual candidate ``kernel_name`` records and finite probe times, not only
the reported planned search count. Each solver's ``Done: ... best ...`` identifies
its selected configuration. The user ``*.udb.txt`` stores the exact CK kernel ID;
the FindDB ``*.ufdb.txt`` stores measured cross-backend solver/time/workspace
results. A fresh-process replay with the same directories, ``MIOPEN_FIND_ENFORCE=1``,
``MIOPEN_FIND_MODE=2``, and ``-S 0`` selects the global FindDB winner without retuning.
If that winner is grouped CK, its invocation must match the persisted CK kernel ID.
The driver integration regressions cover both CK-only and unrestricted full tuning.

Selection is fastest among the configured eligible candidates for that problem,
under MIOpen's measurement policy. It is not a universal optimal-kernel guarantee.
Do not force a Tile winner when a measured legacy candidate is faster.

Verified selection example
--------------------------

On gfx1250 GPU3 with ROCm 10.2/Clang 24, the FP16 NHWC command above measured
251 grouped-CK configurations, including all 18 Tile candidates, alongside
``GemmFwdRest``, ``ConvHipConv``, and ``ConvDirectNaiveConvFwd``. CPU-reference
verification passed. A legacy grouped-CK candidate won this particular problem;
its cross-solver measured time was approximately 0.00841 ms. A fresh immediate-mode
process loaded the persisted CK kernel ID and verified its output.

The 222 numerical/support cases and three driver selection/replay regressions
passed on GPU3. After a subsequent GPU-management/SMU stall interrupted a
post-format smoke rerun, the final formatted binaries were verified on reserved
GPU0: all 222 numerical/support cases and all three direct-driver regressions
passed, including unrestricted full tuning and fresh-process winner replay.
No matching GPU driver warnings were found in the journal window covering those
GPU0 runs. These results establish catalog participation and selection, not a
general Tile speedup or proof of the earlier stall's cause.
