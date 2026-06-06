---
name: verify and fix risk1
overview: Create a standalone regression test that triggers Risk 1 (the latent multi-tile-per-CTA LDS race in MXFlatmmKernel's persistent path), verify the bug is present, apply the same one-line block_sync_lds() fix that worked for GroupedMXFlatmmKernel, then verify the test passes.
todos:
  - id: draft-test
    content: "Draft test/ck_tile/flatmm/test_mx_flatmm_persistent.cpp: standalone gtest binary, inline kernel build, FP8xFP8 trait, parameterized on init_method (0 random / 1 constants), always validates against reference_mx_gemm; 4 test cases = 2 sizes (Single_Tile_Sanity 128x256x256, Multi_Tile_Per_Block 512x4096x256) x 2 init methods (Const, Random)"
    status: in_progress
  - id: cmake-wire
    content: Add test_tile_mx_flatmm_persistent executable to test/ck_tile/flatmm/CMakeLists.txt (no instance-library extension); add to umbrella target test_tile_mx_flatmm_all
    status: pending
  - id: build-verify-bug
    content: "Build test_tile_mx_flatmm_persistent on gfx1250; run on FFM simulator; confirm Single_Tile_Sanity_{Const,Random} PASS and Multi_Tile_Per_Block_{Const,Random} FAIL (Const: characteristic 2.125 corruption; Random: check_err against reference_mx_gemm reports out-of-tolerance elements at last SFC iAccess) - verifies Risk 1 is present under both data regimes"
    status: pending
  - id: apply-kernel-fix
    content: Apply block_sync_lds() at the top of MXFlatmmKernel::operator()'s do-while body in mx_flatmm_kernel.hpp:395, with doc comment matching grouped_mx_flatmm_kernel.hpp:99-113
    status: pending
  - id: build-verify-fix
    content: Rebuild test_tile_mx_flatmm_persistent + regression suite (5 single-problem MX FLATMM + 2 grouped MX FLATMM); run all on FFM gfx1250; expect 72/72 PASS (32 single-problem + 36 grouped + 4 new persistent)
    status: pending
isProject: false
---

# Verify and fix Risk 1: MXFlatmmKernel persistent-path LDS race

## Goal

Lock down the latent defect at [`mx_flatmm_kernel.hpp:395-424`](include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp) (`MXFlatmmKernel::operator()`'s `do { … } while(UsePersistentKernel && partition_idx < total_work_tile_cnt)` persistent loop) with a regression test, then fix it. This is the same class of bug we just fixed in `GroupedMXFlatmmKernel`: the persistent loop reuses the per-CTA `__shared__ smem_ptr` across iterations with no inter-iteration barrier, so the CShuffleEpilogue's last `ds_read` from iter N races the pipeline's `async_load_tile_` write at the start of iter N+1.

## Trigger conditions (must hold for the bug to surface)

- `MXFlatmmKernel<...>` instantiated with `UsePersistentKernel = true` (via `TileGemmUniversalTraits<..., Persistent=true, ..., UseAsyncCopy=true>`).
- Total work tile count > persistent grid size. On the FFM gfx1250 simulator the persistent grid is 48 blocks; we need `(M/M_Tile) * (N/N_Tile) > 48`.
- For FP8xFP8 with `MXFlatmmConfigBase16` (M_Tile=128, N_Tile=256, K_Tile=256): `M=512, N=4096` gives 4*16=64 tiles. K=256 keeps simulator runtime small (~5s/test).

## Changes

### 1. NEW [test/ck_tile/flatmm/test_mx_flatmm_persistent.cpp](test/ck_tile/flatmm/test_mx_flatmm_persistent.cpp)

Standalone gtest binary that builds the `MXFlatmmKernel` inline (no instance-library extension). Inherits the FP8xFP8 trait `MXFlatmm_GFX1250_FP8FP8_Traits` from [mx_flatmm_arch_traits.hpp](test/ck_tile/flatmm/mx_flatmm_arch_traits.hpp).

Structure mirrors [test_grouped_gemm_mx_flatmm_util.hpp](test/ck_tile/grouped_gemm_mx/test_grouped_gemm_mx_flatmm_util.hpp)'s single-call inline-build pattern, but for the single-problem MXFlatmmKernel. The fixture is parameterized on `init_method` (0 = random uniform, 1 = constants) and **always validates against `reference_mx_gemm`** so both data regimes use the same correctness oracle:

```cpp
// Build GemmTraits with Persistent=true, UseAsyncCopy=true (mirrors
// the grouped-mx test's GemmTraits setup that already works post-fix).
using GemmTraits = ck_tile::TileGemmUniversalTraits<
    kPadM, kPadN, kPadK, DoubleSmemBuffer, ALayout, BLayout, CLayout,
    TransposeC, UseStructuredSparsity, /*Persistent=*/true,
    NumWaveGroups, /*UseAsyncCopy=*/true>;

using MXPipelineProblem = ck_tile::MXFlatmmPipelineProblem<
    ADataType, BDataType, AccDataType, FlatmmShape, GemmTraits,
    Scheduler::Default, /*HasHotLoop=*/true, TailNumber::Full>;

using MXFlatmmPipeline = MXFlatmmArchTraits::MXFlatmmPipeline<MXPipelineProblem>;
using GemmEpilogue     = ck_tile::CShuffleEpilogue<...>;
using Kernel           = ck_tile::MXFlatmmKernel<TilePartitioner,
                                                  MXFlatmmPipeline,
                                                  GemmEpilogue>;

// --- Tensor init parameterized on init_method (mirror the fill_tensor /
// fill_scale helpers from test_grouped_gemm_mx_flatmm_util.hpp lines 105-137) ---
//
//   init_method == 0: A ~ FillUniformDistribution<>{0.0f, 1.0f},
//                     B ~ FillUniformDistribution<>{-0.5f, 0.5f},
//                     scale_a, scale_b ~ FillUniformDistribution<>{-2.f, 2.f}.
//                     No closed-form expected value, so the CPU reference is
//                     the only correctness oracle.
//   init_method == 1: A = 2.0 const, B = 0.5 const,
//                     scale_a = 0.5 const, scale_b = 2.0 const.
//                     Per-element expected = K. This regime amplifies the bug
//                     into a near-uniform value (2.125) at the corrupted lanes,
//                     making the failure pattern visually obvious in test logs.
//
// FP6 init proxy: the test is FP8xFP8 only (per the scope question answered
// earlier), so we do NOT need the pk_fp6x16_t init-proxy machinery. If the
// scope is later expanded to FP6, the mx_flatmm_init_proxy_t pattern from
// test_grouped_gemm_mx_flatmm_util.hpp:25-27 should be reused verbatim.

auto kargs        = Kernel::MakeKernelArgs(args);
const dim3 grids  = Kernel::GridSize(kargs);   // returns min(persistent_block_size,
                                                //              total_work_tile_cnt) for persistent path
const dim3 blocks = Kernel::BlockSize();
ck_tile::launch_kernel(...);

// --- Validation against reference_mx_gemm (always, regardless of init_method) ---
ck_tile::HostTensor<CDataType> c_ref(...);
c_ref.SetZero();
ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, ScaleType,
                           AccDataType, CDataType>(
    a_host, b_origin_host, c_ref, scale_a, scale_b);

const float rtol = 1e-2f, atol = 1e-2f;
EXPECT_TRUE(ck_tile::check_err(c_rslt_host, c_ref,
                               "MX persistent flatmm result mismatch",
                               rtol, atol));
```

Test cases (each problem size runs **twice** — once with `init_method=0`, once with `init_method=1`):

| Test name | M | N | K | total_tiles | grid (expected) | Multi-tile? |
|---|---|---|---|---|---|---|
| `Single_Tile_Sanity_Const` (init=1) | 128 | 256 | 256 | 1 | 1 | No (control) |
| `Single_Tile_Sanity_Random` (init=0) | 128 | 256 | 256 | 1 | 1 | No (control) |
| `Multi_Tile_Per_Block_Const` (init=1) | 512 | 4096 | 256 | 64 | 48 | Yes (triggers bug) |
| `Multi_Tile_Per_Block_Random` (init=0) | 512 | 4096 | 256 | 64 | 48 | Yes (triggers bug) |

Why both init methods:
- `init_method=1` (constants) makes the bug signature **diagnostic** — corrupted lanes return the same `2.125` value and the rtol/atol tolerance is effectively zero noise, so any failure points squarely at the LDS race.
- `init_method=0` (random) makes the bug signature **representative** of real workloads — random A/B values amplify into different garbage at corrupted lanes; without the reference oracle this would be undetectable. It also rules out the trivial-input class of false negatives (e.g., a fix that only handles constant zeros).
- Both regimes share the `reference_mx_gemm` oracle so the validation code is identical; only the input data differs.

The sanity case (M=128, N=256, K=256, total_tiles=1) proves the test infra works end-to-end with `UsePersistentKernel=true` for both init regimes. The multi-tile case (total_tiles > grid) triggers the bug under both regimes; before-fix expect both to FAIL, after-fix both to PASS.

### 2. [test/ck_tile/flatmm/CMakeLists.txt](test/ck_tile/flatmm/CMakeLists.txt)

Add one executable inside the existing `if(GPU_TARGETS MATCHES "gfx95|gfx125")` block, alongside the dtype-typed `test_tile_mx_flatmm_*` targets:

```cmake
add_gtest_executable(test_tile_mx_flatmm_persistent
    test_mx_flatmm_persistent.cpp)
target_include_directories(test_tile_mx_flatmm_persistent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_tile_mx_flatmm_persistent PRIVATE ${TEST_FLATMM_COMPILE_OPTIONS})
# No link to mx_flatmm_test_instances - this test builds its kernel inline.

add_dependencies(test_tile_mx_flatmm_all test_tile_mx_flatmm_persistent)
```

No changes to the instance-library `foreach` loops (so no doubling of single-problem compile time).

### 3. Verify the bug is PRESENT (before applying the fix)

Build `test_tile_mx_flatmm_persistent` and run it on FFM gfx1250. Expected outcome **before fix**:
- `Single_Tile_Sanity_Const` and `Single_Tile_Sanity_Random`: PASS (no multi-tile interaction).
- `Multi_Tile_Per_Block_Const`: FAIL with the characteristic `2.125`-instead-of-`K` corruption on a sub-percent of elements at the last SFC `iAccess` position. Constant-input regime makes the failure value diagnostic.
- `Multi_Tile_Per_Block_Random`: FAIL — `check_err` against `reference_mx_gemm` reports elements outside the `rtol=atol=1e-2` envelope at the same SFC last-`iAccess` position (different numerical garbage per element since A/B are random, but detectable via the reference oracle).

Both failure modes confirm Risk 1 is present. The random case proves the bug is detectable on representative inputs (not just amplified by constants).

### 4. Apply the fix to [include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp](include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp)

Add `block_sync_lds();` as the first statement inside the `do { … }` body at [line 395](include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp), with a doc comment matching the one we already placed in [grouped_mx_flatmm_kernel.hpp:99-113](include/ck_tile/ops/flatmm/kernel/grouped_mx_flatmm_kernel.hpp):

```cpp
do
{
    // Drain the prior tile's in-flight LDS reads (notably the
    // CShuffleEpilogue's final `ds_read` of the C-shuffle tile)
    // before the next tile's MXFlatmmPipeline::Run_ issues
    // `async_load_tile_` (buffer_load_lds) writes into the same
    // per-CTA `__shared__ smem_ptr` region. On gfx1250 the async
    // writes are tracked by `asynccnt` which is not ordered against
    // in-flight `ds_read`s on `dscnt`, so without this barrier the
    // leading wave's prefetch races and clobbers bytes that a lagging
    // wave's last `iAccess` is still reading. Mirrors the same fix at
    // grouped_mx_flatmm_kernel.hpp and universal_gemm_kernel.hpp:1316
    // (commit b664f4b6).
    block_sync_lds();
    const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(partition_idx);
    ...
```

First-iteration barrier is a no-op (no prior LDS activity in a fresh CTA) — correct and cheap.

### 5. Verify the fix WORKS (after applying)

Rebuild the test, re-run on FFM gfx1250. Expected outcome **after fix**:
- All 4 `test_tile_mx_flatmm_persistent` tests PASS (both Const and Random variants of both Single_Tile_Sanity and Multi_Tile_Per_Block).

### 6. Regression suite

Rebuild and rerun on FFM gfx1250 to confirm no collateral damage:
- 5 single-problem MX FLATMM binaries (`test_tile_mx_flatmm_fp{8fp8,4fp4,6fp6,8fp4,4fp8}`) - 32 tests, expect 32/32 PASS unchanged.
- 2 grouped MX FLATMM binaries (`test_ck_tile_grouped_gemm_mx_flatmm_{non_tdm,tdm}`) - 36 tests, expect 36/36 PASS unchanged.
- New `test_tile_mx_flatmm_persistent` binary - **4 tests** (Single_Tile_Sanity x {Const, Random}, Multi_Tile_Per_Block x {Const, Random}), expect 4/4 PASS.

Final expected: **72/72 PASS** (was 68/68 before this work; +4 from the new persistent test's two cases x two init regimes).

## Expert dispatch

Per the ck-code skill's selection matrix, "Bug fix in GPU kernel" task type calls for Code Expert + GPU Expert + Testing Expert + Style Expert. However:
- The fix is literally one line + comment, semantically identical to the change we just applied to `grouped_mx_flatmm_kernel.hpp` (which already went through ck-debug and was empirically verified).
- The new test file is a straightforward clone of the established `test_grouped_gemm_mx_flatmm_util.hpp` pattern, parameterized for the single-problem `MXFlatmmKernel`.
- Both Code Expert and GPU Expert already analyzed the identical bug class at the immediate prior ck-debug step and converged on this exact fix.

**Proposal: skip expert dispatch for the kernel fix (~5 lines incl. comment)**, since it is mechanically the same as the verified GroupedMXFlatmmKernel fix. **Dispatch experts only if the test-side draft requires non-trivial choices** (e.g., problem-size selection, args struct, kargs wiring), which I'll know after Step 1. If empirical results in Step 3 don't match the predicted failure pattern, escalate to ck-debug instead of guessing.

If you prefer the conservative interpretation of the skill (always dispatch on GPU kernel changes), I can re-run with Code Expert + GPU Expert in parallel before applying the fix - costs ~10 minutes of subagent time.

## Files touched (2)

- New: [test/ck_tile/flatmm/test_mx_flatmm_persistent.cpp](test/ck_tile/flatmm/test_mx_flatmm_persistent.cpp) - standalone gtest binary with 2 test cases (~150 LoC).
- Modified: [test/ck_tile/flatmm/CMakeLists.txt](test/ck_tile/flatmm/CMakeLists.txt) - add executable + umbrella-target dependency (~6 lines).
- Modified: [include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp](include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp) - 1 functional line + comment inside the do-while body (Step 4).