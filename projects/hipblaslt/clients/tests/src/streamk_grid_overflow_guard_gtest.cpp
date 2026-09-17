/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Selection-level regression coverage for ROCM-31016
// (https://github.com/ROCm/hipBLASLt/issues/2299): on gfx950, a large-M bf16
// matmul silently left most of D unwritten.
//
// The cause is a 32-bit narrowing in the launch path. ContractionSolution
// computes numWorkItems as workGroupSize * numWorkGroups in size_t, then
// HipSolutionAdapter passes it to hipExtModuleLaunchKernel, whose
// globalWorkSize parameters are `unsigned int`. Stream-K normally sizes its
// grid to the CU count, which stays small, but several launch-time fallbacks
// in getSKGridImpl() hand it one workgroup per output tile instead. Measured
// against that API on gfx950 at 256 threads:
//
//   tiles         work items    narrowed   observed
//   2^24 - 1      2^32 - 256    unchanged  correct
//   2^24          2^32          0          launch rejected, invalid argument
//   2^24 + 16     2^32 + 4096   4096       16 workgroups ran, 16,777,216
//                                          missing, no error reported
//
// StreamKWorkgroupNumberCheck (ContractionProblemPredicates.hpp) keeps
// dispatch away from those tile counts. These tests assert the resulting
// invariant against the real library: every Stream-K solution the heuristic
// offers for a shape must have a tile count whose work-item product fits in
// 32 bits.
//
// Why an invariant instead of comparing how many algorithms come back: the
// returned count already changes with M for unrelated reasons (on a pre-fix
// library this shape family drops from 457 candidates to 327 across the
// boundary, mostly via the existing WorkgroupNumberCheck on the handful of
// non-Stream-K solutions), so a count comparison passes with or without this
// guard and proves nothing. The invariant fails specifically and only when an
// overflowing candidate is offered: measured on the pre-fix ROCm 7.13 library,
// 42 offered Stream-K solutions violate it at tiles == 2^24 and 61 do at the
// originally reported shape.
//
// Nothing here allocates device memory or runs a GEMM.
// hipblasLtMatmulAlgoGetHeuristic evaluates predicates from the descriptor and
// layouts alone, so these tests are cheap no matter how large a shape they
// describe. Execution-level coverage that actually dispatches the reproducer
// and checks D needs about 10-21 GiB and is tracked separately.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace
{
    // Mirrors StreamKWorkgroupNumberCheck::MAX_STREAMK_TILE_COUNT. The
    // predicate does not carry NumThreads, so it bounds the 256-thread case
    // that dominates the shipped Stream-K solutions; this test asserts that
    // same contract. The one shipped family that is looser (MacroTile 448x128
    // at 512 threads, which narrows at 2^23) needs roughly 896 GiB of D to
    // reach its own limit and is tracked as follow-up work.
    constexpr uint64_t kMaxStreamKTiles = UINT32_MAX / 256; // 16,777,215

    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    // Tensile encodes the solution's parameters in the kernel name. StreamK's
    // valid values are [0, 3, 4, 5] and appear as an _SK<n>_ token; matching
    // "_SK" alone would also hit the neighbouring _SKFTR0_ and _SKXCCM0_.
    bool isStreamKKernel(const std::string& name)
    {
        for(const char* token : {"_SK3_", "_SK4_", "_SK5_"})
            if(name.find(token) != std::string::npos)
                return true;
        return false;
    }

    // Macro tile appears as _MT<MacroTile0>x<MacroTile1>x<DepthU>.
    bool parseMacroTile(const std::string& name, uint64_t& macroTile0, uint64_t& macroTile1)
    {
        const auto pos = name.find("_MT");
        if(pos == std::string::npos)
            return false;
        return std::sscanf(name.c_str() + pos, "_MT%" SCNu64 "x%" SCNu64, &macroTile0, &macroTile1)
                   == 2
               && macroTile0 > 0 && macroTile1 > 0;
    }

    struct ScopeGuard
    {
        std::vector<std::function<void()>>& cleanups;
        ~ScopeGuard()
        {
            for(auto it = cleanups.rbegin(); it != cleanups.rend(); ++it)
                (*it)();
        }
    };

    struct OfferedSolutions
    {
        bool        queried          = false; // the heuristic call itself succeeded
        int         count            = 0; // candidates returned
        int         streamKCount     = 0;
        int         unparsedCount    = 0; // Stream-K candidates whose name did not parse
        int         overflowingCount = 0;
        uint64_t    worstTiles       = 0;
        std::string worstName;
    };

    // Asks the heuristic for every candidate it will offer for a bf16 NN shape
    // and measures the tile count each Stream-K candidate implies.
    OfferedSolutions offeredSolutionsFor(int64_t M, int64_t N, int64_t K)
    {
        OfferedSolutions result;

        std::vector<std::function<void()>> cleanups;
        ScopeGuard                         guard{cleanups};

        hipblasLtHandle_t handle{};
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return result;
        cleanups.push_back([handle] { static_cast<void>(hipblasLtDestroy(handle)); });

        hipblasLtMatrixLayout_t layoutA{};
        if(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16BF, M, K, M) != HIPBLAS_STATUS_SUCCESS)
            return result;
        cleanups.push_back([layoutA] { static_cast<void>(hipblasLtMatrixLayoutDestroy(layoutA)); });

        hipblasLtMatrixLayout_t layoutB{};
        if(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16BF, K, N, K) != HIPBLAS_STATUS_SUCCESS)
            return result;
        cleanups.push_back([layoutB] { static_cast<void>(hipblasLtMatrixLayoutDestroy(layoutB)); });

        hipblasLtMatrixLayout_t layoutD{};
        if(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_16BF, M, N, M) != HIPBLAS_STATUS_SUCCESS)
            return result;
        cleanups.push_back([layoutD] { static_cast<void>(hipblasLtMatrixLayoutDestroy(layoutD)); });

        hipblasLtMatmulDesc_t desc{};
        if(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
           != HIPBLAS_STATUS_SUCCESS)
            return result;
        cleanups.push_back([desc] { static_cast<void>(hipblasLtMatmulDescDestroy(desc)); });

        hipblasOperation_t opN = HIPBLAS_OP_N;
        if(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN))
               != HIPBLAS_STATUS_SUCCESS
           || hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))
                  != HIPBLAS_STATUS_SUCCESS)
            return result;

        hipblasLtMatmulPreference_t pref{};
        if(hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS)
            return result;
        cleanups.push_back([pref] { static_cast<void>(hipblasLtMatmulPreferenceDestroy(pref)); });

        size_t maxWorkspace = 128ull << 20;
        if(hipblasLtMatmulPreferenceSetAttribute(
               pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspace, sizeof(maxWorkspace))
           != HIPBLAS_STATUS_SUCCESS)
            return result;

        // Deliberately far more than any tuned library holds for one type
        // combo, so the returned count is never clipped by the request.
        constexpr int                                 kRequestedAlgoCount = 8192;
        std::vector<hipblasLtMatmulHeuristicResult_t> candidates(kRequestedAlgoCount);
        int                                           found = 0;

        if(hipblasLtMatmulAlgoGetHeuristic(handle,
                                           desc,
                                           layoutA,
                                           layoutB,
                                           layoutD,
                                           layoutD,
                                           pref,
                                           kRequestedAlgoCount,
                                           candidates.data(),
                                           &found)
           != HIPBLAS_STATUS_SUCCESS)
        {
            // No solution for this shape is a legitimate outcome once the
            // guard is in place, and is not a failure by itself.
            result.queried = true;
            return result;
        }

        result.queried = true;
        result.count   = found;

        for(int i = 0; i < found; ++i)
        {
            const std::string name
                = hipblaslt_ext::getKernelNameFromAlgo(handle, candidates[i].algo);
            if(!isStreamKKernel(name))
                continue;
            ++result.streamKCount;

            uint64_t macroTile0 = 0, macroTile1 = 0;
            if(!parseMacroTile(name, macroTile0, macroTile1))
            {
                ++result.unparsedCount;
                continue;
            }

            // Same tile product the predicate computes: batch is 1 here.
            const uint64_t tiles = ((uint64_t)M + macroTile0 - 1) / macroTile0
                                   * (((uint64_t)N + macroTile1 - 1) / macroTile1);
            if(tiles > kMaxStreamKTiles)
            {
                ++result.overflowingCount;
                if(tiles > result.worstTiles)
                {
                    result.worstTiles = tiles;
                    result.worstName  = name;
                }
            }
        }

        return result;
    }

    // With N=256 and a 16x16 macro tile the tile count is exactly M, which is
    // the geometry the hardware investigation used.
    constexpr int64_t kK = 48;
    constexpr int64_t kN = 256;

    // Without this, a device whose tuned library failed to load offers zero
    // candidates, finds zero violations among them, and passes without having
    // tested anything. The guarded shapes below keep their large-macro-tile
    // Stream-K candidates (only the small tiles cross the limit), so an empty
    // result means the library is missing, not that the guard rejected
    // everything.
#define SKIP_UNLESS_CANDIDATES_WERE_OFFERED(offered)                                   \
    do                                                                                 \
    {                                                                                  \
        if(!gpuAvailable())                                                            \
            GTEST_SKIP() << "no GPU available";                                        \
        ASSERT_TRUE((offered).queried) << "could not set up a heuristic query";        \
        if((offered).count == 0)                                                       \
            GTEST_SKIP() << "the heuristic offered no candidates at all, so there is " \
                            "no tuned library here to check";                          \
    } while(0)
}

// Below the limit the library must still offer Stream-K solutions, and every
// one of them must be within it. This is the control: it proves the guard is
// not simply rejecting the whole shape family, and that the assertion in the
// tests below is not vacuous.
TEST(StreamKGridOverflowGuard_pre_checkin, BelowLimitStillOffersSafeStreamKSolutions)
{
    // tiles == 2^24 - 16, the largest multiple of the N-tile count under the
    // limit for this geometry.
    const auto offered = offeredSolutionsFor(/*M=*/16777200, kN, kK);
    SKIP_UNLESS_CANDIDATES_WERE_OFFERED(offered);

    if(offered.streamKCount == 0)
        GTEST_SKIP() << "no Stream-K solutions tuned for bf16 NN on this device; "
                        "nothing for this guard to bound";

    EXPECT_EQ(offered.overflowingCount, 0)
        << "a shape under the limit must not lose solutions to this guard; worst was "
        << offered.worstTiles << " tiles (limit " << kMaxStreamKTiles << ")";
    EXPECT_EQ(offered.unparsedCount, 0)
        << "could not read MacroTile out of " << offered.unparsedCount
        << " Stream-K kernel names, so the assertion above did not actually cover them";
}

// tiles == 2^24 exactly: 2^32 work items at 256 threads, which narrows to a
// zero-sized grid. 42 offered Stream-K solutions violated this on the pre-fix
// ROCm 7.13 library.
TEST(StreamKGridOverflowGuard_pre_checkin, AtLimitOffersNoOverflowingStreamKSolution)
{
    const auto offered = offeredSolutionsFor(/*M=*/16777216, kN, kK);
    SKIP_UNLESS_CANDIDATES_WERE_OFFERED(offered);

    EXPECT_EQ(offered.overflowingCount, 0)
        << offered.overflowingCount << " of " << offered.streamKCount
        << " offered Stream-K solutions imply more than " << kMaxStreamKTiles
        << " tiles; worst was " << offered.worstTiles << " tiles via " << offered.worstName;
}

// The shape ROCM-31016 was reported against. 61 offered Stream-K solutions
// violated this on the pre-fix ROCm 7.13 library.
TEST(StreamKGridOverflowGuard_pre_checkin, AtReportedShapeOffersNoOverflowingStreamKSolution)
{
    const auto offered = offeredSolutionsFor(/*M=*/33554560, kN, kK);
    SKIP_UNLESS_CANDIDATES_WERE_OFFERED(offered);

    EXPECT_EQ(offered.overflowingCount, 0)
        << offered.overflowingCount << " of " << offered.streamKCount
        << " offered Stream-K solutions imply more than " << kMaxStreamKTiles
        << " tiles; worst was " << offered.worstTiles << " tiles via " << offered.worstName;
}
