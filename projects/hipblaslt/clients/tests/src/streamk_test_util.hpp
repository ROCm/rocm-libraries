// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Shared by the Stream-K stream-isolation tests: the problem setup they build
// their handle, layouts and descriptor from.
//
// A test that has to observe the flag region being read rests on the heuristic
// answering the shape below with a Stream-K solution, which is a property of the
// shipped tuning files rather than of the library: when it stops holding they
// must skip, not pass as no-ops. A test that only needs a flag block to be
// claimed does not -- that happens for every matmul, before the solution is
// known -- and passes createProblem a shape of its own instead.

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <vector>

namespace streamk_test
{
    // The shape the customer workload deadlocked on, and the one the heuristic
    // picks a Stream-K solution for on gfx950. createProblem defaults to it.
    constexpr int64_t kM = 1024;
    constexpr int64_t kN = 7168;
    constexpr int64_t kK = 4096;

    // The hazard only exists on the Stream-K remainder path, reached when the
    // tile count does not divide evenly by the grid. This has to be forced:
    // left alone the heuristic sizes both grid and tiles at 256 for this shape,
    // and a test on the even path passes against a broken library.
    //
    // getSKGrid folds smCountTarget into its CU budget, so a 96-CU target gives
    // grid 96 against 180 tiles, leaving a remainder of 84. Checked on gfx950
    // with TENSILE_DB=0xFFFF; 64 and 128 both come back evenly divided and are
    // not usable here. This is the supported API spelling of
    // TENSILE_STREAMK_MAX_CUS: those environment variables are cached in
    // function-local statics on first read, so a test binary cannot set them
    // reliably once anything else has touched the library.
    constexpr int32_t kSmCountTarget = 96;

    // Only the ceiling offered to the heuristic. What actually gets allocated is
    // the workspace the chosen solution reports needing.
    constexpr size_t kWsBudgetBytes = 128ull << 20;

    inline bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    // Frees whatever was allocated. Only safe while the queue still drains.
    struct Resources
    {
        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulDesc_t       desc   = nullptr;
        hipblasLtMatrixLayout_t     layA = nullptr, layB = nullptr, layD = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        void *                      dA = nullptr, *dB = nullptr;
        std::vector<void*>          dD, dWs;
        std::vector<hipStream_t>    streams;

        ~Resources()
        {
            for(auto s : streams)
                if(s)
                    static_cast<void>(hipStreamDestroy(s));
            for(auto p : dD)
                static_cast<void>(hipFree(p));
            for(auto p : dWs)
                static_cast<void>(hipFree(p));
            static_cast<void>(hipFree(dA));
            static_cast<void>(hipFree(dB));
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(layA)
                hipblasLtMatrixLayoutDestroy(layA);
            if(layB)
                hipblasLtMatrixLayoutDestroy(layB);
            if(layD)
                hipblasLtMatrixLayoutDestroy(layD);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
        }
    };

    // Handle, bf16 m x n x k layouts, the descriptor carrying kSmCountTarget and
    // the preference carrying kWsBudgetBytes. Asserts and skips, so callers must
    // check IsSkipped()/HasFatalFailure() before using `r`. The dimensions
    // default to the Stream-K shape above; a caller that does not need a
    // Stream-K solution should pass something cheaper.
    inline void
        createProblem(Resources& r, int64_t m = kM, int64_t n = kN, int64_t k = kK)
    {
        ASSERT_EQ(hipblasLtCreate(&r.handle), HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layA, HIP_R_16BF, m, k, m),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layB, HIP_R_16BF, k, n, k),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&r.layD, HIP_R_16BF, m, n, m),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatmulDescCreate(&r.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(r.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(r.desc,
                                                  HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                                  &kSmCountTarget,
                                                  sizeof(kSmCountTarget)),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&r.pref), HIPBLAS_STATUS_SUCCESS);
        const uint64_t wsBudget = kWsBudgetBytes;
        hipblasLtMatmulPreferenceSetAttribute(
            r.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBudget, sizeof(wsBudget));
    }
} // namespace streamk_test
