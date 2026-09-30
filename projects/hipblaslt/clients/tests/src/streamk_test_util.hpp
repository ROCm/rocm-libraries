// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU resources and BF16 problem setup for the null-stream accounting test.

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <vector>

namespace streamk_test
{
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

    // Callers check IsSkipped()/HasFatalFailure() before using partially built resources.
    inline void createProblem(Resources& r, int64_t m, int64_t n, int64_t k)
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
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&r.pref), HIPBLAS_STATUS_SUCCESS);
        const uint64_t wsBudget = kWsBudgetBytes;
        hipblasLtMatmulPreferenceSetAttribute(
            r.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBudget, sizeof(wsBudget));
    }
} // namespace streamk_test
