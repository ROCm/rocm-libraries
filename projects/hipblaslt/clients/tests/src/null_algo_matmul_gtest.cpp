// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// hipblasLtMatmul with a null algo selects the algorithm itself. When no algorithm supports the
// problem it must say so with HIPBLAS_STATUS_NOT_SUPPORTED, which callers can act on, rather than
// HIPBLAS_STATUS_INTERNAL_ERROR.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>

#include "../../../library/src/amd_detail/rocblaslt/src/include/solution_lookup_miss_for_test.hpp"

namespace
{
    constexpr int64_t kSize = 64;

    // Restores normal lookups when a test ends, including on a failed assertion.
    class LookupMiss
    {
    public:
        explicit LookupMiss(rocblaslt_testing_solution_lookup_miss miss)
        {
            rocblaslt_testing_set_solution_lookup_miss_for_test(miss);
        }
        ~LookupMiss()
        {
            rocblaslt_testing_set_solution_lookup_miss_for_test(
                rocblaslt_testing_solution_lookup_miss_none);
        }
        LookupMiss(const LookupMiss&)            = delete;
        LookupMiss& operator=(const LookupMiss&) = delete;
    };

    // Column-major FP32 D = A * B with square kSize matrices and no workspace. The buffers hold
    // no data: the tests check statuses only.
    class NullAlgoMatmul : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulDesc_t       desc   = nullptr;
        hipblasLtMatrixLayout_t     layout = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        float*                      a      = nullptr;
        float*                      b      = nullptr;
        float*                      d      = nullptr;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout, HIP_R_32F, kSize, kSize, kSize),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            for(float** buffer : {&a, &b, &d})
                ASSERT_EQ(hipMalloc(buffer, kSize * kSize * sizeof(float)), hipSuccess);
        }

        void TearDown() override
        {
            for(float* buffer : {a, b, d})
                static_cast<void>(hipFree(buffer));
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(layout);
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(handle);
        }

        int heuristicCount()
        {
            hipblasLtMatmulHeuristicResult_t result{};
            int                              returned = -1;
            EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                          handle, desc, layout, layout, layout, layout, pref, 1, &result, &returned),
                      HIPBLAS_STATUS_SUCCESS);
            return returned;
        }

        hipblasStatus_t nullAlgoMatmul()
        {
            const float           alpha = 1.0f, beta = 0.0f;
            const hipblasStatus_t status = hipblasLtMatmul(handle,
                                                           desc,
                                                           &alpha,
                                                           a,
                                                           layout,
                                                           b,
                                                           layout,
                                                           &beta,
                                                           d,
                                                           layout,
                                                           d,
                                                           layout,
                                                           nullptr,
                                                           nullptr,
                                                           0,
                                                           nullptr);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            return status;
        }
    };

    TEST_F(NullAlgoMatmul, smoke_NoAlgorithmIsNotSupported)
    {
        ASSERT_EQ(nullAlgoMatmul(), HIPBLAS_STATUS_SUCCESS) << "control, with normal lookups";

        const LookupMiss miss(rocblaslt_testing_solution_lookup_miss_all);
        EXPECT_EQ(heuristicCount(), 0);
        EXPECT_EQ(nullAlgoMatmul(), HIPBLAS_STATUS_NOT_SUPPORTED);
    }
} // namespace
