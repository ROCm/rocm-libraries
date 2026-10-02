// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Gemm::setProblem rejects a null A or B when alpha is nonzero, even when K is zero, and
// accepts them when alpha is zero.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <vector>

namespace
{
    constexpr int64_t     kM    = 256;
    constexpr int64_t     kN    = 128;
    constexpr hipDataType kType = HIP_R_16F;

    // One FP16 NN problem with K = 0. C and D are real buffers that are never read: every
    // call under test returns before a kernel is launched.
    class GemmPointerCheck : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t       handle  = nullptr;
        hipblasLtMatmulDesc_t   desc    = nullptr;
        hipblasLtMatrixLayout_t layoutA = nullptr;
        hipblasLtMatrixLayout_t layoutB = nullptr;
        hipblasLtMatrixLayout_t layoutC = nullptr;
        void*                   c       = nullptr;
        void*                   d       = nullptr;
        float                   alpha   = 1.0f;
        float                   beta    = 0.5f;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layoutA, kType, kM, 0, kM),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layoutB, kType, 0, kN, 1),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layoutC, kType, kM, kN, kM),
                      HIPBLAS_STATUS_SUCCESS);
            for(void** buffer : {&c, &d})
                ASSERT_EQ(hipMalloc(buffer, kM * kN * sizeof(uint16_t)), hipSuccess);
        }

        void TearDown() override
        {
            for(void* buffer : {c, d})
                static_cast<void>(hipFree(buffer));
            for(hipblasLtMatrixLayout_t layout : {layoutC, layoutB, layoutA})
                if(layout)
                    hipblasLtMatrixLayoutDestroy(layout);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
        }

        hipblasStatus_t setProblem(hipblaslt_ext::Gemm& gemm)
        {
            return gemm.setProblem(
                desc, &alpha, nullptr, layoutA, nullptr, layoutB, &beta, c, layoutC, d, layoutC);
        }
    };

    TEST_F(GemmPointerCheck, smoke_SetProblemRejectsNullABWithZeroK)
    {
        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        EXPECT_EQ(setProblem(gemm), HIPBLAS_STATUS_INVALID_VALUE);

        hipblaslt_ext::GemmPreference                 pref;
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        EXPECT_EQ(gemm.algoGetHeuristic(1, pref, results), HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_TRUE(results.empty());
    }

    TEST_F(GemmPointerCheck, smoke_DimensionSetProblemRejectsNullABWithZeroK)
    {
        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        hipblaslt_ext::GemmEpilogue epilogue;
        hipblaslt_ext::GemmInputs   inputs;
        inputs.setC(c);
        inputs.setD(d);
        inputs.setAlpha(&alpha);
        inputs.setBeta(&beta);
        EXPECT_EQ(gemm.setProblem(kM, kN, 0, 1, epilogue, inputs), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(GemmPointerCheck, smoke_SetProblemAcceptsNullABWithZeroAlpha)
    {
        alpha = 0.0f;
        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        EXPECT_EQ(setProblem(gemm), HIPBLAS_STATUS_SUCCESS);
    }
} // namespace
