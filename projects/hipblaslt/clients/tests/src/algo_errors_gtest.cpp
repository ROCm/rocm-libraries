// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Error paths of algorithm selection and use, which must return a status rather than crash.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <vector>

namespace
{
    // Square, so A, B, C and D have the same size.
    constexpr int64_t     kSize = 128;
    constexpr hipDataType kType = HIP_R_16F;

    // No solution library holds this many solutions.
    constexpr int kUnknownIndex = 0x3fffffff;

    hipblasLtMatmulAlgo_t algoWithUnknownIndex()
    {
        hipblasLtMatmulAlgo_t algo{};
        *reinterpret_cast<int*>(algo.data) = kUnknownIndex;
        return algo;
    }

    // One FP16 NN problem. The buffers are never read: every call under test
    // returns before a kernel is launched.
    class AlgoErrors : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t handle = nullptr;
        void*             a      = nullptr;
        void*             b      = nullptr;
        void*             c      = nullptr;
        void*             d      = nullptr;
        float             alpha  = 1.0f;
        float             beta   = 0.0f;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            for(void** buffer : {&a, &b, &c, &d})
                ASSERT_EQ(hipMalloc(buffer, kSize * kSize * sizeof(uint16_t)), hipSuccess);
        }

        void TearDown() override
        {
            for(void* buffer : {a, b, c, d})
                static_cast<void>(hipFree(buffer));
            if(handle)
                hipblasLtDestroy(handle);
        }

        hipblaslt_ext::GemmInputs inputs()
        {
            hipblaslt_ext::GemmInputs in;
            in.setA(a);
            in.setB(b);
            in.setC(c);
            in.setD(d);
            in.setAlpha(&alpha);
            in.setBeta(&beta);
            return in;
        }
    };

    TEST_F(AlgoErrors, smoke_GemmInitializeRejectsUnknownIndex)
    {
        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        hipblaslt_ext::GemmEpilogue epilogue;
        hipblaslt_ext::GemmInputs   in = inputs();
        ASSERT_EQ(gemm.setProblem(kSize, kSize, kSize, 1, epilogue, in), HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(gemm.initialize(algoWithUnknownIndex(), nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(AlgoErrors, smoke_GroupedGemmInitializeRejectsUnknownIndex)
    {
        hipblaslt_ext::GroupedGemm gemm(
            handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kType, kType, kType, kType, HIPBLAS_COMPUTE_32F);
        std::vector<int64_t>                     size(2, kSize), batch(2, 1);
        std::vector<hipblaslt_ext::GemmEpilogue> epilogue(2);
        std::vector<hipblaslt_ext::GemmInputs>   in(2, inputs());
        ASSERT_EQ(gemm.setProblem(size, size, size, batch, epilogue, in), HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(gemm.initialize(algoWithUnknownIndex(), nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    }
} // namespace
