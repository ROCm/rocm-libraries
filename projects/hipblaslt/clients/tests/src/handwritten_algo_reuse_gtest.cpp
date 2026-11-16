// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// An algo for a hand-written kernel is refused, through the C and the ext API, when
// the problem it is launched for fails the kernel's problem predicates or is a
// pointer-array batch, which hand-written kernels cannot address. The gfx950 skinny TN
// kernel for n == 1 asserts n == 1; its algo is reused for n == 2 and for a
// pointer-array batch of the same problem.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{
    constexpr int64_t m = 4096;
    constexpr int64_t k = 4096;

    class HandwrittenAlgoReuse_smoke : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t handle = nullptr;
        void*             a      = nullptr;
        void*             b      = nullptr;
        void*             d      = nullptr;
        void*             arrayA = nullptr;
        void*             arrayB = nullptr;
        void*             arrayD = nullptr;
        const float       alpha  = 1.0f;
        const float       beta   = 0.0f;
        // Room for the n == 2 problem.
        const size_t dBytes = m * 2 * sizeof(uint16_t);

        void SetUp() override
        {
            int             count = 0;
            hipDeviceProp_t props{};
            if(hipGetDeviceCount(&count) != hipSuccess || count == 0
               || hipGetDeviceProperties(&props, 0) != hipSuccess)
                GTEST_SKIP() << "No GPU available";
            if(std::string(props.gcnArchName).rfind("gfx950", 0) != 0)
                GTEST_SKIP() << "The skinny TN kernels are registered for gfx950 only";
            ASSERT_EQ(hipMalloc(&a, m * k * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&b, k * 2 * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d, dBytes), hipSuccess);
            ASSERT_EQ(hipMemset(a, 0, m * k * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipMemset(b, 0, k * 2 * sizeof(uint16_t)), hipSuccess);
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
            for(void* p : {a, b, d, arrayA, arrayB, arrayD})
                if(p)
                    EXPECT_EQ(hipFree(p), hipSuccess);
        }

        // Fills D with a pattern no GEMM of zero inputs writes, so a launch shows.
        void poisonD()
        {
            ASSERT_EQ(hipMemset(d, 0x7f, dBytes), hipSuccess);
        }

        bool dUntouched()
        {
            for(uint8_t byte : download(d, dBytes))
                if(byte != 0x7f)
                    return false;
            return true;
        }

        std::vector<uint8_t> download(const void* p, size_t bytes)
        {
            std::vector<uint8_t> host(bytes);
            EXPECT_EQ(hipMemcpy(host.data(), p, bytes, hipMemcpyDeviceToHost), hipSuccess);
            return host;
        }

        // A one-batch pointer array, as large as the tensor it points to so a kernel
        // that read it as that tensor would stay inside it.
        void makePointerArray(void*& array, void* tensor, size_t bytes)
        {
            ASSERT_EQ(hipMalloc(&array, bytes), hipSuccess);
            ASSERT_EQ(hipMemset(array, 0x7f, bytes), hipSuccess);
            ASSERT_EQ(hipMemcpy(array, &tensor, sizeof(tensor), hipMemcpyHostToDevice), hipSuccess);
        }

        void makePointerArrays()
        {
            makePointerArray(arrayA, a, m * k * sizeof(uint16_t));
            makePointerArray(arrayB, b, k * 2 * sizeof(uint16_t));
            makePointerArray(arrayD, d, dBytes);
        }

        struct CProblem
        {
            hipblasLtMatmulDesc_t   desc = nullptr;
            hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;

            explicit CProblem(int64_t n)
            {
                const hipblasOperation_t opA = HIPBLAS_OP_T;
                const hipblasOperation_t opB = HIPBLAS_OP_N;
                EXPECT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                          HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(
                    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)),
                    HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(
                    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)),
                    HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, k, m, k), HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, k, n, k), HIPBLAS_STATUS_SUCCESS);
                EXPECT_EQ(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16BF, m, n, m), HIPBLAS_STATUS_SUCCESS);
            }

            void setPointerArray()
            {
                const hipblasLtBatchMode_t mode = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
                for(auto* layout : {la, lb, ld})
                    EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                                  layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &mode, sizeof(mode)),
                              HIPBLAS_STATUS_SUCCESS);
            }

            ~CProblem()
            {
                for(auto* layout : {la, lb, ld})
                    hipblasLtMatrixLayoutDestroy(layout);
                hipblasLtMatmulDescDestroy(desc);
            }
        };

        hipblasStatus_t matmul(const CProblem& p, const hipblasLtMatmulAlgo_t& algo)
        {
            return matmul(p, algo, a, b, d);
        }

        hipblasStatus_t matmul(const CProblem&              p,
                               const hipblasLtMatmulAlgo_t& algo,
                               const void*                  A,
                               const void*                  B,
                               void*                        D)
        {
            return hipblasLtMatmul(
                handle, p.desc, &alpha, A, p.la, B, p.lb, &beta, D, p.ld, D, p.ld, &algo, nullptr, 0, nullptr);
        }

        void setExtProblem(hipblaslt_ext::Gemm& gemm, int64_t n)
        {
            hipblaslt_ext::GemmProblemType pt(HIPBLAS_OP_T,
                                              HIPBLAS_OP_N,
                                              HIP_R_16BF,
                                              HIP_R_16BF,
                                              HIP_R_16BF,
                                              HIP_R_16BF,
                                              HIPBLAS_COMPUTE_32F);
            hipblaslt_ext::GemmEpilogue epilogue;
            hipblaslt_ext::GemmInputs   inputs;
            inputs.setA(a);
            inputs.setB(b);
            inputs.setC(d);
            inputs.setD(d);
            inputs.setAlpha(&alpha);
            inputs.setBeta(&beta);
            ASSERT_EQ(gemm.setProblem(m, n, k, 1, k, k, m, m, k * m, k * n, m * n, m * n, epilogue, inputs, pt),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblasLtMatmulAlgo_t skinnyAlgo()
        {
            CProblem                         p(1);
            hipblasLtMatmulPreference_t      pref = nullptr;
            hipblasLtMatmulHeuristicResult_t result{};
            int                              count = 0;
            EXPECT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                          handle, p.desc, p.la, p.lb, p.ld, p.ld, pref, 1, &result, &count),
                      HIPBLAS_STATUS_SUCCESS);
            hipblasLtMatmulPreferenceDestroy(pref);
            EXPECT_EQ(count, 1);
            EXPECT_EQ(hipblaslt_ext::getSolutionNameFromAlgo(handle, result.algo)
                          .rfind("wvSplitK_vllm_bf16_tn_n1_", 0),
                      0u)
                << "the n == 1 pick is not the hand-written skinny kernel";
            return result.algo;
        }
    };

    TEST_F(HandwrittenAlgoReuse_smoke, CApiRefusesAlgoForAnotherProblem)
    {
        const hipblasLtMatmulAlgo_t algo = skinnyAlgo();
        if(HasFailure())
            return;

        EXPECT_EQ(matmul(CProblem(1), algo), HIPBLAS_STATUS_SUCCESS);

        poisonD();
        EXPECT_EQ(matmul(CProblem(2), algo), HIPBLAS_STATUS_INVALID_VALUE);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_TRUE(dUntouched()) << "the refused algo launched";
    }

    TEST_F(HandwrittenAlgoReuse_smoke, ExtApiRefusesAlgoForAnotherProblem)
    {
        const hipblasLtMatmulAlgo_t algo = skinnyAlgo();
        if(HasFailure())
            return;

        hipblaslt_ext::Gemm gemm(
            handle, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
        setExtProblem(gemm, 1);
        EXPECT_EQ(gemm.initialize(algo, nullptr), HIPBLAS_STATUS_SUCCESS);

        setExtProblem(gemm, 2);
        EXPECT_EQ(gemm.initialize(algo, nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(HandwrittenAlgoReuse_smoke, CApiRefusesAlgoForPointerArrayBatch)
    {
        const hipblasLtMatmulAlgo_t algo = skinnyAlgo();
        if(HasFailure())
            return;
        makePointerArrays();
        if(HasFailure())
            return;

        CProblem p(1);
        p.setPointerArray();
        poisonD();
        const std::vector<uint8_t> arrayDBefore = download(arrayD, dBytes);
        EXPECT_EQ(matmul(p, algo, arrayA, arrayB, arrayD), HIPBLAS_STATUS_INVALID_VALUE);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_TRUE(dUntouched()) << "the refused algo launched";
        EXPECT_EQ(download(arrayD, dBytes), arrayDBefore) << "the refused algo launched";
    }

    TEST_F(HandwrittenAlgoReuse_smoke, ExtApiRefusesAlgoForPointerArrayBatch)
    {
        const hipblasLtMatmulAlgo_t algo = skinnyAlgo();
        if(HasFailure())
            return;
        makePointerArrays();
        if(HasFailure())
            return;

        CProblem p(1);
        p.setPointerArray();
        hipblaslt_ext::Gemm gemm(
            handle, p.desc, &alpha, arrayA, p.la, arrayB, p.lb, &beta, arrayD, p.ld, arrayD, p.ld);
        EXPECT_EQ(gemm.initialize(algo, nullptr), HIPBLAS_STATUS_INVALID_VALUE);
    }
}
