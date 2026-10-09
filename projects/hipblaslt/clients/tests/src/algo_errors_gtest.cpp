// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Error paths of algorithm selection and use, which must return a status rather than crash.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace
{
    // Square, so one buffer size and one layout serve A, B, C and D.
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
        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulDesc_t       desc   = nullptr;
        hipblasLtMatrixLayout_t     layout = nullptr;
        hipblasLtMatmulPreference_t pref   = nullptr;
        void*                       a      = nullptr;
        void*                       b      = nullptr;
        void*                       c      = nullptr;
        void*                       d      = nullptr;
        float                       alpha  = 1.0f;
        float                       beta   = 0.0f;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout, kType, kSize, kSize, kSize),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            for(void** buffer : {&a, &b, &c, &d})
                ASSERT_EQ(hipMalloc(buffer, kSize * kSize * sizeof(uint16_t)), hipSuccess);
        }

        void TearDown() override
        {
            for(void* buffer : {a, b, c, d})
                static_cast<void>(hipFree(buffer));
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            if(layout)
                hipblasLtMatrixLayoutDestroy(layout);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
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

    TEST_F(AlgoErrors, smoke_HeuristicReportsZeroAlgosWhenRequestIsRejected)
    {
        hipblasLtMatmulHeuristicResult_t result{};
        int                              returned = -1;
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, layout, layout, layout, layout, pref, 0, &result, &returned),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(returned, 0);
    }

    TEST_F(AlgoErrors, smoke_MatmulRejectsUnknownIndex)
    {
        const hipblasLtMatmulAlgo_t algo = algoWithUnknownIndex();
        EXPECT_NE(hipblasLtMatmul(handle,
                                  desc,
                                  &alpha,
                                  a,
                                  layout,
                                  b,
                                  layout,
                                  &beta,
                                  c,
                                  layout,
                                  d,
                                  layout,
                                  &algo,
                                  nullptr,
                                  0,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
    }

    struct WorkspaceConfig
    {
        hipDataType typeAB;
        hipDataType typeCD;
        bool        scaled; // device scalar A/B scale pointers
    };

    // Returns how many algos the C API rejects for an undersized workspace.
    int checkUndersizedWorkspace(const WorkspaceConfig& config)
    {
        constexpr int64_t m = 256, n = 256, k = 65536;

        const size_t elementAB = config.typeAB == HIP_R_16BF ? 2 : 1;
        const size_t elementCD = 2;

        hipblasLtHandle_t handle = nullptr;
        EXPECT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulDesc_t desc = nullptr;
        EXPECT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        float* scale = nullptr;
        if(config.scaled)
        {
            EXPECT_EQ(hipMalloc(&scale, sizeof(float)), hipSuccess);
            const float one = 1.0f;
            EXPECT_EQ(hipMemcpy(scale, &one, sizeof(float), hipMemcpyHostToDevice), hipSuccess);
            hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale, sizeof(scale));
            hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale, sizeof(scale));
        }

        hipblasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutCD = nullptr;
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layoutA, config.typeAB, k, m, k),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layoutB, config.typeAB, k, n, k),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutCreate(&layoutCD, config.typeCD, m, n, m),
                  HIPBLAS_STATUS_SUCCESS);

        void *a = nullptr, *b = nullptr, *d = nullptr, *workspaceBuffer = nullptr;
        EXPECT_EQ(hipMalloc(&a, k * m * elementAB), hipSuccess);
        EXPECT_EQ(hipMalloc(&b, k * n * elementAB), hipSuccess);
        EXPECT_EQ(hipMalloc(&d, m * n * elementCD), hipSuccess);
        constexpr size_t maxWorkspace = size_t(1) << 30;
        EXPECT_EQ(hipMalloc(&workspaceBuffer, maxWorkspace), hipSuccess);

        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);

        const float         alpha = 1.0f, beta = 0.0f;
        hipblaslt_ext::Gemm gemm(
            handle, desc, &alpha, a, layoutA, b, layoutB, &beta, d, layoutCD, d, layoutCD);
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(maxWorkspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        EXPECT_EQ(gemm.algoGetHeuristic(2000, pref, results), HIPBLAS_STATUS_SUCCESS);

        int rejected = 0;
        for(size_t i = 0; i < results.size(); i++)
        {
            auto&  algo     = results[i].algo;
            size_t required = 0;
            if(gemm.isAlgoSupported(algo, required) != HIPBLAS_STATUS_SUCCESS || required == 0)
                continue;

            // The C API is the reference for which solutions need the full workspace.
            const hipblasStatus_t reference = hipblasLtMatmul(handle,
                                                              desc,
                                                              &alpha,
                                                              a,
                                                              layoutA,
                                                              b,
                                                              layoutB,
                                                              &beta,
                                                              d,
                                                              layoutCD,
                                                              d,
                                                              layoutCD,
                                                              &algo,
                                                              workspaceBuffer,
                                                              required - 1,
                                                              stream);
            if(reference == HIPBLAS_STATUS_INVALID_VALUE)
            {
                rejected++;
                gemm.setMaxWorkspaceBytes(required - 1);
                EXPECT_EQ(gemm.initialize(algo, workspaceBuffer, false, stream),
                          HIPBLAS_STATUS_INVALID_VALUE)
                    << "algo " << i << ": " << hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
            }

            gemm.setMaxWorkspaceBytes(required);
            EXPECT_EQ(gemm.initialize(algo, workspaceBuffer, false, stream), HIPBLAS_STATUS_SUCCESS)
                << "algo " << i << ": " << hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
        }

        static_cast<void>(hipStreamSynchronize(stream));
        static_cast<void>(hipStreamDestroy(stream));
        for(void* buffer : {a, b, d, workspaceBuffer, static_cast<void*>(scale)})
            static_cast<void>(hipFree(buffer));
        hipblasLtMatrixLayoutDestroy(layoutA);
        hipblasLtMatrixLayoutDestroy(layoutB);
        hipblasLtMatrixLayoutDestroy(layoutCD);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
        return rejected;
    }

    // Wherever the C API rejects a workspace smaller than a solution requires, the ext API
    // must too. Stream-K solutions run with a short workspace in both APIs.
    TEST(AlgoWorkspace, smoke_GemmInitializeRejectsUndersizedWorkspace)
    {
        int rejected = 0;
        for(const WorkspaceConfig& config : {WorkspaceConfig{HIP_R_8F_E4M3, HIP_R_16BF, true},
                                             WorkspaceConfig{HIP_R_16BF, HIP_R_16BF, false}})
            rejected += checkUndersizedWorkspace(config);

        if(rejected == 0)
            GTEST_SKIP() << "no solution in the loaded library requires a fixed workspace";
    }
} // namespace
