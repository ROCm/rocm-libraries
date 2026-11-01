// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <vector>

namespace
{
    // A nonsquare problem also covers sizes outside the old gfx12 Equality tables.
    constexpr int64_t m = 128, n = 129, k = 128;
    constexpr size_t  maxWorkspace = 4 * 1024 * 1024;

    class Xf32Fallback : public ::testing::TestWithParam<bool>
    {
    protected:
        hipblasLtHandle_t                handle = nullptr;
        hipblasLtMatmulDesc_t            fp32 = nullptr, xf32 = nullptr;
        hipblasLtMatrixLayout_t          aLayout = nullptr, bLayout = nullptr, cdLayout = nullptr;
        hipblasLtMatmulPreference_t      pref = nullptr;
        hipblasLtMatmulHeuristicResult_t candidate{};
        float *            a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *bias = nullptr;
        void*              workspace = nullptr;
        float              alpha = 1.0f, beta = 0.0f;
        std::vector<float> ha, hb, hBias;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&fp32, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&xf32, HIPBLAS_COMPUTE_32F_FAST_TF32, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&aLayout, HIP_R_32F, k, m, k),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&bLayout, HIP_R_32F, k, n, k),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&cdLayout, HIP_R_32F, m, n, m),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                hipblasLtMatmulPreferenceSetAttribute(pref,
                                                      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &maxWorkspace,
                                                      sizeof(maxWorkspace)),
                HIPBLAS_STATUS_SUCCESS);

            ha.resize(m * k);
            hb.resize(n * k);
            hBias.resize(m);
            for(size_t i = 0; i < ha.size(); ++i)
                ha[i] = (static_cast<int>(i % 13) - 6) / 8.0f;
            for(size_t i = 0; i < hb.size(); ++i)
                hb[i] = (static_cast<int>(i % 17) - 8) / 8.0f;
            for(size_t i = 0; i < hBias.size(); ++i)
                hBias[i] = static_cast<float>(i % 7) / 4.0f;

            ASSERT_EQ(hipMalloc(&a, ha.size() * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMalloc(&b, hb.size() * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMalloc(&c, m * n * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d, m * n * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMalloc(&bias, m * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMemcpy(a, ha.data(), ha.size() * sizeof(float), hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(hipMemcpy(b, hb.data(), hb.size() * sizeof(float), hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(hipMemcpy(bias, hBias.data(), m * sizeof(float), hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(hipMemset(c, 0, m * n * sizeof(float)), hipSuccess);

            const hipblasOperation_t  transA = HIPBLAS_OP_T;
            const hipblasLtEpilogue_t epilogue
                = GetParam() ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT;
            for(auto desc : {fp32, xf32})
            {
                ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                              desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)),
                          HIPBLAS_STATUS_SUCCESS);
                ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                              desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
                          HIPBLAS_STATUS_SUCCESS);
                if(GetParam())
                    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                                  desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)),
                              HIPBLAS_STATUS_SUCCESS);
            }

            // Obtain a real FP32 candidate, then validate it directly for XF32.
            // This reaches the support check even when GridBased selection succeeds.
            int returned = 0;
            ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                      fp32,
                                                      aLayout,
                                                      bLayout,
                                                      cdLayout,
                                                      cdLayout,
                                                      pref,
                                                      1,
                                                      &candidate,
                                                      &returned),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(returned, 1);
            ASSERT_EQ(candidate.state, HIPBLAS_STATUS_SUCCESS);
            if(candidate.workspaceSize)
                ASSERT_EQ(hipMalloc(&workspace, candidate.workspaceSize), hipSuccess);
        }

        void TearDown() override
        {
            for(void* ptr : {static_cast<void*>(a),
                             static_cast<void*>(b),
                             static_cast<void*>(c),
                             static_cast<void*>(d),
                             static_cast<void*>(bias),
                             workspace})
                if(ptr)
                    static_cast<void>(hipFree(ptr));
            if(pref)
                hipblasLtMatmulPreferenceDestroy(pref);
            for(auto layout : {aLayout, bLayout, cdLayout})
                if(layout)
                    hipblasLtMatrixLayoutDestroy(layout);
            for(auto desc : {fp32, xf32})
                if(desc)
                    hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
        }

        void setProblem(hipblaslt_ext::Gemm& gemm)
        {
            hipblaslt_ext::GemmEpilogue epilogue;
            epilogue.setMode(GetParam() ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT);
            hipblaslt_ext::GemmInputs inputs;
            inputs.setA(a);
            inputs.setB(b);
            inputs.setC(c);
            inputs.setD(d);
            inputs.setAlpha(&alpha);
            inputs.setBeta(&beta);
            if(GetParam())
                inputs.setBias(bias);
            ASSERT_EQ(gemm.setProblem(m, n, k, 1, epilogue, inputs), HIPBLAS_STATUS_SUCCESS);
        }
    };

    TEST_P(Xf32Fallback, smoke_MatmulSupportAcceptsFp32Candidate)
    {
        size_t required = 0;
        ASSERT_EQ(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                       xf32,
                                                       &alpha,
                                                       aLayout,
                                                       bLayout,
                                                       &beta,
                                                       cdLayout,
                                                       cdLayout,
                                                       candidate.algo,
                                                       required),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_LE(required, candidate.workspaceSize);
        ASSERT_EQ(hipblasLtMatmul(handle,
                                  xf32,
                                  &alpha,
                                  a,
                                  aLayout,
                                  b,
                                  bLayout,
                                  &beta,
                                  c,
                                  cdLayout,
                                  d,
                                  cdLayout,
                                  &candidate.algo,
                                  workspace,
                                  candidate.workspaceSize,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        std::vector<float> output(m * n);
        ASSERT_EQ(hipMemcpy(output.data(), d, output.size() * sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        for(int64_t col = 0; col < n; ++col)
            for(int64_t row = 0; row < m; ++row)
            {
                double expected = GetParam() ? hBias[row] : 0.0;
                for(int64_t inner = 0; inner < k; ++inner)
                    expected += static_cast<double>(ha[row * k + inner]) * hb[col * k + inner];
                // All operands and sums are exactly representable binary fractions.
                EXPECT_EQ(output[col * m + row], expected) << row << ", " << col;
            }
    }

    TEST_P(Xf32Fallback, smoke_GemmSupportAcceptsFp32Candidate)
    {
        hipblaslt_ext::Gemm gemm(handle,
                                 HIPBLAS_OP_T,
                                 HIPBLAS_OP_N,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIPBLAS_COMPUTE_32F_FAST_TF32);
        ASSERT_NO_FATAL_FAILURE(setProblem(gemm));
        size_t required = 0;
        EXPECT_EQ(gemm.isAlgoSupported(candidate.algo, required), HIPBLAS_STATUS_SUCCESS);
        EXPECT_LE(required, candidate.workspaceSize);
    }

    TEST_P(Xf32Fallback, smoke_FallbackCheckPreservesPreferredAlgorithm)
    {
        hipblaslt_ext::Gemm gemm(handle,
                                 HIPBLAS_OP_T,
                                 HIPBLAS_OP_N,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIP_R_32F,
                                 HIPBLAS_COMPUTE_32F_FAST_TF32);
        ASSERT_NO_FATAL_FAILURE(setProblem(gemm));
        hipblaslt_ext::GemmPreference preference;
        preference.setMaxWorkspaceBytes(maxWorkspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> before, after;
        ASSERT_EQ(gemm.algoGetHeuristic(1, preference, before), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(before.size(), 1);
        size_t required = 0;
        ASSERT_EQ(gemm.isAlgoSupported(candidate.algo, required), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(gemm.algoGetHeuristic(1, preference, after), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(after.size(), 1);
        // On devices with native XF32, checking a FP32 candidate must not change
        // the next query to FP32. gfx12 continues to use its FP32 fallback.
        EXPECT_EQ(hipblaslt_ext::getIndexFromAlgo(before[0].algo),
                  hipblaslt_ext::getIndexFromAlgo(after[0].algo));
    }

    INSTANTIATE_TEST_SUITE_P(Bias, Xf32Fallback, ::testing::Bool());
} // namespace
