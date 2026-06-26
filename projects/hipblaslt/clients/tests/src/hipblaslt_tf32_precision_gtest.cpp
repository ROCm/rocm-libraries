// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Regression test for ROCM-3139.
//
// On gfx942 (MI300-class), HIPBLAS_COMPUTE_32F_FAST_TF32 must not silently route FP32
// GEMM through a lower-precision path that diverges from HIPBLAS_COMPUTE_32F. This test
// runs the same FP32 NN GEMM twice and requires the outputs to match within a tight bound.
//
// Placed in hipblaslt-test (not a YAML matmul case) because the assertion is a pairwise
// compute-type comparison, not a reference-GEMM unit check. The test name carries the
// `smoke` category token so TheRock PR CI (`--gtest_filter=*smoke*`) executes it on
// gfx942 hardware; it is skipped elsewhere.

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace
{
    constexpr int    kMatrixDim     = 2048;
    constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;
    constexpr float  kFixThreshold  = 1e-4f;

    bool gpuAvailable()
    {
        int deviceCount = 0;
        return hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0;
    }

    bool isGfx942()
    {
        char* archName = nullptr;
        if(hipblasLtGetArchName(&archName) != HIPBLAS_STATUS_SUCCESS || archName == nullptr)
            return false;
        const std::string arch(archName);
        free(archName);
        return arch.compare(0, 6, "gfx942") == 0;
    }

    void fillUniform01(std::vector<float>& data, uint32_t seed)
    {
        std::mt19937                         rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for(float& v : data)
            v = dist(rng);
    }

    void runFp32Gemm(hipblasLtHandle_t         handle,
                     hipStream_t               stream,
                     hipblasComputeType_t      computeType,
                     const std::vector<float>& hA,
                     const std::vector<float>& hB,
                     const std::vector<float>& hC,
                     std::vector<float>&       hD)
    {
        const int64_t m = kMatrixDim;
        const int64_t n = kMatrixDim;
        const int64_t k = kMatrixDim;

        float* dA = nullptr;
        float* dB = nullptr;
        float* dC = nullptr;
        float* dD = nullptr;
        void*  workspace = nullptr;

        hipblasLtMatrixLayout_t matA = nullptr;
        hipblasLtMatrixLayout_t matB = nullptr;
        hipblasLtMatrixLayout_t matC = nullptr;
        hipblasLtMatrixLayout_t matD = nullptr;
        hipblasLtMatmulDesc_t   matmul = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;

        hD.assign(static_cast<size_t>(m * n), 0.0f);

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dC, hC.size() * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMalloc(&dD, hD.size() * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMalloc(&workspace, kWorkspaceBytes), hipSuccess);

        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, m, k, m), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, k, n, k), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, m, n, m), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, m, n, m), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescCreate(&matmul, computeType, HIP_R_32F), HIPBLAS_STATUS_SUCCESS);

        const hipblasOperation_t opN = HIPBLAS_OP_N;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
                  HIPBLAS_STATUS_SUCCESS);

        const hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
        ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                      matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
                  HIPBLAS_STATUS_SUCCESS);

        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        const uint64_t workspacePref = kWorkspaceBytes;
        ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                      HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspacePref,
                                                      sizeof(workspacePref)),
                  HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulHeuristicResult_t heuristic{};
        int                              returnedAlgoCount = 0;
        ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                  matmul,
                                                  matA,
                                                  matB,
                                                  matC,
                                                  matD,
                                                  pref,
                                                  1,
                                                  &heuristic,
                                                  &returnedAlgoCount),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_GE(returnedAlgoCount, 1);
        ASSERT_EQ(heuristic.state, HIPBLAS_STATUS_SUCCESS);

        const size_t workspaceSize = std::max<size_t>(heuristic.workspaceSize, 1);

        ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dC, hC.data(), hC.size() * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

        ASSERT_EQ(hipblasLtMatmul(handle,
                                  matmul,
                                  &alpha,
                                  dA,
                                  matA,
                                  dB,
                                  matB,
                                  &beta,
                                  dC,
                                  matC,
                                  dD,
                                  matD,
                                  &heuristic.algo,
                                  workspace,
                                  workspaceSize,
                                  stream),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

        (void)hipblasLtMatmulPreferenceDestroy(pref);
        (void)hipblasLtMatmulDescDestroy(matmul);
        (void)hipblasLtMatrixLayoutDestroy(matA);
        (void)hipblasLtMatrixLayoutDestroy(matB);
        (void)hipblasLtMatrixLayoutDestroy(matC);
        (void)hipblasLtMatrixLayoutDestroy(matD);
        (void)hipFree(workspace);
        (void)hipFree(dD);
        (void)hipFree(dC);
        (void)hipFree(dB);
        (void)hipFree(dA);
    }

    TEST(HipblasltTf32Precision, smoke_FastTf32MatchesFullFp32OnGfx942)
    {
        if(!gpuAvailable())
            GTEST_SKIP() << "No GPU available";

        if(!isGfx942())
            GTEST_SKIP() << "ROCM-3139 applies to gfx942 only";

        hipblasLtHandle_t handle = nullptr;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

        const size_t elems = static_cast<size_t>(kMatrixDim) * static_cast<size_t>(kMatrixDim);
        std::vector<float> hA(elems);
        std::vector<float> hB(elems);
        std::vector<float> hC(elems, 0.0f);
        fillUniform01(hA, 42);
        fillUniform01(hB, 42);

        std::vector<float> outF32;
        std::vector<float> outTf32;
        runFp32Gemm(handle, stream, HIPBLAS_COMPUTE_32F, hA, hB, hC, outF32);
        runFp32Gemm(handle, stream, HIPBLAS_COMPUTE_32F_FAST_TF32, hA, hB, hC, outTf32);

        float maxDiff = 0.0f;
        for(size_t i = 0; i < elems; ++i)
            maxDiff = std::max(maxDiff, std::fabs(outTf32[i] - outF32[i]));

        EXPECT_LE(maxDiff, kFixThreshold)
            << "ROCM-3139: HIPBLAS_COMPUTE_32F_FAST_TF32 must match HIPBLAS_COMPUTE_32F on gfx942 "
               "(max diff was "
            << maxDiff << ")";

        (void)hipStreamDestroy(stream);
        (void)hipblasLtDestroy(handle);
    }

} // namespace
