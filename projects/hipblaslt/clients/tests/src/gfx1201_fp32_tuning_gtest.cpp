// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace
{
    struct Fp32TuningCase
    {
        int    padding;
        float  alpha;
        float  beta;
        size_t workspaceBytes;
        bool   biasRelu = false;
    };

    void PrintTo(const Fp32TuningCase& param, std::ostream* out)
    {
        *out << "padding=" << param.padding << ", alpha=" << param.alpha << ", beta=" << param.beta
             << ", workspaceBytes=" << param.workspaceBytes << ", biasRelu=" << param.biasRelu;
    }

    // The pre_checkin token selects these tests in the standard CI tier.
    class Gfx1201Fp32Tuning_pre_checkin : public ::testing::TestWithParam<Fp32TuningCase>
    {
    protected:
        hipblasLtHandle_t           handle = nullptr;
        hipblasLtMatmulDesc_t       op     = nullptr;
        hipblasLtMatrixLayout_t     a = nullptr, b = nullptr, c = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        std::vector<void*>          allocations;

        void SetUp() override
        {
            int count = 0;
            ASSERT_EQ(hipGetDeviceCount(&count), hipSuccess);
            if(count == 0)
                GTEST_SKIP() << "No GPU available";
            int device = 0;
            ASSERT_EQ(hipGetDevice(&device), hipSuccess);
            hipDeviceProp_t props{};
            ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
            if(std::string(props.gcnArchName).substr(0, 7) != "gfx1201")
                GTEST_SKIP() << "gfx1201-specific tuning";
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            for(auto ptr : allocations)
                EXPECT_EQ(hipFree(ptr), hipSuccess);
            if(pref)
                EXPECT_EQ(hipblasLtMatmulPreferenceDestroy(pref), HIPBLAS_STATUS_SUCCESS);
            if(c)
                EXPECT_EQ(hipblasLtMatrixLayoutDestroy(c), HIPBLAS_STATUS_SUCCESS);
            if(b)
                EXPECT_EQ(hipblasLtMatrixLayoutDestroy(b), HIPBLAS_STATUS_SUCCESS);
            if(a)
                EXPECT_EQ(hipblasLtMatrixLayoutDestroy(a), HIPBLAS_STATUS_SUCCESS);
            if(op)
                EXPECT_EQ(hipblasLtMatmulDescDestroy(op), HIPBLAS_STATUS_SUCCESS);
            if(handle)
                EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
        }

        void* allocate(size_t bytes)
        {
            void* ptr = nullptr;
            auto  err = hipMalloc(&ptr, bytes);
            EXPECT_EQ(err, hipSuccess);
            if(err == hipSuccess)
                allocations.push_back(ptr);
            return ptr;
        }
    };

    TEST_P(Gfx1201Fp32Tuning_pre_checkin, SelectsTunedKernelAndComputesFullOutput)
    {
        constexpr int M = 128, N = 4096, K = 4096;
        const auto    param = GetParam();
        const int     lda = M + param.padding, ldb = K + param.padding;
        const int     ldc = M + param.padding;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        std::vector<float> hostBias(M);
        if(param.biasRelu)
        {
            for(int i = 0; i < M; ++i)
                hostBias[i] = i % 3 - 1;
            auto deviceBias = static_cast<float*>(allocate(M * sizeof(float)));
            ASSERT_NE(deviceBias, nullptr);
            ASSERT_EQ(
                hipMemcpy(deviceBias, hostBias.data(), M * sizeof(float), hipMemcpyHostToDevice),
                hipSuccess);
            const auto epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
            const auto biasType = HIP_R_32F;
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                          op, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                          op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &deviceBias, sizeof(deviceBias)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                          op, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType)),
                      HIPBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&a, HIP_R_32F, M, K, lda), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&b, HIP_R_32F, K, N, ldb), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, M, N, ldc), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &param.workspaceBytes,
                                                        sizeof(param.workspaceBytes)),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatmulHeuristicResult_t result{};
        int                              returned = 0;
        ASSERT_EQ(
            hipblasLtMatmulAlgoGetHeuristic(handle, op, a, b, c, c, pref, 1, &result, &returned),
            HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(returned, 1);
        ASSERT_EQ(result.state, HIPBLAS_STATUS_SUCCESS);
        ASSERT_LE(result.workspaceSize, param.workspaceBytes);
        if(param.workspaceBytes != 0)
        {
            // No global algorithm index: indices change with the compiled pool.
            // Removing the exact entry must fail this dispatch assertion even
            // though the old default kernel is numerically correct.
            const auto name = hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo);
            EXPECT_NE(name.find("_MT128x64x16_"), std::string::npos) << name;
            EXPECT_EQ(hipblaslt_ext::matmulIsTuned(handle, op, a, b, c, c), 1);
        }

        constexpr float    paddingValue = -12345.0f;
        std::vector<float> hostA(size_t(lda) * K, paddingValue);
        std::vector<float> hostB(size_t(ldb) * N, paddingValue);
        std::vector<float> hostC(size_t(ldc) * N, paddingValue);
        std::vector<float> hostD(size_t(ldc) * N, paddingValue);
        // A(i,p) = row(i)*sign(p), B(p,j) = sign(p)*column(j).
        // The closed-form FP64 oracle checks EVERY output without another GEMM.
        // Integer inputs and alpha/beta make the result exactly representable
        // in FP32, independent of accumulation order (max magnitude < 2^24).
        for(int p = 0; p < K; ++p)
            for(int i = 0; i < M; ++i)
                hostA[size_t(p) * lda + i] = (i % 7 - 3) * (p % 2 ? -1 : 1);
        for(int j = 0; j < N; ++j)
        {
            for(int p = 0; p < K; ++p)
                hostB[size_t(j) * ldb + p] = (j % 5 - 2) * (p % 2 ? -1 : 1);
            for(int i = 0; i < M; ++i)
                hostC[size_t(j) * ldc + i] = (i + 3 * j) % 11 - 5;
        }
        auto deviceA = static_cast<float*>(allocate(hostA.size() * sizeof(float)));
        auto deviceB = static_cast<float*>(allocate(hostB.size() * sizeof(float)));
        auto deviceC = static_cast<float*>(allocate(hostC.size() * sizeof(float)));
        auto deviceD = static_cast<float*>(allocate(hostD.size() * sizeof(float)));
        ASSERT_NE(deviceA, nullptr);
        ASSERT_NE(deviceB, nullptr);
        ASSERT_NE(deviceC, nullptr);
        ASSERT_NE(deviceD, nullptr);
        void* workspace = nullptr;
        if(result.workspaceSize)
        {
            workspace = allocate(result.workspaceSize);
            ASSERT_NE(workspace, nullptr);
        }
        ASSERT_EQ(
            hipMemcpy(deviceA, hostA.data(), hostA.size() * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(deviceB, hostB.data(), hostB.size() * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(deviceC, hostC.data(), hostC.size() * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);
        for(int repeat = 0; repeat < 3; ++repeat)
        {
            for(int j = 0; j < N; ++j)
                for(int i = 0; i < M; ++i)
                    hostD[size_t(j) * ldc + i] = NAN;
            ASSERT_EQ(
                hipMemcpy(
                    deviceD, hostD.data(), hostD.size() * sizeof(float), hipMemcpyHostToDevice),
                hipSuccess);
            ASSERT_EQ(hipblasLtMatmul(handle,
                                      op,
                                      &param.alpha,
                                      deviceA,
                                      a,
                                      deviceB,
                                      b,
                                      &param.beta,
                                      deviceC,
                                      c,
                                      deviceD,
                                      c,
                                      &result.algo,
                                      workspace,
                                      result.workspaceSize,
                                      nullptr),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                hipMemcpy(
                    hostD.data(), deviceD, hostD.size() * sizeof(float), hipMemcpyDeviceToHost),
                hipSuccess);
            for(int j = 0; j < N; ++j)
                for(int i = 0; i < ldc; ++i)
                {
                    double expected = i < M ? double(param.alpha) * K * (i % 7 - 3) * (j % 5 - 2)
                                                  + double(param.beta) * hostC[size_t(j) * ldc + i]
                                            : paddingValue;
                    if(i < M && param.biasRelu)
                        expected = std::max(0.0, expected + hostBias[i]);
                    ASSERT_EQ(double(hostD[size_t(j) * ldc + i]), expected)
                        << "repeat=" << repeat << " row=" << i << " column=" << j;
                }
        }
    }

    INSTANTIATE_TEST_SUITE_P(_,
                             Gfx1201Fp32Tuning_pre_checkin,
                             ::testing::Values(Fp32TuningCase{0, 1.0f, 0.0f, 64 * 1024 * 1024},
                                               Fp32TuningCase{3, 2.0f, -2.0f, 64 * 1024 * 1024},
                                               Fp32TuningCase{
                                                   3, 2.0f, -2.0f, 64 * 1024 * 1024, true},
                                               Fp32TuningCase{0, 1.0f, 1.0f, 0}));
} // namespace
