/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Complex contractions with alpha and beta stored at addresses that are not aligned to the
// size of the complex scalar type, as when a complex double is passed as double[2].

#include <complex>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

namespace
{
    // Owns everything runComplexBilinear creates, so an early return from a failed ASSERT
    // releases it as well.
    struct ContractionResources
    {
        hiptensorHandle_t              handle = nullptr;
        hiptensorTensorDescriptor_t    descA = nullptr, descB = nullptr, descD = nullptr;
        hiptensorOperationDescriptor_t opDesc = nullptr;
        hiptensorPlanPreference_t      pref   = nullptr;
        hiptensorPlan_t                plan   = nullptr;
        void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dWs = nullptr;

        ~ContractionResources()
        {
            for(void* buffer : {dA, dB, dC, dD, dWs})
            {
                if(buffer != nullptr)
                {
                    (void)hipFree(buffer);
                }
            }
            if(plan != nullptr)
            {
                hiptensorDestroyPlan(plan);
            }
            if(pref != nullptr)
            {
                hiptensorDestroyPlanPreference(pref);
            }
            if(opDesc != nullptr)
            {
                hiptensorDestroyOperationDescriptor(opDesc);
            }
            for(auto desc : {descA, descB, descD})
            {
                if(desc != nullptr)
                {
                    hiptensorDestroyTensorDescriptor(desc);
                }
            }
            if(handle != nullptr)
            {
                hiptensorDestroy(handle);
            }
        }
    };

    // D[m,n] = alpha * sum_k A[m,k] * B[n,k] + beta * C[m,n], column-major (first mode fastest),
    // with alpha and beta read from scalarBuf + scalarOffset and scalarBuf + 32 + scalarOffset.
    template <typename Real>
    void runComplexBilinear(hiptensorDataType_t          dataType,
                            hiptensorComputeDescriptor_t computeType,
                            size_t                       scalarOffset,
                            double                       tolerance)
    {
        using Complex = std::complex<Real>;

        constexpr int64_t M = 32;
        constexpr int64_t N = 24;
        constexpr int64_t K = 16;

        std::vector<Complex> hA(M * K), hB(N * K), hC(M * N), hD(M * N);
        for(int64_t i = 0; i < M * K; i++)
        {
            hA[i] = Complex(Real((i % 7) - 3) / 4, Real((i % 5) - 2) / 8);
        }
        for(int64_t i = 0; i < N * K; i++)
        {
            hB[i] = Complex(Real((i % 3) - 1) / 2, Real((i % 11) - 5) / 16);
        }
        for(int64_t i = 0; i < M * N; i++)
        {
            hC[i] = Complex(Real((i % 13) - 6) / 8, Real((i % 9) - 4) / 4);
        }

        Complex const alpha(Real(1.5), Real(-0.75));
        Complex const beta(Real(-0.5), Real(1.25));

        // The scalars sit scalarOffset bytes past a 16-byte boundary.
        alignas(16) unsigned char scalarBuf[64] = {};
        void*                     alphaPtr      = scalarBuf + scalarOffset;
        void*                     betaPtr       = scalarBuf + 32 + scalarOffset;
        std::memcpy(alphaPtr, &alpha, sizeof(alpha));
        std::memcpy(betaPtr, &beta, sizeof(beta));

        ContractionResources r;
        ASSERT_EQ(hiptensorCreate(&r.handle), HIPTENSOR_STATUS_SUCCESS);

        int64_t lensA[2]    = {M, K};
        int64_t stridesA[2] = {1, M};
        int64_t lensB[2]    = {N, K};
        int64_t stridesB[2] = {1, N};
        int64_t lensD[2]    = {M, N};
        int64_t stridesD[2] = {1, M};
        int32_t modeA[2]    = {'m', 'k'};
        int32_t modeB[2]    = {'n', 'k'};
        int32_t modeD[2]    = {'m', 'n'};

        ASSERT_EQ(
            hiptensorCreateTensorDescriptor(r.handle, &r.descA, 2, lensA, stridesA, dataType, 128),
            HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(
            hiptensorCreateTensorDescriptor(r.handle, &r.descB, 2, lensB, stridesB, dataType, 128),
            HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(
            hiptensorCreateTensorDescriptor(r.handle, &r.descD, 2, lensD, stridesD, dataType, 128),
            HIPTENSOR_STATUS_SUCCESS);

        // The library decides whether this device supports the compute type.
        auto const status = hiptensorCreateContraction(r.handle,
                                                       &r.opDesc,
                                                       r.descA,
                                                       modeA,
                                                       HIPTENSOR_OP_IDENTITY,
                                                       r.descB,
                                                       modeB,
                                                       HIPTENSOR_OP_IDENTITY,
                                                       r.descD,
                                                       modeD,
                                                       HIPTENSOR_OP_IDENTITY,
                                                       r.descD,
                                                       modeD,
                                                       computeType);
        if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            GTEST_SKIP() << "This compute type is not supported on this device";
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

        ASSERT_EQ(hiptensorCreatePlanPreference(
                      r.handle, &r.pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);

        uint64_t wsSize = 0;
        ASSERT_EQ(hiptensorEstimateWorkspaceSize(
                      r.handle, r.opDesc, r.pref, HIPTENSOR_WORKSPACE_DEFAULT, &wsSize),
                  HIPTENSOR_STATUS_SUCCESS);

        ASSERT_EQ(hiptensorCreatePlan(r.handle, &r.plan, r.opDesc, r.pref, wsSize),
                  HIPTENSOR_STATUS_SUCCESS);

        size_t const bytesA = sizeof(Complex) * hA.size();
        size_t const bytesB = sizeof(Complex) * hB.size();
        size_t const bytesD = sizeof(Complex) * hD.size();

        ASSERT_EQ(hipMalloc(&r.dA, bytesA), hipSuccess);
        ASSERT_EQ(hipMalloc(&r.dB, bytesB), hipSuccess);
        ASSERT_EQ(hipMalloc(&r.dC, bytesD), hipSuccess);
        ASSERT_EQ(hipMalloc(&r.dD, bytesD), hipSuccess);
        if(wsSize > 0)
        {
            ASSERT_EQ(hipMalloc(&r.dWs, wsSize), hipSuccess);
        }
        ASSERT_EQ(hipMemcpy(r.dA, hA.data(), bytesA, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(r.dB, hB.data(), bytesB, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(r.dC, hC.data(), bytesD, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(r.dD, 0, bytesD), hipSuccess);

        ASSERT_EQ(hiptensorContract(r.handle,
                                    r.plan,
                                    alphaPtr,
                                    r.dA,
                                    r.dB,
                                    betaPtr,
                                    r.dC,
                                    r.dD,
                                    r.dWs,
                                    wsSize,
                                    nullptr),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipMemcpy(hD.data(), r.dD, bytesD, hipMemcpyDeviceToHost), hipSuccess);

        using Ref = std::complex<double>;
        for(int64_t n = 0; n < N; n++)
        {
            for(int64_t m = 0; m < M; m++)
            {
                Ref acc = 0;
                for(int64_t k = 0; k < K; k++)
                {
                    acc += Ref(hA[m + k * M]) * Ref(hB[n + k * N]);
                }
                Ref const expected = Ref(alpha) * acc + Ref(beta) * Ref(hC[m + n * M]);
                Ref const actual(hD[m + n * M]);
                EXPECT_LE(std::abs(actual - expected), tolerance * (1.0 + std::abs(expected)))
                    << "D[" << m << "," << n << "] = " << actual << ", expected " << expected;
            }
        }
    }
} // namespace

// alpha and beta 8 bytes past a 16-byte boundary, as for a complex double passed as double[2].
TEST(ScalarAlignmentTest, ComplexF64ContractionScalarsAt8Mod16)
{
    runComplexBilinear<double>(HIPTENSOR_C_64F, HIPTENSOR_COMPUTE_DESC_C64F, 8, 1e-12);
}

// alpha and beta 4 bytes past an 8-byte boundary, as for a complex float passed as float[2].
TEST(ScalarAlignmentTest, ComplexF32ContractionScalarsAt4Mod8)
{
    runComplexBilinear<float>(HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, 4, 1e-4);
}
