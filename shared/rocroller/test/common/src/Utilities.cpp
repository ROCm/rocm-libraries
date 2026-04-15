// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Test suite utilities.
 */

#include <cblas.h>

#include "common/Utilities.hpp"

int countSubstring(const std::string& str, const std::string& sub)
{
    if(sub.length() == 0)
        return 0;
    int count = 0;
    for(size_t offset = str.find(sub); offset != std::string::npos;
        offset        = str.find(sub, offset + sub.length()))
    {
        ++count;
    }
    return count;
}

std::shared_ptr<void> make_shared_device(rocRoller::CommandArgumentValue const& arg)
{
    auto visitor = [](auto const& arg) -> std::shared_ptr<void> {
        using T = std::decay_t<decltype(arg)>;

        auto rv = make_shared_device<int8_t>(sizeof(T));

        auto result = hipMemcpy(rv.get(), &arg, sizeof(T), hipMemcpyHostToDevice);
        if(result != hipSuccess)
        {
            throw std::runtime_error(hipGetErrorString(result));
        }
        return rv;
    };
    return std::visit(visitor, arg);
}

namespace rocRoller
{
    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<float>& A,
               const std::vector<float>& B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA,
               bool                      transB)
    {
        D = C;
        cblas_sgemm(CblasColMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    M,
                    N,
                    K,
                    alpha,
                    A.data(),
                    transA ? K : M,
                    B.data(),
                    transB ? N : K,
                    beta,
                    D.data(),
                    M);
    }

    /**
     * @brief CPU reference solution for scaled matrix multiply
     *
     * @param D     output buffer
     * @param C     input buffer
     * @param floatA the unpacked F8F6F4 values in FP32 format for matrix A
     * @param floatB the unpacked F8F6F4 values in FP32 format for matrix B
     * @param AX    vector for scale values of matrix A
     * @param BX    vector for scale values of matrix A
     * @param M     A matrix is M * K
     * @param N     B matrix is K * N
     * @param K
     * @param alpha scalar value: alpha * A * B
     * @param beta  scalar value: beta * C
     */
    void ScaledCPUMM(std::vector<float>&         D,
                     const std::vector<float>&   C,
                     const std::vector<float>&   floatA,
                     const std::vector<float>&   floatB,
                     const std::vector<uint8_t>& AX,
                     const std::vector<uint8_t>& BX,
                     int                         M,
                     int                         N,
                     int                         K,
                     float                       alpha,
                     float                       beta,
                     bool                        transA,
                     bool                        transB,
                     const uint                  scaleBlockSize,
                     const DataType              scaleTypeA,
                     const DataType              scaleTypeB)
    {
        auto scaledA = floatA;
        auto scaledB = floatB;

        if(AX.size() > 1)
        {
            AssertFatal(floatA.size() % AX.size() == 0
                            && floatA.size() / AX.size() == scaleBlockSize,
                        "Matrix A size must be scaleBlockSize times the scale vector size.",
                        ShowValue(floatA.size()),
                        ShowValue(AX.size()),
                        ShowValue(scaleBlockSize));

            if(transA)
            {
#pragma omp parallel for
                for(size_t mk = 0; mk < M * K; ++mk)
                {
                    auto  m      = mk / K;
                    auto  k      = mk % K;
                    auto  idx    = m * (K / scaleBlockSize) + (k / scaleBlockSize);
                    float aScale = scaleToFloat(scaleTypeA, AX[idx]);
                    scaledA[mk] *= aScale;
                }
            }
            else
            {
#pragma omp parallel for
                for(size_t mk = 0; mk < M * K; ++mk)
                {
                    auto  m      = mk % M;
                    auto  k      = mk / M;
                    auto  idx    = (k / scaleBlockSize) * M + m;
                    float aScale = scaleToFloat(scaleTypeA, AX[idx]);
                    scaledA[mk] *= aScale;
                }
            }
        }
        else if(AX.size() == 1)
        {
            float aScale = scaleToFloat(scaleTypeA, AX[0]);
#pragma omp parallel for
            for(size_t mk = 0; mk < M * K; ++mk)
            {
                scaledA[mk] *= aScale;
            }
        }
        else
        {
            Throw<FatalError>("Invalid AX.");
        }

        if(BX.size() > 1)
        {
            AssertFatal(floatB.size() % BX.size() == 0
                            && floatB.size() / BX.size() == scaleBlockSize,
                        "Matrix B size must be scaleBlockSize times the scale vector size.",
                        ShowValue(floatB.size()),
                        ShowValue(BX.size()),
                        ShowValue(scaleBlockSize));

            if(transB)
            {
#pragma omp parallel for
                for(size_t kn = 0; kn < K * N; ++kn)
                {
                    auto  k      = kn / N;
                    auto  n      = kn % N;
                    auto  idx    = (k / scaleBlockSize) * N + n;
                    float bScale = scaleToFloat(scaleTypeB, BX[idx]);
                    scaledB[kn] *= bScale;
                }
            }
            else
            {
#pragma omp parallel for
                for(size_t kn = 0; kn < K * N; ++kn)
                {
                    auto  k      = kn % K;
                    auto  n      = kn / K;
                    auto  idx    = n * (K / scaleBlockSize) + (k / scaleBlockSize);
                    float bScale = scaleToFloat(scaleTypeB, BX[idx]);
                    scaledB[kn] *= bScale;
                }
            }
        }
        else if(BX.size() == 1)
        {
            float bScale = scaleToFloat(scaleTypeB, BX[0]);
#pragma omp parallel for
            for(size_t kn = 0; kn < K * N; ++kn)
            {
                scaledB[kn] *= bScale;
            }
        }
        else
        {
            Throw<FatalError>("Invalid BX.");
        }

        CPUMM(D, C, scaledA, scaledB, M, N, K, alpha, beta, transA, transB);
    }
}
