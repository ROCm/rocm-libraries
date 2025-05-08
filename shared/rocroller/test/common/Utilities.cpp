/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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
                     bool                        transB)
    {
        constexpr int elementsPerMXBlock = 32;

        auto scaledA = floatA;
        auto scaledB = floatB;

        if(AX.size() > 1)
        {
            AssertFatal(floatA.size() % AX.size() == 0
                            && floatA.size() / AX.size() == elementsPerMXBlock,
                        "Matrix A size must be 32 times of the scale vector size.",
                        ShowValue(floatA.size()),
                        ShowValue(AX.size()));

            if(transA)
            {
#pragma omp parallel for
                for(size_t mk = 0; mk < M * K; ++mk)
                {
                    auto  m      = mk / K;
                    auto  k      = mk % K;
                    auto  idx    = m * (K / elementsPerMXBlock) + (k / elementsPerMXBlock);
                    float aScale = scaleToFloat(AX[idx]);
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
                    auto  idx    = (k / elementsPerMXBlock) * M + m;
                    float aScale = scaleToFloat(AX[idx]);
                    scaledA[mk] *= aScale;
                }
            }
        }
        else if(AX.size() == 1)
        {
            float aScale = scaleToFloat(AX[0]);
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
                            && floatB.size() / BX.size() == elementsPerMXBlock,
                        "Matrix B size must be 32 times of the scale vector size.",
                        ShowValue(floatB.size()),
                        ShowValue(BX.size()));

            if(transB)
            {
#pragma omp parallel for
                for(size_t kn = 0; kn < K * N; ++kn)
                {
                    auto  k      = kn / N;
                    auto  n      = kn % N;
                    auto  idx    = (k / elementsPerMXBlock) * N + n;
                    float bScale = scaleToFloat(BX[idx]);
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
                    auto  idx    = n * (K / elementsPerMXBlock) + (k / elementsPerMXBlock);
                    float bScale = scaleToFloat(BX[idx]);
                    scaledB[kn] *= bScale;
                }
            }
        }
        else if(BX.size() == 1)
        {
            float bScale = scaleToFloat(BX[0]);
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
