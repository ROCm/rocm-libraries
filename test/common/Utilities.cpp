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
        AssertFatal(floatA.size() % AX.size() == 0 && floatA.size() / AX.size() == 32,
                    "Matrix A size must be 32 times of the scale vector size.",
                    ShowValue(floatA.size()),
                    ShowValue(AX.size()));
        AssertFatal(floatB.size() % BX.size() == 0 && floatB.size() / BX.size() == 32,
                    "Matrix B size must be 32 times of the scale vector size.",
                    ShowValue(floatB.size()),
                    ShowValue(BX.size()));

        auto scaledA = floatA;
        auto scaledB = floatB;

        if(transA)
        {
#pragma omp parallel for
            for(size_t mk = 0; mk < M * K; ++mk)
            {
                auto  m      = mk / K;
                auto  k      = mk % K;
                auto  idx    = m * (K / 32) + (k / 32);
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
                auto  idx    = (k / 32) * M + m;
                float aScale = scaleToFloat(AX[idx]);
                scaledA[mk] *= aScale;
            }
        }

        if(transB)
        {
#pragma omp parallel for
            for(size_t kn = 0; kn < K * N; ++kn)
            {
                auto  k      = kn / N;
                auto  n      = kn % N;
                auto  idx    = (k / 32) * N + n;
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
                auto  idx    = n * (K / 32) + (k / 32);
                float bScale = scaleToFloat(BX[idx]);
                scaledB[kn] *= bScale;
            }
        }

        CPUMM(D, C, scaledA, scaledB, M, N, K, alpha, beta, transA, transB);
    }
}
