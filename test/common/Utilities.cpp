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

    void CPUMM(std::vector<__half>&       D,
               const std::vector<__half>& C,
               const std::vector<__half>& A,
               const std::vector<__half>& B,
               int                        M,
               int                        N,
               int                        K,
               float                      alpha,
               float                      beta,
               bool                       transA,
               bool                       transB)
    {
        std::vector<float> floatA(A.size());
        std::vector<float> floatB(B.size());
        std::vector<float> floatD(C.size());

#pragma omp parallel for
        for(std::size_t i = 0; i != A.size(); ++i)
        {
            floatA[i] = __half2float(A[i]);
        }

#pragma omp parallel for
        for(std::size_t i = 0; i != B.size(); ++i)
        {
            floatB[i] = __half2float(B[i]);
        }

#pragma omp parallel for
        for(std::size_t i = 0; i != C.size(); ++i)
        {
            floatD[i] = __half2float(C[i]);
        }

        cblas_sgemm(CblasColMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    M,
                    N,
                    K,
                    alpha,
                    floatA.data(),
                    transA ? K : M,
                    floatB.data(),
                    transB ? N : K,
                    beta,
                    floatD.data(),
                    M);

#pragma omp parallel for
        for(std::size_t i = 0; i != floatD.size(); ++i)
        {
            D[i] = __float2half(floatD[i]);
        }
    }

    void CPUMM(std::vector<BFloat16>&       D,
               const std::vector<BFloat16>& C,
               const std::vector<BFloat16>& A,
               const std::vector<BFloat16>& B,
               int                          M,
               int                          N,
               int                          K,
               float                        alpha,
               float                        beta,
               bool                         transA,
               bool                         transB)
    {
        std::vector<float> floatA(A.size());
        std::vector<float> floatB(B.size());
        std::vector<float> floatD(C.size());

#pragma omp parallel for
        for(std::size_t i = 0; i != A.size(); ++i)
        {
            floatA[i] = float(A[i]);
        }

#pragma omp parallel for
        for(std::size_t i = 0; i != B.size(); ++i)
        {
            floatB[i] = float(B[i]);
        }

#pragma omp parallel for
        for(std::size_t i = 0; i != C.size(); ++i)
        {
            floatD[i] = float(C[i]);
        }

        cblas_sgemm(CblasColMajor,
                    transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans,
                    M,
                    N,
                    K,
                    alpha,
                    floatA.data(),
                    transA ? K : M,
                    floatB.data(),
                    transB ? N : K,
                    beta,
                    floatD.data(),
                    M);

#pragma omp parallel for
        for(std::size_t i = 0; i != floatD.size(); ++i)
        {
            D[i] = BFloat16(floatD[i]);
        }
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

        auto idxA = [transA, M, K](auto i, auto j) {
            if(transA)
                return i * K + j;
            else
                return j * M + i;
        };

        auto idxB = [transB, N, K](auto i, auto j) {
            if(!transB)
                return i * K + j;
            else
                return j * N + i;
        };

        auto idxScaleA = [transA, M, K](auto i, auto j) {
            if(transA)
                return i * (K / 32) + (j / 32);
            else
                return (j / 32) * M + i;
        };

        auto idxScaleB = [transB, N, K](auto i, auto j) {
            if(!transB)
                return i * (K / 32) + (j / 32);
            else
                return (j / 32) * N + i;
        };

        for(int m = 0; m < M; m++)
        {
            for(int n = 0; n < N; n++)
            {
                double scaledAcc = 0.0;

                for(int k = 0; k < K; k++)
                {
                    float aVal = floatA[idxA(m, k)];
                    float bVal = floatB[idxB(n, k)];

                    float aScale = scaleToFloat(AX[idxScaleA(m, k)]);
                    float bScale = scaleToFloat(BX[idxScaleB(n, k)]);
                    scaledAcc += aScale * aVal * bScale * bVal;
                }

                D[n * M + m] = alpha * scaledAcc + beta * C[n * M + m];
            }
        }
    }
}
