/**
 * Test suite utilites.
 */

#include <cblas.h>

#include "Utilities.hpp"

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

    template <typename F8Type>
    void CPUMM_F8(std::vector<float>&        D,
                  const std::vector<float>&  C,
                  const std::vector<F8Type>& A,
                  const std::vector<F8Type>& B,
                  int                        M,
                  int                        N,
                  int                        K,
                  float                      alpha,
                  float                      beta,
                  bool                       transA,
                  bool                       transB)
    {
        static_assert(
            std::is_same_v<F8Type, rocRoller::FP8> || std::is_same_v<F8Type, rocRoller::BF8>);

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
            floatD[i] = C[i];
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
            D[i] = floatD[i];
        }
    }

    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<BF8>&   A,
               const std::vector<BF8>&   B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA,
               bool                      transB)
    {
        CPUMM_F8(D, C, A, B, M, N, K, alpha, beta, transA, transB);
    }

    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<FP8>&   A,
               const std::vector<FP8>&   B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA,
               bool                      transB)
    {
        CPUMM_F8(D, C, A, B, M, N, K, alpha, beta, transA, transB);
    }

    void CPUMM(std::vector<float>&        D,
               const std::vector<float>&  C,
               const std::vector<FP6x16>& A,
               const std::vector<FP6x16>& B,
               int                        M,
               int                        N,
               int                        K,
               float                      alpha,
               float                      beta,
               bool                       transA,
               bool                       transB)
    {
        auto c_A = unpackFP6x16(A);
        auto c_B = unpackFP6x16(B);
        CPUMM(D, C, c_A, c_B, M, N, K, 1.0, 0.0, transA, transB);
    }

    void CPUMM(std::vector<float>&        D,
               const std::vector<float>&  C,
               const std::vector<BF6x16>& A,
               const std::vector<BF6x16>& B,
               int                        M,
               int                        N,
               int                        K,
               float                      alpha,
               float                      beta,
               bool                       transA,
               bool                       transB)
    {
        auto c_A = unpackBF6x16(A);
        auto c_B = unpackBF6x16(B);
        CPUMM(D, C, c_A, c_B, M, N, K, 1.0, 0.0, transA, transB);
    }

    void CPUMM(std::vector<float>&       D,
               const std::vector<float>& C,
               const std::vector<FP4x8>& A,
               const std::vector<FP4x8>& B,
               int                       M,
               int                       N,
               int                       K,
               float                     alpha,
               float                     beta,
               bool                      transA,
               bool                      transB)
    {
        auto c_A = unpackFP4x8(A);
        auto c_B = unpackFP4x8(B);

        CPUMM(D, C, c_A, c_B, M, N, K, 1.0, 0.0, transA, transB);
    }

}
