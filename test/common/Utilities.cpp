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
}
