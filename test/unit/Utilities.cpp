/**
 * Test suite utilites.
 */

#include <cblas.h>

#include "Utilities.hpp"

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
               bool                      transposeB)
    {
        D = C;
        cblas_sgemm(CblasColMajor,
                    CblasNoTrans,
                    transposeB ? CblasTrans : CblasNoTrans,
                    M,
                    N,
                    K,
                    alpha,
                    A.data(),
                    M,
                    B.data(),
                    transposeB ? N : K,
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
               bool                       transposeB)
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
                    CblasNoTrans,
                    transposeB ? CblasTrans : CblasNoTrans,
                    M,
                    N,
                    K,
                    alpha,
                    floatA.data(),
                    M,
                    floatB.data(),
                    transposeB ? N : K,
                    beta,
                    floatD.data(),
                    M);

#pragma omp parallel for
        for(std::size_t i = 0; i != floatD.size(); ++i)
        {
            D[i] = __float2half(floatD[i]);
        }
    }

}
