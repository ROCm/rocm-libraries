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

}
