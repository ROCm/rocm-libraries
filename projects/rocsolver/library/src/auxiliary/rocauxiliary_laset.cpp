#include "laset.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

//------------------------------------------------------------------------------
// Export C wrappers.
extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_slaset(rocblas_handle handle,
                                                 char const uplo,
                                                 rocblas_int const m,
                                                 rocblas_int const n,
                                                 float const alpha,
                                                 float const beta,

                                                 float* A,
                                                 rocblas_stride const shiftA,
                                                 rocblas_int const lda,
                                                 rocblas_stride const strideA,

                                                 rocblas_int const batch_count)
{
    return rocsolver::laset(handle, uplo, m, n, alpha, beta, A, shiftA, lda, strideA, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlaset(rocblas_handle handle,
                                                 char const uplo,
                                                 rocblas_int const m,
                                                 rocblas_int const n,
                                                 double const alpha,
                                                 double const beta,

                                                 double* A,
                                                 rocblas_stride const shiftA,
                                                 rocblas_int const lda,
                                                 rocblas_stride const strideA,

                                                 rocblas_int const batch_count)
{
    return rocsolver::laset(handle, uplo, m, n, alpha, beta, A, shiftA, lda, strideA, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_claset(rocblas_handle handle,
                                                 char const uplo,
                                                 rocblas_int const m,
                                                 rocblas_int const n,
                                                 rocblas_float_complex const alpha,
                                                 rocblas_float_complex const beta,

                                                 rocblas_float_complex* A,
                                                 rocblas_stride const shiftA,
                                                 rocblas_int const lda,
                                                 rocblas_stride const strideA,

                                                 rocblas_int const batch_count)
{
    return rocsolver::laset(handle, uplo, m, n, alpha, beta, A, shiftA, lda, strideA, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zlaset(rocblas_handle handle,
                                                 char const uplo,
                                                 rocblas_int const m,
                                                 rocblas_int const n,
                                                 rocblas_double_complex const alpha,
                                                 rocblas_double_complex const beta,

                                                 rocblas_double_complex* A,
                                                 rocblas_stride const shiftA,
                                                 rocblas_int const lda,
                                                 rocblas_stride const strideA,

                                                 rocblas_int const batch_count)
{
    return rocsolver::laset(handle, uplo, m, n, alpha, beta, A, shiftA, lda, strideA, batch_count);
}

} // end extern "C"
