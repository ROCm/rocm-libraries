#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "laset.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// Explicit instantiations.

template
void laset(rocblas_handle handle,
                  char const uplo_c,
                  rocblas_int const m,
                  rocblas_int const n,
                  float const alpha,
                  float const beta,

                  float* A_,
                  rocblas_stride const shiftA,
                  rocblas_int const lda,
                  rocblas_stride const strideA,

                  rocblas_int const batch_count);

template
void laset(rocblas_handle handle,
                  char const uplo_c,
                  rocblas_int const m,
                  rocblas_int const n,
                  double const alpha,
                  double const beta,

                  double* A_,
                  rocblas_stride const shiftA,
                  rocblas_int const lda,
                  rocblas_stride const strideA,

                  rocblas_int const batch_count);

template
void laset(rocblas_handle handle,
                  char const uplo_c,
                  rocblas_int const m,
                  rocblas_int const n,
                  rocblas_float_complex const alpha,
                  rocblas_float_complex const beta,

                  rocblas_float_complex* A_,
                  rocblas_stride const shiftA,
                  rocblas_int const lda,
                  rocblas_stride const strideA,

                  rocblas_int const batch_count);

template
void laset(rocblas_handle handle,
                  char const uplo_c,
                  rocblas_int const m,
                  rocblas_int const n,
                  rocblas_double_complex const alpha,
                  rocblas_double_complex const beta,

                  rocblas_double_complex* A_,
                  rocblas_stride const shiftA,
                  rocblas_int const lda,
                  rocblas_stride const strideA,

                  rocblas_int const batch_count);

ROCSOLVER_END_NAMESPACE
