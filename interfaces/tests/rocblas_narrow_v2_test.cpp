// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <rocblas/rocblas.h>

#include <cstdlib>

int main() {
#if defined(_WIN32)
    _putenv_s("ROCM_INTERFACES_BLAS_V2_PROVIDER", BLAS_PROVIDER_PATH);
#else
    setenv("ROCM_INTERFACES_BLAS_V2_PROVIDER", BLAS_PROVIDER_PATH, 1);
#endif
    rocblas_handle h = nullptr;
    if (rocblas_create_handle(&h) != rocblas_status_success) return 1;
    float alpha = 1, beta = 0, x[4]{}, y[4]{}, a[16]{}, b[16]{}, c[16]{};
    // One public spelling for each execution callback (matmul_query is the
    // provider discovery half of matmul and has no direct classic spelling).
    if (rocblas_saxpy(h, 4, &alpha, x, 1, y, 1) != rocblas_status_success) return 2;
    if (rocblas_sdot(h, 4, x, 1, y, 1, x) != rocblas_status_success) return 3;
    if (rocblas_srot(h, 4, x, 1, y, 1, &alpha, &beta) != rocblas_status_success) return 4;
    if (rocblas_sgemv(h, rocblas_operation_none, 4, 4, &alpha, a, 4, x, 1, &beta, y, 1) !=
        rocblas_status_success)
        return 5;
    if (rocblas_sger(h, 4, 4, &alpha, x, 1, y, 1, a, 4) != rocblas_status_success) return 6;
    if (rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, 4, 4, 4, &alpha, a, 4, b,
                      4, &beta, c, 4) != rocblas_status_success)
        return 7;
    if (rocblas_ssymm(h, rocblas_side_left, rocblas_fill_upper, 4, 4, &alpha, a, 4, b, 4, &beta, c,
                      4) != rocblas_status_success)
        return 8;
    if (rocblas_strsm(h, rocblas_side_left, rocblas_fill_upper, rocblas_operation_none,
                      rocblas_diagonal_non_unit, 4, 4, &alpha, a, 4, b,
                      4) != rocblas_status_success)
        return 9;
    if (rocblas_sgeam(h, rocblas_operation_none, rocblas_operation_none, 4, 4, &alpha, a, 4, &beta,
                      b, 4, c, 4) != rocblas_status_success)
        return 10;
    // Grouped GEMM carries per-group shape and scalar arrays that the current
    // narrow request cannot represent. It remains an explicit bridge-only
    // spelling rather than being misreported as an ordinary pointer batch.
    const rocblas_operation operations[]{rocblas_operation_none};
    const rocblas_int dimensions[]{4};
    const rocblas_int group_sizes[]{1};
    const float alphas[]{1};
    const float betas[]{0};
    const float* a_groups[]{a};
    const float* b_groups[]{b};
    float* c_groups[]{c};
    if (rocblas_sgemm_grouped_batched(h, operations, operations, dimensions, dimensions, dimensions,
                                      alphas, a_groups, dimensions, b_groups, dimensions, betas,
                                      c_groups, dimensions, 1,
                                      group_sizes) != rocblas_status_not_implemented)
        return 11;
    return rocblas_destroy_handle(h) == rocblas_status_success ? 0 : 12;
}
