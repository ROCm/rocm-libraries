// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <rocblas/rocblas.h>

#include <cstdlib>
#include <iostream>

int main() {
#if defined(_WIN32)
#if defined(ROCBLAS_COMPLETE_BRIDGE_TEST)
    _putenv_s("ROCM_INTERFACES_ROCBLAS_BRIDGE_PROVIDER", BLAS_PROVIDER_PATH);
#else
    _putenv_s("ROCM_INTERFACES_BLAS_PROVIDER", BLAS_PROVIDER_PATH);
#endif
#else
#if defined(ROCBLAS_COMPLETE_BRIDGE_TEST)
    setenv("ROCM_INTERFACES_ROCBLAS_BRIDGE_PROVIDER", BLAS_PROVIDER_PATH, 1);
#else
    setenv("ROCM_INTERFACES_BLAS_PROVIDER", BLAS_PROVIDER_PATH, 1);
#endif
#endif
    rocblas_handle handle = nullptr;
    if (rocblas_create_handle(&handle) != rocblas_status_success || !handle) return 1;
    const rocblas_status expected_operation_status =
#if defined(ROCBLAS_COMPLETE_BRIDGE_TEST)
        rocblas_status_not_implemented;
#else
        rocblas_status_success;
#endif
    if (rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host) != expected_operation_status)
        return 2;
    float alpha = 1.0f;
    float beta = 0.0f;
    float x[4]{};
    float y[4]{};
    float c[16]{};
    if (rocblas_saxpy(handle, 4, &alpha, x, 1, y, 1) != expected_operation_status) return 3;
    if (rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, 4, 4, 4, &alpha, c, 4,
                      c, 4, &beta, c, 4) != expected_operation_status)
        return 4;
    if (rocblas_sgemm_64(handle, rocblas_operation_none, rocblas_operation_none, 4, 4, 4, &alpha, c,
                         4, c, 4, &beta, c, 4) != expected_operation_status)
        return 5;
#if defined(ROCBLAS_COMPLETE_BRIDGE_TEST)
    if (rocblas_set_optimal_device_memory_size_impl(handle, 2, size_t{16}, size_t{32}) !=
        expected_operation_status)
        return 7;
    struct rocblas_device_malloc_base* allocation = nullptr;
    if (rocblas_device_malloc_alloc(handle, &allocation, 2, size_t{16}, size_t{32}) !=
        expected_operation_status)
        return 8;
#endif
    if (rocblas_destroy_handle(handle) != rocblas_status_success) return 6;
    std::cout << "rocBLAS shadow loader test passed\n";
    return 0;
}
