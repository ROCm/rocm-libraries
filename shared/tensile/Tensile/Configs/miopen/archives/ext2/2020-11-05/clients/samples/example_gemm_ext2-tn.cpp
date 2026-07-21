/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas.h"
#include "utility.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>
#include <memory>

template <typename Ti, typename To, typename Tc>
void mat_mat_mult(rocblas_int    M,
                  rocblas_int    N,
                  rocblas_int    K,
                  Tc             alpha,
                  const Ti*      A,
                  rocblas_stride row_stride_a,
                  rocblas_stride col_stride_a,
                  const Ti*      B,
                  rocblas_stride row_stride_b,
                  rocblas_stride col_stride_b,
                  Tc             beta,
                  const To*      C,
                  rocblas_stride row_stride_c,
                  rocblas_stride col_stride_c,
                  To*            D,
                  rocblas_stride row_stride_d,
                  rocblas_stride col_stride_d)
{
    for(rocblas_int row = 0; row < M; row++)
        for(rocblas_int col = 0; col < N; col++)
        {
            Tc t{};
            if(alpha)
                for(rocblas_int k = 0; k < K; k++)
                    t += Tc{A[row * row_stride_a + k * col_stride_a]}
                         * Tc{B[k * row_stride_b + col * col_stride_b]};
            D[row * row_stride_d + col * col_stride_d] = static_cast<To>(
                beta ? beta * C[row * row_stride_c + col * col_stride_c] + alpha * t : alpha * t);
        }
}

// - { rocblas_function: "rocblas_gemm_ext2", atomics_mode: atomics_allowed, a_type: "f32_r", b_type: "f32_r", c_type: "f32_r", d_type: "f32_r", compute_type: "f32_r", M: 1024, N: 1024, K: 1024, alpha: 1.0, row_stride_a: 1024, col_stride_a: 1, row_stride_b: 1, col_stride_b: 1024, beta: 1.0, row_stride_c: 1, col_stride_c: 0, row_stride_d: 1, col_stride_d: 1024, algo: 0, solution_index: 0, flags: none, call_count: 600 }

int main()
{
    rocblas_datatype a_type       = rocblas_datatype_f32_r;
    rocblas_datatype b_type       = rocblas_datatype_f32_r;
    rocblas_datatype c_type       = rocblas_datatype_f32_r;
    rocblas_datatype d_type       = rocblas_datatype_f32_r;
    rocblas_datatype compute_type = rocblas_datatype_f32_r;

    using a_t = float;
    using b_t = float;
    using c_t = float;
    using d_t = float;

    rocblas_int m = 960;
    rocblas_int n = 1024;
    rocblas_int k = 1024;

    float alpha = 1.0;
    float beta  = 1.0;

    rocblas_stride row_stride_a = m, col_stride_a = 1;
    rocblas_stride row_stride_b = 1, col_stride_b = k;
    rocblas_stride row_stride_c = 1, col_stride_c = 0;
    rocblas_stride row_stride_d = 1, col_stride_d = m;

    std::cout << "gemm_ext2-tn example" << std::endl;

    size_t size_a = size_t(m - 1) * row_stride_a + size_t(k - 1) * col_stride_a + 1;
    size_t size_b = size_t(k - 1) * row_stride_b + size_t(n - 1) * col_stride_b + 1;
    size_t size_c = size_t(m - 1) * row_stride_c + size_t(n - 1) * col_stride_c + 1;
    size_t size_d = size_t(m - 1) * row_stride_d + size_t(n - 1) * col_stride_d + 1;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    auto ha      = std::make_unique<a_t[]>(size_a);
    auto hb      = std::make_unique<b_t[]>(size_b);
    auto hc      = std::make_unique<c_t[]>(size_c);
    auto hd      = std::make_unique<d_t[]>(size_d);
    auto hd_gold = std::make_unique<d_t[]>(size_d);

    // initial data on host
    srand(1);
    for(size_t i = 0; i < size_a; ++i)
        ha[i] = rand() % 17;
    for(size_t i = 0; i < size_b; ++i)
        hb[i] = rand() % 17;
    for(size_t i = 0; i < size_c; ++i)
        hc[i] = rand() % 17;
    for(size_t i = 0; i < size_d; ++i)
        hd[i] = std::numeric_limits<d_t>::signaling_NaN();

    // allocate memory on device
    a_t* da;
    b_t* db;
    c_t* dc;
    d_t* dd;

    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(a_t)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(b_t)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(c_t)));
    CHECK_HIP_ERROR(hipMalloc(&dd, size_d * sizeof(d_t)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, &ha[0], sizeof(a_t) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, &hb[0], sizeof(b_t) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, &hc[0], sizeof(c_t) * size_c, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dd, &hd[0], sizeof(d_t) * size_d, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    auto     algo           = rocblas_gemm_algo_standard;
    int32_t  solution_index = 0;
    uint32_t flags          = 0;

    CHECK_ROCBLAS_ERROR(rocblas_gemm_ext2(handle,
                                          m,
                                          n,
                                          k,
                                          &alpha,
                                          da,
                                          a_type,
                                          row_stride_a,
                                          col_stride_a,
                                          db,
                                          b_type,
                                          row_stride_b,
                                          col_stride_b,
                                          &beta,
                                          dc,
                                          c_type,
                                          row_stride_c,
                                          col_stride_c,
                                          dd,
                                          d_type,
                                          row_stride_d,
                                          col_stride_d,
                                          compute_type,
                                          algo,
                                          solution_index,
                                          flags));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(&hd[0], dd, sizeof(d_t) * size_d, hipMemcpyDeviceToHost));

    std::cout << "a_type, b_type, c_type, d_type, m, n, k, alpha, row_stride_a, col_stride_a, "
                 "row_stride_b, "
                 "col_stride_b, beta, row_stride_c, col_stride_c, row_stride_d, col_stride_d = "
              << rocblas_datatype2string(a_type) << ", " << rocblas_datatype2string(b_type) << ", "
              << rocblas_datatype2string(c_type) << ", " << rocblas_datatype2string(d_type) << ", "
              << m << ", " << n << ", " << k << ", " << alpha << ", " << row_stride_a << ", "
              << col_stride_a << ", " << row_stride_b << ", " << col_stride_b << ", " << beta
              << ", " << row_stride_c << ", " << col_stride_c << ", " << row_stride_d << ", "
              << col_stride_d << std::endl;

    // calculate golden or correct result
    mat_mat_mult(m,
                 n,
                 k,
                 alpha,
                 &ha[0],
                 row_stride_a,
                 col_stride_a,
                 &hb[0],
                 row_stride_b,
                 col_stride_b,
                 beta,
                 &hc[0],
                 row_stride_c,
                 col_stride_c,
                 &hd_gold[0],
                 row_stride_d,
                 col_stride_d);

    double max_relative_error = 0;
    for(size_t i = 0; i < size_d; i++)
    {
        auto relative_error = fabs(double(hd_gold[i] - hd[i]) / hd_gold[i]);
        if(relative_error > max_relative_error)
            max_relative_error = relative_error;
    }

    auto eps       = std::numeric_limits<d_t>::epsilon();
    auto tolerance = 10.0;
    if(max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
        std::cout << "FAIL: ";
    else
        std::cout << "PASS: ";
    std::cout << "max_relative_error = " << max_relative_error << std::endl;

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
