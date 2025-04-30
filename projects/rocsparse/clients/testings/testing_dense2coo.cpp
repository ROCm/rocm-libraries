/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_dense2coo_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    rocsparse_handle          handle       = local_handle;
    rocsparse_int             m            = safe_size;
    rocsparse_int             n            = safe_size;
    const rocsparse_mat_descr descr        = local_descr;
    const T*                  A            = (const T*)0x4;
    rocsparse_int             ld           = safe_size;
    const rocsparse_int*      nnz_per_rows = (const rocsparse_int*)0x4;
    T*                        coo_val      = (T*)0x4;
    rocsparse_int*            coo_row_ind  = (rocsparse_int*)0x4;
    rocsparse_int*            coo_col_ind  = (rocsparse_int*)0x4;

#define PARAMS handle, m, n, descr, A, ld, nnz_per_rows, coo_val, coo_row_ind, coo_col_ind
    bad_arg_analysis(rocsparse_dense2coo<T>, PARAMS);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_dense2coo<T>(PARAMS),
                            rocsparse_status_requires_sorted_storage);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

    // Check ld < m
    ld = m / 2;
    EXPECT_ROCSPARSE_STATUS(rocsparse_dense2coo<T>(PARAMS), rocsparse_status_invalid_size);
#undef PARAMS
}

template <typename T>
void testing_dense2coo(const Arguments& arg)
{
    rocsparse_int        M     = arg.M;
    rocsparse_int        N     = arg.N;
    rocsparse_int        LD    = arg.denseld;
    rocsparse_index_base baseA = arg.baseA;

    rocsparse_local_handle handle;

    rocsparse_local_mat_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, baseA));

    if(LD < M)
    {
        return;
    }

    // Allocate memory.
    host_vector<T>   h_dense_val(LD * N);
    device_vector<T> d_dense_val(LD * N);

    host_vector<rocsparse_int>   h_nnz_per_row(M);
    device_vector<rocsparse_int> d_nnz_per_row(M);

    rocsparse_seedrand();

    // Initialize the entire allocated memory.
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < LD; ++i)
        {
            h_dense_val[j * LD + i] = -1;
        }
    }

    // Random initialization of the matrix.
    for(rocsparse_int j = 0; j < N; ++j)
    {
        for(rocsparse_int i = 0; i < M; ++i)
        {
            h_dense_val[j * LD + i] = random_cached_generator<T>(0, 4);
        }
    }

    // Transfer.
    CHECK_HIP_ERROR(hipMemcpy(d_dense_val, h_dense_val, sizeof(T) * LD * N, hipMemcpyHostToDevice));

    rocsparse_int nnz;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_nnz<T>(handle,
                                           rocsparse_direction_row,
                                           M,
                                           N,
                                           descr,
                                           (const T*)d_dense_val,
                                           LD,
                                           d_nnz_per_row,
                                           &nnz));

    // Transfer.
    CHECK_HIP_ERROR(
        hipMemcpy(h_nnz_per_row, d_nnz_per_row, sizeof(rocsparse_int) * M, hipMemcpyDeviceToHost));

    device_vector<rocsparse_int> d_coo_row_ind(nnz);
    device_vector<T>             d_coo_val(nnz);
    device_vector<rocsparse_int> d_coo_col_ind(nnz);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_dense2coo<T>(handle,
                                                     M,
                                                     N,
                                                     descr,
                                                     d_dense_val,
                                                     LD,
                                                     d_nnz_per_row,
                                                     (T*)d_coo_val,
                                                     (rocsparse_int*)d_coo_row_ind,
                                                     (rocsparse_int*)d_coo_col_ind));

        host_vector<rocsparse_int> gpu_coo_row_ind(nnz);
        host_vector<T>             gpu_coo_val(nnz);
        host_vector<rocsparse_int> gpu_coo_col_ind(nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            gpu_coo_row_ind, d_coo_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            gpu_coo_col_ind, d_coo_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(gpu_coo_val, d_coo_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        host_vector<rocsparse_int> cpu_coo_row_ind(nnz);
        host_vector<T>             cpu_coo_val(nnz);
        host_vector<rocsparse_int> cpu_coo_col_ind(nnz);

        // Compute the reference host first.
        host_dense_to_coo(M,
                          N,
                          rocsparse_get_mat_index_base(descr),
                          h_dense_val,
                          LD,
                          rocsparse_order_column,
                          h_nnz_per_row,
                          cpu_coo_val,
                          cpu_coo_row_ind,
                          cpu_coo_col_ind);

        cpu_coo_row_ind.unit_check(gpu_coo_row_ind);
        cpu_coo_col_ind.unit_check(gpu_coo_col_ind);
        cpu_coo_val.unit_check(gpu_coo_val);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_dense2coo<T>,
                                                                      handle,
                                                                      M,
                                                                      N,
                                                                      descr,
                                                                      d_dense_val,
                                                                      LD,
                                                                      d_nnz_per_row,
                                                                      (T*)d_coo_val,
                                                                      d_coo_row_ind,
                                                                      d_coo_col_ind);

        double gbyte_count = dense2coo_gbyte_count<T>(M, N, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::LD,
                            LD,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_dense2coo_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_dense2coo<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_dense2coo_extra(const Arguments& arg) {}
