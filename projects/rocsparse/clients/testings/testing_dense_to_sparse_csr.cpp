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

#include "testing.hpp"

template <typename I, typename J, typename T>
void testing_dense_to_sparse_csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle              handle      = local_handle;
    J                             m           = safe_size;
    J                             n           = safe_size;
    I                             nnz         = safe_size;
    int64_t                       ld          = safe_size;
    void*                         dense_val   = (void*)0x4;
    void*                         csr_val     = (void*)0x4;
    void*                         csr_row_ptr = (void*)0x4;
    void*                         csr_col_ind = (void*)0x4;
    rocsparse_index_base          base        = rocsparse_index_base_zero;
    rocsparse_order               order       = rocsparse_order_column;
    rocsparse_dense_to_sparse_alg alg         = rocsparse_dense_to_sparse_alg_default;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Dense and sparse matrix structures
    rocsparse_local_dnmat local_mat_A(m, n, ld, dense_val, ttype, order);
    rocsparse_local_spmat local_mat_B(
        m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, itype, jtype, base, ttype);

    rocsparse_dnmat_descr mat_A = local_mat_A;
    rocsparse_spmat_descr mat_B = local_mat_B;

    int       nargs_to_exclude   = 2;
    const int args_to_exclude[2] = {4, 5};

#define PARAMS handle, mat_A, mat_B, alg, buffer_size, temp_buffer
    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(
            rocsparse_dense_to_sparse, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = nullptr;
        select_bad_arg_analysis(
            rocsparse_dense_to_sparse, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(
            rocsparse_dense_to_sparse, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = nullptr;
        select_bad_arg_analysis(
            rocsparse_dense_to_sparse, nargs_to_exclude, args_to_exclude, PARAMS);
    }
#undef PARAMS

    EXPECT_ROCSPARSE_STATUS(rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, nullptr, nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename I, typename J, typename T>
void testing_dense_to_sparse_csr(const Arguments& arg)
{
    J                             m     = arg.M;
    J                             n     = arg.N;
    int64_t                       ld    = arg.denseld;
    rocsparse_index_base          base  = arg.baseA;
    rocsparse_dense_to_sparse_alg alg   = arg.dense_to_sparse_alg;
    rocsparse_order               order = arg.order;

    J mn = (order == rocsparse_order_column) ? m : n;
    J nm = (order == rocsparse_order_column) ? n : m;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Argument sanity check before allocating invalid memory
    if(m <= 0 || n <= 0 || ld < mn)
    {
        // M == N == 0 means nnz can only be 0, too

        static const I safe_size = 100;

        // Allocate memory on device
        device_vector<I> d_csr_row_ptr(safe_size);
        device_vector<J> d_csr_col_ind(safe_size);
        device_vector<T> d_csr_val(safe_size);
        device_vector<T> d_dense_val(safe_size);

        if(!d_csr_row_ptr || !d_csr_col_ind || !d_csr_val || !d_dense_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        if(m == 0 && n == 0 && ld >= mn)
        {
            I nnz = 0;

            rocsparse_local_dnmat mat_A(m, n, ld, d_dense_val, ttype, order);
            rocsparse_local_spmat mat_B(
                m, n, nnz, d_csr_row_ptr, d_csr_col_ind, d_csr_val, itype, jtype, base, ttype);

            // Determine buffer size
            size_t buffer_size;
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, &buffer_size, nullptr),
                rocsparse_status_success);

            void* dbuffer;
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, safe_size));

            // Perform analysis
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, nullptr, dbuffer),
                rocsparse_status_success);

            // Complete conversion
            EXPECT_ROCSPARSE_STATUS(
                rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, &buffer_size, dbuffer),
                rocsparse_status_success);
            CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        }

        return;
    }

    // Allocate memory.
    host_vector<T>   h_dense_val(ld * nm);
    device_vector<T> d_dense_val(ld * nm);

    rocsparse_seedrand();

    // Random initialization of the matrix.
    for(J j = 0; j < nm; ++j)
    {
        for(int64_t i = 0; i < ld; ++i)
        {
            h_dense_val[j * ld + i] = static_cast<T>(-2);
        }
    }

    for(J j = 0; j < nm; ++j)
    {
        for(J i = 0; i < mn; ++i)
        {
            h_dense_val[j * ld + i] = random_cached_generator<T>(0, 9);
        }
    }

    // Transfer.
    CHECK_HIP_ERROR(
        hipMemcpy(d_dense_val, h_dense_val, sizeof(T) * ld * nm, hipMemcpyHostToDevice));

    // Allocate memory on device for row pointer
    device_vector<I> d_csr_row_ptr(m + 1);

    rocsparse_local_dnmat mat_dense(m, n, ld, d_dense_val, ttype, order);
    rocsparse_local_spmat mat_sparse(
        m, n, 0, d_csr_row_ptr, nullptr, nullptr, itype, jtype, base, ttype);

    // Find size of required temporary buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(handle,
                                                    mat_dense,
                                                    mat_sparse,
                                                    rocsparse_dense_to_sparse_alg_default,
                                                    &buffer_size,
                                                    nullptr));

    // Allocate temporary buffer on device
    device_vector<I> d_temp_buffer(buffer_size);

    if(!d_temp_buffer)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Perform analysis
    CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(handle,
                                                    mat_dense,
                                                    mat_sparse,
                                                    rocsparse_dense_to_sparse_alg_default,
                                                    nullptr,
                                                    d_temp_buffer));

    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_size(mat_sparse, &num_rows_tmp, &num_cols_tmp, &nnz));

    host_vector<I> nnz_per_row(m);
    CHECK_HIP_ERROR(
        hipMemcpy(nnz_per_row.data(), d_temp_buffer, sizeof(I) * m, hipMemcpyDeviceToHost));

    // Allocate memory on device
    device_vector<J> d_csr_col_ind(nnz);
    device_vector<T> d_csr_val(nnz);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_csr_set_pointers(mat_sparse, d_csr_row_ptr, d_csr_col_ind, d_csr_val));

    if(arg.unit_check)
    {
        // Complete conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(handle,
                                                        mat_dense,
                                                        mat_sparse,
                                                        rocsparse_dense_to_sparse_alg_default,
                                                        &buffer_size,
                                                        d_temp_buffer));

        host_vector<I> h_csr_row_ptr_gpu(m + 1);
        host_vector<J> h_csr_col_ind_gpu(nnz);
        host_vector<T> h_csr_val_gpu(nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            h_csr_row_ptr_gpu.data(), d_csr_row_ptr, sizeof(I) * (m + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            h_csr_col_ind_gpu.data(), d_csr_col_ind, sizeof(J) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(h_csr_val_gpu.data(), d_csr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        host_vector<I> h_csr_row_ptr_cpu(m + 1);
        host_vector<J> h_csr_col_ind_cpu(nnz);
        host_vector<T> h_csr_val_cpu(nnz);

        host_dense2csx<rocsparse_direction_row>(m,
                                                n,
                                                base,
                                                h_dense_val.data(),
                                                ld,
                                                order,
                                                nnz_per_row.data(),
                                                h_csr_val_cpu.data(),
                                                h_csr_row_ptr_cpu.data(),
                                                h_csr_col_ind_cpu.data());

        h_csr_row_ptr_cpu.unit_check(h_csr_row_ptr_gpu);
        h_csr_col_ind_cpu.unit_check(h_csr_col_ind_gpu);
        h_csr_val_cpu.unit_check(h_csr_val_gpu);
    }

    if(arg.timing)
    {

        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_dense_to_sparse,
                                               handle,
                                               mat_dense,
                                               mat_sparse,
                                               rocsparse_dense_to_sparse_alg_default,
                                               &buffer_size,
                                               d_temp_buffer);

        double gbyte_count = dense2csx_gbyte_count<rocsparse_direction_row, T>(m, n, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::order,
                            order,
                            display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::LD,
                            ld,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TYPE)                                                          \
    template void testing_dense_to_sparse_csr_bad_arg<ITYPE, JTYPE, TYPE>(const Arguments& arg); \
    template void testing_dense_to_sparse_csr<ITYPE, JTYPE, TYPE>(const Arguments& arg)
INSTANTIATE(int32_t, int32_t, _Float16);
INSTANTIATE(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, _Float16);
INSTANTIATE(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, _Float16);
INSTANTIATE(int64_t, int64_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
void testing_dense_to_sparse_csr_extra(const Arguments& arg) {}
