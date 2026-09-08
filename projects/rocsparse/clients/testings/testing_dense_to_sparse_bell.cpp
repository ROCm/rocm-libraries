/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

template <typename I, typename T>
void testing_dense_to_sparse_bell_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle              handle       = local_handle;
    I                             m            = safe_size;
    I                             n            = safe_size;
    int64_t                       nnz          = safe_size;
    int64_t                       ld           = safe_size;
    void*                         dense_val    = (void*)0x4;
    void*                         bell_val     = (void*)0x4;
    void*                         bell_col_ind = (void*)0x4;
    rocsparse_index_base          base         = rocsparse_index_base_zero;
    rocsparse_order               order        = rocsparse_order_column;
    rocsparse_dense_to_sparse_alg alg          = rocsparse_dense_to_sparse_alg_default;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Dense and sparse matrix structures
    rocsparse_local_dnmat local_mat_A(m, n, ld, dense_val, ttype, order);
    rocsparse_local_spmat local_mat_B(m, n, nnz, bell_col_ind, bell_val, itype, base, ttype);

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

template <typename I, typename T>
void testing_dense_to_sparse_bell(const Arguments& arg)
{
    I                             m         = arg.M;
    I                             n         = arg.N;
    I                             block_dim = arg.block_dim;
    int64_t                       ld        = arg.denseld;
    rocsparse_index_base          base      = arg.baseA;
    rocsparse_dense_to_sparse_alg alg       = arg.dense_to_sparse_alg;
    rocsparse_order               order     = arg.order;

    I ell_block_size = block_dim;

    I mb = (m + block_dim - 1) / block_dim;
    I nb = (n + block_dim - 1) / block_dim;

    I mn = (order == rocsparse_order_column) ? m : n;
    I nm = (order == rocsparse_order_column) ? n : m;

    // Index and data type
    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_datatype  ttype = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Allocate memory.
    host_vector<T>   h_dense_val(ld * nm);
    device_vector<T> d_dense_val(ld * nm);

    rocsparse_seedrand();

    // Random initialization of the matrix.
    for(I j = 0; j < nm; ++j)
    {
        for(int64_t i = 0; i < ld; ++i)
        {
            h_dense_val[j * ld + i] = static_cast<T>(-2);
        }
    }

    // Randomly mark whole ELL blocks as zero so that the converted blocked ELL matrix
    // actually contains empty blocks. Roughly 40% of the blocks are kept non-zero.
    std::vector<char> block_nonzero(mb * nb);
    for(I bi = 0; bi < mb; ++bi)
    {
        for(I bj = 0; bj < nb; ++bj)
        {
            block_nonzero[bi * nb + bj] = (random_cached_generator_exact<int>(1, 10) <= 4) ? 1 : 0;
        }
    }

    for(I j = 0; j < nm; ++j)
    {
        for(I i = 0; i < mn; ++i)
        {
            const I row  = (order == rocsparse_order_column) ? i : j;
            const I col  = (order == rocsparse_order_column) ? j : i;
            const I brow = row / ell_block_size;
            const I bcol = col / ell_block_size;

            h_dense_val[j * ld + i] = block_nonzero[brow * nb + bcol]
                                          ? random_cached_generator<T>(0, 9)
                                          : static_cast<T>(0);
        }
    }

    // Transfer.
    CHECK_HIP_ERROR(
        hipMemcpy(d_dense_val, h_dense_val, sizeof(T) * ld * nm, hipMemcpyHostToDevice));

    rocsparse_local_dnmat mat_dense(m, n, ld, d_dense_val, ttype, order);

    rocsparse_local_spmat mat_sparse(mb * ell_block_size,
                                     nb * ell_block_size,
                                     rocsparse_direction_row,
                                     ell_block_size,
                                     0,
                                     nullptr,
                                     nullptr,
                                     itype,
                                     base,
                                     ttype);

    // Find size of required temporary buffer
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(
        rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, &buffer_size, nullptr));

    // Allocate temporary buffer on device
    device_vector<I> d_temp_buffer(buffer_size);

    // Perform analysis
    CHECK_ROCSPARSE_ERROR(
        rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, nullptr, d_temp_buffer));

    int64_t num_rows_tmp;
    int64_t num_cols_tmp;
    int64_t ell_block_dim_tmp;
    int64_t ell_cols_tmp;

    rocsparse_direction  ell_block_dir_tmp;
    rocsparse_indextype  idx_type_tmp;
    rocsparse_index_base idx_base_tmp;
    rocsparse_datatype   data_type_tmp;

    void* ell_col_ind_tmp;
    void* ell_val_tmp;

    CHECK_ROCSPARSE_ERROR(rocsparse_bell_get(mat_sparse,
                                             &num_rows_tmp,
                                             &num_cols_tmp,
                                             &ell_block_dir_tmp,
                                             &ell_block_dim_tmp,
                                             &ell_cols_tmp,
                                             &ell_col_ind_tmp,
                                             &ell_val_tmp,
                                             &idx_type_tmp,
                                             &idx_base_tmp,
                                             &data_type_tmp));

    // Allocate memory on device
    device_vector<I> d_bell_col_ind(mb * ell_cols_tmp / ell_block_dim_tmp);
    device_vector<T> d_bell_val(m * ell_cols_tmp);

    CHECK_ROCSPARSE_ERROR(rocsparse_bell_set_pointers(mat_sparse, d_bell_col_ind, d_bell_val));

    if(arg.unit_check)
    {
        // Complete conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(
            handle, mat_dense, mat_sparse, alg, &buffer_size, d_temp_buffer));

        host_vector<I> h_bell_col_ind_gpu(mb * ell_cols_tmp / ell_block_dim_tmp);
        host_vector<T> h_bell_val_gpu(m * ell_cols_tmp);

        CHECK_HIP_ERROR(hipMemcpy(h_bell_col_ind_gpu.data(),
                                  d_bell_col_ind,
                                  sizeof(I) * mb * ell_cols_tmp / ell_block_dim_tmp,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(h_bell_val_gpu.data(),
                                  d_bell_val,
                                  sizeof(T) * m * ell_cols_tmp,
                                  hipMemcpyDeviceToHost));

        host_vector<I> h_bell_col_ind_cpu(mb * ell_cols_tmp / ell_block_dim_tmp);
        host_vector<T> h_bell_val_cpu(m * ell_cols_tmp);

        I ell_cols_cpu = 0;
        host_dense_to_bell(m,
                           n,
                           base,
                           h_dense_val,
                           ld,
                           order,
                           ell_block_size,
                           ell_cols_cpu,
                           h_bell_val_cpu,
                           h_bell_col_ind_cpu);

        h_bell_col_ind_cpu.unit_check(h_bell_col_ind_gpu);
        h_bell_val_cpu.unit_check(h_bell_val_gpu);
    }

    if(arg.timing)
    {
        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_dense_to_sparse,
                                                                      handle,
                                                                      mat_dense,
                                                                      mat_sparse,
                                                                      alg,
                                                                      &buffer_size,
                                                                      d_temp_buffer);

        double gbyte_count = dense2bell_gbyte_count<T>(m, n, (I)ell_cols_tmp, block_dim);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::LD,
                            ld,
                            display_key_t::order,
                            order,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(ITYPE, TYPE)                                                           \
    template void testing_dense_to_sparse_bell_bad_arg<ITYPE, TYPE>(const Arguments& arg); \
    template void testing_dense_to_sparse_bell<ITYPE, TYPE>(const Arguments& arg)
INSTANTIATE(int32_t, _Float16);
INSTANTIATE(int32_t, rocsparse_bfloat16);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, _Float16);
INSTANTIATE(int64_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);

void testing_dense_to_sparse_bell_extra_687(const Arguments& arg)
{
    if(!arg.unit_check)
    {
        return;
    }

    using I = int32_t;
    using T = float;

    int             device;
    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDevice(&device));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, device));

    // One thread block converts one block-row, so a block size of one makes the number of
    // block-rows exceed the maximum grid dimension by exactly one block-row. The single
    // trailing block-row is only reached if the grid is clamped and the kernels stride.
    const int64_t ell_block_size = 1;
    const int64_t m              = static_cast<int64_t>(prop.maxGridSize[0]) + 1;
    const int64_t n              = 1;
    const int64_t ld             = m;

    const int64_t mb = (m - 1) / ell_block_size + 1;
    const int64_t nb = (n - 1) / ell_block_size + 1;

    const rocsparse_index_base          base  = rocsparse_index_base_one;
    const rocsparse_order               order = rocsparse_order_column;
    const rocsparse_dense_to_sparse_alg alg   = rocsparse_dense_to_sparse_alg_default;

    const rocsparse_indextype itype = get_indextype<I>();
    const rocsparse_datatype  ttype = get_datatype<T>();

    rocsparse_local_handle handle;

    // The dense matrix is zero everywhere but in its very last row, so that the converted
    // blocked ELL matrix has a single occupied slot and that slot sits in the trailing
    // block-row that an unclamped or non-striding grid does not reach.
    const T h_last_val = static_cast<T>(3);

    device_vector<T> d_dense_val(ld * n);
    CHECK_HIP_ERROR(hipMemset(d_dense_val, 0, sizeof(T) * ld * n));
    CHECK_HIP_ERROR(
        hipMemcpy(d_dense_val.data() + (m - 1), &h_last_val, sizeof(T), hipMemcpyHostToDevice));

    rocsparse_local_dnmat mat_dense(m, n, ld, d_dense_val, ttype, order);

    rocsparse_local_spmat mat_sparse(mb * ell_block_size,
                                     nb * ell_block_size,
                                     rocsparse_direction_row,
                                     ell_block_size,
                                     0,
                                     nullptr,
                                     nullptr,
                                     itype,
                                     base,
                                     ttype);

    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(
        rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, &buffer_size, nullptr));

    device_vector<int64_t> d_temp_buffer(buffer_size / sizeof(int64_t));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, nullptr, d_temp_buffer));

    int64_t              rows_tmp;
    int64_t              cols_tmp;
    int64_t              ell_block_dim_tmp;
    int64_t              ell_cols;
    rocsparse_direction  ell_block_dir_tmp;
    rocsparse_indextype  itype_tmp;
    rocsparse_index_base base_tmp;
    rocsparse_datatype   ttype_tmp;
    void*                ell_col_ind_tmp;
    void*                ell_val_tmp;

    CHECK_ROCSPARSE_ERROR(rocsparse_bell_get(mat_sparse,
                                             &rows_tmp,
                                             &cols_tmp,
                                             &ell_block_dir_tmp,
                                             &ell_block_dim_tmp,
                                             &ell_cols,
                                             &ell_col_ind_tmp,
                                             &ell_val_tmp,
                                             &itype_tmp,
                                             &base_tmp,
                                             &ttype_tmp));

    unit_check_scalar<int64_t>(ell_cols, ell_block_size);

    const int64_t ell_block_width = ell_cols / ell_block_size;

    device_vector<I> d_bell_col_ind(mb * ell_block_width);
    device_vector<T> d_bell_val(m * ell_cols);

    // Poison the column indices so that a block-row the fill kernel never visits cannot be
    // mistaken for either an occupied or a padded slot.
    CHECK_HIP_ERROR(hipMemset(d_bell_col_ind, 0xff, sizeof(I) * mb * ell_block_width));

    CHECK_ROCSPARSE_ERROR(rocsparse_bell_set_pointers(mat_sparse, d_bell_col_ind, d_bell_val));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, &buffer_size, d_temp_buffer));

    I h_col_ind[2];
    T h_val[2];

    // Leading block-row: empty, so its single slot is padded and its value is zero.
    CHECK_HIP_ERROR(
        hipMemcpy(&h_col_ind[0], d_bell_col_ind.data(), sizeof(I), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(&h_val[0], d_bell_val.data(), sizeof(T), hipMemcpyDeviceToHost));

    // Trailing block-row: holds the only non-zero of the matrix in block-column zero.
    CHECK_HIP_ERROR(hipMemcpy(&h_col_ind[1],
                              d_bell_col_ind.data() + (mb - 1) * ell_block_width,
                              sizeof(I),
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        &h_val[1], d_bell_val.data() + (m - 1) * ell_cols, sizeof(T), hipMemcpyDeviceToHost));

    unit_check_scalar<I>(h_col_ind[0], static_cast<I>(base) - 1);
    unit_check_scalar<T>(h_val[0], static_cast<T>(0));

    unit_check_scalar<I>(h_col_ind[1], static_cast<I>(base));
    unit_check_scalar<T>(h_val[1], h_last_val);
}

void testing_dense_to_sparse_bell_extra(const Arguments& arg)
{
    testing_dense_to_sparse_bell_extra_687(arg);
}
