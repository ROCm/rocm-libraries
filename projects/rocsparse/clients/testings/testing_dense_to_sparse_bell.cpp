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

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <tuple>

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

        unit_check_scalar<int64_t>(ell_cols_tmp, ell_cols_cpu);
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

template <typename I, typename T>
void testing_dense_to_sparse_bell_extra_687(const Arguments& arg)
{
    if(!arg.unit_check)
    {
        return;
    }

    int             device;
    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDevice(&device));
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, device));

    // The dense2bell kernels launch one 256-thread block per block-row, and their grid is
    // clamped to min(maxGridSize[0], (2^32 - 1) / 256) blocks, i.e. 16,777,215 blocks. With a
    // block size of one, m exceeds that clamp by exactly one block-row. The single trailing
    // block-row is only reached if the grid is clamped and the kernels stride, and it is
    // block 0 that reaches it, right after block-row 0.
    const int64_t block_threads  = 256;
    const int64_t grid_x_max     = std::min(static_cast<int64_t>(prop.maxGridSize[0]),
                                        static_cast<int64_t>(UINT32_MAX) / block_threads);
    const int64_t ell_block_size = 1;
    const int64_t m              = grid_x_max + 1;
    const int64_t n              = 2;
    const int64_t ld             = m;

    const int64_t mb = (m - 1) / ell_block_size + 1;
    const int64_t nb = (n - 1) / ell_block_size + 1;

    const rocsparse_index_base          base  = rocsparse_index_base_one;
    const rocsparse_order               order = rocsparse_order_column;
    const rocsparse_dense_to_sparse_alg alg   = rocsparse_dense_to_sparse_alg_default;

    const rocsparse_indextype itype = get_indextype<I>();
    const rocsparse_datatype  ttype = get_datatype<T>();

    rocsparse_local_handle handle;

    // The dense matrix is zero except for A(0, 0) and the whole last row. Block-row 0 fills
    // one slot, so block 0 enters the trailing block-row with s_slot == 1 unless it resets
    // it. The trailing block-row is the only one with two occupied slots, so ell_cols is
    // only right if the nnz kernel reached it.
    const T h_first_val   = static_cast<T>(2);
    const T h_last_val[2] = {static_cast<T>(3), static_cast<T>(5)};

    device_vector<T> d_dense_val(ld * n);
    CHECK_HIP_ERROR(hipMemset(d_dense_val, 0, sizeof(T) * ld * n));
    CHECK_HIP_ERROR(hipMemcpy(d_dense_val.data(), &h_first_val, sizeof(T), hipMemcpyHostToDevice));
    for(int64_t j = 0; j < n; ++j)
    {
        CHECK_HIP_ERROR(hipMemcpy(d_dense_val.data() + ld * j + (m - 1),
                                  &h_last_val[j],
                                  sizeof(T),
                                  hipMemcpyHostToDevice));
    }

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

    // A block-row the nnz kernel never visits must count as empty, not as whatever the
    // allocation held.
    CHECK_HIP_ERROR(hipMemset(d_temp_buffer, 0, buffer_size));

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

    const int64_t ell_block_width = 2;

    unit_check_scalar<int64_t>(ell_cols, ell_block_width * ell_block_size);
    if(ell_cols != ell_block_width * ell_block_size)
    {
        return;
    }

    // One spare block-row absorbs the slot past the end that a block entering the trailing
    // block-row with a stale s_slot writes.
    device_vector<I> d_bell_col_ind((mb + 1) * ell_block_width);
    device_vector<T> d_bell_val((m + ell_block_size) * ell_cols);

    // Poison the column indices so that a slot the fill kernel never writes cannot be
    // mistaken for either an occupied or a padded slot.
    CHECK_HIP_ERROR(hipMemset(d_bell_col_ind, 0xff, sizeof(I) * (mb + 1) * ell_block_width));

    CHECK_ROCSPARSE_ERROR(rocsparse_bell_set_pointers(mat_sparse, d_bell_col_ind, d_bell_val));

    CHECK_ROCSPARSE_ERROR(
        rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, &buffer_size, d_temp_buffer));

    I h_first_col_ind[2];
    T h_first_val_out[2];
    I h_last_col_ind[2];
    T h_last_val_out[2];

    CHECK_HIP_ERROR(
        hipMemcpy(h_first_col_ind, d_bell_col_ind.data(), sizeof(I) * 2, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(h_first_val_out, d_bell_val.data(), sizeof(T) * 2, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(h_last_col_ind,
                              d_bell_col_ind.data() + (mb - 1) * ell_block_width,
                              sizeof(I) * 2,
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(h_last_val_out,
                              d_bell_val.data() + (m - 1) * ell_cols,
                              sizeof(T) * 2,
                              hipMemcpyDeviceToHost));

    // Leading block-row: block-column 0 in slot 0, slot 1 padded.
    unit_check_scalar<I>(h_first_col_ind[0], static_cast<I>(base));
    unit_check_scalar<I>(h_first_col_ind[1], static_cast<I>(base) - 1);
    unit_check_scalar<T>(h_first_val_out[0], h_first_val);
    unit_check_scalar<T>(h_first_val_out[1], static_cast<T>(0));

    // Trailing block-row: block-columns 0 and 1 in slots 0 and 1.
    unit_check_scalar<I>(h_last_col_ind[0], static_cast<I>(base));
    unit_check_scalar<I>(h_last_col_ind[1], static_cast<I>(base) + 1);
    unit_check_scalar<T>(h_last_val_out[0], h_last_val[0]);
    unit_check_scalar<T>(h_last_val_out[1], h_last_val[1]);
}

namespace
{
    // A dense m x n matrix given by its non-zero entries (row, column, value), converted with
    // block dimension bd.
    struct dense2bell_case
    {
        int64_t                                        m;
        int64_t                                        n;
        int64_t                                        bd;
        std::vector<std::tuple<int64_t, int64_t, int>> nonzeros;
        // Block width of the descriptor the conversion writes into, if it is not the one the
        // analysis returns (a hand-set ell_cols).
        int64_t ell_block_width = -1;
        // The output of the conversion would not fit in memory; only the analysis runs.
        bool analysis_only = false;
    };

    int64_t dense2bell_ceil_div(int64_t count, int64_t block_size)
    {
        return (count > 0) ? (count - 1) / block_size + 1 : 0;
    }

    // Block columns of each block row that hold a non-zero, in increasing order.
    std::vector<std::vector<int64_t>> dense2bell_nonzero_blocks(const dense2bell_case& c)
    {
        std::vector<std::set<int64_t>> blocks(dense2bell_ceil_div(c.m, c.bd));
        for(const auto& [row, col, value] : c.nonzeros)
        {
            if(value != 0)
            {
                blocks[row / c.bd].insert(col / c.bd);
            }
        }

        std::vector<std::vector<int64_t>> result;
        for(const auto& b : blocks)
        {
            result.emplace_back(b.begin(), b.end());
        }
        return result;
    }

    // Converts one case and checks it against the closed form of the blocked ELL format:
    //   ell_cols = bd * max_ib K_ib, K_ib the number of non-zero blocks of block row ib,
    //   bell_col_ind[ib * W + s] = J_ib[s] + base, padded with base - 1,
    //   bell_val[r * ell_cols + s * bd + c] = A(r, J_ib[s] * bd + c), zero elsewhere.
    // All of them are integers or copies of small integers, so they have to match exactly.
    template <typename I, typename T>
    void testing_dense_to_sparse_bell_case(rocsparse_handle       handle,
                                           const dense2bell_case& c,
                                           rocsparse_index_base   base,
                                           rocsparse_order        order)
    {
#ifdef GOOGLE_TEST
        SCOPED_TRACE("m " + std::to_string(c.m) + " n " + std::to_string(c.n) + " bd "
                     + std::to_string(c.bd) + " base " + std::to_string(base) + " order "
                     + std::to_string(order) + " index bytes " + std::to_string(sizeof(I)));
#endif

        const rocsparse_dense_to_sparse_alg alg   = rocsparse_dense_to_sparse_alg_default;
        const rocsparse_indextype           itype = get_indextype<I>();
        const rocsparse_datatype            ttype = get_datatype<T>();

        const int64_t m  = c.m;
        const int64_t n  = c.n;
        const int64_t bd = c.bd;
        const int64_t mb = dense2bell_ceil_div(m, bd);

        const std::vector<std::vector<int64_t>> blocks = dense2bell_nonzero_blocks(c);

        int64_t max_blocks = 0;
        for(const auto& b : blocks)
        {
            max_blocks = std::max<int64_t>(max_blocks, b.size());
        }
        const int64_t expected_ell_cols = bd * max_blocks;

        // The padding of each row or column holds a non-zero, so that reading it changes the
        // result.
        const int64_t  ld = ((order == rocsparse_order_row) ? n : m) + 3;
        const int64_t  nm = (order == rocsparse_order_row) ? m : n;
        host_vector<T> h_dense(std::max<int64_t>(ld * nm, 1), static_cast<T>(99));
        for(int64_t r = 0; r < m; ++r)
        {
            for(int64_t col = 0; col < n; ++col)
            {
                h_dense[(order == rocsparse_order_row) ? ld * r + col : ld * col + r]
                    = static_cast<T>(0);
            }
        }
        for(const auto& [row, col, value] : c.nonzeros)
        {
            h_dense[(order == rocsparse_order_row) ? ld * row + col : ld * col + row]
                = static_cast<T>(value);
        }

        device_vector<T> d_dense(h_dense.size());
        d_dense.transfer_from(h_dense);

        rocsparse_local_dnmat mat_dense(m, n, ld, d_dense, ttype, order);
        rocsparse_local_spmat mat_sparse(
            m, n, rocsparse_direction_row, bd, 0, nullptr, nullptr, itype, base, ttype);

        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, &buffer_size, nullptr));
        unit_check_scalar<size_t>(buffer_size, sizeof(int64_t) * mb);

        device_vector<int64_t> d_temp_buffer(std::max<int64_t>(mb, 1));
        CHECK_ROCSPARSE_ERROR(
            rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, nullptr, d_temp_buffer));

        int64_t              rows_tmp, cols_tmp, ell_block_dim_tmp, ell_cols;
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

        unit_check_scalar<int64_t>(ell_cols, expected_ell_cols);
        if(ell_cols != expected_ell_cols || c.analysis_only)
        {
            return;
        }

        const int64_t width    = (c.ell_block_width < 0) ? max_blocks : c.ell_block_width;
        const int64_t cols_out = width * bd;

        // Guard entries after both arrays catch a write past their ends.
        constexpr int64_t guard     = 64;
        const int64_t     ind_count = mb * width;
        const int64_t     val_count = m * cols_out;

        I ind_poison;
        std::memset(&ind_poison, 0x7f, sizeof(I));
        const T val_poison = static_cast<T>(77);

        host_vector<I> h_expected_ind(ind_count + guard, ind_poison);
        host_vector<T> h_expected_val(val_count + guard, val_poison);
        for(int64_t ib = 0; ib < mb; ++ib)
        {
            for(int64_t s = 0; s < width; ++s)
            {
                h_expected_ind[ib * width + s] = (s < int64_t(blocks[ib].size()))
                                                     ? static_cast<I>(blocks[ib][s] + base)
                                                     : static_cast<I>(base) - 1;
            }
        }
        for(int64_t i = 0; i < val_count; ++i)
        {
            h_expected_val[i] = static_cast<T>(0);
        }
        for(const auto& [row, col, value] : c.nonzeros)
        {
            const auto&   b    = blocks[row / bd];
            const int64_t slot = std::find(b.begin(), b.end(), col / bd) - b.begin();
            if(slot < width)
            {
                h_expected_val[row * cols_out + slot * bd + col % bd] = static_cast<T>(value);
            }
        }

        host_vector<I>   h_ind(ind_count + guard, ind_poison);
        host_vector<T>   h_val(val_count + guard, val_poison);
        device_vector<I> d_ind(h_ind.size());
        device_vector<T> d_val(h_val.size());
        d_ind.transfer_from(h_ind);
        d_val.transfer_from(h_val);

        if(c.ell_block_width < 0)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_bell_set_pointers(mat_sparse, d_ind, d_val));
            CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(
                handle, mat_dense, mat_sparse, alg, &buffer_size, d_temp_buffer));
        }
        else
        {
            rocsparse_local_spmat mat_hand(
                m, n, rocsparse_direction_row, bd, cols_out, d_ind, d_val, itype, base, ttype);
            CHECK_ROCSPARSE_ERROR(rocsparse_dense_to_sparse(
                handle, mat_dense, mat_hand, alg, &buffer_size, d_temp_buffer));
        }

        h_ind.transfer_from(d_ind);
        h_val.transfer_from(d_val);
        h_expected_ind.unit_check(h_ind);
        h_expected_val.unit_check(h_val);
    }

    // Runs every case for each index base and dense order.
    template <typename I, typename T>
    void testing_dense_to_sparse_bell_cases(const std::vector<dense2bell_case>& cases)
    {
        rocsparse_local_handle handle;
        for(const auto& c : cases)
        {
            for(auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
            {
                for(auto order : {rocsparse_order_row, rocsparse_order_column})
                {
                    testing_dense_to_sparse_bell_case<I, T>(handle, c, base, order);
                }
            }
        }
    }

    // A 1 x 2bd row whose entries are all non-zero: two full block columns.
    dense2bell_case dense2bell_full_row(int64_t bd)
    {
        dense2bell_case c{1, 2 * bd, bd, {}};
        for(int64_t j = 0; j < 2 * bd; ++j)
        {
            c.nonzeros.emplace_back(0, j, 1 + j % 7);
        }
        return c;
    }

    // Block dimensions up to the 256 threads of the dense-to-BELL analysis kernel. Each window
    // of the analysis holds one or more whole block columns.
    std::vector<dense2bell_case> dense2bell_cases_up_to_256()
    {
        std::vector<dense2bell_case> cases;
        for(int64_t bd : {1, 2, 3, 128, 255, 256})
        {
            cases.push_back({1, 1, bd, {{0, 0, 3}}});
        }
        cases.push_back(dense2bell_full_row(256));
        // Row and column offset 255 inside a block, partial last block row and block column,
        // and a padded slot.
        cases.push_back({300, 600, 256, {{255, 511, 7}, {0, 599, 3}, {299, 0, 5}}});
        cases.push_back({300, 600, 256, {}});
        cases.push_back({0, 5, 2, {}});
        cases.push_back({5, 0, 2, {}});
        cases.push_back({0, 5, 257, {}});
        cases.push_back({5, 0, 257, {}});
        // A descriptor with one slot per block row for a block row with two non-zero blocks:
        // the conversion keeps the first block and writes nothing past the arrays.
        dense2bell_case narrow{2, 4, 2, {{0, 0, 1}, {0, 3, 2}}};
        narrow.ell_block_width = 1;
        cases.push_back(narrow);
        return cases;
    }

    // Block dimensions above 256. The analysis kernel used to compute a column stride of
    // (256 / bd) * bd = 0 for them and never return (issue 15902).
    std::vector<dense2bell_case> dense2bell_cases_above_256()
    {
        std::vector<dense2bell_case> cases;
        for(int64_t bd : {257, 300, 512})
        {
            cases.push_back({1, 1, bd, {{0, 0, 3}}});
        }
        // The only non-zero lies past the first 256 columns of the block.
        for(int64_t bd : {257, 300})
        {
            cases.push_back({1, bd, bd, {{0, bd - 1, 5}}});
        }
        for(int64_t bd : {257, 300})
        {
            cases.push_back(dense2bell_full_row(bd));
        }
        // ell_cols = 514: block row 0 holds block columns 1 and 2, block row 1 block column 0.
        cases.push_back({300, 600, 257, {{256, 513, 7}, {0, 599, 3}, {299, 0, 5}}});
        cases.push_back({1, 1, 257, {}});
        cases.push_back({300, 600, 257, {}});
        // ell_cols = 2^32 is too large to convert, but the analysis must count it.
        dense2bell_case wide{1, 1, int64_t(1) << 32, {{0, 0, 3}}};
        wide.analysis_only = true;
        cases.push_back(wide);
        return cases;
    }

    // n + bd - 1 does not fit in int64_t: the analysis rejects the block dimension and leaves
    // ell_cols alone.
    template <typename I, typename T>
    void testing_dense_to_sparse_bell_block_dim_overflow()
    {
        rocsparse_local_handle handle;

        const int64_t m = 1, n = 2, ld = 2;
        const int64_t bd = std::numeric_limits<int64_t>::max();

        device_vector<T> d_dense(n);
        CHECK_HIP_ERROR(hipMemset(d_dense, 0, sizeof(T) * n));

        rocsparse_local_dnmat mat_dense(m, n, ld, d_dense, get_datatype<T>(), rocsparse_order_row);
        rocsparse_local_spmat mat_sparse(m,
                                         n,
                                         rocsparse_direction_row,
                                         bd,
                                         0,
                                         nullptr,
                                         nullptr,
                                         get_indextype<I>(),
                                         rocsparse_index_base_zero,
                                         get_datatype<T>());

        const rocsparse_dense_to_sparse_alg alg = rocsparse_dense_to_sparse_alg_default;

        size_t buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, &buffer_size, nullptr));
        unit_check_scalar<size_t>(buffer_size, sizeof(int64_t));

        device_vector<int64_t> d_temp_buffer(1);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_dense_to_sparse(handle, mat_dense, mat_sparse, alg, nullptr, d_temp_buffer),
            rocsparse_status_invalid_size);

        int64_t              rows_tmp, cols_tmp, ell_block_dim_tmp, ell_cols = -1;
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
        unit_check_scalar<int64_t>(ell_cols, 0);
    }

    template <typename I, typename T>
    void testing_dense_to_sparse_bell_extra_block_dim(bool above_256)
    {
        const std::vector<dense2bell_case> cases
            = above_256 ? dense2bell_cases_above_256() : dense2bell_cases_up_to_256();
        testing_dense_to_sparse_bell_cases<I, T>(cases);
        if(above_256)
        {
            testing_dense_to_sparse_bell_block_dim_overflow<I, T>();
        }
    }
}

void testing_dense_to_sparse_bell_extra(const Arguments& arg)
{
    const std::string name(arg.name);
    if(name == "dense_to_sparse_bell_extra_block_dim"
       || name == "dense_to_sparse_bell_extra_block_dim_above_256")
    {
        if(!arg.unit_check)
        {
            return;
        }
        const bool above_256 = (name == "dense_to_sparse_bell_extra_block_dim_above_256");
        testing_dense_to_sparse_bell_extra_block_dim<int32_t, float>(above_256);
        testing_dense_to_sparse_bell_extra_block_dim<int64_t, double>(above_256);
        testing_dense_to_sparse_bell_extra_block_dim<int64_t, rocsparse_float_complex>(above_256);
        return;
    }

    testing_dense_to_sparse_bell_extra_687<int32_t, float>(arg);
}
