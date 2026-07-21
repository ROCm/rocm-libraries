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
void testing_csr2gebsr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 1;

    host_dense_vector<rocsparse_int> hptr(safe_size + 1);
    hptr[0] = 0;
    hptr[1] = 1;
    device_dense_vector<rocsparse_int> dcsr_row_ptr(hptr);
    device_dense_vector<rocsparse_int> dbsr_row_ptr(hptr);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create descriptors
    rocsparse_local_mat_descr local_csr_descr;
    rocsparse_local_mat_descr local_bsr_descr;

    rocsparse_handle          handle          = local_handle;
    rocsparse_direction       dir             = rocsparse_direction_row;
    rocsparse_int             m               = safe_size;
    rocsparse_int             n               = safe_size;
    const rocsparse_mat_descr csr_descr       = local_csr_descr;
    const T*                  csr_val         = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr     = (const rocsparse_int*)dcsr_row_ptr;
    const rocsparse_int*      csr_col_ind     = (const rocsparse_int*)0x4;
    const rocsparse_mat_descr bsr_descr       = local_bsr_descr;
    T*                        bsr_val         = (T*)0x4;
    rocsparse_int*            bsr_row_ptr     = (rocsparse_int*)dbsr_row_ptr;
    rocsparse_int*            bsr_col_ind     = (rocsparse_int*)0x4;
    rocsparse_int             row_block_dim   = safe_size;
    rocsparse_int             col_block_dim   = safe_size;
    rocsparse_int*            bsr_nnz_devhost = (rocsparse_int*)0x4;
    size_t*                   buffer_size     = (size_t*)0x4;
    void*                     temp_buffer     = (void*)0x4;

    static constexpr int nargs_to_exclude_nnz                      = 2;
    const int            args_to_exclude_nnz[nargs_to_exclude_nnz] = {6, 12};

    static constexpr int nargs_to_exclude_solve                        = 1;
    const int            args_to_exclude_solve[nargs_to_exclude_solve] = {14};

#define PARAMS_BUFFER_SIZE                                                                         \
    handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, csr_col_ind, row_block_dim, col_block_dim, \
        buffer_size
#define PARAMS_NNZ                                                                                 \
    handle, dir, m, n, csr_descr, csr_row_ptr, csr_col_ind, bsr_descr, bsr_row_ptr, row_block_dim, \
        col_block_dim, bsr_nnz_devhost, temp_buffer
#define PARAMS                                                                           \
    handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, csr_col_ind, bsr_descr, bsr_val, \
        bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, temp_buffer
    bad_arg_analysis(rocsparse_csr2gebsr_buffer_size<T>, PARAMS_BUFFER_SIZE);
    select_bad_arg_analysis(
        rocsparse_csr2gebsr_nnz, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);
    select_bad_arg_analysis(
        rocsparse_csr2gebsr<T>, nargs_to_exclude_solve, args_to_exclude_solve, PARAMS);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr_nnz(PARAMS_NNZ),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr<T>(PARAMS),
                            rocsparse_status_requires_sorted_storage);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_sorted));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_sorted));

    // Check row_block_dim == 0
    row_block_dim = 0;
    col_block_dim = safe_size;
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr_nnz(PARAMS_NNZ), rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr<T>(PARAMS), rocsparse_status_invalid_size);

    // Check col_block_dim == 0
    row_block_dim = safe_size;
    col_block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr_nnz(PARAMS_NNZ), rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2gebsr<T>(PARAMS), rocsparse_status_invalid_size);
#undef PARAMS
#undef PARAMS_NNZ
#undef PARAMS_BUFFER_SIZE
}

template <typename T>
void testing_csr2gebsr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M             = arg.M;
    rocsparse_int               N             = arg.N;
    rocsparse_index_base        csr_base      = arg.baseA;
    rocsparse_index_base        bsr_base      = arg.baseB;
    rocsparse_direction         direction     = arg.direction;
    rocsparse_int               row_block_dim = arg.row_block_dimA;
    rocsparse_int               col_block_dim = arg.col_block_dimA;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_descr bsr_descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, bsr_base));

    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA, M, N);

    // M and N can be modified in init_csr
    rocsparse_int Mb = (M + row_block_dim - 1) / row_block_dim;
    rocsparse_int Nb = (N + col_block_dim - 1) / col_block_dim;

    device_csr_matrix<T>   dA(hA);
    device_gebsr_matrix<T> dC(direction, Mb, Nb, 0, row_block_dim, col_block_dim, bsr_base);

    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_buffer_size<T>(handle,
                                                             direction,
                                                             dA.m,
                                                             dA.n,
                                                             csr_descr,
                                                             dA.val,
                                                             dA.ptr,
                                                             dA.ind,
                                                             row_block_dim,
                                                             col_block_dim,
                                                             &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    host_scalar<rocsparse_int> hbsr_nnzb;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                  direction,
                                                  dA.m,
                                                  dA.n,
                                                  csr_descr,
                                                  dA.ptr,
                                                  dA.ind,
                                                  bsr_descr,
                                                  dC.ptr,
                                                  row_block_dim,
                                                  col_block_dim,
                                                  hbsr_nnzb,
                                                  dbuffer));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

    device_scalar<rocsparse_int> dbsr_nnzb;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                  direction,
                                                  dA.m,
                                                  dA.n,
                                                  csr_descr,
                                                  dA.ptr,
                                                  dA.ind,
                                                  bsr_descr,
                                                  dC.ptr,
                                                  row_block_dim,
                                                  col_block_dim,
                                                  dbsr_nnzb,
                                                  dbuffer));

    dC.define(direction, Mb, Nb, *hbsr_nnzb, row_block_dim, col_block_dim, bsr_base);

    if(arg.unit_check)
    {
        dbsr_nnzb.unit_check(hbsr_nnzb);

        // Finish conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2gebsr<T>(handle,
                                                     direction,
                                                     dA.m,
                                                     dA.n,
                                                     csr_descr,
                                                     dA.val,
                                                     dA.ptr,
                                                     dA.ind,
                                                     bsr_descr,
                                                     dC.val,
                                                     dC.ptr,
                                                     dC.ind,
                                                     row_block_dim,
                                                     col_block_dim,
                                                     dbuffer));

        // Host solution
        host_gebsr_matrix<T> hC_gold(
            direction, Mb, Nb, *hbsr_nnzb, row_block_dim, col_block_dim, bsr_base);
        host_csr_to_gebsr(direction,
                          hA.m,
                          hA.n,
                          hA.nnz,
                          hA.val,
                          hA.ptr,
                          hA.ind,
                          row_block_dim,
                          col_block_dim,
                          csr_base,
                          hC_gold.val,
                          hC_gold.ptr,
                          hC_gold.ind,
                          bsr_base);

        hC_gold.near_check(dC);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_csr2gebsr<T>,
                                                                      handle,
                                                                      direction,
                                                                      dA.m,
                                                                      dA.n,
                                                                      csr_descr,
                                                                      dA.val,
                                                                      dA.ptr,
                                                                      dA.ind,
                                                                      bsr_descr,
                                                                      dC.val,
                                                                      dC.ptr,
                                                                      dC.ind,
                                                                      row_block_dim,
                                                                      col_block_dim,
                                                                      dbuffer);

        double gbyte_count
            = csr2gebsr_gbyte_count<T>(M, Mb, hA.nnz, *hbsr_nnzb, row_block_dim, col_block_dim);
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::Mb,
                            Mb,
                            display_key_t::Nb,
                            Nb,
                            display_key_t::rbdim,
                            row_block_dim,
                            display_key_t::cbdim,
                            col_block_dim,
                            display_key_t::nnzb,
                            *hbsr_nnzb,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                                \
    template void testing_csr2gebsr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2gebsr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csr2gebsr_extra(const Arguments& arg) {}
