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

static void testing_spattern_create_csr_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle    local_handle;
    rocsparse_handle          handle   = local_handle;
    rocsparse_spattern_descr* p_descr  = (rocsparse_spattern_descr*)0x4;
    int64_t                   rows     = 1;
    int64_t                   cols     = 1;
    int64_t                   nnz      = 1;
    rocsparse_idvec_descr     row_data = (rocsparse_idvec_descr)0x4;
    rocsparse_idvec_descr     col_data = (rocsparse_idvec_descr)0x4;
    rocsparse_error*          p_error  = nullptr;

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {7};
        select_bad_arg_analysis(rocsparse_spattern_create_csr,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                p_descr,
                                rows,
                                cols,
                                nnz,
                                row_data,
                                col_data,
                                p_error);
    }
}

static void testing_spattern_create_csc_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle    local_handle;
    rocsparse_handle          handle   = local_handle;
    rocsparse_spattern_descr* p_descr  = (rocsparse_spattern_descr*)0x4;
    int64_t                   rows     = 1;
    int64_t                   cols     = 1;
    int64_t                   nnz      = 1;
    rocsparse_idvec_descr     row_data = (rocsparse_idvec_descr)0x4;
    rocsparse_idvec_descr     col_data = (rocsparse_idvec_descr)0x4;
    rocsparse_error*          p_error  = nullptr;

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {7};
        select_bad_arg_analysis(rocsparse_spattern_create_csc,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                p_descr,
                                rows,
                                cols,
                                nnz,
                                row_data,
                                col_data,
                                p_error);
    }
}

static void testing_spattern_create_ell_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle    local_handle;
    rocsparse_handle          handle   = local_handle;
    rocsparse_spattern_descr* p_descr  = (rocsparse_spattern_descr*)0x4;
    int64_t                   rows     = 1;
    int64_t                   cols     = 1;
    int64_t                   width    = 1;
    rocsparse_idvec_descr     col_data = (rocsparse_idvec_descr)0x4;
    rocsparse_error*          p_error  = nullptr;

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {6};
        select_bad_arg_analysis(rocsparse_spattern_create_ell,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                p_descr,
                                rows,
                                cols,
                                width,
                                col_data,
                                p_error);
    }
}

static void testing_spattern_create_bell_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle    local_handle;
    rocsparse_handle          handle          = local_handle;
    rocsparse_spattern_descr* p_descr         = (rocsparse_spattern_descr*)0x4;
    int64_t                   rowsb           = 1;
    int64_t                   colsb           = 1;
    int64_t                   width           = 1;
    rocsparse_direction       block_direction = rocsparse_direction_row;
    int64_t                   block_dim       = 1;
    rocsparse_idvec_descr     col_data        = (rocsparse_idvec_descr)0x4;
    rocsparse_error*          p_error         = nullptr;

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {8};
        select_bad_arg_analysis(rocsparse_spattern_create_bell,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                p_descr,
                                rowsb,
                                colsb,
                                width,
                                block_direction,
                                block_dim,
                                col_data,
                                p_error);
    }
}

static void testing_spattern_create_sell_bad_arg(const Arguments& arg) {}

static void testing_spattern_create_bsr_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle    local_handle;
    rocsparse_handle          handle          = local_handle;
    rocsparse_spattern_descr* p_descr         = (rocsparse_spattern_descr*)0x4;
    int64_t                   rowsb           = 1;
    int64_t                   colsb           = 1;
    int64_t                   nnzb            = 1;
    rocsparse_direction       block_direction = rocsparse_direction_row;
    int64_t                   block_dim       = 1;
    rocsparse_idvec_descr     row_data        = (rocsparse_idvec_descr)0x4;
    rocsparse_idvec_descr     col_data        = (rocsparse_idvec_descr)0x4;
    rocsparse_error*          p_error         = nullptr;

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {9};
        select_bad_arg_analysis(rocsparse_spattern_create_bsr,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                p_descr,
                                rowsb,
                                colsb,
                                nnzb,
                                block_direction,
                                block_dim,
                                row_data,
                                col_data,
                                p_error);
    }
}

static void testing_spattern_create_coo_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle    local_handle;
    rocsparse_handle          handle   = local_handle;
    rocsparse_spattern_descr* p_descr  = (rocsparse_spattern_descr*)0x4;
    int64_t                   rows     = 1;
    int64_t                   cols     = 1;
    int64_t                   nnz      = 1;
    rocsparse_idvec_descr     row_data = (rocsparse_idvec_descr)0x4;
    rocsparse_idvec_descr     col_data = (rocsparse_idvec_descr)0x4;
    rocsparse_error*          p_error  = nullptr;

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {7};
        select_bad_arg_analysis(rocsparse_spattern_create_coo,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                p_descr,
                                rows,
                                cols,
                                nnz,
                                row_data,
                                col_data,
                                p_error);
    }
}

static void testing_spattern_create_coo_aos_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle    local_handle;
    rocsparse_handle          handle   = local_handle;
    rocsparse_spattern_descr* p_descr  = (rocsparse_spattern_descr*)0x4;
    int64_t                   rows     = 1;
    int64_t                   cols     = 1;
    int64_t                   nnz      = 1;
    rocsparse_idvec_descr     row_data = (rocsparse_idvec_descr)0x4;
    rocsparse_idvec_descr     col_data = (rocsparse_idvec_descr)0x4;
    rocsparse_error*          p_error  = nullptr;

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {7};
        select_bad_arg_analysis(rocsparse_spattern_create_coo_aos,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                p_descr,
                                rows,
                                cols,
                                nnz,
                                row_data,
                                col_data,
                                p_error);
    }
}

static void testing_spattern_get_data_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle   local_handle;
    rocsparse_handle         handle = local_handle;
    rocsparse_spattern_descr descr;
    rocsparse_error*         p_error = nullptr;

    const int64_t         rows               = 4;
    const int64_t         cols               = 4;
    int64_t               nnz                = 4;
    int32_t*              row_raw_data       = (int32_t*)0x4;
    int32_t*              col_raw_data       = (int32_t*)0x4;
    const int32_t*        const_row_raw_data = (const int32_t*)0x4;
    const int32_t*        const_col_raw_data = (const int32_t*)0x4;
    rocsparse_idvec_descr row_data;
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_create(handle,
                                                 &row_data,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rows + 1,
                                                 1,
                                                 const_row_raw_data,
                                                 row_raw_data,
                                                 p_error));

    rocsparse_idvec_descr col_data;
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_create(handle,
                                                 &col_data,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 cols + 1,
                                                 1,
                                                 const_col_raw_data,
                                                 col_raw_data,
                                                 p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_create_csr(
        handle, &descr, rows, cols, nnz, row_data, col_data, p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_destroy(handle, descr, p_error));
}

static void testing_spattern_set_data_bad_arg(const Arguments& arg) {}

static void testing_spattern_get_prop_bad_arg(const Arguments& arg) {}

static void testing_spattern_set_prop_bad_arg(const Arguments& arg) {}

template <typename T>
void testing_spattern_descr_bad_arg(const Arguments& arg)
{
    testing_spattern_create_csr_bad_arg(arg);
    testing_spattern_create_csc_bad_arg(arg);
    testing_spattern_create_ell_bad_arg(arg);
    testing_spattern_create_bell_bad_arg(arg);
    testing_spattern_create_sell_bad_arg(arg);
    testing_spattern_create_bsr_bad_arg(arg);
    testing_spattern_create_coo_bad_arg(arg);
    testing_spattern_create_coo_aos_bad_arg(arg);
    testing_spattern_get_data_bad_arg(arg);
    testing_spattern_set_data_bad_arg(arg);
    testing_spattern_get_prop_bad_arg(arg);
    testing_spattern_set_prop_bad_arg(arg);
}

#include "rocsparse_enum.hpp"

void testing_spattern_descr_extra(const Arguments& arg) {}

template <typename T>
void testing_spattern_descr(const Arguments& arg)
{
}

#define INSTANTIATE(TTYPE)                                                     \
    template void testing_spattern_descr_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_spattern_descr<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
