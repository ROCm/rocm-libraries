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

#include "rocsparse_enum.hpp"
#include "testing.hpp"

static void testing_spattern_descr_create_csr_bad_arg(const Arguments& arg)
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
        select_bad_arg_analysis(rocsparse_spattern_descr_create_csr,
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

static void testing_spattern_descr_create_csc_bad_arg(const Arguments& arg)
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
        select_bad_arg_analysis(rocsparse_spattern_descr_create_csc,
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

static void testing_spattern_descr_create_ell_bad_arg(const Arguments& arg)
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
        select_bad_arg_analysis(rocsparse_spattern_descr_create_ell,
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

static void testing_spattern_descr_create_bell_bad_arg(const Arguments& arg)
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
        select_bad_arg_analysis(rocsparse_spattern_descr_create_bell,
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

static void testing_spattern_descr_create_sell_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle    local_handle;
    rocsparse_handle          handle           = local_handle;
    rocsparse_spattern_descr* p_descr          = (rocsparse_spattern_descr*)0x4;
    int64_t                   rows             = 1;
    int64_t                   cols             = 1;
    int64_t                   nnz              = 1;
    int64_t                   sell_slice_size  = 1;
    int64_t                   sell_colval_size = 1;
    rocsparse_idvec_descr     row_data         = (rocsparse_idvec_descr)0x4;
    rocsparse_idvec_descr     col_data         = (rocsparse_idvec_descr)0x4;
    rocsparse_error*          p_error          = nullptr;

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {9};
        select_bad_arg_analysis(rocsparse_spattern_descr_create_sell,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                p_descr,
                                rows,
                                cols,
                                nnz,
                                sell_slice_size,
                                sell_colval_size,
                                row_data,
                                col_data,
                                p_error);
    }
}

static void testing_spattern_descr_create_bsr_bad_arg(const Arguments& arg)
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
        select_bad_arg_analysis(rocsparse_spattern_descr_create_bsr,
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

static void testing_spattern_descr_create_coo_bad_arg(const Arguments& arg)
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
        select_bad_arg_analysis(rocsparse_spattern_descr_create_coo,
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

static void testing_spattern_descr_create_coo_aos_bad_arg(const Arguments& arg)
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
        select_bad_arg_analysis(rocsparse_spattern_descr_create_coo_aos,
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
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_descr_create(handle,
                                                       &row_data,
                                                       rocsparse_indextype_i32,
                                                       rocsparse_index_base_zero,
                                                       rows + 1,
                                                       1,
                                                       const_row_raw_data,
                                                       row_raw_data,
                                                       p_error));

    rocsparse_idvec_descr col_data;
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_descr_create(handle,
                                                       &col_data,
                                                       rocsparse_indextype_i32,
                                                       rocsparse_index_base_zero,
                                                       nnz,
                                                       1,
                                                       const_col_raw_data,
                                                       col_raw_data,
                                                       p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_csr(
        handle, &descr, rows, cols, nnz, row_data, col_data, p_error));

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {4};
        rocsparse_spattern_data  spattern_data                     = rocsparse_spattern_data_row;
        rocsparse_idvec_descr*   p_data                            = (rocsparse_idvec_descr*)0x4;
        select_bad_arg_analysis(rocsparse_spattern_get_data,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                descr,
                                spattern_data,
                                p_data,
                                p_error);
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_destroy(handle, descr, p_error));
}

static void testing_spattern_set_data_bad_arg(const Arguments& arg)
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
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_descr_create(handle,
                                                       &row_data,
                                                       rocsparse_indextype_i32,
                                                       rocsparse_index_base_zero,
                                                       rows + 1,
                                                       1,
                                                       const_row_raw_data,
                                                       row_raw_data,
                                                       p_error));

    rocsparse_idvec_descr col_data;
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_descr_create(handle,
                                                       &col_data,
                                                       rocsparse_indextype_i32,
                                                       rocsparse_index_base_zero,
                                                       nnz,
                                                       1,
                                                       const_col_raw_data,
                                                       col_raw_data,
                                                       p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_csr(
        handle, &descr, rows, cols, nnz, row_data, col_data, p_error));

    {
        static constexpr int32_t nargs_to_exclude                  = 1;
        const int32_t            args_to_exclude[nargs_to_exclude] = {4};
        rocsparse_spattern_data  spattern_data                     = rocsparse_spattern_data_row;
        rocsparse_idvec_descr    data                              = (rocsparse_idvec_descr)0x4;
        select_bad_arg_analysis(rocsparse_spattern_set_data,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                descr,
                                spattern_data,
                                data,
                                p_error);
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_destroy(handle, descr, p_error));
}

static void testing_spattern_get_prop_bad_arg(const Arguments& arg)
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
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_descr_create(handle,
                                                       &row_data,
                                                       rocsparse_indextype_i32,
                                                       rocsparse_index_base_zero,
                                                       rows + 1,
                                                       1,
                                                       const_row_raw_data,
                                                       row_raw_data,
                                                       p_error));

    rocsparse_idvec_descr col_data;
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_descr_create(handle,
                                                       &col_data,
                                                       rocsparse_indextype_i32,
                                                       rocsparse_index_base_zero,
                                                       nnz,
                                                       1,
                                                       const_col_raw_data,
                                                       col_raw_data,
                                                       p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_csr(
        handle, &descr, rows, cols, nnz, row_data, col_data, p_error));

    {
        static constexpr int32_t nargs_to_exclude                  = 2;
        const int32_t            args_to_exclude[nargs_to_exclude] = {4, 5};
        rocsparse_spattern_prop  prop                              = rocsparse_spattern_prop_rows;
        void*                    p_value                           = (void*)0x4;
        size_t                   datasize                          = sizeof(int32_t);
        select_bad_arg_analysis(rocsparse_spattern_get_prop,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                descr,
                                prop,
                                p_value,
                                datasize,
                                p_error);
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_destroy(handle, descr, p_error));
}

static void testing_spattern_set_prop_bad_arg(const Arguments& arg)
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
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_descr_create(handle,
                                                       &row_data,
                                                       rocsparse_indextype_i32,
                                                       rocsparse_index_base_zero,
                                                       rows + 1,
                                                       1,
                                                       const_row_raw_data,
                                                       row_raw_data,
                                                       p_error));

    rocsparse_idvec_descr col_data;
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_descr_create(handle,
                                                       &col_data,
                                                       rocsparse_indextype_i32,
                                                       rocsparse_index_base_zero,
                                                       nnz,
                                                       1,
                                                       const_col_raw_data,
                                                       col_raw_data,
                                                       p_error));

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_csr(
        handle, &descr, rows, cols, nnz, row_data, col_data, p_error));

    {
        static constexpr int32_t nargs_to_exclude                  = 2;
        const int32_t            args_to_exclude[nargs_to_exclude] = {4, 5};
        rocsparse_spattern_prop  prop                              = rocsparse_spattern_prop_rows;
        void*                    value                             = (void*)0x4;
        size_t                   datasize                          = sizeof(int32_t);
        select_bad_arg_analysis(rocsparse_spattern_set_prop,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                descr,
                                prop,
                                value,
                                datasize,
                                p_error);
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_destroy(handle, descr, p_error));
}

template <typename T>
void testing_spattern_descr_bad_arg(const Arguments& arg)
{
    testing_spattern_descr_create_csr_bad_arg(arg);
    testing_spattern_descr_create_csc_bad_arg(arg);
    testing_spattern_descr_create_ell_bad_arg(arg);
    testing_spattern_descr_create_bell_bad_arg(arg);
    testing_spattern_descr_create_sell_bad_arg(arg);
    testing_spattern_descr_create_bsr_bad_arg(arg);
    testing_spattern_descr_create_coo_bad_arg(arg);
    testing_spattern_descr_create_coo_aos_bad_arg(arg);
    testing_spattern_get_data_bad_arg(arg);
    testing_spattern_set_data_bad_arg(arg);
    testing_spattern_get_prop_bad_arg(arg);
    testing_spattern_set_prop_bad_arg(arg);
}

void testing_spattern_descr_extra(const Arguments& arg) {}

template <typename T>
void testing_spattern_descr(const Arguments& arg)
{

    rocsparse_local_handle local_handle;
    rocsparse_handle       handle  = local_handle;
    static constexpr bool  verbose = true;

    //
    // Create coo pattern.
    //
    for(auto format : rocsparse_format_t::values)
    {
        if(verbose)
            std::cout << "format :" << rocsparse_format2string(format) << std::endl;
        rocsparse_spattern_descr   descr            = nullptr;
        int64_t                    width            = 2;
        int64_t                    bell_width       = 2;
        int64_t                    batch_count      = 1;
        int64_t                    block_dim        = 2;
        int64_t                    rows             = 4;
        int64_t                    cols             = 4;
        int64_t                    nnz              = 16;
        int64_t                    rowsb            = rows / block_dim;
        int64_t                    colsb            = cols / block_dim;
        int64_t                    nnzb             = nnz / block_dim;
        rocsparse_direction        block_dir        = rocsparse_direction_row;
        const rocsparse_index_base base             = arg.baseA;
        int64_t                    sell_slice_size  = 4;
        int64_t                    sell_colval_size = 4;
        int64_t                    row_size         = 0;
        int64_t                    col_size         = 0;
        switch(format)
        {

        case rocsparse_format_coo:
        {
            row_size = nnz;
            col_size = nnz;
            break;
        }

        case rocsparse_format_coo_aos:
        {
            row_size = nnz;
            col_size = nnz;
            break;
        }

        case rocsparse_format_csr:
        {
            row_size = rows + 1;
            col_size = nnz;
            break;
        }

        case rocsparse_format_bsr:
        {
            row_size = rowsb + 1;
            col_size = nnzb;
            break;
        }

        case rocsparse_format_bell:
        {
            row_size = 0;
            col_size = rowsb * bell_width;
            break;
        }

        case rocsparse_format_sell:
        {
            row_size = (rows - 1) / sell_slice_size + 1;
            col_size = sell_colval_size;
            break;
        }

        case rocsparse_format_csc:
        {
            row_size = nnz;
            col_size = cols + 1;
            break;
        }

        case rocsparse_format_ell:
        {
            row_size = 0;
            col_size = rows * width;
            break;
        }
        }

        device_dense_vector<int32_t> drow_indices(row_size);
        device_dense_vector<int32_t> dcol_indices(col_size);
        rocsparse_local_idvec        row_data(handle, drow_indices, base);
        rocsparse_local_idvec        col_data(handle, dcol_indices, base);
        rocsparse_error*             p_error = nullptr;
        switch(format)
        {
        case rocsparse_format_coo:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_coo(
                handle, &descr, rows, cols, nnz, row_data, col_data, p_error));
            block_dir        = rocsparse_direction_column;
            block_dim        = 1;
            width            = 0;
            bell_width       = 0;
            sell_slice_size  = 0;
            sell_colval_size = 0;
            break;
        }

        case rocsparse_format_coo_aos:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_coo_aos(
                handle, &descr, rows, cols, nnz, row_data, col_data, p_error));
            block_dir        = rocsparse_direction_column;
            block_dim        = 1;
            width            = 0;
            bell_width       = 0;
            sell_slice_size  = 0;
            sell_colval_size = 0;
            break;
        }

        case rocsparse_format_csr:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_csr(
                handle, &descr, rows, cols, nnz, row_data, col_data, p_error));
            block_dir        = rocsparse_direction_column;
            block_dim        = 1;
            bell_width       = 0;
            width            = 0;
            sell_slice_size  = 0;
            sell_colval_size = 0;
            break;
        }

        case rocsparse_format_bsr:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_bsr(handle,
                                                                      &descr,
                                                                      rowsb,
                                                                      colsb,
                                                                      nnzb,
                                                                      block_dir,
                                                                      block_dim,
                                                                      row_data,
                                                                      col_data,
                                                                      p_error));

            rows             = rowsb;
            cols             = colsb;
            nnz              = nnzb;
            width            = 0;
            bell_width       = 0;
            sell_slice_size  = 0;
            sell_colval_size = 0;
            break;
        }

        case rocsparse_format_bell:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_bell(
                handle, &descr, rowsb, colsb, bell_width, block_dir, block_dim, col_data, p_error));

            rows             = rowsb;
            cols             = colsb;
            nnz              = nnzb;
            width            = 0;
            nnz              = rows * bell_width;
            sell_slice_size  = 0;
            sell_colval_size = 0;
            break;
        }

        case rocsparse_format_sell:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_sell(handle,
                                                                       &descr,
                                                                       rows,
                                                                       cols,
                                                                       nnz,
                                                                       sell_slice_size,
                                                                       sell_colval_size,
                                                                       row_data,
                                                                       col_data,
                                                                       p_error));
            block_dir  = rocsparse_direction_column;
            block_dim  = 1;
            width      = 0;
            bell_width = 0;

            break;
        }

        case rocsparse_format_csc:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_csc(
                handle, &descr, rows, cols, nnz, row_data, col_data, p_error));
            block_dir = rocsparse_direction_column;
            block_dim = 1;
            width     = 0;

            bell_width       = 0;
            sell_slice_size  = 0;
            sell_colval_size = 0;
            break;
        }

        case rocsparse_format_ell:
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_create_ell(
                handle, &descr, rows, cols, width, col_data, p_error));
            block_dir        = rocsparse_direction_column;
            block_dim        = 1;
            bell_width       = 0;
            nnz              = rows * width;
            sell_slice_size  = 0;
            sell_colval_size = 0;
            break;
        }
        }

        rocsparse_idvec_descr data;
        if(row_size > 0)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_data(
                handle, descr, rocsparse_spattern_data_row, &data, p_error));

            if(data != row_data)
            {
                unit_check_scalar(0, 1);
            }

            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_data(
                handle, descr, rocsparse_spattern_data_row, ((rocsparse_idvec_descr)0x4), p_error));

            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_data(
                handle, descr, rocsparse_spattern_data_row, &data, p_error));

            if(data != ((rocsparse_idvec_descr)0x4))
            {
                unit_check_scalar(0, 1);
            }

            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_data(
                handle, descr, rocsparse_spattern_data_row, col_data, p_error));
        }

        if(col_size > 0)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_data(
                handle, descr, rocsparse_spattern_data_column, &data, p_error));
            if(data != col_data)
            {
                unit_check_scalar(0, 1);
            }

            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_data(handle,
                                                              descr,
                                                              rocsparse_spattern_data_column,
                                                              ((rocsparse_idvec_descr)0x4),
                                                              p_error));

            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_data(
                handle, descr, rocsparse_spattern_data_column, &data, p_error));

            if(data != ((rocsparse_idvec_descr)0x4))
            {
                unit_check_scalar(0, 1);
            }

            CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_data(
                handle, descr, rocsparse_spattern_data_column, col_data, p_error));
        }
        //
        // Test get_prop
        //
        for(auto spattern_prop : rocsparse_spattern_prop_t::values)
        {
            if(verbose)
                std::cout << "prop :" << rocsparse_spattern_prop_t::to_string(spattern_prop)
                          << std::endl;
            switch(spattern_prop)
            {
            case rocsparse_spattern_prop_format:
            {
                rocsparse_format value;

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_enum(format, value);

                auto backup_value = value;
                value             = rocsparse_format_coo;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = rocsparse_format_csr;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_enum(value, rocsparse_format_coo);

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));

                break;
            }

            case rocsparse_spattern_prop_rows:
            {
                int64_t value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, rows);

                value = 77;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = 0;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, int64_t(77));

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &rows, sizeof(rows), p_error));
                break;
            }

            case rocsparse_spattern_prop_cols:
            {
                int64_t value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, cols);
                int64_t backup_value = value;
                value                = 77;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = 57;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, int64_t(77));

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));
                break;
            }

            case rocsparse_spattern_prop_nnz:
            {
                int64_t value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, nnz);

                int64_t backup_value = value;
                value                = 77;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = 0;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, int64_t(77));

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));

                break;
            }

            case rocsparse_spattern_prop_batch_count:
            {
                int64_t value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, batch_count);

                int64_t backup_value = value;
                value                = 77;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = 0;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, int64_t(77));

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));

                break;
            }

            case rocsparse_spattern_prop_block_dim:
            {
                int64_t value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                unit_check_scalar(value, block_dim);

                int64_t backup_value = value;
                value                = 77;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = 0;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, int64_t(77));

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));
                break;
            }

            case rocsparse_spattern_prop_block_dir:
            {
                rocsparse_direction value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_enum(value, block_dir);

                auto backup_value = value;
                value             = rocsparse_direction_column;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = rocsparse_direction_row;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_enum(value, rocsparse_direction_column);

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));

                break;
            }

            case rocsparse_spattern_prop_ell_width:
            {
                int64_t value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, width);

                auto backup_value = value;
                value             = 77;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = 0;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, int64_t(77));

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));

                break;
            }
            case rocsparse_spattern_prop_bell_width:
            {
                int64_t value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, bell_width);

                auto backup_value = value;
                value             = 77;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = 0;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, int64_t(77));

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));

                break;
            }
            case rocsparse_spattern_prop_sell_slice_size:
            {
                int64_t value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, sell_slice_size);

                auto backup_value = value;
                value             = 77;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = 0;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, int64_t(77));

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));

                break;
            }
            case rocsparse_spattern_prop_sell_colval_size:
            {
                int64_t value;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, sell_colval_size);

                auto backup_value = value;
                value             = 77;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));

                value = 0;
                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_get_prop(
                    handle, descr, spattern_prop, &value, sizeof(value), p_error));
                unit_check_scalar(value, int64_t(77));

                CHECK_ROCSPARSE_ERROR(rocsparse_spattern_set_prop(
                    handle, descr, spattern_prop, &backup_value, sizeof(backup_value), p_error));

                break;
            }
            }
        } // end for : prop

        CHECK_ROCSPARSE_ERROR(rocsparse_spattern_descr_destroy(handle, descr, p_error));
    }
}

#define INSTANTIATE(TTYPE)                                                     \
    template void testing_spattern_descr_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_spattern_descr<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
