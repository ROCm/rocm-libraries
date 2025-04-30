/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_const_spmat_descr_bad_arg(const Arguments& arg)
{
    static const size_t         safe_size = 100;
    rocsparse_const_spmat_descr local_descr{};
    int64_t                     local_rows          = safe_size;
    int64_t                     local_cols          = safe_size;
    int64_t                     local_nnz           = safe_size;
    rocsparse_direction         local_ell_block_dir = rocsparse_direction_row;
    int64_t                     local_ell_block_dim = safe_size;
    int64_t                     local_ell_cols      = safe_size;
    rocsparse_index_base        local_base          = rocsparse_index_base_zero;
    rocsparse_format            local_format        = rocsparse_format_csr;
    rocsparse_indextype         local_itype         = get_indextype<I>();
    rocsparse_indextype         local_jtype         = get_indextype<J>();
    rocsparse_datatype          local_ttype         = get_datatype<T>();
    rocsparse_int               local_batch_count   = safe_size;

    {
        rocsparse_const_spmat_descr* descr         = &local_descr;
        int64_t                      rows          = local_rows;
        int64_t                      cols          = local_cols;
        int64_t                      nnz           = local_nnz;
        rocsparse_direction          ell_block_dir = local_ell_block_dir;
        int64_t                      ell_block_dim = local_ell_block_dim;
        int64_t                      ell_cols      = local_ell_cols;
        rocsparse_index_base         idx_base      = local_base;

        rocsparse_indextype idx_type     = local_itype;
        rocsparse_datatype  data_type    = local_ttype;
        rocsparse_indextype row_ptr_type = local_itype;
        rocsparse_indextype col_ind_type = local_jtype;
        rocsparse_indextype col_ptr_type = local_itype;
        rocsparse_indextype row_ind_type = local_jtype;

        void* coo_row_ind = (void*)0x4;
        void* coo_col_ind = (void*)0x4;
        void* coo_val     = (void*)0x4;

#define PARAMS_CREATE_COO \
    descr, rows, cols, nnz, coo_row_ind, coo_col_ind, coo_val, idx_type, idx_base, data_type
        bad_arg_analysis(rocsparse_create_const_coo_descr, PARAMS_CREATE_COO);
#undef PARAMS_CREATE_COO

        // nnz > rows * cols
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_coo_descr(descr,
                                                                 rows,
                                                                 cols,
                                                                 (rows * cols + 1),
                                                                 coo_row_ind,
                                                                 coo_col_ind,
                                                                 coo_val,
                                                                 idx_type,
                                                                 idx_base,
                                                                 data_type),
                                rocsparse_status_invalid_size);
        void* csr_row_ptr = (void*)0x4;
        void* csr_col_ind = (void*)0x4;
        void* csr_val     = (void*)0x4;

#define PARAMS_CREATE_CSR                                                                  \
    descr, rows, cols, nnz, csr_row_ptr, csr_col_ind, csr_val, row_ptr_type, col_ind_type, \
        idx_base, data_type
        bad_arg_analysis(rocsparse_create_const_csr_descr, PARAMS_CREATE_CSR);
#undef PARAMS_CREATE_CSR

        // nnz > rows * cols
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csr_descr(descr,
                                                                 rows,
                                                                 cols,
                                                                 (rows * cols + 1),
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
                                                                 csr_val,
                                                                 row_ptr_type,
                                                                 col_ind_type,
                                                                 idx_base,
                                                                 data_type),
                                rocsparse_status_invalid_size);

        void* csc_row_ind = (void*)0x4;
        void* csc_col_ptr = (void*)0x4;
        void* csc_val     = (void*)0x4;

#define PARAMS_CREATE_CSC                                                                  \
    descr, rows, cols, nnz, csc_col_ptr, csc_row_ind, csc_val, col_ptr_type, row_ind_type, \
        idx_base, data_type
        bad_arg_analysis(rocsparse_create_const_csc_descr, PARAMS_CREATE_CSC);
#undef PARAMS_CREATE_CSC

        // nnz > rows * cols
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csc_descr(descr,
                                                                 rows,
                                                                 cols,
                                                                 (rows * cols + 1),
                                                                 csc_col_ptr,
                                                                 csc_row_ind,
                                                                 csc_val,
                                                                 col_ptr_type,
                                                                 row_ind_type,
                                                                 idx_base,
                                                                 data_type),
                                rocsparse_status_invalid_size);

        void* ell_col_ind = (void*)0x4;
        void* ell_val     = (void*)0x4;

#define PARAMS_CREATE_ELL                                                                      \
    descr, rows, cols, ell_block_dir, ell_block_dim, ell_cols, ell_col_ind, ell_val, idx_type, \
        idx_base, data_type
        bad_arg_analysis(rocsparse_create_const_bell_descr, PARAMS_CREATE_ELL);
#undef PARAMS_CREATE_ELL

        // block_dim = 0
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(descr,
                                                                  rows,
                                                                  cols,
                                                                  ell_block_dir,
                                                                  0,
                                                                  ell_cols,
                                                                  ell_col_ind,
                                                                  ell_val,
                                                                  idx_type,
                                                                  idx_base,
                                                                  data_type),
                                rocsparse_status_invalid_size);

        // ell_cols > cols
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(descr,
                                                                  rows,
                                                                  cols,
                                                                  ell_block_dir,
                                                                  ell_block_dim,
                                                                  (cols + 1),
                                                                  ell_col_ind,
                                                                  ell_val,
                                                                  idx_type,
                                                                  idx_base,
                                                                  data_type),
                                rocsparse_status_invalid_size);

        // rocsparse_destroy_spmat_descr_ex
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(nullptr),
                                rocsparse_status_invalid_pointer);

        // Check valid descriptor creations
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_const_coo_descr(
                descr, 0, cols, 0, nullptr, nullptr, nullptr, idx_type, idx_base, data_type),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_const_coo_descr(
                descr, rows, 0, 0, nullptr, nullptr, nullptr, idx_type, idx_base, data_type),
            rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_const_coo_descr(
                descr, rows, cols, 0, nullptr, nullptr, nullptr, idx_type, idx_base, data_type),
            rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csr_descr(descr,
                                                                 0,
                                                                 cols,
                                                                 0,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 row_ptr_type,
                                                                 col_ind_type,
                                                                 idx_base,
                                                                 data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csr_descr(descr,
                                                                 rows,
                                                                 0,
                                                                 0,
                                                                 csr_row_ptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 row_ptr_type,
                                                                 col_ind_type,
                                                                 idx_base,
                                                                 data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csr_descr(descr,
                                                                 rows,
                                                                 cols,
                                                                 0,
                                                                 csr_row_ptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 row_ptr_type,
                                                                 col_ind_type,
                                                                 idx_base,
                                                                 data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
#if 0
    //
    // SWDEV-340500
    //
    EXPECT_ROCSPARSE_STATUS(
			    rocsparse_create_const_csr_descr(
							     descr, rows, cols, 0, nullptr, nullptr, nullptr, row_ptr_type, col_ind_type, idx_base, data_type),
			    rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr), rocsparse_status_success);
#endif

        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csc_descr(descr,
                                                                 rows,
                                                                 0,
                                                                 0,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 row_ptr_type,
                                                                 col_ind_type,
                                                                 idx_base,
                                                                 data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csc_descr(descr,
                                                                 0,
                                                                 cols,
                                                                 0,
                                                                 csc_col_ptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 row_ptr_type,
                                                                 col_ind_type,
                                                                 idx_base,
                                                                 data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csc_descr(descr,
                                                                 rows,
                                                                 cols,
                                                                 0,
                                                                 csc_col_ptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 row_ptr_type,
                                                                 col_ind_type,
                                                                 idx_base,
                                                                 data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);

#if 0
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(descr,
                                                              0,
                                                              cols,
                                                              ell_block_dir,
                                                              ell_block_dim,
                                                              ell_cols,
                                                              nullptr,
                                                              nullptr,
                                                              idx_type,
                                                              idx_base,
                                                              data_type),
                            rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr), rocsparse_status_success);
#endif
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(descr,
                                                                  rows,
                                                                  0,
                                                                  ell_block_dir,
                                                                  ell_block_dim,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  idx_type,
                                                                  idx_base,
                                                                  data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(descr,
                                                                  rows,
                                                                  cols,
                                                                  ell_block_dir,
                                                                  ell_block_dim,
                                                                  0,
                                                                  nullptr,
                                                                  nullptr,
                                                                  idx_type,
                                                                  idx_base,
                                                                  data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
    }

    {
        int64_t*              rows          = &local_rows;
        int64_t*              cols          = &local_cols;
        int64_t*              nnz           = &local_nnz;
        rocsparse_direction*  ell_block_dir = &local_ell_block_dir;
        int64_t*              ell_block_dim = &local_ell_block_dim;
        int64_t*              ell_cols      = &local_ell_cols;
        rocsparse_index_base* idx_base      = &local_base;
        rocsparse_format*     format        = &local_format;

        rocsparse_indextype* idx_type     = &local_itype;
        rocsparse_datatype*  data_type    = &local_ttype;
        rocsparse_indextype* row_ptr_type = &local_itype;
        rocsparse_indextype* col_ind_type = &local_jtype;
        rocsparse_indextype* col_ptr_type = &local_itype;
        rocsparse_indextype* row_ind_type = &local_jtype;

        const void** coo_row_ind = (const void**)0x4;
        const void** coo_col_ind = (const void**)0x4;
        const void** coo_val     = (const void**)0x4;
        const void** csr_row_ptr = (const void**)0x4;
        const void** csr_col_ind = (const void**)0x4;
        const void** csr_val     = (const void**)0x4;
        const void** csc_row_ind = (const void**)0x4;
        const void** csc_col_ptr = (const void**)0x4;
        const void** csc_val     = (const void**)0x4;
        const void** ell_col_ind = (const void**)0x4;
        const void** ell_val     = (const void**)0x4;

        rocsparse_int* batch_count = &local_batch_count;

        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_coo_descr(&local_descr,
                                                                     local_rows,
                                                                     local_cols,
                                                                     local_nnz,
                                                                     (const void*)0x4,
                                                                     (const void*)0x4,
                                                                     (const void*)0x4,
                                                                     local_itype,
                                                                     local_base,
                                                                     local_ttype),
                                    rocsparse_status_success);
            rocsparse_const_spmat_descr descr = local_descr;

#define PARAMS_GET_COO \
    descr, rows, cols, nnz, coo_row_ind, coo_col_ind, coo_val, idx_type, idx_base, data_type
            bad_arg_analysis(rocsparse_const_coo_get, PARAMS_GET_COO);
#undef PARAMS_GET_COO

#define PARAMS_GET_SIZE descr, rows, cols, nnz
            bad_arg_analysis(rocsparse_spmat_get_size, PARAMS_GET_SIZE);
#undef PARAMS_GET_SIZE

#define PARAMS_GET_FORMAT descr, format
            bad_arg_analysis(rocsparse_spmat_get_format, PARAMS_GET_FORMAT);
#undef PARAMS_GET_FORMAT

#define PARAMS_GET_INDEX_BASE descr, idx_base
            bad_arg_analysis(rocsparse_spmat_get_index_base, PARAMS_GET_INDEX_BASE);
#undef PARAMS_GET_INDEX_BASE

            const void** values = (const void**)0x4;
#define PARAMS_GET_VALUES descr, values
            bad_arg_analysis(rocsparse_const_spmat_get_values, PARAMS_GET_VALUES);
#undef PARAMS_GET_VALUES

#define PARAMS_GET_STRIDED_BATCH descr, batch_count
            bad_arg_analysis(rocsparse_spmat_get_strided_batch, PARAMS_GET_STRIDED_BATCH);
#undef PARAMS_GET_STRIDED_BATCH

            // Destroy valid descriptors
            EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
        }

        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csr_descr(&local_descr,
                                                                     local_rows,
                                                                     local_cols,
                                                                     local_nnz,
                                                                     (const void*)0x4,
                                                                     (const void*)0x4,
                                                                     (const void*)0x4,
                                                                     local_itype,
                                                                     local_jtype,
                                                                     local_base,
                                                                     local_ttype),
                                    rocsparse_status_success);
            rocsparse_const_spmat_descr descr = local_descr;

#define PARAMS_GET_CSR                                                                     \
    descr, rows, cols, nnz, csr_row_ptr, csr_col_ind, csr_val, row_ptr_type, col_ind_type, \
        idx_base, data_type
            bad_arg_analysis(rocsparse_const_csr_get, PARAMS_GET_CSR);
#undef PARAMS_GET_CSR

#define PARAMS_GET_SIZE descr, rows, cols, nnz
            bad_arg_analysis(rocsparse_spmat_get_size, PARAMS_GET_SIZE);
#undef PARAMS_GET_SIZE

#define PARAMS_GET_FORMAT descr, format
            bad_arg_analysis(rocsparse_spmat_get_format, PARAMS_GET_FORMAT);
#undef PARAMS_GET_FORMAT

#define PARAMS_GET_INDEX_BASE descr, idx_base
            bad_arg_analysis(rocsparse_spmat_get_index_base, PARAMS_GET_INDEX_BASE);
#undef PARAMS_GET_INDEX_BASE

            const void** values = (const void**)0x4;
#define PARAMS_GET_VALUES descr, values
            bad_arg_analysis(rocsparse_const_spmat_get_values, PARAMS_GET_VALUES);
#undef PARAMS_GET_VALUES

#define PARAMS_GET_STRIDED_BATCH descr, batch_count
            bad_arg_analysis(rocsparse_spmat_get_strided_batch, PARAMS_GET_STRIDED_BATCH);
#undef PARAMS_GET_STRIDED_BATCH

            // Destroy valid descriptors
            EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
        }

        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_csc_descr(&local_descr,
                                                                     local_rows,
                                                                     local_cols,
                                                                     local_nnz,
                                                                     (const void*)0x4,
                                                                     (const void*)0x4,
                                                                     (const void*)0x4,
                                                                     local_itype,
                                                                     local_jtype,
                                                                     local_base,
                                                                     local_ttype),
                                    rocsparse_status_success);
            rocsparse_const_spmat_descr descr = local_descr;

#define PARAMS_GET_CSC                                                                     \
    descr, rows, cols, nnz, csc_col_ptr, csc_row_ind, csc_val, col_ptr_type, row_ind_type, \
        idx_base, data_type
            bad_arg_analysis(rocsparse_const_csc_get, PARAMS_GET_CSC);
#undef PARAMS_GET_CSC

#define PARAMS_GET_SIZE descr, rows, cols, nnz
            bad_arg_analysis(rocsparse_spmat_get_size, PARAMS_GET_SIZE);
#undef PARAMS_GET_SIZE

#define PARAMS_GET_FORMAT descr, format
            bad_arg_analysis(rocsparse_spmat_get_format, PARAMS_GET_FORMAT);
#undef PARAMS_GET_FORMAT

#define PARAMS_GET_INDEX_BASE descr, idx_base
            bad_arg_analysis(rocsparse_spmat_get_index_base, PARAMS_GET_INDEX_BASE);
#undef PARAMS_GET_INDEX_BASE

            const void** values = (const void**)0x4;
#define PARAMS_GET_VALUES descr, values
            bad_arg_analysis(rocsparse_const_spmat_get_values, PARAMS_GET_VALUES);
#undef PARAMS_GET_VALUES

#define PARAMS_GET_STRIDED_BATCH descr, batch_count
            bad_arg_analysis(rocsparse_spmat_get_strided_batch, PARAMS_GET_STRIDED_BATCH);
#undef PARAMS_GET_STRIDED_BATCH

            // Destroy valid descriptors
            EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
        }

        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_create_const_bell_descr(&local_descr,
                                                                      local_rows,
                                                                      local_cols,
                                                                      local_ell_block_dir,
                                                                      local_ell_block_dim,
                                                                      local_ell_cols,
                                                                      (const void*)0x4,
                                                                      (const void*)0x4,
                                                                      local_itype,
                                                                      local_base,
                                                                      local_ttype),
                                    rocsparse_status_success);
            rocsparse_const_spmat_descr descr = local_descr;

#define PARAMS_GET_ELL                                                                         \
    descr, rows, cols, ell_block_dir, ell_block_dim, ell_cols, ell_col_ind, ell_val, idx_type, \
        idx_base, data_type
            bad_arg_analysis(rocsparse_const_bell_get, PARAMS_GET_ELL);
#undef PARAMS_GET_ELL

#define PARAMS_GET_SIZE descr, rows, cols, nnz
            bad_arg_analysis(rocsparse_spmat_get_size, PARAMS_GET_SIZE);
#undef PARAMS_GET_SIZE

#define PARAMS_GET_FORMAT descr, format
            bad_arg_analysis(rocsparse_spmat_get_format, PARAMS_GET_FORMAT);
#undef PARAMS_GET_FORMAT

#define PARAMS_GET_INDEX_BASE descr, idx_base
            bad_arg_analysis(rocsparse_spmat_get_index_base, PARAMS_GET_INDEX_BASE);
#undef PARAMS_GET_INDEX_BASE

            const void** values = (const void**)0x4;
#define PARAMS_GET_VALUES descr, values
            bad_arg_analysis(rocsparse_const_spmat_get_values, PARAMS_GET_VALUES);
#undef PARAMS_GET_VALUES

#define PARAMS_GET_STRIDED_BATCH descr, batch_count
            bad_arg_analysis(rocsparse_spmat_get_strided_batch, PARAMS_GET_STRIDED_BATCH);
#undef PARAMS_GET_STRIDED_BATCH

            // Destroy valid descriptors
            EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
        }
    }
}

template <typename I, typename J, typename T>
void testing_const_spmat_descr(const Arguments& arg)
{
    int64_t              m         = arg.M;
    int64_t              n         = arg.N;
    int64_t              nnz       = arg.nnz;
    rocsparse_direction  block_dir = arg.direction;
    int64_t              block_dim = arg.block_dim;
    rocsparse_index_base base      = arg.baseA;

    int64_t col_width = n;
    int64_t mb        = (m + block_dim - 1) / block_dim;
    int64_t nb        = (col_width + block_dim - 1) / block_dim;

    rocsparse_indextype itype = get_indextype<I>();
    rocsparse_indextype jtype = get_indextype<J>();
    rocsparse_datatype  ttype = get_datatype<T>();

    device_vector<I> row_data(nnz);
    device_vector<I> col_data(nnz);
    device_vector<J> idx_data(m + 1);
    device_vector<T> val_data(nnz);

    device_vector<I> bell_col_data(mb * nb);
    device_vector<T> bell_val_data(mb * block_dim * col_width);

    if(arg.unit_check)
    {
        /*
        *   COO
        */
        rocsparse_const_spmat_descr coo;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_const_coo_descr(
            &coo, m, n, nnz, row_data, col_data, val_data, itype, base, ttype));

        int64_t              coo_m;
        int64_t              coo_n;
        int64_t              coo_nnz;
        rocsparse_index_base coo_base;
        rocsparse_indextype  coo_itype;
        rocsparse_datatype   coo_ttype;
        const T*             coo_val;
        const I*             coo_row;
        const I*             coo_col;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_coo_get(coo,
                                                      &coo_m,
                                                      &coo_n,
                                                      &coo_nnz,
                                                      (const void**)&coo_row,
                                                      (const void**)&coo_col,
                                                      (const void**)&coo_val,
                                                      &coo_itype,
                                                      &coo_base,
                                                      &coo_ttype));

        unit_check_scalar(m, coo_m);
        unit_check_scalar(n, coo_n);
        unit_check_scalar(nnz, coo_nnz);
        unit_check_enum(base, coo_base);
        unit_check_enum(itype, coo_itype);
        unit_check_enum(ttype, coo_ttype);
        ASSERT_EQ(val_data, coo_val);
        ASSERT_EQ(row_data, coo_row);
        ASSERT_EQ(col_data, coo_col);

        coo_val = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_spmat_get_values(coo, (const void**)&coo_val));
        ASSERT_EQ(val_data, coo_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(coo));

        /*
        *   CSR
        */
        rocsparse_const_spmat_descr csr;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_const_csr_descr(
            &csr, m, n, nnz, idx_data, col_data, val_data, jtype, itype, base, ttype));

        int64_t              csr_m;
        int64_t              csr_n;
        int64_t              csr_nnz;
        rocsparse_index_base csr_base;
        rocsparse_indextype  csr_itype;
        rocsparse_indextype  csr_jtype;
        rocsparse_datatype   csr_ttype;
        const T*             csr_val;
        const J*             csr_idx;
        const I*             csr_col;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_csr_get(csr,
                                                      &csr_m,
                                                      &csr_n,
                                                      &csr_nnz,
                                                      (const void**)&csr_idx,
                                                      (const void**)&csr_col,
                                                      (const void**)&csr_val,
                                                      &csr_jtype,
                                                      &csr_itype,
                                                      &csr_base,
                                                      &csr_ttype));

        unit_check_scalar(m, csr_m);
        unit_check_scalar(n, csr_n);
        unit_check_scalar(nnz, csr_nnz);
        unit_check_enum(base, csr_base);
        unit_check_enum(itype, csr_itype);
        unit_check_enum(jtype, csr_jtype);
        unit_check_enum(ttype, csr_ttype);
        ASSERT_EQ(val_data, csr_val);
        ASSERT_EQ(idx_data, csr_idx);
        ASSERT_EQ(col_data, csr_col);

        csr_val = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_spmat_get_values(csr, (const void**)&csr_val));
        ASSERT_EQ(val_data, csr_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(csr));

        /*
        *   CSC
        */
        rocsparse_const_spmat_descr csc;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_const_csc_descr(
            &csc, n, m, nnz, idx_data, row_data, val_data, jtype, itype, base, ttype));

        int64_t              csc_m;
        int64_t              csc_n;
        int64_t              csc_nnz;
        rocsparse_index_base csc_base;
        rocsparse_indextype  csc_itype;
        rocsparse_indextype  csc_jtype;
        rocsparse_datatype   csc_ttype;
        const T*             csc_val;
        const J*             csc_idx;
        const I*             csc_row;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_csc_get(csc,
                                                      &csc_n,
                                                      &csc_m,
                                                      &csc_nnz,
                                                      (const void**)&csc_idx,
                                                      (const void**)&csc_row,
                                                      (const void**)&csc_val,
                                                      &csc_jtype,
                                                      &csc_itype,
                                                      &csc_base,
                                                      &csc_ttype));

        unit_check_scalar(m, csc_m);
        unit_check_scalar(n, csc_n);
        unit_check_scalar(nnz, csc_nnz);
        unit_check_enum(base, csc_base);
        unit_check_enum(itype, csc_itype);
        unit_check_enum(jtype, csc_jtype);
        unit_check_enum(ttype, csc_ttype);
        ASSERT_EQ(val_data, csc_val);
        ASSERT_EQ(idx_data, csc_idx);
        ASSERT_EQ(row_data, csc_row);

        csc_val = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_spmat_get_values(csc, (const void**)&csc_val));
        ASSERT_EQ(val_data, csc_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(csc));

        /*
        *   BELL
        */
        rocsparse_const_spmat_descr bell;
        CHECK_ROCSPARSE_ERROR(rocsparse_create_const_bell_descr(&bell,
                                                                m,
                                                                n,
                                                                block_dir,
                                                                block_dim,
                                                                col_width,
                                                                bell_col_data,
                                                                bell_val_data,
                                                                itype,
                                                                base,
                                                                ttype));

        int64_t              bell_m;
        int64_t              bell_n;
        int64_t              bell_col_width;
        int64_t              bell_block_dim;
        rocsparse_direction  bell_block_dir;
        rocsparse_index_base bell_base;
        rocsparse_indextype  bell_itype;
        rocsparse_datatype   bell_ttype;
        const T*             bell_val;
        const I*             bell_col;

        CHECK_ROCSPARSE_ERROR(rocsparse_const_bell_get(bell,
                                                       &bell_m,
                                                       &bell_n,
                                                       &bell_block_dir,
                                                       &bell_block_dim,
                                                       &bell_col_width,
                                                       (const void**)&bell_col,
                                                       (const void**)&bell_val,
                                                       &bell_itype,
                                                       &bell_base,
                                                       &bell_ttype));

        unit_check_scalar(m, bell_m);
        unit_check_scalar(n, bell_n);
        unit_check_scalar(col_width, bell_col_width);
        unit_check_scalar(block_dim, bell_block_dim);
        unit_check_enum(block_dir, bell_block_dir);
        unit_check_enum(base, bell_base);
        unit_check_enum(itype, bell_itype);
        unit_check_enum(ttype, bell_ttype);
        ASSERT_EQ(bell_val_data, bell_val);
        ASSERT_EQ(bell_col_data, bell_col);

        bell_val = nullptr;
        CHECK_ROCSPARSE_ERROR(rocsparse_const_spmat_get_values(bell, (const void**)&bell_val));
        ASSERT_EQ(bell_val_data, bell_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(bell));
    }

    if(arg.timing)
    {
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                        \
    template void testing_const_spmat_descr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_const_spmat_descr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
void testing_const_spmat_descr_extra(const Arguments& arg) {}
