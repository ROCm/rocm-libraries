/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_spmat_descr_bad_arg(const Arguments& arg)
{
    static const size_t   safe_size = 100;
    rocsparse_spmat_descr local_descr{};
    int64_t               local_rows          = safe_size;
    int64_t               local_cols          = safe_size;
    int64_t               local_nnz           = safe_size;
    rocsparse_direction   local_ell_block_dir = rocsparse_direction_row;
    int64_t               local_ell_block_dim = safe_size;
    int64_t               local_ell_cols      = safe_size;
    rocsparse_index_base  local_base          = rocsparse_index_base_zero;
    rocsparse_format      local_format        = rocsparse_format_csr;
    rocsparse_indextype   local_itype         = get_indextype<I>();
    rocsparse_indextype   local_jtype         = get_indextype<J>();
    rocsparse_datatype    local_ttype         = get_datatype<T>();
    rocsparse_int         local_batch_count   = safe_size;

    {
        rocsparse_spmat_descr* descr         = &local_descr;
        int64_t                rows          = local_rows;
        int64_t                cols          = local_cols;
        int64_t                nnz           = local_nnz;
        rocsparse_direction    ell_block_dir = local_ell_block_dir;
        int64_t                ell_block_dim = local_ell_block_dim;
        int64_t                ell_cols      = local_ell_cols;
        rocsparse_index_base   idx_base      = local_base;

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
        bad_arg_analysis(rocsparse_create_coo_descr, PARAMS_CREATE_COO);
#undef PARAMS_CREATE_COO

        // nnz > rows * cols
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_coo_descr(descr,
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
        bad_arg_analysis(rocsparse_create_csr_descr, PARAMS_CREATE_CSR);
#undef PARAMS_CREATE_CSR

        // nnz > rows * cols
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_csr_descr(descr,
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
        bad_arg_analysis(rocsparse_create_csc_descr, PARAMS_CREATE_CSC);
#undef PARAMS_CREATE_CSC

        // nnz > rows * cols
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_csc_descr(descr,
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
        bad_arg_analysis(rocsparse_create_bell_descr, PARAMS_CREATE_ELL);
#undef PARAMS_CREATE_ELL

        // block_dim = 0
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_bell_descr(descr,
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
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_bell_descr(descr,
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
            rocsparse_create_coo_descr(
                descr, 0, cols, 0, nullptr, nullptr, nullptr, idx_type, idx_base, data_type),
            rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_coo_descr(
                descr, rows, 0, 0, nullptr, nullptr, nullptr, idx_type, idx_base, data_type),
            rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_coo_descr(
                descr, rows, cols, 0, nullptr, nullptr, nullptr, idx_type, idx_base, data_type),
            rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_create_csr_descr(descr,
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
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_csr_descr(descr,
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
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_csr_descr(descr,
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
    EXPECT_ROCSPARSE_STATUS(
			    rocsparse_create_csr_descr(
							     descr, rows, cols, 0, nullptr, nullptr, nullptr, row_ptr_type, col_ind_type, idx_base, data_type),
			    rocsparse_status_success);
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(local_descr), rocsparse_status_success);
#endif
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_csc_descr(descr,
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
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_csc_descr(descr,
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
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_csc_descr(descr,
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
    EXPECT_ROCSPARSE_STATUS(rocsparse_create_bell_descr(descr,
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
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_bell_descr(descr,
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
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_bell_descr(descr,
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

        void** coo_row_ind = (void**)0x4;
        void** coo_col_ind = (void**)0x4;
        void** coo_val     = (void**)0x4;
        void** csr_row_ptr = (void**)0x4;
        void** csr_col_ind = (void**)0x4;
        void** csr_val     = (void**)0x4;
        void** ell_col_ind = (void**)0x4;
        void** ell_val     = (void**)0x4;

        rocsparse_int* batch_count = &local_batch_count;

        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_create_coo_descr(&local_descr,
                                                               local_rows,
                                                               local_cols,
                                                               local_nnz,
                                                               (void*)0x4,
                                                               (void*)0x4,
                                                               (void*)0x4,
                                                               local_itype,
                                                               local_base,
                                                               local_ttype),
                                    rocsparse_status_success);
            rocsparse_spmat_descr descr = local_descr;

#define PARAMS_GET_COO \
    descr, rows, cols, nnz, coo_row_ind, coo_col_ind, coo_val, idx_type, idx_base, data_type
            bad_arg_analysis(rocsparse_coo_get, PARAMS_GET_COO);
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

            void** values = (void**)0x4;
#define PARAMS_GET_VALUES descr, values
            bad_arg_analysis(rocsparse_spmat_get_values, PARAMS_GET_VALUES);
#undef PARAMS_GET_VALUES

#define PARAMS_GET_STRIDED_BATCH descr, batch_count
            bad_arg_analysis(rocsparse_spmat_get_strided_batch, PARAMS_GET_STRIDED_BATCH);
#undef PARAMS_GET_STRIDED_BATCH

            // Destroy valid descriptors
            EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
        }

        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_create_csr_descr(&local_descr,
                                                               local_rows,
                                                               local_cols,
                                                               local_nnz,
                                                               (void*)0x4,
                                                               (void*)0x4,
                                                               (void*)0x4,
                                                               local_itype,
                                                               local_jtype,
                                                               local_base,
                                                               local_ttype),
                                    rocsparse_status_success);
            rocsparse_spmat_descr descr = local_descr;

#define PARAMS_GET_CSR                                                                     \
    descr, rows, cols, nnz, csr_row_ptr, csr_col_ind, csr_val, row_ptr_type, col_ind_type, \
        idx_base, data_type
            bad_arg_analysis(rocsparse_csr_get, PARAMS_GET_CSR);
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

            void** values = (void**)0x4;
#define PARAMS_GET_VALUES descr, values
            bad_arg_analysis(rocsparse_spmat_get_values, PARAMS_GET_VALUES);
#undef PARAMS_GET_VALUES

#define PARAMS_GET_STRIDED_BATCH descr, batch_count
            bad_arg_analysis(rocsparse_spmat_get_strided_batch, PARAMS_GET_STRIDED_BATCH);
#undef PARAMS_GET_STRIDED_BATCH

            // Destroy valid descriptors
            EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
        }

        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_create_csc_descr(&local_descr,
                                                               local_rows,
                                                               local_cols,
                                                               local_nnz,
                                                               (void*)0x4,
                                                               (void*)0x4,
                                                               (void*)0x4,
                                                               local_itype,
                                                               local_jtype,
                                                               local_base,
                                                               local_ttype),
                                    rocsparse_status_success);
            rocsparse_spmat_descr descr = local_descr;

            rocsparse_indextype* col_ptr_type = &local_itype;
            rocsparse_indextype* row_ind_type = &local_jtype;
            void**               csc_row_ind  = (void**)0x4;
            void**               csc_col_ptr  = (void**)0x4;
            void**               csc_val      = (void**)0x4;
#define PARAMS_GET_CSC                                                                     \
    descr, rows, cols, nnz, csc_col_ptr, csc_row_ind, csc_val, col_ptr_type, row_ind_type, \
        idx_base, data_type
            bad_arg_analysis(rocsparse_csc_get, PARAMS_GET_CSC);
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

            void** values = (void**)0x4;
#define PARAMS_GET_VALUES descr, values
            bad_arg_analysis(rocsparse_spmat_get_values, PARAMS_GET_VALUES);
#undef PARAMS_GET_VALUES

#define PARAMS_GET_STRIDED_BATCH descr, batch_count
            bad_arg_analysis(rocsparse_spmat_get_strided_batch, PARAMS_GET_STRIDED_BATCH);
#undef PARAMS_GET_STRIDED_BATCH

            // Destroy valid descriptors
            EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spmat_descr(descr), rocsparse_status_success);
        }

        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_create_bell_descr(&local_descr,
                                                                local_rows,
                                                                local_cols,
                                                                local_ell_block_dir,
                                                                local_ell_block_dim,
                                                                local_ell_cols,
                                                                (void*)0x4,
                                                                (void*)0x4,
                                                                local_itype,
                                                                local_base,
                                                                local_ttype),
                                    rocsparse_status_success);
            rocsparse_spmat_descr descr = local_descr;

#define PARAMS_GET_ELL                                                                         \
    descr, rows, cols, ell_block_dir, ell_block_dim, ell_cols, ell_col_ind, ell_val, idx_type, \
        idx_base, data_type
            bad_arg_analysis(rocsparse_bell_get, PARAMS_GET_ELL);
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

            void** values = (void**)0x4;
#define PARAMS_GET_VALUES descr, values
            bad_arg_analysis(rocsparse_spmat_get_values, PARAMS_GET_VALUES);
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
void testing_spmat_descr(const Arguments& arg)
{
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                  \
    template void testing_spmat_descr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spmat_descr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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
void testing_spmat_descr_extra(const Arguments& arg) {}
