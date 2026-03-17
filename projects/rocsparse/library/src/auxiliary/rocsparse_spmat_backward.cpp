/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/auxiliary/rocsparse_idvec_descr.h"
#include "rocsparse_argdescr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_logging.hpp"
#include "rocsparse_spattern_descr.hpp"

static void define_coo(rocsparse_spmat_descr that,
                       int64_t               rows,
                       int64_t               cols,
                       int64_t               nnz,
                       const void*           const_row_data,
                       void*                 row_data,
                       const void*           const_col_data,
                       void*                 col_data,
                       const void*           const_val_data,
                       void*                 val_data,
                       rocsparse_indextype   idx_type,
                       rocsparse_index_base  idx_base,
                       rocsparse_datatype    val_type,
                       rocsparse_mat_descr   mat_descr,
                       rocsparse_mat_info    mat_info)
{
    auto spattern = that->get_spattern();
    auto row      = spattern->get_row_data();
    auto col      = spattern->get_col_data();
    auto val      = that->get_values();
    row->define(idx_type, idx_base, nnz, 1, const_row_data, row_data);
    col->define(idx_type, idx_base, nnz, 1, const_col_data, col_data);
    val->define(val_type, nnz, 1, const_val_data, val_data);
    spattern->define_coo(rows, cols, nnz, row, col, mat_descr, mat_info);
    that->define(spattern, val, mat_info);
}

static void define_coo_aos(rocsparse_spmat_descr that,
                           int64_t               rows,
                           int64_t               cols,
                           int64_t               nnz,
                           const void*           const_ind_data,
                           void*                 ind_data,
                           const void*           const_val_data,
                           void*                 val_data,
                           rocsparse_indextype   idx_type,
                           rocsparse_index_base  idx_base,
                           rocsparse_datatype    val_type,
                           rocsparse_mat_descr   mat_descr,
                           rocsparse_mat_info    mat_info)
{

    auto spattern = that->get_spattern();
    auto row      = spattern->get_row_data();
    auto col      = spattern->get_col_data();
    auto val      = that->get_values();

    row->define(idx_type, idx_base, nnz, 2, const_ind_data, ind_data);

    const void* s_const_ind_data
        = reinterpret_cast<const char*>(const_ind_data) + rocsparse::indextype_sizeof(idx_type);
    void* s_ind_data
        = (ind_data != nullptr)
              ? (reinterpret_cast<char*>(ind_data) + rocsparse::indextype_sizeof(idx_type))
              : nullptr;

    col->define(idx_type, idx_base, nnz, 2, s_const_ind_data, s_ind_data);
    val->define(val_type, nnz, 1, const_val_data, val_data);
    spattern->define_coo_aos(rows, cols, nnz, row, col, mat_descr, mat_info);
    that->define(spattern, val, mat_info);
}

static void define_bsr(rocsparse_spmat_descr that,
                       int64_t               mb,
                       int64_t               nb,
                       int64_t               nnzb,
                       rocsparse_direction   block_dir,
                       int64_t               block_dim,

                       const void*          const_row_data,
                       void*                row_data,
                       const void*          const_col_data,
                       void*                col_data,
                       const void*          const_val_data,
                       void*                val_data,
                       rocsparse_indextype  row_type,
                       rocsparse_indextype  col_type,
                       rocsparse_index_base idx_base,
                       rocsparse_datatype   val_type,
                       rocsparse_mat_descr  mat_descr,
                       rocsparse_mat_info   mat_info)
{
    auto spattern = that->get_spattern();
    auto row      = spattern->get_row_data();
    auto col      = spattern->get_col_data();
    auto val      = that->get_values();
    row->define(row_type, idx_base, mb + 1, 1, const_row_data, row_data);
    col->define(col_type, idx_base, nnzb, 1, const_col_data, col_data);
    val->define(val_type, nnzb * block_dim * block_dim, 1, const_val_data, val_data);
    spattern->define_bsr(mb, nb, nnzb, block_dir, block_dim, row, col, mat_descr, mat_info);
    that->define(spattern, val, mat_info);
}

static void define_sell(rocsparse_spmat_descr that,
                        int64_t               rows,
                        int64_t               cols,
                        int64_t               nnz,
                        int64_t               sell_slice_size,
                        int64_t               sell_colval_size,
                        const void*           const_row_data,
                        void*                 row_data,
                        const void*           const_col_data,
                        void*                 col_data,
                        const void*           const_val_data,
                        void*                 val_data,
                        rocsparse_indextype   row_type,
                        rocsparse_indextype   col_type,
                        rocsparse_index_base  idx_base,
                        rocsparse_datatype    val_type,
                        rocsparse_mat_descr   mat_descr,
                        rocsparse_mat_info    mat_info)
{
    auto          spattern = that->get_spattern();
    const int64_t nslices  = (rows - 1) / sell_slice_size + 1;
    auto          row      = spattern->get_row_data();
    auto          col      = spattern->get_col_data();
    auto          val      = that->get_values();
    row->define(row_type, idx_base, nslices + 1, 1, const_row_data, row_data);
    col->define(col_type, idx_base, sell_colval_size, 1, const_col_data, col_data);
    val->define(val_type, sell_colval_size, 1, const_val_data, val_data);
    spattern->define_sell(
        rows, cols, nnz, sell_slice_size, sell_colval_size, row, col, mat_descr, mat_info);
    that->define(spattern, val, mat_info);
}

static void define_csr(rocsparse_spmat_descr that,
                       int64_t               rows,
                       int64_t               cols,
                       int64_t               nnz,
                       const void*           const_row_data,
                       void*                 row_data,
                       const void*           const_col_data,
                       void*                 col_data,
                       const void*           const_val_data,
                       void*                 val_data,
                       rocsparse_indextype   row_type,
                       rocsparse_indextype   col_type,
                       rocsparse_index_base  idx_base,
                       rocsparse_datatype    val_type,
                       rocsparse_mat_descr   mat_descr,
                       rocsparse_mat_info    mat_info)
{
    auto spattern = that->get_spattern();
    auto row      = spattern->get_row_data();
    auto col      = spattern->get_col_data();
    auto val      = that->get_values();
    row->define(row_type, idx_base, rows + 1, 1, const_row_data, row_data);
    col->define(col_type, idx_base, nnz, 1, const_col_data, col_data);
    val->define(val_type, nnz, 1, const_val_data, val_data);
    spattern->define_csr(rows, cols, nnz, row, col, mat_descr, mat_info);
    that->define(spattern, val, mat_info);
}

static void define_csc(rocsparse_spmat_descr that,
                       int64_t               rows,
                       int64_t               cols,
                       int64_t               nnz,
                       const void*           const_row_data,
                       void*                 row_data,
                       const void*           const_col_data,
                       void*                 col_data,
                       const void*           const_val_data,
                       void*                 val_data,
                       rocsparse_indextype   row_type,
                       rocsparse_indextype   col_type,
                       rocsparse_index_base  idx_base,
                       rocsparse_datatype    val_type,
                       rocsparse_mat_descr   mat_descr,
                       rocsparse_mat_info    mat_info)
{
    auto spattern = that->get_spattern();
    auto row      = spattern->get_row_data();
    auto col      = spattern->get_col_data();
    auto val      = that->get_values();
    row->define(row_type, idx_base, nnz, 1, const_row_data, row_data);
    col->define(col_type, idx_base, cols + 1, 1, const_col_data, col_data);
    val->define(val_type, nnz, 1, const_val_data, val_data);
    spattern->define_csc(rows, cols, nnz, row, col, mat_descr, mat_info);
    that->define(spattern, val, mat_info);
}

static void define_bell(rocsparse_spmat_descr that,
                        int64_t               rows,
                        int64_t               cols,
                        rocsparse_direction   block_dir,
                        int64_t               block_dim,
                        const void*           const_ind_data,
                        void*                 ind_data,
                        const void*           const_val_data,
                        void*                 val_data,
                        int64_t               width,
                        rocsparse_indextype   idx_type,
                        rocsparse_index_base  idx_base,
                        rocsparse_datatype    val_type,
                        rocsparse_mat_descr   mat_descr,
                        rocsparse_mat_info    mat_info)
{

    auto          spattern = that->get_spattern();
    auto          col      = spattern->get_col_data();
    auto          val      = that->get_values();
    const int64_t nnz_s    = rows * width;
    const int64_t nnz_n    = rows * width * block_dim * block_dim;
    col->define(idx_type, idx_base, nnz_s, 1, const_ind_data, ind_data);
    val->define(val_type, nnz_n, 1, const_val_data, val_data);
    spattern->define_bell(rows, cols, width, block_dir, block_dim, col, mat_descr, mat_info);
    that->define(spattern, val, mat_info);
}

static void define_ell(rocsparse_spmat_descr that,
                       int64_t               rows,
                       int64_t               cols,
                       const void*           const_ind_data,
                       void*                 ind_data,
                       const void*           const_val_data,
                       void*                 val_data,
                       int64_t               width,
                       rocsparse_indextype   idx_type,
                       rocsparse_index_base  idx_base,
                       rocsparse_datatype    val_type,
                       rocsparse_mat_descr   mat_descr,
                       rocsparse_mat_info    mat_info)
{
    auto spattern = that->get_spattern();
    auto col      = spattern->get_col_data();
    auto val      = that->get_values();
    col->define(idx_type, idx_base, width * rows, 1, const_ind_data, ind_data);
    val->define(val_type, width * rows, 1, const_val_data, val_data);
    spattern->define_ell(rows, cols, width, col, mat_descr, mat_info);
    that->define(spattern, val, mat_info);
}
#if 0
  void define_sell(int64_t              rows,
		   int64_t              cols,
		   int64_t              nnz,
		   int64_t              sell_slice_size,
		   int64_t              sell_colval_size,
		   const void*          const_row_data,
		   void*                row_data,
		   const void*          const_col_data,
		   void*                col_data,
		   const void*          const_val_data,
		   void*                val_data,
		   rocsparse_indextype  row_type,
		   rocsparse_indextype  col_type,
		   rocsparse_index_base idx_base,
		   rocsparse_datatype   val_type,
		 rocsparse_mat_descr mat_descr,
		 rocsparse_mat_info mat_info);

  void define_coo(int64_t              rows,
		  int64_t              cols,
		  int64_t              nnz,
		  const void*          const_row_data,
		  void*                row_data,
		  const void*          const_col_data,
		  void*                col_data,
		  const void*          const_val,
		  void*                val,
		  rocsparse_indextype  idx_type,
		  rocsparse_index_base idx_base,
		  rocsparse_datatype   val_type,
		  rocsparse_mat_descr mat_descr,
		  rocsparse_mat_info mat_info);

  void define_coo_aos(int64_t     rows,
		    int64_t     cols,
		    int64_t     nnz,
		    const void* const_ind_data,
		    void*       ind_data,
		    const void*          const_val,
		    void*                val,
		    rocsparse_indextype  idx_type,
		    rocsparse_index_base idx_base,
		    rocsparse_datatype   val_type,
		  rocsparse_mat_descr mat_descr,
		  rocsparse_mat_info mat_info);

  void define_bsr(int64_t             mb,
		  int64_t             nb,
		  int64_t             nnzb,
		  rocsparse_direction block_dir,
		  int64_t             block_dim,
		  const void*          const_row_data,
		  void*                row_data,
		  const void*          const_col_data,
		  void*                col_data,
		  const void*          const_val,
		  void*                val,
		  rocsparse_indextype  row_type,
		  rocsparse_indextype  col_type,
		  rocsparse_index_base idx_base,
		  rocsparse_datatype   val_type,
		  rocsparse_mat_descr mat_descr,
		  rocsparse_mat_info mat_info);

  void define_csr(int64_t              rows,
		  int64_t              cols,
		  int64_t              nnz,
		  const void*          const_row_data,
		  void*                row_data,
		  const void*          const_col_data,
		  void*                col_data,
		  const void*          const_val,
		  void*                val,
		  rocsparse_indextype  row_type,
		  rocsparse_indextype  col_type,
		  rocsparse_index_base idx_base,
		  rocsparse_datatype   val_type,
		  rocsparse_mat_descr mat_descr,
		  rocsparse_mat_info mat_info);

  void define_csc(int64_t              rows,
		  int64_t              cols,
                                            int64_t              nnz,
		  const void*          const_row_data,
		  void*                row_data,
		  const void*          const_col_data,
		  void*                col_data,
		  const void*          const_val,
		  void*                val,
		  rocsparse_indextype  row_type,
		  rocsparse_indextype  col_type,
		  rocsparse_index_base idx_base,
		  rocsparse_datatype   val_type,
		  rocsparse_mat_descr mat_descr,
		  rocsparse_mat_info mat_info);

  void define_bell(int64_t              rows,
		   int64_t              cols,
		   rocsparse_direction  block_dir,
		   int64_t              block_dim,
		   const void*          const_ind_data,
		   void*                ind_data,
		   const void*          const_val,
		   void*                val,
		   int64_t              width,
		   rocsparse_indextype  idx_type,
		   rocsparse_index_base idx_base,
		   rocsparse_datatype   val_type,
		   rocsparse_mat_descr mat_descr,
		   rocsparse_mat_info mat_info);

  void define_ell(int64_t              rows,
		  int64_t              cols,
		  const void*          const_ind_data,
		  void*                ind_data,
		  const void*          const_val,
		  void*                val,
		  int64_t              width,
		  rocsparse_indextype  idx_type,
		  rocsparse_index_base idx_base,
		  rocsparse_datatype   val_type,
		  rocsparse_mat_descr mat_descr,
		  rocsparse_mat_info mat_info);
#endif
#ifdef __cplusplus
extern "C" {
#endif

rocsparse_status rocsparse_create_csr_descr_SWDEV_453599(rocsparse_spmat_descr* descr,
                                                         int64_t                rows,
                                                         int64_t                cols,
                                                         int64_t                nnz,
                                                         void*                  csr_row_ptr,
                                                         void*                  csr_col_ind,
                                                         void*                  csr_val,
                                                         rocsparse_indextype    row_ptr_type,
                                                         rocsparse_indextype    col_ind_type,
                                                         rocsparse_index_base   idx_base,
                                                         rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);

    //
    // SWDEV-340500, this is a non-sense.
    // cusparse parity behavior should be fixed in hipsparse, not here.
    //
    // cusparse allows setting NULL for the pointers when nnz == 0. See SWDEV_453599 for reproducer.
    // This function exists so that hipsparse can follow this behaviour without affecting rocsparse.
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_val);
    ROCSPARSE_CHECKARG_ENUM(7, row_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_csr(spmat_descr,
               rows,
               cols,
               nnz,
               csr_row_ptr,
               csr_row_ptr,
               csr_col_ind,
               csr_col_ind,
               csr_val,
               csr_val,
               row_ptr_type,
               col_ind_type,
               idx_base,
               data_type,
               nullptr,
               nullptr);
    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_coo_descr creates a descriptor holding the COO matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_coo_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  coo_row_ind,
                                            void*                  coo_col_ind,
                                            void*                  coo_val,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, coo_val);
    ROCSPARSE_CHECKARG_ENUM(7, idx_type);
    ROCSPARSE_CHECKARG_ENUM(8, idx_base);
    ROCSPARSE_CHECKARG_ENUM(9, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_coo(spmat_descr,
               rows,
               cols,
               nnz,
               coo_row_ind,
               coo_row_ind,
               coo_col_ind,
               coo_col_ind,
               coo_val,
               coo_val,
               idx_type,
               idx_base,
               data_type,
               nullptr,
               nullptr);

    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_coo_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  coo_row_ind,
                                                  const void*                  coo_col_ind,
                                                  const void*                  coo_val,
                                                  rocsparse_indextype          idx_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, coo_val);
    ROCSPARSE_CHECKARG_ENUM(7, idx_type);
    ROCSPARSE_CHECKARG_ENUM(8, idx_base);
    ROCSPARSE_CHECKARG_ENUM(9, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_coo(spmat_descr,
               rows,
               cols,
               nnz,
               coo_row_ind,
               nullptr,
               coo_col_ind,
               nullptr,
               coo_val,
               nullptr,
               idx_type,
               idx_base,
               data_type,
               nullptr,
               nullptr);
    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_coo_aos_descr creates a descriptor holding the COO matrix
 * data, sizes and properties where the row pointer and column indices are stored
 * using array of structure (AoS) format. It must be called prior to all subsequent
 * library function calls that involve sparse matrices. It should be destroyed at
 * the end using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_coo_aos_descr(rocsparse_spmat_descr* descr,
                                                int64_t                rows,
                                                int64_t                cols,
                                                int64_t                nnz,
                                                void*                  coo_ind,
                                                void*                  coo_val,
                                                rocsparse_indextype    idx_type,
                                                rocsparse_index_base   idx_base,
                                                rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_val);
    ROCSPARSE_CHECKARG_ENUM(6, idx_type);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);
    ROCSPARSE_CHECKARG_ENUM(8, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_coo_aos(spmat_descr,
                   rows,
                   cols,
                   nnz,
                   coo_ind,
                   coo_ind,
                   coo_val,
                   coo_val,
                   idx_type,
                   idx_base,
                   data_type,
                   nullptr,
                   nullptr);
    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_csr_descr creates a descriptor holding the CSR matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_csr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csr_row_ptr,
                                            void*                  csr_col_ind,
                                            void*                  csr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);

    //
    // SWDEV-340500, this is a non-sense.
    // cusparse parity behavior should be fixed in hipsparse, not here.
    //
    //    ROCSPARSE_CHECKARG(4, (rows > 0 && nnz > 0 && csr_row_ptr == nullptr), csr_row_ptr, rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG_ARRAY(4, rows, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_val);
    ROCSPARSE_CHECKARG_ENUM(7, row_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);
    // TOTO
    auto spmat_descr = new _rocsparse_spmat_descr;
    define_csr(spmat_descr,
               rows,
               cols,
               nnz,
               csr_row_ptr,
               csr_row_ptr,
               csr_col_ind,
               csr_col_ind,
               csr_val,
               csr_val,
               row_ptr_type,
               col_ind_type,
               idx_base,
               data_type,
               nullptr,
               nullptr);
    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_csr_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csr_row_ptr,
                                                  const void*                  csr_col_ind,
                                                  const void*                  csr_val,
                                                  rocsparse_indextype          row_ptr_type,
                                                  rocsparse_indextype          col_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);

    //
    // SWDEV-340500, this is a non-sense.
    // cusparse parity behavior should be fixed in hipsparse, not here.
    //
    //    ROCSPARSE_CHECKARG(4, (rows > 0 && nnz > 0 && csr_row_ptr == nullptr), csr_row_ptr, rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG_ARRAY(4, rows, csr_row_ptr);

    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_val);
    ROCSPARSE_CHECKARG_ENUM(7, row_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_csr(spmat_descr,
               rows,
               cols,
               nnz,
               csr_row_ptr,
               nullptr,
               csr_col_ind,
               nullptr,
               csr_val,
               nullptr,
               row_ptr_type,
               col_ind_type,
               idx_base,
               data_type,
               nullptr,
               nullptr);
    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_csc_descr creates a descriptor holding the CSC matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_csc_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csc_col_ptr,
                                            void*                  csc_row_ind,
                                            void*                  csc_val,
                                            rocsparse_indextype    col_ptr_type,
                                            rocsparse_indextype    row_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, cols, csc_col_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csc_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csc_val);
    ROCSPARSE_CHECKARG_ENUM(7, col_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, row_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);
    auto spmat_descr = new _rocsparse_spmat_descr;
    define_csc(spmat_descr,
               rows,
               cols,
               nnz,
               csc_col_ptr,
               csc_col_ptr,
               csc_row_ind,
               csc_row_ind,
               csc_val,
               csc_val,
               col_ptr_type,
               row_ind_type,
               idx_base,
               data_type,
               nullptr,
               nullptr);
    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_csc_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csc_col_ptr,
                                                  const void*                  csc_row_ind,
                                                  const void*                  csc_val,
                                                  rocsparse_indextype          col_ptr_type,
                                                  rocsparse_indextype          row_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, cols, csc_col_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csc_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csc_val);
    ROCSPARSE_CHECKARG_ENUM(7, col_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, row_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_csc(spmat_descr,
               rows,
               cols,
               nnz,
               csc_col_ptr,
               nullptr,
               csc_row_ind,
               nullptr,
               csc_val,
               nullptr,
               col_ptr_type,
               row_ind_type,
               idx_base,
               data_type,
               nullptr,
               nullptr);

    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_ell_descr creates a descriptor holding the ELL matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_ell_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            void*                  ell_col_ind,
                                            void*                  ell_val,
                                            int64_t                ell_width,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(5, ell_width);
    ROCSPARSE_CHECKARG_ARRAY(3, rows * ell_width, ell_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(4, rows * ell_width, ell_val);
    ROCSPARSE_CHECKARG(5, ell_width, (ell_width > cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ENUM(6, idx_type);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);
    ROCSPARSE_CHECKARG_ENUM(8, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_ell(spmat_descr,
               rows,
               cols,
               ell_col_ind,
               ell_col_ind,
               ell_val,
               ell_val,
               ell_width,
               idx_type,
               idx_base,
               data_type,
               nullptr,
               nullptr);
    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_bell_descr creates a descriptor holding the
 * BLOCKED ELL matrix data, sizes and properties. It must be called prior to all
 * subsequent library function calls that involve sparse matrices.
 * It should be destroyed at the end using rocsparse_destroy_spmat_descr().
 * All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_bell_descr(rocsparse_spmat_descr* descr,
                                             int64_t                rows,
                                             int64_t                cols,
                                             rocsparse_direction    ell_block_dir,
                                             int64_t                ell_block_dim,
                                             int64_t                ell_cols,
                                             void*                  ell_col_ind,
                                             void*                  ell_val,
                                             rocsparse_indextype    idx_type,
                                             rocsparse_index_base   idx_base,
                                             rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_ENUM(3, ell_block_dir);
    ROCSPARSE_CHECKARG_SIZE(4, ell_block_dim);
    ROCSPARSE_CHECKARG(4, ell_block_dim, (ell_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(5, ell_cols);
    ROCSPARSE_CHECKARG(5, ell_cols, (ell_cols > cols), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(6, ell_cols * ell_block_dim, ell_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(7, ell_cols * ell_block_dim, ell_val);

    ROCSPARSE_CHECKARG_ENUM(8, idx_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_bell(spmat_descr,
                rows,
                cols,
                ell_block_dir,
                ell_block_dim,
                ell_col_ind,
                ell_col_ind,
                ell_val,
                ell_val,
                ell_cols,
                idx_type,
                idx_base,
                data_type,
                nullptr,
                nullptr);
    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_bell_descr(rocsparse_const_spmat_descr* descr,
                                                   int64_t                      rows,
                                                   int64_t                      cols,
                                                   rocsparse_direction          ell_block_dir,
                                                   int64_t                      ell_block_dim,
                                                   int64_t                      ell_cols,
                                                   const void*                  ell_col_ind,
                                                   const void*                  ell_val,
                                                   rocsparse_indextype          idx_type,
                                                   rocsparse_index_base         idx_base,
                                                   rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_ENUM(3, ell_block_dir);
    ROCSPARSE_CHECKARG_SIZE(4, ell_block_dim);
    ROCSPARSE_CHECKARG(4, ell_block_dim, (ell_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(5, ell_cols);
    ROCSPARSE_CHECKARG(5, ell_cols, ell_cols > cols, rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(6, rows * ell_cols * ell_block_dim, ell_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(7, rows * ell_cols * ell_block_dim, ell_val);

    ROCSPARSE_CHECKARG_ENUM(8, idx_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_bell(spmat_descr,
                rows,
                cols,
                ell_block_dir,
                ell_block_dim,
                ell_col_ind,
                nullptr,
                ell_val,
                nullptr,
                ell_cols,
                idx_type,
                idx_base,
                data_type,
                nullptr,
                nullptr);

    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_sell_descr creates a descriptor holding the
 * SLICED ELL matrix data, sizes and properties. It must be called prior to all
 * subsequent library function calls that involve sparse matrices.
 * It should be destroyed at the end using rocsparse_destroy_spmat_descr().
 * All data pointers remain valid.
 *******************************************************************************/

extern "C" rocsparse_status rocsparse_create_sell_descr(rocsparse_spmat_descr* descr,
                                                        int64_t                rows,
                                                        int64_t                cols,
                                                        int64_t                nnz,
                                                        int64_t                sell_slice_size,
                                                        int64_t                sell_colval_size,
                                                        void*                  sell_slice_offsets,
                                                        void*                  sell_col_ind,
                                                        void*                  sell_val,
                                                        rocsparse_indextype sell_slice_offsets_type,
                                                        rocsparse_indextype sell_col_ind_type,
                                                        rocsparse_index_base idx_base,
                                                        rocsparse_datatype   data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_SIZE(4, sell_slice_size);
    ROCSPARSE_CHECKARG(4, sell_slice_size, (sell_slice_size == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(5, sell_colval_size);

    ROCSPARSE_CHECKARG(3, nnz, (nnz > sell_colval_size), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(4, sell_slice_size, (sell_slice_size > rows), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(6, rows, sell_slice_offsets);
    ROCSPARSE_CHECKARG_ARRAY(7, sell_colval_size, sell_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, sell_colval_size, sell_val);

    ROCSPARSE_CHECKARG_ENUM(9, sell_slice_offsets_type);
    ROCSPARSE_CHECKARG_ENUM(10, sell_col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(11, idx_base);
    ROCSPARSE_CHECKARG_ENUM(12, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_sell(spmat_descr,
                rows,
                cols,
                nnz,
                sell_slice_size,
                sell_colval_size,
                sell_slice_offsets,
                sell_slice_offsets,
                sell_col_ind,
                sell_col_ind,
                sell_val,
                sell_val,
                sell_slice_offsets_type,
                sell_col_ind_type,
                idx_base,
                data_type,
                nullptr,
                nullptr);
    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_sell_descr(rocsparse_const_spmat_descr* descr,
                                                   int64_t                      rows,
                                                   int64_t                      cols,
                                                   int64_t                      nnz,
                                                   int64_t                      sell_slice_size,
                                                   int64_t                      sell_colval_size,
                                                   const void*                  sell_slice_offsets,
                                                   const void*                  sell_col_ind,
                                                   const void*                  sell_val,
                                                   rocsparse_indextype  sell_slice_offsets_type,
                                                   rocsparse_indextype  sell_col_ind_type,
                                                   rocsparse_index_base idx_base,
                                                   rocsparse_datatype   data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_SIZE(4, sell_slice_size);
    ROCSPARSE_CHECKARG(4, sell_slice_size, (sell_slice_size == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(5, sell_colval_size);

    ROCSPARSE_CHECKARG(3, nnz, (nnz > sell_colval_size), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(4, sell_slice_size, (sell_slice_size > rows), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(6, (rows / sell_slice_size + 1), sell_slice_offsets);
    ROCSPARSE_CHECKARG_ARRAY(7, sell_colval_size, sell_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, sell_colval_size, sell_val);

    ROCSPARSE_CHECKARG_ENUM(9, sell_slice_offsets_type);
    ROCSPARSE_CHECKARG_ENUM(10, sell_col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(11, idx_base);
    ROCSPARSE_CHECKARG_ENUM(12, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_sell(spmat_descr,
                rows,
                cols,
                nnz,
                sell_slice_size,
                sell_colval_size,
                sell_slice_offsets,
                nullptr,
                sell_col_ind,
                nullptr,
                sell_val,
                nullptr,
                sell_slice_offsets_type,
                sell_col_ind_type,
                idx_base,
                data_type,
                nullptr,
                nullptr);

    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_bsr_descr creates a descriptor holding the BSR matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_bsr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                mb,
                                            int64_t                nb,
                                            int64_t                nnzb,
                                            rocsparse_direction    block_dir,
                                            int64_t                block_dim,
                                            void*                  bsr_row_ptr,
                                            void*                  bsr_col_ind,
                                            void*                  bsr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, mb);
    ROCSPARSE_CHECKARG_SIZE(2, nb);
    ROCSPARSE_CHECKARG_SIZE(3, nnzb);
    ROCSPARSE_CHECKARG(3, nnzb, (nnzb > mb * nb), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ENUM(4, block_dir);
    ROCSPARSE_CHECKARG_SIZE(5, block_dim);
    ROCSPARSE_CHECKARG(5, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(6, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ENUM(9, row_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(10, col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(11, idx_base);
    ROCSPARSE_CHECKARG_ENUM(12, data_type);

    auto spmat_descr = new _rocsparse_spmat_descr;
    define_bsr(spmat_descr,
               mb,
               nb,
               nnzb,
               block_dir,
               block_dim,
               bsr_row_ptr,
               bsr_row_ptr,
               bsr_col_ind,
               bsr_col_ind,
               bsr_val,
               bsr_val,
               row_ptr_type,
               col_ind_type,
               idx_base,
               data_type,
               nullptr,
               nullptr);

    *descr = spmat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_destroy_spmat_descr destroys a sparse matrix descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_const_spmat_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);

    // Check if descriptor has been initialized
    if(descr->get_init() == false)
    {
        // Do nothing
        return rocsparse_status_success;
    }

    //    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr->get_descr() ));
    //    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(descr->get_info() ));

    delete descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_get returns the sparse COO matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_coo_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      coo_row_ind,
                                   void**                      coo_col_ind,
                                   void**                      coo_val,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, coo_row_ind);
    ROCSPARSE_CHECKARG_POINTER(5, coo_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, coo_val);
    ROCSPARSE_CHECKARG_POINTER(7, idx_type);
    ROCSPARSE_CHECKARG_POINTER(8, idx_base);
    ROCSPARSE_CHECKARG_POINTER(9, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();
    *nnz  = descr->get_nnz();

    *coo_row_ind = descr->get_row_data();
    *coo_col_ind = descr->get_col_data();
    *coo_val     = descr->get_val_data();

    *idx_type  = descr->get_row_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_coo_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                coo_row_ind,
                                         const void**                coo_col_ind,
                                         const void**                coo_val,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, coo_row_ind);
    ROCSPARSE_CHECKARG_POINTER(5, coo_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, coo_val);
    ROCSPARSE_CHECKARG_POINTER(7, idx_type);
    ROCSPARSE_CHECKARG_POINTER(8, idx_base);
    ROCSPARSE_CHECKARG_POINTER(9, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();
    *nnz  = descr->get_nnz();

    *coo_row_ind = descr->get_const_row_data();
    *coo_col_ind = descr->get_const_col_data();
    *coo_val     = descr->get_const_val_data();

    *idx_type  = descr->get_row_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_aos_get returns the sparse COO (AoS) matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_coo_aos_get(const rocsparse_spmat_descr descr,
                                       int64_t*                    rows,
                                       int64_t*                    cols,
                                       int64_t*                    nnz,
                                       void**                      coo_ind,
                                       void**                      coo_val,
                                       rocsparse_indextype*        idx_type,
                                       rocsparse_index_base*       idx_base,
                                       rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, coo_ind);
    ROCSPARSE_CHECKARG_POINTER(5, coo_val);
    ROCSPARSE_CHECKARG_POINTER(6, idx_type);
    ROCSPARSE_CHECKARG_POINTER(7, idx_base);
    ROCSPARSE_CHECKARG_POINTER(8, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();
    *nnz  = descr->get_nnz();

    *coo_ind = descr->get_ind_data();
    *coo_val = descr->get_val_data();

    *idx_type  = descr->get_row_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_coo_aos_get(rocsparse_const_spmat_descr descr,
                                             int64_t*                    rows,
                                             int64_t*                    cols,
                                             int64_t*                    nnz,
                                             const void**                coo_ind,
                                             const void**                coo_val,
                                             rocsparse_indextype*        idx_type,
                                             rocsparse_index_base*       idx_base,
                                             rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, coo_ind);
    ROCSPARSE_CHECKARG_POINTER(5, coo_val);
    ROCSPARSE_CHECKARG_POINTER(6, idx_type);
    ROCSPARSE_CHECKARG_POINTER(7, idx_base);
    ROCSPARSE_CHECKARG_POINTER(8, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();
    *nnz  = descr->get_nnz();

    *coo_ind = descr->get_const_ind_data();
    *coo_val = descr->get_const_val_data();

    *idx_type  = descr->get_row_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csr_get returns the sparse CSR matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_csr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csr_row_ptr,
                                   void**                      csr_col_ind,
                                   void**                      csr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, csr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(5, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, csr_val);
    ROCSPARSE_CHECKARG_POINTER(7, row_ptr_type);
    ROCSPARSE_CHECKARG_POINTER(8, col_ind_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();
    *nnz  = descr->get_nnz();

    *csr_row_ptr = descr->get_row_data();
    *csr_col_ind = descr->get_col_data();
    *csr_val     = descr->get_val_data();

    *row_ptr_type = descr->get_row_type();
    *col_ind_type = descr->get_col_type();
    *idx_base     = descr->get_idx_base();
    *data_type    = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_csr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csr_row_ptr,
                                         const void**                csr_col_ind,
                                         const void**                csr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, csr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(5, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, csr_val);
    ROCSPARSE_CHECKARG_POINTER(7, row_ptr_type);
    ROCSPARSE_CHECKARG_POINTER(8, col_ind_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();
    *nnz  = descr->get_nnz();

    *csr_row_ptr = descr->get_const_row_data();
    *csr_col_ind = descr->get_const_col_data();
    *csr_val     = descr->get_const_val_data();

    *row_ptr_type = descr->get_row_type();
    *col_ind_type = descr->get_col_type();
    *idx_base     = descr->get_idx_base();
    *data_type    = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_bsr_get returns the sparse BSR matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_const_bsr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    brows,
                                         int64_t*                    bcols,
                                         int64_t*                    bnnz,
                                         rocsparse_direction*        bdir,
                                         int64_t*                    bdim,
                                         const void**                bsr_row_ptr,
                                         const void**                bsr_col_ind,
                                         const void**                bsr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(brows == nullptr || bcols == nullptr || bnnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(bsr_row_ptr == nullptr || bsr_col_ind == nullptr || bsr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(row_ptr_type == nullptr || col_ind_type == nullptr || idx_base == nullptr
       || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->get_init() == false)
    {
        return rocsparse_status_not_initialized;
    }

    *brows = descr->get_rows();
    *bcols = descr->get_cols();
    *bnnz  = descr->get_nnz();

    *bsr_row_ptr = descr->get_const_row_data();
    *bsr_col_ind = descr->get_const_col_data();
    *bsr_val     = descr->get_const_val_data();

    *row_ptr_type = descr->get_row_type();
    *col_ind_type = descr->get_col_type();
    *idx_base     = descr->get_idx_base();
    *data_type    = descr->get_data_type();
    *bdim         = descr->get_block_dim();
    *bdir         = descr->get_block_dir();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_bsr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    brows,
                                   int64_t*                    bcols,
                                   int64_t*                    bnnz,
                                   rocsparse_direction*        bdir,
                                   int64_t*                    bdim,
                                   void**                      bsr_row_ptr,
                                   void**                      bsr_col_ind,
                                   void**                      bsr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(brows == nullptr || bcols == nullptr || bnnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(bsr_row_ptr == nullptr || bsr_col_ind == nullptr || bsr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(row_ptr_type == nullptr || col_ind_type == nullptr || idx_base == nullptr
       || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->get_init() == false)
    {
        return rocsparse_status_not_initialized;
    }

    *brows = descr->get_rows();
    *bcols = descr->get_cols();
    *bnnz  = descr->get_nnz();

    *bsr_row_ptr = descr->get_row_data();
    *bsr_col_ind = descr->get_col_data();
    *bsr_val     = descr->get_val_data();

    *row_ptr_type = descr->get_row_type();
    *col_ind_type = descr->get_col_type();
    *idx_base     = descr->get_idx_base();
    *data_type    = descr->get_data_type();
    *bdim         = descr->get_block_dim();
    *bdir         = descr->get_block_dir();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csc_get returns the sparse CSC matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_csc_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csc_col_ptr,
                                   void**                      csc_row_ind,
                                   void**                      csc_val,
                                   rocsparse_indextype*        col_ptr_type,
                                   rocsparse_indextype*        row_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, csc_col_ptr);
    ROCSPARSE_CHECKARG_POINTER(5, csc_row_ind);
    ROCSPARSE_CHECKARG_POINTER(6, csc_val);
    ROCSPARSE_CHECKARG_POINTER(7, col_ptr_type);
    ROCSPARSE_CHECKARG_POINTER(8, row_ind_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();
    *nnz  = descr->get_nnz();

    *csc_col_ptr = descr->get_col_data();
    *csc_row_ind = descr->get_row_data();
    *csc_val     = descr->get_val_data();

    *col_ptr_type = descr->get_col_type();
    *row_ind_type = descr->get_row_type();
    *idx_base     = descr->get_idx_base();
    *data_type    = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_csc_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csc_col_ptr,
                                         const void**                csc_row_ind,
                                         const void**                csc_val,
                                         rocsparse_indextype*        col_ptr_type,
                                         rocsparse_indextype*        row_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, csc_col_ptr);
    ROCSPARSE_CHECKARG_POINTER(5, csc_row_ind);
    ROCSPARSE_CHECKARG_POINTER(6, csc_val);
    ROCSPARSE_CHECKARG_POINTER(7, col_ptr_type);
    ROCSPARSE_CHECKARG_POINTER(8, row_ind_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();
    *nnz  = descr->get_nnz();

    *csc_col_ptr = descr->get_const_col_data();
    *csc_row_ind = descr->get_const_row_data();
    *csc_val     = descr->get_const_val_data();

    *row_ind_type = descr->get_row_type();
    *col_ptr_type = descr->get_col_type();
    *idx_base     = descr->get_idx_base();
    *data_type    = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_ell_get returns the sparse ELL matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_ell_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   void**                      ell_col_ind,
                                   void**                      ell_val,
                                   int64_t*                    ell_width,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(4, ell_val);
    ROCSPARSE_CHECKARG_POINTER(5, ell_width);
    ROCSPARSE_CHECKARG_POINTER(6, idx_type);
    ROCSPARSE_CHECKARG_POINTER(7, idx_base);
    ROCSPARSE_CHECKARG_POINTER(8, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();

    *ell_col_ind = descr->get_col_data();
    *ell_val     = descr->get_val_data();
    *ell_width   = descr->get_ell_width();

    *idx_type  = descr->get_row_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_ell_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         const void**                ell_col_ind,
                                         const void**                ell_val,
                                         int64_t*                    ell_width,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(4, ell_val);
    ROCSPARSE_CHECKARG_POINTER(5, ell_width);
    ROCSPARSE_CHECKARG_POINTER(6, idx_type);
    ROCSPARSE_CHECKARG_POINTER(7, idx_base);
    ROCSPARSE_CHECKARG_POINTER(8, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();

    *ell_col_ind = descr->get_const_col_data();
    *ell_val     = descr->get_const_val_data();
    *ell_width   = descr->get_ell_width();

    *idx_type  = descr->get_row_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_bell_get returns the sparse BLOCKED ELL matrix data,
 * sizes and properties.
 *******************************************************************************/
rocsparse_status rocsparse_bell_get(const rocsparse_spmat_descr descr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    rocsparse_direction*        ell_block_dir,
                                    int64_t*                    ell_block_dim,
                                    int64_t*                    ell_cols,
                                    void**                      ell_col_ind,
                                    void**                      ell_val,
                                    rocsparse_indextype*        idx_type,
                                    rocsparse_index_base*       idx_base,
                                    rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ell_block_dir);
    ROCSPARSE_CHECKARG_POINTER(4, ell_block_dim);
    ROCSPARSE_CHECKARG_POINTER(5, ell_cols);
    ROCSPARSE_CHECKARG_POINTER(6, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(7, ell_val);
    ROCSPARSE_CHECKARG_POINTER(8, idx_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();

    *ell_col_ind   = descr->get_col_data();
    *ell_val       = descr->get_val_data();
    *ell_cols      = descr->get_ell_cols();
    *ell_block_dir = descr->get_block_dir();
    *ell_block_dim = descr->get_block_dim();

    *idx_type  = descr->get_row_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_bell_get(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          rocsparse_direction*        ell_block_dir,
                                          int64_t*                    ell_block_dim,
                                          int64_t*                    ell_cols,
                                          const void**                ell_col_ind,
                                          const void**                ell_val,
                                          rocsparse_indextype*        idx_type,
                                          rocsparse_index_base*       idx_base,
                                          rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ell_block_dir);
    ROCSPARSE_CHECKARG_POINTER(4, ell_block_dim);
    ROCSPARSE_CHECKARG_POINTER(5, ell_cols);
    ROCSPARSE_CHECKARG_POINTER(6, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(7, ell_val);
    ROCSPARSE_CHECKARG_POINTER(8, idx_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->get_rows();
    *cols = descr->get_cols();

    *ell_col_ind   = descr->get_const_col_data();
    *ell_val       = descr->get_const_val_data();
    *ell_cols      = descr->get_ell_cols();
    *ell_block_dir = descr->get_block_dir();
    *ell_block_dim = descr->get_block_dim();

    *idx_type  = descr->get_row_type();
    *idx_base  = descr->get_idx_base();
    *data_type = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_sell_get returns the sparse SLICED ELL matrix data,
 * sizes and properties.
 *******************************************************************************/
rocsparse_status rocsparse_sell_get(const rocsparse_spmat_descr descr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    int64_t*                    nnz,
                                    int64_t*                    sell_slice_size,
                                    int64_t*                    sell_colval_size,
                                    void**                      sell_slice_offsets,
                                    void**                      sell_col_ind,
                                    void**                      sell_val,
                                    rocsparse_indextype*        sell_slice_offsets_type,
                                    rocsparse_indextype*        sell_col_ind_type,
                                    rocsparse_index_base*       idx_base,
                                    rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, sell_slice_size);
    ROCSPARSE_CHECKARG_POINTER(5, sell_colval_size);
    ROCSPARSE_CHECKARG_POINTER(6, sell_slice_offsets);
    ROCSPARSE_CHECKARG_POINTER(7, sell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(8, sell_val);
    ROCSPARSE_CHECKARG_POINTER(9, sell_slice_offsets_type);
    ROCSPARSE_CHECKARG_POINTER(10, sell_col_ind_type);
    ROCSPARSE_CHECKARG_POINTER(11, idx_base);
    ROCSPARSE_CHECKARG_POINTER(12, data_type);

    *rows             = descr->get_rows();
    *cols             = descr->get_cols();
    *nnz              = descr->get_nnz();
    *sell_slice_size  = descr->get_sell_slice_size();
    *sell_colval_size = descr->get_sell_colval_size();

    *sell_slice_offsets = descr->get_row_data();
    *sell_col_ind       = descr->get_col_data();
    *sell_val           = descr->get_val_data();

    *sell_slice_offsets_type = descr->get_row_type();
    *sell_col_ind_type       = descr->get_col_type();
    *idx_base                = descr->get_idx_base();
    *data_type               = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_sell_get(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          int64_t*                    nnz,
                                          int64_t*                    sell_slice_size,
                                          int64_t*                    sell_colval_size,
                                          const void**                sell_slice_offsets,
                                          const void**                sell_col_ind,
                                          const void**                sell_val,
                                          rocsparse_indextype*        sell_slice_offsets_type,
                                          rocsparse_indextype*        sell_col_ind_type,
                                          rocsparse_index_base*       idx_base,
                                          rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, sell_slice_size);
    ROCSPARSE_CHECKARG_POINTER(5, sell_colval_size);
    ROCSPARSE_CHECKARG_POINTER(6, sell_slice_offsets);
    ROCSPARSE_CHECKARG_POINTER(7, sell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(8, sell_val);
    ROCSPARSE_CHECKARG_POINTER(9, sell_slice_offsets_type);
    ROCSPARSE_CHECKARG_POINTER(10, sell_col_ind_type);
    ROCSPARSE_CHECKARG_POINTER(11, idx_base);
    ROCSPARSE_CHECKARG_POINTER(12, data_type);

    *rows             = descr->get_rows();
    *cols             = descr->get_cols();
    *nnz              = descr->get_nnz();
    *sell_slice_size  = descr->get_sell_slice_size();
    *sell_colval_size = descr->get_sell_colval_size();

    *sell_slice_offsets = descr->get_const_row_data();
    *sell_col_ind       = descr->get_const_col_data();
    *sell_val           = descr->get_const_val_data();

    *sell_slice_offsets_type = descr->get_row_type();
    *sell_col_ind_type       = descr->get_col_type();
    *idx_base                = descr->get_idx_base();
    *data_type               = descr->get_data_type();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_set_pointers sets the sparse COO matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_coo_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 coo_row_ind,
                                            void*                 coo_col_ind,
                                            void*                 coo_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, coo_row_ind);
    ROCSPARSE_CHECKARG_POINTER(2, coo_col_ind);
    ROCSPARSE_CHECKARG_POINTER(3, coo_val);

    descr->set_row_data(coo_row_ind);
    descr->set_col_data(coo_col_ind);
    descr->set_val_data(coo_val);

    descr->set_const_row_data(coo_row_ind);
    descr->set_const_col_data(coo_col_ind);
    descr->set_const_val_data(coo_val);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_aos_set_pointers sets the sparse COO (AoS) matrix data pointers.
 *******************************************************************************/
rocsparse_status
    rocsparse_coo_aos_set_pointers(rocsparse_spmat_descr descr, void* coo_ind, void* coo_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, coo_ind);
    ROCSPARSE_CHECKARG_POINTER(2, coo_val);

    descr->set_ind_data(coo_ind);
    descr->set_val_data(coo_val);

    descr->set_const_ind_data(coo_ind);
    descr->set_const_val_data(coo_val);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csr_set_pointers sets the sparse CSR matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_csr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csr_row_ptr,
                                            void*                 csr_col_ind,
                                            void*                 csr_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(1, csr_row_ptr);
    ROCSPARSE_CHECKARG(2,
                       csr_col_ind,
                       descr->get_nnz() > 0 && csr_col_ind == nullptr,
                       rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(
        3, csr_val, descr->get_nnz() > 0 && csr_val == nullptr, rocsparse_status_invalid_pointer);

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->set_analysed(false);

    descr->set_row_data(csr_row_ptr);
    descr->set_col_data(csr_col_ind);
    descr->set_val_data(csr_val);

    descr->set_const_row_data(csr_row_ptr);
    descr->set_const_col_data(csr_col_ind);
    descr->set_const_val_data(csr_val);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csc_set_pointers sets the sparse CSR matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_csc_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csc_col_ptr,
                                            void*                 csc_row_ind,
                                            void*                 csc_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(1, csc_col_ptr);
    ROCSPARSE_CHECKARG(2,
                       csc_row_ind,
                       descr->get_nnz() > 0 && csc_row_ind == nullptr,
                       rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(
        3, csc_val, descr->get_nnz() > 0 && csc_val == nullptr, rocsparse_status_invalid_pointer);

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->set_analysed(false);

    descr->set_row_data(csc_row_ind);
    descr->set_col_data(csc_col_ptr);
    descr->set_val_data(csc_val);

    descr->set_const_row_data(csc_row_ind);
    descr->set_const_col_data(csc_col_ptr);
    descr->set_const_val_data(csc_val);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_ell_set_pointers sets the sparse ELL matrix data pointers.
 *******************************************************************************/
rocsparse_status
    rocsparse_ell_set_pointers(rocsparse_spmat_descr descr, void* ell_col_ind, void* ell_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(2, ell_val);

    descr->set_col_data(ell_col_ind);
    descr->set_val_data(ell_val);

    descr->set_const_col_data(ell_col_ind);
    descr->set_const_val_data(ell_val);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_bsr_set_pointers sets the sparse BSR matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_bsr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 bsr_row_ptr,
                                            void*                 bsr_col_ind,
                                            void*                 bsr_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(1, bsr_row_ptr);
    ROCSPARSE_CHECKARG(2,
                       bsr_col_ind,
                       descr->get_nnz() > 0 && bsr_col_ind == nullptr,
                       rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(
        3, bsr_val, descr->get_nnz() > 0 && bsr_val == nullptr, rocsparse_status_invalid_pointer);

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->set_analysed(false);

    descr->set_row_data(bsr_row_ptr);
    descr->set_col_data(bsr_col_ind);
    descr->set_val_data(bsr_val);

    descr->set_const_row_data(bsr_row_ptr);
    descr->set_const_col_data(bsr_col_ind);
    descr->set_const_val_data(bsr_val);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_size returns the sparse matrix sizes.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_size(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          int64_t*                    nnz)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);

    *rows = descr->get_rows();
    *cols = descr->get_cols();
    *nnz  = descr->get_nnz();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_format returns the sparse matrix format.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_format(rocsparse_const_spmat_descr descr,
                                            rocsparse_format*           format)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, format);

    *format = descr->get_format();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_index_base returns the sparse matrix index base.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_index_base(rocsparse_const_spmat_descr descr,
                                                rocsparse_index_base*       idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, idx_base);

    *idx_base = descr->get_idx_base();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_values returns the sparse matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_values(rocsparse_spmat_descr descr, void** values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    *values = descr->get_val_data();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_spmat_get_values(rocsparse_const_spmat_descr descr,
                                                  const void**                values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    *values = descr->get_const_val_data();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_set_values sets the sparse matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_set_values(rocsparse_spmat_descr descr, void* values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    descr->set_val_data(values);
    descr->set_const_val_data(values);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_strided_batch gets the sparse matrix batch count.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_strided_batch(rocsparse_const_spmat_descr descr,
                                                   rocsparse_int*              batch_count)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, batch_count);

    *batch_count = descr->get_batch_count();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/****************************************************************************
 * \brief rocsparse_spmat_get_nnz gets the sparse matrix number of non-zeros.
 ****************************************************************************/
rocsparse_status rocsparse_spmat_get_nnz(rocsparse_const_spmat_descr descr, int64_t* nnz)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, nnz);

    switch(descr->get_format())
    {
    case rocsparse_format_bell:
    {
        nnz[0] = descr->get_ell_cols() * descr->get_rows() * descr->get_block_dim()
                 * descr->get_block_dim();
        break;
    }

    case rocsparse_format_ell:
    {
        nnz[0] = descr->get_ell_width() * descr->get_rows();
        break;
    }

    case rocsparse_format_bsr:
    {
        nnz[0] = descr->get_nnz() * descr->get_block_dim() * descr->get_block_dim();
        break;
    }

    case rocsparse_format_csc:
    case rocsparse_format_csr:
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    case rocsparse_format_sell:
    {
        nnz[0] = descr->get_nnz();
        break;
    }
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/****************************************************************************
 * \brief rocsparse_spmat_get_nnz sets the sparse matrix number of non-zeros.
 ****************************************************************************/
rocsparse_status rocsparse_spmat_set_nnz(rocsparse_spmat_descr descr, int64_t nnz)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_SIZE(1, nnz);

    switch(descr->get_format())
    {
    case rocsparse_format_bell:
    {
        // LCOV_EXCL_START
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "Cannot set the number of non-zeros of a Block ELL sparse matrix.");
        // LCOV_EXCL_STOP
        break;
    }

    case rocsparse_format_ell:
    {
        // LCOV_EXCL_START
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "Cannot set the number of non-zeros of an ELL sparse matrix.");
        // LCOV_EXCL_STOP
        break;
    }

    case rocsparse_format_bsr:
    {
        descr->set_nnz(nnz);
        break;
    }
    case rocsparse_format_csc:
    {
        descr->set_nnz(nnz);
        break;
    }
    case rocsparse_format_csr:
    {
        descr->set_nnz(nnz);
        break;
    }
    case rocsparse_format_coo:
    {
        descr->set_nnz(nnz);
        break;
    }
    case rocsparse_format_coo_aos:
    {
        descr->set_nnz(nnz);
        break;
    }
    case rocsparse_format_sell:
    {
        descr->set_nnz(nnz);
        break;
    }
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_set_strided_batch sets the sparse matrix batch count.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_set_strided_batch(rocsparse_spmat_descr descr,
                                                   rocsparse_int         batch_count)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);

    descr->set_batch_count(batch_count);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_set_strided_batch sets the COO sparse matrix batch count
 * and batch stride.
 *******************************************************************************/
rocsparse_status rocsparse_coo_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               batch_stride)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(2, batch_stride, (batch_stride < 0), rocsparse_status_invalid_value);

    descr->set_batch_count(batch_count);
    descr->set_batch_stride(batch_stride);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csr_set_strided_batch sets the CSR sparse matrix batch count
 * and batch stride.
 *******************************************************************************/
rocsparse_status rocsparse_csr_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               columns_values_batch_stride)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(
        2, offsets_batch_stride, (offsets_batch_stride < 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(3,
                       columns_values_batch_stride,
                       (columns_values_batch_stride < 0),
                       rocsparse_status_invalid_value);

    descr->set_batch_count(batch_count);
    descr->set_offsets_batch_stride(offsets_batch_stride);
    descr->set_columns_values_batch_stride(columns_values_batch_stride);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csc_set_strided_batch sets the CSC sparse matrix batch count
 * and batch stride.
 *******************************************************************************/
rocsparse_status rocsparse_csc_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               rows_values_batch_stride)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(
        2, offsets_batch_stride, (offsets_batch_stride < 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(3,
                       rows_values_batch_stride,
                       (rows_values_batch_stride < 0),
                       rocsparse_status_invalid_value);

    descr->set_batch_count(batch_count);
    descr->set_offsets_batch_stride(offsets_batch_stride);
    descr->set_columns_values_batch_stride(rows_values_batch_stride);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_attribute gets the sparse matrix attribute.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_attribute(rocsparse_const_spmat_descr descr,
                                               rocsparse_spmat_attribute   attribute,
                                               void*                       data,
                                               size_t                      data_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_ENUM(1, attribute);
    ROCSPARSE_CHECKARG_POINTER(2, data);
    switch(attribute)
    {
    case rocsparse_spmat_fill_mode:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_fill_mode),
                           rocsparse_status_invalid_size);
        rocsparse_fill_mode* uplo = reinterpret_cast<rocsparse_fill_mode*>(data);
        *uplo                     = rocsparse_get_mat_fill_mode(descr->get_descr());
        return rocsparse_status_success;
    }
    case rocsparse_spmat_diag_type:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_diag_type),
                           rocsparse_status_invalid_size);
        rocsparse_diag_type* uplo = reinterpret_cast<rocsparse_diag_type*>(data);
        *uplo                     = rocsparse_get_mat_diag_type(descr->get_descr());
        return rocsparse_status_success;
    }
    case rocsparse_spmat_matrix_type:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_matrix_type),
                           rocsparse_status_invalid_size);
        rocsparse_matrix_type* matrix = reinterpret_cast<rocsparse_matrix_type*>(data);
        *matrix                       = rocsparse_get_mat_type(descr->get_descr());
        return rocsparse_status_success;
    }
    case rocsparse_spmat_storage_mode:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_storage_mode),
                           rocsparse_status_invalid_size);
        rocsparse_storage_mode* storage = reinterpret_cast<rocsparse_storage_mode*>(data);
        *storage                        = rocsparse_get_mat_storage_mode(descr->get_descr());
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    return rocsparse_status_invalid_value;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_set_attribute sets the sparse matrix attribute.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_set_attribute(rocsparse_spmat_descr     descr,
                                               rocsparse_spmat_attribute attribute,
                                               const void*               data,
                                               size_t                    data_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_ENUM(1, attribute);
    ROCSPARSE_CHECKARG_POINTER(2, data);

    switch(attribute)
    {
    case rocsparse_spmat_fill_mode:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_fill_mode),
                           rocsparse_status_invalid_size);
        rocsparse_fill_mode uplo = *reinterpret_cast<const rocsparse_fill_mode*>(data);
        return rocsparse_set_mat_fill_mode(descr->get_descr(), uplo);
    }
    case rocsparse_spmat_diag_type:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_diag_type),
                           rocsparse_status_invalid_size);
        rocsparse_diag_type diag = *reinterpret_cast<const rocsparse_diag_type*>(data);
        return rocsparse_set_mat_diag_type(descr->get_descr(), diag);
    }

    case rocsparse_spmat_matrix_type:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_matrix_type),
                           rocsparse_status_invalid_size);
        rocsparse_matrix_type matrix = *reinterpret_cast<const rocsparse_matrix_type*>(data);
        return rocsparse_set_mat_type(descr->get_descr(), matrix);
    }
    case rocsparse_spmat_storage_mode:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_storage_mode),
                           rocsparse_status_invalid_size);
        rocsparse_storage_mode storage = *reinterpret_cast<const rocsparse_storage_mode*>(data);
        return rocsparse_set_mat_storage_mode(descr->get_descr(), storage);
    }
    }
    // LCOV_EXCL_START
    return rocsparse_status_invalid_value;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#ifdef __cplusplus
}
#endif
