/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_EXTRACT_H
#define ROCSPARSE_EXTRACT_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
   *  \details
   *  \p rocsparse_extract_buffer_size calculates the required buffer size in bytes for a given stage \p stage.
   *  This routine is used in conjunction with \ref rocsparse_extract_nnz and \ref rocsparse_extract to extract
   *  a lower or upper triangular sparse matrix from an input sparse matrix. See \ref rocsparse_extract for more
   *  details.
   *
   *  \note
   *  This routine is asynchronous with respect to the host.
   *  This routine does support execution in a hipGraph context.
   *
   *  @param[in]
   *  handle       handle to the rocsparse library context queue.
   *  @param[in]
   *  descr        descriptor of the extract algorithm.
   *  @param[in]
   *  source       source sparse matrix descriptor.
   *  @param[in]
   *  target       target sparse matrix descriptor.
   *  @param[in]
   *  stage        stage of the extract computation.
   *  @param[out]
   *  buffer_size_in_bytes  size in bytes of the buffer.
   *
   *  \retval      rocsparse_status_success the operation completed successfully.
   *  \retval      rocsparse_status_invalid_handle the library context was not initialized.
   *  \retval      rocsparse_status_invalid_value if \p stage is invalid.
   *  \retval      rocsparse_status_invalid_pointer \p descr, \p source, \p target, or \p buffer_size_in_bytes
   *               pointer is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_extract_buffer_size(rocsparse_handle            handle,
                                               rocsparse_extract_descr     descr,
                                               rocsparse_const_spmat_descr source,
                                               rocsparse_spmat_descr       target,
                                               rocsparse_extract_stage     stage,
                                               size_t*                     buffer_size_in_bytes);

/*! \ingroup generic_module
   *  \details
   *  \p rocsparse_extract_nnz returns the number of non-zeros in the extracted matrix. The value is
   *  available after the analysis phase \ref rocsparse_extract_stage_analysis has been executed. This routine
   *  is used in conjunction with \ref rocsparse_extract_buffer_size and \ref rocsparse_extract to extract a lower
   *  or upper triangular sparse matrix from an input sparse matrix. See \ref rocsparse_extract for more
   *  details.
   *
   *  \note
   *  This routine is asynchronous with respect to the host.
   *  This routine does support execution in a hipGraph context.
   *
   *  @param[in]
   *  handle       handle to the rocsparse library context queue.
   *  @param[in]
   *  descr        descriptor of the extract algorithm.
   *  @param[out]
   *  nnz          the number of non-zeros.
   *
   *  \retval      rocsparse_status_success the operation completed successfully.
   *  \retval      rocsparse_status_invalid_handle the library context was not initialized.
   *  \retval      rocsparse_status_invalid_pointer \p descr or \p nnz pointer is invalid.
   */
ROCSPARSE_EXPORT
rocsparse_status
    rocsparse_extract_nnz(rocsparse_handle handle, rocsparse_extract_descr descr, int64_t* nnz);

/*! \ingroup generic_module
   *  \brief Sparse matrix extraction.
   *
   *  \details
   *  \p rocsparse_extract performs the extraction of the lower or upper part of a sparse matrix into a new matrix.
   *
   *  \p rocsparse_extract requires multiple steps to complete. First, the user creates the source and target sparse matrix
   *  descriptors. For example, in the case of CSR matrix format this might look like:
   *  \code{.c}
   *   // Build Source
   *   rocsparse_spmat_descr source;
   *   rocsparse_create_csr_descr(&source,
   *                              M,
   *                              N,
   *                              nnz,
   *                              dsource_row_ptr,
   *                              dsource_col_ind,
   *                              dsource_val,
   *                              rocsparse_indextype_i32,
   *                              rocsparse_indextype_i32,
   *                              rocsparse_index_base_zero,
   *                              rocsparse_datatype_f32_r);
   *
   *   // Build target
   *   void * dtarget_row_ptr;
   *   hipMalloc(&dtarget_row_ptr, sizeof(int32_t) * (M + 1));
   *   rocsparse_spmat_descr target;
   *   rocsparse_create_csr_descr(&target,
   *                              M,
   *                              N,
   *                              0,
   *                              dtarget_row_ptr,
   *                              nullptr,
   *                              nullptr,
   *                              rocsparse_indextype_i32,
   *                              rocsparse_indextype_i32,
   *                              rocsparse_index_base_zero,
   *                              rocsparse_datatype_f32_r);
   *  \endcode
   *  Next, the user creates the extraction descriptor and calls \ref rocsparse_extract_buffer_size with the stage
   *  \ref rocsparse_extract_stage_analysis in order to determine the amount of temporary storage required.
   *  The user allocates this temporary storage buffer and passes it to \p rocsparse_extract with the stage
   *  \ref rocsparse_extract_stage_analysis
   *  \code{.c}
   *  // Create descriptor
   *  rocsparse_extract_descr descr;
   *  rocsparse_create_extract_descr(&descr,
   *                                 source,
   *                                 target,
   *                                 rocsparse_extract_alg_default);
   *
   *  // Analysis phase
   *  size_t buffer_size;
   *  rocsparse_extract_buffer_size(handle,
   *                                descr,
   *                                source,
   *                                target,
   *                                rocsparse_extract_stage_analysis,
   *                                &buffer_size);
   *  void* dbuffer = nullptr;
   *  hipMalloc(&dbuffer, buffer_size);
   *  rocsparse_extract(handle,
   *                    descr,
   *                    source,
   *                    target,
   *                    rocsparse_extract_stage_analysis,
   *                    buffer_size,
   *                    dbuffer);
   *  hipFree(dbuffer);
   *  \endcode
   *  The user then calls \ref rocsparse_extract_nnz in order to determine the number of non-zeros that will exist in the
   *  target matrix. Once determined, the user can allocate the column indices and values arrays of the target sparse
   *  matrix:
   *  \code{.c}
   *  int64_t target_nnz;
   *  rocsparse_extract_nnz(handle, descr, &target_nnz);
   *
   *  void* dtarget_col_ind,
   *  void* dtarget_val;
   *  hipMalloc(&dtarget_col_ind, sizeof(int32_t) * target_nnz);
   *  hipMalloc(&dtarget_val, sizeof(float) * target_nnz);
   *  rocsparse_csr_set_pointers(target, dtarget_row_ptr, dtarget_col_ind, dtarget_val);
   *  \endcode
   *  Finally, the user calls \ref rocsparse_extract_buffer_size with the stage \ref rocsparse_extract_stage_compute in order
   *  to determine the size of the temporary user allocated storage needed for the computation of the column indices and values
   *  in the sparse target. The user allocates this buffer and completes the conversion by calling \p rocsparse_extract using
   *  the \ref rocsparse_extract_stage_compute stage:
   *  \code{.c}
   *  // Calculation phase
   *  rocsparse_extract_buffer_size(handle,
   *                                descr,
   *                                source,
   *                                target,
   *                                rocsparse_extract_stage_compute,
   *                                &buffer_size);
   *  hipMalloc(&dbuffer, buffer_size);
   *  rocsparse_extract(handle,
   *                    descr,
   *                    source,
   *                    target,
   *                    rocsparse_extract_stage_compute,
   *                    buffer_size,
   *                    dbuffer);
   *  hipFree(dbuffer);
   *  \endcode
   *  The target row pointer, column indices, and values arrays will now be filled with the upper or lower part of the source matrix.
   *
   *  The source and the target matrices must have the same format (see \ref rocsparse_format) and the same storage mode (see
   *  \ref rocsparse_storage_mode). The attributes of the target matrix, the fill mode \ref rocsparse_fill_mode and the diagonal
   *  type \ref rocsparse_diag_type are used to parametrise the algorithm. These can be set on the target matrix using
   *  \ref rocsparse_spmat_set_attribute. See full example below.
   *
   *  \note
   *  This routine is asynchronous with respect to the host.
   *  This routine does support execution in a hipGraph context.
   *  \note
   *  Supported formats are \ref rocsparse_format_csr and  \ref rocsparse_format_csc.
   *
   *  @param[in]
   *  handle       handle to the rocsparse library context queue.
   *  @param[in]
   *  descr        descriptor of the extract algorithm.
   *  @param[in]
   *  source       sparse matrix descriptor.
   *  @param[in]
   *  target       sparse matrix descriptor.
   *  @param[in]
   *  stage        stage of the extract computation.
   *  @param[in]
   *  buffer_size_in_bytes  size in bytes of the \p buffer
   *  @param[in]
   *  buffer  temporary storage buffer allocated by the user.
   *
   *  \retval      rocsparse_status_success the operation completed successfully.
   *  \retval      rocsparse_status_invalid_handle the library context was not initialized.
   *  \retval      rocsparse_status_invalid_value if \p stage is invalid.
   *  \retval      rocsparse_status_invalid_pointer \p descr, \p source, \p target, or \p buffer
   *               pointer is invalid.
   *  \par Example
   *  This example extracts the lower part of CSR matrix into a CSR matrix.
   *  \snippet example_rocsparse_extract.cpp doc example
   */
ROCSPARSE_EXPORT
rocsparse_status rocsparse_extract(rocsparse_handle            handle,
                                   rocsparse_extract_descr     descr,
                                   rocsparse_const_spmat_descr source,
                                   rocsparse_spmat_descr       target,
                                   rocsparse_extract_stage     stage,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       buffer);

#ifdef __cplusplus
}
#endif

#endif
