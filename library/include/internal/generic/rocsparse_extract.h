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
   *  \brief Sparse matrix extraction.
   *
   *  \details
   *  \p rocsparse_extract_buffer_size calculates the required buffer size in bytes for a given stage \p stage.
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
   *  \brief Sparse matrix extraction.
   *
   *  \details
   *  \p rocsparse_extract_nnz returns the number of non-zeros of the extracted matrix. The value is available after the analysis phase \ref rocsparse_extract_stage_analysis being executed.
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
   *  \p rocsparse_extract performs the extraction of the lower or upper part of a sparse matrix.
   *
   *  \note
   *  Supported formats are \ref rocsparse_format_csr and  \ref rocsparse_format_csc.
   *
   *  The source and the target matrices must have the same format \ref rocsparse_format.
   *  The source and the target matrices must have the same storage mode \ref rocsparse_storage_mode.
   *  The attributes of the target matrix, the fill mode \ref rocsparse_fill_mode and the diagonal type \ref rocsparse_diag_type are used
   *  to parametrise the algorithm.
   *
   *  The required allocation size (in bytes) to \p buffer_size_in_bytes must be obtained from \ref rocsparse_extract_buffer_size
   *  for each stage, since the required buffer size can be different between stages.
   *
   *  The value of the number of non-zeros is available after the analysis phase \ref rocsparse_extract_stage_analysis being executed.
   *  This value can be fetched with \ref rocsparse_extract_nnz.
   *
   *  This routine is asynchronous with respect to the host.
   *  This routine does support execution in a hipGraph context.
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
   *  \code{.c}
   *
   *      // It assumes the CSR arrays (ptr, ind, val) have already been allocated and filled.
   *      // Build Source
   *      rocsparse_spmat_descr source;
   *      rocsparse_create_csr_descr(&source,
   *                                 M,
   *                                 N,
   *                                 nnz,
   *                                 ptr,
   *                                 ind,
   *                                 val,
   *                                 rocsparse_indextype_i32,
   *                                 rocsparse_indextype_i32,
   *                                 rocsparse_index_base_zero,
   *                                 rocsparse_datatype_f32_r);
   *
   *      // Build target
   *      void * target_ptr;
   *      hipMalloc(&target_ptr,sizeof(int32_t)*(M+1));
   *      rocsparse_spmat_descr target;
   *      rocsparse_create_csr_descr(&target,
   *                                 M,
   *                                 N,
   *                                 0,
   *                                 target_ptr,
   *                                 nullptr,
   *                                 nullptr,
   *                                 rocsparse_indextype_i32,
   *                                 rocsparse_indextype_i32,
   *                                 rocsparse_index_base_zero,
   *                                 rocsparse_datatype_f32_r);
   *
   *      const rocsparse_fill_mode 		fill_mode	= rocsparse_fill_mode_lower;
   *      const rocsparse_diag_type 		diag_type	= rocsparse_diag_type_non_unit;
   *
   *      rocsparse_spmat_set_attribute(target,
   *                                    rocsparse_spmat_fill_mode,
   *                                    &fill_mode,
   *                                    sizeof(fill_mode));
   *
   *      rocsparse_spmat_set_attribute(target,
   *                                    rocsparse_spmat_diag_type,
   *                                    &diag_type,
   *                                    sizeof(diag_type));
   *
   *      // Create descriptor
   *      rocsparse_extract_descr descr;
   *      rocsparse_create_extract_descr(&descr,
   *                                     source,
   *                                     target,
   *                                     rocsparse_extract_alg_default);
   *
   *      // Analysis phase
   *      rocsparse_extract_buffer_size(handle,
   *                                    descr,
   *                                    source,
   *                                    target,
   *                                    rocsparse_extract_stage_analysis,
   *                                    &buffer_size);
   *      hipMalloc(&buffer,buffer_size);
   *      rocsparse_extract(handle,
   *                        descr,
   *                        source,
   *                        target,
   *                        rocsparse_extract_stage_analysis,
   *                        buffer_size,
   *                        buffer);
   *      hipFree(buffer);
   *
   *      //
   *      // The user is responsible to allocate target arrays after the analysis phase.
   *      //
   *
   *      { int64_t target_nnz;
   *        rocsparse_extract_nnz(handle,
   *                              descr,
   *                              &target_nnz);
   *        void * target_ind, * target_val;
   *        hipMalloc(&target_ind, target_nnz * sizeof(int32_t));
   *        hipMalloc(&target_val, target_nnz* sizeof(float)));
   *        rocsparse_csr_set_pointers(target,
   *                                   target_ptr,
   *                                   target_ind,
   *                                   target_val); }
   *
   *      // Calculation phase
   *      rocsparse_extract_buffer_size(handle,
   *                                    descr,
   *                                    source,
   *                                    target,
   *                                    rocsparse_extract_stage_compute,
   *                                    &buffer_size);
   *      hipMalloc(&buffer,buffer_size);
   *      rocsparse_extract(handle,
   *                        descr,
   *                        source,
   *                        target,
   *                        rocsparse_extract_stage_compute,
   *                        buffer_size,
   *                        buffer);
   *      hipFree(buffer);
   *
   *     rocsparse_destroy_extract_descr(descr);
   *  \endcode
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
