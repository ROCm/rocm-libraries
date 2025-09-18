/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_SPARSE_TO_SPARSE_H
#define ROCSPARSE_SPARSE_TO_SPARSE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p rocsparse_sparse_to_sparse_buffer_size calculates the required buffer size in bytes for a given stage \p stage.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  descr        descriptor of the sparse_to_sparse algorithm.
*  @param[in]
*  source       source sparse matrix descriptor.
*  @param[in]
*  target       target sparse matrix descriptor.
*  @param[in]
*  stage        stage of the sparse_to_sparse computation.
*  @param[out]
*  buffer_size_in_bytes  size in bytes of the \p buffer
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_value if any required enumeration is invalid.
*  \retval      rocsparse_status_invalid_pointer \p mat_A, \p mat_B, or \p buffer_size_in_bytes
*               pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sparse_to_sparse_buffer_size(rocsparse_handle                 handle,
                                                        rocsparse_sparse_to_sparse_descr descr,
                                                        rocsparse_const_spmat_descr      source,
                                                        rocsparse_spmat_descr            target,
                                                        rocsparse_sparse_to_sparse_stage stage,
                                                        size_t* buffer_size_in_bytes);

/*! \ingroup generic_module
*  \brief Sparse matrix to sparse matrix conversion.
*
*  \details
*  \p rocsparse_sparse_to_sparse performs the conversion of a sparse matrix to a sparse matrix.
*
*  \note
*  The required allocation size (in bytes) to \p buffer_size_in_bytes must be obtained from \ref rocsparse_sparse_to_sparse_buffer_size
*  for each stage, indeed the required buffer size can be different between stages.
*
*  \note
*  The format \ref rocsparse_format_bell is not supported.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  descr        descriptor of the sparse_to_sparse algorithm.
*  @param[in]
*  source       sparse matrix descriptor.
*  @param[in]
*  target       sparse matrix descriptor.
*  @param[in]
*  stage        stage of the sparse_to_sparse computation.
*  @param[in]
*  buffer_size_in_bytes  size in bytes of the \p buffer
*  @param[in]
*  buffer  temporary storage buffer allocated by the user.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \par Example
*  This example converts a CSR matrix into an ELL matrix.
*  \snippet example_rocsparse_sparse_to_sparse.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sparse_to_sparse(rocsparse_handle                 handle,
                                            rocsparse_sparse_to_sparse_descr descr,
                                            rocsparse_const_spmat_descr      source,
                                            rocsparse_spmat_descr            target,
                                            rocsparse_sparse_to_sparse_stage stage,
                                            size_t                           buffer_size_in_bytes,
                                            void*                            buffer);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPARSE_TO_sparse_H */
