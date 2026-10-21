/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
*  handle       handle to the rocSPARSE library context queue.
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
*  \note
*  This routine does not support batched computation.
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
*  for each stage. The required buffer size can be different between stages.
*
*  \note
*  The \ref rocsparse_format_bell and \ref rocsparse_format_sell formats are not supported.
*
*  \note
*  If \p source and \p target have different data types, only the conversions
*  \ref rocsparse_datatype_f32_r <-> \ref rocsparse_datatype_f64_r,
*  \ref rocsparse_datatype_f32_r or \ref rocsparse_datatype_f64_r -> \ref rocsparse_datatype_f32_c
*  and \ref rocsparse_datatype_f32_c <-> \ref rocsparse_datatype_f64_c are supported, depending on
*  the format of \p target:
*  - same format as \p source: the values are converted.
*  - \ref rocsparse_format_coo or \ref rocsparse_format_coo_aos: the values are converted. From
*    the CSC, ELL and BSR formats, this needs \ref rocsparse_sparse_to_sparse_permissive, and the
*    conversion goes through an intermediate CSR matrix.
*  - \ref rocsparse_format_csr: the values are also converted from the COO and COO AoS formats.
*  - \ref rocsparse_format_csc, \ref rocsparse_format_ell, \ref rocsparse_format_bsr: only from
*    the same format.
*
*  Other combinations return \ref rocsparse_status_not_implemented.
*
*  \note
*  This routine does not support batched computation.
*
*  @param[in]
*  handle       handle to the rocSPARSE library context queue.
*  @param[in]
*  descr        descriptor of the sparse_to_sparse algorithm.
*  @param[in]
*  source       sparse matrix descriptor.
*  @param[in]
*  target       sparse matrix descriptor.
*  @param[in]
*  stage        stage of the sparse_to_sparse computation.
*  @param[in]
*  buffer_size_in_bytes  size in bytes of the \p buffer.
*  @param[in]
*  buffer  temporary storage buffer allocated by the user.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_not_implemented the conversion is not supported for the formats or data types of \p source and \p target.
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
