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
*  \retval      rocsparse_status_invalid_size \p source has a batch count greater than one.
*  \retval      rocsparse_status_type_mismatch \p source and \p target differ in matrix type,
*               fill mode, diagonal type, index base, or storage mode.
*  \retval      rocsparse_status_not_implemented the conversion is not supported for the
*               formats or data types of \p source and \p target. A conversion that does
*               not need \ref rocsparse_sparse_to_sparse_permissive checks the data types
*               only in \ref rocsparse_sparse_to_sparse. A permissive conversion converts
*               \p source into its intermediate CSR matrix during this query for
*               \ref rocsparse_sparse_to_sparse_stage_analysis, so this query returns
*               the status when that conversion does not support the data type of
*               \p source.
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
*  \ref rocsparse_format_bell and \ref rocsparse_format_sell are not supported.
*  See the table below for the supported format and data type combinations.
*
*  <table>
*  <tr><th>source / target <th>CSR <th>CSC <th>COO <th>COO AoS <th>ELL <th>BSR
*  <tr><td>CSR
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_same_a "Same (a)"
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_same_b "Same (b)"
*      <td>\ref sparse_to_sparse_same_b "Same (b)"
*  <tr><td>CSC
*      <td>\ref sparse_to_sparse_same_a "Same (a)"
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_permissive_conversions "Permissive"
*      <td>\ref sparse_to_sparse_permissive_conversions "Permissive"
*      <td>\ref sparse_to_sparse_permissive_b "Permissive (b)"
*      <td>\ref sparse_to_sparse_permissive_b "Permissive (b)"
*  <tr><td>COO
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_permissive_a "Permissive (a)"
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_permissive_b "Permissive (b)"
*      <td>\ref sparse_to_sparse_permissive_b "Permissive (b)"
*  <tr><td>COO AoS
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_permissive_a "Permissive (a)"
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_permissive_b "Permissive (b)"
*      <td>\ref sparse_to_sparse_permissive_b "Permissive (b)"
*  <tr><td>ELL
*      <td>\ref sparse_to_sparse_same_c "Same (c)"
*      <td>\ref sparse_to_sparse_permissive_c "Permissive (c)"
*      <td>\ref sparse_to_sparse_permissive_conversions "Permissive"
*      <td>\ref sparse_to_sparse_permissive_conversions "Permissive"
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*      <td>\ref sparse_to_sparse_permissive_c "Permissive (c)"
*  <tr><td>BSR
*      <td>\ref sparse_to_sparse_same_b "Same (b)"
*      <td>\ref sparse_to_sparse_permissive_b "Permissive (b)"
*      <td>\ref sparse_to_sparse_permissive_conversions "Permissive"
*      <td>\ref sparse_to_sparse_permissive_conversions "Permissive"
*      <td>\ref sparse_to_sparse_permissive_b "Permissive (b)"
*      <td>\ref sparse_to_sparse_mixed_conversions "Mixed"
*  </table>
*
*  Mixed copies every identical \ref rocsparse_datatype and converts different data types with
*  the pairs listed in \ref sparse_to_sparse_mixed_conversions "Mixed conversions".
*  Same (a), Same (b), and Same (c) keep the source data type and accept only that set.
*  Any other data type returns \ref rocsparse_status_not_implemented.
*  \ref rocsparse_status_invalid_value is returned when the data type is not a valid
*  \ref rocsparse_datatype.
*
*  The ELL to CSR and CSR to BSR conversions report an unsupported data type in the
*  \ref rocsparse_sparse_to_sparse_stage_compute stage, after the
*  \ref rocsparse_sparse_to_sparse_stage_analysis stage has written the row pointer array of
*  \p target. A permissive conversion to BSR does the same when the conversion into the
*  intermediate CSR matrix succeeds.
*
*  A conversion that writes values returns \ref rocsparse_status_not_implemented for an
*  unsupported data type pair even when the matrix has zero nonzeros, if both value arrays
*  are set. This includes the permissive conversions, which check the data types of
*  \p source and \p target in the \ref rocsparse_sparse_to_sparse_stage_compute stage: a
*  COO or COO AoS target needs a supported conversion between them, and a CSC, ELL or BSR
*  target needs the same data type. The CSR to CSC and CSC to CSR conversions compare data
*  types only when both value arrays are set, and otherwise dispatch on the source data
*  type. BSR to CSR compares data types only when both value arrays are set and dispatches
*  on the target data type. A dispatched type that the routine does not implement still
*  returns an error.
*
*  Conversions to a single-precision type round each value (each real and imaginary part) to
*  nearest. Values that overflow single precision become infinity of the same sign, NaN is
*  preserved, and \ref rocsparse_status_success is returned.
*
*  \section sparse_to_sparse_type_rules Data type rules
*
*  \subsection sparse_to_sparse_mixed_conversions Mixed conversions
*  <table>
*  <tr><th>source                   <th>target
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f32_c <td>rocsparse_datatype_f64_c
*  <tr><td>rocsparse_datatype_f64_c <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f64_c
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f64_c
*  <tr><td>rocsparse_datatype_f16_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f16_r <td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f16_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f16_r <td>rocsparse_datatype_f64_c
*  <tr><td>rocsparse_datatype_bf16_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_bf16_r <td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_bf16_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_bf16_r <td>rocsparse_datatype_f64_c
*  </table>
*
*  These real widenings are exact: every source value is representable in the target, and a
*  real value stored in a complex target has an imaginary part of zero.
*  \ref rocsparse_datatype_f64_r into \ref rocsparse_datatype_f32_c rounds the real part.
*
*  \subsection sparse_to_sparse_same_a Same (a)
*  <table>
*  <tr><th>data type
*  <tr><td>rocsparse_datatype_i8_r
*  <tr><td>rocsparse_datatype_u8_r
*  <tr><td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_u32_r
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \subsection sparse_to_sparse_same_b Same (b)
*  <table>
*  <tr><th>data type
*  <tr><td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \subsection sparse_to_sparse_same_c Same (c)
*  <table>
*  <tr><th>data type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \subsection sparse_to_sparse_permissive_conversions Permissive
*  These pairs require \ref rocsparse_sparse_to_sparse_permissive. The conversion goes through
*  an intermediate CSR matrix, which adds a pass over the matrix. The supported conversions are:
*  <table>
*  <tr><th>source                   <th>target
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f32_c <td>rocsparse_datatype_f64_c
*  <tr><td>rocsparse_datatype_f64_c <td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f32_r <td>rocsparse_datatype_f64_c
*  <tr><td>rocsparse_datatype_f64_r <td>rocsparse_datatype_f64_c
*  </table>
*
*  CSC, ELL and BSR to COO with an \ref rocsparse_datatype_f16_r or
*  \ref rocsparse_datatype_bf16_r source still go through the existing same-type CSR
*  conversion, which does not support those two types:
*  \ref rocsparse_sparse_to_sparse_buffer_size for
*  \ref rocsparse_sparse_to_sparse_stage_analysis returns
*  \ref rocsparse_status_not_implemented. COO AoS from CSC, ELL and BSR uses the same
*  intermediate conversion.
*
*  \subsection sparse_to_sparse_permissive_a Permissive (a)
*  These pairs require \ref rocsparse_sparse_to_sparse_permissive. The conversion goes through
*  an intermediate CSR matrix and keeps one of these source data types.
*  <table>
*  <tr><th>data type
*  <tr><td>rocsparse_datatype_i8_r
*  <tr><td>rocsparse_datatype_u8_r
*  <tr><td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_u32_r
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \subsection sparse_to_sparse_permissive_b Permissive (b)
*  These pairs require \ref rocsparse_sparse_to_sparse_permissive. The conversion goes through
*  an intermediate CSR matrix and keeps one of these source data types.
*  <table>
*  <tr><th>data type
*  <tr><td>rocsparse_datatype_i32_r
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \subsection sparse_to_sparse_permissive_c Permissive (c)
*  These pairs require \ref rocsparse_sparse_to_sparse_permissive. The conversion goes through
*  an intermediate CSR matrix and keeps one of these source data types.
*  <table>
*  <tr><th>data type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \note
*  The required allocation size (in bytes) to \p buffer_size_in_bytes must be obtained from \ref rocsparse_sparse_to_sparse_buffer_size
*  for each stage. The required buffer size can be different between stages.
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
*  \retval      rocsparse_status_invalid_handle the library context was not initialized.
*  \retval      rocsparse_status_invalid_pointer \p descr, \p source, \p target, or \p buffer
*               is invalid.
*  \retval      rocsparse_status_invalid_value \p stage is not a valid enumeration.
*  \retval      rocsparse_status_invalid_size a conversion to the same format has a
*               \p source and a \p target with different numbers of rows or columns.
*  \retval      rocsparse_status_type_mismatch \p source and \p target differ in matrix type,
*               fill mode, diagonal type, index base, or storage mode, or an index value
*               does not fit in the target index type.
*  \retval      rocsparse_status_not_implemented the conversion is not supported for the formats
*               or data types of \p source and \p target, or the matrix is batched.
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
