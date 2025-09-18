/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_SPGEAM_H
#define ROCSPARSE_SPGEAM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p rocsparse_spgeam_buffer_size returns the size of the required buffer to execute the given stage of the SpGEAM operation.
*  This routine is used in conjunction with \ref rocsparse_spgeam(). See \ref rocsparse_spgeam for full description and example.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  descr        SpGEAM descriptor
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  mat_B        sparse matrix \f$B\f$ descriptor.
*  @param[in]
*  mat_C        sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  stage        SpGEAM stage for the SpGEAM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p mat_A, \p mat_B, \p descr or \p buffer_size pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spgeam_buffer_size(rocsparse_handle            handle,
                                              rocsparse_spgeam_descr      descr,
                                              rocsparse_const_spmat_descr mat_A,
                                              rocsparse_const_spmat_descr mat_B,
                                              rocsparse_const_spmat_descr mat_C,
                                              rocsparse_spgeam_stage      stage,
                                              size_t*                     buffer_size,
                                              rocsparse_error*            error);

/*! \ingroup generic_module
*  \brief Sparse matrix sparse matrix addition
*
*  \details
*  \p rocsparse_spgeam multiplies the scalar \f$\alpha\f$ with the sparse \f$m \times n\f$ CSR matrix \f$op(A)\f$
*  and adds it to \f$\beta\f$ multiplied by the sparse \f$m \times n\f$ matrix \f$op(B)\f$. The final result is
*  stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha op(A) + \beta op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == rocsparse_operation_none}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == rocsparse_operation_none}
*    \end{array}
*    \right.
*  \f]
*
*  \p rocsparse_spgeam requires multiple steps to complete. First, the user must create a \ref rocsparse_spgeam_descr by
*  calling \ref rocsparse_create_spgeam_descr. The user sets the SpGEAM algorithm (currently only
*  \ref rocsparse_spgeam_alg_default supported) as well as the compute type and the transpose operation type for the sparse
*  matrices \f$op(A)\f$ and \f$op(B)\f$ using \ref rocsparse_spgeam_set_input. Next, the user must calculate the total nonzeros
*  that will exist in the sparse matrix \f$C\f$. To do so, the user calls \ref rocsparse_spgeam_buffer_size with the stage set
*  to \ref rocsparse_spgeam_stage_analysis. This will fill the \p buffer_size parameter allowing the user to then allocate this buffer.
*  Once the buffer has been allocated, the user calls \p rocsparse_spgeam with the same stage \ref rocsparse_spgeam_stage_analysis.
*  The total nonzeros and the row offset array for \f$C\f$ has now been calculated and is stored internally in the
*  \ref rocsparse_spgeam_descr. The user now needs to retrieve the nonzero count using \ref rocsparse_spgeam_get_output and then
*  allocate the \f$C\f$ matrix. To complete the computation, the user repeats the process (this time passing the stage
*  \ref rocsparse_spgeam_stage_compute) by calling \ref rocsparse_spgeam_buffer_size to determine the required buffer size, then
*  allocating the buffer, and finally calling \p rocsparse_spgeam. The user allocated buffers can be freed after each call to
*  \p rocsparse_spgeam. Once the computation is complete and the SpGEAM descriptor is no longer needed, the user must call
*  \ref rocsparse_destroy_spgeam_descr. See full code example below.
*
*  The stage \ref rocsparse_spgeam_stage_compute computes the symbolic part and the numeric of the resulting matrix C. If the user wants to perform multiple operations involving matrices of same sparsity patterns but with different numerical values, then the symbolic stages (\ref rocsparse_spgeam_stage_symbolic_analysis and \ref rocsparse_spgeam_stage_symbolic_compute) and the numeric stages (\ref rocsparse_spgeam_stage_numeric_analysis and \ref rocsparse_spgeam_stage_numeric_compute) can be used to separate the symbolic calculation from the numeric calculation.
*
*  \note The stages  \ref rocsparse_spgeam_stage_analysis and \ref rocsparse_spgeam_stage_compute cannot be mixed with the stages \ref rocsparse_spgeam_stage_symbolic_analysis, \ref rocsparse_spgeam_stage_symbolic_compute, \ref rocsparse_spgeam_stage_numeric_analysis, and \ref rocsparse_spgeam_stage_numeric_compute.
*  \note The stage \ref rocsparse_spgeam_stage_analysis must precede the stage \ref rocsparse_spgeam_stage_compute.
*  \note The stage \ref rocsparse_spgeam_stage_symbolic_analysis must precede the stage \ref rocsparse_spgeam_stage_symbolic_compute.
*  \note The stage \ref rocsparse_spgeam_stage_numeric_analysis must precede the stage \ref rocsparse_spgeam_stage_numeric_compute.
*  \note The symbolic stages are not required to perform the numeric stages.
*  \note The stage \ref rocsparse_spgeam_stage_numeric_analysis must be re-applied if the numeric values of the input matrices \p mat_A and \p mat_B have changed between subsquent calls of the stage \ref rocsparse_spgeam_stage_numeric_compute.
*
*  \p rocsparse_spgeam supports multiple combinations of index types, data types, and compute types. The tables below indicate
*  the currently supported different index and data types that can be used for the sparse matrices \f$op(A)\f$, \f$op(B)\f$, and
*  \f$C\f$, and the compute type for \f$\alpha\f$ and \f$\beta\f$. The advantage of using different index and data types is to save on
*  memory bandwidth and storage when a user application allows while performing the actual computation in a higher precision.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spgeam_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Uniform Index Types:
*  <table>
*  <caption id="spgeam_csr_uniform_indextype">CSR Uniform Index Types</caption>
*  <tr><th>CSR Row offset                 <th>CSR Column indices
*  <tr><td>rocsparse_datatype_f32_r</td>  <td>rocsparse_datatype_f32_r</td>
*  <tr><td>rocsparse_datatype_f64_r</td>  <td>rocsparse_datatype_f64_r</td>
*  </table>
*
*  \par Mixed Index Types:
*  <table>
*  <caption id="spgeam_csr_mixed_indextype">CSR Mixed Index Types</caption>
*  <tr><th>CSR Row offset                 <th>CSR Column indices
*  <tr><td>rocsparse_datatype_f64_r</td>  <td>rocsparse_datatype_f32_r</td>
*  </table>
*
*  In general, when adding two sparse matrices together, it is entirely possible that the resulting matrix will require a
*  a larger index representation to store correctly. For example, when adding \f$A + B\f$ using
*  \ref rocsparse_indextype_i32 index types for the row pointer and column indices arrays, it may be the case that the row pointer
*  of the resulting \f$C\f$ matrix would require index type \ref rocsparse_indextype_i64. This is currently not supported. In this
*  scenario, the user would need to store the \f$A\f$, \f$B\f$, and \f$C\f$ matrices using the higher index precision.
*
*  Additionally, all three matrices \f$A\f$, \f$B\f$, and \f$C\f$ must use the same index types. For example, if \f$A\f$ uses the
*  index type \ref rocsparse_datatype_f32_r for the row offset array and the index type \ref rocsparse_datatype_f32_r for the column
*  indices array, then both \f$B\f$ and \f$C\f$ must also use these same index types for their respective row offset and column index
*  arrays. In the scenario where \f$C\f$ requires a larger index type for the row offset array, the user would need to store all three
*  matrices using the larger index type \ref rocsparse_datatype_f64_r for the row offsets array.
*
*  \note Currently only CSR format is supported.
*  \note Currently, only \p trans_A == \ref rocsparse_operation_none is supported.
*  \note Currently, only \p trans_B == \ref rocsparse_operation_none is supported.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  descr        SpGEAM descriptor
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  mat_B        sparse matrix \f$B\f$ descriptor.
*  @param[out]
*  mat_C        sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  stage        SpGEAM stage for the SpGEAM computation.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer. \p buffer_size is
*               determined by calling \ref rocsparse_spgeam_buffer_size.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*  @param[out]
*  error        error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if the user is not interested in obtaining an error descriptor.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p mat_A, \p mat_B, \p mat_C, \p descr or \p buffer_size pointer is invalid.
*
*  \par First Example
*  \snippet example_rocsparse_spgeam_1.cpp doc example
*
*  \par Second Example
*  \snippet example_rocsparse_spgeam_2.cpp doc example
*
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spgeam(rocsparse_handle            handle,
                                  rocsparse_spgeam_descr      descr,
                                  rocsparse_const_spmat_descr mat_A,
                                  rocsparse_const_spmat_descr mat_B,
                                  rocsparse_spmat_descr       mat_C,
                                  rocsparse_spgeam_stage      stage,
                                  size_t                      buffer_size,
                                  void*                       temp_buffer,
                                  rocsparse_error*            error);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPGEAM_H */
