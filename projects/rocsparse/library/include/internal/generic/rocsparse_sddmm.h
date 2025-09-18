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

#ifndef ROCSPARSE_SDDMM_H
#define ROCSPARSE_SDDMM_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \details
*  \p rocsparse_sddmm_buffer_size returns the size of the required buffer to execute the SDDMM operation from a given configuration.
*  This routine is used in conjunction with \ref rocsparse_sddmm_preprocess() and \ref rocsparse_sddmm().
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  opA          dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB          dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat_A        dense matrix \f$A\f$ descriptor.
*  @param[in]
*  mat_B        dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  mat_C        sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  compute_type floating point precision for the SDDMM computation.
*  @param[in]
*  alg specification of the algorithm to use.
*  @param[out]
*  buffer_size  number of bytes of the temporary storage buffer.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_value the value of \p opA or \p opB is incorrect.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p mat_A, \p mat_B, \p mat_C or \p buffer_size pointer is invalid.
*  \retval rocsparse_status_not_implemented
*          \p opA == \ref rocsparse_operation_conjugate_transpose or
*          \p opB == \ref rocsparse_operation_conjugate_transpose.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sddmm_buffer_size(rocsparse_handle            handle,
                                             rocsparse_operation         opA,
                                             rocsparse_operation         opB,
                                             const void*                 alpha,
                                             rocsparse_const_dnmat_descr mat_A,
                                             rocsparse_const_dnmat_descr mat_B,
                                             const void*                 beta,
                                             rocsparse_spmat_descr       mat_C,
                                             rocsparse_datatype          compute_type,
                                             rocsparse_sddmm_alg         alg,
                                             size_t*                     buffer_size);

/*! \ingroup generic_module
*  \details
*  \p rocsparse_sddmm_preprocess executes a part of the algorithm that can be calculated once in the context of multiple
*  calls of the \ref rocsparse_sddmm with the same sparsity pattern.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  opA          dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB          dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat_A        dense matrix \f$A\f$ descriptor.
*  @param[in]
*  mat_B        dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  mat_C        sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  compute_type floating point precision for the SDDMM computation.
*  @param[in]
*  alg specification of the algorithm to use.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*  The size must be greater or equal to the size obtained with \ref rocsparse_sddmm_buffer_size.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_value the value of \p opA or \p opB is incorrect.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p mat_A, \p mat_B, \p mat_C or \p temp_buffer pointer is invalid.
*  \retval rocsparse_status_not_implemented
*          \p opA == \ref rocsparse_operation_conjugate_transpose or
*          \p opB == \ref rocsparse_operation_conjugate_transpose.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sddmm_preprocess(rocsparse_handle            handle,
                                            rocsparse_operation         opA,
                                            rocsparse_operation         opB,
                                            const void*                 alpha,
                                            rocsparse_const_dnmat_descr mat_A,
                                            rocsparse_const_dnmat_descr mat_B,
                                            const void*                 beta,
                                            rocsparse_spmat_descr       mat_C,
                                            rocsparse_datatype          compute_type,
                                            rocsparse_sddmm_alg         alg,
                                            void*                       temp_buffer);

/*! \ingroup generic_module
*  \brief  Sampled Dense-Dense Matrix Multiplication.
*
*  \details
*  \p rocsparse_sddmm multiplies the scalar \f$\alpha\f$ with the dense
*  \f$m \times k\f$ matrix \f$op(A)\f$, the dense \f$k \times n\f$ matrix \f$op(B)\f$, filtered by the sparsity pattern
*  of the \f$m \times n\f$ sparse matrix \f$C\f$ and adds the result to \f$C\f$ scaled by
*  \f$\beta\f$. The final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha ( op(A) \cdot op(B) ) \circ spy(C) + \beta C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if op(A) == rocsparse_operation_none} \\
*        A^T, & \text{if op(A) == rocsparse_operation_transpose} \\
*    \end{array}
*    \right.
*  \f],
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if op(B) == rocsparse_operation_none} \\
*        B^T, & \text{if op(B) == rocsparse_operation_transpose} \\
*    \end{array}
*    \right.
*  \f]
*   and
*  \f[
*    spy(C)_{ij} = \left\{
*    \begin{array}{ll}
*        1, & \text{ if C_{ij} != 0} \\
*        0, & \text{ otherwise} \\
*    \end{array}
*    \right.
*  \f]
*
*  Computing the above sampled dense-dense multiplication requires three steps to complete. First, the user calls
*  \ref rocsparse_sddmm_buffer_size to determine the size of the required temporary storage buffer. Next, the user
*  allocates this buffer and calls \ref rocsparse_sddmm_preprocess which performs any analysis of the input matrices
*  that may be required. Finally, the user calls \p rocsparse_sddmm to complete the computation. Once all calls to
*  \p rocsparse_sddmm are complete, the temporary buffer can be deallocated.
*
*  \p rocsparse_sddmm supports different algorithms which can provide better performance for different matrices.
*
*  <table>
*  <caption id="sddmm_algorithms">Algorithms</caption>
*  <tr><th>CSR/CSC Algorithms                <th>Deterministic  <th>Preprocessing  <th>Notes
*  <tr><td>rocsparse_sddmm_alg_default</td>  <td>Yes</td>       <td>No</td>        <td>Uses the sparsity pattern of matrix C to perform a limited set of dot products </td>
*  <tr><td>rocsparse_sddmm_alg_dense</td>    <td>Yes</td>       <td>No</td>        <td>Explicitly converts the matrix C into a dense matrix to perform a dense matrix multiply and add </td>
*  </table>
*
*  Currently, \p rocsparse_sddmm only supports the uniform precisions indicated in the table below. For the sparse matrix \f$C\f$, \p rocsparse_sddmm supports the index types
*  \ref rocsparse_indextype_i32 and \ref rocsparse_indextype_i64.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="sddmm_uniform">Uniform Precisions</caption>
*  <tr><th>A / B / C / compute_type
*  <tr><td>rocsparse_datatype_f16_r
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  \par Mixed precisions:
*  <table>
*  <caption id="sddmm_mixed">Mixed Precisions</caption>
*  <tr><th>A / B                     <th>C                         <th>compute_type
*  <tr><td>rocsparse_datatype_f16_r  <td>rocsparse_datatype_f32_r  <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f16_r  <td>rocsparse_datatype_f16_r  <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_bf16_r <td>rocsparse_datatype_f32_r  <td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_bf16_r <td>rocsparse_datatype_bf16_r <td>rocsparse_datatype_f32_r
*  </table>
*
*  \note
*  The sparse matrix formats currently supported are: \ref rocsparse_format_csr.
*
*  \note \p opA == \ref rocsparse_operation_conjugate_transpose is not supported.
*  \note \p opB == \ref rocsparse_operation_conjugate_transpose is not supported.
*
*  \note
*  This routine supports execution in a hipGraph context only when \p alg == \ref rocsparse_sddmm_alg_default.
*
*  @param[in]
*  handle       handle to the rocsparse library context queue.
*  @param[in]
*  opA          dense matrix \f$A\f$ operation type.
*  @param[in]
*  opB          dense matrix \f$B\f$ operation type.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat_A        dense matrix \f$A\f$ descriptor.
*  @param[in]
*  mat_B        dense matrix \f$B\f$ descriptor.
*  @param[in]
*  beta         scalar \f$\beta\f$.
*  @param[inout]
*  mat_C        sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  compute_type floating point precision for the SDDMM computation.
*  @param[in]
*  alg specification of the algorithm to use.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*  The size must be greater or equal to the size obtained with \ref rocsparse_sddmm_buffer_size.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_value the value of \p opA, \p opB, \p compute\_type or alg is incorrect.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha and \p beta are invalid,
*          \p mat_A, \p mat_B, \p mat_C or \p temp_buffer pointer is invalid.
*  \retval rocsparse_status_not_implemented
*          \p opA == \ref rocsparse_operation_conjugate_transpose or
*          \p opB == \ref rocsparse_operation_conjugate_transpose.
*
*  \par Example
*  This example performs sampled dense-dense matrix product, \f$C := \alpha ( A \cdot B ) \circ spy(C) + \beta C\f$
*  where \f$\circ\f$ is the hadamard product
*  \snippet example_rocsparse_sddmm.cpp doc example
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_sddmm(rocsparse_handle            handle,
                                 rocsparse_operation         opA,
                                 rocsparse_operation         opB,
                                 const void*                 alpha,
                                 rocsparse_const_dnmat_descr mat_A,
                                 rocsparse_const_dnmat_descr mat_B,
                                 const void*                 beta,
                                 rocsparse_spmat_descr       mat_C,
                                 rocsparse_datatype          compute_type,
                                 rocsparse_sddmm_alg         alg,
                                 void*                       temp_buffer);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SDDMM_H */
