/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_SPSCALE_H
#define ROCSPARSE_SPSCALE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Buffer size step of the sparse matrix scaling.
*
*  \details
*  \p rocsparse_spscale_buffer_size returns the size in bytes of the temporary storage buffer
*  required by \ref rocsparse_spscale(). The same buffer is passed to \ref rocsparse_spscale().
*  For the currently supported formats no additional workspace is required and
*  \p buffer_size_in_bytes is set to zero; the routine is provided for interface consistency and
*  forward compatibility.
*
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  @param[in]
*  handle       handle to the rocSPARSE library context queue.
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor.
*  @param[in]
*  mat_C        sparse matrix \f$C\f$ descriptor.
*  @param[out]
*  buffer_size_in_bytes  number of bytes of the temporary storage buffer.
*  @param[out]
*  p_error      error descriptor created if the returned status is not
*               \ref rocsparse_status_success. A null pointer can be passed if an error
*               descriptor is not required.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p mat_A, \p mat_C or \p buffer_size_in_bytes
*          pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spscale_buffer_size(rocsparse_handle            handle,
                                               rocsparse_const_spmat_descr mat_A,
                                               rocsparse_spmat_descr       mat_C,
                                               size_t*                     buffer_size_in_bytes,
                                               rocsparse_error*            p_error);

/*! \ingroup generic_module
*  \brief Sparse matrix scaling.
*
*  \details
*  \p rocsparse_spscale multiplies the sparse matrix \f$A\f$ by the scalar \f$\alpha\f$ and
*  stores the result in the sparse matrix \f$C\f$:
*  \f[
*    C := \alpha \cdot A.
*  \f]
*  The output matrix \f$C\f$ has the same sparsity pattern as \f$A\f$ (its structure is copied
*  from \f$A\f$ and \f$nnz(C) = nnz(A)\f$) and each value is scaled by \f$\alpha\f$. This is a
*  uniform scalar multiply of the matrix values; it is unrelated to row/column equilibration
*  scaling.
*
*  \p C must be created with the same format and dimensions as \p A and with its arrays
*  allocated for \f$nnz(A)\f$ nonzeros; the column indices and value arrays are set through the
*  format specific set-pointers routine (e.g. \ref rocsparse_csr_set_pointers). A value of
*  \f$\alpha = 0\f$ produces \f$C\f$ with the sparsity pattern of \f$A\f$ and all values equal
*  to zero (the pattern is not dropped).
*
*  \p alpha can be passed in host or device memory according to the pointer mode set on the
*  handle (\ref rocsparse_set_pointer_mode).
*
*  \note The following formats are supported: \ref rocsparse_format_coo,
*  \ref rocsparse_format_coo_aos, \ref rocsparse_format_csr, \ref rocsparse_format_csc,
*  \ref rocsparse_format_bsr, \ref rocsparse_format_ell, \ref rocsparse_format_bell and
*  \ref rocsparse_format_sell. \p A and \p C must use the same format.
*  \note Only the non-transpose operation is supported.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spscale_uniform">Uniform Precisions</caption>
*  <tr><th>A / C
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  @param[in]
*  handle       handle to the rocSPARSE library context queue.
*  @param[in]
*  alpha        scalar \f$\alpha\f$.
*  @param[in]
*  mat_A        sparse matrix \f$A\f$ descriptor.
*  @param[out]
*  mat_C        sparse matrix \f$C\f$ descriptor.
*  @param[in]
*  buffer_size_in_bytes  number of bytes of the temporary storage buffer, as returned by
*               \ref rocsparse_spscale_buffer_size.
*  @param[in]
*  temp_buffer  temporary storage buffer allocated by the user.
*  @param[out]
*  p_error      error descriptor created if the returned status is not
*               \ref rocsparse_status_success. A null pointer can be passed if an error
*               descriptor is not required.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha, \p mat_A or \p mat_C pointer is invalid.
*  \retval rocsparse_status_not_implemented the formats of \p mat_A and \p mat_C differ, or the
*          format is not one of the supported formats.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spscale(rocsparse_handle            handle,
                                   const void*                 alpha,
                                   rocsparse_const_spmat_descr mat_A,
                                   rocsparse_spmat_descr       mat_C,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       temp_buffer,
                                   rocsparse_error*            p_error);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPSCALE_H */
