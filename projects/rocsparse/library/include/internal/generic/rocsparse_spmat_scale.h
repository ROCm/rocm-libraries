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

#ifndef ROCSPARSE_SPMAT_SCALE_H
#define ROCSPARSE_SPMAT_SCALE_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup generic_module
*  \brief Sparse matrix scaling.
*
*  \details
*  \p rocsparse_spmat_scale multiplies the values of the sparse matrix \p source (\f$A\f$) by the
*  scalar \f$\alpha\f$ and writes them into the sparse matrix \p target (\f$C\f$):
*  \f[
*    C := \alpha \cdot A.
*  \f]
*  Only the value array of \p target is written; its data type is scaled by \f$\alpha\f$. This is a
*  uniform scalar multiply of the matrix values; it is unrelated to row/column equilibration
*  scaling. No temporary storage buffer is required.
*
*  This routine does not copy the sparsity pattern. \p target is assumed to already describe the
*  same sparsity pattern as \p source (same format, dimensions and nonzero count), so its index
*  arrays are left untouched. When \p target and \p source alias the same value array the scaling
*  is performed in place; otherwise the scaled source values are written into \p target. In-place
*  operation is the common case. A value of \f$\alpha = 0\f$ writes all-zero values into \p target
*  (the pattern is not dropped).
*
*  The scaling factor \f$\alpha\f$ is passed as a size-one dense vector descriptor. It can live in
*  host or device memory; the memory space is taken from the descriptor itself (see
*  \ref rocsparse_dnvec_descr_create_scalar), so the handle pointer mode does not affect it. The
*  data type of \p alpha must match the data type of the matrices.
*
*  \note The following formats are supported: \ref rocsparse_format_coo,
*  \ref rocsparse_format_coo_aos, \ref rocsparse_format_csr, \ref rocsparse_format_csc,
*  \ref rocsparse_format_bsr, \ref rocsparse_format_ell, \ref rocsparse_format_bell and
*  \ref rocsparse_format_sell. \p source and \p target must use the same format.
*  \note Batched matrices are not supported.
*  \note
*  This routine does not support execution in a hipGraph context.
*
*  \par Uniform Precisions:
*  <table>
*  <caption id="spmat_scale_uniform">Uniform Precisions</caption>
*  <tr><th>alpha / A / C
*  <tr><td>rocsparse_datatype_f32_r
*  <tr><td>rocsparse_datatype_f64_r
*  <tr><td>rocsparse_datatype_f32_c
*  <tr><td>rocsparse_datatype_f64_c
*  </table>
*
*  @param[in]
*  handle       handle to the rocSPARSE library context queue.
*  @param[in]
*  alpha        size-one dense vector descriptor holding the scalar \f$\alpha\f$.
*  @param[in]
*  source       sparse matrix \f$A\f$ descriptor.
*  @param[out]
*  target       sparse matrix \f$C\f$ descriptor.
*  @param[out]
*  p_error      error descriptor created if the returned status is not
*               \ref rocsparse_status_success. A null pointer can be passed if an error
*               descriptor is not required.
*
*  \retval rocsparse_status_success the operation completed successfully.
*  \retval rocsparse_status_invalid_handle the library context was not initialized.
*  \retval rocsparse_status_invalid_pointer \p alpha, \p source or \p target pointer is invalid.
*  \retval rocsparse_status_not_implemented the formats of \p source and \p target differ, the
*          format is not one of the supported formats, or a batched matrix is passed.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_scale(rocsparse_handle            handle,
                                       rocsparse_const_dnvec_descr alpha,
                                       rocsparse_const_spmat_descr source,
                                       rocsparse_spmat_descr       target,
                                       rocsparse_error*            p_error);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_SPMAT_SCALE_H */
