/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifdef ROCSPARSE_DNMAT_COPY_DATA_H
#define ROCSPARSE_DNMAT_COPY_DATA_H

#include "rocsparse-types.h"
#include "rocsparse-version.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Copy data between two dense matrices.
 *  \details
 *  \p rocsparse_dnmat_copy_data copies the data from a \p X dense matrix to a \p Y dense matrix.
 *  \f[
 *    Y =  \alpha \cdot X,
 *  \f]
 *  where \alpha, or \p alpha is a scalar.
 *
 *  \note Dense matrix orders (\ref rocsparse_order) can be arbitrary.
 *  The number of rows and the number of columns must be the same.
 *  All the data types, i.e. from \p alpha, \p X and \p Y are the same.
 *
 *  \note Complete strided batch operation are supported. The batch_count of the operation is given by the batch_count of the matrix \p Y,
 *  meaning that batch_count values for \p alpha and \p X must be either the same or one, i.e. the following operations are supported\f$0 \le k \lt batch_count\f$
 *  \f[
 *    Y_k =  \alpha_k \cdot X_k,
 *  \f]
 *  \f[
 *    Y_k =  \alpha_k \cdot X,
 *  \f]
 *  or
 *  \f[
 *    Y_k =  \alpha \cdot X_k, or
 *  \f]
 *  \f[
 *    Y_k =  \alpha \cdot X.
 *  \f]
 *
 *
 *  @param[in]
 *  handle      handle to the rocSPARSE library context queue.
 *  @param[in]
 *  alpha       optional, scale the operation if not null.
 *  @param[in]
 *  X           source matrix.
 *  @param[inout]
 *  Y           target matrix.
 *  @param[out]
 *  p_error      error descriptor created if the returned status is not \ref rocsparse_status_success. A null pointer can be passed if an error descriptor is not required.
 *
 *  \retval rocsparse_status_success the operation completed successfully.
 *  \retval rocsparse_status_invalid_pointer if \p Y, or \p X is invalid.
 *  \retval rocsparse_status_invalid_value if dimensions or data types are invalid.
 *  \par Example
 *  \snippet example_rocsparse_dnmat_copy_data.cpp doc example
 */
ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_copy_data(rocsparse_handle            handle,
                                                            rocsparse_const_dnvec_descr alpha,
                                                            rocsparse_const_dnmat_descr X,
                                                            rocsparse_dnmat_descr       Y,
                                                            rocsparse_error*            p_error);

#ifdef __cplusplus
}
#endif

#endif
