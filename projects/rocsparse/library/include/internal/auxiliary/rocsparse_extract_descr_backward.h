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

/*! \file
 *  \brief rocsparse_extract_descr_backward.h provides auxilary functions in rocsparse
 *  but without using a rocsparse_handle.
 * Causing a disruption in the stream to use, as the default is the only available.
 */

#ifndef ROCSPARSE_EXTRACT_DESCR_BACKWARD_H
#define ROCSPARSE_EXTRACT_DESCR_BACKWARD_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
*  \brief Sparse matrix extraction.
*
*  \details
*  \p rocsparse_create_extract_descr creates the descriptor of the extract algorithm.

*  @param[out]
*  descr        pointer to the descriptor of the extract algorithm.
*  @param[in]
*  source       source sparse matrix descriptor.
*  @param[in]
*  target       target sparse matrix descriptor.
*  @param[in]
*  alg          algorithm for the extract computation.
*
*  \retval      rocsparse_status_success the operation completed successfully.
*  \retval      rocsparse_status_invalid_value if any required enumeration is invalid.
*  \retval      rocsparse_status_invalid_pointer \p descr, \p source, or \p target
*               pointer is invalid.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_extract_descr(rocsparse_extract_descr*    descr,
                                                rocsparse_const_spmat_descr source,
                                                rocsparse_spmat_descr       target,
                                                rocsparse_extract_alg       alg);

/*! \ingroup aux_module
*  \brief Sparse matrix extraction.
*
*  \details
*  \p rocsparse_destroy_extract_descr destroys the descriptor of the \ref rocsparse_extract routine.
*
*  @param[in]
*  descr        descriptor of the extract routine.
*  \retval      rocsparse_status_success the operation completed successfully.
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_destroy_extract_descr(rocsparse_extract_descr descr);

#ifdef __cplusplus
}
#endif

#endif
