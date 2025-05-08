/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCSPARSE_INVERSE_PERMUTATION_H
#define ROCSPARSE_INVERSE_PERMUTATION_H

#include "../../rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup conv_module
*  \brief Create the identity map
*
*  \details
*  \p rocsparse_create_identity_permutation stores the identity map in \p p, such that
*  \f$p = 0:1:(n-1)\f$.
*
*  \code{.c}
*      for(i = 0; i < n; ++i)
*      {
*          p[i] = i;
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  n           size of the map \p p.
*  @param[out]
*  p           array of \p n integers containing the map.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p n is invalid.
*  \retval     rocsparse_status_invalid_pointer \p p pointer is invalid.
*
*  \par Example
*  The following example creates an identity permutation.
*  \code{.c}
*      rocsparse_int size = 200;
*
*      // Allocate memory to hold the identity map
*      rocsparse_int* perm;
*      hipMalloc((void**)&perm, sizeof(rocsparse_int) * size);
*
*      // Fill perm with the identity permutation
*      rocsparse_create_identity_permutation(handle, size, perm);
*  \endcode
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_identity_permutation(rocsparse_handle handle,
                                                       rocsparse_int    n,
                                                       rocsparse_int*   p);

/*! \ingroup conv_module
*  \brief Inverse a permutation vector.
*
*  \details
*  \p rocsparse_inverse_permutation computes
*
*  \code{.c}
*      for(i = 0; i < n; ++i)
*      {
*          q[p[i]- base] = i + base;
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  n           size of the permutation vector \p p.
*  @param[in]
*  p           array of \p n integers containing the permutation vector to inverse.
*  @param[out]
*  q           array of \p n integers containing the invsrse of the permutation vector.
*  @param[in]
*  base        \ref rocsparse_index_base_zero or \ref rocsparse_index_base_one.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p n is invalid.
*  \retval     rocsparse_status_invalid_pointer \p p pointer is invalid or \p q pointer is invalid.
*  \retval     rocsparse_status_invalid_value \p base is invalid.
*
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_inverse_permutation(rocsparse_handle     handle,
                                               rocsparse_int        n,
                                               const rocsparse_int* p,
                                               rocsparse_int*       q,
                                               rocsparse_index_base base);

/*! \ingroup conv_module
*  \brief Create the identity map
*
*  \details
*  \p rocsparse_set_identity_permutation stores the identity map in \p p, such that
*  \f$p = 0:1:(n-1)\f$.
*
*  \code{.c}
*      for(i = 0; i < n; ++i)
*      {
*          p[i] = i;
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  This routine supports execution in a hipGraph context.
*
*  @param[in]
*  handle      handle to the rocsparse library context queue.
*  @param[in]
*  n           size of the map \p p.
*  @param[out]
*  p           array of \p n integers containing the map.
*  @param[in]
*  indextype   the integer type of \p p. Can be \p rocsparse_indextype_i32 or \p rocsparse_indextype_i64.
*
*  \retval     rocsparse_status_success the operation completed successfully.
*  \retval     rocsparse_status_invalid_handle the library context was not initialized.
*  \retval     rocsparse_status_invalid_size \p n is invalid.
*  \retval     rocsparse_status_not_implemented if \p indextype is rocsparse_indextype_u16.
*  \retval     rocsparse_status_invalid_pointer \p p pointer is invalid.
*
*  \par Example
*  The following example creates an identity permutation.
*  \code{.c}
*      int32_t size = 200;
*
*      // Allocate memory to hold the identity map
*      int32_t* perm;
*      hipMalloc((void**)&perm, sizeof(int32_t) * size);
*
*      // Fill perm with the identity permutation
*      rocsparse_set_identity_permutation(handle, size, (void*)perm, rocsparse_indextype_i32);
*  \endcode
*/
ROCSPARSE_EXPORT
rocsparse_status rocsparse_set_identity_permutation(rocsparse_handle    handle,
                                                    int64_t             n,
                                                    void*               p,
                                                    rocsparse_indextype indextype);

#ifdef __cplusplus
}
#endif

#endif /* ROCSPARSE_INVERSE_PERMUTATION_H */
