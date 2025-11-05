/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 *  \brief rocsparse_spmat_descr.h provides auxilary functions in rocsparse
 */

#ifndef ROCSPARSE_SPMAT_DESCR_H
#define ROCSPARSE_SPMAT_DESCR_H

#include "internal/auxiliary/rocsparse_spmat_descr_backward.h"
#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_create(rocsparse_handle         handle,
                                        rocsparse_spmat_descr*   p_descr,
                                        rocsparse_spattern_descr spattern,
                                        rocsparse_dnvec_descr    values,
                                        rocsparse_error*         p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_create_block(rocsparse_handle         handle,
                                              rocsparse_spmat_descr*   p_descr,
                                              rocsparse_spattern_descr spattern,
                                              rocsparse_direction      block_dir,
                                              int64_t                  block_dim,
                                              rocsparse_dnvec_descr    values,
                                              rocsparse_error*         p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_prop(rocsparse_handle            handle,
                                          rocsparse_const_spmat_descr descr,
                                          rocsparse_spmat_prop        prop,
                                          void*                       value,
                                          size_t                      value_size_in_bytes,
                                          rocsparse_error*            p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_prop(rocsparse_handle      handle,
                                          rocsparse_spmat_descr descr,
                                          rocsparse_spmat_prop  prop,
                                          const void*           value,
                                          size_t                value_size_in_bytes,
                                          rocsparse_error*      p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_spattern(rocsparse_handle          handle,
                                              rocsparse_spmat_descr     descr,
                                              rocsparse_spattern_descr* p_value,
                                              rocsparse_error*          p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_spattern(rocsparse_handle         handle,
                                              rocsparse_spmat_descr    descr,
                                              rocsparse_spattern_descr value,
                                              rocsparse_error*         p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_get_data(rocsparse_handle       handle,
                                          rocsparse_spmat_descr  descr,
                                          rocsparse_dnvec_descr* p_value,
                                          rocsparse_error*       p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spmat_set_data(rocsparse_handle      handle,
                                          rocsparse_spmat_descr descr,
                                          rocsparse_dnvec_descr value,
                                          rocsparse_error*      p_error);

#ifdef __cplusplus
}
#endif

#endif
