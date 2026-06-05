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

#pragma once

#include "rocsparse-types.h"
#include "rocsparse_pivot_info_t.hpp"
#include "rocsparse_singular_info_t.hpp"

namespace rocsparse
{
    rocsparse_status singularity_get_position_async(rocsparse_handle                  handle,
                                                    int64_t                           batch_count,
                                                    const rocsparse::pivot_info_t*    symbolic,
                                                    const rocsparse::singular_info_t* exact,
                                                    const rocsparse::singular_info_t* near,
                                                    rocsparse_pointer_mode position_pointer_mode,
                                                    rocsparse_indextype    position_indextype,
                                                    void*                  position);
    rocsparse_status singularity_get_async(rocsparse_handle                  handle,
                                           int64_t                           batch_count,
                                           const rocsparse::pivot_info_t*    symbolic,
                                           const rocsparse::singular_info_t* exact,
                                           const rocsparse::singular_info_t* near,
                                           rocsparse_pointer_mode singularity_pointer_mode,
                                           void*                  singularities);

}
