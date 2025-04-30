/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "control.h"
#include "utility.h"

namespace rocsparse
{
    template <typename T>
    rocsparse_status gtsv_no_pivot_buffer_size_template(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        rocsparse_int    n,
                                                        const T*         dl,
                                                        const T*         d,
                                                        const T*         du,
                                                        const T*         B,
                                                        rocsparse_int    ldb,
                                                        size_t*          buffer_size);

    template <typename T>
    rocsparse_status gtsv_no_pivot_template(rocsparse_handle handle,
                                            rocsparse_int    m,
                                            rocsparse_int    n,
                                            const T*         dl,
                                            const T*         d,
                                            const T*         du,
                                            T*               B,
                                            rocsparse_int    ldb,
                                            void*            temp_buffer);
}
