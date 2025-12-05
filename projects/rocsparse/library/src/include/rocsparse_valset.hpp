/*! \file */
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

#pragma once

#include "rocsparse-types.h"
#include "rocsparse_control.hpp"
#include "rocsparse_indextype_utils.hpp"

namespace rocsparse
{
    rocsparse_status valset(rocsparse_handle    handle,
                            int64_t             length,
                            int64_t             value,
                            rocsparse_indextype array_indextype,
                            void*               array);

    template <typename T>
    inline rocsparse_status valset(rocsparse_handle handle, int64_t length, T value, T* array)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::valset(handle, length, value, rocsparse::get_indextype<T>(), array));
        return rocsparse_status_success;
    }

    template <typename T>
    inline rocsparse_status valset(rocsparse_handle handle, int32_t length, T value, T* array)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::valset(handle, length, value, rocsparse::get_indextype<T>(), array));
        return rocsparse_status_success;
    }

}
