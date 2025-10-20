/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_floating_data_t.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_logging.hpp"
#include "rocsparse_memstat.hpp"
#include "rocsparse_scalar.hpp"

namespace rocsparse
{

// Return the leftmost significant bit position
#if defined(rocsparse_ILP64)
    static inline rocsparse_int clz(rocsparse_int n)
    {
        // __builtin_clzll is undefined for n == 0
        if(n == 0)
        {
            return 0;
        }
        return 64 - __builtin_clzll(n);
    }
#else
    static inline rocsparse_int clz(rocsparse_int n)
    {
        // __builtin_clz is undefined for n == 0
        if(n == 0)
        {
            return 0;
        }
        return 32 - __builtin_clz(n);
    }
#endif

}
