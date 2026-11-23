/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsparse_singularity.hpp"

#include <limits>

namespace rocsparse
{
    // a and b are non-negative.
    inline bool size_product_overflows(int64_t a, int64_t b)
    {
        return a > 0 && b > std::numeric_limits<int64_t>::max() / a;
    }

    // A BSR descriptor stores block counts in rows and cols, so compare in
    // scalar units. Dividing the dense size cannot overflow, unlike
    // multiplying the block counts by block_dim.
    inline bool spmat_dnmat_dims_differ(rocsparse_const_spmat_descr sparse,
                                        rocsparse_const_dnmat_descr dense)
    {
        if(sparse->format == rocsparse_format_bsr)
        {
            const int64_t block_dim = sparse->block_dim;
            return (dense->rows % block_dim) != 0 || (dense->cols % block_dim) != 0
                   || dense->rows / block_dim != sparse->rows
                   || dense->cols / block_dim != sparse->cols;
        }
        return sparse->rows != dense->rows || sparse->cols != dense->cols;
    }

    template <typename T>
    inline size_t align_size(size_t size)
    {
        return ((sizeof(T) * size + 255) / 256) * 256;
    }

    template <uint32_t VALUE = 256>
    inline size_t align_size(const size_t size)
    {
        return ((size - 1) / VALUE + 1) * VALUE;
    }

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
