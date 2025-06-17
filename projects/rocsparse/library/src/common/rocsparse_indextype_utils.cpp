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
#include "rocsparse_indextype_utils.hpp"

template <>
rocsparse_indextype rocsparse::get_indextype<int32_t>()
{
    return rocsparse_indextype_i32;
}

template <>
rocsparse_indextype rocsparse::get_indextype<uint16_t>()
{
    return rocsparse_indextype_u16;
}

template <>
rocsparse_indextype rocsparse::get_indextype<int64_t>()
{
    return rocsparse_indextype_i64;
}

size_t rocsparse::indextype_sizeof(rocsparse_indextype that)
{
    switch(that)
    {

    case rocsparse_indextype_i32:
    {
        return sizeof(int32_t);
    }
    case rocsparse_indextype_i64:
    {
        return sizeof(int64_t);
    }
    case rocsparse_indextype_u16:
    {
        return sizeof(uint16_t);
    }
    }
}
