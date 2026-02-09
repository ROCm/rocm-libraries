/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TYPE_DISPATCH_HPP
#define TYPE_DISPATCH_HPP

#include "hipsparse_arguments.hpp"

// ----------------------------------------------------------------------------
// Calls TEST template based on the argument types. TEST<> is expected to
// return a functor which takes a const Arguments& argument. If the types do
// not match a recognized type combination, then TEST<void> is called.  This
// function returns the same type as TEST<...>{}(arg), usually bool or void.
// ----------------------------------------------------------------------------

// Simple functions which take only one datatype
//
// Even if the function can take mixed datatypes, this function can handle the
// cases where the types are uniform, in which case one template type argument
// is passed to TEST, and the rest are assumed to match the first.
template <template <typename...> class TEST>
auto hipsparse_simple_dispatch(const Arguments& arg)
{
    switch(arg.compute_type)
    {
    case HIP_R_32F:
        return TEST<float>{}(arg);
    case HIP_R_64F:
        return TEST<double>{}(arg);
    case HIP_C_32F:
        return TEST<hipComplex>{}(arg);
    case HIP_C_64F:
        return TEST<hipDoubleComplex>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto hipsparse_it_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    if(I == HIPSPARSE_INDEX_32I)
    {
        switch(arg.compute_type)
        {
        case HIP_R_32F:
            return TEST<int32_t, float>{}(arg);
        case HIP_R_64F:
            return TEST<int32_t, double>{}(arg);
        case HIP_C_32F:
            return TEST<int32_t, hipComplex>{}(arg);
        case HIP_C_64F:
            return TEST<int32_t, hipDoubleComplex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == HIPSPARSE_INDEX_64I)
    {
        switch(arg.compute_type)
        {
        case HIP_R_32F:
            return TEST<int64_t, float>{}(arg);
        case HIP_R_64F:
            return TEST<int64_t, double>{}(arg);
        case HIP_C_32F:
            return TEST<int64_t, hipComplex>{}(arg);
        case HIP_C_64F:
            return TEST<int64_t, hipDoubleComplex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto hipsparse_ijt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;

    if(I == HIPSPARSE_INDEX_32I && J == HIPSPARSE_INDEX_32I)
    {
        switch(arg.compute_type)
        {
        case HIP_R_32F:
            return TEST<int32_t, int32_t, float>{}(arg);
        case HIP_R_64F:
            return TEST<int32_t, int32_t, double>{}(arg);
        case HIP_C_32F:
            return TEST<int32_t, int32_t, hipComplex>{}(arg);
        case HIP_C_64F:
            return TEST<int32_t, int32_t, hipDoubleComplex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == HIPSPARSE_INDEX_64I && J == HIPSPARSE_INDEX_32I)
    {
        switch(arg.compute_type)
        {
        case HIP_R_32F:
            return TEST<int64_t, int32_t, float>{}(arg);
        case HIP_R_64F:
            return TEST<int64_t, int32_t, double>{}(arg);
        case HIP_C_32F:
            return TEST<int64_t, int32_t, hipComplex>{}(arg);
        case HIP_C_64F:
            return TEST<int64_t, int32_t, hipDoubleComplex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }
    else if(I == HIPSPARSE_INDEX_64I && J == HIPSPARSE_INDEX_64I)
    {
        switch(arg.compute_type)
        {
        case HIP_R_32F:
            return TEST<int64_t, int64_t, float>{}(arg);
        case HIP_R_64F:
            return TEST<int64_t, int64_t, double>{}(arg);
        case HIP_C_32F:
            return TEST<int64_t, int64_t, hipComplex>{}(arg);
        case HIP_C_64F:
            return TEST<int64_t, int64_t, hipDoubleComplex>{}(arg);
        default:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

#endif // TYPE_DISPATCH_HPP