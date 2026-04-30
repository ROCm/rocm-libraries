/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

template <template <typename...> class TEST>
auto hipsparse_spmv_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;
    const auto A = arg.a_type;
    const auto X = arg.x_type;
    const auto Y = arg.y_type;
    const auto T = arg.compute_type;

#define HIPSPARSE_SPMV_DISPATCH_BLOCK(I_t, J_t)                                                    \
    do                                                                                             \
    {                                                                                              \
        /* Uniform precisions: A == X == Y == T */                                                 \
        if(A == HIP_R_32F && X == HIP_R_32F && Y == HIP_R_32F && T == HIP_R_32F)                   \
            return TEST<I_t, J_t, float, float, float, float>{}(arg);                              \
        if(A == HIP_R_64F && X == HIP_R_64F && Y == HIP_R_64F && T == HIP_R_64F)                   \
            return TEST<I_t, J_t, double, double, double, double>{}(arg);                          \
        if(A == HIP_C_32F && X == HIP_C_32F && Y == HIP_C_32F && T == HIP_C_32F)                   \
            return TEST<I_t, J_t, hipComplex, hipComplex, hipComplex, hipComplex>{}(arg);          \
        if(A == HIP_C_64F && X == HIP_C_64F && Y == HIP_C_64F && T == HIP_C_64F)                   \
            return TEST<I_t,                                                                       \
                        J_t,                                                                       \
                        hipDoubleComplex,                                                          \
                        hipDoubleComplex,                                                          \
                        hipDoubleComplex,                                                          \
                        hipDoubleComplex>{}(arg);                                                  \
        /* Basic mixed precisions: A == X, Y / T may differ */                                     \
        if(A == HIP_R_8I && X == HIP_R_8I && Y == HIP_R_32I && T == HIP_R_32I)                     \
            return TEST<I_t, J_t, int8_t, int8_t, int32_t, int32_t>{}(arg);                        \
        if(A == HIP_R_8I && X == HIP_R_8I && Y == HIP_R_32F && T == HIP_R_32F)                     \
            return TEST<I_t, J_t, int8_t, int8_t, float, float>{}(arg);                            \
        if(A == HIP_R_16F && X == HIP_R_16F && Y == HIP_R_32F && T == HIP_R_32F)                   \
            return TEST<I_t, J_t, hipsparseFloat16, hipsparseFloat16, float, float>{}(arg);        \
        if(A == HIP_R_16F && X == HIP_R_16F && Y == HIP_R_16F && T == HIP_R_32F)                   \
            return TEST<I_t, J_t, hipsparseFloat16, hipsparseFloat16, hipsparseFloat16, float>{}(  \
                arg);                                                                              \
        if(A == HIP_R_16BF && X == HIP_R_16BF && Y == HIP_R_32F && T == HIP_R_32F)                 \
            return TEST<I_t, J_t, hipsparseBfloat16, hipsparseBfloat16, float, float>{}(arg);      \
        if(A == HIP_R_16BF && X == HIP_R_16BF && Y == HIP_R_16BF && T == HIP_R_32F)                \
            return TEST<I_t,                                                                       \
                        J_t,                                                                       \
                        hipsparseBfloat16,                                                         \
                        hipsparseBfloat16,                                                         \
                        hipsparseBfloat16,                                                         \
                        float>{}(arg);                                                             \
        if(A == HIP_R_32F && X == HIP_R_64F && Y == HIP_R_64F && T == HIP_R_64F)                   \
            return TEST<I_t, J_t, float, double, double, double>{}(arg);                           \
        if(A == HIP_C_32F && X == HIP_C_64F && Y == HIP_C_64F && T == HIP_C_64F)                   \
            return TEST<I_t,                                                                       \
                        J_t,                                                                       \
                        hipComplex,                                                                \
                        hipDoubleComplex,                                                          \
                        hipDoubleComplex,                                                          \
                        hipDoubleComplex>{}(arg);                                                  \
        if(A == HIP_R_32F && X == HIP_C_32F && Y == HIP_C_32F && T == HIP_C_32F)                   \
            return TEST<I_t, J_t, float, hipComplex, hipComplex, hipComplex>{}(arg);               \
        if(A == HIP_R_64F && X == HIP_C_64F && Y == HIP_C_64F && T == HIP_C_64F)                   \
            return TEST<I_t, J_t, double, hipDoubleComplex, hipDoubleComplex, hipDoubleComplex>{}( \
                arg);                                                                              \
        return TEST<void>{}(arg);                                                                  \
    } while(0)

    if(I == HIPSPARSE_INDEX_32I && J == HIPSPARSE_INDEX_32I)
    {
        HIPSPARSE_SPMV_DISPATCH_BLOCK(int32_t, int32_t);
    }
    else if(I == HIPSPARSE_INDEX_64I && J == HIPSPARSE_INDEX_32I)
    {
        HIPSPARSE_SPMV_DISPATCH_BLOCK(int64_t, int32_t);
    }
    else if(I == HIPSPARSE_INDEX_64I && J == HIPSPARSE_INDEX_64I)
    {
        HIPSPARSE_SPMV_DISPATCH_BLOCK(int64_t, int64_t);
    }

#undef HIPSPARSE_SPMV_DISPATCH_BLOCK

    return TEST<void>{}(arg);
}

#endif // TYPE_DISPATCH_HPP