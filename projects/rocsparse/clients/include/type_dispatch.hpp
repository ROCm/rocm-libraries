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

#include "rocsparse_arguments.hpp"

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
auto rocsparse_simple_dispatch(const Arguments& arg)
{
    switch(arg.compute_type)
    {
    case rocsparse_datatype_f32_r:
        return TEST<float>{}(arg);
    case rocsparse_datatype_f64_r:
        return TEST<double>{}(arg);
    case rocsparse_datatype_f32_c:
        return TEST<rocsparse_float_complex>{}(arg);
    case rocsparse_datatype_f64_c:
        return TEST<rocsparse_double_complex>{}(arg);
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    case rocsparse_datatype_f16_r:
    case rocsparse_datatype_bf16_r:
        return TEST<void>{}(arg);
    }
    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_it_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    if(I == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int32_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_it_gather_scatter_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    if(I == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_i8_r:
            return TEST<int32_t, int8_t>{}(arg);
        case rocsparse_datatype_f16_r:
            return TEST<int32_t, _Float16>{}(arg);
        case rocsparse_datatype_bf16_r:
            return TEST<int32_t, rocsparse_bfloat16>{}(arg);
        case rocsparse_datatype_f32_r:
            return TEST<int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int32_t, rocsparse_double_complex>{}(arg);

        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_i8_r:
            return TEST<int64_t, int8_t>{}(arg);
        case rocsparse_datatype_f16_r:
            return TEST<int64_t, _Float16>{}(arg);
        case rocsparse_datatype_bf16_r:
            return TEST<int64_t, rocsparse_bfloat16>{}(arg);
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_ijt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;

    if(I == rocsparse_indextype_i32 && J == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int32_t, int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int32_t, int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int32_t, int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int32_t, int32_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64 && J == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, int32_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64 && J == rocsparse_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, int64_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, int64_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, int64_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, int64_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_ixyt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;

    const auto X = arg.x_type;
    const auto Y = arg.y_type;

    const auto T = arg.compute_type;

    const bool f32r_case = (X == rocsparse_datatype_f32_r && X == Y && X == T);
    const bool f64r_case = (X == rocsparse_datatype_f64_r && X == Y && X == T);
    const bool f32c_case = (X == rocsparse_datatype_f32_c && X == Y && X == T);
    const bool f64c_case = (X == rocsparse_datatype_f64_c && X == Y && X == T);

    const bool i8r_i8r_i32r_case = (X == rocsparse_datatype_i8_r && Y == rocsparse_datatype_i8_r
                                    && T == rocsparse_datatype_i32_r);

    const bool i8r_i8r_f32r_case = (X == rocsparse_datatype_i8_r && Y == rocsparse_datatype_i8_r
                                    && T == rocsparse_datatype_f32_r);

    const bool f16r_f16r_f32r_case = (X == rocsparse_datatype_f16_r && Y == rocsparse_datatype_f16_r
                                      && T == rocsparse_datatype_f32_r);

    const bool bf16r_bf16r_f32r_case
        = (X == rocsparse_datatype_bf16_r && Y == rocsparse_datatype_bf16_r
           && T == rocsparse_datatype_f32_r);

#define DISPATCH_TEST(ITYPE)                                                      \
    if(f32r_case)                                                                 \
    {                                                                             \
        return TEST<ITYPE, float, float, float>{}(arg);                           \
    }                                                                             \
    else if(f64r_case)                                                            \
    {                                                                             \
        return TEST<ITYPE, double, double, double>{}(arg);                        \
    }                                                                             \
    else if(f32c_case)                                                            \
    {                                                                             \
        return TEST<ITYPE,                                                        \
                    rocsparse_float_complex,                                      \
                    rocsparse_float_complex,                                      \
                    rocsparse_float_complex>{}(arg);                              \
    }                                                                             \
    else if(f64c_case)                                                            \
    {                                                                             \
        return TEST<ITYPE,                                                        \
                    rocsparse_double_complex,                                     \
                    rocsparse_double_complex,                                     \
                    rocsparse_double_complex>{}(arg);                             \
    }                                                                             \
    else if(i8r_i8r_i32r_case)                                                    \
    {                                                                             \
        return TEST<ITYPE, int8_t, int8_t, int32_t>{}(arg);                       \
    }                                                                             \
    else if(i8r_i8r_f32r_case)                                                    \
    {                                                                             \
        return TEST<ITYPE, int8_t, int8_t, float>{}(arg);                         \
    }                                                                             \
    else if(f16r_f16r_f32r_case)                                                  \
    {                                                                             \
        return TEST<ITYPE, _Float16, _Float16, float>{}(arg);                     \
    }                                                                             \
    else if(bf16r_bf16r_f32r_case)                                                \
    {                                                                             \
        return TEST<ITYPE, rocsparse_bfloat16, rocsparse_bfloat16, float>{}(arg); \
    }

    switch(I)
    {
    case rocsparse_indextype_u16:
    {
        return TEST<void, void, void, void>{}(arg);
    }
    case rocsparse_indextype_i32:
    {
        DISPATCH_TEST(int32_t);
    }
    case rocsparse_indextype_i64:
    {
        DISPATCH_TEST(int64_t);
    }
    }
#undef DISPATCH_TEST

    return TEST<void, void, void, void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_ixyt_axpby_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;

    const auto X = arg.x_type;
    const auto Y = arg.y_type;

    const auto T = arg.compute_type;

    const bool f32r_case = (X == rocsparse_datatype_f32_r && X == Y && X == T);
    const bool f64r_case = (X == rocsparse_datatype_f64_r && X == Y && X == T);
    const bool f32c_case = (X == rocsparse_datatype_f32_c && X == Y && X == T);
    const bool f64c_case = (X == rocsparse_datatype_f64_c && X == Y && X == T);

    const bool f16r_f16r_f32r_case = (X == rocsparse_datatype_f16_r && Y == rocsparse_datatype_f16_r
                                      && T == rocsparse_datatype_f32_r);
    const bool bf16r_bf16r_f32r_case
        = (X == rocsparse_datatype_bf16_r && Y == rocsparse_datatype_bf16_r
           && T == rocsparse_datatype_f32_r);

#define DISPATCH_TEST(ITYPE)                                                      \
    if(f32r_case)                                                                 \
    {                                                                             \
        return TEST<ITYPE, float, float, float>{}(arg);                           \
    }                                                                             \
    else if(f64r_case)                                                            \
    {                                                                             \
        return TEST<ITYPE, double, double, double>{}(arg);                        \
    }                                                                             \
    else if(f32c_case)                                                            \
    {                                                                             \
        return TEST<ITYPE,                                                        \
                    rocsparse_float_complex,                                      \
                    rocsparse_float_complex,                                      \
                    rocsparse_float_complex>{}(arg);                              \
    }                                                                             \
    else if(f64c_case)                                                            \
    {                                                                             \
        return TEST<ITYPE,                                                        \
                    rocsparse_double_complex,                                     \
                    rocsparse_double_complex,                                     \
                    rocsparse_double_complex>{}(arg);                             \
    }                                                                             \
    else if(f16r_f16r_f32r_case)                                                  \
    {                                                                             \
        return TEST<ITYPE, _Float16, _Float16, float>{}(arg);                     \
    }                                                                             \
    else if(bf16r_bf16r_f32r_case)                                                \
    {                                                                             \
        return TEST<ITYPE, rocsparse_bfloat16, rocsparse_bfloat16, float>{}(arg); \
    }

    switch(I)
    {
    case rocsparse_indextype_u16:
    {
        return TEST<void, void, void, void>{}(arg);
    }
    case rocsparse_indextype_i32:
    {
        DISPATCH_TEST(int32_t);
    }
    case rocsparse_indextype_i64:
    {
        DISPATCH_TEST(int64_t);
    }
    }
#undef DISPATCH_TEST

    return TEST<void, void, void, void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_iaxyt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;

    const auto A = arg.a_type;
    const auto X = arg.x_type;
    const auto Y = arg.y_type;

    const auto T = arg.compute_type;

    const bool f32r_case = (A == rocsparse_datatype_f32_r && A == X && A == Y && A == T);
    const bool f64r_case = (A == rocsparse_datatype_f64_r && A == X && A == Y && A == T);
    const bool f32c_case = (A == rocsparse_datatype_f32_c && A == X && A == Y && A == T);
    const bool f64c_case = (A == rocsparse_datatype_f64_c && A == X && A == Y && A == T);

    const bool i8r_i8r_i32r_i32r_case
        = (A == rocsparse_datatype_i8_r && X == rocsparse_datatype_i8_r
           && Y == rocsparse_datatype_i32_r && T == rocsparse_datatype_i32_r);

    const bool i8r_i8r_f32r_f32r_case
        = (A == rocsparse_datatype_i8_r && X == rocsparse_datatype_i8_r
           && Y == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool f16r_f16r_f32r_f32r_case
        = (A == rocsparse_datatype_f16_r && X == rocsparse_datatype_f16_r
           && Y == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool bf16r_bf16r_f32r_f32r_case
        = (A == rocsparse_datatype_bf16_r && X == rocsparse_datatype_bf16_r
           && Y == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool f32r_f32c_f32c_f32c_case
        = (A == rocsparse_datatype_f32_r && X == rocsparse_datatype_f32_c
           && Y == rocsparse_datatype_f32_c && T == rocsparse_datatype_f32_c);

    const bool f64r_f64c_f64c_f64c_case
        = (A == rocsparse_datatype_f64_r && X == rocsparse_datatype_f64_c
           && Y == rocsparse_datatype_f64_c && T == rocsparse_datatype_f64_c);

#define DISPATCH_TEST(ITYPE)                                                             \
    if(f32r_case)                                                                        \
    {                                                                                    \
        return TEST<ITYPE, float, float, float, float>{}(arg);                           \
    }                                                                                    \
    else if(f64r_case)                                                                   \
    {                                                                                    \
        return TEST<ITYPE, double, double, double, double>{}(arg);                       \
    }                                                                                    \
    else if(f32c_case)                                                                   \
    {                                                                                    \
        return TEST<ITYPE,                                                               \
                    rocsparse_float_complex,                                             \
                    rocsparse_float_complex,                                             \
                    rocsparse_float_complex,                                             \
                    rocsparse_float_complex>{}(arg);                                     \
    }                                                                                    \
    else if(f64c_case)                                                                   \
    {                                                                                    \
        return TEST<ITYPE,                                                               \
                    rocsparse_double_complex,                                            \
                    rocsparse_double_complex,                                            \
                    rocsparse_double_complex,                                            \
                    rocsparse_double_complex>{}(arg);                                    \
    }                                                                                    \
    else if(i8r_i8r_i32r_i32r_case)                                                      \
    {                                                                                    \
        return TEST<ITYPE, int8_t, int8_t, int32_t, int32_t>{}(arg);                     \
    }                                                                                    \
    else if(i8r_i8r_f32r_f32r_case)                                                      \
    {                                                                                    \
        return TEST<ITYPE, int8_t, int8_t, float, float>{}(arg);                         \
    }                                                                                    \
    else if(f16r_f16r_f32r_f32r_case)                                                    \
    {                                                                                    \
        return TEST<ITYPE, _Float16, _Float16, float, float>{}(arg);                     \
    }                                                                                    \
    else if(bf16r_bf16r_f32r_f32r_case)                                                  \
    {                                                                                    \
        return TEST<ITYPE, rocsparse_bfloat16, rocsparse_bfloat16, float, float>{}(arg); \
    }                                                                                    \
    else if(f32r_f32c_f32c_f32c_case)                                                    \
    {                                                                                    \
        return TEST<ITYPE,                                                               \
                    float,                                                               \
                    rocsparse_float_complex,                                             \
                    rocsparse_float_complex,                                             \
                    rocsparse_float_complex>{}(arg);                                     \
    }                                                                                    \
    else if(f64r_f64c_f64c_f64c_case)                                                    \
    {                                                                                    \
        return TEST<ITYPE,                                                               \
                    double,                                                              \
                    rocsparse_double_complex,                                            \
                    rocsparse_double_complex,                                            \
                    rocsparse_double_complex>{}(arg);                                    \
    }

    switch(I)
    {
    case rocsparse_indextype_u16:
    {
        return TEST<void, void, void, void, void>{}(arg);
    }
    case rocsparse_indextype_i32:
    {
        DISPATCH_TEST(int32_t);
    }
    case rocsparse_indextype_i64:
    {
        DISPATCH_TEST(int64_t);
    }
    }
#undef DISPATCH_TEST

    return TEST<void, void, void, void, void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_ijaxyt_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;

    const auto A = arg.a_type;
    const auto X = arg.x_type;
    const auto Y = arg.y_type;

    const auto T = arg.compute_type;

    const bool f32r_case = (A == rocsparse_datatype_f32_r && A == X && A == Y && A == T);
    const bool f64r_case = (A == rocsparse_datatype_f64_r && A == X && A == Y && A == T);
    const bool f32c_case = (A == rocsparse_datatype_f32_c && A == X && A == Y && A == T);
    const bool f64c_case = (A == rocsparse_datatype_f64_c && A == X && A == Y && A == T);

    const bool i8r_i8r_i32r_i32r_case
        = (A == rocsparse_datatype_i8_r && X == rocsparse_datatype_i8_r
           && Y == rocsparse_datatype_i32_r && T == rocsparse_datatype_i32_r);

    const bool i8r_i8r_f32r_f32r_case
        = (A == rocsparse_datatype_i8_r && X == rocsparse_datatype_i8_r
           && Y == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool f16r_f16r_f32r_f32r_case
        = (A == rocsparse_datatype_f16_r && X == rocsparse_datatype_f16_r
           && Y == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool bf16r_bf16r_f32r_f32r_case
        = (A == rocsparse_datatype_bf16_r && X == rocsparse_datatype_bf16_r
           && Y == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool f32r_f32c_f32c_f32c_case
        = (A == rocsparse_datatype_f32_r && X == rocsparse_datatype_f32_c
           && Y == rocsparse_datatype_f32_c && T == rocsparse_datatype_f32_c);

    const bool f64r_f64c_f64c_f64c_case
        = (A == rocsparse_datatype_f64_r && X == rocsparse_datatype_f64_c
           && Y == rocsparse_datatype_f64_c && T == rocsparse_datatype_f64_c);

#define DISPATCH_TEST(ITYPE, JTYPE)                                                             \
    if(f32r_case)                                                                               \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, float, float, float, float>{}(arg);                           \
    }                                                                                           \
    else if(f64r_case)                                                                          \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, double, double, double, double>{}(arg);                       \
    }                                                                                           \
    else if(f32c_case)                                                                          \
    {                                                                                           \
        return TEST<ITYPE,                                                                      \
                    JTYPE,                                                                      \
                    rocsparse_float_complex,                                                    \
                    rocsparse_float_complex,                                                    \
                    rocsparse_float_complex,                                                    \
                    rocsparse_float_complex>{}(arg);                                            \
    }                                                                                           \
    else if(f64c_case)                                                                          \
    {                                                                                           \
        return TEST<ITYPE,                                                                      \
                    JTYPE,                                                                      \
                    rocsparse_double_complex,                                                   \
                    rocsparse_double_complex,                                                   \
                    rocsparse_double_complex,                                                   \
                    rocsparse_double_complex>{}(arg);                                           \
    }                                                                                           \
    else if(i8r_i8r_i32r_i32r_case)                                                             \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, int8_t, int8_t, int32_t, int32_t>{}(arg);                     \
    }                                                                                           \
    else if(i8r_i8r_f32r_f32r_case)                                                             \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, int8_t, int8_t, float, float>{}(arg);                         \
    }                                                                                           \
    else if(f16r_f16r_f32r_f32r_case)                                                           \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, _Float16, _Float16, float, float>{}(arg);                     \
    }                                                                                           \
    else if(bf16r_bf16r_f32r_f32r_case)                                                         \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, rocsparse_bfloat16, rocsparse_bfloat16, float, float>{}(arg); \
    }                                                                                           \
    else if(f32r_f32c_f32c_f32c_case)                                                           \
    {                                                                                           \
        return TEST<ITYPE,                                                                      \
                    JTYPE,                                                                      \
                    float,                                                                      \
                    rocsparse_float_complex,                                                    \
                    rocsparse_float_complex,                                                    \
                    rocsparse_float_complex>{}(arg);                                            \
    }                                                                                           \
    else if(f64r_f64c_f64c_f64c_case)                                                           \
    {                                                                                           \
        return TEST<ITYPE,                                                                      \
                    JTYPE,                                                                      \
                    double,                                                                     \
                    rocsparse_double_complex,                                                   \
                    rocsparse_double_complex,                                                   \
                    rocsparse_double_complex>{}(arg);                                           \
    }

    switch(I)
    {
    case rocsparse_indextype_u16:
    {
        return TEST<void, void, void, void, void, void>{}(arg);
    }
    case rocsparse_indextype_i32:
    {
        switch(J)
        {
        case rocsparse_indextype_u16:
        case rocsparse_indextype_i64:
        {
            return TEST<void, void, void, void, void, void>{}(arg);
        }
        case rocsparse_indextype_i32:
        {
            DISPATCH_TEST(int32_t, int32_t);
        }
        }
    }
    case rocsparse_indextype_i64:
    {
        switch(J)
        {
        case rocsparse_indextype_u16:
        {
            return TEST<void, void, void, void, void, void>{}(arg);
        }
        case rocsparse_indextype_i32:
        {
            DISPATCH_TEST(int64_t, int32_t);
        }
        case rocsparse_indextype_i64:
        {
            DISPATCH_TEST(int64_t, int64_t);
        }
        }
    }
    }
#undef DISPATCH_TEST

    return TEST<void, void, void, void, void, void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_iabct_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;

    const auto A = arg.a_type;
    const auto B = arg.b_type;
    const auto C = arg.c_type;

    const auto T = arg.compute_type;

    const bool f32r_case = (A == rocsparse_datatype_f32_r && A == B && A == C && A == T);
    const bool f64r_case = (A == rocsparse_datatype_f64_r && A == B && A == C && A == T);
    const bool f32c_case = (A == rocsparse_datatype_f32_c && A == B && A == C && A == T);
    const bool f64c_case = (A == rocsparse_datatype_f64_c && A == B && A == C && A == T);

    const bool i8r_i8r_i32r_i32r_case
        = (A == rocsparse_datatype_i8_r && B == rocsparse_datatype_i8_r
           && C == rocsparse_datatype_i32_r && T == rocsparse_datatype_i32_r);

    const bool i8r_i8r_f32r_f32r_case
        = (A == rocsparse_datatype_i8_r && B == rocsparse_datatype_i8_r
           && C == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool f16r_f16r_f32r_f32r_case
        = (A == rocsparse_datatype_f16_r && B == rocsparse_datatype_f16_r
           && C == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool bf16r_bf16r_f32r_f32r_case
        = (A == rocsparse_datatype_bf16_r && B == rocsparse_datatype_bf16_r
           && C == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

#define DISPATCH_TEST(ITYPE)                                                             \
    if(f32r_case)                                                                        \
    {                                                                                    \
        return TEST<ITYPE, float, float, float, float>{}(arg);                           \
    }                                                                                    \
    else if(f64r_case)                                                                   \
    {                                                                                    \
        return TEST<ITYPE, double, double, double, double>{}(arg);                       \
    }                                                                                    \
    else if(f32c_case)                                                                   \
    {                                                                                    \
        return TEST<ITYPE,                                                               \
                    rocsparse_float_complex,                                             \
                    rocsparse_float_complex,                                             \
                    rocsparse_float_complex,                                             \
                    rocsparse_float_complex>{}(arg);                                     \
    }                                                                                    \
    else if(f64c_case)                                                                   \
    {                                                                                    \
        return TEST<ITYPE,                                                               \
                    rocsparse_double_complex,                                            \
                    rocsparse_double_complex,                                            \
                    rocsparse_double_complex,                                            \
                    rocsparse_double_complex>{}(arg);                                    \
    }                                                                                    \
    else if(i8r_i8r_i32r_i32r_case)                                                      \
    {                                                                                    \
        return TEST<ITYPE, int8_t, int8_t, int32_t, int32_t>{}(arg);                     \
    }                                                                                    \
    else if(i8r_i8r_f32r_f32r_case)                                                      \
    {                                                                                    \
        return TEST<ITYPE, int8_t, int8_t, float, float>{}(arg);                         \
    }                                                                                    \
    else if(f16r_f16r_f32r_f32r_case)                                                    \
    {                                                                                    \
        return TEST<ITYPE, _Float16, _Float16, float, float>{}(arg);                     \
    }                                                                                    \
    else if(bf16r_bf16r_f32r_f32r_case)                                                  \
    {                                                                                    \
        return TEST<ITYPE, rocsparse_bfloat16, rocsparse_bfloat16, float, float>{}(arg); \
    }

    switch(I)
    {
    case rocsparse_indextype_u16:
    {
        return TEST<void, void, void, void, void>{}(arg);
    }
    case rocsparse_indextype_i32:
    {
        DISPATCH_TEST(int32_t);
    }
    case rocsparse_indextype_i64:
    {
        DISPATCH_TEST(int64_t);
    }
    }
#undef DISPATCH_TEST

    return TEST<void, void, void, void, void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_ijabct_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;

    const auto A = arg.a_type;
    const auto B = arg.b_type;
    const auto C = arg.c_type;

    const auto T = arg.compute_type;

    const bool f32r_case = (A == rocsparse_datatype_f32_r && A == B && A == C && A == T);
    const bool f64r_case = (A == rocsparse_datatype_f64_r && A == B && A == C && A == T);
    const bool f32c_case = (A == rocsparse_datatype_f32_c && A == B && A == C && A == T);
    const bool f64c_case = (A == rocsparse_datatype_f64_c && A == B && A == C && A == T);

    const bool i8r_i8r_i32r_i32r_case
        = (A == rocsparse_datatype_i8_r && B == rocsparse_datatype_i8_r
           && C == rocsparse_datatype_i32_r && T == rocsparse_datatype_i32_r);

    const bool i8r_i8r_f32r_f32r_case
        = (A == rocsparse_datatype_i8_r && B == rocsparse_datatype_i8_r
           && C == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool f16r_f16r_f32r_f32r_case
        = (A == rocsparse_datatype_f16_r && B == rocsparse_datatype_f16_r
           && C == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool bf16r_bf16r_f32r_f32r_case
        = (A == rocsparse_datatype_bf16_r && B == rocsparse_datatype_bf16_r
           && C == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

#define DISPATCH_TEST(ITYPE, JTYPE)                                                             \
    if(f32r_case)                                                                               \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, float, float, float, float>{}(arg);                           \
    }                                                                                           \
    else if(f64r_case)                                                                          \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, double, double, double, double>{}(arg);                       \
    }                                                                                           \
    else if(f32c_case)                                                                          \
    {                                                                                           \
        return TEST<ITYPE,                                                                      \
                    JTYPE,                                                                      \
                    rocsparse_float_complex,                                                    \
                    rocsparse_float_complex,                                                    \
                    rocsparse_float_complex,                                                    \
                    rocsparse_float_complex>{}(arg);                                            \
    }                                                                                           \
    else if(f64c_case)                                                                          \
    {                                                                                           \
        return TEST<ITYPE,                                                                      \
                    JTYPE,                                                                      \
                    rocsparse_double_complex,                                                   \
                    rocsparse_double_complex,                                                   \
                    rocsparse_double_complex,                                                   \
                    rocsparse_double_complex>{}(arg);                                           \
    }                                                                                           \
    else if(i8r_i8r_i32r_i32r_case)                                                             \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, int8_t, int8_t, int32_t, int32_t>{}(arg);                     \
    }                                                                                           \
    else if(i8r_i8r_f32r_f32r_case)                                                             \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, int8_t, int8_t, float, float>{}(arg);                         \
    }                                                                                           \
    else if(f16r_f16r_f32r_f32r_case)                                                           \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, _Float16, _Float16, float, float>{}(arg);                     \
    }                                                                                           \
    else if(bf16r_bf16r_f32r_f32r_case)                                                         \
    {                                                                                           \
        return TEST<ITYPE, JTYPE, rocsparse_bfloat16, rocsparse_bfloat16, float, float>{}(arg); \
    }

    switch(I)
    {
    case rocsparse_indextype_u16:
    {
        return TEST<void, void, void, void, void, void>{}(arg);
    }
    case rocsparse_indextype_i32:
    {
        switch(J)
        {
        case rocsparse_indextype_u16:
        case rocsparse_indextype_i64:
        {
            return TEST<void, void, void, void, void, void>{}(arg);
        }
        case rocsparse_indextype_i32:
        {
            DISPATCH_TEST(int32_t, int32_t);
        }
        }
    }
    case rocsparse_indextype_i64:
    {
        switch(J)
        {
        case rocsparse_indextype_u16:
        {
            return TEST<void, void, void, void, void, void>{}(arg);
        }
        case rocsparse_indextype_i32:
        {
            DISPATCH_TEST(int64_t, int32_t);
        }
        case rocsparse_indextype_i64:
        {
            DISPATCH_TEST(int64_t, int64_t);
        }
        }
    }
    }
#undef DISPATCH_TEST

    return TEST<void, void, void, void, void, void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_ijabct_sddmm_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;

    const auto A = arg.a_type;
    const auto B = arg.b_type;
    const auto C = arg.c_type;

    const auto T = arg.compute_type;

    const bool f16r_case = (A == rocsparse_datatype_f16_r && A == B && A == C && A == T);
    const bool f32r_case = (A == rocsparse_datatype_f32_r && A == B && A == C && A == T);
    const bool f64r_case = (A == rocsparse_datatype_f64_r && A == B && A == C && A == T);
    const bool f32c_case = (A == rocsparse_datatype_f32_c && A == B && A == C && A == T);
    const bool f64c_case = (A == rocsparse_datatype_f64_c && A == B && A == C && A == T);

    const bool f16r_f16r_f32r_f32r_case
        = (A == rocsparse_datatype_f16_r && B == rocsparse_datatype_f16_r
           && C == rocsparse_datatype_f32_r && T == rocsparse_datatype_f32_r);

    const bool f16r_f16r_f16r_f32r_case
        = (A == rocsparse_datatype_f16_r && B == rocsparse_datatype_f16_r
           && C == rocsparse_datatype_f16_r && T == rocsparse_datatype_f32_r);

#define DISPATCH_TEST(ITYPE, JTYPE)                                               \
    if(f16r_case)                                                                 \
    {                                                                             \
        return TEST<ITYPE, JTYPE, _Float16, _Float16, _Float16, _Float16>{}(arg); \
    }                                                                             \
    else if(f32r_case)                                                            \
    {                                                                             \
        return TEST<ITYPE, JTYPE, float, float, float, float>{}(arg);             \
    }                                                                             \
    else if(f64r_case)                                                            \
    {                                                                             \
        return TEST<ITYPE, JTYPE, double, double, double, double>{}(arg);         \
    }                                                                             \
    else if(f32c_case)                                                            \
    {                                                                             \
        return TEST<ITYPE,                                                        \
                    JTYPE,                                                        \
                    rocsparse_float_complex,                                      \
                    rocsparse_float_complex,                                      \
                    rocsparse_float_complex,                                      \
                    rocsparse_float_complex>{}(arg);                              \
    }                                                                             \
    else if(f64c_case)                                                            \
    {                                                                             \
        return TEST<ITYPE,                                                        \
                    JTYPE,                                                        \
                    rocsparse_double_complex,                                     \
                    rocsparse_double_complex,                                     \
                    rocsparse_double_complex,                                     \
                    rocsparse_double_complex>{}(arg);                             \
    }                                                                             \
    else if(f16r_f16r_f32r_f32r_case)                                             \
    {                                                                             \
        return TEST<ITYPE, JTYPE, _Float16, _Float16, float, float>{}(arg);       \
    }                                                                             \
    else if(f16r_f16r_f16r_f32r_case)                                             \
    {                                                                             \
        return TEST<ITYPE, JTYPE, _Float16, _Float16, _Float16, float>{}(arg);    \
    }

    switch(I)
    {
    case rocsparse_indextype_u16:
    {
        return TEST<void, void, void, void, void, void>{}(arg);
    }
    case rocsparse_indextype_i32:
    {
        switch(J)
        {
        case rocsparse_indextype_u16:
        case rocsparse_indextype_i64:
        {
            return TEST<void, void, void, void, void, void>{}(arg);
        }
        case rocsparse_indextype_i32:
        {
            DISPATCH_TEST(int32_t, int32_t);
        }
        }
    }
    case rocsparse_indextype_i64:
    {
        switch(J)
        {
        case rocsparse_indextype_u16:
        {
            return TEST<void, void, void, void, void, void>{}(arg);
        }
        case rocsparse_indextype_i32:
        {
            DISPATCH_TEST(int64_t, int32_t);
        }
        case rocsparse_indextype_i64:
        {
            DISPATCH_TEST(int64_t, int64_t);
        }
        }
    }
    }
#undef DISPATCH_TEST

    return TEST<void, void, void, void, void, void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_it_sparse_to_dense_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    if(I == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f16_r:
            return TEST<int32_t, _Float16>{}(arg);
        case rocsparse_datatype_bf16_r:
            return TEST<int32_t, rocsparse_bfloat16>{}(arg);
        case rocsparse_datatype_f32_r:
            return TEST<int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int32_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f16_r:
            return TEST<int64_t, _Float16>{}(arg);
        case rocsparse_datatype_bf16_r:
            return TEST<int64_t, rocsparse_bfloat16>{}(arg);
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_it_dense_to_sparse_dispatch(const Arguments& arg)
{
    return rocsparse_it_sparse_to_dense_dispatch<TEST>(arg);
}

template <template <typename...> class TEST>
auto rocsparse_ijt_sparse_to_dense_dispatch(const Arguments& arg)
{
    const auto I = arg.index_type_I;
    const auto J = arg.index_type_J;

    if(I == rocsparse_indextype_i32 && J == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f16_r:
            return TEST<int32_t, int32_t, _Float16>{}(arg);
        case rocsparse_datatype_bf16_r:
            return TEST<int32_t, int32_t, rocsparse_bfloat16>{}(arg);
        case rocsparse_datatype_f32_r:
            return TEST<int32_t, int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int32_t, int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int32_t, int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int32_t, int32_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64 && J == rocsparse_indextype_i32)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f16_r:
            return TEST<int64_t, int32_t, _Float16>{}(arg);
        case rocsparse_datatype_bf16_r:
            return TEST<int64_t, int32_t, rocsparse_bfloat16>{}(arg);
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, int32_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, int32_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, int32_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, int32_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
            return TEST<void>{}(arg);
        }
    }
    else if(I == rocsparse_indextype_i64 && J == rocsparse_indextype_i64)
    {
        switch(arg.compute_type)
        {
        case rocsparse_datatype_f16_r:
            return TEST<int64_t, int64_t, _Float16>{}(arg);
        case rocsparse_datatype_bf16_r:
            return TEST<int64_t, int64_t, rocsparse_bfloat16>{}(arg);
        case rocsparse_datatype_f32_r:
            return TEST<int64_t, int64_t, float>{}(arg);
        case rocsparse_datatype_f64_r:
            return TEST<int64_t, int64_t, double>{}(arg);
        case rocsparse_datatype_f32_c:
            return TEST<int64_t, int64_t, rocsparse_float_complex>{}(arg);
        case rocsparse_datatype_f64_c:
            return TEST<int64_t, int64_t, rocsparse_double_complex>{}(arg);
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
            return TEST<void>{}(arg);
        }
    }

    return TEST<void>{}(arg);
}

template <template <typename...> class TEST>
auto rocsparse_ijt_dense_to_sparse_dispatch(const Arguments& arg)
{
    return rocsparse_ijt_sparse_to_dense_dispatch<TEST>(arg);
}

#endif // TYPE_DISPATCH_HPP
