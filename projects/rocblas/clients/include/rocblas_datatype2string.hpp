/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "rocblas.h"
#include <string>

// these enum should have unique value names including those defined in the rocblas library
// API enums as yaml data file processing doesn't have class scoping so can get confused to values

enum class rocblas_initialization
{
    rand_int          = 111,
    trig_float        = 222,
    hpl               = 333,
    denorm            = 444,
    denorm2           = 555,
    rand_int_zero_one = 666,
    zero              = 777,
};

/* ============================================================================================ */
/*  Convert rocblas constants to lapack char. */

constexpr auto rocblas2char_operation(rocblas_operation value)
{
    switch(value)
    {
    case rocblas_operation_none:
        return 'N';
    case rocblas_operation_transpose:
        return 'T';
    case rocblas_operation_conjugate_transpose:
        return 'C';
    }
    return '\0';
}

constexpr auto rocblas2char_fill(rocblas_fill value)
{
    switch(value)
    {
    case rocblas_fill_upper:
        return 'U';
    case rocblas_fill_lower:
        return 'L';
    case rocblas_fill_full:
        return 'F';
    }
    return '\0';
}

constexpr auto rocblas2char_diagonal(rocblas_diagonal value)
{
    switch(value)
    {
    case rocblas_diagonal_unit:
        return 'U';
    case rocblas_diagonal_non_unit:
        return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_side(rocblas_side value)
{
    switch(value)
    {
    case rocblas_side_left:
        return 'L';
    case rocblas_side_right:
        return 'R';
    case rocblas_side_both:
        return 'B';
    }
    return '\0';
}

// return precision string for rocblas_datatype
constexpr auto rocblas_datatype2string(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:
        return "f16_r";
    case rocblas_datatype_f32_r:
        return "f32_r";
    case rocblas_datatype_f64_r:
        return "f64_r";
    case rocblas_datatype_f16_c:
        return "f16_k";
    case rocblas_datatype_f32_c:
        return "f32_c";
    case rocblas_datatype_f64_c:
        return "f64_c";
    case rocblas_datatype_i8_r:
        return "i8_r";
    case rocblas_datatype_u8_r:
        return "u8_r";
    case rocblas_datatype_i32_r:
        return "i32_r";
    case rocblas_datatype_u32_r:
        return "u32_r";
    case rocblas_datatype_i8_c:
        return "i8_c";
    case rocblas_datatype_u8_c:
        return "u8_c";
    case rocblas_datatype_i32_c:
        return "i32_c";
    case rocblas_datatype_u32_c:
        return "u32_c";
    case rocblas_datatype_bf16_r:
        return "bf16_r";
    case rocblas_datatype_bf16_c:
        return "bf16_c";
    case rocblas_datatype_invalid:
        return "invalid";
    }
    return "invalid";
}

constexpr auto rocblas_initialization2string(rocblas_initialization init)
{
    switch(init)
    {
    case rocblas_initialization::rand_int:
        return "rand_int";
    case rocblas_initialization::trig_float:
        return "trig_float";
    case rocblas_initialization::hpl:
        return "hpl";
    case rocblas_initialization::denorm:
        return "denorm";
    case rocblas_initialization::denorm2:
        return "denorm2";
    case rocblas_initialization::rand_int_zero_one:
        return "rand_int_zero_one";
    case rocblas_initialization::zero:
        return "zero";
    }
    return "invalid";
}

inline rocblas_internal_ostream& operator<<(rocblas_internal_ostream& os,
                                            rocblas_initialization    init)
{
    return os << rocblas_initialization2string(init);
}

inline rocblas_internal_ostream& operator<<(rocblas_internal_ostream& os, uint8_t val)
{
    return os << (unsigned int)(val); // avoid 0 btye in stream as passed to gtest
}

inline rocblas_internal_ostream& operator<<(rocblas_internal_ostream& os, int8_t val)
{
    return os << (int)(val); // avoid 0 btye in stream as passed to gtest
}

// these next two << instantiations for std::pair simply allow the enums to be logged without quotes
// like the rocblas API enums

inline rocblas_internal_ostream& operator<<(rocblas_internal_ostream&                      os,
                                            std::pair<char const*, rocblas_initialization> p)
{
    os << p.first << ": ";
#define CASE(x) \
    case x:     \
        return os << rocblas_initialization2string(x)
    switch(p.second)
    {
        CASE(rocblas_initialization::rand_int);
        CASE(rocblas_initialization::trig_float);
        CASE(rocblas_initialization::hpl);
        CASE(rocblas_initialization::denorm);
        CASE(rocblas_initialization::denorm2);
        CASE(rocblas_initialization::rand_int_zero_one);
        CASE(rocblas_initialization::zero);
    }
    return os << "invalid";
}
#undef CASE

/* ============================================================================================ */
/*  Convert lapack char constants to rocblas type. */

constexpr rocblas_operation char2rocblas_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n':
        return rocblas_operation_none;
    case 'T':
    case 't':
        return rocblas_operation_transpose;
    case 'C':
    case 'c':
        return rocblas_operation_conjugate_transpose;
    default:
        return static_cast<rocblas_operation>(0); // zero not in enum
    }
}

constexpr rocblas_fill char2rocblas_fill(char value)
{
    switch(value)
    {
    case 'U':
    case 'u':
        return rocblas_fill_upper;
    case 'L':
    case 'l':
        return rocblas_fill_lower;
    default:
        return static_cast<rocblas_fill>(0); // zero not in enum
    }
}

constexpr rocblas_diagonal char2rocblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
    case 'u':
        return rocblas_diagonal_unit;
    case 'N':
    case 'n':
        return rocblas_diagonal_non_unit;
    default:
        return static_cast<rocblas_diagonal>(0); // zero not in enum
    }
}

constexpr rocblas_side char2rocblas_side(char value)
{
    switch(value)
    {
    case 'L':
    case 'l':
        return rocblas_side_left;
    case 'R':
    case 'r':
        return rocblas_side_right;
    default:
        return static_cast<rocblas_side>(0); // zero not in enum
    }
}

// clang-format off
inline rocblas_initialization string2rocblas_initialization(const std::string& value)
{
    return
        value == "rand_int"   ? rocblas_initialization::rand_int   :
        value == "trig_float" ? rocblas_initialization::trig_float :
        value == "hpl"        ? rocblas_initialization::hpl        :
        value == "denorm"     ? rocblas_initialization::denorm     :
        value == "denorm2"    ? rocblas_initialization::denorm2    :
        value == "rand_int_zero_one"    ? rocblas_initialization::rand_int_zero_one        :
        value == "zero"       ? rocblas_initialization::zero       :
        static_cast<rocblas_initialization>(0); // zero not in enum
}

inline rocblas_datatype string2rocblas_datatype(const std::string& value)
{
    return
        value == "f16_r" || value == "h" ? rocblas_datatype_f16_r  :
        value == "f32_r" || value == "s" ? rocblas_datatype_f32_r  :
        value == "f64_r" || value == "d" ? rocblas_datatype_f64_r  :
        value == "bf16_r"                ? rocblas_datatype_bf16_r :
        value == "f16_c"                 ? rocblas_datatype_f16_c  :
        value == "f32_c" || value == "c" ? rocblas_datatype_f32_c  :
        value == "f64_c" || value == "z" ? rocblas_datatype_f64_c  :
        value == "bf16_c"                ? rocblas_datatype_bf16_c :
        value == "i8_r"                  ? rocblas_datatype_i8_r   :
        value == "i32_r"                 ? rocblas_datatype_i32_r  :
        value == "i8_c"                  ? rocblas_datatype_i8_c   :
        value == "i32_c"                 ? rocblas_datatype_i32_c  :
        value == "u8_r"                  ? rocblas_datatype_u8_r   :
        value == "u32_r"                 ? rocblas_datatype_u32_r  :
        value == "u8_c"                  ? rocblas_datatype_u8_c   :
        value == "u32_c"                 ? rocblas_datatype_u32_c  :
        rocblas_datatype_invalid;
}

// clang-format on
