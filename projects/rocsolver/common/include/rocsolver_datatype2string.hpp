/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#include <string>

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_BEGIN_NAMESPACE
#endif

#define ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES \
    (ROCBLAS_VERSION_MAJOR >= 4 || (ROCBLAS_VERSION_MAJOR == 3 && ROCBLAS_VERSION_MINOR >= 1))

// return char from type
template <typename>
static constexpr char rocblas2char_precision = '\0';
template <>
static constexpr auto rocblas2char_precision<float> = 's';
template <>
static constexpr auto rocblas2char_precision<double> = 'd';
template <>
static constexpr auto rocblas2char_precision<rocblas_float_complex> = 'c';
template <>
static constexpr auto rocblas2char_precision<rocblas_double_complex> = 'z';

/* ============================================================================================
 */
/*  Convert rocblas constants to lapack char. */

constexpr auto rocblas2char_operation(rocblas_operation value)
{
    switch(value)
    {
    case rocblas_operation_none: return 'N';
    case rocblas_operation_transpose: return 'T';
    case rocblas_operation_conjugate_transpose: return 'C';
    }
    return '\0';
}

constexpr auto rocblas2char_fill(rocblas_fill value)
{
    switch(value)
    {
    case rocblas_fill_upper: return 'U';
    case rocblas_fill_lower: return 'L';
    case rocblas_fill_full: return 'F';
    }
    return '\0';
}

constexpr auto rocblas2char_diagonal(rocblas_diagonal value)
{
    switch(value)
    {
    case rocblas_diagonal_unit: return 'U';
    case rocblas_diagonal_non_unit: return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_side(rocblas_side value)
{
    switch(value)
    {
    case rocblas_side_left: return 'L';
    case rocblas_side_right: return 'R';
    case rocblas_side_both: return 'B';
    }
    return '\0';
}

constexpr auto rocblas2char_direct(rocblas_direct value)
{
    switch(value)
    {
    case rocblas_forward_direction: return 'F';
    case rocblas_backward_direction: return 'B';
    }
    return '\0';
}

constexpr auto rocblas2char_pivot(rocblas_pivot value)
{
    switch(value)
    {
    case rocblas_pivot_variable: return 'V';
    case rocblas_pivot_top: return 'T';
    case rocblas_pivot_bottom: return 'B';
    }
    return '\0';
}

constexpr auto rocblas2char_storev(rocblas_storev value)
{
    switch(value)
    {
    case rocblas_column_wise: return 'C';
    case rocblas_row_wise: return 'R';
    }
    return '\0';
}

constexpr auto rocblas2char_workmode(rocblas_workmode value)
{
    switch(value)
    {
    case rocblas_outofplace: return 'O';
    case rocblas_inplace: return 'I';
    }
    return '\0';
}

constexpr auto rocblas2char_svect(rocblas_svect value, bool use_V = false)
{
    switch(value)
    {
    case rocblas_svect_all: return 'A';
    case rocblas_svect_singular: return (use_V ? 'V' : 'S');
    case rocblas_svect_overwrite: return 'O';
    case rocblas_svect_none: return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_srange(rocblas_srange value)
{
    switch(value)
    {
    case rocblas_srange_all: return 'A';
    case rocblas_srange_value: return 'V';
    case rocblas_srange_index: return 'I';
    }
    return '\0';
}

constexpr auto rocblas2char_evect(rocblas_evect value)
{
    switch(value)
    {
    case rocblas_evect_original: return 'V';
    case rocblas_evect_tridiagonal: return 'I';
    case rocblas_evect_none: return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_eform(rocblas_eform value)
{
    switch(value)
    {
    case rocblas_eform_ax: return '1';
    case rocblas_eform_abx: return '2';
    case rocblas_eform_bax: return '3';
    }
    return '\0';
}

constexpr auto rocblas2char_erange(rocblas_erange value)
{
    switch(value)
    {
    case rocblas_erange_all: return 'A';
    case rocblas_erange_value: return 'V';
    case rocblas_erange_index: return 'I';
    }
    return '\0';
}

constexpr auto rocblas2char_eorder(rocblas_eorder value)
{
    switch(value)
    {
    case rocblas_eorder_blocks: return 'B';
    case rocblas_eorder_entire: return 'E';
    }
    return '\0';
}

constexpr auto rocblas2char_esort(rocblas_esort value)
{
    switch(value)
    {
    case rocblas_esort_none: return 'N';
    case rocblas_esort_ascending: return 'A';
    }
    return '\0';
}

constexpr auto rocblas2string_datatype(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r: return "f16_r";
    case rocblas_datatype_f32_r: return "f32_r";
    case rocblas_datatype_f64_r: return "f64_r";
    case rocblas_datatype_f16_c: return "f16_k";
    case rocblas_datatype_f32_c: return "f32_c";
    case rocblas_datatype_f64_c: return "f64_c";
    case rocblas_datatype_i8_r: return "i8_r";
    case rocblas_datatype_u8_r: return "u8_r";
    case rocblas_datatype_i32_r: return "i32_r";
    case rocblas_datatype_u32_r: return "u32_r";
    case rocblas_datatype_i8_c: return "i8_c";
    case rocblas_datatype_u8_c: return "u8_c";
    case rocblas_datatype_i32_c: return "i32_c";
    case rocblas_datatype_u32_c: return "u32_c";
    case rocblas_datatype_bf16_r: return "bf16_r";
    case rocblas_datatype_bf16_c: return "bf16_c";
    case rocblas_datatype_invalid: return "invalid";
#if ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES
    case rocblas_datatype_f8_r: return "f8_r";
    case rocblas_datatype_bf8_r: return "bf8_r";
#endif
    }
    return "invalid";
}

constexpr auto rocsolver2char_rfinfo_mode(rocsolver_rfinfo_mode value)
{
    switch(value)
    {
    case rocsolver_rfinfo_mode_lu: return '1';
    case rocsolver_rfinfo_mode_cholesky: return '2';
    }
    return '\0';
}

/* ============================================================================================
 */
/*  Convert lapack char constants to rocblas type. */

constexpr rocblas_operation char2rocblas_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n': return rocblas_operation_none;
    case 'T':
    case 't': return rocblas_operation_transpose;
    case 'C':
    case 'c': return rocblas_operation_conjugate_transpose;
    default: return static_cast<rocblas_operation>(0);
    }
}

constexpr rocblas_fill char2rocblas_fill(char value)
{
    switch(value)
    {
    case 'U':
    case 'u': return rocblas_fill_upper;
    case 'L':
    case 'l': return rocblas_fill_lower;
    default: return static_cast<rocblas_fill>(0);
    }
}

constexpr rocblas_diagonal char2rocblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
    case 'u': return rocblas_diagonal_unit;
    case 'N':
    case 'n': return rocblas_diagonal_non_unit;
    default: return static_cast<rocblas_diagonal>(0);
    }
}

constexpr rocblas_side char2rocblas_side(char value)
{
    switch(value)
    {
    case 'L':
    case 'l': return rocblas_side_left;
    case 'R':
    case 'r': return rocblas_side_right;
    default: return static_cast<rocblas_side>(0);
    }
}

constexpr rocblas_direct char2rocblas_direct(char value)
{
    switch(value)
    {
    case 'F': return rocblas_forward_direction;
    case 'B': return rocblas_backward_direction;
    default: return static_cast<rocblas_direct>(0);
    }
}

constexpr rocblas_pivot char2rocblas_pivot(char value)
{
    switch(value)
    {
    case 'V': return rocblas_pivot_variable;
    case 'T': return rocblas_pivot_top;
    case 'B': return rocblas_pivot_bottom;
    default: return static_cast<rocblas_pivot>(0);
    }
}

constexpr rocblas_storev char2rocblas_storev(char value)
{
    switch(value)
    {
    case 'C': return rocblas_column_wise;
    case 'R': return rocblas_row_wise;
    default: return static_cast<rocblas_storev>(0);
    }
}

constexpr rocblas_workmode char2rocblas_workmode(char value)
{
    switch(value)
    {
    case 'O': return rocblas_outofplace;
    case 'I': return rocblas_inplace;
    default: return static_cast<rocblas_workmode>(0);
    }
}

constexpr rocblas_svect char2rocblas_svect(char value)
{
    switch(value)
    {
    case 'A': return rocblas_svect_all;
    case 'S':
    case 'V': return rocblas_svect_singular;
    case 'O': return rocblas_svect_overwrite;
    case 'N': return rocblas_svect_none;
    default: return static_cast<rocblas_svect>(0);
    }
}

constexpr rocblas_srange char2rocblas_srange(char value)
{
    switch(value)
    {
    case 'A': return rocblas_srange_all;
    case 'V': return rocblas_srange_value;
    case 'I': return rocblas_srange_index;
    default: return static_cast<rocblas_srange>(-1);
    }
}

constexpr rocblas_evect char2rocblas_evect(char value)
{
    switch(value)
    {
    case 'V': return rocblas_evect_original;
    case 'I': return rocblas_evect_tridiagonal;
    case 'N': return rocblas_evect_none;
    default: return static_cast<rocblas_evect>(0);
    }
}

constexpr rocblas_eform char2rocblas_eform(char value)
{
    switch(value)
    {
    case '1': return rocblas_eform_ax;
    case '2': return rocblas_eform_abx;
    case '3': return rocblas_eform_bax;
    default: return static_cast<rocblas_eform>(0);
    }
}

constexpr rocblas_erange char2rocblas_erange(char value)
{
    switch(value)
    {
    case 'A': return rocblas_erange_all;
    case 'V': return rocblas_erange_value;
    case 'I': return rocblas_erange_index;
    default: return static_cast<rocblas_erange>(0);
    }
}

constexpr rocblas_eorder char2rocblas_eorder(char value)
{
    switch(value)
    {
    case 'B': return rocblas_eorder_blocks;
    case 'E': return rocblas_eorder_entire;
    default: return static_cast<rocblas_eorder>(0);
    }
}

constexpr rocblas_esort char2rocblas_esort(char value)
{
    switch(value)
    {
    case 'N': return rocblas_esort_none;
    case 'A': return rocblas_esort_ascending;
    default: return static_cast<rocblas_esort>(0);
    }
}

// clang-format off
inline rocblas_datatype string2rocblas_datatype(const std::string& value)
{
    return
        value == "f16_r" || value == "h" ? rocblas_datatype_f16_r :
        value == "f32_r" || value == "s" ? rocblas_datatype_f32_r :
        value == "f64_r" || value == "d" ? rocblas_datatype_f64_r :
        value == "bf16_r"                ? rocblas_datatype_bf16_r :
        value == "f16_c"                 ? rocblas_datatype_f16_c :
        value == "f32_c" || value == "c" ? rocblas_datatype_f32_c :
        value == "f64_c" || value == "z" ? rocblas_datatype_f64_c :
        value == "bf16_c"                ? rocblas_datatype_bf16_c :
        value == "i8_r"                  ? rocblas_datatype_i8_r  :
        value == "i32_r"                 ? rocblas_datatype_i32_r :
        value == "i8_c"                  ? rocblas_datatype_i8_c  :
        value == "i32_c"                 ? rocblas_datatype_i32_c :
        value == "u8_r"                  ? rocblas_datatype_u8_r  :
        value == "u32_r"                 ? rocblas_datatype_u32_r :
        value == "u8_c"                  ? rocblas_datatype_u8_c  :
        value == "u32_c"                 ? rocblas_datatype_u32_c :
#if ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES
        value == "f8_r"                  ? rocblas_datatype_f8_r  :
        value == "bf8_r"                 ? rocblas_datatype_bf8_r :
#endif
        rocblas_datatype_invalid;
}
// clang-format on

constexpr rocsolver_rfinfo_mode char2rocsolver_rfinfo_mode(char value)
{
    switch(value)
    {
    case '1': return rocsolver_rfinfo_mode_lu;
    case '2': return rocsolver_rfinfo_mode_cholesky;
    default: return static_cast<rocsolver_rfinfo_mode>(0);
    }
}

#undef ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_END_NAMESPACE
#endif
