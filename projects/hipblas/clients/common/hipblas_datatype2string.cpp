/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "hipblas_datatype2string.hpp"
#include "hipblas.h"

// clang-format off
hipblas_initialization string2hipblas_initialization(const std::string& value)
{
    return
        value == "rand_int"   ? hipblas_initialization::rand_int   :
        value == "trig_float" ? hipblas_initialization::trig_float :
        value == "hpl"        ? hipblas_initialization::hpl        :
        static_cast<hipblas_initialization>(0); // invalid enum
}
// clang-format on

/* ============================================================================================ */
/*  Convert hipblas constants to lapack char. */

char hipblas2char_operation(hipblasOperation_t value)
{
    switch(value)
    {
    case HIPBLAS_OP_N:
        return 'N';
    case HIPBLAS_OP_T:
        return 'T';
    case HIPBLAS_OP_C:
        return 'C';
    }
    return '\0';
}

char hipblas2char_fill(hipblasFillMode_t value)
{
    switch(value)
    {
    case HIPBLAS_FILL_MODE_UPPER:
        return 'U';
    case HIPBLAS_FILL_MODE_LOWER:
        return 'L';
    case HIPBLAS_FILL_MODE_FULL:
        return 'F';
    }
    return '\0';
}

char hipblas2char_diagonal(hipblasDiagType_t value)
{
    switch(value)
    {
    case HIPBLAS_DIAG_UNIT:
        return 'U';
    case HIPBLAS_DIAG_NON_UNIT:
        return 'N';
    }
    return '\0';
}

char hipblas2char_side(hipblasSideMode_t value)
{
    switch(value)
    {
    case HIPBLAS_SIDE_LEFT:
        return 'L';
    case HIPBLAS_SIDE_RIGHT:
        return 'R';
    case HIPBLAS_SIDE_BOTH:
        return 'B';
    }
    return '\0';
}

/* ============================================================================================ */
/*  Convert lapack char constants to hipblas type. */

hipblasOperation_t char2hipblas_operation(char value)
{
    switch(value)
    {
    case 'N':
        return HIPBLAS_OP_N;
    case 'T':
        return HIPBLAS_OP_T;
    case 'C':
        return HIPBLAS_OP_C;
    case 'n':
        return HIPBLAS_OP_N;
    case 't':
        return HIPBLAS_OP_T;
    case 'c':
        return HIPBLAS_OP_C;
    }
    return HIPBLAS_OP_N;
}

hipblasFillMode_t char2hipblas_fill(char value)
{
    switch(value)
    {
    case 'U':
        return HIPBLAS_FILL_MODE_UPPER;
    case 'L':
        return HIPBLAS_FILL_MODE_LOWER;
    case 'u':
        return HIPBLAS_FILL_MODE_UPPER;
    case 'l':
        return HIPBLAS_FILL_MODE_LOWER;
    }
    return HIPBLAS_FILL_MODE_LOWER;
}

hipblasDiagType_t char2hipblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
        return HIPBLAS_DIAG_UNIT;
    case 'N':
        return HIPBLAS_DIAG_NON_UNIT;
    case 'u':
        return HIPBLAS_DIAG_UNIT;
    case 'n':
        return HIPBLAS_DIAG_NON_UNIT;
    }
    return HIPBLAS_DIAG_NON_UNIT;
}

hipblasSideMode_t char2hipblas_side(char value)
{
    switch(value)
    {
    case 'L':
        return HIPBLAS_SIDE_LEFT;
    case 'R':
        return HIPBLAS_SIDE_RIGHT;
    case 'l':
        return HIPBLAS_SIDE_LEFT;
    case 'r':
        return HIPBLAS_SIDE_RIGHT;
    }
    return HIPBLAS_SIDE_LEFT;
}

// clang-format off
hipblasDatatype_t string2hipblas_datatype(const std::string& value)
{
    return
        value == "f16_r" || value == "h" ? HIPBLAS_R_16F  :
        value == "f32_r" || value == "s" ? HIPBLAS_R_32F  :
        value == "f64_r" || value == "d" ? HIPBLAS_R_64F  :
        value == "bf16_r"                ? HIPBLAS_R_16B :
        value == "f16_c"                 ? HIPBLAS_C_16B  :
        value == "f32_c" || value == "c" ? HIPBLAS_C_32F  :
        value == "f64_c" || value == "z" ? HIPBLAS_C_64F  :
        value == "bf16_c"                ? HIPBLAS_C_16B :
        value == "i8_r"                  ? HIPBLAS_R_8I   :
        value == "i32_r"                 ? HIPBLAS_R_32I  :
        value == "i8_c"                  ? HIPBLAS_C_8I   :
        value == "i32_c"                 ? HIPBLAS_C_32I  :
        value == "u8_r"                  ? HIPBLAS_R_8U   :
        value == "u32_r"                 ? HIPBLAS_R_32U  :
        value == "u8_c"                  ? HIPBLAS_C_8U   :
        value == "u32_c"                 ? HIPBLAS_C_32U  :
        HIPBLAS_DATATYPE_INVALID;
}

hipblasComputeType_t string2hipblas_computetype(const std::string& value)
{
    return value == "c16f"           ? HIPBLAS_COMPUTE_16F :
           value == "c16f_pedantic"  ? HIPBLAS_COMPUTE_16F_PEDANTIC :
           value == "c32f"           ? HIPBLAS_COMPUTE_32F :
           value == "c32f_pedantic"  ? HIPBLAS_COMPUTE_32F_PEDANTIC :
           value == "c32f_fast_16f"  ? HIPBLAS_COMPUTE_32F_FAST_16F :
           value == "c32f_fast_16Bf" ? HIPBLAS_COMPUTE_32F_FAST_16BF :
           value == "c32f_fast_tf32" ? HIPBLAS_COMPUTE_32F_FAST_TF32 :
           value == "c64f"           ? HIPBLAS_COMPUTE_64F :
           value == "c64f_pedantic"  ? HIPBLAS_COMPUTE_64F_PEDANTIC :
           value == "c32i"           ? HIPBLAS_COMPUTE_32I :
           value == "c32i_pedantic"  ? HIPBLAS_COMPUTE_32I_PEDANTIC :
           HIPBLAS_COMPUTE_32F; // Default
}
// clang-format on
