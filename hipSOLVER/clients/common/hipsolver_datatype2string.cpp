/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * ************************************************************************ */

#include "../include/hipsolver_datatype2string.hpp"

/* ============================================================================================ */
/*  Convert hipsolver constants to lapack char. */

char hipsolver2char_operation(hipsolverOperation_t value)
{
    switch(value)
    {
    case HIPSOLVER_OP_N:
        return 'N';
    case HIPSOLVER_OP_T:
        return 'T';
    case HIPSOLVER_OP_C:
        return 'C';
    default:
        throw std::invalid_argument("Invalid enum");
    }
}

char hipsolver2char_fill(hipsolverFillMode_t value)
{
    switch(value)
    {
    case HIPSOLVER_FILL_MODE_UPPER:
        return 'U';
    case HIPSOLVER_FILL_MODE_LOWER:
        return 'L';
    default:
        throw std::invalid_argument("Invalid enum");
    }
}

char hipsolver2char_side(hipsolverSideMode_t value)
{
    switch(value)
    {
    case HIPSOLVER_SIDE_LEFT:
        return 'L';
    case HIPSOLVER_SIDE_RIGHT:
        return 'R';
    default:
        throw std::invalid_argument("Invalid enum");
    }
}

char hipsolver2char_evect(hipsolverEigMode_t value)
{
    switch(value)
    {
    case HIPSOLVER_EIG_MODE_NOVECTOR:
        return 'N';
    case HIPSOLVER_EIG_MODE_VECTOR:
        return 'V';
    default:
        throw std::invalid_argument("Invalid enum");
    }
}

char hipsolver2char_eform(hipsolverEigType_t value)
{
    switch(value)
    {
    case HIPSOLVER_EIG_TYPE_1:
        return '1';
    case HIPSOLVER_EIG_TYPE_2:
        return '2';
    case HIPSOLVER_EIG_TYPE_3:
        return '3';
    default:
        throw std::invalid_argument("Invalid enum");
    }
}

char hipsolver2char_erange(hipsolverEigRange_t value)
{
    switch(value)
    {
    case HIPSOLVER_EIG_RANGE_ALL:
        return 'A';
    case HIPSOLVER_EIG_RANGE_V:
        return 'V';
    case HIPSOLVER_EIG_RANGE_I:
        return 'I';
    default:
        throw std::invalid_argument("Invalid enum");
    }
}

/* ============================================================================================ */
/*  Convert lapack char constants to hipsolver type. */

hipsolverStatus_t string2hipsolver_status(const std::string& value)
{
    return value == "HIPSOLVER_STATUS_SUCCESS"             ? HIPSOLVER_STATUS_SUCCESS
           : value == "HIPSOLVER_STATUS_NOT_INITIALIZED"   ? HIPSOLVER_STATUS_NOT_INITIALIZED
           : value == "HIPSOLVER_STATUS_ALLOC_FAILED"      ? HIPSOLVER_STATUS_ALLOC_FAILED
           : value == "HIPSOLVER_STATUS_INVALID_VALUE"     ? HIPSOLVER_STATUS_INVALID_VALUE
           : value == "HIPSOLVER_STATUS_MAPPING_ERROR"     ? HIPSOLVER_STATUS_MAPPING_ERROR
           : value == "HIPSOLVER_STATUS_EXECUTION_FAILED"  ? HIPSOLVER_STATUS_EXECUTION_FAILED
           : value == "HIPSOLVER_STATUS_INTERNAL_ERROR"    ? HIPSOLVER_STATUS_INTERNAL_ERROR
           : value == "HIPSOLVER_STATUS_NOT_SUPPORTED"     ? HIPSOLVER_STATUS_NOT_SUPPORTED
           : value == "HIPSOLVER_STATUS_ARCH_MISMATCH"     ? HIPSOLVER_STATUS_ARCH_MISMATCH
           : value == "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR" ? HIPSOLVER_STATUS_HANDLE_IS_NULLPTR
           : value == "HIPSOLVER_STATUS_INVALID_ENUM"      ? HIPSOLVER_STATUS_INVALID_ENUM
           : value == "HIPSOLVER_STATUS_UNKNOWN"           ? HIPSOLVER_STATUS_UNKNOWN
                                                           : static_cast<hipsolverStatus_t>(-1);
}

hipsolverOperation_t char2hipsolver_operation(char value)
{
    switch(value)
    {
    case 'n':
    case 'N':
        return HIPSOLVER_OP_N;
    case 't':
    case 'T':
        return HIPSOLVER_OP_T;
    case 'c':
    case 'C':
        return HIPSOLVER_OP_C;
    default:
        throw std::invalid_argument("Invalid character");
    }
}

hipsolverFillMode_t char2hipsolver_fill(char value)
{
    switch(value)
    {
    case 'u':
    case 'U':
        return HIPSOLVER_FILL_MODE_UPPER;
    case 'l':
    case 'L':
        return HIPSOLVER_FILL_MODE_LOWER;
    default:
        throw std::invalid_argument("Invalid character");
    }
}

hipsolverSideMode_t char2hipsolver_side(char value)
{
    switch(value)
    {
    case 'l':
    case 'L':
        return HIPSOLVER_SIDE_LEFT;
    case 'r':
    case 'R':
        return HIPSOLVER_SIDE_RIGHT;
    default:
        throw std::invalid_argument("Invalid character");
    }
}

hipsolverEigMode_t char2hipsolver_evect(char value)
{
    switch(value)
    {
    case 'n':
    case 'N':
        return HIPSOLVER_EIG_MODE_NOVECTOR;
    case 'v':
    case 'V':
        return HIPSOLVER_EIG_MODE_VECTOR;
    default:
        throw std::invalid_argument("Invalid character");
    }
}

hipsolverEigType_t char2hipsolver_eform(char value)
{
    switch(value)
    {
    case '1':
        return HIPSOLVER_EIG_TYPE_1;
    case '2':
        return HIPSOLVER_EIG_TYPE_2;
    case '3':
        return HIPSOLVER_EIG_TYPE_3;
    default:
        throw std::invalid_argument("Invalid character");
    }
}

hipsolverEigRange_t char2hipsolver_erange(char value)
{
    switch(value)
    {
    case 'A':
        return HIPSOLVER_EIG_RANGE_ALL;
    case 'V':
        return HIPSOLVER_EIG_RANGE_V;
    case 'I':
        return HIPSOLVER_EIG_RANGE_I;
    default:
        throw std::invalid_argument("Invalid character");
    }
}
