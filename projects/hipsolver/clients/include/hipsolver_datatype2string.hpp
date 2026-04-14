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

#pragma once

#include <ostream>
#include <string>

#include "complex.hpp"
#include "hipsolver.h"

// Complex output
inline std::ostream& operator<<(std::ostream& os, const hipsolverComplex& x)
{
    os << "'(" << x.real() << "," << x.imag() << ")'";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipsolverDoubleComplex& x)
{
    os << "'(" << x.real() << "," << x.imag() << ")'";
    return os;
}

/* ============================================================================================ */
/*  Convert hipsolver constants to lapack char. */

inline constexpr auto hipsolver2string_status(hipsolverStatus_t value)
{
    switch(value)
    {
    case HIPSOLVER_STATUS_SUCCESS:
        return "HIPSOLVER_STATUS_SUCCESS";
    case HIPSOLVER_STATUS_NOT_INITIALIZED:
        return "HIPSOLVER_STATUS_NOT_INITIALIZED";
    case HIPSOLVER_STATUS_ALLOC_FAILED:
        return "HIPSOLVER_STATUS_ALLOC_FAILED";
    case HIPSOLVER_STATUS_INVALID_VALUE:
        return "HIPSOLVER_STATUS_INVALID_VALUE";
    case HIPSOLVER_STATUS_MAPPING_ERROR:
        return "HIPSOLVER_STATUS_MAPPING_ERROR";
    case HIPSOLVER_STATUS_EXECUTION_FAILED:
        return "HIPSOLVER_STATUS_EXECUTION_FAILED";
    case HIPSOLVER_STATUS_INTERNAL_ERROR:
        return "HIPSOLVER_STATUS_INTERNAL_ERROR";
    case HIPSOLVER_STATUS_NOT_SUPPORTED:
        return "HIPSOLVER_STATUS_NOT_SUPPORTED";
    case HIPSOLVER_STATUS_ARCH_MISMATCH:
        return "HIPSOLVER_STATUS_ARCH_MISMATCH";
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
        return "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR";
    case HIPSOLVER_STATUS_INVALID_ENUM:
        return "HIPSOLVER_STATUS_INVALID_ENUM";
    case HIPSOLVER_STATUS_UNKNOWN:
        return "HIPSOLVER_STATUS_UNKNOWN";
    default:
        throw std::invalid_argument("Invalid enum");
    }
}

char hipsolver2char_operation(hipsolverOperation_t value);

char hipsolver2char_fill(hipsolverFillMode_t value);

char hipsolver2char_side(hipsolverSideMode_t value);

char hipsolver2char_evect(hipsolverEigMode_t value);

char hipsolver2char_eform(hipsolverEigType_t value);

char hipsolver2char_erange(hipsolverEigRange_t value);

/* ============================================================================================ */
/*  Convert lapack char constants to hipsolver type. */

hipsolverStatus_t string2hipsolver_status(const std::string& value);

hipsolverOperation_t char2hipsolver_operation(char value);

hipsolverFillMode_t char2hipsolver_fill(char value);

hipsolverSideMode_t char2hipsolver_side(char value);

hipsolverEigMode_t char2hipsolver_evect(char value);

hipsolverEigType_t char2hipsolver_eform(char value);

hipsolverEigRange_t char2hipsolver_erange(char value);
