/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <exception>

#include "hipsolver.h"
#include "lib_macros.hpp"

HIPSOLVER_BEGIN_NAMESPACE

// Convert the current C++ exception to hipsolverStatus_t
// This allows extern "C" functions to return this function in a catch(...) block
// while converting all C++ exceptions to an equivalent hipsolverStatus_t here
inline hipsolverStatus_t exception2hip_status(std::exception_ptr e = std::current_exception())
try
{
    if(e)
        std::rethrow_exception(e);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(const hipsolverStatus_t& status)
{
    return status;
}
catch(const std::bad_alloc&)
{
    return HIPSOLVER_STATUS_ALLOC_FAILED;
}
catch(...)
{
    return HIPSOLVER_STATUS_INTERNAL_ERROR;
}

HIPSOLVER_END_NAMESPACE
