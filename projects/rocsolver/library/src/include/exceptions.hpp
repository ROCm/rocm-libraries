/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "rocblas_utility.hpp"
#include "rocsparse.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// Convert the current C++ exception to rocblas_status
// This allows extern "C" functions to return this function in a catch(...) block
// while converting all C++ exceptions to an equivalent rocblas_status here
inline rocblas_status exception2rocblas_status(std::exception_ptr e = std::current_exception())
try
{
    if(e)
        std::rethrow_exception(e);
    return rocblas_status_success;
}
catch(const rocblas_status& status)
{
    return status;
}
catch(const rocsparse_status& status)
{
    return rocsparse2rocblas_status(status);
}
catch(const std::bad_alloc&)
{
    return rocblas_status_memory_error;
}
catch(...)
{
    return rocblas_status_internal_error;
}

ROCSOLVER_END_NAMESPACE
