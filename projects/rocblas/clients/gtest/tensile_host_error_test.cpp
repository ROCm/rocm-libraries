/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocblas/rocblas.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

// Contract behind the Tensile-host-init error-logging fix (tensile_host.cpp
// get_library_and_adapter). A bare rocblas_status enum is NOT a std::exception,
// so a load failure thrown as a bare enum slips past catch(const std::exception&)
// and, without a typed catch(const rocblas_status&), lands in the blank catch(...)
// that prints only "Unknown exception thrown". This pins that dispatch semantic;
// the real init path is GPU-bound and not exercised here.
TEST(TensileHostInitError, RocblasStatusBypassesStdExceptionCatch)
{
    enum Handler
    {
        NONE,
        STD_EXCEPTION,
        ROCBLAS_STATUS,
        BLACKOUT
    };

    // With the typed handler present (the fix): a bare rocblas_status is
    // caught by catch(const rocblas_status&), never by catch(const std::exception&),
    // and never falls through to the catch(...) blackout.
    Handler hit = NONE;
    try
    {
        throw rocblas_status_internal_error;
    }
    catch(const std::exception&)
    {
        hit = STD_EXCEPTION;
    }
    catch(const rocblas_status& status)
    {
        hit = ROCBLAS_STATUS;
        // The typed handler recovers the detail the blank catch(...) discarded.
        EXPECT_STRNE(rocblas_status_to_string(status), "");
    }
    catch(...)
    {
        hit = BLACKOUT;
    }
    EXPECT_EQ(hit, ROCBLAS_STATUS);

    // Without the typed handler (pre-fix ordering), the same throw reaches
    // catch(...) — the blackout the fix eliminates.
    Handler preFix = NONE;
    try
    {
        throw rocblas_status_internal_error;
    }
    catch(const std::exception&)
    {
        preFix = STD_EXCEPTION;
    }
    catch(...)
    {
        preFix = BLACKOUT;
    }
    EXPECT_EQ(preFix, BLACKOUT);
}

// The switched-to macro (HIP_CHECK_EXC_MESSAGE) throws std::runtime_error, which
// IS a std::exception, so its detail survives through catch(const std::exception&).
// This asserts the "detail is preserved" half of the same contract.
TEST(TensileHostInitError, RuntimeErrorDetailSurvivesStdExceptionCatch)
{
    const std::string detail = "loading code object: /path/to/gfx.co";
    std::string       caught;
    try
    {
        throw std::runtime_error(detail);
    }
    catch(const std::exception& e)
    {
        caught = e.what();
    }
    catch(...)
    {
        caught = "unknown";
    }
    EXPECT_EQ(caught, detail);
}
