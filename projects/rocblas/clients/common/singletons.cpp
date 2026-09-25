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
 * ************************************************************************ */

// rocblas_test.hpp supplies rocblas_cerr, and Google Test itself when GOOGLE_TEST is
// defined. Included the same way host_alloc.cpp in this directory does, so that this file
// still compiles for the samples, which build it without Google Test on their include path.
#include "singletons.hpp"
#include "rocblas_test.hpp"

#include <cstdlib>

// global for device memory padding see d_vector.hpp
size_t g_DVEC_PAD = 4096;

void d_vector_set_pad_length(size_t pad)
{
    g_DVEC_PAD = pad;
}

void d_vector_report_failure(const std::string& message)
{
    // Conditional here rather than in d_vector.hpp, which is the whole point: this is an
    // ordinary function, compiled exactly once per binary, so a body that varies with the
    // macro gives no symbol two definitions. The samples compile this file without
    // GOOGLE_TEST and without linking Google Test, so the reference to ADD_FAILURE has to
    // be compiled out for them rather than merely unused.
#ifdef GOOGLE_TEST
    // Recorded unconditionally. With a running test this attributes the failure to it;
    // between tests Google Test keeps it as an ad-hoc failure, which still makes
    // RUN_ALL_TESTS return non-zero. The EXPECT_EQ this replaced also recorded
    // unconditionally, and reporting only when a test happens to be running would let a
    // guard failure during teardown between tests go unnoticed.
    ADD_FAILURE() << message;
    if(::testing::UnitTest::GetInstance()->current_test_info())
        return;

    // No test to attribute it to, so print it rather than leave it in the ad-hoc results.
    // That is rocblas-bench and rocblas-gemm-tune, for which a failed free has never been
    // fatal: both compile with GOOGLE_TEST, so both already got the assertion form.
    rocblas_cerr << "rocBLAS client: " << message << std::endl;
#else
    // No Google Test anywhere in this binary, so this is a sample. Two of them build
    // device containers, and before this function existed they reached the non-test form
    // of CHECK_HIP_ERROR, which printed and exited. Keep that exit status, or a sample
    // would report success after failing to free device memory.
    rocblas_cerr << "rocBLAS client: " << message << std::endl;
    exit(EXIT_FAILURE);
#endif
}
