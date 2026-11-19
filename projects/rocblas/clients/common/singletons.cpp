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
#include <sstream>

// global for device memory padding see d_vector.hpp
size_t g_DVEC_PAD = 4096;

void d_vector_set_pad_length(size_t pad)
{
    g_DVEC_PAD = pad;
}

// The macro may be tested here, but not in d_vector.hpp: see singletons.hpp.
void d_vector_report_failure(const std::string& message)
{
#ifdef GOOGLE_TEST
    // Recorded unconditionally: between tests Google Test keeps it as an ad-hoc failure,
    // which still makes RUN_ALL_TESTS return non-zero.
    ADD_FAILURE() << message;
    if(::testing::UnitTest::GetInstance()->current_test_info())
        return;

    // No running test to attribute it to, so print it as well rather than leave it buried
    // in the ad-hoc results. This is rocblas-bench and rocblas-gemm-tune.
    rocblas_cerr << "rocBLAS client: " << message << std::endl;
#else
    // A sample: no Google Test anywhere in the binary. These previously reached the
    // non-test CHECK_HIP_ERROR, which printed and exited, so keep the exit status or a
    // sample would report success after failing to free device memory.
    rocblas_cerr << "rocBLAS client: " << message << std::endl;
    exit(EXIT_FAILURE);
#endif
}

void d_vector_report_guard_corruption(const unsigned char* host,
                                      const unsigned char* reference,
                                      size_t               guard_bytes,
                                      const char*          tag)
{
    size_t differing = 0, first = 0;
    for(size_t i = 0; i < guard_bytes; ++i)
        if(host[i] != reference[i])
        {
            if(!differing) // first differing byte only
                first = i;
            ++differing;
        }

    std::ostringstream msg;
    if(differing > 0)
        // Each number is delimited on both sides, so a test matching on "corrupted: 6 of "
        // cannot also be satisfied by a reported 16.
        msg << tag << "-guard corrupted: " << differing << " of " << guard_bytes
            << " byte(s) differ, first at offset " << first << " (expected 0x" << std::hex
            << static_cast<unsigned>(reference[first]) << ", got 0x"
            << static_cast<unsigned>(host[first]) << std::dec << ")";
    else
        // The caller has already established that memcmp found a difference, so finding
        // none here means the two disagree. Report that rather than lose the corruption.
        msg << tag << "-guard mismatch reported, but no byte differs across " << guard_bytes
            << " byte(s); device_vector_check is inconsistent";

    d_vector_report_failure(msg.str());
}
