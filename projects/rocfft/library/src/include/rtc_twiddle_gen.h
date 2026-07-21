// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef RTC_TWIDDLE_GEN
#define RTC_TWIDDLE_GEN

#include "rocfft/rocfft.h"
#include <hip/hip_runtime_api.h>
#include <string>

static const unsigned int TWIDDLES_THREADS         = 32;
static const unsigned int TWIDDLES_RTC_MAX_RADICES = 8;
static const double       TWO_PI                   = -6.283185307179586476925286766559;

// structure to pass fixed-length array of radices by value in
// kernargs instead of by reference in global memory
struct radices_t
{
    size_t data[TWIDDLES_RTC_MAX_RADICES];
};
// stringified version of same structure, for RTC
static const char* radices_t_str{
    R"_RADICES_T(
struct radices_t
{
    size_t data[TWIDDLES_MAX_RADICES];
};
)_RADICES_T"};

enum struct TwiddleTableType
{
    // "stacked" table generated from radices (radices multiply out
    // to length N)
    RADICES,
    // N twiddles for length N
    LENGTH_N,
    // half-length table for length N
    HALF_N,
    // "large" twiddle table, decomposed into pow2 base and multiple
    // steps
    LARGE,
    // N * N table for length N partial pass
    PARTIAL_PASS_N
};

// generate name for twiddle-compute kernel
std::string twiddle_rtc_kernel_name(TwiddleTableType type, rocfft_precision precision);
// generate source for twiddle-compute kernel
std::string
    twiddle_rtc(const std::string& kernel_name, TwiddleTableType type, rocfft_precision precision);

#endif
