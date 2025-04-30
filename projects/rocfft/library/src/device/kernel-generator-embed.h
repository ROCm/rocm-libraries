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

#ifndef ROCFFT_KERNEL_GENERATOR_EMBED_H
#define ROCFFT_KERNEL_GENERATOR_EMBED_H

#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

extern const char* rocfft_complex_h;
extern const char* common_h;
extern const char* memory_gfx_h;
extern const char* callback_h;
extern const char* butterfly_constant_h;
extern const char* real2complex_device_h;
extern const char* large_twiddles_h;
extern const char* radix_2_h;
extern const char* radix_3_h;
extern const char* radix_4_h;
extern const char* radix_5_h;
extern const char* radix_6_h;
extern const char* radix_7_h;
extern const char* radix_8_h;
extern const char* radix_9_h;
extern const char* radix_10_h;
extern const char* radix_11_h;
extern const char* radix_13_h;
extern const char* radix_16_h;
extern const char* radix_17_h;

extern const char* device_properties_h;
extern const char* rocfft_hip_h;
extern const char* gpubuf_h;
extern const char* rtc_kernel_h;
extern const char* rtc_kernel_cpp;
extern const char* rtc_test_harness_helper_cpp;

const std::array<char, 32> generator_sum();

// append the necessary radix headers to src, for the given factors
static void append_radix_h(std::string& src, const std::vector<unsigned int>& factors)
{
    static const std::unordered_map<size_t, const char*> butterfly_funcs = {{2, radix_2_h},
                                                                            {3, radix_3_h},
                                                                            {4, radix_4_h},
                                                                            {5, radix_5_h},
                                                                            {6, radix_6_h},
                                                                            {7, radix_7_h},
                                                                            {8, radix_8_h},
                                                                            {9, radix_9_h},
                                                                            {10, radix_10_h},
                                                                            {11, radix_11_h},
                                                                            {13, radix_13_h},
                                                                            {16, radix_16_h},
                                                                            {17, radix_17_h}};

    // factors may contain duplicates, so uniquify them
    std::vector<unsigned int> factors_uniq = factors;
    std::sort(factors_uniq.begin(), factors_uniq.end());
    factors_uniq.erase(std::unique(factors_uniq.begin(), factors_uniq.end()), factors_uniq.end());

    for(auto f : factors_uniq)
    {
        // we can build a length-1 FFT kernel, which will ask for the
        // no-op radix 1
        if(f == 1)
            continue;
        src += butterfly_funcs.at(f);
    }
}
#endif
