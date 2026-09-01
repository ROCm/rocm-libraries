// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_ROCRAND_COMMON_HPP_
#define TEST_ROCRAND_COMMON_HPP_

#include <rocrand/rocrand.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

constexpr rocrand_rng_type rng_types[] = {ROCRAND_RNG_PSEUDO_PHILOX4_32_10,
                                          ROCRAND_RNG_PSEUDO_MRG31K3P,
                                          ROCRAND_RNG_PSEUDO_MRG32K3A,
                                          ROCRAND_RNG_PSEUDO_XORWOW,
                                          ROCRAND_RNG_PSEUDO_MTGP32,
                                          ROCRAND_RNG_PSEUDO_LFSR113,
                                          ROCRAND_RNG_PSEUDO_THREEFRY2_32_20,
                                          ROCRAND_RNG_PSEUDO_THREEFRY2_64_20,
                                          ROCRAND_RNG_PSEUDO_THREEFRY4_32_20,
                                          ROCRAND_RNG_PSEUDO_THREEFRY4_64_20,
                                          ROCRAND_RNG_QUASI_SOBOL32,
                                          ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32,
                                          ROCRAND_RNG_QUASI_SOBOL64,
                                          ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64};

constexpr rocrand_rng_type int_rng_types[] = {ROCRAND_RNG_PSEUDO_PHILOX4_32_10,
                                              ROCRAND_RNG_PSEUDO_MRG31K3P,
                                              ROCRAND_RNG_PSEUDO_MRG32K3A,
                                              ROCRAND_RNG_PSEUDO_XORWOW,
                                              ROCRAND_RNG_PSEUDO_MTGP32,
                                              ROCRAND_RNG_PSEUDO_LFSR113,
                                              ROCRAND_RNG_PSEUDO_THREEFRY2_32_20,
                                              ROCRAND_RNG_PSEUDO_THREEFRY4_32_20,
                                              ROCRAND_RNG_QUASI_SOBOL32,
                                              ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32};

constexpr rocrand_rng_type long_long_rng_types[] = {ROCRAND_RNG_PSEUDO_THREEFRY2_64_20,
                                                    ROCRAND_RNG_PSEUDO_THREEFRY4_64_20,
                                                    ROCRAND_RNG_QUASI_SOBOL64,
                                                    ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64};

// Map values of rocrand_rng_type to names so test filters can target a test
// parameter by name, instead of positional index, which changes upon re-ordering.
inline std::string rocrand_rng_name(rocrand_rng_type rng_type)
{
    switch(rng_type)
    {
        case ROCRAND_RNG_PSEUDO_PHILOX4_32_10: return "Philox4_32_10";
        case ROCRAND_RNG_PSEUDO_MRG31K3P: return "Mrg31k3p";
        case ROCRAND_RNG_PSEUDO_MRG32K3A: return "Mrg32k3a";
        case ROCRAND_RNG_PSEUDO_XORWOW: return "Xorwow";
        case ROCRAND_RNG_PSEUDO_MTGP32: return "Mtgp32";
        case ROCRAND_RNG_PSEUDO_LFSR113: return "Lfsr113";
        case ROCRAND_RNG_PSEUDO_MT19937: return "Mt19937";
        case ROCRAND_RNG_PSEUDO_THREEFRY2_32_20: return "Threefry2_32_20";
        case ROCRAND_RNG_PSEUDO_THREEFRY2_64_20: return "Threefry2_64_20";
        case ROCRAND_RNG_PSEUDO_THREEFRY4_32_20: return "Threefry4_32_20";
        case ROCRAND_RNG_PSEUDO_THREEFRY4_64_20: return "Threefry4_64_20";
        case ROCRAND_RNG_QUASI_SOBOL32: return "Sobol32";
        case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32: return "ScrambledSobol32";
        case ROCRAND_RNG_QUASI_SOBOL64: return "Sobol64";
        case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64: return "ScrambledSobol64";
        default: return "Unknown";
    }
}

inline std::string
    rocrand_rng_type_test_name(const ::testing::TestParamInfo<rocrand_rng_type>& info)
{
    return rocrand_rng_name(info.param);
}

// Convert test parameters in "double lambdas[]{...}" to names, e.g. 5.5 -> "Lambda5_5".
inline std::string lambda_param_name(const ::testing::TestParamInfo<double>& info)
{
    std::string s    = std::to_string(info.param);
    const auto  last = s.find_last_not_of('0');
    if(last != std::string::npos)
        s.erase(last + 1);
    if(!s.empty() && s.back() == '.')
        s.pop_back();
    for(char& c : s)
        if(c == '.')
            c = '_';
    return "Lambda" + s;
}

#endif // TEST_ROCRAND_COMMON_HPP_
