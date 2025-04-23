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

#include <cstdlib>

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

#endif // TEST_ROCRAND_COMMON_HPP_
