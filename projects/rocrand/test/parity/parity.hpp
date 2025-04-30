// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TEST_PARITY_PARITY_HPP_
#define TEST_PARITY_PARITY_HPP_

#include <cstddef>
#include <vector>

// The different generators that can be tested. These are the generators
// which rocrand and curand both implement.
enum class generator_type
{
    XORWOW,
    MRG32K3A,
    MTGP32,
    PHILOX4_32_10,
    MT19937,
    SOBOL32,
    SCRAMBLED_SOBOL32,
    SOBOL64,
    SCRAMBLED_SOBOL64
};

struct test_case
{
    generator_type rng_type;
    size_t         size;
    long long      prng_seed       = -1; // ignored if not prng
    int            qrng_dimensions = -1; // ignored if not qrng
    long long      offset          = -1;
};

inline bool generator_is_pseudo(const generator_type rng_type)
{
    switch(rng_type)
    {
        case generator_type::XORWOW:
        case generator_type::MRG32K3A:
        case generator_type::MTGP32:
        case generator_type::PHILOX4_32_10:
        case generator_type::MT19937: return true;
        default: return false;
    }
}

std::vector<unsigned int>       test_rocrand_generate(const test_case& test_case);
std::vector<unsigned long long> test_rocrand_generate_long_long(const test_case& test_case);

std::vector<unsigned int>       test_curand_generate(const test_case& test_case);
std::vector<unsigned long long> test_curand_generate_long_long(const test_case& test_case);

#endif
