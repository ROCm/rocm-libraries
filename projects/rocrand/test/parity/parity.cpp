// MIT License
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "parity.hpp"

#include <iostream>
#include <vector>

#ifndef DEFAULT_RAND_N
    #define DEFAULT_RAND_N (1024 * 1024 * 128)
#endif

constexpr long long seeds[]   = {-1ll, 12345ll};
constexpr int       dims[]    = {1, 3};
constexpr long long offsets[] = {-1ll, 112121116104111110ll};

static const char* generator_name(const generator_type rng_type)
{
    switch(rng_type)
    {
        case generator_type::XORWOW: return "xorwow";
        case generator_type::MRG32K3A: return "mrg32k3a";
        case generator_type::MTGP32: return "mtgp32";
        case generator_type::PHILOX4_32_10: return "philox";
        case generator_type::MT19937: return "mt19937";
        case generator_type::SOBOL32: return "sobol32";
        case generator_type::SCRAMBLED_SOBOL32: return "scrambled_sobol32";
        case generator_type::SOBOL64: return "sobol64";
        case generator_type::SCRAMBLED_SOBOL64: return "scrambled_sobol64";
    }
}

static bool generator_is_64bit(const generator_type rng_type)
{
    return rng_type == generator_type::SOBOL64 || rng_type == generator_type::SCRAMBLED_SOBOL64;
}

static bool generator_supports_offset(const generator_type rng_type)
{
    return rng_type != generator_type::MTGP32 && rng_type != generator_type::MT19937;
}

template<typename T, typename F, typename G>
static void run_test(const test_case& test_case, F generate_rocrand, G generate_curand)
{
    constexpr size_t n = DEFAULT_RAND_N;

    std::cout << generator_name(test_case.rng_type) << "," << test_case.size;

    if(test_case.prng_seed < 0)
        std::cout << ",default";
    else
        std::cout << "," << test_case.prng_seed;

    if(test_case.qrng_dimensions < 0)
        std::cout << ",default";
    else
        std::cout << "," << test_case.qrng_dimensions;

    if(test_case.offset < 0)
        std::cout << ",default";
    else
        std::cout << "," << test_case.offset;

    const std::vector<T> rocrand_results = generate_rocrand(test_case);
    const std::vector<T> curand_results  = generate_curand(test_case);

    size_t i = 0;
    for(; i < n; ++i)
    {
        if(rocrand_results[i] != curand_results[i])
        {
            break;
        }
    }

    if(i < n)
    {
        std::cout << ",mismatch at index " << i << std::endl;
    }
    else
    {
        std::cout << ",results are equal" << std::endl;
    }
}

static void run_case(const test_case& test_case)
{
    if(generator_is_64bit(test_case.rng_type))
    {
        run_test<unsigned long long>(test_case,
                                     test_rocrand_generate_long_long,
                                     test_curand_generate_long_long);
    }
    else
    {
        run_test<unsigned int>(test_case, test_rocrand_generate, test_curand_generate);
    }
}

static void test_offset(const generator_type rng_type, size_t size, long long offset)
{
    if(generator_is_pseudo(rng_type))
    {
        for(const long long seed : seeds)
        {
            test_case test_case;
            test_case.rng_type  = rng_type;
            test_case.size      = size;
            test_case.prng_seed = seed;
            test_case.offset    = offset;
            run_case(test_case);
        }
    }
    else
    {
        for(const long long dim : dims)
        {
            test_case test_case;
            test_case.rng_type = rng_type;
            // Round down to multiple of dimensions
            test_case.size            = size - size % dim;
            test_case.qrng_dimensions = dim;
            test_case.offset          = offset;
            run_case(test_case);
        }
    }
}

int main()
{
    // CSV header
    std::cout << "generator,size,seed,dims,offset,result" << std::endl;

    constexpr generator_type generator_types[] = {generator_type::XORWOW,
                                                  generator_type::MRG32K3A,
                                                  generator_type::MTGP32,
                                                  generator_type::PHILOX4_32_10,
                                                  generator_type::MT19937,
                                                  generator_type::SOBOL32,
                                                  generator_type::SCRAMBLED_SOBOL32,
                                                  generator_type::SOBOL64,
                                                  generator_type::SCRAMBLED_SOBOL64};

    for(const generator_type rng_type : generator_types)
    {
        for(const long long offset : offsets)
        {
            test_offset(rng_type, DEFAULT_RAND_N, offset);
            if(!generator_supports_offset(rng_type))
                break;
        }
    }
}
