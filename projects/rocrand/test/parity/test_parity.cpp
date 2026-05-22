// MIT License
//
// Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "parity_curand.hpp"
#include "parity_rocrand.hpp"

#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#ifndef DEFAULT_RAND_N
    #define DEFAULT_RAND_N (1024 * 1024 * 128)
#endif

class GeneratorName
{
public:
    static const std::string Print(const generator_type& rng_type)
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
        return "unknown";
    }
};

std::string GeneratorTestName(const ::testing::TestParamInfo<generator_type>& info)
{
    return GeneratorName::Print(info.param);
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
static void
    run_test(const test_case& test_case, F generate_rocrand, G generate_curand, bool should_match)
{
    const std::vector<T> rocrand_results = generate_rocrand(test_case);
    const std::vector<T> curand_results  = generate_curand(test_case);

    ASSERT_EQ(rocrand_results.size(), curand_results.size());

    for(size_t i = 0; i < rocrand_results.size(); ++i)
    {
        if(should_match)
        {
            EXPECT_EQ(rocrand_results[i], curand_results[i]) << "Mismatch at index " << i;
        }
        else if(rocrand_results[i] != curand_results[i])
        {
            std::cout << "Expected failure: "
                      << "mismatch at index " << i << std::endl;
            return;
        }
    }
}

static void run_case(const test_case& test_case, bool should_match)
{
    if(generator_is_64bit(test_case.rng_type))
    {
        run_test<unsigned long long>(test_case,
                                     rocrand_parity::test_generate_long_long,
                                     curand_parity::test_generate_long_long,
                                     should_match);
    }
    else
    {
        run_test<unsigned int>(test_case,
                               rocrand_parity::test_generate,
                               curand_parity::test_generate,
                               should_match);
    }
}

class ParityTest : public ::testing::TestWithParam<generator_type>
{};

TEST_P(ParityTest, BasicParity)
{
    constexpr long long      seeds[]   = {-1ll, 12345ll};
    constexpr int            dims[]    = {1, 3};
    constexpr long long      offsets[] = {-1ll, 112121116104111110ll};
    constexpr generator_type excpected_parity[]
        = {generator_type::SOBOL32, generator_type::SOBOL64};

    const generator_type rng_type = GetParam();

    const bool should_match
        = std::find(std::begin(excpected_parity), std::end(excpected_parity), rng_type)
          != std::end(excpected_parity);

    for(const long long offset : offsets)
    {
        SCOPED_TRACE(testing::Message() << "with offset = " << offset);
        const size_t size = DEFAULT_RAND_N;
        if(generator_is_pseudo(rng_type))
        {
            for(const long long seed : seeds)
            {
                SCOPED_TRACE(testing::Message() << "with seed = " << seed);
                test_case test_case;
                test_case.rng_type  = rng_type;
                test_case.size      = size;
                test_case.prng_seed = seed;
                test_case.offset    = offset;
                run_case(test_case, should_match);
            }
        }
        else
        {
            for(const long long dim : dims)
            {
                SCOPED_TRACE(testing::Message() << "with dim = " << dim);
                test_case test_case;
                test_case.rng_type = rng_type;
                // Round down to multiple of dimensions
                test_case.size            = size - size % dim;
                test_case.qrng_dimensions = dim;
                test_case.offset          = offset;
                run_case(test_case, should_match);
            }
        }

        if(!generator_supports_offset(rng_type))
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(AllGenerators,
                         ParityTest,
                         ::testing::Values(generator_type::XORWOW,
                                           generator_type::MRG32K3A,
                                           generator_type::MTGP32,
                                           generator_type::PHILOX4_32_10,
                                           generator_type::MT19937,
                                           generator_type::SOBOL32,
                                           generator_type::SCRAMBLED_SOBOL32,
                                           generator_type::SOBOL64,
                                           generator_type::SCRAMBLED_SOBOL64),
                         GeneratorTestName);
