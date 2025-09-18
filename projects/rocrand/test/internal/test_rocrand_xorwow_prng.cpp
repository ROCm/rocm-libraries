// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_rocrand_prng.hpp"
#include <rocrand/rocrand.h>

#include <rng/xorwow.hpp>

#include <gtest/gtest.h>

using rocrand_impl::host::xorwow_generator;

// Generator API tests
using xorwow_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<xorwow_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<xorwow_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using xorwow_generator_prng_offset_tests_types = ::testing::Types<
    generator_prng_offset_tests_params<unsigned int,
                                       xorwow_generator,
                                       ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<unsigned int,
                                       xorwow_generator,
                                       ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_offset_tests_params<float, xorwow_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<float, xorwow_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(xorwow_generator,
                               generator_prng_tests,
                               xorwow_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(xorwow_generator,
                               generator_prng_continuity_tests,
                               xorwow_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(xorwow_generator,
                               generator_prng_offset_tests,
                               xorwow_generator_prng_offset_tests_types);

// Engine API tests
class xorwow_engine_type_test : public xorwow_generator::engine_type
{};

TEST(xorwow_engine_type_test, discard_test)
{
    const unsigned long long      seed = 1234567890123ULL;
    xorwow_generator::engine_type engine1(seed, 0, 678ULL);
    xorwow_generator::engine_type engine2(seed, 0, 677ULL);

    (void)engine2.next();

    EXPECT_EQ(engine1(), engine2());

    const unsigned long long ds[] = {1ULL,
                                     4ULL,
                                     37ULL,
                                     583ULL,
                                     7452ULL,
                                     21032ULL,
                                     35678ULL,
                                     66778ULL,
                                     10313475ULL,
                                     82120230ULL};

    for(auto d : ds)
    {
        for(unsigned long long i = 0; i < d; i++)
        {
            (void)engine1.next();
        }
        engine2.discard(d);

        EXPECT_EQ(engine1(), engine2());
    }
}

TEST(xorwow_engine_type_test, discard_sequence_test)
{
    const unsigned long long      seed = ~1234567890123ULL;
    xorwow_generator::engine_type engine1(seed, 0, 444ULL);
    xorwow_generator::engine_type engine2(seed, 123ULL, 444ULL);

    engine1.discard_subsequence(123ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard(5356446450ULL);
    engine1.discard_subsequence(123ULL);
    engine1.discard(30000000006ULL);

    engine2.discard_subsequence(3ULL);
    engine2.discard(35356446456ULL);
    engine2.discard_subsequence(120ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard_subsequence(3456000ULL);
    engine1.discard_subsequence(1000005ULL);

    engine2.discard_subsequence(4456005ULL);

    EXPECT_EQ(engine1(), engine2());
}
