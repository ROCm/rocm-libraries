// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rng/philox4x32_10.hpp>

#include <gtest/gtest.h>

using rocrand_impl::host::philox4x32_10_generator;

// Generator API tests
using philox_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<philox4x32_10_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<philox4x32_10_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using philox_generator_prng_offset_tests_types
    = ::testing::Types<generator_prng_offset_tests_params<unsigned int,
                                                          philox4x32_10_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_tests_params<unsigned int,
                                                          philox4x32_10_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
                       generator_prng_offset_tests_params<float,
                                                          philox4x32_10_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_tests_params<float,
                                                          philox4x32_10_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(philox4x32_10_generator,
                               generator_prng_tests,
                               philox_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(philox4x32_10_generator,
                               generator_prng_continuity_tests,
                               philox_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(philox4x32_10_generator,
                               generator_prng_offset_tests,
                               philox_generator_prng_offset_tests_types);
#ifdef CODE_COVERAGE_ENABLED
#include "test_rocrand_host_prng.hpp"

using rocrand_impl::host::philox4x32_10_generator_host;
using philox4x32_10_generator_prng_host_tests_types
    = ::testing::Types<generator_prng_host_tests_params<philox4x32_10_generator_host<true>,
                                                        ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

using philox4x32_10_generator_prng_offset_host_tests_types
    = ::testing::Types<generator_prng_offset_host_tests_params<unsigned int,
                                                               philox4x32_10_generator_host<true>,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_host_tests_params<float,
                                                               philox4x32_10_generator_host<true>,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

INSTANTIATE_TYPED_TEST_SUITE_P(philox4x32_10_host_generator,
                               generator_prng_host_tests,
                               philox4x32_10_generator_prng_host_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(philox4x32_10_host_generator,
                               generator_prng_continuity_host_tests,
                               philox4x32_10_generator_prng_host_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(philox4x32_10_host_generator,
                               generator_prng_offset_host_tests,
                               philox4x32_10_generator_prng_offset_host_tests_types);
#endif //CODE_COVERAGE_ENABLED

// Engine API tests
class philox4x32_10_engine_type_test : public philox4x32_10_generator::engine_type
{
public:
    __host__ philox4x32_10_engine_type_test() : philox4x32_10_generator::engine_type(0, 0, 0) {}

    __host__
    state_type& internal_state_ref()
    {
        return m_state;
    }
};

TEST(philox_prng_state_tests, seed_test)
{
    philox4x32_10_engine_type_test              engine;
    philox4x32_10_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 1U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.seed(3331, 0, 5 * 4ULL);
    EXPECT_EQ(state.counter.x, 5U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
}

// Check if the philox state counter is calculated correctly during
// random number generation.
TEST(philox_prng_state_tests, discard_test)
{
    philox4x32_10_engine_type_test              engine;
    philox4x32_10_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard(UINT_MAX * 4ULL);
    EXPECT_EQ(state.counter.x, UINT_MAX);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard(UINT_MAX * 4ULL);
    EXPECT_EQ(state.counter.x, UINT_MAX - 1);
    EXPECT_EQ(state.counter.y, 1U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard(2 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 2U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    state.counter.x = UINT_MAX;
    state.counter.y = UINT_MAX;
    state.counter.z = UINT_MAX;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 1U);

    state.counter.x = UINT_MAX;
    state.counter.y = UINT_MAX;
    state.counter.z = UINT_MAX;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 2U);

    state.counter.x = 123;
    state.counter.y = 456;
    state.counter.z = 789;
    state.counter.w = 999;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124U);
    EXPECT_EQ(state.counter.y, 456U);
    EXPECT_EQ(state.counter.z, 789U);
    EXPECT_EQ(state.counter.w, 999U);

    state.counter.x = 123;
    state.counter.y = 0;
    state.counter.z = 0;
    state.counter.w = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    state.counter.x = UINT_MAX - 1;
    state.counter.y = 2;
    state.counter.z = 3;
    state.counter.w = 4;
    engine.discard(((1ull << 32) + 2ull) * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 4U);
    EXPECT_EQ(state.counter.z, 3U);
    EXPECT_EQ(state.counter.w, 4U);
}

TEST(philox_prng_state_tests, discard_sequence_test)
{
    philox4x32_10_engine_type_test              engine;
    philox4x32_10_engine_type_test::state_type& state = engine.internal_state_ref();

    engine.discard_subsequence(UINT_MAX);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, UINT_MAX);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard_subsequence(UINT_MAX);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, UINT_MAX - 1);
    EXPECT_EQ(state.counter.w, 1U);

    engine.discard_subsequence(2);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 2U);

    state.counter.x = 123;
    state.counter.y = 456;
    state.counter.z = 789;
    state.counter.w = 999;
    engine.discard_subsequence(1);
    EXPECT_EQ(state.counter.x, 123U);
    EXPECT_EQ(state.counter.y, 456U);
    EXPECT_EQ(state.counter.z, 790U);
    EXPECT_EQ(state.counter.w, 999U);

    state.counter.x = 1;
    state.counter.y = 2;
    state.counter.z = UINT_MAX - 1;
    state.counter.w = 4;
    engine.discard_subsequence((1ull << 32) + 2ull);
    EXPECT_EQ(state.counter.x, 1U);
    EXPECT_EQ(state.counter.y, 2U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 6U);
}
