// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_rocrand_prng.hpp"
#include "test_rocrand_threefryNx64_20_prng.hpp"
#include <rocrand/rocrand.h>

#include <rng/threefry.hpp>

using rocrand_impl::host::threefry4x64_20_generator;

// Generator API tests
using threefry4x64_20_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<threefry4x64_20_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<threefry4x64_20_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using threefry4x64_20_generator_prng_offset_tests_types
    = ::testing::Types<generator_prng_offset_tests_params<unsigned long long,
                                                          threefry4x64_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_tests_params<unsigned long long,
                                                          threefry4x64_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
                       generator_prng_offset_tests_params<float,
                                                          threefry4x64_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_tests_params<float,
                                                          threefry4x64_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x64_20_generator,
                               generator_prng_tests,
                               threefry4x64_20_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x64_20_generator,
                               generator_prng_continuity_tests,
                               threefry4x64_20_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x64_20_generator,
                               generator_prng_offset_tests,
                               threefry4x64_20_generator_prng_offset_tests_types);

// threefry4x64_20-specific generator API tests
INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x64_20_generator,
                               threefryNx64_20_generator_prng_tests,
                               threefry4x64_20_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x64_20_generator,
                               threefryNx64_20_generator_prng_continuity_tests,
                               threefry4x64_20_generator_prng_tests_types);

// Engine API tests
class threefry4x64_engine_type_test : public threefry4x64_20_generator::engine_type
{
public:
    __host__ threefry4x64_engine_type_test() : threefry4x64_20_generator::engine_type(0, 0, 0) {}

    __host__ state_type& internal_state_ref()
    {
        return m_state;
    }
};

TEST(threefry_prng_state_tests, seed_test)
{
    threefry4x64_engine_type_test              engine;
    threefry4x64_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 0ULL);
    EXPECT_EQ(state.substate, 0U);

    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 1ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 0ULL);
    EXPECT_EQ(state.substate, 0U);

    engine.seed(3331, 0, 5 * 4ULL);
    EXPECT_EQ(state.counter.x, 5ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 0ULL);
    EXPECT_EQ(state.substate, 0U);
}

// Check if the threefry state counter is calculated correctly during
// random number generation.
TEST(threefry_prng_state_tests, discard_test)
{
    threefry4x64_engine_type_test              engine;
    threefry4x64_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 0ULL);
    EXPECT_EQ(state.substate, 0ULL);

    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    EXPECT_EQ(state.counter.x, ULLONG_MAX);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 0ULL);
    EXPECT_EQ(state.substate, 0ULL);

    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    EXPECT_EQ(state.counter.x, ULLONG_MAX - 1ULL);
    EXPECT_EQ(state.counter.y, 1ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 0ULL);
    EXPECT_EQ(state.substate, 0ULL);

    engine.discard(2 * 4ULL);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 2ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 0ULL);
    EXPECT_EQ(state.substate, 0ULL);

    state.counter.x = ULLONG_MAX;
    state.counter.y = ULLONG_MAX;
    state.counter.z = ULLONG_MAX;
    state.counter.w = 0ULL;
    state.substate  = 0ULL;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 1ULL);
    EXPECT_EQ(state.substate, 0ULL);

    state.counter.x = ULLONG_MAX;
    state.counter.y = ULLONG_MAX;
    state.counter.z = ULLONG_MAX;
    state.counter.w = 0ULL;
    state.substate  = 0ULL;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 1ULL);
    EXPECT_EQ(state.substate, 0ULL);

    state.counter.x = 123ULL;
    state.counter.y = 456ULL;
    state.counter.z = 789ULL;
    state.counter.w = 999ULL;
    state.substate  = 0ULL;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124ULL);
    EXPECT_EQ(state.counter.y, 456ULL);
    EXPECT_EQ(state.counter.z, 789ULL);
    EXPECT_EQ(state.counter.w, 999ULL);
    EXPECT_EQ(state.substate, 0ULL);

    state.counter.x = 123ULL;
    state.counter.y = 0ULL;
    state.counter.z = 0ULL;
    state.counter.w = 0ULL;
    state.substate  = 0ULL;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 0ULL);
    EXPECT_EQ(state.substate, 0ULL);

    state.counter.x = ULLONG_MAX - 1;
    state.counter.y = 2ULL;
    state.counter.z = 3ULL;
    state.counter.w = 4ULL;
    state.substate  = 0ULL;
    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    engine.discard(12ULL);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 4ULL);
    EXPECT_EQ(state.counter.z, 3ULL);
    EXPECT_EQ(state.counter.w, 4ULL);
    EXPECT_EQ(state.substate, 0ULL);
}

TEST(threefry_prng_state_tests, discard_sequence_test)
{
    threefry4x64_engine_type_test              engine;
    threefry4x64_engine_type_test::state_type& state = engine.internal_state_ref();

    engine.discard_subsequence(ULLONG_MAX);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, ULLONG_MAX);
    EXPECT_EQ(state.counter.w, 0ULL);
    EXPECT_EQ(state.substate, 0U);

    engine.discard_subsequence(ULLONG_MAX);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, ULLONG_MAX - 1);
    EXPECT_EQ(state.counter.w, 1ULL);
    EXPECT_EQ(state.substate, 0U);

    engine.discard_subsequence(2);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 2ULL);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 123ULL;
    state.counter.y = 456ULL;
    state.counter.z = 789ULL;
    state.counter.w = 999ULL;
    state.substate  = 0U;
    engine.discard_subsequence(1);
    EXPECT_EQ(state.counter.x, 123ULL);
    EXPECT_EQ(state.counter.y, 456ULL);
    EXPECT_EQ(state.counter.z, 790ULL);
    EXPECT_EQ(state.counter.w, 999ULL);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 1ULL;
    state.counter.y = 2ULL;
    state.counter.z = ULLONG_MAX - 1;
    state.counter.w = 4ULL;
    state.substate  = 0U;
    engine.discard_subsequence(2);
    engine.discard_subsequence(ULLONG_MAX);
    engine.discard_subsequence(1);
    EXPECT_EQ(state.counter.x, 1ULL);
    EXPECT_EQ(state.counter.y, 2ULL);
    EXPECT_EQ(state.counter.z, 0ULL);
    EXPECT_EQ(state.counter.w, 6ULL);
    EXPECT_EQ(state.substate, 0U);
}
