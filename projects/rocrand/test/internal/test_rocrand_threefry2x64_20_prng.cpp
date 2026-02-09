// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_rocrand_host_prng.hpp"
#include "test_rocrand_prng.hpp"
#include "test_rocrand_threefryNx64_20_prng.hpp"
#include <rocrand/rocrand.h>

#include <rng/threefry.hpp>

using rocrand_impl::host::threefry2x64_20_generator;

// Generator API tests
using threefry2x64_20_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<threefry2x64_20_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<threefry2x64_20_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using threefry2x64_20_generator_prng_offset_tests_types
    = ::testing::Types<generator_prng_offset_tests_params<unsigned long long,
                                                          threefry2x64_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_tests_params<unsigned long long,
                                                          threefry2x64_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
                       generator_prng_offset_tests_params<float,
                                                          threefry2x64_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_tests_params<float,
                                                          threefry2x64_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(threefry2x64_20_generator,
                               generator_prng_tests,
                               threefry2x64_20_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry2x64_20_generator,
                               generator_prng_continuity_tests,
                               threefry2x64_20_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry2x64_20_generator,
                               generator_prng_offset_tests,
                               threefry2x64_20_generator_prng_offset_tests_types);

#ifdef CODE_COVERAGE_ENABLED
#include "test_rocrand_host_prng.hpp"

using rocrand_impl::host::threefry2x64_20_generator_host;
using threefry2x64_20_generator_prng_host_tests_types
    = ::testing::Types<generator_prng_host_tests_params<threefry2x64_20_generator_host<true>,
                                                        ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

using threefry2x64_20_generator_prng_offset_host_tests_types
    = ::testing::Types<generator_prng_offset_host_tests_params<unsigned long long,
                                                               threefry2x64_20_generator_host<true>,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_host_tests_params<float,
                                                               threefry2x64_20_generator_host<true>,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

INSTANTIATE_TYPED_TEST_SUITE_P(threefry2x64_20_host_generator,
                               generator_prng_host_tests,
                               threefry2x64_20_generator_prng_host_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry2x64_20_host_generator,
                               generator_prng_continuity_host_tests,
                               threefry2x64_20_generator_prng_host_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry2x64_20_host_generator,
                               generator_prng_offset_host_tests,
                               threefry2x64_20_generator_prng_offset_host_tests_types);
#endif //CODE_COVERAGE_ENABLED
// threefry2x64_20-specific generator API tests
INSTANTIATE_TYPED_TEST_SUITE_P(threefry2x64_20_generator,
                               threefryNx64_20_generator_prng_tests,
                               threefry2x64_20_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry2x64_20_generator,
                               threefryNx64_20_generator_prng_continuity_tests,
                               threefry2x64_20_generator_prng_tests_types);

// Engine API tests
class threefry2x64_engine_type_test : public threefry2x64_20_generator::engine_type
{
public:
    __host__ threefry2x64_engine_type_test() : threefry2x64_20_generator::engine_type(0, 0, 0) {}

    __host__
    state_type& internal_state_ref()
    {
        return m_state;
    }
};

TEST(threefry_prng_state_tests, seed_test)
{
    threefry2x64_engine_type_test              engine;
    threefry2x64_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 0ULL);

    engine.discard(1 * 2ULL);
    EXPECT_EQ(state.counter.x, 1ULL);
    EXPECT_EQ(state.counter.y, 0ULL);

    engine.seed(3331, 0, 5 * 2ULL);
    EXPECT_EQ(state.counter.x, 5ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
}

// Check if the threefry state counter is calculated correctly during
// random number generation.
TEST(threefry_prng_state_tests, discard_test)
{
    threefry2x64_engine_type_test              engine;
    threefry2x64_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.substate, 0ULL);

    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    EXPECT_EQ(state.counter.x, ULLONG_MAX);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.substate, 0ULL);

    engine.discard(ULLONG_MAX - 1ULL);
    engine.discard(ULLONG_MAX - 1ULL);
    EXPECT_EQ(state.counter.x, ULLONG_MAX - 2ULL);
    EXPECT_EQ(state.counter.y, 1ULL);
    EXPECT_EQ(state.substate, 0ULL);

    engine.discard(3 * 2ULL);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 2ULL);
    EXPECT_EQ(state.substate, 0ULL);

    state.counter.x = 123;
    state.counter.y = 456;
    state.substate  = 0;
    engine.discard(1 * 2ULL);
    EXPECT_EQ(state.counter.x, 124ULL);
    EXPECT_EQ(state.counter.y, 456ULL);
    EXPECT_EQ(state.substate, 0ULL);

    state.counter.x = 123;
    state.counter.y = 0;
    state.substate  = 0;
    engine.discard(1 * 2ULL);
    EXPECT_EQ(state.counter.x, 124ULL);
    EXPECT_EQ(state.counter.y, 0ULL);
    EXPECT_EQ(state.substate, 0ULL);

    state.counter.x = ULLONG_MAX - 1;
    state.counter.y = 2;
    state.substate  = 0;
    engine.discard(2ULL);
    engine.discard(ULLONG_MAX);
    engine.discard(ULLONG_MAX);
    engine.discard(4ULL);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, 4ULL);
    EXPECT_EQ(state.substate, 0ULL);
}

TEST(threefry_prng_state_tests, discard_sequence_test)
{
    threefry2x64_engine_type_test              engine;
    threefry2x64_engine_type_test::state_type& state = engine.internal_state_ref();

    engine.discard_subsequence(ULLONG_MAX);
    EXPECT_EQ(state.counter.x, 0ULL);
    EXPECT_EQ(state.counter.y, ULLONG_MAX);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 123;
    state.counter.y = 456;
    state.substate  = 0;
    engine.discard_subsequence(1);
    EXPECT_EQ(state.counter.x, 123ULL);
    EXPECT_EQ(state.counter.y, 457ULL);
    EXPECT_EQ(state.substate, 0U);
}

TEST(threefry_additional_tests, rocrand_test)
{
    // making sure the outputs are uniformly distributed!
    rocrand_state_threefry2x64_20 state;

    rocrand_init(0, 0, 0, &state);
    size_t testSize = 40000;

    unsigned long long* output = new unsigned long long[testSize];

    double mean = 0;
    for(size_t i = 0; i < testSize; i++)
    {
        output[i] = rocrand(&state);
        mean += static_cast<double>(output[i]);
    }
    mean /= testSize;

    double std = 0.0;
    for(size_t i = 0; i < testSize; i++)
        std += std::pow(output[i] - mean, 2);

    std = std::sqrt(std / testSize);

    double maxi  = (double)std::numeric_limits<unsigned long long>::max();
    double eMean = 0.5 * (maxi); // 0.5(a + b)
    double eStd  = (maxi) / (2 * std::sqrt(3)); // (b - a) / (2*3^0.5)

    ASSERT_NEAR(mean, eMean, eMean * 0.1);
    ASSERT_NEAR(std, eStd, eStd * 0.1);

    delete[] output;
}

TEST(threefry_additional_tests, rocrand2_test)
{
    // making sure the outputs are uniformly distributed!
    rocrand_state_threefry2x64_20 state;

    rocrand_init(0, 0, 0, &state);
    size_t testSize = 40000;

    unsigned long long* output = new unsigned long long[testSize];

    double mean = 0;
    for(size_t i = 0; i < testSize; i += 2)
    {
        ulonglong2 t  = rocrand2(&state);
        output[i]     = t.x;
        output[i + 1] = t.y;
        mean += static_cast<double>(output[i]);
        mean += static_cast<double>(output[i + 1]);
    }
    mean /= testSize;

    double std = 0.0;
    for(size_t i = 0; i < testSize; i++)
        std += std::pow(output[i] - mean, 2);

    std = std::sqrt(std / testSize);

    double maxi = (double)std::numeric_limits<unsigned long long>::max();
    // min val is 0
    double eMean = 0.5 * (maxi); // 0.5(a + b)
    double eStd  = (maxi) / (2 * std::sqrt(3)); // (b - a) / (2*3^0.5)

    ASSERT_NEAR(mean, eMean, eMean * 0.1);
    ASSERT_NEAR(std, eStd, eStd * 0.1);

    delete[] output;
}
