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
#include "test_rocrand_threefryNx32_20_prng.hpp"
#include <rocrand/rocrand.h>

#include <rng/threefry.hpp>

using rocrand_impl::host::threefry4x32_20_generator;

// Generator API tests
using threefry4x32_20_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<threefry4x32_20_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<threefry4x32_20_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using threefry4x32_20_generator_prng_offset_tests_types
    = ::testing::Types<generator_prng_offset_tests_params<unsigned int,
                                                          threefry4x32_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_tests_params<unsigned int,
                                                          threefry4x32_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
                       generator_prng_offset_tests_params<float,
                                                          threefry4x32_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_tests_params<float,
                                                          threefry4x32_20_generator,
                                                          ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x32_20_generator,
                               generator_prng_tests,
                               threefry4x32_20_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x32_20_generator,
                               generator_prng_continuity_tests,
                               threefry4x32_20_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x32_20_generator,
                               generator_prng_offset_tests,
                               threefry4x32_20_generator_prng_offset_tests_types);

#ifdef CODE_COVERAGE_ENABLED
#include "test_rocrand_host_prng.hpp"

using rocrand_impl::host::threefry4x32_20_generator_host;
using threefry4x32_20_generator_prng_host_tests_types
    = ::testing::Types<generator_prng_host_tests_params<threefry4x32_20_generator_host<true>,
                                                        ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

using threefry4x32_20_generator_prng_offset_host_tests_types
    = ::testing::Types<generator_prng_offset_host_tests_params<unsigned int,
                                                               threefry4x32_20_generator_host<true>,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_host_tests_params<float,
                                                               threefry4x32_20_generator_host<true>,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x32_20_host_generator,
                               generator_prng_host_tests,
                               threefry4x32_20_generator_prng_host_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x32_20_host_generator,
                               generator_prng_continuity_host_tests,
                               threefry4x32_20_generator_prng_host_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x32_20_host_generator,
                               generator_prng_offset_host_tests,
                               threefry4x32_20_generator_prng_offset_host_tests_types);
#endif //CODE_COVERAGE_ENABLED

// threefry4x32_20-specific generator API tests
INSTANTIATE_TYPED_TEST_SUITE_P(threefry4x32_20_generator,
                               threefryNx32_20_generator_prng_tests,
                               threefry4x32_20_generator_prng_tests_types);

// Engine API tests
class threefry4x32_engine_type_test : public threefry4x32_20_generator::engine_type
{
public:
    __host__ threefry4x32_engine_type_test() : threefry4x32_20_generator::engine_type(0, 0, 0) {}

    __host__
    state_type& internal_state_ref()
    {
        return m_state;
    }
};

TEST(threefry_prng_state_tests, seed_test)
{
    threefry4x32_engine_type_test              engine;
    threefry4x32_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 1U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.seed(3331, 0, 5 * 4ULL);
    EXPECT_EQ(state.counter.x, 5U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);
}

// Check if the threefry state counter is calculated correctly during
// random number generation.
TEST(threefry_prng_state_tests, discard_test)
{
    threefry4x32_engine_type_test              engine;
    threefry4x32_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard(UINT_MAX * 4ULL);
    EXPECT_EQ(state.counter.x, UINT_MAX);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard(UINT_MAX * 4ULL);
    EXPECT_EQ(state.counter.x, UINT_MAX - 1);
    EXPECT_EQ(state.counter.y, 1U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard(2 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 2U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = UINT_MAX;
    state.counter.y = UINT_MAX;
    state.counter.z = UINT_MAX;
    state.counter.w = 0;
    state.substate  = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 1U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = UINT_MAX;
    state.counter.y = UINT_MAX;
    state.counter.z = UINT_MAX;
    state.counter.w = 1;
    state.substate  = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 2U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 123;
    state.counter.y = 456;
    state.counter.z = 789;
    state.counter.w = 999;
    state.substate  = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124U);
    EXPECT_EQ(state.counter.y, 456U);
    EXPECT_EQ(state.counter.z, 789U);
    EXPECT_EQ(state.counter.w, 999U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 123;
    state.counter.y = 0;
    state.counter.z = 0;
    state.counter.w = 0;
    state.substate  = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = UINT_MAX - 1;
    state.counter.y = 2;
    state.counter.z = 3;
    state.counter.w = 4;
    state.substate  = 0;
    engine.discard(((1ull << 32) + 2ull) * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 4U);
    EXPECT_EQ(state.counter.z, 3U);
    EXPECT_EQ(state.counter.w, 4U);
    EXPECT_EQ(state.substate, 0U);
}

TEST(threefry_prng_state_tests, discard_sequence_test)
{
    threefry4x32_engine_type_test              engine;
    threefry4x32_engine_type_test::state_type& state = engine.internal_state_ref();

    engine.discard_subsequence(UINT_MAX);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, UINT_MAX);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard_subsequence(UINT_MAX);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, UINT_MAX - 1);
    EXPECT_EQ(state.counter.w, 1U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard_subsequence(2);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 2U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 123;
    state.counter.y = 456;
    state.counter.z = 789;
    state.counter.w = 999;
    state.substate  = 0;
    engine.discard_subsequence(1);
    EXPECT_EQ(state.counter.x, 123U);
    EXPECT_EQ(state.counter.y, 456U);
    EXPECT_EQ(state.counter.z, 790U);
    EXPECT_EQ(state.counter.w, 999U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 1;
    state.counter.y = 2;
    state.counter.z = UINT_MAX - 1;
    state.counter.w = 4;
    state.substate  = 0;
    engine.discard_subsequence((1ull << 32) + 2ull);
    EXPECT_EQ(state.counter.x, 1U);
    EXPECT_EQ(state.counter.y, 2U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 6U);
    EXPECT_EQ(state.substate, 0U);
}

TEST(threefry_additional_tests, rocrand_init_test)
{
    // making sure the outputs are the same when initialized with same parameters
    rocrand_state_threefry4x32_20 state1, state2;

    using ull = unsigned long long;

    ull seeds[]        = {0, 123, 321, 123456, 654321};
    ull subsequences[] = {0xf, 0xff, 0x1f, 0x1ff, 0x1f1};
    ull offsets[]      = {0, 1, 2, 3, 4};

    for(int i = 0; i < 5; i++)
    {
        rocrand_init(seeds[i], subsequences[i], offsets[i], &state1);
        rocrand_init(seeds[i], subsequences[i], offsets[i], &state2);

        for(int j = 0; j < 5000; j++)
            ASSERT_EQ(rocrand(&state1), rocrand(&state2));
    }
}

TEST(threefry_additional_tests, rocrand_test)
{
    // making sure the outputs are uniformly distributed!
    rocrand_state_threefry4x32_20 state;

    rocrand_init(0, 0, 0, &state);
    size_t testSize = 40000;

    unsigned int* output = new unsigned int[testSize];

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

    double maxi  = (double)std::numeric_limits<unsigned int>::max();
    double eMean = 0.5 * (maxi); // 0.5(a + b)
    double eStd  = (maxi) / (2 * std::sqrt(3)); // (b - a) / (2*3^0.5)

    ASSERT_NEAR(mean, eMean, eMean * 0.1);
    ASSERT_NEAR(std, eStd, eStd * 0.1);

    delete[] output;
}

TEST(threefry_additional_tests, rocrand4_test)
{
    // making sure the outputs are uniformly distributed!
    rocrand_state_threefry4x32_20 state;

    rocrand_init(0, 0, 0, &state);
    size_t testSize = 40000;

    unsigned int* output = new unsigned int[testSize];

    double mean = 0;
    for(size_t i = 0; i < testSize; i += 4)
    {
        uint4 t       = rocrand4(&state);
        output[i]     = t.w;
        output[i + 1] = t.x;
        output[i + 2] = t.y;
        output[i + 3] = t.z;
        mean += static_cast<double>(output[i]);
        mean += static_cast<double>(output[i + 1]);
        mean += static_cast<double>(output[i + 2]);
        mean += static_cast<double>(output[i + 3]);
    }
    mean /= testSize;

    double std = 0.0;
    for(size_t i = 0; i < testSize; i++)
        std += std::pow(output[i] - mean, 2);

    std = std::sqrt(std / testSize);

    double maxi = (double)std::numeric_limits<unsigned int>::max();
    // min val is 0
    double eMean = 0.5 * (maxi); // 0.5(a + b)
    double eStd  = (maxi) / (2 * std::sqrt(3)); // (b - a) / (2*3^0.5)

    ASSERT_NEAR(mean, eMean, eMean * 0.1);
    ASSERT_NEAR(std, eStd, eStd * 0.1);

    delete[] output;
}
