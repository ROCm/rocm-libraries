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

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_rocrand_prng.hpp"
#include <rocrand/rocrand.h>

#include <rng/lfsr113.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <stdexcept>

using rocrand_impl::host::lfsr113_generator;

// Generator API tests
using lfsr113_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<lfsr113_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<lfsr113_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using lfsr113_generator_prng_offset_tests_types = ::testing::Types<
    generator_prng_offset_tests_params<unsigned int,
                                       lfsr113_generator,
                                       ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<unsigned int,
                                       lfsr113_generator,
                                       ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_offset_tests_params<float, lfsr113_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<float, lfsr113_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(lfsr113_generator,
                               generator_prng_tests,
                               lfsr113_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(lfsr113_generator,
                               generator_prng_continuity_tests,
                               lfsr113_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(lfsr113_generator,
                               generator_prng_offset_tests,
                               lfsr113_generator_prng_offset_tests_types);

#ifdef CODE_COVERAGE_ENABLED
#include "test_rocrand_host_prng.hpp"

using rocrand_impl::host::lfsr113_generator_host;
using lfsr113_generator_prng_host_tests_types
    = ::testing::Types<generator_prng_host_tests_params<lfsr113_generator_host<true>,
                                                        ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

using lfsr113_generator_prng_offset_host_tests_types
    = ::testing::Types<generator_prng_offset_host_tests_params<unsigned int,
                                                               lfsr113_generator_host<true>,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       generator_prng_offset_host_tests_params<float,
                                                               lfsr113_generator_host<true>,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

INSTANTIATE_TYPED_TEST_SUITE_P(lfsr113_host_generator,
                               generator_prng_host_tests,
                               lfsr113_generator_prng_host_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(lfsr113_host_generator,
                               generator_prng_continuity_host_tests,
                               lfsr113_generator_prng_host_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(lfsr113_host_generator,
                               generator_prng_offset_host_tests,
                               lfsr113_generator_prng_offset_host_tests_types);
#endif //CODE_COVERAGE_ENABLED

// lfsr113-specific generator API tests
template<class Params>
struct lfsr113_generator_prng_tests : public testing::Test
{
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;

    auto get_generator() const
    {
        generator_t g;
        if(g.set_order(ordering) != ROCRAND_STATUS_SUCCESS)
        {
            throw std::runtime_error("Could not set ordering for generator");
        }
        return g;
    }
};

TYPED_TEST_SUITE(lfsr113_generator_prng_tests, lfsr113_generator_prng_tests_types);

TYPED_TEST(lfsr113_generator_prng_tests, different_seed_test)
{
    const unsigned long long seed0 = 5ULL;
    const unsigned long long seed1 = 10ULL;

    // Device side data
    const size_t  size = 1024;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    // Generators
    auto g0 = TestFixture::get_generator(), g1 = TestFixture::get_generator();
    // Set different seeds
    g0.set_seed(seed0);
    g1.set_seed(seed1);

    const uint4 get_seed0 = g0.get_seed_uint4();
    const uint4 get_seed1 = g1.get_seed_uint4();

    ASSERT_NE(get_seed0.x, get_seed1.x);

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g0_host_data[size];
    HIP_CHECK(hipMemcpy(g0_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g1 and copy to host
    ROCRAND_CHECK(g1.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g1_host_data[size];
    HIP_CHECK(hipMemcpy(g1_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    size_t same = 0;
    for(size_t i = 0; i < size; i++)
    {
        if(g1_host_data[i] == g0_host_data[i])
            same++;
    }
    // It may happen that numbers are the same, so we
    // just make sure that most of them are different.
    EXPECT_LT(same, static_cast<size_t>(0.01f * size));

    HIP_CHECK(hipFree(data));
}

// Checks if generators with the same seed and in the same state generate
// the same numbers
TYPED_TEST(lfsr113_generator_prng_tests, different_seed_uint4_test)
{
    const uint4 seeds0[] = {
        { 0,  2,  4,  6},
        { 0,  2,  4,  6},
        {11, 12, 13, 14}
    };
    const uint4 seeds1[] = {
        { 3,  5, 22, 1000},
        {10, 30, 60,  900},
        {12, 13, 14,  150}
    };

    for(unsigned int i = 0; i < 3; i++)
    {
        const uint4 seed0 = seeds0[i];
        const uint4 seed1 = seeds1[i];

        // Device side data
        const size_t  size = 1024;
        unsigned int* data;
        HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

        // Generators
        auto g0 = TestFixture::get_generator(), g1 = TestFixture::get_generator();
        // Set different seeds
        g0.set_seed_uint4(seed0);
        g1.set_seed_uint4(seed1);

        const uint4 get_seed0 = g0.get_seed_uint4();
        const uint4 get_seed1 = g1.get_seed_uint4();

        ASSERT_NE(get_seed0.x, get_seed1.x);
        ASSERT_NE(get_seed0.y, get_seed1.y);
        ASSERT_NE(get_seed0.z, get_seed1.z);
        ASSERT_NE(get_seed0.w, get_seed1.w);

        // Generate using g0 and copy to host
        ROCRAND_CHECK(g0.generate(data, size));
        HIP_CHECK(hipDeviceSynchronize());

        unsigned int g0_host_data[size];
        HIP_CHECK(
            hipMemcpy(g0_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        // Generate using g1 and copy to host
        ROCRAND_CHECK(g1.generate(data, size));
        HIP_CHECK(hipDeviceSynchronize());

        unsigned int g1_host_data[size];
        HIP_CHECK(
            hipMemcpy(g1_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        size_t same = 0;
        for(size_t i = 0; i < size; i++)
        {
            if(g1_host_data[i] == g0_host_data[i])
                same++;
        }
        // It may happen that numbers are the same, so we
        // just make sure that most of them are different.
        EXPECT_LT(same, static_cast<size_t>(0.01f * size));

        HIP_CHECK(hipFree(data));
    }
}

// Engine API tests
struct lfsr113_engine_api_tests : public lfsr113_generator::engine_type
{};

TEST(lfsr113_engine_api_tests, discard_test)
{
    using generator_t = lfsr113_generator;
    using engine_t    = typename generator_t::engine_type;

    const uint4 seed = {1234567U, 12345678U, 123456789U, 1234567890U};
    engine_t    engine1(seed, 0, 678U);
    engine_t    engine2(seed, 0, 677U);

    // Check next() function
    (void)engine2.next();

    EXPECT_EQ(engine1(), engine2());

    // Check discard() function
    (void)engine1.next();
    engine2.discard();

    EXPECT_EQ(engine1(), engine2());

    // Check discard(offset) function
    const unsigned int offsets[]
        = {1U, 4U, 37U, 583U, 7452U, 21032U, 35678U, 66778U, 10313475U, 82120230U};

    for(auto offset : offsets)
    {
        for(unsigned int i = 0; i < offset; i++)
        {
            (void)engine1.next();
        }
        engine2.discard(offset);

        EXPECT_EQ(engine1(), engine2());
    }
}

TEST(lfsr113_engine_api_tests, discard_sequence_test)
{
    using generator_t = lfsr113_generator;
    using engine_t    = typename generator_t::engine_type;

    const uint4 seed = {1234567U, 12345678U, 123456789U, 1234567890U};
    engine_t    engine1(seed, 0, 444U);
    engine_t    engine2(seed, 123U, 444U);

    engine1.discard_subsequence(123U);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard(5356446450U);
    engine1.discard_subsequence(123U);
    engine1.discard(30000000006U);

    engine2.discard_subsequence(3U);
    engine2.discard(35356446456U);
    engine2.discard_subsequence(120U);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard_subsequence(3456000U);
    engine1.discard_subsequence(1000005U);

    engine2.discard_subsequence(4456005U);

    EXPECT_EQ(engine1(), engine2());
}
