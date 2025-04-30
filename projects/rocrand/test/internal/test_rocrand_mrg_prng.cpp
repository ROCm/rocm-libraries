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

#include <rng/mrg.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <stdexcept>

using rocrand_impl::host::mrg31k3p_generator;
using rocrand_impl::host::mrg32k3a_generator;

// Generator API tests
using mrg_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<mrg31k3p_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<mrg31k3p_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_tests_params<mrg32k3a_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<mrg32k3a_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using mrg_generator_prng_offset_tests_types = ::testing::Types<
    generator_prng_offset_tests_params<unsigned int,
                                       mrg31k3p_generator,
                                       ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<unsigned int,
                                       mrg31k3p_generator,
                                       ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_offset_tests_params<unsigned int,
                                       mrg32k3a_generator,
                                       ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<unsigned int,
                                       mrg32k3a_generator,
                                       ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_offset_tests_params<float, mrg31k3p_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<float, mrg31k3p_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_offset_tests_params<float, mrg32k3a_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<float, mrg32k3a_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mrg, generator_prng_tests, mrg_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mrg,
                               generator_prng_continuity_tests,
                               mrg_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mrg,
                               generator_prng_offset_tests,
                               mrg_generator_prng_offset_tests_types);

// mrg-specific generator API tests
template<class Params>
struct mrg_generator_prng_tests : public ::testing::Test
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

TYPED_TEST_SUITE(mrg_generator_prng_tests, mrg_generator_prng_tests_types);

template<class Generator, class T>
void uniform_floating_point_range_test(rocrand_ordering ordering)
{
    const size_t size = 1 << 26;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    Generator g;
    g.set_order(ordering);
    ROCRAND_CHECK(g.generate_uniform(data, size));

    T* host_data = new T[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_GT(host_data[i], 0.0f);
        ASSERT_LE(host_data[i], 1.0f);
    }

    HIP_CHECK(hipFree(data));
    delete[] host_data;
}

TYPED_TEST(mrg_generator_prng_tests, uniform_float_range_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;

    uniform_floating_point_range_test<generator_t, float>(ordering);
}

TYPED_TEST(mrg_generator_prng_tests, uniform_double_range_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;

    uniform_floating_point_range_test<generator_t, double>(ordering);
}

// Engine API tests
template<class Generator>
struct mrg_prng_engine_tests : public ::testing::Test
{
    using mrg_type = Generator;
};

using mrg_prng_engine_tests_types = ::testing::Types<mrg31k3p_generator, mrg32k3a_generator>;

TYPED_TEST_SUITE(mrg_prng_engine_tests, mrg_prng_engine_tests_types);

TYPED_TEST(mrg_prng_engine_tests, discard_test)
{
    typedef typename TestFixture::mrg_type mrg_type;
    const unsigned long long               seed = 12345ULL;
    typename mrg_type::engine_type         engine1(seed, 0, 678ULL);
    typename mrg_type::engine_type         engine2(seed, 0, 677ULL);

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

TYPED_TEST(mrg_prng_engine_tests, discard_sequence_test)
{
    typedef typename TestFixture::mrg_type mrg_type;
    const unsigned long long               seed = 23456ULL;
    typename mrg_type::engine_type         engine1(seed, 123ULL, 444ULL);
    typename mrg_type::engine_type         engine2(seed, 123ULL, 444ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard(5356446450ULL);
    engine1.discard_sequence(123ULL);
    engine1.discard(30000000006ULL);

    engine2.discard_sequence(3ULL);
    engine2.discard(35356446456ULL);
    engine2.discard_sequence(120ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard_sequence(3456000ULL);
    engine1.discard_sequence(1000005ULL);

    engine2.discard_sequence(4456005ULL);

    EXPECT_EQ(engine1(), engine2());
}

TYPED_TEST(mrg_prng_engine_tests, discard_subsequence_test)
{
    typedef typename TestFixture::mrg_type mrg_type;
    const unsigned long long               seed = 23456ULL;
    typename mrg_type::engine_type         engine1(seed, 0, 444ULL);
    typename mrg_type::engine_type         engine2(seed, 123ULL, 444ULL);

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
