// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_TEST_INTERNAL_TEST_ROCRAND_THREEFRY_PRNG_HPP_
#define ROCRAND_TEST_INTERNAL_TEST_ROCRAND_THREEFRY_PRNG_HPP_

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include <rocrand/rocrand.h>

#include <rng/threefry.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>

// threefryNx64_20-specific generator API tests
template<class Params>
struct threefryNx64_20_generator_prng_tests : public ::testing::Test
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

TYPED_TEST_SUITE_P(threefryNx64_20_generator_prng_tests);

// Assert that the kernel arguments are trivially copyable and destructible.
TYPED_TEST_P(threefryNx64_20_generator_prng_tests, type)
{
    using generator_t = typename TestFixture::generator_t;
    using engine_type = typename generator_t::engine_type::base_type;
    // TODO: Enable once uint2 is trivially copyable.
    // EXPECT_TRUE(std::is_trivially_copyable<engine_type>::value);
    EXPECT_TRUE(std::is_trivially_destructible<engine_type>::value);
}

TYPED_TEST_P(threefryNx64_20_generator_prng_tests, uniform_ulonglong_test)
{
    const size_t        size = 1313;
    unsigned long long* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned long long) * (size + 1)));

    auto g = TestFixture::get_generator();
    ROCRAND_CHECK(g.generate_uniform(data + 1, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long host_data[size];
    HIP_CHECK(
        hipMemcpy(host_data, data + 1, sizeof(unsigned long long) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i] / static_cast<double>(size);
    }
    ASSERT_NEAR(mean, static_cast<double>(ULLONG_MAX / 2), static_cast<double>(ULLONG_MAX / 20));

    HIP_CHECK(hipFree(data));
}

REGISTER_TYPED_TEST_SUITE_P(threefryNx64_20_generator_prng_tests, type, uniform_ulonglong_test);

// threefryNx64_20-specific generator API continuity tests
template<class Params>
struct threefryNx64_20_generator_prng_continuity_tests : public ::testing::Test
{
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;
};

TYPED_TEST_SUITE_P(threefryNx64_20_generator_prng_continuity_tests);

TYPED_TEST_P(threefryNx64_20_generator_prng_continuity_tests, continuity_uniform_ullong_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef unsigned long long int output_t;

    continuity_test<output_t, generator_t>(
        [](generator_t& g, output_t* data, size_t s) { g.generate_uniform(data, s); },
        ordering,
        rocrand_impl::host::uniform_distribution<output_t, unsigned long long int>::output_width);
}

REGISTER_TYPED_TEST_SUITE_P(threefryNx64_20_generator_prng_continuity_tests,
                            continuity_uniform_ullong_test);

#endif // ROCRAND_TEST_INTERNAL_TEST_ROCRAND_THREEFRY_PRNG_HPP_
