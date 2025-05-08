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

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>

// threefryNx32_20-specific generator API tests
template<class Params>
struct threefryNx32_20_generator_prng_tests : public ::testing::Test
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

TYPED_TEST_SUITE_P(threefryNx32_20_generator_prng_tests);

// Assert that the kernel arguments are trivially copyable and destructible.
TYPED_TEST_P(threefryNx32_20_generator_prng_tests, type)
{
    using generator_t = typename TestFixture::generator_t;
    using engine_type = typename generator_t::engine_type::base_type;
    // TODO: Enable once uint2 is trivially copyable.
    // EXPECT_TRUE(std::is_trivially_copyable<engine_type>::value);
    EXPECT_TRUE(std::is_trivially_destructible<engine_type>::value);
}

REGISTER_TYPED_TEST_SUITE_P(threefryNx32_20_generator_prng_tests, type);

#endif // ROCRAND_TEST_INTERNAL_TEST_ROCRAND_THREEFRY_PRNG_HPP_
