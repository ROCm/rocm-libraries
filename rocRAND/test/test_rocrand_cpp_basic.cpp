// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>

#include <rocrand/rocrand.hpp>

#include <hip/hip_runtime_api.h>

template<typename GeneratorType>
class rocrand_cpp_basic_tests : public ::testing::Test
{
public:
    using generator_type = GeneratorType;
};

using GeneratorTypes = testing::Types<rocrand_cpp::lfsr113,
                                      rocrand_cpp::mrg31k3p,
                                      rocrand_cpp::mrg32k3a,
                                      rocrand_cpp::mt19937,
                                      rocrand_cpp::mtgp32,
                                      rocrand_cpp::philox4x32_10,
                                      rocrand_cpp::scrambled_sobol32,
                                      rocrand_cpp::scrambled_sobol64,
                                      rocrand_cpp::sobol32,
                                      rocrand_cpp::sobol64,
                                      rocrand_cpp::threefry2x32,
                                      rocrand_cpp::threefry2x64,
                                      rocrand_cpp::threefry4x32,
                                      rocrand_cpp::threefry4x64,
                                      rocrand_cpp::xorwow>;

TYPED_TEST_SUITE(rocrand_cpp_basic_tests, GeneratorTypes);

TYPED_TEST(rocrand_cpp_basic_tests, move_construction)
{
    using generator_type = typename TestFixture::generator_type;

    rocrand_cpp::uniform_real_distribution<float> dist;

    float* d_data;
    HIP_CHECK(hipMallocHelper(&d_data, sizeof(*d_data)));

    float expected;
    {
        // Generate two values to verify that moving from a generator transfers the state
        generator_type g;
        dist(g, d_data, 1);
        dist(g, d_data, 1);

        HIP_CHECK(hipMemcpy(&expected, d_data, sizeof(expected), hipMemcpyDeviceToHost));
    }

    // Create g2 in a new scope so that it gets destroyed before g1.
    // Have to have some kind of indirection here for the deferred lifetimes, unique_ptr is chosen
    // because std::optional would be C++17
    std::unique_ptr<generator_type> g1;
    {
        generator_type g2;
        dist(g2, d_data, 1);
        g1.reset(new generator_type(std::move(g2)));
    }

    dist(*g1, d_data, 1);

    float actual;
    HIP_CHECK(hipMemcpy(&actual, d_data, sizeof(actual), hipMemcpyDeviceToHost));
    ASSERT_EQ(expected, actual);
    
    HIP_CHECK(hipFree(d_data));
}

TYPED_TEST(rocrand_cpp_basic_tests, move_assignment)
{
    using generator_type = typename TestFixture::generator_type;

    rocrand_cpp::uniform_real_distribution<float> dist;

    float* d_data;
    HIP_CHECK(hipMallocHelper(&d_data, sizeof(*d_data)));

    float expected;
    {
        // Generate two values to verify that moving from a generator transfers the state
        generator_type g;
        dist(g, d_data, 1);
        dist(g, d_data, 1);

        HIP_CHECK(hipMemcpy(&expected, d_data, sizeof(expected), hipMemcpyDeviceToHost));
    }

    // Create g2 in a new scope so that it gets destroyed before g1.
    generator_type g1;
    {
        generator_type g2;
        dist(g2, d_data, 1);
        g1 = std::move(g2);
    }

    dist(g1, d_data, 1);

    float actual;
    HIP_CHECK(hipMemcpy(&actual, d_data, sizeof(actual), hipMemcpyDeviceToHost));
    ASSERT_EQ(expected, actual);
    
    HIP_CHECK(hipFree(d_data));
}
