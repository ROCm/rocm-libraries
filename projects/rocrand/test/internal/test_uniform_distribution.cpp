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

#include <gtest/gtest.h>
#include <stdio.h>

#include <random>

#include <rng/distribution/uniform.hpp>

using namespace rocrand_impl::host;

TEST(uniform_distribution_tests, uint_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    uniform_distribution<unsigned int> u;
    unsigned int                       input[1];
    unsigned int                       output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        unsigned int x = dis(gen);
        input[0]       = x;
        u(input, output);
        EXPECT_EQ(output[0], x);
    }

    input[0] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(output[0], UINT_MAX);
    input[0] = 0U;
    u(input, output);
    EXPECT_EQ(output[0], 0U);
}

TEST(uniform_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    uniform_distribution<float> u;
    unsigned int                input[1];
    float                       output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0f);
        EXPECT_GT(output[0], 0.0f);
    }

    input[0] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(output[0], 1.0f);
    input[0] = 0U;
    u(input, output);
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1e-9f);
}

TEST(uniform_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    uniform_distribution<double> u;
    unsigned int                 input[2];
    double                       output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0);
        EXPECT_GT(output[0], 0.0);
    }

    input[0] = UINT_MAX;
    input[1] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(output[0], 1.0);
    input[0] = 0U;
    input[1] = 0U;
    u(input, output);
    EXPECT_GT(output[0], 0.0);
    EXPECT_LT(output[0], 1e-9);
}

TEST(uniform_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    uniform_distribution<half> u;
    unsigned int               input[1];
    half                       output[2];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(__half2float(output[0]), 1.0f);
        EXPECT_LE(__half2float(output[1]), 1.0f);
        EXPECT_GT(__half2float(output[0]), 0.0f);
        EXPECT_GT(__half2float(output[1]), 0.0f);
    }

    input[0] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(__half2float(output[0]), 1.0f);
    EXPECT_EQ(__half2float(output[1]), 1.0f);
    input[0] = 0U;
    u(input, output);
    EXPECT_GT(__half2float(output[0]), 0.0f);
    EXPECT_LT(__half2float(output[0]), 1e-4f);
    EXPECT_GT(__half2float(output[1]), 0.0f);
    EXPECT_LT(__half2float(output[1]), 1e-4f);
}

template<typename mrg, unsigned int m1>
struct mrg_uniform_distribution_test_type
{
    typedef mrg                   mrg_type;
    static constexpr unsigned int mrg_m1 = m1;
};

template<typename test_type>
struct mrg_uniform_distribution_tests : public ::testing::Test
{
    typedef typename test_type::mrg_type mrg_type;
    static constexpr unsigned int        mrg_m1 = test_type::mrg_m1;
};

typedef ::testing::Types<
    mrg_uniform_distribution_test_type<rocrand_state_mrg31k3p, ROCRAND_MRG31K3P_M1>,
    mrg_uniform_distribution_test_type<rocrand_state_mrg32k3a, ROCRAND_MRG32K3A_M1>>
    mrg_uniform_distribution_test_types;

TYPED_TEST_SUITE(mrg_uniform_distribution_tests, mrg_uniform_distribution_test_types);

TYPED_TEST(mrg_uniform_distribution_tests, uint_test)
{
    mrg_engine_uniform_distribution<unsigned int, typename TestFixture::mrg_type> u;
    unsigned int                                                                  input[1];
    unsigned int                                                                  output[1];

    input[0] = TestFixture::mrg_m1;
    u(input, output);
    EXPECT_EQ(output[0], UINT_MAX);
    input[0] = 1U;
    u(input, output);
    EXPECT_EQ(output[0], 0U);
}

TYPED_TEST(mrg_uniform_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    mrg_engine_uniform_distribution<float, typename TestFixture::mrg_type> u;
    unsigned int                                                           input[1];
    float                                                                  output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0f);
        EXPECT_GT(output[0], 0.0f);
    }

    input[0] = TestFixture::mrg_m1;
    u(input, output);
    EXPECT_EQ(output[0], 1.0f);
    input[0] = 1U;
    u(input, output);
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1e-9f);
}

TYPED_TEST(mrg_uniform_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    mrg_engine_uniform_distribution<double, typename TestFixture::mrg_type> u;
    unsigned int                                                            input[1];
    double                                                                  output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0);
        EXPECT_GT(output[0], 0.0);
    }

    input[0] = TestFixture::mrg_m1;
    u(input, output);
    EXPECT_EQ(output[0], 1.0);
    input[0] = 1U;
    u(input, output);
    EXPECT_GT(output[0], 0.0);
    EXPECT_LT(output[0], 1e-9);
}

TYPED_TEST(mrg_uniform_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    mrg_engine_uniform_distribution<half, typename TestFixture::mrg_type> u;
    unsigned int                                                          input[1];
    half                                                                  output[2];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(__half2float(output[0]), 1.0f);
        EXPECT_LE(__half2float(output[1]), 1.0f);
        EXPECT_GT(__half2float(output[0]), 0.0f);
        EXPECT_GT(__half2float(output[1]), 0.0f);
    }

    input[0] = TestFixture::mrg_m1;
    u(input, output);
    EXPECT_EQ(__half2float(output[0]), 1.0f);
    EXPECT_EQ(__half2float(output[1]), 1.0f);
    input[0] = 1U;
    u(input, output);
    EXPECT_GT(__half2float(output[0]), 0.0f);
    EXPECT_LT(__half2float(output[0]), 1e-4f);
    EXPECT_GT(__half2float(output[1]), 0.0f);
    EXPECT_LT(__half2float(output[1]), 1e-4f);
}

TEST(sobol_uniform_distribution_tests, uint_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    sobol_uniform_distribution<unsigned int> u;
    unsigned int                             input[1];
    unsigned int                             output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        unsigned int x = dis(gen);
        input[0]       = x;
        output[0]      = u(input[0]);
        EXPECT_EQ(output[0], x);
    }

    input[0]  = UINT_MAX;
    output[0] = u(input[0]);
    EXPECT_EQ(output[0], UINT_MAX);
    input[0]  = 0U;
    output[0] = u(input[0]);
    EXPECT_EQ(output[0], 0U);
}

TEST(sobol_uniform_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    sobol_uniform_distribution<float> u;
    unsigned int                      input[1];
    float                             output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0]  = dis(gen);
        output[0] = u(input[0]);
        EXPECT_LE(output[0], 1.0f);
        EXPECT_GT(output[0], 0.0f);
    }

    input[0]  = UINT_MAX;
    output[0] = u(input[0]);
    EXPECT_EQ(output[0], 1.0f);
    input[0]  = 0U;
    output[0] = u(input[0]);
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1e-9f);
}

TEST(sobol_uniform_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    sobol_uniform_distribution<double> u;
    unsigned int                       input[1];
    double                             output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0]  = dis(gen);
        output[0] = u(input[0]);
        EXPECT_LE(output[0], 1.0);
        EXPECT_GT(output[0], 0.0);
    }

    input[0]  = UINT_MAX;
    output[0] = u(input[0]);
    EXPECT_EQ(output[0], 1.0);
    input[0]  = 0U;
    output[0] = u(input[0]);
    EXPECT_GT(output[0], 0.0);
    EXPECT_LT(output[0], 1e-9);
}

TEST(sobol_uniform_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    sobol_uniform_distribution<half> u;
    unsigned int                     input[1];
    half                             output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0]  = dis(gen);
        output[0] = u(input[0]);
        EXPECT_LE(__half2float(output[0]), 1.0f);
        EXPECT_GT(__half2float(output[0]), 0.0f);
    }

    input[0]  = UINT_MAX;
    output[0] = u(input[0]);
    EXPECT_EQ(__half2float(output[0]), 1.0f);
    input[0]  = 0U;
    output[0] = u(input[0]);
    EXPECT_GT(__half2float(output[0]), 0.0f);
    EXPECT_LT(__half2float(output[0]), 1e-4f);
}
