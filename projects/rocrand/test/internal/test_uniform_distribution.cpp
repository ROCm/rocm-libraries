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

#include <gtest/gtest.h>
#include <stdio.h>

#include "../test_common.hpp"

#include <cmath>
#include <numeric>
#include <random>

#include <rng/distribution/uniform.hpp>
#include <rocrand/rocrand_mtgp32_11213.h>

#define ROCRAND_CHECK(cmd)                                                                \
    do                                                                                    \
    {                                                                                     \
        auto status = cmd;                                                                \
        if(status != 0)                                                                   \
        {                                                                                 \
            std::cerr << "Encountered ROCRAND error: " << status << "at line" << __LINE__ \
                      << " in file " << __FILE__ << "\n";                                 \
            exit(status);                                                                 \
        }                                                                                 \
    }                                                                                     \
    while(0)

// If x is small then get withing 0.001 otherwise 5%
#define GET_EPS(x) x < (T)0.01 ? (T)0.001 : x*(T)0.05

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

template <typename OutType, typename InType, typename UD>
struct NumericUD{

    template <typename FuncCall>
    void run_test(UD & dis, const FuncCall & f){
        std::random_device rd;
        std::mt19937 gen(rd());

        const size_t testSize = 4000000;

        float * output = new float [testSize];

        InType input;
        OutType out;

        double mean = 0;

        for(size_t i = 0; i < testSize; i += 4){
            input = {dis(gen), dis(gen), dis(gen), dis(gen)};

            f(input, out);

            output[i] = out.w;
            output[i + 1] = out.x;
            output[i + 2] = out.y;
            output[i + 3] = out.z;

            ASSERT_GT(out.w, 0);
            ASSERT_GT(out.x, 0);
            ASSERT_GT(out.y, 0);
            ASSERT_GT(out.z, 0);

            ASSERT_LE(out.w, 1);
            ASSERT_LE(out.x, 1);
            ASSERT_LE(out.y, 1);
            ASSERT_LE(out.z, 1);

            mean += out.w + out.x + out.y + out.z;
        }

        mean /= testSize;

        double std = 0.0;
        for(size_t i = 0; i < testSize; i++)
            std += std::pow(output[i] - mean, 2);

        std = std::sqrt(std / testSize);

        double eMean = 0.5 * (0 + 1); // 0.5(a + b)
        double eStd = (1 - 0) / (2 * std::sqrt(3)); // (b - a) / (2*3^0.5)

        ASSERT_NEAR(mean, eMean, eMean * 0.1) << "Expected Mean: " << eMean << " Actual Mean: " << mean << " Eps: " << eMean * 0.1;
        ASSERT_NEAR(std, eStd, eStd * 0.1) << "Expected Std: " << eStd << " Actual Std: " << std << " Eps: " << eStd * 0.1;

        delete [] output;
    }

};

TEST(uniform_distribution_tests, float4_uint4_in_test){
    unsigned int mini = 0;
    unsigned int maxi = std::numeric_limits<unsigned int>::max();
    std::uniform_int_distribution<unsigned int> dis(mini, maxi);

    NumericUD<float4, uint4, std::uniform_int_distribution<unsigned int>> test;
    test.run_test(dis,
                  [](uint4& input, float4& output)
                  { output = rocrand_device::detail::uniform_distribution4(input); });
}

TEST(uniform_distribution_tests, float4_ulonglong4_test){
    unsigned long long mini = 0;
    unsigned long long maxi = std::numeric_limits<unsigned long long>::max();
    std::uniform_int_distribution<unsigned long long> dis(mini, maxi);

    NumericUD<float4, ulonglong4, std::uniform_int_distribution<unsigned long long>> test;
    test.run_test(dis,
                  [](ulonglong4& input, float4& output)
                  { output = rocrand_device::detail::uniform_distribution4(input); });
}

TEST(uniform_distribution_tests, double4_uint4_test){
    unsigned int mini = 0;
    unsigned int maxi = std::numeric_limits<unsigned int>::max();
    std::uniform_int_distribution<unsigned int> dis(mini, maxi);

    std::random_device rd;
    std::mt19937 gen(rd());

    NumericUD<double4, uint4, std::uniform_int_distribution<unsigned int>> test;
    test.run_test(dis,
                  [&](uint4& input, double4& output)
                  { output = rocrand_device::detail::uniform_distribution_double4(input, input); });
}

TEST(uniform_distribution_tests, double4_ulonglong4_test){
    unsigned long long mini = 0;
    unsigned long long maxi = std::numeric_limits<unsigned long long>::max();
    std::uniform_int_distribution<unsigned long long> dis(mini, maxi);

    NumericUD<double4, ulonglong4, std::uniform_int_distribution<unsigned long long>> test;
    test.run_test(dis,
                  [](ulonglong4& input, double4& output)
                  { output = rocrand_device::detail::uniform_distribution_double4(input); });
}

TEST(uniform_distribution_tests, double2_uint4_in_test){
    unsigned int mini = 0;
    unsigned int maxi = std::numeric_limits<unsigned int>::max();
    std::uniform_int_distribution<unsigned int> dis(mini, maxi);

    std::random_device rd;
    std::mt19937 gen(rd());

    NumericUD<double4, uint4, std::uniform_int_distribution<unsigned int>> test;
    test.run_test(dis,
                  [&](uint4& input, double4& output)
                  {
                      uint4   secondInput = {dis(gen), dis(gen), dis(gen), dis(gen)};
                      double2 o1 = rocrand_device::detail::uniform_distribution_double2(input);
                      double2 o2
                          = rocrand_device::detail::uniform_distribution_double2(secondInput);

                      output.w = o1.x;
                      output.x = o1.y;
                      output.y = o2.x;
                      output.z = o2.y;
                  });
}

TEST(uniform_distribution_tests, double2_ulonglong2_in_test){
    unsigned long long mini = 0;
    unsigned long long maxi = std::numeric_limits<unsigned long long>::max();
    std::uniform_int_distribution<unsigned long long> dis(mini, maxi);

    NumericUD<double4, ulonglong4, std::uniform_int_distribution<unsigned long long>> test;
    test.run_test(dis,
                  [&](ulonglong4& input, double4& output)
                  {
                      ulonglong2 i1 = {input.w, input.x};
                      ulonglong2 i2 = {input.y, input.z};

                      double2 o1 = rocrand_device::detail::uniform_distribution_double2(i1);
                      double2 o2 = rocrand_device::detail::uniform_distribution_double2(i2);

                      output.w = o1.x;
                      output.x = o1.y;
                      output.y = o2.x;
                      output.z = o2.y;
                  });
}

TEST(uniform_distribution_tests, double2_ulonglong4_in_test){
    unsigned long long mini = 0;
    unsigned long long maxi = std::numeric_limits<unsigned long long>::max();
    std::uniform_int_distribution<unsigned long long> dis(mini, maxi);

    std::random_device rd;
    std::mt19937 gen(rd());

    NumericUD<double4, ulonglong4, std::uniform_int_distribution<unsigned long long>> test;
    test.run_test(dis,
                  [&](ulonglong4& input, double4& output)
                  {
                      ulonglong4 secondInput = {dis(gen), dis(gen), dis(gen), dis(gen)};
                      double2    o1 = rocrand_device::detail::uniform_distribution_double2(input);
                      double2    o2
                          = rocrand_device::detail::uniform_distribution_double2(secondInput);

                      output.w = o1.x;
                      output.x = o1.y;
                      output.y = o2.x;
                      output.z = o2.y;
                  });
}

template <typename OutType>
struct StatesUD{
    template <typename FuncCall>
    void run_test(const FuncCall & f, size_t testSize = 4000000){
        float * output = new float [testSize];
        OutType out;

        double mean = 0;

        for(size_t i = 0; i < testSize; i += 4){
            f(out);

            output[i] = out.w;
            output[i + 1] = out.x;
            output[i + 2] = out.y;
            output[i + 3] = out.z;

            ASSERT_GT(out.w, 0);
            ASSERT_GT(out.x, 0);
            ASSERT_GT(out.y, 0);
            ASSERT_GT(out.z, 0);

            ASSERT_LE(out.w, 1);
            ASSERT_LE(out.x, 1);
            ASSERT_LE(out.y, 1);
            ASSERT_LE(out.z, 1);

            mean += out.w + out.x + out.y + out.z;
        }

        mean /= testSize;

        double std = 0.0;
        for(size_t i = 0; i < testSize; i++)
            std += std::pow(output[i] - mean, 2);

        std = std::sqrt(std / testSize);

        double eMean = 0.5 * (0 + 1); // 0.5(a + b)
        double eStd = (1 - 0) / (2 * std::sqrt(3)); // (b - a) / (2*3^0.5)

        ASSERT_NEAR(mean, eMean, eMean * 0.1) << "Expected Mean: " << eMean << " Actual Mean: " << mean << " Eps: " << eMean * 0.1;
        ASSERT_NEAR(std, eStd, eStd * 0.1) << "Expected Std: " << eStd << " Actual Std: " << std << " Eps: " << eStd * 0.1;

        delete [] output;
    }
};

TEST(uniform_distribution_tests, philox4x32_10_test){
    rocrand_state_philox4x32_10 states;
    rocrand_init(123456, 654321, 0, &states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    testFloat.run_test(
        [&](float4& output)
        {
            float2 o1 = rocrand_uniform2(&states);
            float2 o2 = rocrand_uniform2(&states);

            output.w = o1.x;
            output.x = o1.y;
            output.y = o2.x;
            output.z = o2.y;
        });

    testFloat.run_test([&](float4& output) { output = rocrand_uniform4(&states); });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });

    testDouble.run_test(
        [&](double4& output)
        {
            double2 o1 = rocrand_uniform_double2(&states);
            double2 o2 = rocrand_uniform_double2(&states);

            output.w = o1.x;
            output.x = o1.y;
            output.y = o2.x;
            output.z = o2.y;
        });

    testDouble.run_test([&](double4& output) { output = rocrand_uniform_double4(&states); });
}

TEST(uniform_distribution_tests, mrg31k3p_test){
    rocrand_state_mrg31k3p states;
    rocrand_init(123456, 654321, 0, &states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, mrg32k3a_test){
    rocrand_state_mrg32k3a states;
    rocrand_init(123456, 654321, 0, &states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, xorwow_test){
    rocrand_state_xorwow states;
    rocrand_init(123456, 654321, 0, &states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, sobol32_test){
    rocrand_state_sobol32 states;
    const unsigned int* directions;
    ROCRAND_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    rocrand_init(directions, 0, &states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, scrambled_sobol32_test){
    rocrand_state_scrambled_sobol32 states;
    const unsigned int* directions;
    ROCRAND_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    rocrand_init(directions, 123456, 0, &states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, sobol64_test){
    rocrand_state_sobol64 states;
    const unsigned long long* directions;
    ROCRAND_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
    rocrand_init(directions, 0, &states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, scrambled_sobol64_test){
    rocrand_state_scrambled_sobol64 states;
    const unsigned long long* directions;
    ROCRAND_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
    rocrand_init(directions, 123456, 0, &states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, lfsr113_test){
    rocrand_state_lfsr113 states;
    rocrand_init(uint4{12}, 0, &states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, threefry2x32_20_test){
    rocrand_state_threefry2x32_20 states;
    rocrand_init(123456, 654321, 0, & states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, threefry2x64_20_test){
    rocrand_state_threefry2x64_20 states;
    rocrand_init(123456, 654321, 0, & states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, threefry4x32_20_test){
    rocrand_state_threefry4x32_20 states;
    rocrand_init(123456, 654321, 0, & states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

TEST(uniform_distribution_tests, threefry4x64_20_test){
    rocrand_state_threefry4x64_20 states;
    rocrand_init(123456, 654321, 0, & states);

    StatesUD<float4> testFloat;

    testFloat.run_test(
        [&](float4& output)
        {
            output = {
                rocrand_uniform(&states), rocrand_uniform(&states),
                rocrand_uniform(&states), rocrand_uniform(&states)
            };
        });

    StatesUD<double4> testDouble;

    testDouble.run_test(
        [&](double4& output)
        {
            output = {
                rocrand_uniform_double(&states), rocrand_uniform_double(&states),
                rocrand_uniform_double(&states), rocrand_uniform_double(&states)
            };
        });
}

template <typename T, typename UDFunction>
__global__ void mtgp32_kernel (rocrand_state_mtgp32 * states, T * output, const size_t N, const UDFunction & f){
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= N)
        return;

    __shared__ rocrand_state_mtgp32 state;
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    // output[index] = rocrand_uniform(&state);
    output[index] = f(&state);

    if(thread_id == 0)
        states[state_id] = state;
}

struct rocrand_state_mtgp32_uniform
{
    __device__
    auto operator()(rocrand_state_mtgp32* state) const
    {
        return rocrand_uniform(state);
    }
};

struct rocrand_state_mtgp32_uniform_double
{
    __device__
    auto operator()(rocrand_state_mtgp32* state) const
    {
        return rocrand_uniform_double(state);
    }
};

TEST(uniform_distribution_tests, float_mtgp32_test){
    size_t testSize = 40192;
    size_t threads = 256;
    size_t blocks = std::ceil(static_cast<double>(testSize) / static_cast<double>(threads));

    rocrand_state_mtgp32 * states;
    size_t state_size = blocks, seed = 654321;
    HIP_CHECK(hipMalloc(&states, state_size * sizeof(rocrand_state_mtgp32)));
    rocrand_make_state_mtgp32(states,  mtgp32dc_params_fast_11213, state_size, seed);

    float * hOut = new float[testSize];
    float * dOut;
    HIP_CHECK(hipMalloc(&dOut, sizeof(float) * testSize));
    HIP_CHECK(hipDeviceSynchronize());

    mtgp32_kernel<float><<<dim3(blocks), dim3(threads), 0, 0>>>(states,
                                                                dOut,
                                                                testSize,
                                                                rocrand_state_mtgp32_uniform{});

    HIP_CHECK(hipMemcpy(hOut, dOut, sizeof(float) * testSize, hipMemcpyDeviceToHost));

    double mean = 0.0;
    for(size_t i = 0; i < testSize; i++){
        ASSERT_GT(hOut[i], 0.0);
        ASSERT_LE(hOut[i], 1.0);

        mean += hOut[i];
    }

    mean /= testSize;

    double std = 0.0;
    for(size_t i = 0; i < testSize; i++)
        std += std::pow(hOut[i] - mean, 2);

    std = std::sqrt(std / testSize);

    double eMean = 0.5 * (0 + 1); // 0.5(a + b)
    double eStd = (1 - 0) / (2 * std::sqrt(3)); // (b - a) / (2*3^0.5)

    ASSERT_NEAR(mean, eMean, eMean * 0.1) << "Expected Mean: " << eMean << " Actual Mean: " << mean << " Eps: " << eMean * 0.1;
    ASSERT_NEAR(std, eStd, eStd * 0.1) << "Expected Std: " << eStd << " Actual Std: " << std << " Eps: " << eStd * 0.1;

    HIP_CHECK(hipFree(states));
    HIP_CHECK(hipFree(dOut));

    delete [] hOut;
}

TEST(uniform_distribution_tests, double_mtgp32_test){
    size_t testSize = 40192;
    size_t threads = 256;
    size_t blocks = std::ceil(static_cast<double>(testSize) / static_cast<double>(threads));

    rocrand_state_mtgp32 * states;
    size_t state_size = blocks, seed = 654321;
    HIP_CHECK(hipMalloc(&states, state_size * sizeof(rocrand_state_mtgp32)));
    rocrand_make_state_mtgp32(states,  mtgp32dc_params_fast_11213, state_size, seed);

    double * hOut = new double[testSize];
    double * dOut;
    HIP_CHECK(hipMalloc(&dOut, sizeof(double) * testSize));
    HIP_CHECK(hipDeviceSynchronize());

    mtgp32_kernel<double>
        <<<dim3(blocks), dim3(threads), 0, 0>>>(states,
                                                dOut,
                                                testSize,
                                                rocrand_state_mtgp32_uniform_double{});

    HIP_CHECK(hipMemcpy(hOut, dOut, sizeof(double) * testSize, hipMemcpyDeviceToHost));

    double mean = 0.0;
    for(size_t i = 0; i < testSize; i++){
        ASSERT_GT(hOut[i], 0.0);
        ASSERT_LE(hOut[i], 1.0);

        mean += hOut[i];
    }

    mean /= testSize;

    double std = 0.0;
    for(size_t i = 0; i < testSize; i++)
        std += std::pow(hOut[i] - mean, 2);

    std = std::sqrt(std / testSize);

    double eMean = 0.5 * (0 + 1); // 0.5(a + b)
    double eStd = (1 - 0) / (2 * std::sqrt(3)); // (b - a) / (2*3^0.5)

    ASSERT_NEAR(mean, eMean, eMean * 0.1) << "Expected Mean: " << eMean << " Actual Mean: " << mean << " Eps: " << eMean * 0.1;
    ASSERT_NEAR(std, eStd, eStd * 0.1) << "Expected Std: " << eStd << " Actual Std: " << std << " Eps: " << eStd * 0.1;

    HIP_CHECK(hipFree(states));
    HIP_CHECK(hipFree(dOut));

    delete [] hOut;
}

/* #################################################

                TEST HOST SIDE

   ###############################################*/

template<typename OutType,
         typename T,
         size_t OutSize,
         class GenFunc,
         class ReadMeanFunc,
         class ReadStdFunc>
void run_host_num_test(const GenFunc& gf, const ReadMeanFunc& rmf, const ReadStdFunc& rsf)
{
    constexpr size_t test_size = 50000;

    const T expected_mean    = 0.5;
    const T expected_std_dev = 0.288675134595; //sqrt(1/12)

    std::vector<OutType> output(test_size);

    std::random_device                                rd;
    std::mt19937                                      gen(rd());
    std::uniform_int_distribution<unsigned long long> dis(
        std::numeric_limits<unsigned long long>::min(),
        std::numeric_limits<unsigned long long>::max());

    for(size_t i = 0; i < test_size; i++)
    {
        output[i] = gf(dis, gen);
    }

    T actual_mean = std::accumulate(output.begin(),
                                    output.end(),
                                    (T)0,
                                    [=](T acc, OutType x) { return acc + rmf(x); })
                    / static_cast<T>(test_size * OutSize);

    T actual_std_dev = std::accumulate(output.begin(),
                                       output.end(),
                                       (T)0,
                                       [=](T acc, OutType x) { return acc + rsf(x, actual_mean); });
    actual_std_dev   = std::sqrt(actual_std_dev / static_cast<T>(test_size * OutSize - 1));

    T mean_eps    = GET_EPS(expected_mean);
    T std_dev_eps = GET_EPS(expected_std_dev);

    ASSERT_NEAR(expected_mean, actual_mean, mean_eps);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, std_dev_eps);
}

TEST(UniformHostTest, float_out_uint_in)
{
    run_host_num_test<float, float, 1>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        { return rocrand_device::detail::uniform_distribution((unsigned int)(dis(gen) >> 32)); },
        [](float x) { return x; },
        [](float x, float actual_mean) { return POWF(x - actual_mean, 2); });
}
TEST(UniformHostTest, float_out_ulonglongint_in)
{
    run_host_num_test<float, float, 1>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        { return rocrand_device::detail::uniform_distribution(dis(gen)); },
        [](float x) { return x; },
        [](float x, float actual_mean) { return POWF(x - actual_mean, 2); });
}

TEST(UniformHostTest, float4_out_uint4_in)
{
    run_host_num_test<float4, float, 4>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        {
            unsigned long long a = dis(gen);
            unsigned long long b = dis(gen);

            return rocrand_device::detail::uniform_distribution4(
                uint4{static_cast<unsigned int>(a & 0xffffffff),
                      static_cast<unsigned int>(a >> 32),
                      static_cast<unsigned int>(b & 0xffffffff),
                      static_cast<unsigned int>(b >> 32)});
        },
        [](float4 x) { return x.w + x.x + x.y + x.z; },
        [](float4 x, float actual_mean)
        {
            return POWF(x.w - actual_mean, 2) + POWF(x.x - actual_mean, 2)
                   + POWF(x.y - actual_mean, 2) + POWF(x.z - actual_mean, 2);
        });
}

TEST(UniformHostTest, float4_out_ulonglong4_in)
{
    run_host_num_test<float4, float, 4>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        {
            return rocrand_device::detail::uniform_distribution4(
                ulonglong4{dis(gen), dis(gen), dis(gen), dis(gen)});
        },
        [](float4 x) { return x.w + x.x + x.y + x.z; },
        [](float4 x, float actual_mean)
        {
            return POWF(x.w - actual_mean, 2) + POWF(x.x - actual_mean, 2)
                   + POWF(x.y - actual_mean, 2) + POWF(x.z - actual_mean, 2);
        });
}

TEST(UniformHostTest, double_out_uint_in)
{
    run_host_num_test<double, double, 1>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen) {
            return rocrand_device::detail::uniform_distribution_double(
                (unsigned int)(dis(gen) >> 32));
        },
        [](double x) { return x; },
        [](double x, double actual_mean) { return POWF(x - actual_mean, 2); });
}

TEST(UniformHostTest, double_out_2uint_in)
{
    run_host_num_test<double, double, 1>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        {
            unsigned long long temp = dis(gen);
            return rocrand_device::detail::uniform_distribution_double(
                (unsigned int)(temp >> 32),
                (unsigned int)(temp & 0xffffffff));
        },
        [](double x) { return x; },
        [](double x, double actual_mean) { return POWF(x - actual_mean, 2); });
}

TEST(UniformHostTest, double_out_ulonglong_in)
{
    run_host_num_test<double, double, 1>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        { return rocrand_device::detail::uniform_distribution_double(dis(gen)); },
        [](double x) { return x; },
        [](double x, double actual_mean) { return POWF(x - actual_mean, 2); });
}

TEST(UniformHostTest, double2_out_uint4_in)
{
    run_host_num_test<double2, double, 2>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        {
            unsigned long long a = dis(gen);
            unsigned long long b = dis(gen);
            return rocrand_device::detail::uniform_distribution_double2(
                uint4{static_cast<unsigned int>(a & 0xffffffff),
                      static_cast<unsigned int>(a >> 32),
                      static_cast<unsigned int>(b & 0xffffffff),
                      static_cast<unsigned int>(b >> 32)});
        },
        [](double2 x) { return x.x + x.y; },
        [](double2 x, double actual_mean)
        { return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2); });
}

TEST(UniformHostTest, double4_out_2uint4_in)
{
    run_host_num_test<double4, double, 4>(

        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        {
            unsigned long long a = dis(gen);
            unsigned long long b = dis(gen);
            unsigned long long c = dis(gen);
            unsigned long long d = dis(gen);

            return rocrand_device::detail::uniform_distribution_double4(
                uint4{static_cast<unsigned int>(a & 0xffffffff),
                      static_cast<unsigned int>(a >> 32),
                      static_cast<unsigned int>(b & 0xffffffff),
                      static_cast<unsigned int>(b >> 32)},
                uint4{static_cast<unsigned int>(c & 0xffffffff),
                      static_cast<unsigned int>(c >> 32),
                      static_cast<unsigned int>(d & 0xffffffff),
                      static_cast<unsigned int>(d >> 32)});
        },
        [](double4 x) { return x.x + x.y + x.z + x.w; },

        [](double4 x, double actual_mean)
        {
            return std::pow(x.x - actual_mean, 2) + std::pow(x.y - actual_mean, 2)
                   + std::pow(x.z - actual_mean, 2) + std::pow(x.w - actual_mean, 2);
        });
}

TEST(UniformHostTest, double2_out_ulonglong2_in)
{
    run_host_num_test<double2, double, 2>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen) {
            return rocrand_device::detail::uniform_distribution_double2(
                ulonglong2{dis(gen), dis(gen)});
        },
        [](double2 x) { return x.x + x.y; },

        [](double2 x, double actual_mean)
        { return std::pow(x.x - actual_mean, 2) + std::pow(x.y - actual_mean, 2); });
}

TEST(UniformHostTest, double2_out_ulonglong4_in)
{
    run_host_num_test<double2, double, 2>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        {
            return rocrand_device::detail::uniform_distribution_double2(
                ulonglong4{dis(gen), dis(gen), dis(gen), dis(gen)});
        },
        [](double2 x) { return x.x + x.y; },

        [](double2 x, double actual_mean)
        { return std::pow(x.x - actual_mean, 2) + std::pow(x.y - actual_mean, 2); });
}

TEST(UniformHostTest, double4_out_ulonglong4_in)
{
    run_host_num_test<double4, double, 4>(
        [](std::uniform_int_distribution<unsigned long long>& dis, std::mt19937& gen)
        {
            return rocrand_device::detail::uniform_distribution_double4(
                ulonglong4{dis(gen), dis(gen), dis(gen), dis(gen)});
        },
        [](double4 x) { return x.x + x.y + x.z + x.w; },

        [](double4 x, double actual_mean)
        {
            return std::pow(x.x - actual_mean, 2) + std::pow(x.y - actual_mean, 2)
                   + std::pow(x.z - actual_mean, 2) + std::pow(x.w - actual_mean, 2);
        });
}

template<class RocrandPRNGType>
inline void GetRocrandState(RocrandPRNGType* host_state)
{
    if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol32>)
    {
        const unsigned int* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
        rocrand_init(directions, 123456, host_state);
    }
    // scrambled sobol32 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol32>)
    {
        const unsigned int* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
        rocrand_init(directions, 123456, 654321, host_state);
    }
    // sobol64 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol64>)
    {
        const unsigned long long* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
        rocrand_init(directions, 123456, host_state);
    }
    // scrambled sobol64 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol64>)
    {
        const unsigned long long* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
        rocrand_init(directions, 123456, 654321, host_state);
    }
    // lfsr113 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_lfsr113>)
    {
        rocrand_init({0xabcd, 0xdabc, 0xcdab, 0xbcda}, 0, 0, host_state);
    }
    else
    {
        rocrand_init(123456, 654321, 0, host_state);
    }
}

template<typename OutputType, class RocrandPRNGType, size_t OutSize>
struct HostParams
{
    using out_type                   = OutputType;
    using rng                        = RocrandPRNGType;
    static constexpr size_t out_size = OutSize;
};

using UniformParams = ::testing::Types<HostParams<float, rocrand_state_philox4x32_10, 1>,
                                       HostParams<float, rocrand_state_mrg31k3p, 1>,
                                       HostParams<float, rocrand_state_mrg32k3a, 1>,
                                       HostParams<float, rocrand_state_xorwow, 1>,
                                       HostParams<float, rocrand_state_sobol32, 1>,
                                       HostParams<float, rocrand_state_scrambled_sobol32, 1>,
                                       HostParams<float, rocrand_state_sobol64, 1>,
                                       HostParams<float, rocrand_state_scrambled_sobol64, 1>,
                                       HostParams<float, rocrand_state_lfsr113, 1>,
                                       HostParams<float, rocrand_state_threefry2x32_20, 1>,
                                       HostParams<float, rocrand_state_threefry2x64_20, 1>,
                                       HostParams<float, rocrand_state_threefry4x32_20, 1>,
                                       HostParams<float, rocrand_state_threefry4x64_20, 1>,
                                       HostParams<double, rocrand_state_philox4x32_10, 1>,
                                       HostParams<double, rocrand_state_mrg31k3p, 1>,
                                       HostParams<double, rocrand_state_mrg32k3a, 1>,
                                       HostParams<double, rocrand_state_xorwow, 1>,
                                       HostParams<double, rocrand_state_sobol32, 1>,
                                       HostParams<double, rocrand_state_scrambled_sobol32, 1>,
                                       HostParams<double, rocrand_state_sobol64, 1>,
                                       HostParams<double, rocrand_state_scrambled_sobol64, 1>,
                                       HostParams<double, rocrand_state_lfsr113, 1>,
                                       HostParams<double, rocrand_state_threefry2x32_20, 1>,
                                       HostParams<double, rocrand_state_threefry2x64_20, 1>,
                                       HostParams<double, rocrand_state_threefry4x32_20, 1>,
                                       HostParams<double, rocrand_state_threefry4x64_20, 1>,
                                       HostParams<float2, rocrand_state_philox4x32_10, 2>,
                                       HostParams<double2, rocrand_state_philox4x32_10, 2>,
                                       HostParams<float4, rocrand_state_philox4x32_10, 4>,
                                       HostParams<double4, rocrand_state_philox4x32_10, 4>>;

template<typename OutputType,
         typename T,
         size_t OutSize,
         class RocrandPRNGType,
         class UniformFunc,
         class ReadMeanFunc,
         class ReadStdFunc>
void run_host_prng_test(const UniformFunc& lnf, const ReadMeanFunc& rmf, ReadStdFunc& rsf)
{
    constexpr size_t test_size        = 50000;
    const T          expected_mean    = 0.5;
    const T          expected_std_dev = 0.288675134595; //sqrt(1/12)

    RocrandPRNGType generator;
    GetRocrandState(&generator);

    std::vector<OutputType> output(test_size);

    for(size_t i = 0; i < test_size; i++)
    {
        output[i] = lnf(&generator);
    }

    T actual_mean = std::accumulate(output.begin(),
                                    output.end(),
                                    (T)0,
                                    [=](T acc, OutputType x) { return acc + rmf(x); })
                    / static_cast<T>(test_size * OutSize);

    T actual_std_dev
        = std::accumulate(output.begin(),
                          output.end(),
                          (T)0,
                          [=](T acc, OutputType x) { return acc + rsf(x, actual_mean); });
    actual_std_dev = std::sqrt(actual_std_dev / static_cast<T>(test_size * OutSize - 1));

    T mean_eps    = GET_EPS(expected_mean);
    T std_dev_eps = GET_EPS(expected_std_dev);

    ASSERT_NEAR(expected_mean, actual_mean, mean_eps);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, std_dev_eps);
}

template<class HostParams>
class UniformRocRandStateHostTest : public ::testing::Test
{
public:
    using out_type                   = typename HostParams::out_type;
    using prng_type                  = typename HostParams::rng;
    static constexpr size_t out_size = HostParams::out_size;
};

TYPED_TEST_SUITE(UniformRocRandStateHostTest, UniformParams);
TYPED_TEST(UniformRocRandStateHostTest, rocrand_state_tests)
{
    using out_type            = typename TestFixture::out_type;
    using rocrand_state       = typename TestFixture::prng_type;
    constexpr size_t out_size = TestFixture::out_size;
    using T
        = std::conditional_t<(std::is_same_v<out_type, float> || std::is_same_v<out_type, float2>
                              || std::is_same_v<out_type, float4>),
                             float,
                             double>;

    if constexpr(out_size == 1)
    {
        auto mean_func = [](out_type x) { return x; };
        auto std_dev_func
            = [](out_type x, out_type actual_mean) { return POWF(x - actual_mean, 2); };
        if constexpr(std::is_same_v<out_type, float>)
        {
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_uniform(state); },
                mean_func,
                std_dev_func);
        }
        else
        {
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_uniform_double(state); },
                mean_func,
                std_dev_func);
        }
    }
    else if constexpr(out_size == 2)
    {
        auto mean_func    = [](out_type x) { return x.x + x.y; };
        auto std_dev_func = [](out_type x, T actual_mean)
        { return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2); };

        if constexpr(std::is_same_v<out_type, float2>)
        {
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_uniform2(state); },
                mean_func,
                std_dev_func);
        }
        else
        {
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_uniform_double2(state); },
                mean_func,
                std_dev_func);
        }
    }
    else
    {
        auto mean_func    = [](out_type x) { return x.x + x.y + x.w + x.z; };
        auto std_dev_func = [](out_type x, T actual_mean)
        {
            return POWF(x.x - actual_mean, 2) + POWF(x.y - actual_mean, 2)
                   + POWF(x.w - actual_mean, 2) + POWF(x.z - actual_mean, 2);
        };

        if constexpr(std::is_same_v<out_type, float4>)
        {
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_uniform4(state); },
                mean_func,
                std_dev_func);
        }
        else
        {
            run_host_prng_test<out_type, T, out_size, rocrand_state>(
                [=](rocrand_state* state) { return rocrand_uniform_double4(state); },
                mean_func,
                std_dev_func);
        }
    }
}
