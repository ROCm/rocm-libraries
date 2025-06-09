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

#include <rng/distribution/log_normal.hpp>
#include <rocrand/rocrand_mtgp32_11213.h>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)

using namespace rocrand_impl::host;

TEST(log_normal_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                   size = 4000;
    float                          val[size];
    log_normal_distribution<float> u(0.2f, 0.5f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[2];
        float        output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std
        = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}

TEST(log_normal_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                    size = 4000;
    double                          val[size];
    log_normal_distribution<double> u(0.2, 0.5);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[4];
        double       output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        input[2] = dis(gen);
        input[3] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    double expected_mean = std::exp(0.2 + 0.5 * 0.5 / 2);
    double expected_std  = std::sqrt((std::exp(0.5 * 0.5) - 1.0) * std::exp(2 * 0.2 + 0.5 * 0.5));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1);
}

TEST(log_normal_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t                  size = 4000;
    half                          val[size];
    log_normal_distribution<half> u(__float2half(0.2f), __float2half(0.5f));

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[1];
        half         output[2];
        input[0] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += __half2float(output[0]) + __half2float(output[1]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std
        = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}

template<typename mrg, unsigned int m1>
struct mrg_log_normal_distribution_test_type
{
    typedef mrg                   mrg_type;
    static constexpr unsigned int mrg_m1 = m1;
};

template<typename test_type>
struct mrg_log_normal_distribution_tests : public ::testing::Test
{
    typedef typename test_type::mrg_type mrg_type;
    static constexpr unsigned int        mrg_m1 = test_type::mrg_m1;
};

typedef ::testing::Types<
    mrg_log_normal_distribution_test_type<rocrand_state_mrg31k3p, ROCRAND_MRG31K3P_M1>,
    mrg_log_normal_distribution_test_type<rocrand_state_mrg32k3a, ROCRAND_MRG32K3A_M1>>
    mrg_log_normal_distribution_test_types;

TYPED_TEST_SUITE(mrg_log_normal_distribution_tests, mrg_log_normal_distribution_test_types);

TYPED_TEST(mrg_log_normal_distribution_tests, float_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    const size_t                                                              size = 4000;
    float                                                                     val[size];
    mrg_engine_log_normal_distribution<float, typename TestFixture::mrg_type> u(0.2f, 0.5f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[2];
        float        output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std
        = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}

TYPED_TEST(mrg_log_normal_distribution_tests, double_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    const size_t                                                               size = 4000;
    double                                                                     val[size];
    mrg_engine_log_normal_distribution<double, typename TestFixture::mrg_type> u(0.2, 0.5);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[2];
        double       output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    double expected_mean = std::exp(0.2 + 0.5 * 0.5 / 2);
    double expected_std  = std::sqrt((std::exp(0.5 * 0.5) - 1.0) * std::exp(2 * 0.2 + 0.5 * 0.5));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1);
}

TYPED_TEST(mrg_log_normal_distribution_tests, half_test)
{
    std::random_device                          rd;
    std::mt19937                                gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, TestFixture::mrg_m1);

    const size_t                                                             size = 4000;
    half                                                                     val[size];
    mrg_engine_log_normal_distribution<half, typename TestFixture::mrg_type> u(__float2half(0.2f),
                                                                               __float2half(0.5f));

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 2)
    {
        unsigned int input[1];
        half         output[2];
        input[0] = dis(gen);
        u(input, output);
        val[i]     = output[0];
        val[i + 1] = output[1];
        mean += __half2float(output[0]) + __half2float(output[1]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std
        = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}

template<typename T>
struct sobol_log_normal_distribution_tests : public ::testing::Test
{
    using sobol_type = T;
};

using SobolReturnTypes = ::testing::Types<unsigned int, unsigned long long int>;

TYPED_TEST_SUITE(sobol_log_normal_distribution_tests, SobolReturnTypes);

TYPED_TEST(sobol_log_normal_distribution_tests, float_test)
{
    using T = typename TestFixture::sobol_type;

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_int_distribution<T> dis;

    const size_t                         size = 4000;
    float                                val[size];
    sobol_log_normal_distribution<float> u(0.2f, 0.5f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 1)
    {
        val[i] = u(dis(gen));
        mean += val[i];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std
        = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}
TYPED_TEST(sobol_log_normal_distribution_tests, double_test)
{
    using T = typename TestFixture::sobol_type;

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_int_distribution<T> dis;

    const size_t                          size = 4000;
    double                                val[size];
    sobol_log_normal_distribution<double> u(0.2, 0.5);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i += 1)
    {
        val[i] = u(dis(gen));
        mean += val[i];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    double expected_mean = std::exp(0.2 + 0.5 * 0.5 / 2);
    double expected_std  = std::sqrt((std::exp(0.5 * 0.5) - 1.0) * std::exp(2 * 0.2 + 0.5 * 0.5));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1);
}

TYPED_TEST(sobol_log_normal_distribution_tests, half_test)
{
    using T = typename TestFixture::sobol_type;

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_int_distribution<T> dis;

    const size_t                        size = 4000;
    half                                val[size];
    sobol_log_normal_distribution<half> u(__float2half(0.2f), __float2half(0.5f));

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i += 1)
    {
        val[i] = u(dis(gen));
        mean += __half2float(val[i]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);
    
    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std
    = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));
    
    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}


template <typename OutType>
struct StatesLND{
    template <typename FuncCall>
    void run_test(const FuncCall & f, size_t testSize = 4000000){
        double iMean = 0;
        double iStd = 1;

        float * output = new float [testSize];
        OutType out;
    
        double mean = 0;
    
        for(size_t i = 0; i <= testSize; i += 4){
            f(out, iMean, iStd);
    
            output[i] = out.w;
            output[i + 1] = out.x;
            output[i + 2] = out.y;
            output[i + 3] = out.z;
            mean += out.w + out.x + out.y + out.z;
        }
    
        mean /= testSize;
    
        double std = 0.0;
        for(size_t i = 0; i < testSize; i++)
            std += std::pow(output[i] - mean, 2);
    
        std = std::sqrt(std / testSize);
    
        double eMean = std::exp(iMean + (iStd * iStd) / 2);
        double eStd = std::sqrt(std::log(1 + (iStd * iStd)/(iMean * iMean)));
        ASSERT_NEAR(mean, eMean, eMean * 0.1) << "Expected Mean: " << eMean << " Actual Mean: " << mean << " Eps: " << eMean * 0.1;
        ASSERT_NEAR(std, eStd, eStd * 0.1) << "Expected Std: " << eStd << " Actual Std: " << std << " Eps: " << eStd * 0.1;
    }
};

TEST(log_normal_distribution_tests, philox4x32_10_test){
    rocrand_state_philox4x32_10 states;
    rocrand_init(123456, 654321, 0, &states);

    StatesLND<float4> testFloat;
    StatesLND<double4> testDouble;

    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        testFloat.run_test()(
            [&] __host__ __device__ (float4 & output, float mean, float std){
                output = {
                    rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                    rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
                };
        )
        testDouble.run_test()(
            [&] __host__ __device__ (double4 & output, double mean, double std){
                output = {
                    rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                    rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
                };
            } 
        )
    #endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            float2 o1 = rocrand_log_normal2(&states, mean, std);
            float2 o2 = rocrand_log_normal2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = rocrand_log_normal4(&states, mean, std);
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            double2 o1 = rocrand_log_normal_double2(&states, mean, std);
            double2 o2 = rocrand_log_normal_double2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = rocrand_log_normal_double4(&states, mean, std);
        }
    );
}

TEST(log_normal_distribution_tests, mrg31k3p_test){
    rocrand_state_mrg31k3p states;
    rocrand_init(123456, 654321, 0, &states);

    StatesLND<float4> testFloat;
    StatesLND<double4> testDouble;

    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        testFloat.run_test()(
            [&] __host__ __device__ (float4 & output, float mean, float std){
                output = {
                    rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                    rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
                };
        )
        testDouble.run_test()(
            [&] __host__ __device__ (double4 & output, double mean, double std){
                output = {
                    rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                    rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
                };
            } 
        )
    #endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            float2 o1 = rocrand_log_normal2(&states, mean, std);
            float2 o2 = rocrand_log_normal2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            double2 o1 = rocrand_log_normal_double2(&states, mean, std);
            double2 o2 = rocrand_log_normal_double2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );
}

TEST(log_normal_distribution_tests, mrg32k3a_test){
    rocrand_state_mrg32k3a states;
    rocrand_init(123456, 654321, 0, &states);

    StatesLND<float4> testFloat;
    StatesLND<double4> testDouble;

    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        testFloat.run_test()(
            [&] __host__ __device__ (float4 & output, float mean, float std){
                output = {
                    rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                    rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
                };
        )
        testDouble.run_test()(
            [&] __host__ __device__ (double4 & output, double mean, double std){
                output = {
                    rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                    rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
                };
            } 
        )
    #endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            float2 o1 = rocrand_log_normal2(&states, mean, std);
            float2 o2 = rocrand_log_normal2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            double2 o1 = rocrand_log_normal_double2(&states, mean, std);
            double2 o2 = rocrand_log_normal_double2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );
}

TEST(log_normal_distribution_tests, xorwow_test){
    rocrand_state_xorwow states;
    rocrand_init(123456, 654321, 0, &states);

    StatesLND<float4> testFloat;
    StatesLND<double4> testDouble;

    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        testFloat.run_test()(
            [&] __host__ __device__ (float4 & output, float mean, float std){
                output = {
                    rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                    rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
                };
        )
        testDouble.run_test()(
            [&] __host__ __device__ (double4 & output, double mean, double std){
                output = {
                    rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                    rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
                };
            } 
        )
    #endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            float2 o1 = rocrand_log_normal2(&states, mean, std);
            float2 o2 = rocrand_log_normal2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            double2 o1 = rocrand_log_normal_double2(&states, mean, std);
            double2 o2 = rocrand_log_normal_double2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );
}

TEST(log_normal_distribution_tests, sobol32_test){
    rocrand_state_sobol32 states;
    const unsigned int* directions;
    HIP_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    rocrand_init(directions, 0, &states);

    StatesLND<float4> testFloat;

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = {
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
            };
        }
    );

    StatesLND<double4> testDouble;

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = {
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
            };
        }
    );
}

TEST(log_normal_distribution_tests, scrambled_sobol32_test){
    rocrand_state_scrambled_sobol32 states;
    const unsigned int* directions;
    HIP_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    rocrand_init(directions, 123456, 0, &states);

    StatesLND<float4> testFloat;

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = {
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
            };
        }
    );

    StatesLND<double4> testDouble;

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = {
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
            };
        }
    );
}

TEST(log_normal_distribution_tests, sobol64_test){
    rocrand_state_sobol64 states;
    const unsigned long long* directions;
    HIP_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
    rocrand_init(directions, 0, &states);

    StatesLND<float4> testFloat;

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = {
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
            };
        }
    );

    StatesLND<double4> testDouble;

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = {
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
            };
        }
    );
}

TEST(log_normal_distribution_tests, scrambled_sobol64_test){
    rocrand_state_scrambled_sobol64 states;
    const unsigned long long* directions;
    HIP_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
    rocrand_init(directions, 123456, 0, &states);

    StatesLND<float4> testFloat;

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = {
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
            };
        }
    );

    StatesLND<double4> testDouble;

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = {
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
            };
        }
    );
}

TEST(log_normal_distribution_tests, lfsr113_test){
    rocrand_state_lfsr113 states;
    rocrand_init(static_cast<uint4>(12), 0, &states);

    StatesLND<float4> testFloat;

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = {
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
            };
        }
    );

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            float2 o1 = rocrand_log_normal2(&states, mean, std);
            float2 o2 = rocrand_log_normal2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    StatesLND<double4> testDouble;

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = {
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
            };
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            double2 o1 = rocrand_log_normal_double2(&states, mean, std);
            double2 o2 = rocrand_log_normal_double2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );
}

TEST(log_normal_distribution_tests, threefry2x32_20_test){
    rocrand_state_threefry2x32_20 states;
    rocrand_init(123456, 654321, 0, &states);

    StatesLND<float4> testFloat;

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = {
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
            };
        }
    );

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            float2 o1 = rocrand_log_normal2(&states, mean, std);
            float2 o2 = rocrand_log_normal2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    StatesLND<double4> testDouble;

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = {
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
            };
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            double2 o1 = rocrand_log_normal_double2(&states, mean, std);
            double2 o2 = rocrand_log_normal_double2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );
}

TEST(log_normal_distribution_tests, threefry2x64_20_test){
    rocrand_state_threefry2x64_20 states;
    rocrand_init(123456, 654321, 0, &states);

    StatesLND<float4> testFloat;

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = {
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
            };
        }
    );

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            float2 o1 = rocrand_log_normal2(&states, mean, std);
            float2 o2 = rocrand_log_normal2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    StatesLND<double4> testDouble;

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = {
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
            };
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            double2 o1 = rocrand_log_normal_double2(&states, mean, std);
            double2 o2 = rocrand_log_normal_double2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );
}

TEST(log_normal_distribution_tests, threefry4x32_20_test){
    rocrand_state_threefry4x32_20 states;
    rocrand_init(123456, 654321, 0, &states);

    StatesLND<float4> testFloat;

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = {
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
            };
        }
    );

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            float2 o1 = rocrand_log_normal2(&states, mean, std);
            float2 o2 = rocrand_log_normal2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    StatesLND<double4> testDouble;

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = {
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
            };
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            double2 o1 = rocrand_log_normal_double2(&states, mean, std);
            double2 o2 = rocrand_log_normal_double2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );
}

TEST(log_normal_distribution_tests, threefry4x64_20_test){
    rocrand_state_threefry4x64_20 states;
    rocrand_init(123456, 654321, 0, &states);

    StatesLND<float4> testFloat;

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            output = {
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std),
                rocrand_log_normal(&states, mean, std), rocrand_log_normal(&states, mean, std)
            };
        }
    );

    testFloat.run_test(
        [&] __host__ __device__ (float4 & output, float mean, float std){
            float2 o1 = rocrand_log_normal2(&states, mean, std);
            float2 o2 = rocrand_log_normal2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );

    StatesLND<double4> testDouble;

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            output = {
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std),
                rocrand_log_normal_double(&states, mean, std), rocrand_log_normal_double(&states, mean, std)
            };
        }
    );

    testDouble.run_test(
        [&] __host__ __device__ (double4 & output, double mean, double std){
            double2 o1 = rocrand_log_normal_double2(&states, mean, std);
            double2 o2 = rocrand_log_normal_double2(&states, mean, std);
            output = {
                o1.x, o2.x, o1.y, o2.y
            };
        }
    );
}

template <typename T, typename pType, typename LNDFunction>
__global__ void mtgp32_kernel (rocrand_state_mtgp32 * states, T * output, const size_t N, const pType mean, const pType std ,const LNDFunction & f){
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(index >= N)
        return;

    __shared__ rocrand_state_mtgp32 state;
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    output[index] = f(&state, mean, std);

    if(thread_id == 0)
        states[state_id] = state; 
}

TEST(log_normal_distribution_tests, float_mtgp32_test){
    size_t testSize = 40192;
    size_t threads = 256;
    size_t blocks = std::ceil(static_cast<double>(testSize) / static_cast<double>(threads));

    float iMean = 0;
    float iStd = 1;

    rocrand_state_mtgp32 * states;
    size_t state_size = blocks, seed = 654321;
    HIP_CHECK(hipMalloc(&states, state_size * sizeof(rocrand_state_mtgp32)));
    rocrand_make_state_mtgp32(states,  mtgp32dc_params_fast_11213, state_size, seed);

    float * hOut = new float[testSize];
    float * dOut;
    HIP_CHECK(hipMalloc(&dOut, sizeof(float) * testSize));
    HIP_CHECK(hipDeviceSynchronize());


    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(mtgp32_kernel<float, float>),
        dim3(blocks),
        dim3(threads),
        0,
        0,
        states,
        dOut,
        testSize,
        iMean,
        iStd,

        [] __device__ (rocrand_state_mtgp32 * state, float mean, float std){
            return rocrand_log_normal(state, mean, std);
        }
    );

    HIP_CHECK(hipMemcpy(hOut, dOut, sizeof(float) * testSize, hipMemcpyDeviceToHost));

    double mean = 0.0;
    for(size_t i = 0; i < testSize; i++)
        mean += hOut[i];
    

    mean /= testSize;

    double std = 0.0;
    for(size_t i = 0; i < testSize; i++)
        std += std::pow(hOut[i] - mean, 2);

    std = std::sqrt(std / testSize);

    double eMean = std::exp(iMean + (iStd * iStd) / 2);
    double eStd = std::sqrt(std::log(1 + (iStd * iStd)/(iMean * iMean)));

    ASSERT_NEAR(mean, eMean, eMean * 0.1) << "Expected Mean: " << eMean << " Actual Mean: " << mean << " Eps: " << eMean * 0.1;
    ASSERT_NEAR(std, eStd, eStd * 0.1) << "Expected Std: " << eStd << " Actual Std: " << std << " Eps: " << eStd * 0.1;

    HIP_CHECK(hipFree(states));
    HIP_CHECK(hipFree(dOut));

    delete [] hOut;
}

TEST(log_normal_distribution_tests, float2_mtgp32_test){
    size_t testSize = 40192;
    size_t threads = 256;
    size_t blocks = std::ceil(static_cast<double>(testSize) / static_cast<double>(threads));

    float iMean = 0;
    float iStd = 1;

    rocrand_state_mtgp32 * states;
    size_t state_size = blocks, seed = 654321;
    HIP_CHECK(hipMalloc(&states, state_size * sizeof(rocrand_state_mtgp32)));
    rocrand_make_state_mtgp32(states,  mtgp32dc_params_fast_11213, state_size, seed);

    float2 * hOut = new float2[testSize];
    float2 * dOut;
    HIP_CHECK(hipMalloc(&dOut, sizeof(float2) * testSize));
    HIP_CHECK(hipDeviceSynchronize());


    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(mtgp32_kernel<float2, float>),
        dim3(blocks),
        dim3(threads),
        0,
        0,
        states,
        dOut,
        testSize,
        iMean,
        iStd,
        [] __device__ (rocrand_state_mtgp32 * state, float mean, float std){
            return rocrand_log_normal2(state, mean, std);
        }
    );

    HIP_CHECK(hipMemcpy(hOut, dOut, sizeof(float2) * testSize, hipMemcpyDeviceToHost));

    double mean = 0.0;
    for(size_t i = 0; i < testSize; i++){
        mean += hOut[i].x;
        mean += hOut[i].y;
    }
    

    mean /= (testSize * 2);

    double std = 0.0;
    for(size_t i = 0; i < testSize; i++){
        std += std::pow(hOut[i].x - mean, 2);
        std += std::pow(hOut[i].y - mean, 2);
    }

    std = std::sqrt(std / (testSize * 2));

    double eMean = std::exp(iMean + (iStd * iStd) / 2);
    double eStd = std::sqrt(std::log(1 + (iStd * iStd)/(iMean * iMean)));

    ASSERT_NEAR(mean, eMean, eMean * 0.1) << "Expected Mean: " << eMean << " Actual Mean: " << mean << " Eps: " << eMean * 0.1;
    ASSERT_NEAR(std, eStd, eStd * 0.1) << "Expected Std: " << eStd << " Actual Std: " << std << " Eps: " << eStd * 0.1;

    HIP_CHECK(hipFree(states));
    HIP_CHECK(hipFree(dOut));

    delete [] hOut;
}

TEST(log_normal_distribution_tests, double_mtgp32_test){
    size_t testSize = 40192;
    size_t threads = 256;
    size_t blocks = std::ceil(static_cast<double>(testSize) / static_cast<double>(threads));

    double iMean = 0;
    double iStd = 1;

    rocrand_state_mtgp32 * states;
    size_t state_size = blocks, seed = 654321;
    HIP_CHECK(hipMalloc(&states, state_size * sizeof(rocrand_state_mtgp32)));
    rocrand_make_state_mtgp32(states,  mtgp32dc_params_fast_11213, state_size, seed);

    double * hOut = new double[testSize];
    double * dOut;
    HIP_CHECK(hipMalloc(&dOut, sizeof(double) * testSize));
    HIP_CHECK(hipDeviceSynchronize());


    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(mtgp32_kernel<double, double>),
        dim3(blocks),
        dim3(threads),
        0,
        0,
        states,
        dOut,
        testSize,
        iMean,
        iStd,

        [] __device__ (rocrand_state_mtgp32 * state, double mean, double std){
            return rocrand_log_normal_double(state, mean, std);
        }
    );

    HIP_CHECK(hipMemcpy(hOut, dOut, sizeof(double) * testSize, hipMemcpyDeviceToHost));

    double mean = 0.0;
    for(size_t i = 0; i < testSize; i++)
        mean += hOut[i];
    

    mean /= testSize;

    double std = 0.0;
    for(size_t i = 0; i < testSize; i++)
        std += std::pow(hOut[i] - mean, 2);

    std = std::sqrt(std / testSize);

    double eMean = std::exp(iMean + (iStd * iStd) / 2);
    double eStd = std::sqrt(std::log(1 + (iStd * iStd)/(iMean * iMean)));

    ASSERT_NEAR(mean, eMean, eMean * 0.1) << "Expected Mean: " << eMean << " Actual Mean: " << mean << " Eps: " << eMean * 0.1;
    ASSERT_NEAR(std, eStd, eStd * 0.1) << "Expected Std: " << eStd << " Actual Std: " << std << " Eps: " << eStd * 0.1;

    HIP_CHECK(hipFree(states));
    HIP_CHECK(hipFree(dOut));

    delete [] hOut;
}

TEST(log_normal_distribution_tests, double2_mtgp32_test){
    size_t testSize = 40192;
    size_t threads = 256;
    size_t blocks = std::ceil(static_cast<double>(testSize) / static_cast<double>(threads));

    double iMean = 0;
    double iStd = 1;

    rocrand_state_mtgp32 * states;
    size_t state_size = blocks, seed = 654321;
    HIP_CHECK(hipMalloc(&states, state_size * sizeof(rocrand_state_mtgp32)));
    rocrand_make_state_mtgp32(states,  mtgp32dc_params_fast_11213, state_size, seed);

    double2 * hOut = new double2[testSize];
    double2 * dOut;
    HIP_CHECK(hipMalloc(&dOut, sizeof(double2) * testSize));
    HIP_CHECK(hipDeviceSynchronize());


    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(mtgp32_kernel<double2, double>),
        dim3(blocks),
        dim3(threads),
        0,
        0,
        states,
        dOut,
        testSize,
        iMean,
        iStd,
        [] __device__ (rocrand_state_mtgp32 * state, double mean, double std){
            return rocrand_log_normal_double2(state, mean, std);
        }
    );

    HIP_CHECK(hipMemcpy(hOut, dOut, sizeof(double2) * testSize, hipMemcpyDeviceToHost));

    double mean = 0.0;
    for(size_t i = 0; i < testSize; i++){
        mean += hOut[i].x;
        mean += hOut[i].y;
    }
    

    mean /= (testSize * 2);

    double std = 0.0;
    for(size_t i = 0; i < testSize; i++){
        std += std::pow(hOut[i].x - mean, 2);
        std += std::pow(hOut[i].y - mean, 2);
    }

    std = std::sqrt(std / (testSize * 2));

    double eMean = std::exp(iMean + (iStd * iStd) / 2);
    double eStd = std::sqrt(std::log(1 + (iStd * iStd)/(iMean * iMean)));

    ASSERT_NEAR(mean, eMean, eMean * 0.1) << "Expected Mean: " << eMean << " Actual Mean: " << mean << " Eps: " << eMean * 0.1;
    ASSERT_NEAR(std, eStd, eStd * 0.1) << "Expected Std: " << eStd << " Actual Std: " << std << " Eps: " << eStd * 0.1;

    HIP_CHECK(hipFree(states));
    HIP_CHECK(hipFree(dOut));

    delete [] hOut;
}
