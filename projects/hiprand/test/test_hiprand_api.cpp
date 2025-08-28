// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hiprand/hiprand.h>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

constexpr hiprandRngType_t hiprand_rng_types[] = {HIPRAND_RNG_PSEUDO_XORWOW,
                                                  HIPRAND_RNG_PSEUDO_MRG32K3A,
                                                  HIPRAND_RNG_PSEUDO_MTGP32,
                                                  HIPRAND_RNG_PSEUDO_MT19937,
                                                  HIPRAND_RNG_PSEUDO_PHILOX4_32_10,
                                                  HIPRAND_RNG_QUASI_SOBOL32,
                                                  HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32,
                                                  HIPRAND_RNG_QUASI_SOBOL64,
                                                  HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64};

constexpr hiprandRngType_t hiprand_rng_types_32[] = {HIPRAND_RNG_PSEUDO_XORWOW,
                                                     HIPRAND_RNG_PSEUDO_MRG32K3A,
                                                     HIPRAND_RNG_PSEUDO_MTGP32,
                                                     HIPRAND_RNG_PSEUDO_MT19937,
                                                     HIPRAND_RNG_PSEUDO_PHILOX4_32_10,
                                                     HIPRAND_RNG_QUASI_SOBOL32,
                                                     HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32};

constexpr hiprandRngType_t hiprand_rng_types_64[]
    = {HIPRAND_RNG_QUASI_SOBOL64, HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64};

constexpr hiprandOrdering_t hiprand_ordering_types[] = {HIPRAND_ORDERING_PSEUDO_BEST,
                                                        HIPRAND_ORDERING_PSEUDO_DEFAULT,
                                                        HIPRAND_ORDERING_PSEUDO_SEEDED,
                                                        HIPRAND_ORDERING_PSEUDO_LEGACY,
                                                        HIPRAND_ORDERING_PSEUDO_DYNAMIC,
                                                        HIPRAND_ORDERING_QUASI_DEFAULT};

// Not all generators have been implemented on the host yet.
constexpr hiprandRngType_t hiprand_host_rng_types[] = {HIPRAND_RNG_PSEUDO_PHILOX4_32_10};

class hiprand_api : public ::testing::TestWithParam<hiprandRngType_t>
{};

class hiprand_api_32 : public ::testing::TestWithParam<hiprandRngType_t>
{};

class hiprand_api_64 : public ::testing::TestWithParam<hiprandRngType_t>
{};

class hiprand_ordering : public ::testing::TestWithParam<hiprandOrdering_t>
{};

class hiprand_host : public ::testing::TestWithParam<hiprandRngType_t>
{};

void hiprand_generate_test_func(hiprandRngType_t rng_type)
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(unsigned int))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    HIPRAND_CHECK(hiprandGenerate(generator, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / UINT_MAX;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_api_32, hiprand_generate_test)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_test_func(rng_type);
}

void hiprand_generate_long_long_test_func(hiprandRngType_t rng_type)
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t            output_size = 8192;
    unsigned long long int* output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(unsigned long long int)));
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    HIPRAND_CHECK(hiprandGenerateLongLong(generator, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned long long int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned long long int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / static_cast<double>(ULLONG_MAX);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_api_64, hiprand_generate_long_long_test)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_long_long_test_func(rng_type);
}

void hiprand_generate_uniform_test_func(hiprandRngType_t rng_type)
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(float))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    HIPRAND_CHECK(hiprandGenerateUniform(generator, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(float),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_api, hiprand_generate_uniform_test)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_uniform_test_func(rng_type);
}

void hiprand_generate_uniform_double_test_func(hiprandRngType_t rng_type)
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    double * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(double))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    HIPRAND_CHECK(hiprandGenerateUniformDouble(generator, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(double),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_api, hiprand_generate_uniform_double_test)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_uniform_double_test_func(rng_type);
}

void hiprand_generate_normal_test_func(hiprandRngType_t rng_type)
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(float))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    HIPRAND_CHECK(hiprandGenerateNormal(generator, output, output_size, 0, 1));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(float),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_api, hiprand_generate_normal_test)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_normal_test_func(rng_type);
}

void hiprand_generate_normal_double_test_func(hiprandRngType_t rng_type)
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    double * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(double))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    HIPRAND_CHECK(hiprandGenerateNormalDouble(generator, output, output_size, 0, 1));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(double),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_api, hiprand_generate_normal_double_test)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_normal_double_test_func(rng_type);
}

void hiprand_generate_lognormal_test_func(hiprandRngType_t rng_type)
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(float))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    HIPRAND_CHECK(hiprandGenerateLogNormal(generator, output, output_size, 1.6f, 0.25f));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(float),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_size);

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd = std::sqrt(std::log(1.0f + stddev/(mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_api, hiprand_generate_lognormal_test)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_lognormal_test_func(rng_type);
}

void hiprand_generate_lognormal_double_test_func(hiprandRngType_t rng_type)
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    double * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(double))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    HIPRAND_CHECK(hiprandGenerateLogNormalDouble(generator, output, output_size, 1.6, 0.25));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(double),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_size);

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd = std::sqrt(std::log(1.0f + stddev/(mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_api, hiprand_generate_lognormal_double_test)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_lognormal_double_test_func(rng_type);
}

void hiprand_generate_poisson_test_func(hiprandRngType_t rng_type)
{
    double lambda = 20.0;

    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(
        hipMallocHelper((void **)&output,
        output_size * sizeof(unsigned int))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    HIPRAND_CHECK(hiprandGeneratePoisson(generator, output, output_size, lambda));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_api, hiprand_generate_poisson_test)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_poisson_test_func(rng_type);
}

INSTANTIATE_TEST_SUITE_P(hiprand_32, hiprand_api_32, ::testing::ValuesIn(hiprand_rng_types_32));

INSTANTIATE_TEST_SUITE_P(hiprand_64, hiprand_api_64, ::testing::ValuesIn(hiprand_rng_types_64));

INSTANTIATE_TEST_SUITE_P(hiprand, hiprand_api, ::testing::ValuesIn(hiprand_rng_types));

TEST_P(hiprand_ordering, hiprand_ordering_test)
{
    const hiprandOrdering_t ordering = GetParam();

    hiprandGenerator_t     generator = 0;
    const hiprandRngType_t rng_type  = ordering == HIPRAND_ORDERING_QUASI_DEFAULT
                                           ? HIPRAND_RNG_QUASI_SOBOL32
                                           : HIPRAND_RNG_PSEUDO_XORWOW;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));
    HIPRAND_CHECK(hiprandSetGeneratorOrdering(generator, ordering));

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper((void**)&output, output_size * sizeof(unsigned int)));

    HIPRAND_CHECK(hiprandGenerate(generator, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(output));

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_ordering, hiprand_invalid_ordering_test)
{
    const hiprandOrdering_t ordering = GetParam();
    const hiprandRngType_t  rng_type = ordering == HIPRAND_ORDERING_QUASI_DEFAULT
                                           ? HIPRAND_RNG_PSEUDO_XORWOW
                                           : HIPRAND_RNG_QUASI_SOBOL32;

    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));
    ASSERT_EQ(hiprandSetGeneratorOrdering(generator, ordering), HIPRAND_STATUS_OUT_OF_RANGE);
    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

INSTANTIATE_TEST_SUITE_P(hiprand, hiprand_ordering, ::testing::ValuesIn(hiprand_ordering_types));

void hiprand_generate_uniform_host_test_func(hiprandRngType_t rng_type)
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGeneratorHost(&generator, rng_type));

    constexpr size_t   output_size = 8192;
    std::vector<float> output_host(output_size);

    // generate
    HIPRAND_CHECK(hiprandGenerateUniform(generator, output_host.data(), output_size));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST_P(hiprand_host, hiprand_generate_uniform_host)
{
    const hiprandRngType_t rng_type = GetParam();
    hiprand_generate_uniform_host_test_func(rng_type);
}

INSTANTIATE_TEST_SUITE_P(hiprand, hiprand_host, ::testing::ValuesIn(hiprand_host_rng_types));
