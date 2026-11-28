// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"

#include <cstring>
#include <limits>
#include <random>
#include <vector>

class rocrand_generate_poisson_tests : public ::testing::TestWithParam<rocrand_rng_type>
{};

const double invalid_lambdas[] = {0.0,
                                  -0.0,
                                  -1.0,
                                  -std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::quiet_NaN()};

template<typename T, typename GenerateFunc>
void test_generate(GenerateFunc generate_func)
{
    const rocrand_rng_type rng_type = rocrand_generate_poisson_tests::GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    const size_t size   = 12563;
    double       lambda = 100.0;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(generate_func(generator, data, size, lambda));

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

template<typename T>
void test_generate_host()
{
    const rocrand_rng_type rng_type = rocrand_generate_poisson_tests::GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator_host_blocking(&generator, rng_type));

    const size_t size   = 12563;
    double       lambda = 100.0;
    T*           data   = new T[size];

    ROCRAND_CHECK(rocrand_generate_poisson(generator, data, size, lambda));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));

    delete[] data;
}

template<typename T, typename GenerateFunc>
void test_out_of_range(GenerateFunc generate_func)
{
    const rocrand_rng_type rng_type = rocrand_generate_poisson_tests::GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    const size_t size = 256;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(T)));
    HIP_CHECK(hipMemset(data, 0xAB, size * sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());

    for(const double lambda : invalid_lambdas)
    {
        SCOPED_TRACE(testing::Message() << "with lambda = " << lambda);
        EXPECT_EQ(generate_func(generator, data, size, lambda), ROCRAND_STATUS_OUT_OF_RANGE);
    }

    // The output must be left untouched
    std::vector<T> h_data(size);
    HIP_CHECK(hipMemcpy(h_data.data(), data, size * sizeof(T), hipMemcpyDeviceToHost));
    std::vector<T> expected(size);
    std::memset(expected.data(), 0xAB, size * sizeof(T));
    EXPECT_EQ(h_data, expected);

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

void test_multiple_lambdas(const rocrand_rng_type rng_type, const hipStream_t stream)
{
    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));
    ROCRAND_CHECK(rocrand_set_stream(generator, stream));

    constexpr size_t       num_invocations = 20;
    constexpr size_t       size            = 125638;
    constexpr double       min_lambda      = 0.001;
    constexpr double       max_lambda      = 5000;
    constexpr unsigned int seed            = 654321;

    std::uniform_real_distribution<double> lambda_distribution(min_lambda, max_lambda);
    std::default_random_engine             rng(seed);
    std::vector<double>                    lambdas(num_invocations);
    for(auto& lambda : lambdas)
    {
        lambda = lambda_distribution(rng);
    }

    std::vector<unsigned int*> d_results(num_invocations);
    for(auto& d_ptr : d_results)
    {
        HIP_CHECK(hipMallocHelper(&d_ptr, sizeof(*d_ptr) * size));
    }

    std::vector<std::vector<unsigned int>> h_results(num_invocations);
    for(auto& h_vec : h_results)
    {
        h_vec.resize(size);
    }

    for(size_t i = 0; i < num_invocations; ++i)
    {
        ROCRAND_CHECK(rocrand_generate_poisson(generator, d_results[i], size, lambdas[i]));
    }

    HIP_CHECK(hipStreamSynchronize(stream));

    for(size_t i = 0; i < num_invocations; ++i)
    {
        const auto lambda = lambdas[i];
        auto&      values = h_results[i];
        HIP_CHECK(hipMemcpy(values.data(),
                            d_results[i],
                            sizeof(*d_results[i]) * size,
                            hipMemcpyDeviceToHost));

        const double mean     = get_mean(values);
        const double variance = get_variance(values, mean);

        EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 3e-2));
        EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 2e-2));
    }

    for(auto* d_ptr : d_results)
    {
        HIP_CHECK(hipFree(d_ptr));
    }

    // No output pointer
    ROCRAND_CHECK(rocrand_generate_poisson(generator, nullptr, size, lambdas[0]));

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

// Number of positions where the two equally sized buffers differ.
size_t count_mismatches(const std::vector<unsigned int>& a, const std::vector<unsigned int>& b)
{
    size_t mismatches = 0;
    for(size_t i = 0; i < a.size(); ++i)
    {
        mismatches += a[i] != b[i];
    }
    return mismatches;
}

// Generates the given lambdas back to back on one generator and copies each buffer to h_results.
// If sync_each is set, the stream is synchronized after every call.
void generate_lambda_sequence(const rocrand_rng_type                  rng_type,
                              const hipStream_t                       stream,
                              const std::vector<double>&              lambdas,
                              const size_t                            size,
                              const bool                              sync_each,
                              std::vector<std::vector<unsigned int>>& h_results)
{
    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));
    ROCRAND_CHECK(rocrand_set_stream(generator, stream));

    std::vector<unsigned int*> d_results(lambdas.size());
    for(auto& d_ptr : d_results)
    {
        HIP_CHECK(hipMallocHelper(&d_ptr, sizeof(*d_ptr) * size));
    }

    for(size_t i = 0; i < lambdas.size(); ++i)
    {
        ROCRAND_CHECK(rocrand_generate_poisson(generator, d_results[i], size, lambdas[i]));
        if(sync_each)
        {
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    h_results.assign(lambdas.size(), std::vector<unsigned int>(size));
    for(size_t i = 0; i < lambdas.size(); ++i)
    {
        HIP_CHECK(hipMemcpy(h_results[i].data(),
                            d_results[i],
                            sizeof(*d_results[i]) * size,
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_results[i]));
    }

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

// Switching lambda and back without synchronizing must give the same output as
// synchronizing after each call: the third call must not reuse a stale table.
void test_lambda_change_back_to_back(const rocrand_rng_type rng_type, const hipStream_t stream)
{
    constexpr size_t          size = 65536;
    const std::vector<double> lambdas{1000.0, 10.0, 1000.0};

    std::vector<std::vector<unsigned int>> expected;
    std::vector<std::vector<unsigned int>> actual;
    generate_lambda_sequence(rng_type, stream, lambdas, size, true, expected);
    generate_lambda_sequence(rng_type, stream, lambdas, size, false, actual);

    for(size_t i = 0; i < lambdas.size(); ++i)
    {
        SCOPED_TRACE(testing::Message() << "call " << i << ", lambda " << lambdas[i]);
        EXPECT_EQ(count_mismatches(actual[i], expected[i]), 0u);
    }
}

TEST_P(rocrand_generate_poisson_tests, generate_test)
{
    test_generate<unsigned int>(
        [](rocrand_generator gen, unsigned int* data, size_t size, double lambda)
        { return rocrand_generate_poisson(gen, data, size, lambda); });

    test_generate_host<unsigned int>();
}

TEST(rocrand_generate_poisson_tests, neg_test)
{
    const size_t size   = 256;
    double       lambda = 100.0;
    void*        data   = nullptr;

    rocrand_generator generator = nullptr;

    EXPECT_EQ(rocrand_generate_poisson(generator, static_cast<unsigned int*>(data), size, lambda),
              ROCRAND_STATUS_NOT_CREATED);
}

TEST(rocrand_generate_poisson_tests, create_poisson_distribution_out_of_range_test)
{
    EXPECT_EQ(rocrand_create_poisson_distribution(100.0, nullptr), ROCRAND_STATUS_OUT_OF_RANGE);

    for(const double lambda : invalid_lambdas)
    {
        SCOPED_TRACE(testing::Message() << "with lambda = " << lambda);
        rocrand_discrete_distribution distribution = nullptr;
        EXPECT_EQ(rocrand_create_poisson_distribution(lambda, &distribution),
                  ROCRAND_STATUS_OUT_OF_RANGE);
    }
}

TEST(rocrand_generate_poisson_tests, create_discrete_distribution_out_of_range_test)
{
    const double probabilities[] = {0.1, 0.2, 0.3, 0.4};

    rocrand_discrete_distribution distribution = nullptr;
    EXPECT_EQ(rocrand_create_discrete_distribution(nullptr, 4, 0, &distribution),
              ROCRAND_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(rocrand_create_discrete_distribution(probabilities, 0, 0, &distribution),
              ROCRAND_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(rocrand_create_discrete_distribution(probabilities, 4, 0, nullptr),
              ROCRAND_STATUS_OUT_OF_RANGE);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double max = std::numeric_limits<double>::max();

    const std::vector<std::vector<double>> invalid_probabilities = {
        {0.0, 0.0, 0.0, 0.0},
        {0.1, -0.2, 0.3, 0.4},
        {0.1, nan, 0.3, 0.4},
        {0.1, inf, 0.3, 0.4},
        {0.1, -inf, 0.3, 0.4},
        {max, max},
    };
    for(const std::vector<double>& p : invalid_probabilities)
    {
        SCOPED_TRACE(testing::PrintToString(p));
        EXPECT_EQ(rocrand_create_discrete_distribution(p.data(), p.size(), 0, &distribution),
                  ROCRAND_STATUS_OUT_OF_RANGE);
    }
}

TEST(rocrand_generate_poisson_tests, create_discrete_distribution_zero_probability_test)
{
    const double probabilities[] = {0.0, 0.5, 0.0, 0.5};

    rocrand_discrete_distribution distribution = nullptr;
    ASSERT_EQ(rocrand_create_discrete_distribution(probabilities, 4, 0, &distribution),
              ROCRAND_STATUS_SUCCESS);
    EXPECT_EQ(rocrand_destroy_discrete_distribution(distribution), ROCRAND_STATUS_SUCCESS);
}

TEST_P(rocrand_generate_poisson_tests, out_of_range_test)
{
    test_out_of_range<unsigned int>(
        [](rocrand_generator gen, unsigned int* data, size_t size, double lambda)
        { return rocrand_generate_poisson(gen, data, size, lambda); });
}

TEST_P(rocrand_generate_poisson_tests, multiple_lambdas_default_stream)
{
    test_multiple_lambdas(GetParam(), hipStreamDefault);
}

TEST_P(rocrand_generate_poisson_tests, multiple_lambdas_non_blocking_stream)
{
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    test_multiple_lambdas(GetParam(), stream);
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST_P(rocrand_generate_poisson_tests, lambda_change_back_to_back_null_stream)
{
    test_lambda_change_back_to_back(GetParam(), nullptr);
}

TEST_P(rocrand_generate_poisson_tests, lambda_change_back_to_back_blocking_stream)
{
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamDefault));
    test_lambda_change_back_to_back(GetParam(), stream);
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST_P(rocrand_generate_poisson_tests, lambda_change_back_to_back_non_blocking_stream)
{
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    test_lambda_change_back_to_back(GetParam(), stream);
    HIP_CHECK(hipStreamDestroy(stream));
}

INSTANTIATE_TEST_SUITE_P(rocrand_generate_poisson_tests,
                         rocrand_generate_poisson_tests,
                         ::testing::ValuesIn(rng_types),
                         rocrand_rng_type_test_name);
