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

#include <rocrand/rocrand_discrete.h>
#include <rocrand/rocrand_log_normal.h>
#include <rocrand/rocrand_normal.h>
#include <rocrand/rocrand_poisson.h>
#include <rocrand/rocrand_uniform.h>

#include <rocrand/rocrand_common.h>
#include <rocrand/rocrand_scrambled_sobol64.h>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <vector>

struct rocrand_f
{
    __device__ __forceinline__
    unsigned long long int
        operator()(rocrand_state_scrambled_sobol64* state_ptr)
    {
        return rocrand(state_ptr);
    }
};

struct rocrand_uniform_f
{
    __device__ __forceinline__
    float
        operator()(rocrand_state_scrambled_sobol64* state_ptr)
    {
        return rocrand_uniform(state_ptr);
    }
};

struct rocrand_uniform_double_f
{
    __device__ __forceinline__
    double
        operator()(rocrand_state_scrambled_sobol64* state_ptr)
    {
        return rocrand_uniform_double(state_ptr);
    }
};

struct rocrand_normal_f
{
    __device__ __forceinline__
    float
        operator()(rocrand_state_scrambled_sobol64* state_ptr)
    {
        return rocrand_normal(state_ptr);
    }
};

struct rocrand_normal_double_f
{
    __device__ __forceinline__
    double
        operator()(rocrand_state_scrambled_sobol64* state_ptr)
    {
        return rocrand_normal_double(state_ptr);
    }
};

struct rocrand_log_normal_f
{
    __device__ __forceinline__
    float
        operator()(rocrand_state_scrambled_sobol64* state_ptr, float mean, float std)
    {
        return rocrand_log_normal(state_ptr, mean, std);
    }
};

struct rocrand_log_normal_double_f
{
    __device__ __forceinline__
    double
        operator()(rocrand_state_scrambled_sobol64* state_ptr, double mean, double std)
    {
        return rocrand_log_normal_double(state_ptr, mean, std);
    }
};

struct rocrand_poisson_f
{
    __device__ __forceinline__
    unsigned int
        operator()(rocrand_state_scrambled_sobol64* state_ptr, double lambda)
    {
        return rocrand_poisson(state_ptr, lambda);
    }
};

struct rocrand_discrete_f
{
    __device__ __forceinline__
    unsigned int
        operator()(rocrand_state_scrambled_sobol64* state_ptr,
                   rocrand_discrete_distribution    discrete_distribution)
    {
        return rocrand_discrete(state_ptr, discrete_distribution);
    }
};

template<class Distribution, typename OutputType, typename... Args>
__global__
void rocrand_kernel(OutputType*                   output,
                    const unsigned long long int* vectors,
                    const unsigned long long int* scramble_constants,
                    const size_t                  size_per_dimension,
                    Args... args)
{
    const unsigned int state_id  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int dimension = blockIdx.y;

    // `size_per_dimension` elements are generated for each dimension
    const unsigned int threads_per_dim = gridDim.x * blockDim.x;
    const unsigned int items_per_thread
        = (size_per_dimension + threads_per_dim - 1) / threads_per_dim;
    // offset of results of thread inside block
    const unsigned int block_offset = state_id * items_per_thread;
    // offset of results of the block inside grid
    const unsigned int global_offset = dimension * size_per_dimension;

    rocrand_state_scrambled_sobol64 state;
    rocrand_init(&(vectors[64 * dimension]),
                 scramble_constants[dimension],
                 1234 + block_offset,
                 &state);

    // `size_per_dimension` is not necessarily divisible by the amount of threads in a dimension
    for(unsigned int i = 0; block_offset + i < size_per_dimension; i++)
    {
        output[global_offset + block_offset + i] = Distribution{}(&state, args...);
    }
}

void load_scrambled_sobol64_constants_to_gpu(const unsigned int       dimensions,
                                             unsigned long long int** direction_vectors,
                                             unsigned long long int** scramble_constants)
{
    const unsigned long long* h_directions;
    const unsigned long long* h_constants;

    rocrand_get_direction_vectors64(&h_directions, ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);
    rocrand_get_scramble_constants64(&h_constants);

    HIP_CHECK(hipMallocHelper(direction_vectors, sizeof(unsigned long long int) * dimensions * 64));
    HIP_CHECK(hipMemcpy(*direction_vectors,
                        h_directions,
                        sizeof(unsigned long long int) * dimensions * 64,
                        hipMemcpyHostToDevice));

    HIP_CHECK(hipMallocHelper(scramble_constants, sizeof(unsigned long long int) * dimensions));
    HIP_CHECK(hipMemcpy(*scramble_constants,
                        h_constants,
                        sizeof(unsigned long long int) * dimensions,
                        hipMemcpyHostToDevice));
}

template<typename ResultType, class Distribution, typename... Args>
void call_rocrand_kernel(std::vector<ResultType>& output_host,
                         const unsigned int       dimensions,
                         const size_t             size_per_dimension,
                         Args... args)
{
    // output_size has to be a multiple of the dimensions for sobol
    const size_t output_size = dimensions * size_per_dimension;
    output_host.resize(output_size);

    ResultType* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(ResultType)));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long int* m_vector;
    unsigned long long int* m_scramble_constants;
    load_scrambled_sobol64_constants_to_gpu(dimensions, &m_vector, &m_scramble_constants);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_kernel<Distribution>),
                       dim3(8, dimensions),
                       dim3(32),
                       0,
                       0,
                       output,
                       m_vector,
                       m_scramble_constants,
                       size_per_dimension,
                       args...);
    HIP_CHECK(hipGetLastError());

    //std::vector<ResultType> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(ResultType),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(m_vector));
    HIP_CHECK(hipFree(m_scramble_constants));
}

TEST(rocrand_kernel_scrambled_sobol64, rocrand_state_scrambled_sobol64_type)
{
    typedef rocrand_state_scrambled_sobol64 state_type;
    EXPECT_EQ(sizeof(state_type), 67 * sizeof(unsigned long long int));
    EXPECT_EQ(sizeof(state_type[64]), 64 * sizeof(state_type));
    EXPECT_TRUE(std::is_trivially_copyable<state_type>::value);
    EXPECT_TRUE(std::is_trivially_destructible<state_type>::value);
}

TEST(rocrand_kernel_scrambled_sobol64, rocrand)
{
    using ResultType   = unsigned long long int;
    using Distribution = rocrand_f;

    // amount of generated numbers has to be a multiple of the dimensions for sobol, so size is given per dimension
    constexpr size_t       size_per_dimension = 8192;
    constexpr unsigned int dimensions         = 8;

    std::vector<ResultType> output_host;
    call_rocrand_kernel<ResultType, Distribution>(output_host, dimensions, size_per_dimension);

    double mean = 0;
    for(ResultType v : output_host)
    {
        // conversion from ULLONG_MAX to double loses precision -> implicit conversion generates a warning
        mean += static_cast<double>(v) / static_cast<double>(ULLONG_MAX);
    }
    mean = mean / output_host.size();
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_scrambled_sobol64, rocrand_uniform)
{
    using ResultType   = float;
    using Distribution = rocrand_uniform_f;

    // amount of generated numbers has to be a multiple of the dimensions for sobol, so size is given per dimension
    constexpr size_t       size_per_dimension = 8192;
    constexpr unsigned int dimensions         = 8;

    std::vector<ResultType> output_host;
    call_rocrand_kernel<ResultType, Distribution>(output_host, dimensions, size_per_dimension);

    double mean = 0;
    for(ResultType v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_host.size();
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_scrambled_sobol64, rocrand_uniform_double)
{
    using ResultType   = double;
    using Distribution = rocrand_uniform_double_f;

    // amount of generated numbers has to be a multiple of the dimensions for sobol, so size is given per dimension
    constexpr size_t       size_per_dimension = 8192;
    constexpr unsigned int dimensions         = 8;

    std::vector<ResultType> output_host;
    call_rocrand_kernel<ResultType, Distribution>(output_host, dimensions, size_per_dimension);

    double mean = 0;
    for(ResultType v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_host.size();
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_scrambled_sobol64, rocrand_normal)
{
    using ResultType   = float;
    using Distribution = rocrand_normal_f;

    // amount of generated numbers has to be a multiple of the dimensions for sobol, so size is given per dimension
    constexpr size_t       size_per_dimension = 8192;
    constexpr unsigned int dimensions         = 8;

    std::vector<ResultType> output_host;
    call_rocrand_kernel<ResultType, Distribution>(output_host, dimensions, size_per_dimension);

    double mean = 0;
    for(ResultType v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_host.size();
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(ResultType v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_host.size();
    EXPECT_NEAR(stddev, 1.0, 0.2);
}

TEST(rocrand_kernel_scrambled_sobol64, rocrand_normal_double)
{
    using ResultType   = double;
    using Distribution = rocrand_normal_double_f;

    // amount of generated numbers has to be a multiple of the dimensions for sobol, so size is given per dimension
    constexpr size_t       size_per_dimension = 8192;
    constexpr unsigned int dimensions         = 8;

    std::vector<ResultType> output_host;
    call_rocrand_kernel<ResultType, Distribution>(output_host, dimensions, size_per_dimension);

    double mean = 0;
    for(ResultType v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_host.size();
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(ResultType v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_host.size();
    EXPECT_NEAR(stddev, 1.0, 0.2);
}

TEST(rocrand_kernel_scrambled_sobol64, rocrand_log_normal)
{
    using ResultType   = float;
    using Distribution = rocrand_log_normal_f;

    // amount of generated numbers has to be a multiple of the dimensions for sobol, so size is given per dimension
    constexpr size_t       size_per_dimension = 8192;
    constexpr unsigned int dimensions         = 8;

    constexpr ResultType ExpectedMean = 1.6f;
    constexpr ResultType ExpectedStd  = 0.25f;

    std::vector<ResultType> output_host;
    call_rocrand_kernel<ResultType, Distribution>(output_host,
                                                  dimensions,
                                                  size_per_dimension,
                                                  ExpectedMean,
                                                  ExpectedStd);

    double mean = 0;
    for(ResultType v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_host.size();

    double stddev = 0;
    for(ResultType v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_host.size());

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd  = std::sqrt(std::log(1.0f + stddev / (mean * mean)));

    EXPECT_NEAR(ExpectedMean, logmean, ExpectedMean * 0.2);
    EXPECT_NEAR(ExpectedStd, logstd, ExpectedStd * 0.2);
}

TEST(rocrand_kernel_scrambled_sobol64, rocrand_log_normal_double)
{
    using ResultType   = double;
    using Distribution = rocrand_log_normal_double_f;

    // amount of generated numbers has to be a multiple of the dimensions for sobol, so size is given per dimension
    constexpr size_t       size_per_dimension = 8192;
    constexpr unsigned int dimensions         = 8;

    constexpr ResultType ExpectedMean = 1.6f;
    constexpr ResultType ExpectedStd  = 0.25f;

    std::vector<ResultType> output_host;
    call_rocrand_kernel<ResultType, Distribution>(output_host,
                                                  dimensions,
                                                  size_per_dimension,
                                                  ExpectedMean,
                                                  ExpectedStd);

    double mean = 0;
    for(ResultType v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_host.size();

    double stddev = 0;
    for(ResultType v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_host.size());

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd  = std::sqrt(std::log(1.0f + stddev / (mean * mean)));

    EXPECT_NEAR(ExpectedMean, logmean, ExpectedMean * 0.2);
    EXPECT_NEAR(ExpectedStd, logstd, ExpectedStd * 0.2);
}

class rocrand_kernel_scrambled_sobol64_poisson : public ::testing::TestWithParam<double>
{};

TEST_P(rocrand_kernel_scrambled_sobol64_poisson, rocrand_poisson)
{
    const double lambda = GetParam();

    using ResultType = unsigned int;

    constexpr size_t       size_per_dimension = 8192;
    constexpr unsigned int dimensions         = 8;
    // output_size has to be a multiple of the dimensions for sobol
    constexpr size_t output_size = dimensions * size_per_dimension;

    ResultType* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(ResultType)));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long int* m_vector;
    unsigned long long int* m_scramble_constants;
    load_scrambled_sobol64_constants_to_gpu(dimensions, &m_vector, &m_scramble_constants);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_kernel<rocrand_poisson_f>),
                       dim3(8, dimensions),
                       dim3(32),
                       0,
                       0,
                       output,
                       m_vector,
                       m_scramble_constants,
                       size_per_dimension,
                       lambda);
    HIP_CHECK(hipGetLastError());

    std::vector<ResultType> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(ResultType),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(m_vector));
    HIP_CHECK(hipFree(m_scramble_constants));

    double mean = 0;
    for(ResultType v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(ResultType v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

TEST_P(rocrand_kernel_scrambled_sobol64_poisson, rocrand_discrete)
{
    const double lambda = GetParam();

    using ResultType = unsigned int;

    constexpr size_t       size_per_dimension = 8192;
    constexpr unsigned int dimensions         = 8;
    // output_size has to be a multiple of the dimensions for sobol
    constexpr size_t output_size = dimensions * size_per_dimension;

    ResultType* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(ResultType)));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long int* m_vector;
    unsigned long long int* m_scramble_constants;
    load_scrambled_sobol64_constants_to_gpu(dimensions, &m_vector, &m_scramble_constants);

    rocrand_discrete_distribution discrete_distribution;
    ROCRAND_CHECK(rocrand_create_poisson_distribution(lambda, &discrete_distribution));

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_kernel<rocrand_discrete_f>),
                       dim3(8, dimensions),
                       dim3(32),
                       0,
                       0,
                       output,
                       m_vector,
                       m_scramble_constants,
                       size_per_dimension,
                       discrete_distribution);
    HIP_CHECK(hipGetLastError());

    std::vector<ResultType> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(ResultType),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(m_vector));
    HIP_CHECK(hipFree(m_scramble_constants));
    ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));

    double mean = 0;
    for(ResultType v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(ResultType v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

const double lambdas[] = {1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0};

INSTANTIATE_TEST_SUITE_P(rocrand_kernel_scrambled_sobol64_poisson,
                         rocrand_kernel_scrambled_sobol64_poisson,
                         ::testing::ValuesIn(lambdas));
