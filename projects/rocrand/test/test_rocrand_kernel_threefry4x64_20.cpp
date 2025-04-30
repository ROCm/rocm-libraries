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

#include <gtest/gtest.h>
#include <stdio.h>

#include <cmath>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime.h>

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"

template<class GeneratorState>
__global__
void rocrand_init_kernel(GeneratorState*    states,
                         const size_t       states_size,
                         unsigned long long seed,
                         unsigned long long offset)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int subsequence = state_id;
    if(state_id < states_size)
    {
        GeneratorState state;
        rocrand_init(seed, subsequence, offset, &state);
        states[state_id] = state;
    }
}

template<class GeneratorState>
__global__
void rocrand_kernel(unsigned long long* output, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    GeneratorState     state;
    const unsigned int subsequence = state_id;
    rocrand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        output[index] = rocrand(&state);
        index += global_size;
    }
}

template<class GeneratorState>
__global__
void rocrand_uniform_kernel(float* output, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    GeneratorState     state;
    const unsigned int subsequence = state_id;
    rocrand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        output[index] = rocrand_uniform(&state);
        index += global_size;
    }
}

template<class GeneratorState>
__global__
void rocrand_uniform_double_kernel(double* output, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    GeneratorState     state;
    const unsigned int subsequence = state_id;
    rocrand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        output[index] = rocrand_uniform_double(&state);
        index += global_size;
    }
}

template<class GeneratorState>
__global__
void rocrand_normal_kernel(float* output, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    GeneratorState     state;
    const unsigned int subsequence = state_id;
    rocrand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        if(state_id % 2 == 0)
            output[index] = rocrand_normal2(&state).x;
        else
            output[index] = rocrand_normal(&state);
        index += global_size;
    }
}

template<class GeneratorState>
__global__
void rocrand_log_normal_kernel(float* output, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    GeneratorState     state;
    const unsigned int subsequence = state_id;
    rocrand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        if(state_id % 2 == 0)
            output[index] = rocrand_log_normal2(&state, 1.6f, 0.25f).x;
        else
            output[index] = rocrand_log_normal(&state, 1.6f, 0.25f);
        index += global_size;
    }
}

template<class GeneratorState>
__global__
void rocrand_poisson_kernel(unsigned int* output, const size_t size, double lambda)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    GeneratorState     state;
    const unsigned int subsequence = state_id;
    rocrand_init(23456, subsequence, 234ULL, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        output[index] = rocrand_poisson(&state, lambda);
        index += global_size;
    }
}

template<class GeneratorState>
__global__
void rocrand_discrete_kernel(unsigned int*                 output,
                             const size_t                  size,
                             rocrand_discrete_distribution discrete_distribution)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    GeneratorState     state;
    const unsigned int subsequence = state_id;
    rocrand_init(23456, subsequence, 234ULL, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        output[index] = rocrand_discrete(&state, discrete_distribution);
        index += global_size;
    }
}

TEST(rocrand_kernel_threefry4x64_20, rocrand_state_threefry_type)
{
    typedef rocrand_state_threefry4x64_20 state_type;
    EXPECT_EQ(alignof(state_type), alignof(ulonglong4));
    EXPECT_EQ(sizeof(state_type[32]), 32 * sizeof(state_type));
    // TODO: Enable once ulonglong4 trivially copyable.
    //EXPECT_TRUE(std::is_trivially_copyable<state_type>::value);
    EXPECT_TRUE(std::is_trivially_destructible<state_type>::value);
}

TEST(rocrand_kernel_threefry4x64_20, rocrand)
{
    typedef rocrand_state_threefry4x64_20 state_type;

    const size_t        output_size = 8192;
    unsigned long long* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(unsigned long long)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_kernel<state_type>),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       output,
                       output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned long long> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned long long),
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
}

TEST(rocrand_kernel_threefry4x64_20, rocrand_uniform)
{
    typedef rocrand_state_threefry4x64_20 state_type;

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_uniform_kernel<state_type>),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       output,
                       output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_threefry4x64_20, rocrand_uniform_double)
{
    typedef rocrand_state_threefry4x64_20 state_type;

    const size_t output_size = 8192;
    double*      output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(double)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_uniform_double_kernel<state_type>),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       output,
                       output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += v;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_threefry4x64_20, rocrand_uniform_range)
{
    typedef rocrand_state_threefry4x64_20 state_type;

    const size_t output_size = 1 << 26;
    float*       output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_uniform_kernel<state_type>),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       output,
                       output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    for(auto v : output_host)
    {
        ASSERT_GT(v, 0.0f);
        ASSERT_LE(v, 1.0f);
    }
}

TEST(rocrand_kernel_threefry4x64_20, rocrand_uniform_double_range)
{
    typedef rocrand_state_threefry4x64_20 state_type;

    const size_t output_size = 1 << 26;
    double*      output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(double)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_uniform_double_kernel<state_type>),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       output,
                       output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    for(auto v : output_host)
    {
        ASSERT_GT(v, 0.0);
        ASSERT_LE(v, 1.0);
    }
}

TEST(rocrand_kernel_threefry4x64_20, rocrand_normal)
{
    typedef rocrand_state_threefry4x64_20 state_type;

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_normal_kernel<state_type>),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       output,
                       output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
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
}

TEST(rocrand_kernel_threefry4x64_20, rocrand_log_normal)
{
    typedef rocrand_state_threefry4x64_20 state_type;

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_log_normal_kernel<state_type>),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       output,
                       output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
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
    double logstd  = std::sqrt(std::log(1.0f + stddev / (mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);
}

class rocrand_kernel_threefry4x64_20_poisson : public ::testing::TestWithParam<double>
{};

TEST_P(rocrand_kernel_threefry4x64_20_poisson, rocrand_poisson)
{
    typedef rocrand_state_threefry4x64_20 state_type;

    const double lambda = GetParam();

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_poisson_kernel<state_type>),
                       dim3(4),
                       dim3(64),
                       0,
                       0,
                       output,
                       output_size,
                       lambda);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
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
}

TEST_P(rocrand_kernel_threefry4x64_20_poisson, rocrand_discrete)
{
    typedef rocrand_state_threefry4x64_20 state_type;

    const double lambda = GetParam();

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    rocrand_discrete_distribution discrete_distribution;
    ROCRAND_CHECK(rocrand_create_poisson_distribution(lambda, &discrete_distribution));

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_discrete_kernel<state_type>),
                       dim3(4),
                       dim3(64),
                       0,
                       0,
                       output,
                       output_size,
                       discrete_distribution);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));

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
}

const double lambdas[] = {1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0};

INSTANTIATE_TEST_SUITE_P(rocrand_kernel_threefry4x64_20_poisson,
                         rocrand_kernel_threefry4x64_20_poisson,
                         ::testing::ValuesIn(lambdas));
