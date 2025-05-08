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

#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime.h>

#include <rocrand/rocrand_kernel.h>
#include <rocrand/rocrand_mtgp32_11213.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"

template<class GeneratorState>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void rocrand_kernel(GeneratorState* states, unsigned int* output, const size_t size)
{
    const unsigned int state_id = blockIdx.x;
    unsigned int       index    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int       stride   = gridDim.x * blockDim.x;

    __shared__ GeneratorState state;
    rocrand_mtgp32_block_copy(&states[state_id], &state);

    const size_t r               = size % blockDim.x;
    const size_t size_rounded_up = r == 0 ? size : size + (blockDim.x - r);
    while(index < size_rounded_up)
    {
        auto value = rocrand(&state);
        if(index < size)
            output[index] = value;
        // Next position
        index += stride;
    }

    // Save engine with its state
    rocrand_mtgp32_block_copy(&state, &states[state_id]);
}

template<class GeneratorState>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void rocrand_uniform_kernel(GeneratorState* states, float* output, const size_t size)
{
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int       stride    = gridDim.x * blockDim.x;

    __shared__ GeneratorState state;
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r               = size % blockDim.x;
    const size_t size_rounded_up = r == 0 ? size : size + (blockDim.x - r);
    while(index < size_rounded_up)
    {
        auto value = rocrand_uniform(&state);
        if(index < size)
            output[index] = value;
        // Next position
        index += stride;
    }

    // Save engine with its state
    if(thread_id == 0)
        states[state_id] = state;
}

template<class GeneratorState>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void rocrand_normal_kernel(GeneratorState* states, float* output, const size_t size)
{
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int       stride    = gridDim.x * blockDim.x;

    __shared__ GeneratorState state;
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r               = size % blockDim.x;
    const size_t size_rounded_up = r == 0 ? size : size + (blockDim.x - r);
    while(index < size_rounded_up)
    {
        if(index < size)
        {
            if(state_id % 2 == 0)
            {
                output[index] = rocrand_normal_double2(&state).x;
            }
            else
            {
                output[index] = rocrand_normal_double(&state);
            }
        }
        // Next position
        index += stride;
    }

    // Save engine with its state
    if(thread_id == 0)
        states[state_id] = state;
}

template<class GeneratorState>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void rocrand_normal_double_kernel(GeneratorState* states, double* output, const size_t size)
{
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int       stride    = gridDim.x * blockDim.x;

    __shared__ GeneratorState state;
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r               = size % blockDim.x;
    const size_t size_rounded_up = r == 0 ? size : size + (blockDim.x - r);
    while(index < size_rounded_up)
    {
        if(index < size)
        {
            if(state_id % 2 == 0)
            {
                output[index] = rocrand_normal_double2(&state).x;
            }
            else
            {
                output[index] = rocrand_normal_double(&state);
            }
        }
        // Next position
        index += stride;
    }

    // Save engine with its state
    if(thread_id == 0)
        states[state_id] = state;
}

template<class GeneratorState>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void rocrand_log_normal_kernel(GeneratorState* states, float* output, const size_t size)
{
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int       stride    = gridDim.x * blockDim.x;

    __shared__ GeneratorState state;
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r               = size % blockDim.x;
    const size_t size_rounded_up = r == 0 ? size : size + (blockDim.x - r);
    while(index < size_rounded_up)
    {
        if(index < size)
        {
            if(state_id % 2 == 0)
            {
                output[index] = rocrand_log_normal2(&state, 1.6f, 0.25f).x;
            }
            else
            {
                output[index] = rocrand_log_normal(&state, 1.6f, 0.25f);
            }
        }
        // Next position
        index += stride;
    }

    // Save engine with its state
    if(thread_id == 0)
        states[state_id] = state;
}

template<class GeneratorState>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void rocrand_log_normal_double_kernel(GeneratorState* states, double* output, const size_t size)
{
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int       stride    = gridDim.x * blockDim.x;

    __shared__ GeneratorState state;
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r               = size % blockDim.x;
    const size_t size_rounded_up = r == 0 ? size : size + (blockDim.x - r);
    while(index < size_rounded_up)
    {
        if(index < size)
        {
            if(state_id % 2 == 0)
            {
                output[index] = rocrand_log_normal_double2(&state, 1.6f, 0.25f).x;
            }
            else
            {
                output[index] = rocrand_log_normal_double(&state, 1.6f, 0.25f);
            }
        }
        // Next position
        index += stride;
    }

    // Save engine with its state
    if(thread_id == 0)
        states[state_id] = state;
}

template<class GeneratorState>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void rocrand_poisson_kernel(GeneratorState* states,
                            unsigned int*   output,
                            const size_t    size,
                            double          lambda)
{
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int       stride    = gridDim.x * blockDim.x;

    __shared__ GeneratorState state;
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r               = size % blockDim.x;
    const size_t size_rounded_up = r == 0 ? size : size + (blockDim.x - r);
    while(index < size_rounded_up)
    {
        auto value = rocrand_poisson(&state, lambda);
        if(index < size)
            output[index] = value;
        // Next position
        index += stride;
    }

    // Save engine with its state
    if(thread_id == 0)
        states[state_id] = state;
}

template<class GeneratorState>
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void rocrand_discrete_kernel(GeneratorState*               states,
                             unsigned int*                 output,
                             const size_t                  size,
                             rocrand_discrete_distribution discrete_distribution)
{
    const unsigned int state_id  = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    unsigned int       index     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride          = gridDim.x * blockDim.x;

    __shared__ GeneratorState state;
    if(thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r               = size % blockDim.x;
    const size_t size_rounded_up = r == 0 ? size : size + (blockDim.x - r);
    while(index < size_rounded_up)
    {
        auto value = rocrand_discrete(&state, discrete_distribution);
        if(index < size)
            output[index] = value;
        // Next position
        index += stride;
    }

    // Save engine with its state
    if(thread_id == 0)
        states[state_id] = state;
}

TEST(rocrand_kernel_mtgp32, rocrand_state_mtgp32_type)
{
    typedef rocrand_state_mtgp32 state_type;
    EXPECT_EQ(sizeof(state_type), 1078 * sizeof(unsigned int));
    EXPECT_EQ(sizeof(state_type[32]), 32 * sizeof(state_type));
    EXPECT_TRUE(std::is_trivially_copyable<state_type>::value);
    EXPECT_TRUE(std::is_trivially_destructible<state_type>::value);
}

TEST(rocrand_kernel_mtgp32, rocrand)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type* states;
    HIP_CHECK(hipMallocHelper(&states, sizeof(state_type) * 8));

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());
    rocrand_kernel<state_type><<<dim3(8), dim3(256), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / UINT_MAX;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_mtgp32, rocrand_uniform)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type* states;
    HIP_CHECK(hipMallocHelper(&states, sizeof(state_type) * 8));

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());
    rocrand_uniform_kernel<state_type><<<dim3(8), dim3(256), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_mtgp32, rocrand_normal)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type* states;
    HIP_CHECK(hipMallocHelper(&states, sizeof(state_type) * 8));

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());
    rocrand_normal_kernel<state_type><<<dim3(8), dim3(256), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = std::accumulate(output_host.begin(), output_host.end(), 0.0) / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);
}

TEST(rocrand_kernel_mtgp32, rocrand_normal_double)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type* states;
    HIP_CHECK(hipMallocHelper(&states, sizeof(state_type) * 8));

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    double*      output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(double)));
    HIP_CHECK(hipDeviceSynchronize());
    rocrand_normal_double_kernel<state_type>
        <<<dim3(8), dim3(256), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = std::accumulate(output_host.begin(), output_host.end(), 0.0) / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);
}

TEST(rocrand_kernel_mtgp32, rocrand_log_normal)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type* states;
    HIP_CHECK(hipMallocHelper(&states, sizeof(state_type) * 8));

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    float*       output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());
    rocrand_log_normal_kernel<state_type>
        <<<dim3(8), dim3(256), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = std::accumulate(output_host.begin(), output_host.end(), 0.0) / output_size;

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

TEST(rocrand_kernel_mtgp32, rocrand_log_normal_double)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type* states;
    HIP_CHECK(hipMallocHelper(&states, sizeof(state_type) * 8));

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    double*      output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(double)));
    HIP_CHECK(hipDeviceSynchronize());
    rocrand_log_normal_double_kernel<state_type>
        <<<dim3(8), dim3(256), 0, 0>>>(states, output, output_size);
    HIP_CHECK(hipGetLastError());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(output_host.data(), output, output_size * sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = std::accumulate(output_host.begin(), output_host.end(), 0.0) / output_size;

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

class rocrand_kernel_mtgp32_poisson : public ::testing::TestWithParam<double>
{};

TEST_P(rocrand_kernel_mtgp32_poisson, rocrand_poisson)
{
    typedef rocrand_state_mtgp32 state_type;

    const double lambda = GetParam();

    state_type* states;
    HIP_CHECK(hipMallocHelper(&states, sizeof(state_type) * 8));

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());
    rocrand_poisson_kernel<state_type>
        <<<dim3(8), dim3(256), 0, 0>>>(states, output, output_size, lambda);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));

    double mean = std::accumulate(output_host.begin(), output_host.end(), 0.0) / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

TEST_P(rocrand_kernel_mtgp32_poisson, rocrand_discrete)
{
    typedef rocrand_state_mtgp32 state_type;

    const double lambda = GetParam();

    state_type* states;
    HIP_CHECK(hipMallocHelper(&states, sizeof(state_type) * 8));

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    rocrand_discrete_distribution discrete_distribution;
    ROCRAND_CHECK(rocrand_create_poisson_distribution(lambda, &discrete_distribution));

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());
    rocrand_discrete_kernel<state_type>
        <<<dim3(8), dim3(256), 0, 0>>>(states, output, output_size, discrete_distribution);
    HIP_CHECK(hipGetLastError());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(output_host.data(),
                        output,
                        output_size * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(states));
    ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));

    double mean = std::accumulate(output_host.begin(), output_host.end(), 0.0) / output_size;

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

INSTANTIATE_TEST_SUITE_P(rocrand_kernel_mtgp32_poisson,
                         rocrand_kernel_mtgp32_poisson,
                         ::testing::ValuesIn(lambdas));
