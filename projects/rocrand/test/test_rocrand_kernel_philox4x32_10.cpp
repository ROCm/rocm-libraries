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
void rocrand_kernel(unsigned int* output, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    GeneratorState     state;
    const unsigned int subsequence = state_id;
    rocrand_init(0, subsequence, 0, &state);

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
    rocrand_init(0, subsequence, 0, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        if(state_id % 4 == 0)
            output[index] = rocrand_uniform4(&state).x;
        else if(state_id % 2 == 0)
            output[index] = rocrand_uniform2(&state).x;
        else
            output[index] = rocrand_uniform(&state);
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
    rocrand_init(0, subsequence, 0, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        if(state_id % 4 == 0)
            output[index] = rocrand_normal4(&state).x;
        else if(state_id % 2 == 0)
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
    rocrand_init(0, subsequence, 0, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        if(state_id % 4 == 0)
            output[index] = rocrand_log_normal4(&state, 1.6f, 0.25f).x;
        else if(state_id % 2 == 0)
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
    rocrand_init(456, subsequence, 234ULL, &state);

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
    rocrand_init(456, subsequence, 234ULL, &state);

    unsigned int index = state_id;
    while(index < size)
    {
        output[index] = rocrand_discrete(&state, discrete_distribution);
        index += global_size;
    }
}

__global__
void rocrand_philox4x32_10_kernel(const uint4* counters,
                                  const uint2* keys,
                                  uint4*       output,
                                  const size_t size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size)
    {
        output[index] = rocrand_philox4x32_10(counters[index], keys[index]);
    }
}

__global__
void rocrand_philox4x32_10_state_kernel(uint4*             block_output,
                                        uint4*             state_output,
                                        unsigned long long seed,
                                        const size_t       size)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size)
    {
        const unsigned long long subsequence = index * 0x100000001ULL;
        const unsigned long long block       = (UINT_MAX - 3ULL) + index * 0x10001ULL;

        rocrand_state_philox4x32_10 state;
        rocrand_init(seed, subsequence, 4 * block, &state);

        const uint4 counter = {static_cast<unsigned int>(block),
                               static_cast<unsigned int>(block >> 32),
                               static_cast<unsigned int>(subsequence),
                               static_cast<unsigned int>(subsequence >> 32)};
        const uint2 key = {static_cast<unsigned int>(seed), static_cast<unsigned int>(seed >> 32)};

        state_output[index] = rocrand4(&state);
        block_output[index] = rocrand_philox4x32_10(counter, key);
    }
}

TEST(rocrand_kernel_philox4x32_10, rocrand_state_philox4x32_10_type)
{
    typedef rocrand_state_philox4x32_10 state_type;
    EXPECT_EQ(sizeof(state_type), 16 * sizeof(float));
    EXPECT_EQ(sizeof(state_type[32]), 32 * sizeof(state_type));
    EXPECT_TRUE(std::is_trivially_destructible<uint2>::value);
    EXPECT_TRUE(std::is_trivially_destructible<uint4>::value);
    EXPECT_TRUE(std::is_trivially_destructible<state_type>::value);
}

TEST(rocrand_kernel_philox4x32_10, DISABLED_rocrand_state_philox4x32_10_copyable)
{
    typedef rocrand_state_philox4x32_10 state_type;
    EXPECT_TRUE(std::is_trivially_copyable<uint2>::value);
    EXPECT_TRUE(std::is_trivially_copyable<uint4>::value);
    EXPECT_TRUE(std::is_trivially_copyable<state_type>::value);
}

TEST(rocrand_kernel_philox4x32_10, rocrand_init)
{
    // Just get access to internal state
    class rocrand_state_philox4x32_10_test : public rocrand_state_philox4x32_10
    {
        typedef rocrand_state_philox4x32_10::philox4x32_10_state internal_state_type;

    public:
        rocrand_state_philox4x32_10_test() {}

        internal_state_type internal_state() const
        {
            return m_state;
        }
    };

    typedef rocrand_state_philox4x32_10      state_type;
    typedef rocrand_state_philox4x32_10_test state_type_test;

    unsigned long long seed   = 0xdeadbeefbeefdeadULL;
    unsigned long long offset = 4 * ((UINT_MAX * 17ULL) + 17);

    const size_t states_size = 256;
    state_type*  states;
    HIP_CHECK(hipMallocHelper(&states, states_size * sizeof(state_type)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_init_kernel),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       states,
                       states_size,
                       seed,
                       offset);
    HIP_CHECK(hipGetLastError());

    std::vector<state_type_test> states_host(states_size);
    HIP_CHECK(hipMemcpy(states_host.data(),
                        states,
                        states_size * sizeof(state_type),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(states));

    unsigned int subsequence = 0;
    for(auto& state : states_host)
    {
        auto s = state.internal_state();
        EXPECT_EQ(s.key.x, 0xbeefdeadU);
        EXPECT_EQ(s.key.y, 0xdeadbeefU);

        EXPECT_EQ(s.counter.x, 0U);
        EXPECT_EQ(s.counter.y, 17U);
        EXPECT_EQ(s.counter.z, subsequence);
        EXPECT_EQ(s.counter.w, 0U);

        EXPECT_TRUE(s.result.x != 0U || s.result.y != 0U || s.result.z != 0U || s.result.w);

        EXPECT_EQ(s.substate, 0U);

        subsequence++;
    }
}

TEST(rocrand_kernel_philox4x32_10, rocrand)
{
    typedef rocrand_state_philox4x32_10 state_type;

    const size_t  output_size = 8192;
    unsigned int* output;
    HIP_CHECK(hipMallocHelper(&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_kernel<state_type>),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       output,
                       output_size);
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
        mean += static_cast<double>(v) / UINT_MAX;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_philox4x32_10, rocrand_philox4x32_10_known_answers)
{
    // Known-answer vectors from Random123 (tests/kat_vectors)
    const std::vector<uint4> counters = {
        {0x00000000U, 0x00000000U, 0x00000000U, 0x00000000U},
        {0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU},
        {0x243f6a88U, 0x85a308d3U, 0x13198a2eU, 0x03707344U}
    };
    const std::vector<uint2> keys = {
        {0x00000000U, 0x00000000U},
        {0xffffffffU, 0xffffffffU},
        {0xa4093822U, 0x299f31d0U}
    };
    const std::vector<uint4> expected = {
        {0x6627e8d5U, 0xe169c58dU, 0xbc57ac4cU, 0x9b00dbd8U},
        {0x408f276dU, 0x41c83b0eU, 0xa20bc7c6U, 0x6d5451fdU},
        {0xd16cfe09U, 0x94fdccebU, 0x5001e420U, 0x24126ea1U}
    };
    const size_t size = counters.size();

    uint4* counters_device;
    uint2* keys_device;
    uint4* output;
    HIP_CHECK(hipMallocHelper(&counters_device, size * sizeof(uint4)));
    HIP_CHECK(hipMallocHelper(&keys_device, size * sizeof(uint2)));
    HIP_CHECK(hipMallocHelper(&output, size * sizeof(uint4)));
    HIP_CHECK(
        hipMemcpy(counters_device, counters.data(), size * sizeof(uint4), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(keys_device, keys.data(), size * sizeof(uint2), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_philox4x32_10_kernel),
                       dim3(1),
                       dim3(32),
                       0,
                       0,
                       counters_device,
                       keys_device,
                       output,
                       size);
    HIP_CHECK(hipGetLastError());

    std::vector<uint4> output_host(size);
    HIP_CHECK(hipMemcpy(output_host.data(), output, size * sizeof(uint4), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(counters_device));
    HIP_CHECK(hipFree(keys_device));
    HIP_CHECK(hipFree(output));

    for(size_t i = 0; i < size; i++)
    {
        const uint4 host = rocrand_philox4x32_10(counters[i], keys[i]);
        for(const uint4& v : {output_host[i], host})
        {
            EXPECT_EQ(v.x, expected[i].x);
            EXPECT_EQ(v.y, expected[i].y);
            EXPECT_EQ(v.z, expected[i].z);
            EXPECT_EQ(v.w, expected[i].w);
        }
    }
}

TEST(rocrand_kernel_philox4x32_10, rocrand_philox4x32_10_matches_state)
{
    const unsigned long long seed = 0xdeadbeefbeefdeadULL;
    const size_t             size = 256;

    uint4* block_output;
    uint4* state_output;
    HIP_CHECK(hipMallocHelper(&block_output, size * sizeof(uint4)));
    HIP_CHECK(hipMallocHelper(&state_output, size * sizeof(uint4)));

    hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_philox4x32_10_state_kernel),
                       dim3(8),
                       dim3(32),
                       0,
                       0,
                       block_output,
                       state_output,
                       seed,
                       size);
    HIP_CHECK(hipGetLastError());

    std::vector<uint4> block_host(size);
    std::vector<uint4> state_host(size);
    HIP_CHECK(
        hipMemcpy(block_host.data(), block_output, size * sizeof(uint4), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(state_host.data(), state_output, size * sizeof(uint4), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(block_output));
    HIP_CHECK(hipFree(state_output));

    for(size_t i = 0; i < size; i++)
    {
        EXPECT_EQ(block_host[i].x, state_host[i].x);
        EXPECT_EQ(block_host[i].y, state_host[i].y);
        EXPECT_EQ(block_host[i].z, state_host[i].z);
        EXPECT_EQ(block_host[i].w, state_host[i].w);
    }
}

TEST(rocrand_kernel_philox4x32_10, rocrand_uniform)
{
    typedef rocrand_state_philox4x32_10 state_type;

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

TEST(rocrand_kernel_philox4x32_10, rocrand_normal)
{
    typedef rocrand_state_philox4x32_10 state_type;

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

TEST(rocrand_kernel_philox4x32_10, rocrand_log_normal)
{
    typedef rocrand_state_philox4x32_10 state_type;

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

class rocrand_kernel_philox4x32_10_poisson : public ::testing::TestWithParam<double>
{};

TEST_P(rocrand_kernel_philox4x32_10_poisson, rocrand_poisson)
{
    typedef rocrand_state_philox4x32_10 state_type;

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

TEST_P(rocrand_kernel_philox4x32_10_poisson, rocrand_discrete)
{
    typedef rocrand_state_philox4x32_10 state_type;

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

INSTANTIATE_TEST_SUITE_P(rocrand_kernel_philox4x32_10_poisson,
                         rocrand_kernel_philox4x32_10_poisson,
                         ::testing::ValuesIn(lambdas));
