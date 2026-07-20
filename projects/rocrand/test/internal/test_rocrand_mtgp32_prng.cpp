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

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_rocrand_prng.hpp"
#include <rocrand/rocrand.h>

#include <rng/mtgp32.hpp>
#include <rocrand/rocrand_mtgp32_11213.h>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#define ROCRAND_ERROR_CHECK(cmd)                                                          \
    do                                                                                    \
    {                                                                                     \
        auto status = cmd;                                                                \
        if(status != 0)                                                                   \
        {                                                                                 \
            std::cerr << "Encountered ROCRAND error: " << status << "at line" << __LINE__ \
                      << " in file " << __FILE__ << "\n";                                 \
            exit(-1);                                                                     \
        }                                                                                 \
    }                                                                                     \
    while(0)

using rocrand_impl::host::mtgp32_generator;

// Generator API tests
using mtgp32_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<mtgp32_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<mtgp32_generator, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(mtgp32_generator,
                               generator_prng_tests,
                               mtgp32_generator_prng_tests_types);

// Continuity cannot be implemented for MTGP32, as 'offset' is not supported for this
// generator. Therefore, continuity tests fail.
// INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mtgp32,
//                                generator_prng_continuity_tests,
//                                rocrand_mtgp32_generator_prng_tests_types);

#ifdef CODE_COVERAGE_ENABLED
#include "test_rocrand_host_prng.hpp"

using rocrand_impl::host::mtgp32_generator_host;
using mtgp32_generator_prng_host_tests_types = ::testing::Types<
    generator_prng_host_tests_params<mtgp32_generator_host<true>, ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

INSTANTIATE_TYPED_TEST_SUITE_P(mtgp32_host_generator,
                               generator_prng_host_tests,
                               mtgp32_generator_prng_host_tests_types);
#endif //CODE_COVERAGE_ENABLED
TEST(AdditionalTests, rocrand_make_constant)
{
    // test to make sure the copy is working and that all data is being coverted properly
    rocrand_device::mtgp32_fast_params* src_params = mtgp32dc_params_fast_11213;
    mtgp32_params*                      device_params;
    HIP_CHECK(hipMalloc(&device_params, sizeof(mtgp32_params)));

    // Bring it to device side
    ROCRAND_CHECK(rocrand_make_constant(src_params, device_params));

    //Bring it back to host to check that everything is the same
    mtgp32_params host_params[1];

    HIP_CHECK(
        hipMemcpy(host_params, device_params, sizeof(mtgp32_params) * 1, hipMemcpyDeviceToHost));

    for(size_t i = 0; i < mtgpdc_params_11213_num; i++)
    {
        rocrand_device::mtgp32_fast_params original = src_params[i];
        mtgp32_params                      copy     = host_params[0];

        ASSERT_EQ(original.pos, copy.pos_tbl[i]);
        ASSERT_EQ(original.sh1, copy.sh1_tbl[i]);
        ASSERT_EQ(original.sh2, copy.sh2_tbl[i]);

        for(size_t ii = 0; ii < MTGP_TS; ii++)
        {
            ASSERT_EQ(original.tbl[ii], copy.param_tbl[i][ii]);
            ASSERT_EQ(original.tmp_tbl[ii], copy.temper_tbl[i][ii]);
            ASSERT_EQ(original.flt_tmp_tbl[ii], copy.single_temper_tbl[i][ii]);
        }

        ASSERT_EQ(original.mask, copy.mask[0]);
    }
    HIP_CHECK(hipFree(device_params));
}

template<size_t items_per_thread, size_t block_size>
__global__
void rocrand_kernel(rocrand_state_mtgp32* states, unsigned int* device_output)
{
    constexpr size_t items_per_block = items_per_thread * block_size;
    const size_t offset = (items_per_block * blockIdx.x) + (items_per_thread * threadIdx.x);

    __shared__ rocrand_state_mtgp32 state;
    for(size_t i = 0; i < items_per_thread; i++)
    {

        if(threadIdx.x == 0)
            state = states[blockIdx.x];
        __syncthreads();

        device_output[offset + i] = rocrand(&state);

        if(threadIdx.x == 0)
            states[blockIdx.x] = state;
        __syncthreads();
    }
}

TEST(AdditionalTests, rocrand_check_uniform_property)
{
    //Test of rocrand returns a uniformly distributed distribution
    constexpr size_t items_per_thread = 512;
    constexpr size_t block_size       = 16;
    constexpr size_t grid_size        = 16;

    constexpr size_t items_per_block = items_per_thread * block_size;
    constexpr size_t size            = items_per_block * grid_size;

    rocrand_state_mtgp32* states;
    HIP_CHECK(hipMalloc(&states, sizeof(rocrand_state_mtgp32) * grid_size));

    rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, grid_size, 0);

    unsigned int* device_output;
    HIP_CHECK(hipMalloc(&device_output, sizeof(unsigned int) * size));

    rocrand_kernel<items_per_thread, block_size>
        <<<dim3(grid_size), dim3(block_size)>>>(states, device_output);

    unsigned int* host_output = new unsigned int[size];
    HIP_CHECK(
        hipMemcpy(host_output, device_output, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));

    constexpr double a = 0;
    constexpr double b = 1;

    const double expected_mean    = (a + b) / 2;
    const double expected_std_dev = (b - a) / std::sqrt(12);

    const unsigned int mini = std::numeric_limits<unsigned int>::min();
    const unsigned int maxi = std::numeric_limits<unsigned int>::max();

    double actual_mean = std::accumulate(host_output,
                                         host_output + size,
                                         (double)0.0,
                                         [=](double acc, unsigned int x)
                                         {
                                             double converted
                                                 = (a + static_cast<double>(x - mini) * (b - a))
                                                   / (static_cast<double>(maxi - mini));
                                             return acc + converted;
                                         })
                         / static_cast<double>(size);

    double actual_std_dev = std::accumulate(host_output,
                                            host_output + size,
                                            (double)0.0,
                                            [=](double acc, unsigned int x)
                                            {
                                                double converted
                                                    = (a + static_cast<double>(x - mini) * (b - a))
                                                      / (static_cast<double>(maxi - mini));
                                                return acc + std::pow(converted - actual_mean, 2);
                                            });
    actual_std_dev        = std::sqrt(actual_std_dev / static_cast<double>(size));

    // make sure results are within 5% of expected values
    ASSERT_NEAR(expected_mean, actual_mean, expected_mean * 0.05);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, expected_std_dev * 0.05);

    delete[] host_output;

    HIP_CHECK(hipFree(states));
    HIP_CHECK(hipFree(device_output));
}

__global__
void rocrand_mtgp32_block_copy_kernel(rocrand_state_mtgp32* src_states,
                                      rocrand_state_mtgp32* dest_states)
{
    rocrand_mtgp32_block_copy(src_states + blockIdx.x, dest_states + blockIdx.x);
}

template<size_t items_per_thread, size_t block_size>
__global__
void rocrand_kernel(rocrand_state_mtgp32* states1,
                    rocrand_state_mtgp32* states2,
                    unsigned int*         device_output1,
                    unsigned int*         device_output2)
{
    constexpr size_t items_per_block = items_per_thread * block_size;
    const size_t offset = (items_per_block * blockIdx.x) + (items_per_thread * threadIdx.x);

    __shared__ rocrand_state_mtgp32 src_state;
    __shared__ rocrand_state_mtgp32 dest_state;
    for(size_t i = 0; i < items_per_thread; i++)
    {

        if(threadIdx.x == 0)
        {
            src_state  = states1[blockIdx.x];
            dest_state = states2[blockIdx.x];
        }
        __syncthreads();

        device_output1[offset + i] = rocrand(&src_state);
        device_output2[offset + i] = rocrand(&dest_state);

        if(threadIdx.x == 0)
        {
            states1[blockIdx.x] = src_state;
            states2[blockIdx.x] = dest_state;
        }
        __syncthreads();
    }
}

TEST(AdditionalTests, rocrand_mtgp32_block_copy)
{
    //Test of to make sure rocrand_mtgp32_block_copy is coppying corectly
    constexpr size_t items_per_thread = 1024;
    constexpr size_t block_size       = 16;
    constexpr size_t grid_size        = 16;

    constexpr size_t items_per_block = items_per_thread * block_size;
    constexpr size_t size            = items_per_block * grid_size;

    rocrand_state_mtgp32* src_states;
    HIP_CHECK(hipMalloc(&src_states, sizeof(rocrand_state_mtgp32) * grid_size));

    rocrand_state_mtgp32* dest_states;
    HIP_CHECK(hipMalloc(&dest_states, sizeof(rocrand_state_mtgp32) * grid_size));
    rocrand_make_state_mtgp32(src_states, mtgp32dc_params_fast_11213, grid_size, 0);

    unsigned int* src_device_output;
    HIP_CHECK(hipMalloc(&src_device_output, sizeof(unsigned int) * size));

    unsigned int* pram_set_output;
    HIP_CHECK(hipMalloc(&pram_set_output, sizeof(unsigned int) * size));

    rocrand_mtgp32_block_copy_kernel<<<dim3(grid_size), dim3(block_size)>>>(src_states,
                                                                            dest_states);

    rocrand_kernel<items_per_thread, block_size>
        <<<dim3(grid_size), dim3(block_size)>>>(src_states,
                                                dest_states,
                                                src_device_output,
                                                pram_set_output);

    unsigned int* src_host_output = new unsigned int[size];
    HIP_CHECK(hipMemcpy(src_host_output,
                        src_device_output,
                        sizeof(unsigned int) * size,
                        hipMemcpyDeviceToHost));

    unsigned int* dest_host_output = new unsigned int[size];
    HIP_CHECK(hipMemcpy(dest_host_output,
                        pram_set_output,
                        sizeof(unsigned int) * size,
                        hipMemcpyDeviceToHost));

    for(size_t i = 0; i < size; i++)
        ASSERT_EQ(src_host_output[i], dest_host_output[i]) << "Index: " << i;

    delete[] src_host_output;
    delete[] dest_host_output;

    HIP_CHECK(hipFree(src_states));
    HIP_CHECK(hipFree(src_device_output));
    HIP_CHECK(hipFree(dest_states));
    HIP_CHECK(hipFree(pram_set_output));
}

__global__
void rocrand_mtgp32_set_params_kernel(rocrand_state_mtgp32* states, mtgp32_params* params)
{
    rocrand_mtgp32_set_params(states + blockIdx.x, params);
}

TEST(AdditionalTests, rocrand_mtgp32_set_params)
{
    //Test of to make sure rocrand_mtgp32_set_params is setting parameter correctly
    constexpr size_t items_per_thread = 1024;
    constexpr size_t block_size       = 16;
    constexpr size_t grid_size        = 16;

    constexpr size_t items_per_block = items_per_thread * block_size;
    constexpr size_t size            = items_per_block * grid_size;

    rocrand_state_mtgp32* created_states;
    HIP_CHECK(hipMalloc(&created_states, sizeof(rocrand_state_mtgp32) * grid_size));
    rocrand_make_state_mtgp32(created_states, mtgp32dc_params_fast_11213, grid_size, 0);

    rocrand_state_mtgp32* param_set_states;
    HIP_CHECK(hipMalloc(&param_set_states, sizeof(rocrand_state_mtgp32) * grid_size));
    rocrand_make_state_mtgp32(param_set_states, mtgp32dc_params_fast_11213, grid_size, 0);

    unsigned int* src_device_output;
    HIP_CHECK(hipMalloc(&src_device_output, sizeof(unsigned int) * size));

    unsigned int* pram_set_output;
    HIP_CHECK(hipMalloc(&pram_set_output, sizeof(unsigned int) * size));

    mtgp32_params* device_params;
    HIP_CHECK(hipMalloc(&device_params, sizeof(mtgp32_params)));
    ROCRAND_CHECK(rocrand_make_constant(mtgp32dc_params_fast_11213, device_params));

    rocrand_mtgp32_set_params_kernel<<<dim3(grid_size), dim3(1)>>>(param_set_states, device_params);

    rocrand_kernel<items_per_thread, block_size>
        <<<dim3(grid_size), dim3(block_size)>>>(created_states,
                                                param_set_states,
                                                src_device_output,
                                                pram_set_output);
    unsigned int* src_host_output = new unsigned int[size];
    HIP_CHECK(hipMemcpy(src_host_output,
                        src_device_output,
                        sizeof(unsigned int) * size,
                        hipMemcpyDeviceToHost));

    unsigned int* dest_host_output = new unsigned int[size];
    HIP_CHECK(hipMemcpy(dest_host_output,
                        pram_set_output,
                        sizeof(unsigned int) * size,
                        hipMemcpyDeviceToHost));

    for(size_t i = 0; i < size; i++)
        ASSERT_EQ(src_host_output[i], dest_host_output[i]) << "Index: " << i;

    delete[] src_host_output;
    delete[] dest_host_output;

    HIP_CHECK(hipFree(created_states));
    HIP_CHECK(hipFree(src_device_output));
    HIP_CHECK(hipFree(param_set_states));
    HIP_CHECK(hipFree(pram_set_output));
    HIP_CHECK(hipFree(device_params));
}

template<size_t items_per_thread, size_t block_size>
__global__
void operator_kernel(rocrand_state_mtgp32* states, unsigned int* device_output)
{
    constexpr size_t items_per_block = items_per_thread * block_size;
    const size_t offset = (items_per_block * blockIdx.x) + (items_per_thread * threadIdx.x);

    __shared__ rocrand_state_mtgp32 state;
    for(size_t i = 0; i < items_per_thread; i++)
    {

        if(threadIdx.x == 0)
            state = states[blockIdx.x];
        __syncthreads();

        device_output[offset + i] = state();

        if(threadIdx.x == 0)
            states[blockIdx.x] = state;
        __syncthreads();
    }
}

TEST(AdditionalTests, operator_check_uniform_property)
{
    //Test of rocrand returns a uniformly distributed distribution
    constexpr size_t items_per_thread = 1024;
    constexpr size_t block_size       = 16;
    constexpr size_t grid_size        = 16;

    constexpr size_t items_per_block = items_per_thread * block_size;
    constexpr size_t size            = items_per_block * grid_size;

    rocrand_state_mtgp32* states;
    HIP_CHECK(hipMalloc(&states, sizeof(rocrand_state_mtgp32) * grid_size));

    rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, grid_size, 0);

    unsigned int* device_output;
    HIP_CHECK(hipMalloc(&device_output, sizeof(unsigned int) * size));

    operator_kernel<items_per_thread, block_size>
        <<<dim3(grid_size), dim3(block_size)>>>(states, device_output);

    unsigned int* host_output = new unsigned int[size];
    HIP_CHECK(
        hipMemcpy(host_output, device_output, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));

    constexpr double a = 0;
    constexpr double b = 1;

    const double expected_mean    = (a + b) / 2;
    const double expected_std_dev = (b - a) / std::sqrt(12);

    const unsigned int mini = std::numeric_limits<unsigned int>::min();
    const unsigned int maxi = std::numeric_limits<unsigned int>::max();

    double actual_mean = std::accumulate(host_output,
                                         host_output + size,
                                         (double)0.0,
                                         [=](double acc, unsigned int x)
                                         {
                                             double converted
                                                 = (a + static_cast<double>(x - mini) * (b - a))
                                                   / (static_cast<double>(maxi - mini));
                                             return acc + converted;
                                         })
                         / static_cast<double>(size);

    double actual_std_dev = std::accumulate(host_output,
                                            host_output + size,
                                            (double)0.0,
                                            [=](double acc, unsigned int x)
                                            {
                                                double converted
                                                    = (a + static_cast<double>(x - mini) * (b - a))
                                                      / (static_cast<double>(maxi - mini));
                                                return acc + std::pow(converted - actual_mean, 2);
                                            });
    actual_std_dev        = std::sqrt(actual_std_dev / static_cast<double>(size));

    // make sure results are within 5% of expected values
    ASSERT_NEAR(expected_mean, actual_mean, expected_mean * 0.05);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, expected_std_dev * 0.05);

    delete[] host_output;

    HIP_CHECK(hipFree(states));
    HIP_CHECK(hipFree(device_output));
}
