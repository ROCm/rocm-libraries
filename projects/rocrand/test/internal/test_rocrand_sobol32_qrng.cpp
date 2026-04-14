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

#include "rocrand/rocrand.h"
#include "test_rocrand_sobol_qrng.hpp"

#include <numeric>

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

using rocrand_impl::host::sobol32_generator;

using test_sobol32_qrng_types
    = ::testing::Types<sobol_qrng_tests_params<sobol32_generator, ROCRAND_ORDERING_QUASI_DEFAULT>>;

INSTANTIATE_TYPED_TEST_SUITE_P(sobol_qrng_tests, sobol_qrng_tests, test_sobol32_qrng_types);

using uint = unsigned int;

TEST(AdditionalTest, host_rocrand_init_consistent)
{
    //making sure that the the output is consistent when init parameter are the same
    const uint* directions;
    ROCRAND_ERROR_CHECK(
        rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    std::vector<uint> offsets = {1, 2, 4, 8, 32, 64, 128};

    for(const uint& offset : offsets)
    {
        rocrand_state_sobol32 state1, state2;

        rocrand_init(directions, offset, &state1);
        rocrand_init(directions, offset, &state2);

        for(size_t i = 0; i < 10000; i++)
            ASSERT_EQ(rocrand(&state1), rocrand(&state2));
    }
}

TEST(AdditionalTest, host_rocrand_init_offset)
{
    //test the offset functionality
    const uint* directions;
    ROCRAND_ERROR_CHECK(
        rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    std::vector<uint> offsets = {1, 2, 4, 8, 32, 64, 128};

    for(const uint& offset : offsets)
    {
        rocrand_state_sobol32 state1, state2;

        rocrand_init(directions, 0, &state1);
        rocrand_init(directions, offset, &state2);

        for(uint i = 0; i < offset; i++)
            rocrand(&state1);
        ASSERT_EQ(rocrand(&state1), rocrand(&state2));
    }
}

TEST(AdditionalTest, host_rocrand)
{
    // test to make sure rocrand returns uniformly distributed values
    const uint* directions;
    ROCRAND_ERROR_CHECK(
        rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    std::vector<uint> offsets = {1, 2, 4, 8, 32, 64, 128};

    constexpr size_t test_size = 10000;

    std::vector<double> output(test_size);

    constexpr double a = 0;
    constexpr double b = 1;

    const double expected_mean    = (a + b) / 2;
    const double expected_std_dev = (b - a) / std::sqrt(12);

    const uint mini = std::numeric_limits<uint>::min();
    const uint maxi = std::numeric_limits<uint>::max();

    for(const uint& offset : offsets)
    {
        rocrand_state_sobol32 state;

        rocrand_init(directions, offset, &state);

        for(size_t i = 0; i < test_size; i++)
        {
            //converting to range between 0 and 1
            output[i] = (a + static_cast<double>(rocrand(&state) - mini) * (b - a))
                        / (static_cast<double>(maxi - mini));
        }

        double actual_mean
            = std::accumulate(output.begin(), output.end(), 0.0) / static_cast<double>(test_size);
        double actual_std_dev = std::accumulate(output.begin(),
                                                output.end(),
                                                0.0,
                                                [=](double acc, double x)
                                                { return acc + std::pow(x - actual_mean, 2); });
        actual_std_dev        = std::sqrt(actual_std_dev / static_cast<double>(test_size));

        // make sure results are within 5% of expected values
        ASSERT_NEAR(expected_mean, actual_mean, expected_mean * 0.05);
        ASSERT_NEAR(expected_std_dev, actual_std_dev, expected_std_dev * 0.05);
    }
}

TEST(AdditionalTest, host_skipahead)
{
    //test the skipahead functionality
    const uint* directions;
    ROCRAND_ERROR_CHECK(
        rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    std::vector<uint> offsets = {1, 2, 4, 8, 32, 64, 128};

    for(const uint& offset : offsets)
    {
        rocrand_state_sobol32 state1, state2;

        rocrand_init(directions, 0, &state1);
        rocrand_init(directions, offset, &state2);

        skipahead(offset, &state1);

        ASSERT_EQ(rocrand(&state1), rocrand(&state2));
    }
}

template<size_t test_size, size_t offset_size>
__global__
void rocrand_init_consistency_kernel(uint*       device_output1,
                                     uint*       device_output2,
                                     const uint* device_directions,
                                     const uint* offsets)
{

    size_t counter = 0;
    for(size_t i = 0; i < offset_size; i++)
    {
        uint offset = offsets[i];

        rocrand_state_sobol32 state1, state2;
        rocrand_init(device_directions, offset, &state1);
        rocrand_init(device_directions, offset, &state2);

        for(size_t ii = 0; ii < test_size; ii++)
        {
            device_output1[counter]   = rocrand(&state1);
            device_output2[counter++] = rocrand(&state2);
        }
    }
}

TEST(AdditionalTest, device_rocrand_init_consistent)
{
    //making sure that the the output is consistent when init parameter are the same

    constexpr uint test_size   = 1000;
    constexpr uint offset_size = 5;

    const uint* directions;
    ROCRAND_ERROR_CHECK(
        rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    const uint offsets[offset_size] = {8, 16, 32, 64, 128};

    uint host_output1[offset_size * test_size];
    uint host_output2[offset_size * test_size];

    uint* device_directions;
    uint* device_offsets;
    uint* device_output1;
    uint* device_output2;

    // 640000 is the size of ROCRAND_DIRECTION_VECTORS_32_JOEKUO6
    HIP_CHECK(hipMalloc(&device_directions, sizeof(uint) * 640000));
    HIP_CHECK(hipMalloc(&device_offsets, sizeof(uint) * offset_size));
    HIP_CHECK(hipMalloc(&device_output1, sizeof(uint) * offset_size * test_size));
    HIP_CHECK(hipMalloc(&device_output2, sizeof(uint) * offset_size * test_size));

    HIP_CHECK(
        hipMemcpy(device_directions, directions, sizeof(uint) * 640000, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(device_offsets, offsets, sizeof(uint) * offset_size, hipMemcpyHostToDevice));

    rocrand_init_consistency_kernel<test_size, offset_size>
        <<<1, 1>>>(device_output1, device_output2, device_directions, device_offsets);

    HIP_CHECK(hipMemcpy(host_output1,
                        device_output1,
                        sizeof(uint) * offset_size * test_size,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_output2,
                        device_output2,
                        sizeof(uint) * offset_size * test_size,
                        hipMemcpyDeviceToHost));

    for(size_t i = 0; i < offset_size * test_size; i++)
        ASSERT_EQ(host_output1[i], host_output2[i]);

    HIP_CHECK(hipFree(device_directions));
    HIP_CHECK(hipFree(device_offsets));
    HIP_CHECK(hipFree(device_output1));
    HIP_CHECK(hipFree(device_output2));
}

template<size_t offset_size>
__global__
void rocrand_init_offset_kernel(uint*       device_output1,
                                uint*       device_output2,
                                const uint* device_directions,
                                const uint* offsets)
{

    for(size_t i = 0; i < offset_size; i++)
    {
        uint offset = offsets[i];

        rocrand_state_sobol32 state1, state2;
        rocrand_init(device_directions, 0, &state1);
        rocrand_init(device_directions, offset, &state2);

        for(size_t ii = 0; ii < offset; ii++)
            rocrand(&state1);

        device_output1[i] = rocrand(&state1);
        device_output2[i] = rocrand(&state2);
    }
}

TEST(AdditionalTest, device_rocrand_init_offset)
{
    //test the offset functionality

    constexpr uint offset_size = 5;
    const uint*    directions;
    ROCRAND_ERROR_CHECK(
        rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    const uint offsets[offset_size] = {8, 16, 32, 64, 128};

    uint host_output1[offset_size];
    uint host_output2[offset_size];

    uint* device_directions;
    uint* device_offsets;
    uint* device_output1;
    uint* device_output2;

    // 640000 is the size of ROCRAND_DIRECTION_VECTORS_32_JOEKUO6
    HIP_CHECK(hipMalloc(&device_directions, sizeof(uint) * 640000));
    HIP_CHECK(hipMalloc(&device_offsets, sizeof(uint) * offset_size));
    HIP_CHECK(hipMalloc(&device_output1, sizeof(uint) * offset_size));
    HIP_CHECK(hipMalloc(&device_output2, sizeof(uint) * offset_size));

    HIP_CHECK(
        hipMemcpy(device_directions, directions, sizeof(uint) * 640000, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(device_offsets, offsets, sizeof(uint) * offset_size, hipMemcpyHostToDevice));

    rocrand_init_offset_kernel<offset_size>
        <<<1, 1>>>(device_output1, device_output2, device_directions, device_offsets);

    HIP_CHECK(
        hipMemcpy(host_output1, device_output1, sizeof(uint) * offset_size, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(host_output2, device_output2, sizeof(uint) * offset_size, hipMemcpyDeviceToHost));

    for(size_t i = 0; i < offset_size; i++)
        ASSERT_EQ(host_output1[i], host_output2[i]);

    HIP_CHECK(hipFree(device_directions));
    HIP_CHECK(hipFree(device_offsets));
    HIP_CHECK(hipFree(device_output1));
    HIP_CHECK(hipFree(device_output2));
}

template<size_t test_size, size_t offset_size>
__global__
void rocrand_kernel(uint* device_output, const uint* device_directions, const uint* offsets)
{

    size_t counter = 0;
    for(size_t i = 0; i < offset_size; i++)
    {
        uint offset = offsets[i];

        rocrand_state_sobol32 state;
        rocrand_init(device_directions, offset, &state);

        for(size_t ii = 0; ii < test_size; ii++)
        {
            device_output[counter++] = rocrand(&state);
        }
    }
}

TEST(AdditionalTest, device_rocrand)
{
    // test to make sure rocrand returns uniformly distributed values
    const uint* host_directions;
    ROCRAND_ERROR_CHECK(
        rocrand_get_direction_vectors32(&host_directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));

    constexpr size_t test_size   = 10000;
    constexpr size_t offset_size = 8;
    constexpr size_t total_size  = offset_size * test_size;

    uint host_offsets[offset_size] = {1, 2, 4, 8, 16, 32, 64, 128};
    uint host_output[total_size];

    uint* device_directions;
    uint* device_offsets;
    uint* device_output;

    // 640000 is the size of ROCRAND_DIRECTION_VECTORS_32_JOEKUO6
    HIP_CHECK(hipMalloc(&device_directions, sizeof(uint) * 640000));
    HIP_CHECK(hipMalloc(&device_offsets, sizeof(uint) * offset_size));
    HIP_CHECK(hipMalloc(&device_output, sizeof(uint) * total_size));

    HIP_CHECK(hipMemcpy(device_directions,
                        host_directions,
                        sizeof(uint) * 640000,
                        hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(device_offsets, host_offsets, sizeof(uint) * offset_size, hipMemcpyHostToDevice));

    rocrand_kernel<test_size, offset_size>
        <<<1, 1>>>(device_output, device_directions, device_offsets);

    HIP_CHECK(
        hipMemcpy(host_output, device_output, sizeof(uint) * total_size, hipMemcpyDeviceToHost));

    constexpr double a = 0;
    constexpr double b = 1;

    const double expected_mean    = (a + b) / 2;
    const double expected_std_dev = (b - a) / std::sqrt(12);

    const uint mini = std::numeric_limits<uint>::min();
    const uint maxi = std::numeric_limits<uint>::max();

    double actual_mean = std::accumulate(host_output,
                                         host_output + total_size,
                                         (double)0.0,
                                         [=](double acc, uint x)
                                         {
                                             double converted
                                                 = (a + static_cast<double>(x - mini) * (b - a))
                                                   / (static_cast<double>(maxi - mini));
                                             return acc + converted;
                                         })
                         / static_cast<double>(total_size);

    double actual_std_dev = std::accumulate(host_output,
                                            host_output + total_size,
                                            (double)0.0,
                                            [=](double acc, uint x)
                                            {
                                                double converted
                                                    = (a + static_cast<double>(x - mini) * (b - a))
                                                      / (static_cast<double>(maxi - mini));
                                                return acc + std::pow(converted - actual_mean, 2);
                                            });
    actual_std_dev        = std::sqrt(actual_std_dev / static_cast<double>(total_size));

    // make sure results are within 5% of expected values
    ASSERT_NEAR(expected_mean, actual_mean, expected_mean * 0.05);
    ASSERT_NEAR(expected_std_dev, actual_std_dev, expected_std_dev * 0.05);

    HIP_CHECK(hipFree(device_directions));
    HIP_CHECK(hipFree(device_offsets));
    HIP_CHECK(hipFree(device_output));
}

template<size_t offset_size>
__global__
void skipahead_kernel(uint*       device_output1,
                      uint*       device_output2,
                      const uint* device_directions,
                      const uint* offsets)
{

    for(size_t i = 0; i < offset_size; i++)
    {
        uint offset = offsets[i];

        rocrand_state_sobol32 state1, state2;
        rocrand_init(device_directions, 0, &state1);
        rocrand_init(device_directions, offset, &state2);

        skipahead(offset, &state1);

        device_output1[i] = rocrand(&state1);
        device_output2[i] = rocrand(&state2);
    }
}

TEST(AdditionalTest, device_skipahead)
{
    //test the offset functionality

    constexpr uint offset_size = 5;
    const uint*    directions;
    ROCRAND_ERROR_CHECK(
        rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
    const uint offsets[offset_size] = {8, 16, 32, 64, 128};

    uint host_output1[offset_size];
    uint host_output2[offset_size];

    uint* device_directions;
    uint* device_offsets;
    uint* device_output1;
    uint* device_output2;

    // 640000 is the size of ROCRAND_DIRECTION_VECTORS_32_JOEKUO6
    HIP_CHECK(hipMalloc(&device_directions, sizeof(uint) * 640000));
    HIP_CHECK(hipMalloc(&device_offsets, sizeof(uint) * offset_size));
    HIP_CHECK(hipMalloc(&device_output1, sizeof(uint) * offset_size));
    HIP_CHECK(hipMalloc(&device_output2, sizeof(uint) * offset_size));

    HIP_CHECK(
        hipMemcpy(device_directions, directions, sizeof(uint) * 640000, hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(device_offsets, offsets, sizeof(uint) * offset_size, hipMemcpyHostToDevice));

    skipahead_kernel<offset_size>
        <<<1, 1>>>(device_output1, device_output2, device_directions, device_offsets);

    HIP_CHECK(
        hipMemcpy(host_output1, device_output1, sizeof(uint) * offset_size, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(host_output2, device_output2, sizeof(uint) * offset_size, hipMemcpyDeviceToHost));

    for(size_t i = 0; i < offset_size; i++)
        ASSERT_EQ(host_output1[i], host_output2[i]);

    HIP_CHECK(hipFree(device_directions));
    HIP_CHECK(hipFree(device_offsets));
    HIP_CHECK(hipFree(device_output1));
    HIP_CHECK(hipFree(device_output2));
}
