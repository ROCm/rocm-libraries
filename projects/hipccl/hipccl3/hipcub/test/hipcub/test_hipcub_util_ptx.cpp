// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <vector>
#include <cmath>

// Google Test
#include <gtest/gtest.h>
// hipCUB API
#include <hipcub/hipcub.hpp>

#include "test_utils.hpp"

#define HIP_CHECK(error) ASSERT_EQ(error, hipSuccess)

// Custom structure
struct custom_notaligned
{
    short i;
    double d;
    float f;
    unsigned int u;

    HIPCUB_HOST_DEVICE
    custom_notaligned() {};
    HIPCUB_HOST_DEVICE
    ~custom_notaligned() {};
};

HIPCUB_HOST_DEVICE
inline bool operator==(const custom_notaligned& lhs,
                       const custom_notaligned& rhs)
{
    return lhs.i == rhs.i && lhs.d == rhs.d
        && lhs.f == rhs.f &&lhs.u == rhs.u;
}

// Custom structure aligned to 16 bytes
struct custom_16aligned
{
    int i;
    unsigned int u;
    float f;

    HIPCUB_HOST_DEVICE
    custom_16aligned() {};
    HIPCUB_HOST_DEVICE
    ~custom_16aligned() {};
} __attribute__((aligned(16)));

inline HIPCUB_HOST_DEVICE
bool operator==(const custom_16aligned& lhs, const custom_16aligned& rhs)
{
    return lhs.i == rhs.i && lhs.f == rhs.f && lhs.u == rhs.u;
}

// Params for tests
template<class T, unsigned int LogicalWarpSize = HIPCUB_WARP_THREADS>
struct params
{
    using type = T;
    static constexpr unsigned int logical_warp_size = LogicalWarpSize;
};

template<class Params>
class HipcubUtilPtxTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr unsigned int logical_warp_size = Params::logical_warp_size;
};

typedef ::testing::Types<
    params<int, 32>,
    params<int, 16>,
    params<int, 8>,
    params<int, 4>,
    params<int, 2>,
    params<float>,
    params<double>,
    params<unsigned char>
> UtilPtxTestParams;

TYPED_TEST_CASE(HipcubUtilPtxTests, UtilPtxTestParams);

template<unsigned int LOGICAL_WARP_THREADS, class T>
__global__
void shuffle_up_kernel(T* data, unsigned int src_offset)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    T value = data[index];

    // first_thread argument is ignored in hipCUB with rocPRIM-backend
    const unsigned int first_thread = 0;
    // Using mask is not supported in rocPRIM, so we don't test other masks
    const unsigned int member_mask = 0xffffffff;
    value = hipcub::ShuffleUp<LOGICAL_WARP_THREADS>(
        value, src_offset, first_thread, member_mask
    );

    data[index] = value;
}

TYPED_TEST(HipcubUtilPtxTests, ShuffleUp)
{
    using T = typename TestFixture::type;
    constexpr unsigned int logical_warp_size = TestFixture::logical_warp_size;
    const size_t hardware_warp_size = HIPCUB_WARP_THREADS;
    const size_t size = hardware_warp_size;

    // Generate input
    auto input = test_utils::get_random_data<T>(size, T(-100), T(100));
    std::vector<T> output(input.size());

    auto src_offsets = test_utils::get_random_data<unsigned int>(
        std::max<size_t>(1, logical_warp_size/2),
        1U,
        std::max<unsigned int>(1, logical_warp_size - 1)
    );

    T* device_data;
    HIP_CHECK(
        hipMalloc(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)
        )
    );

    for(auto src_offset : src_offsets)
    {
        SCOPED_TRACE(testing::Message() << "where src_offset = " << src_offset);
        // Calculate expected results on host
        std::vector<T> expected(size, 0);
        for(size_t i = 0; i < input.size()/logical_warp_size; i++)
        {
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                size_t index = j + logical_warp_size * i;
                auto up_index = j > src_offset-1 ? index-src_offset : index;
                expected[index] = input[up_index];
            }
        }

        // Writing to device memory
        HIP_CHECK(
            hipMemcpy(
                device_data, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(shuffle_up_kernel<logical_warp_size, T>),
            dim3(1), dim3(hardware_warp_size), 0, 0,
            device_data, src_offset
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_data,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
        }
    }
    hipFree(device_data);
}

template<unsigned int LOGICAL_WARP_THREADS, class T>
__global__
void shuffle_down_kernel(T* data, unsigned int src_offset)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    T value = data[index];

    // last_thread argument is ignored in hipCUB with rocPRIM-backend
    const unsigned int last_thread = LOGICAL_WARP_THREADS - 1;
    // Using mask is not supported in rocPRIM, so we don't test other masks
    const unsigned int member_mask = 0xffffffff;
    value = hipcub::ShuffleDown<LOGICAL_WARP_THREADS>(
        value, src_offset, last_thread, member_mask
    );

    data[index] = value;
}

TYPED_TEST(HipcubUtilPtxTests, ShuffleDown)
{
    using T = typename TestFixture::type;
    constexpr unsigned int logical_warp_size = TestFixture::logical_warp_size;
    const size_t hardware_warp_size = HIPCUB_WARP_THREADS;
    const size_t size = hardware_warp_size;

    // Generate input
    auto input = test_utils::get_random_data<T>(size, T(-100), T(100));
    std::vector<T> output(input.size());

    auto src_offsets = test_utils::get_random_data<unsigned int>(
        std::max<size_t>(1, logical_warp_size/2),
        1U,
        std::max<unsigned int>(1, logical_warp_size - 1)
    );

    T * device_data;
    HIP_CHECK(
        hipMalloc(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)
        )
    );

    for(auto src_offset : src_offsets)
    {
        SCOPED_TRACE(testing::Message() << "where src_offset = " << src_offset);
        // Calculate expected results on host
        std::vector<T> expected(size, 0);
        for(size_t i = 0; i < input.size()/logical_warp_size; i++)
        {
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                size_t index = j + logical_warp_size * i;
                auto down_index = j+src_offset < logical_warp_size ? index+src_offset : index;
                expected[index] = input[down_index];
            }
        }

        // Writing to device memory
        HIP_CHECK(
            hipMemcpy(
                device_data, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(shuffle_down_kernel<logical_warp_size, T>),
            dim3(1), dim3(hardware_warp_size), 0, 0,
            device_data, src_offset
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_data,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
        }
    }
    hipFree(device_data);
}

template<unsigned int LOGICAL_WARP_THREADS, class T>
__global__
void shuffle_index_kernel(T* data, int* src_offsets)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    T value = data[index];

    // Using mask is not supported in rocPRIM, so we don't test other masks
    const unsigned int member_mask = 0xffffffff;
    value = hipcub::ShuffleIndex<LOGICAL_WARP_THREADS>(
        value, src_offsets[hipThreadIdx_x/LOGICAL_WARP_THREADS], member_mask
    );

    data[index] = value;
}

TYPED_TEST(HipcubUtilPtxTests, ShuffleIndex)
{
    using T = typename TestFixture::type;
    constexpr unsigned int logical_warp_size = TestFixture::logical_warp_size;
    const size_t hardware_warp_size = HIPCUB_WARP_THREADS;
    const size_t size = hardware_warp_size;

    // Generate input
    auto input = test_utils::get_random_data<T>(size, T(-100), T(100));
    std::vector<T> output(input.size());

    auto src_offsets = test_utils::get_random_data<int>(
        hardware_warp_size/logical_warp_size, 0, std::max<int>(1, logical_warp_size - 1)
    );

    // Calculate expected results on host
    std::vector<T> expected(size, 0);
    for(size_t i = 0; i < input.size()/logical_warp_size; i++)
    {
        int src_index = src_offsets[i];
        for(size_t j = 0; j < logical_warp_size; j++)
        {
            size_t index = j + logical_warp_size * i;
            if(src_index >= int(logical_warp_size) || src_index < 0) src_index = index;
            expected[index] = input[src_index + logical_warp_size * i];
        }
    }

    // Writing to device memory
    T* device_data;
    int * device_src_offsets;
    HIP_CHECK(
        hipMalloc(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)
        )
    );
    HIP_CHECK(
        hipMalloc(
            &device_src_offsets,
            src_offsets.size() * sizeof(typename decltype(src_offsets)::value_type)
        )
    );
    HIP_CHECK(
        hipMemcpy(
            device_data, input.data(),
            input.size() * sizeof(typename decltype(input)::value_type),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(
        hipMemcpy(
            device_src_offsets, src_offsets.data(),
            src_offsets.size() * sizeof(typename decltype(src_offsets)::value_type),
            hipMemcpyHostToDevice
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(shuffle_index_kernel<logical_warp_size, T>),
        dim3(1), dim3(hardware_warp_size), 0, 0,
        device_data, device_src_offsets
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_data,
            output.size() * sizeof(T),
            hipMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < output.size(); i++)
    {
        ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
    }

    hipFree(device_data);
    hipFree(device_src_offsets);
}

TEST(HipcubUtilPtxTests, ShuffleUpCustomStruct)
{
    using T = custom_notaligned;
    constexpr unsigned int hardware_warp_size = HIPCUB_WARP_THREADS;
    constexpr unsigned int logical_warp_size = hardware_warp_size;
    const size_t size = logical_warp_size;

    // Generate data
    std::vector<double> random_data = test_utils::get_random_data<double>(4 * size, -100, 100);
    std::vector<T> input(size);
    std::vector<T> output(input.size());
    for(size_t i = 0; i < 4 * input.size(); i+=4)
    {
        input[i/4].i = random_data[i];
        input[i/4].d = random_data[i+1];
        input[i/4].f = random_data[i+2];
        input[i/4].u = random_data[i+3];
    }

    auto src_offsets = test_utils::get_random_data<unsigned int>(
        std::max<size_t>(1, logical_warp_size/2),
        1U,
        std::max<unsigned int>(1, logical_warp_size - 1)
    );

    T* device_data;
    HIP_CHECK(
        hipMalloc(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)
        )
    );

    for(auto src_offset : src_offsets)
    {
        // Calculate expected results on host
        std::vector<T> expected(size);
        for(size_t i = 0; i < input.size()/logical_warp_size; i++)
        {
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                size_t index = j + logical_warp_size * i;
                auto up_index = j > src_offset-1 ? index-src_offset : index;
                expected[index] = input[up_index];
            }
        }

        // Writing to device memory
        HIP_CHECK(
            hipMemcpy(
                device_data, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(shuffle_up_kernel<logical_warp_size, T>),
            dim3(1), dim3(hardware_warp_size), 0, 0,
            device_data, src_offset
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_data,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
        }
    }
    hipFree(device_data);
}

TEST(HipcubUtilPtxTests, ShuffleUpCustomAlignedStruct)
{
    using T = custom_16aligned;
    constexpr unsigned int hardware_warp_size = HIPCUB_WARP_THREADS;
    constexpr unsigned int logical_warp_size = hardware_warp_size;
    const size_t size = logical_warp_size;

    // Generate data
    std::vector<double> random_data = test_utils::get_random_data<double>(3 * size, -100, 100);
    std::vector<T> input(size);
    std::vector<T> output(input.size());
    for(size_t i = 0; i < 3 * input.size(); i+=3)
    {
        input[i/3].i = random_data[i];
        input[i/3].u = random_data[i+1];
        input[i/3].f = random_data[i+2];
    }

    auto src_offsets = test_utils::get_random_data<unsigned int>(
        std::max<size_t>(1, logical_warp_size/2),
        1U,
        std::max<unsigned int>(1, logical_warp_size - 1)
    );

    T* device_data;
    HIP_CHECK(
        hipMalloc(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)
        )
    );

    for(auto src_offset : src_offsets)
    {
        // Calculate expected results on host
        std::vector<T> expected(size);
        for(size_t i = 0; i < input.size()/logical_warp_size; i++)
        {
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                size_t index = j + logical_warp_size * i;
                auto up_index = j > src_offset-1 ? index-src_offset : index;
                expected[index] = input[up_index];
            }
        }

        // Writing to device memory
        HIP_CHECK(
            hipMemcpy(
                device_data, input.data(),
                input.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        // Launching kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(shuffle_up_kernel<logical_warp_size, T>),
            dim3(1), dim3(hardware_warp_size), 0, 0,
            device_data, src_offset
        );
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(
            hipMemcpy(
                output.data(), device_data,
                output.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]) << "where index = " << i;
        }
    }
    hipFree(device_data);
}

__global__
void warp_id_kernel(unsigned int* output)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    output[index] = hipcub::WarpId();
}

TEST(HipcubUtilPtxTests, WarpId)
{
    constexpr unsigned int hardware_warp_size = HIPCUB_WARP_THREADS;
    const size_t block_size = 4 * hardware_warp_size;
    const size_t size = 16 * block_size;

    std::vector<unsigned int> output(size);
    unsigned int* device_output;
    HIP_CHECK(
        hipMalloc(
            &device_output,
            output.size() * sizeof(unsigned int)
        )
    );

    // Launching kernel
    hipLaunchKernelGGL(
        warp_id_kernel,
        dim3(size/block_size), dim3(block_size), 0, 0,
        device_output
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(
        hipMemcpy(
            output.data(), device_output,
            output.size() * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );

    std::vector<size_t> warp_ids(block_size/hardware_warp_size, 0);
    for(size_t i = 0; i < output.size()/hardware_warp_size; i++)
    {
        auto prev = output[i * hardware_warp_size];
        for(size_t j = 0; j < hardware_warp_size; j++)
        {
            auto index = j + i * hardware_warp_size;
            // less than number of warps in thread block
            ASSERT_LT(output[index], block_size/hardware_warp_size);
            ASSERT_GE(output[index], 0U); // > 0
            ASSERT_EQ(output[index], prev); // all in warp_ids in warp are the same
        }
        warp_ids[prev]++;
    }
    // Check if each warp_id appears the same number of times.
    for(auto warp_id_no : warp_ids)
    {
        ASSERT_EQ(warp_id_no, size/block_size);
    }
}
