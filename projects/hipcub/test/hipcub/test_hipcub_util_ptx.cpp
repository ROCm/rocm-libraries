// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_data_generation.hpp"

#include <algorithm>
#include <ostream>
#include <utility>

#include <cstdint>

// Custom structure
struct custom_notaligned
{
    short        i;
    double       d;
    float        f;
    unsigned int u;

    HIPCUB_HOST_DEVICE custom_notaligned() : i(0), d(0), f(0), u(0){};
    HIPCUB_HOST_DEVICE ~custom_notaligned(){};
};

HIPCUB_HOST_DEVICE
inline bool
    operator==(const custom_notaligned& lhs, const custom_notaligned& rhs)
{
    return lhs.i == rhs.i && lhs.d == rhs.d && lhs.f == rhs.f && lhs.u == rhs.u;
}

// Custom structure aligned to 16 bytes
struct custom_16aligned
{
    int          i;
    unsigned int u;
    float        f;

    HIPCUB_HOST_DEVICE custom_16aligned(){};
    HIPCUB_HOST_DEVICE ~custom_16aligned(){};
} __attribute__((aligned(16)));

inline HIPCUB_HOST_DEVICE
bool operator==(const custom_16aligned& lhs, const custom_16aligned& rhs)
{
    return lhs.i == rhs.i && lhs.f == rhs.f && lhs.u == rhs.u;
}

// Params for tests
template<class T, unsigned int LogicalWarpSize>
struct params
{
    using type                                      = T;
    static constexpr unsigned int logical_warp_size = LogicalWarpSize;
};

template<class Params>
class HipcubUtilPtxTests : public ::testing::Test
{
public:
    using type                                      = typename Params::type;
    static constexpr unsigned int logical_warp_size = Params::logical_warp_size;
};

using UtilPtxTestParams = ::testing::Types<params<int, 32>,
                                           params<int, 16>,
                                           params<int, 8>,
                                           params<int, 4>,
                                           params<int, 2>,
                                           params<float, HIPCUB_WARP_SIZE_32>,
                                           params<double, HIPCUB_WARP_SIZE_32>,
                                           params<test_utils::half, HIPCUB_WARP_SIZE_32>,
                                           params<test_utils::bfloat16, HIPCUB_WARP_SIZE_32>,
                                           params<unsigned char, HIPCUB_WARP_SIZE_32>
#ifdef __HIP_PLATFORM_AMD__
                                           ,
                                           params<float, HIPCUB_WARP_SIZE_64>,
                                           params<double, HIPCUB_WARP_SIZE_64>,
                                           params<unsigned char, HIPCUB_WARP_SIZE_64>,
                                           params<test_utils::half, HIPCUB_WARP_SIZE_64>,
                                           params<test_utils::bfloat16, HIPCUB_WARP_SIZE_64>
#endif
                                           >;

TYPED_TEST_SUITE(HipcubUtilPtxTests, UtilPtxTestParams);

template<unsigned int LOGICAL_WARP_THREADS, class T>
__global__
void shuffle_up_kernel(T* data, unsigned int src_offset)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    T                  value = data[index];

    // first_thread argument is ignored in hipCUB with rocPRIM-backend
    const unsigned int first_thread = 0;
    // Using mask is not supported in rocPRIM, so we don't test other masks
    const unsigned int member_mask = 0xffffffff;
    value = hipcub::ShuffleUp<LOGICAL_WARP_THREADS>(value, src_offset, first_thread, member_mask);

    data[index] = value;
}

TYPED_TEST(HipcubUtilPtxTests, ShuffleUp)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                         = typename TestFixture::type;
    constexpr unsigned int logical_warp_size        = TestFixture::logical_warp_size;
    const unsigned int     current_device_warp_size = HIPCUB_HOST_WARP_THREADS;
    const size_t           hardware_warp_size = (current_device_warp_size == HIPCUB_WARP_SIZE_32)
                                                    ? HIPCUB_WARP_SIZE_32
                                                    : HIPCUB_WARP_SIZE_64;
    const size_t           size               = hardware_warp_size;

    if(logical_warp_size > current_device_warp_size)
    {
        printf("Unsupported test warp size: %u Current device warp size: %u.    Skipping test\n",
               logical_warp_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate input
        auto           input = test_utils::get_random_data<T>(size,
                                                    test_utils::convert_to_device<T>(-100),
                                                    test_utils::convert_to_device<T>(100),
                                                    seed_value);
        std::vector<T> output(input.size());

        auto src_offsets = test_utils::get_random_data<unsigned int>(
            std::max<size_t>(1, logical_warp_size / 2),
            1U,
            std::max<unsigned int>(1, logical_warp_size - 1),
            seed_value + seed_value_addition);

        T* device_data;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)));

        for(auto src_offset : src_offsets)
        {
            SCOPED_TRACE(testing::Message() << "where src_offset = " << src_offset);
            // Calculate expected results on host
            std::vector<T> expected(size, test_utils::convert_to_device<T>(0));
            for(size_t i = 0; i < input.size() / logical_warp_size; i++)
            {
                for(size_t j = 0; j < logical_warp_size; j++)
                {
                    size_t index    = j + logical_warp_size * i;
                    auto   up_index = j > src_offset - 1 ? index - src_offset : index;
                    expected[index] = input[up_index];
                }
            }

            // Writing to device memory
            HIP_CHECK(hipMemcpy(device_data,
                                input.data(),
                                input.size() * sizeof(T),
                                hipMemcpyHostToDevice));

            // Launching kernel
            hipLaunchKernelGGL(HIP_KERNEL_NAME(shuffle_up_kernel<logical_warp_size, T>),
                               dim3(1),
                               dim3(hardware_warp_size),
                               0,
                               0,
                               device_data,
                               src_offset);
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Read from device memory
            HIP_CHECK(hipMemcpy(output.data(),
                                device_data,
                                output.size() * sizeof(T),
                                hipMemcpyDeviceToHost));

            for(size_t i = 0; i < output.size(); i++)
            {
                ASSERT_EQ(test_utils::convert_to_native(output[i]),
                          test_utils::convert_to_native(expected[i]))
                    << "where index = " << i;
            }
        }
        HIP_CHECK(hipFree(device_data));
    }
}

template<unsigned int LOGICAL_WARP_THREADS, class T>
__global__
void shuffle_down_kernel(T* data, unsigned int src_offset)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    T                  value = data[index];

    // last_thread argument is ignored in hipCUB with rocPRIM-backend
    const unsigned int last_thread = LOGICAL_WARP_THREADS - 1;
    // Using mask is not supported in rocPRIM, so we don't test other masks
    const unsigned int member_mask = 0xffffffff;
    value = hipcub::ShuffleDown<LOGICAL_WARP_THREADS>(value, src_offset, last_thread, member_mask);

    data[index] = value;
}

TYPED_TEST(HipcubUtilPtxTests, ShuffleDown)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                         = typename TestFixture::type;
    constexpr unsigned int logical_warp_size        = TestFixture::logical_warp_size;
    const unsigned int     current_device_warp_size = HIPCUB_HOST_WARP_THREADS;
    const size_t           hardware_warp_size = (current_device_warp_size == HIPCUB_WARP_SIZE_32)
                                                    ? HIPCUB_WARP_SIZE_32
                                                    : HIPCUB_WARP_SIZE_64;
    const size_t           size               = hardware_warp_size;

    if(logical_warp_size > current_device_warp_size)
    {
        printf("Unsupported test warp size: %u Current device warp size: %u.    Skipping test\n",
               logical_warp_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate input
        auto           input = test_utils::get_random_data<T>(size,
                                                    test_utils::convert_to_device<T>(-100),
                                                    test_utils::convert_to_device<T>(100),
                                                    seed_value);
        std::vector<T> output(input.size());

        auto src_offsets = test_utils::get_random_data<unsigned int>(
            std::max<size_t>(1, logical_warp_size / 2),
            1U,
            std::max<unsigned int>(1, logical_warp_size - 1),
            seed_value + seed_value_addition);

        T* device_data;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)));

        for(auto src_offset : src_offsets)
        {
            SCOPED_TRACE(testing::Message() << "where src_offset = " << src_offset);
            // Calculate expected results on host
            std::vector<T> expected(size, test_utils::convert_to_device<T>(0));
            for(size_t i = 0; i < input.size() / logical_warp_size; i++)
            {
                for(size_t j = 0; j < logical_warp_size; j++)
                {
                    size_t index = j + logical_warp_size * i;
                    auto   down_index
                        = j + src_offset < logical_warp_size ? index + src_offset : index;
                    expected[index] = input[down_index];
                }
            }

            // Writing to device memory
            HIP_CHECK(hipMemcpy(device_data,
                                input.data(),
                                input.size() * sizeof(T),
                                hipMemcpyHostToDevice));

            // Launching kernel
            hipLaunchKernelGGL(HIP_KERNEL_NAME(shuffle_down_kernel<logical_warp_size, T>),
                               dim3(1),
                               dim3(hardware_warp_size),
                               0,
                               0,
                               device_data,
                               src_offset);
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Read from device memory
            HIP_CHECK(hipMemcpy(output.data(),
                                device_data,
                                output.size() * sizeof(T),
                                hipMemcpyDeviceToHost));

            for(size_t i = 0; i < output.size(); i++)
            {
                ASSERT_EQ(test_utils::convert_to_native(output[i]),
                          test_utils::convert_to_native(expected[i]))
                    << "where index = " << i;
            }
        }
        HIP_CHECK(hipFree(device_data));
    }
}

template<unsigned int LOGICAL_WARP_THREADS, class T>
__global__
void shuffle_index_kernel(T* data, int* src_offsets)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    T                  value = data[index];

    // Using mask is not supported in rocPRIM, so we don't test other masks
    const unsigned int member_mask = 0xffffffff;
    value                          = hipcub::ShuffleIndex<LOGICAL_WARP_THREADS>(
        value,
        src_offsets[hipThreadIdx_x / LOGICAL_WARP_THREADS],
        member_mask);

    data[index] = value;
}

TYPED_TEST(HipcubUtilPtxTests, ShuffleIndex)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                         = typename TestFixture::type;
    constexpr unsigned int logical_warp_size        = TestFixture::logical_warp_size;
    const unsigned int     current_device_warp_size = HIPCUB_HOST_WARP_THREADS;
    const size_t           hardware_warp_size = (current_device_warp_size == HIPCUB_WARP_SIZE_32)
                                                    ? HIPCUB_WARP_SIZE_32
                                                    : HIPCUB_WARP_SIZE_64;
    const size_t           size               = hardware_warp_size;

    if(logical_warp_size > current_device_warp_size)
    {
        printf("Unsupported test warp size: %u Current device warp size: %u.    Skipping test\n",
               logical_warp_size,
               current_device_warp_size);
        GTEST_SKIP();
    }

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate input
        auto           input = test_utils::get_random_data<T>(size,
                                                    test_utils::convert_to_device<T>(-100),
                                                    test_utils::convert_to_device<T>(100),
                                                    seed_value);
        std::vector<T> output(input.size());

        auto src_offsets = test_utils::get_random_data<int>(hardware_warp_size / logical_warp_size,
                                                            0,
                                                            std::max<int>(1, logical_warp_size - 1),
                                                            seed_value + seed_value_addition);

        // Calculate expected results on host
        std::vector<T> expected(size, test_utils::convert_to_device<T>(0));
        for(size_t i = 0; i < input.size() / logical_warp_size; i++)
        {
            int src_index = src_offsets[i];
            for(size_t j = 0; j < logical_warp_size; j++)
            {
                size_t index = j + logical_warp_size * i;
                if(src_index >= int(logical_warp_size) || src_index < 0)
                    src_index = index;
                expected[index] = input[src_index + logical_warp_size * i];
            }
        }

        // Writing to device memory
        T*   device_data;
        int* device_src_offsets;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)));
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_src_offsets,
            src_offsets.size() * sizeof(typename decltype(src_offsets)::value_type)));
        HIP_CHECK(hipMemcpy(device_data,
                            input.data(),
                            input.size() * sizeof(typename decltype(input)::value_type),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_src_offsets,
                            src_offsets.data(),
                            src_offsets.size() * sizeof(typename decltype(src_offsets)::value_type),
                            hipMemcpyHostToDevice));

        // Launching kernel
        hipLaunchKernelGGL(HIP_KERNEL_NAME(shuffle_index_kernel<logical_warp_size, T>),
                           dim3(1),
                           dim3(hardware_warp_size),
                           0,
                           0,
                           device_data,
                           device_src_offsets);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Read from device memory
        HIP_CHECK(hipMemcpy(output.data(),
                            device_data,
                            output.size() * sizeof(T),
                            hipMemcpyDeviceToHost));

        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(test_utils::convert_to_native(output[i]),
                      test_utils::convert_to_native(expected[i]))
                << "where index = " << i;
        }

        HIP_CHECK(hipFree(device_data));
        HIP_CHECK(hipFree(device_src_offsets));
    }
}

TEST(HipcubUtilPtxTests, ShuffleUpCustomStruct)
{
    using T                                     = custom_notaligned;
    constexpr unsigned int logical_warp_size_32 = HIPCUB_WARP_SIZE_32;
    constexpr unsigned int logical_warp_size_64 = HIPCUB_WARP_SIZE_64;

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;
    const unsigned int logical_warp_size        = (current_device_warp_size == HIPCUB_WARP_SIZE_32)
                                                      ? logical_warp_size_32
                                                      : logical_warp_size_64;
    const size_t size = (current_device_warp_size == HIPCUB_WARP_SIZE_32) ? logical_warp_size_32
                                                                          : logical_warp_size_64;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<double> random_data
            = test_utils::get_random_data<double>(4 * size,
                                                  static_cast<double>(-100),
                                                  static_cast<double>(100),
                                                  seed_value);
        std::vector<T> input(size);
        std::vector<T> output(input.size());
        for(size_t i = 0; i < 4 * input.size(); i += 4)
        {
            input[i / 4].i = random_data[i];
            input[i / 4].d = random_data[i + 1];
            input[i / 4].f = random_data[i + 2];
            input[i / 4].u = random_data[i + 3];
        }

        auto src_offsets = test_utils::get_random_data<unsigned int>(
            std::max<size_t>(1, logical_warp_size / 2),
            1U,
            std::max<unsigned int>(1, logical_warp_size - 1),
            seed_value + seed_value_addition);

        T* device_data;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)));

        for(auto src_offset : src_offsets)
        {
            // Calculate expected results on host
            std::vector<T> expected(size);
            for(size_t i = 0; i < input.size() / logical_warp_size; i++)
            {
                for(size_t j = 0; j < logical_warp_size; j++)
                {
                    size_t index    = j + logical_warp_size * i;
                    auto   up_index = j > src_offset - 1 ? index - src_offset : index;
                    expected[index] = input[up_index];
                }
            }

            // Writing to device memory
            HIP_CHECK(hipMemcpy(device_data,
                                input.data(),
                                input.size() * sizeof(T),
                                hipMemcpyHostToDevice));

            if(logical_warp_size == logical_warp_size_32)
            {
                // Launching kernel
                hipLaunchKernelGGL(HIP_KERNEL_NAME(shuffle_up_kernel<logical_warp_size_32, T>),
                                   dim3(1),
                                   dim3(HIPCUB_WARP_SIZE_32),
                                   0,
                                   0,
                                   device_data,
                                   src_offset);
            }
            else if(logical_warp_size == logical_warp_size_64)
            {
                // Launching kernel
                hipLaunchKernelGGL(HIP_KERNEL_NAME(shuffle_up_kernel<logical_warp_size_64, T>),
                                   dim3(1),
                                   dim3(HIPCUB_WARP_SIZE_64),
                                   0,
                                   0,
                                   device_data,
                                   src_offset);
            }
            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Read from device memory
            HIP_CHECK(hipMemcpy(output.data(),
                                device_data,
                                output.size() * sizeof(T),
                                hipMemcpyDeviceToHost));

            for(size_t i = 0; i < output.size(); i++)
            {
                ASSERT_EQ(test_utils::convert_to_native(output[i]),
                          test_utils::convert_to_native(expected[i]))
                    << "where index = " << i;
            }
        }
        HIP_CHECK(hipFree(device_data));
    }
}

TEST(HipcubUtilPtxTests, ShuffleUpCustomAlignedStruct)
{
    using T                                     = custom_16aligned;
    constexpr unsigned int logical_warp_size_32 = HIPCUB_WARP_SIZE_32;
    constexpr unsigned int logical_warp_size_64 = HIPCUB_WARP_SIZE_64;

    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;
    const unsigned int hardware_warp_size       = (current_device_warp_size == HIPCUB_WARP_SIZE_32)
                                                      ? HIPCUB_WARP_SIZE_32
                                                      : HIPCUB_WARP_SIZE_64;
    const unsigned int logical_warp_size        = (current_device_warp_size == HIPCUB_WARP_SIZE_32)
                                                      ? logical_warp_size_32
                                                      : logical_warp_size_64;
    const size_t size = (current_device_warp_size == HIPCUB_WARP_SIZE_32) ? logical_warp_size_32
                                                                          : logical_warp_size_64;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<double> random_data
            = test_utils::get_random_data<double>(3 * size,
                                                  static_cast<double>(-100),
                                                  static_cast<double>(100),
                                                  seed_value);
        std::vector<T> input(size);
        std::vector<T> output(input.size());
        for(size_t i = 0; i < 3 * input.size(); i += 3)
        {
            input[i / 3].i = random_data[i];
            input[i / 3].u = random_data[i + 1];
            input[i / 3].f = random_data[i + 2];
        }

        auto src_offsets = test_utils::get_random_data<unsigned int>(
            std::max<size_t>(1, logical_warp_size / 2),
            1U,
            std::max<unsigned int>(1, logical_warp_size - 1),
            seed_value + seed_value_addition);

        T* device_data;
        HIP_CHECK(test_common_utils::hipMallocHelper(
            &device_data,
            input.size() * sizeof(typename decltype(input)::value_type)));

        for(auto src_offset : src_offsets)
        {
            // Calculate expected results on host
            std::vector<T> expected(size);
            for(size_t i = 0; i < input.size() / logical_warp_size; i++)
            {
                for(size_t j = 0; j < logical_warp_size; j++)
                {
                    size_t index    = j + logical_warp_size * i;
                    auto   up_index = j > src_offset - 1 ? index - src_offset : index;
                    expected[index] = input[up_index];
                }
            }

            // Writing to device memory
            HIP_CHECK(hipMemcpy(device_data,
                                input.data(),
                                input.size() * sizeof(T),
                                hipMemcpyHostToDevice));

            if(logical_warp_size == logical_warp_size_32)
            {
                // Launching kernel
                hipLaunchKernelGGL(HIP_KERNEL_NAME(shuffle_up_kernel<logical_warp_size_32, T>),
                                   dim3(1),
                                   dim3(hardware_warp_size),
                                   0,
                                   0,
                                   device_data,
                                   src_offset);
            }
            else if(logical_warp_size == logical_warp_size_64)
            {
                // Launching kernel
                hipLaunchKernelGGL(HIP_KERNEL_NAME(shuffle_up_kernel<logical_warp_size_64, T>),
                                   dim3(1),
                                   dim3(hardware_warp_size),
                                   0,
                                   0,
                                   device_data,
                                   src_offset);
            }

            HIP_CHECK(hipPeekAtLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Read from device memory
            HIP_CHECK(hipMemcpy(output.data(),
                                device_data,
                                output.size() * sizeof(T),
                                hipMemcpyDeviceToHost));

            for(size_t i = 0; i < output.size(); i++)
            {
                ASSERT_EQ(test_utils::convert_to_native(output[i]),
                          test_utils::convert_to_native(expected[i]))
                    << "where index = " << i;
            }
        }
        HIP_CHECK(hipFree(device_data));
    }
}

__global__
void warp_id_kernel(unsigned int* output)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    output[index]            = ::rocprim::warp_id();
}

TEST(HipcubUtilPtxTests, WarpId)
{
    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;
    const unsigned int hardware_warp_size       = (current_device_warp_size == HIPCUB_WARP_SIZE_32)
                                                      ? HIPCUB_WARP_SIZE_32
                                                      : HIPCUB_WARP_SIZE_64;
    const size_t       block_size               = (current_device_warp_size == HIPCUB_WARP_SIZE_32)
                                                      ? 4 * HIPCUB_WARP_SIZE_32
                                                      : 4 * HIPCUB_WARP_SIZE_64;
    const size_t       size                     = 16 * block_size;

    std::vector<unsigned int> output(size);
    unsigned int*             device_output;
    HIP_CHECK(
        test_common_utils::hipMallocHelper(&device_output, output.size() * sizeof(unsigned int)));

    // Launching kernel
    hipLaunchKernelGGL(warp_id_kernel,
                       dim3(size / block_size),
                       dim3(block_size),
                       0,
                       0,
                       device_output);
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Read from device memory
    HIP_CHECK(hipMemcpy(output.data(),
                        device_output,
                        output.size() * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));

    std::vector<size_t> warp_ids(block_size / hardware_warp_size, 0);
    for(size_t i = 0; i < output.size() / hardware_warp_size; i++)
    {
        auto prev = output[i * hardware_warp_size];
        for(size_t j = 0; j < hardware_warp_size; j++)
        {
            auto index = j + i * hardware_warp_size;
            // less than number of warps in thread block
            ASSERT_LT(output[index], block_size / hardware_warp_size);
            ASSERT_GE(output[index], 0U); // > 0
            ASSERT_EQ(output[index], prev); // all in warp_ids in warp are the same
        }
        warp_ids[prev]++;
    }
    // Check if each warp_id appears the same number of times.
    for(auto warp_id_no : warp_ids)
    {
        ASSERT_EQ(warp_id_no, size / block_size);
    }

    HIP_CHECK(hipFree(device_output));
}

enum class TestStatus : uint8_t
{
    Failed = 0,
    Passed = 1
};

std::ostream& operator<<(std::ostream& lhs, TestStatus rhs)
{
    switch(rhs)
    {
        case TestStatus::Failed: return lhs << "F";
        case TestStatus::Passed: return lhs << "P";
    }
    return lhs << "Unknown(" << static_cast<int>(rhs) << ")";
}

HIPCUB_DEVICE
bool is_lane_in_mask(const uint64_t mask, const unsigned int lane)
{
    return (uint64_t(1) << lane) & mask;
}

template<unsigned int LogicalWarpSize>
HIPCUB_DEVICE
std::enable_if_t<(HIPCUB_DEVICE_WARP_THREADS >= LogicalWarpSize), TestStatus>
test_warp_mask_pow_two() {
    const unsigned int logical_warp_id = ::rocprim::lane_id() / LogicalWarpSize;
    const uint64_t mask = hipcub::WarpMask<LogicalWarpSize>(logical_warp_id);

    const unsigned int warp_start      = logical_warp_id * LogicalWarpSize;
    const unsigned int next_warp_start = (logical_warp_id + 1) * LogicalWarpSize;
    for(unsigned int lane = 0; lane < warp_start; ++lane)
    {
        if(is_lane_in_mask(mask, lane))
        {
            return TestStatus::Failed;
        }
    }
    for(unsigned int lane = warp_start; lane < next_warp_start; ++lane)
    {
        if(!is_lane_in_mask(mask, lane))
        {
            return TestStatus::Failed;
        }
    }
    for(unsigned int lane = next_warp_start; lane < 64; ++lane)
    {
        if(is_lane_in_mask(mask, lane))
        {
            return TestStatus::Failed;
        }
    }
    return TestStatus::Passed;
}

template<unsigned int LogicalWarpSize>
HIPCUB_DEVICE
std::enable_if_t<!(HIPCUB_DEVICE_WARP_THREADS >= LogicalWarpSize), TestStatus>
    test_warp_mask_pow_two()
{
    return TestStatus::Passed;
}

template<unsigned int LogicalWarpSize>
HIPCUB_DEVICE
std::enable_if_t<(HIPCUB_DEVICE_WARP_THREADS >= LogicalWarpSize), TestStatus>
test_warp_mask_non_pow_two() {
    const unsigned int logical_warp_id = ::rocprim::lane_id() / LogicalWarpSize;
    const uint64_t mask = hipcub::WarpMask<LogicalWarpSize>(logical_warp_id);

    for(unsigned int lane = 0; lane < LogicalWarpSize; ++lane)
    {
        if(!is_lane_in_mask(mask, lane))
        {
            return TestStatus::Failed;
        }
    }
    for(unsigned int lane = LogicalWarpSize; lane < 64; ++lane)
    {
        if(is_lane_in_mask(mask, lane))
        {
            return TestStatus::Failed;
        }
    }
    return TestStatus::Passed;
}

template<unsigned int LogicalWarpSize>
HIPCUB_DEVICE
std::enable_if_t<!(HIPCUB_DEVICE_WARP_THREADS >= LogicalWarpSize), TestStatus>
    test_warp_mask_non_pow_two()
{
    return TestStatus::Passed;
}

template<unsigned int LogicalWarpSize>
__global__
void device_test_warp_mask(TestStatus* statuses)
{
    constexpr bool is_power_of_two = test_utils::is_power_of_two(LogicalWarpSize);
    statuses[threadIdx.x]          = is_power_of_two ? test_warp_mask_pow_two<LogicalWarpSize>()
                                                     : test_warp_mask_non_pow_two<LogicalWarpSize>();
}

template<unsigned int LogicalWarpSize>
void test_warp_size(std::vector<TestStatus>& statuses,
                    TestStatus*              d_statuses,
                    const unsigned int       device_warp_size)
{
    if(LogicalWarpSize > device_warp_size)
    {
        return;
    }

    statuses.clear();
    statuses.insert(statuses.begin(), device_warp_size, TestStatus::Failed);

    HIP_CHECK(hipMemcpy(d_statuses,
                        statuses.data(),
                        statuses.size() * sizeof(statuses[0]),
                        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(device_test_warp_mask<LogicalWarpSize>,
                       dim3(1),
                       dim3(device_warp_size),
                       0,
                       0,
                       d_statuses);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(statuses.data(),
                        d_statuses,
                        statuses.size() * sizeof(statuses[0]),
                        hipMemcpyDeviceToHost));

    SCOPED_TRACE(testing::Message() << "where LogicalWarpSize = " << LogicalWarpSize);
    ASSERT_TRUE(std::all_of(statuses.begin(),
                            statuses.end(),
                            [](const TestStatus status) { return status == TestStatus::Passed; }));
}

template<unsigned int... LogicalWarpSizes>
void test_all_warp_sizes(std::integer_sequence<unsigned int, LogicalWarpSizes...>)
{
    const int device_warp_size = HIPCUB_HOST_WARP_THREADS;
    ASSERT_GT(device_warp_size, 0);

    SCOPED_TRACE(testing::Message() << "where device warp size = " << device_warp_size);

    TestStatus* d_statuses = nullptr;
    HIP_CHECK(
        test_common_utils::hipMallocHelper(&d_statuses, static_cast<size_t>(device_warp_size)));

    // Call the test with each logical warp size in the range [1, 64]
    auto       statuses = std::vector<TestStatus>();
    const auto ignore
        = {(test_warp_size<LogicalWarpSizes + 1>(statuses,
                                                 d_statuses,
                                                 static_cast<unsigned int>(device_warp_size)),
            0)...};
    static_cast<void>(ignore);

    HIP_CHECK(hipFree(d_statuses));
}

TEST(HipcubUtilPtxTests, WarpMask)
{
    using sequence = std::make_integer_sequence<unsigned int, 64>;
    test_all_warp_sizes(sequence{});
}
