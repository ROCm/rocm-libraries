// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

// required rocprim headers
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/intrinsics/warp_shuffle.hpp>

// required test headers
#include "test_utils_types.hpp"

// Custom structure
struct custom_notaligned
{
    short i;
    double d;
    float f;
    unsigned int u;

    ROCPRIM_HOST_DEVICE
    custom_notaligned() {};
    ROCPRIM_HOST_DEVICE
    ~custom_notaligned() {};
};

ROCPRIM_HOST_DEVICE
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

    ROCPRIM_HOST_DEVICE
    custom_16aligned() {};
    ROCPRIM_HOST_DEVICE
    ~custom_16aligned() {};
} __attribute__((aligned(16)));

inline ROCPRIM_HOST_DEVICE
bool operator==(const custom_16aligned& lhs, const custom_16aligned& rhs)
{
    return lhs.i == rhs.i && lhs.f == rhs.f && lhs.u == rhs.u;
}

// Params for tests
template<class T>
struct params
{
    using type = T;
};

template<class Params>
class RocprimIntrinsicsTests : public ::testing::Test
{
public:
    using type = typename Params::type;
};

typedef ::testing::Types<
    params<int>,
    params<float>,
    params<double>,
    params<unsigned char>
> IntrinsicsTestParams;

TYPED_TEST_CASE(RocprimIntrinsicsTests, IntrinsicsTestParams);

template<class T>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void shuffle_up_kernel(T* data, unsigned int delta, unsigned int width)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    T value = data[index];
    value = rocprim::warp_shuffle_up(value, delta, width);
    data[index] = value;
}

TYPED_TEST(RocprimIntrinsicsTests, ShuffleUp)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    const size_t hardware_warp_size = ::rocprim::warp_size();
    const size_t size = hardware_warp_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate input
        auto input = test_utils::get_random_data<T>(size, T(-100), T(100), seed_value);
        std::vector<T> output(input.size());

        T* device_data;
        HIP_CHECK(
            hipMalloc(
                &device_data,
                input.size() * sizeof(typename decltype(input)::value_type)
            )
        );

        for(unsigned int i = hardware_warp_size; i > 1; i = i/2)
        {
            const unsigned int logical_warp_size = i;
            SCOPED_TRACE(testing::Message() << "where logical_warp_size = " << i);

            auto deltas = test_utils::get_random_data<unsigned int>(
                std::max<size_t>(1, logical_warp_size/2),
                1U,
                std::max<unsigned int>(1, logical_warp_size - 1),
                seed_index
            );

            for(auto delta : deltas)
            {
                SCOPED_TRACE(testing::Message() << "where delta = " << delta);
                // Calculate expected results on host
                std::vector<T> expected(size, 0);
                for(size_t i = 0; i < input.size()/logical_warp_size; i++)
                {
                    for(size_t j = 0; j < logical_warp_size; j++)
                    {
                        size_t index = j + logical_warp_size * i;
                        auto up_index = j > delta-1 ? index-delta : index;
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
                    HIP_KERNEL_NAME(shuffle_up_kernel<T>),
                    dim3(1), dim3(hardware_warp_size), 0, 0,
                    device_data, delta, logical_warp_size
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
        }
        hipFree(device_data);
    }

}

template<class T>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void shuffle_down_kernel(T* data, unsigned int delta, unsigned int width)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    T value = data[index];
    value = rocprim::warp_shuffle_down(value, delta, width);
    data[index] = value;
}

TYPED_TEST(RocprimIntrinsicsTests, ShuffleDown)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    const size_t hardware_warp_size = ::rocprim::warp_size();
    const size_t size = hardware_warp_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate input
        auto input = test_utils::get_random_data<T>(size, T(-100), T(100), seed_value);
        std::vector<T> output(input.size());

        T* device_data;
        HIP_CHECK(
            hipMalloc(
                &device_data,
                input.size() * sizeof(typename decltype(input)::value_type)
            )
        );

        for(unsigned int i = hardware_warp_size; i > 1; i = i/2)
        {
            const unsigned int logical_warp_size = i;
            SCOPED_TRACE(testing::Message() << "where logical_warp_size = " << i);

            auto deltas = test_utils::get_random_data<unsigned int>(
                std::max<size_t>(1, logical_warp_size/2),
                1U,
                std::max<unsigned int>(1, logical_warp_size - 1),
                seed_index
            );

            for(auto delta : deltas)
            {
                SCOPED_TRACE(testing::Message() << "where delta = " << delta);
                // Calculate expected results on host
                std::vector<T> expected(size, 0);
                for(size_t i = 0; i < input.size()/logical_warp_size; i++)
                {
                    for(size_t j = 0; j < logical_warp_size; j++)
                    {
                        size_t index = j + logical_warp_size * i;
                        auto down_index = j+delta < logical_warp_size ? index+delta : index;
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
                    HIP_KERNEL_NAME(shuffle_down_kernel<T>),
                    dim3(1), dim3(hardware_warp_size), 0, 0,
                    device_data, delta, logical_warp_size
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
        }
        hipFree(device_data);
    }

}

template<class T>
__global__
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void shuffle_index_kernel(T* data, int* src_lanes, unsigned int width)
{
    const unsigned int index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    T value = data[index];
    value = rocprim::warp_shuffle(
        value, src_lanes[hipThreadIdx_x/width], width
    );
    data[index] = value;
}

TYPED_TEST(RocprimIntrinsicsTests, ShuffleIndex)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::type;
    const size_t hardware_warp_size = ::rocprim::warp_size();
    const size_t size = hardware_warp_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate input
        auto input = test_utils::get_random_data<T>(size, T(-100), T(100), seed_value);
        std::vector<T> output(input.size());

        T* device_data;
        int * device_src_lanes;
        HIP_CHECK(
            hipMalloc(
                &device_data,
                input.size() * sizeof(typename decltype(input)::value_type)
            )
        );
        HIP_CHECK(
            hipMalloc(
                &device_src_lanes,
                hardware_warp_size * sizeof(int)
            )
        );

        for(unsigned int i = hardware_warp_size; i > 1; i = i/2)
        {
            const unsigned int logical_warp_size = i;
            SCOPED_TRACE(testing::Message() << "where logical_warp_size = " << i);

            auto src_lanes = test_utils::get_random_data<int>(
                hardware_warp_size/logical_warp_size,
                0, std::max<int>(0, logical_warp_size-1),
                seed_index
            );

            // Calculate expected results on host
            std::vector<T> expected(size, 0);
            for(size_t i = 0; i < input.size()/logical_warp_size; i++)
            {
                int src_lane = src_lanes[i];
                for(size_t j = 0; j < logical_warp_size; j++)
                {
                    size_t index = j + logical_warp_size * i;
                    if(src_lane >= int(logical_warp_size) || src_lane < 0) src_lane = index;
                    expected[index] = input[src_lane + logical_warp_size * i];
                }
            }

            // Writing to device memory
            HIP_CHECK(
                hipMemcpy(
                    device_data, input.data(),
                    input.size() * sizeof(typename decltype(input)::value_type),
                    hipMemcpyHostToDevice
                )
            );
            HIP_CHECK(
                hipMemcpy(
                    device_src_lanes, src_lanes.data(),
                    src_lanes.size() * sizeof(typename decltype(src_lanes)::value_type),
                    hipMemcpyHostToDevice
                )
            );

            // Launching kernel
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(shuffle_index_kernel<T>),
                dim3(1), dim3(hardware_warp_size), 0, 0,
                device_data, device_src_lanes, logical_warp_size
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
        hipFree(device_src_lanes);
    }

}

TEST(RocprimIntrinsicsTests, ShuffleUpCustomStruct)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = custom_notaligned;
    const size_t hardware_warp_size = ::rocprim::warp_size();
    const size_t size = hardware_warp_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate input
        std::vector<double> random_data = test_utils::get_random_data<double>(4 * size, -100, 100, seed_value);
        std::vector<T> input(size);
        std::vector<T> output(input.size());
        for(size_t i = 0; i < 4 * input.size(); i+=4)
        {
            input[i/4].i = random_data[i];
            input[i/4].d = random_data[i+1];
            input[i/4].f = random_data[i+2];
            input[i/4].u = random_data[i+3];
        }

        T* device_data;
        HIP_CHECK(
            hipMalloc(
                &device_data,
                input.size() * sizeof(typename decltype(input)::value_type)
            )
        );

        for(unsigned int i = hardware_warp_size; i > 1; i = i/2)
        {
            const unsigned int logical_warp_size = i;
            SCOPED_TRACE(testing::Message() << "where logical_warp_size = " << i);

            auto deltas = test_utils::get_random_data<unsigned int>(
                std::max<size_t>(1, logical_warp_size/2),
                1U,
                std::max<unsigned int>(1, logical_warp_size - 1),
                seed_index
            );

            for(auto delta : deltas)
            {
                SCOPED_TRACE(testing::Message() << "where delta = " << delta);
                // Calculate expected results on host
                std::vector<T> expected(size);
                for(size_t i = 0; i < input.size()/logical_warp_size; i++)
                {
                    for(size_t j = 0; j < logical_warp_size; j++)
                    {
                        size_t index = j + logical_warp_size * i;
                        auto up_index = j > delta-1 ? index-delta : index;
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
                    HIP_KERNEL_NAME(shuffle_up_kernel<T>),
                    dim3(1), dim3(hardware_warp_size), 0, 0,
                    device_data, delta, logical_warp_size
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
        }
        hipFree(device_data);
    }

}

TEST(RocprimIntrinsicsTests, ShuffleUpCustomAlignedStruct)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = custom_16aligned;
    const size_t hardware_warp_size = ::rocprim::warp_size();
    const size_t size = hardware_warp_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate input
        std::vector<double> random_data = test_utils::get_random_data<double>(3 * size, -100, 100, seed_value);
        std::vector<T> input(size);
        std::vector<T> output(input.size());
        for(size_t i = 0; i < 3 * input.size(); i+=3)
        {
            input[i/3].i = random_data[i];
            input[i/3].u = random_data[i+1];
            input[i/3].f = random_data[i+2];
        }

        T* device_data;
        HIP_CHECK(
            hipMalloc(
                &device_data,
                input.size() * sizeof(typename decltype(input)::value_type)
            )
        );

        for(unsigned int i = hardware_warp_size; i > 1; i = i/2)
        {
            const unsigned int logical_warp_size = i;
            SCOPED_TRACE(testing::Message() << "where logical_warp_size = " << i);

            auto deltas = test_utils::get_random_data<unsigned int>(
                std::max<size_t>(1, logical_warp_size/2),
                1U,
                std::max<unsigned int>(1, logical_warp_size - 1),
                seed_index
            );

            for(auto delta : deltas)
            {
                SCOPED_TRACE(testing::Message() << "where delta = " << delta);
                // Calculate expected results on host
                std::vector<T> expected(size);
                for(size_t i = 0; i < input.size()/logical_warp_size; i++)
                {
                    for(size_t j = 0; j < logical_warp_size; j++)
                    {
                        size_t index = j + logical_warp_size * i;
                        auto up_index = j > delta-1 ? index-delta : index;
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
                    HIP_KERNEL_NAME(shuffle_up_kernel<T>),
                    dim3(1), dim3(hardware_warp_size), 0, 0,
                    device_data, delta, logical_warp_size
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
        }
        hipFree(device_data);
    }

}
