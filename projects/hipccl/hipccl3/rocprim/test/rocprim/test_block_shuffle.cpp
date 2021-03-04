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
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>
#include <rocprim/block/block_shuffle.hpp>
#include <rocprim/block/block_sort.hpp>

// required test headers
#include "test_utils_types.hpp"

template<typename Params>
class RocprimBlockShuffleTests : public ::testing::Test {
public:
    using type = typename Params::input_type;
    static constexpr unsigned int block_size = Params::block_size;
};

TYPED_TEST_CASE(RocprimBlockShuffleTests, BlockParams);

template<
    unsigned int BlockSize,
    class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void shuffle_offset_kernel(T* device_input, T* device_output)
{
    const unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    rocprim::block_shuffle<T,BlockSize> b_shuffle;
    b_shuffle.offset(device_input[index],device_output[index]);
}

TYPED_TEST(RocprimBlockShuffleTests, BlockOffset)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::type;
    const size_t block_size = 20;
    const size_t size = block_size * 10;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<type> input_data = test_utils::get_random_data<type>(size, -100, 100, seed_value);
        std::vector<type> output_data(input_data);

        // Calculate expected results on host
          // for(size_t i = 0; i < output.size() / block_size; i++)
          // {
          //     std::sort(
          //         expected.begin() + (i * block_size),
          //         expected.begin() + ((i + 1) * block_size),
          //         binary_op
          //     );
          // }

        // Preparing device
        type * device_input;
        type * device_output;

        HIP_CHECK(hipMalloc(&device_input, input_data.size() * sizeof(type)));
        HIP_CHECK(hipMalloc(&device_output, input_data.size() * sizeof(type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input_data.data(),
                input_data.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(shuffle_offset_kernel<block_size, type>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_data.data(), device_output,
                output_data.size() * sizeof(type),
                hipMemcpyDeviceToHost
            )
        );

        test_utils::assert_eq(input_data, output_data);

        HIP_CHECK(hipFree(device_input));
    }

}
