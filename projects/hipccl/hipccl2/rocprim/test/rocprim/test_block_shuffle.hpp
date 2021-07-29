// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

block_reduce_test_suite_type_def(suite_name, name_suffix)

typed_test_suite_def(suite_name, name_suffix, warp_params);

typed_test_def(suite_name, name_suffix, BlockOffset)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::input_type;
    static constexpr size_t block_size = TestFixture::block_size;
    static constexpr size_t size = block_size * 11;
    static constexpr size_t grid_size = size / block_size;
    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        int distance = (rand()%std::min<size_t>(10,block_size/2))-std::min<size_t>(10,block_size/2);
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value <<" & distance = "<<distance);
        // Generate data
        std::vector<type> input_data = test_utils::get_random_data<type>(size, -100, 100, seed_value);
        std::vector<type> output_data(input_data);

        // Preparing device
        type * device_input;
        type * device_output;

        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input_data.size() * sizeof(type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), input_data.size() * sizeof(type)));

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
            device_input, device_output, distance
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_data.data(), device_output,
                output_data.size() * sizeof(type),
                hipMemcpyDeviceToHost
            )
        );

        // Calculate expected results on host
        for(size_t block_index = 0; block_index < grid_size; block_index++)
        {
          for(size_t thread_index = 0; thread_index < block_size; thread_index++)
          {
            int offset = thread_index + distance;
            if((offset >= 0 ) && (offset < (int)block_size))
            {
              test_utils::assert_eq(input_data[block_index*block_size + offset],output_data[block_index*block_size + thread_index]);
            }
          }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));

    }

}

typed_test_def(suite_name, name_suffix, BlockRotate)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::input_type;
    static constexpr size_t block_size = TestFixture::block_size;
    static constexpr size_t size = block_size * 11;
    static constexpr size_t grid_size = size / block_size;
    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        int distance = (rand()%std::min<size_t>(5,block_size/2));
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value <<" & distance = "<<distance);
        // Generate data
        std::vector<type> input_data = test_utils::get_random_data<type>(size, -100, 100, seed_value);
        std::vector<type> output_data(input_data);

        // Preparing device
        type * device_input;
        type * device_output;

        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input_data.size() * sizeof(type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), input_data.size() * sizeof(type)));

        HIP_CHECK(
            hipMemcpy(
                device_input, input_data.data(),
                input_data.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );

        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(shuffle_rotate_kernel<block_size, type>),
            dim3(grid_size), dim3(block_size), 0, 0,
            device_input, device_output, distance
        );

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_data.data(), device_output,
                output_data.size() * sizeof(type),
                hipMemcpyDeviceToHost
            )
        );

        // Calculate expected results on host
        for(size_t block_index = 0; block_index < grid_size; block_index++)
        {
          for(size_t thread_index = 0; thread_index < block_size; thread_index++)
          {
            int offset = thread_index + distance;
            if (offset >= (int)block_size)
                offset -=      block_size;
            test_utils::assert_eq(input_data[block_index*block_size + offset],output_data[block_index*block_size + thread_index]);
          }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));

    }

}

typed_test_def(suite_name, name_suffix, BlockUp)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::input_type;
    static constexpr size_t block_size = TestFixture::block_size;
    static constexpr size_t size = block_size * 11;
    static constexpr size_t grid_size = size / block_size;
    static constexpr unsigned int ItemsPerThread = 128;
    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
        // Generate data
        std::vector<type> input_data = test_utils::get_random_data<type>(ItemsPerThread * size, -100, 100, seed_value);
        std::vector<type> output_data(input_data);

        std::vector<type*>  arr_input(size);
        std::vector<type*> arr_output(size);

        // Preparing device
        type * device_input;
        type * device_output;


        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input_data.size() * sizeof(type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), input_data.size() * sizeof(type)));



        HIP_CHECK(
            hipMemcpy(
                device_input, input_data.data(),
                input_data.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );


        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(shuffle_up_kernel<block_size, ItemsPerThread, type>),
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

        // Calculate expected results on host
        for(size_t block_index = 0; block_index < grid_size; block_index++)
        {
          for(size_t thread_index = 0; thread_index < block_size; thread_index++)
          {
            size_t start_offset = (block_index*block_size + thread_index)*ItemsPerThread;
            for(size_t item_index = 0; item_index < ItemsPerThread; item_index++)
            {
              if(thread_index + item_index>0)
              {
                  test_utils::assert_eq(input_data[start_offset + item_index-1],output_data[start_offset + item_index]);
              }
            }
          }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));

    }

}

typed_test_def(suite_name, name_suffix, BlockDown)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using type = typename TestFixture::input_type;
    static constexpr size_t block_size = TestFixture::block_size;
    static constexpr size_t size = block_size * 11;
    static constexpr size_t grid_size = size / block_size;
    static constexpr unsigned int ItemsPerThread = 128;
    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<type> input_data = test_utils::get_random_data<type>(ItemsPerThread * size, -100, 100, seed_value);
        std::vector<type> output_data(input_data);

        std::vector<type*>  arr_input(size);
        std::vector<type*> arr_output(size);

        // Preparing device
        type * device_input;
        type * device_output;


        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input), input_data.size() * sizeof(type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output), input_data.size() * sizeof(type)));



        HIP_CHECK(
            hipMemcpy(
                device_input, input_data.data(),
                input_data.size() * sizeof(type),
                hipMemcpyHostToDevice
            )
        );


        // Running kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(shuffle_down_kernel<block_size, ItemsPerThread, type>),
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

        // Calculate expected results on host
        for(size_t block_index = 0; block_index < grid_size; block_index++)
        {
          for(size_t thread_index = 0; thread_index < block_size; thread_index++)
          {
            size_t start_offset = (block_index*block_size + thread_index)*ItemsPerThread;
            for(size_t item_index = 0; item_index < ItemsPerThread; item_index++)
            {
              if((thread_index!=block_size-1)&&(item_index!=ItemsPerThread-1))
              {
                  test_utils::assert_eq(input_data[start_offset + item_index+1],output_data[start_offset + item_index]);
              }
            }
          }
        }

        HIP_CHECK(hipFree(device_input));
        HIP_CHECK(hipFree(device_output));

    }

}
