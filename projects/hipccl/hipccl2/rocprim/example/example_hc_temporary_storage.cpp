// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <vector>

// rocPRIM HC API
#include <rocprim/rocprim_hc.hpp>

#include "example_utils.hpp"

template<class T>
void run_example_shared_memory(size_t size, hc::accelerator_view acc_view)
{
    constexpr unsigned int block_size = 256;
    // Make sure size is a multiple of block_size
    unsigned int number_of_tiles = (size + block_size - 1) / block_size;
    size = block_size * number_of_tiles;

    // Host memory allocation
    std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
    std::vector<T> host_output(size);
    std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size);

    // Device memory allocation
    hc::array_view<T> device_input(host_input.size(), host_input.data());
    hc::array_view<T> device_output(host_output.size(), host_output.data());

    // Allocating shared memory inside the kernel for each Primitive
    hc::parallel_for_each(
        acc_view,
        hc::extent<1>(size).tile(block_size),
        [=](hc::tiled_index<1> thread_id) [[hc]]
        {
            int index = thread_id.global[0];
            using block_scan_type = rocprim::block_scan<T, block_size>;

            // Allocate storage in shared memory for the block
            tile_static typename block_scan_type::storage_type temporary_storage;

            T input_value = device_input[index];
            T output_value;

            block_scan_type()
                .inclusive_scan(
                    input_value,
                    output_value,
                    temporary_storage,
                    rocprim::plus<T>()
                );

            device_output[index] = output_value;
        }
    );

    // Getting output data from device
    device_output.synchronize();

    // Validating device output
    OUTPUT_VALIDATION_CHECK(
        validate_device_output(host_output, host_expected_output)
    );

    std::cout << "Shared memory Kernel run was successful!" << std::endl;
}

template<class T>
void run_example_union_storage_types(size_t size, hc::accelerator_view acc_view)
{
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    // Make sure size is a multiple of block_size
    unsigned int number_of_tiles = (size + block_size - 1) / block_size;
    size = block_size * number_of_tiles;

    // Host memory allocation
    std::vector<T> host_input = get_random_data<int>(size, 0, 1000);
    std::vector<T> host_output(size);
    std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size, items_per_thread);

    // Device memory allocation
    hc::array_view<T> device_input(host_input.size(), host_input.data());
    hc::array_view<T> device_output(host_output.size(), host_output.data());

    // Getting output data from device
    device_output.synchronize();

    // storage_type for one primitive union'ed with storage_type of other primitive
    hc::parallel_for_each(
        acc_view,
        hc::extent<1>(size).tile(block_size),
        [=](hc::tiled_index<1> thread_id) [[hc]]
        {
            // Specialize primitives
            using block_scan_type = rocprim::block_scan<
                T, block_size, rocprim::block_scan_algorithm::using_warp_scan
            >;
            using block_load_type = rocprim::block_load<
                T, block_size, items_per_thread, rocprim::block_load_method::block_load_transpose
            >;
            using block_store_type = rocprim::block_store<
                T, block_size, items_per_thread, rocprim::block_store_method::block_store_transpose
            >;

            // Allocate storage in shared memory for load,store and scan operations
            tile_static union
            {
                typename block_scan_type::storage_type scan;
                typename block_load_type::storage_type load;
                typename block_store_type::storage_type store;
            } storage;

            constexpr int items_per_block = block_size * items_per_thread;
            int block_offset = (thread_id.tile[0] * items_per_block);

            // Input/output array for block scan primitive
            T values[items_per_thread];

            // Loading data for this thread
            block_load_type()
                .load(
                    device_input.data() + block_offset,
                    values,
                    storage.load
                );
            rocprim::syncthreads();

            // Perform scan
            block_scan_type()
                .inclusive_scan(
                    values, // as input
                    values, // as output
                    storage.scan,
                    rocprim::plus<T>()
                );
            rocprim::syncthreads();

            // Save elements to output
            block_store_type()
                .store(
                    device_output.data() + block_offset,
                    values,
                    storage.store
                );
        }
    );

    // Getting output data from device
    device_output.synchronize();

    // Validating device output
    OUTPUT_VALIDATION_CHECK(
        validate_device_output(host_output, host_expected_output)
    );

    std::cout << "Union shared memory Kernel run was successful!" << std::endl;
}

template<class T>
void run_example_dynamic_shared_memory(size_t size, hc::accelerator_view acc_view)
{
    constexpr unsigned int block_size = 256;
    // Make sure size is a multiple of block_size
    unsigned int number_of_tiles = (size + block_size - 1) / block_size;
    size = block_size * number_of_tiles;

    // Host memory allocation
    std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
    std::vector<T> host_output(size);
    std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size);

    // Device memory allocation
    hc::array_view<T> device_input(host_input.size(), host_input.data());
    hc::array_view<T> device_output(host_output.size(), host_output.data());

    // Allocating shared memory inside the kernel for each Primitive
    using storage_type = typename rocprim::block_scan<T, block_size>::storage_type;

    hc::parallel_for_each(
        acc_view,
        hc::extent<1>(size)
            .tile_with_dynamic(block_size, sizeof(storage_type)),
        [=](hc::tiled_index<1> thread_id) [[hc]]
        {
            int index = thread_id.global[0];
            using block_scan_type = rocprim::block_scan<T, block_size>;

            // Allocate storage in shared memory for the block
            auto *temporary_storage = static_cast<storage_type*>(hc::get_dynamic_group_segment_base_pointer());

            T input_value = device_input[index];
            T output_value;

            block_scan_type()
                .inclusive_scan(
                    input_value,
                    output_value,
                    *temporary_storage,
                    rocprim::plus<T>()
                );

            device_output[index] = output_value;
        }
    );

    // Getting output data from device
    device_output.synchronize();

    // Validating device output
    OUTPUT_VALIDATION_CHECK(
        validate_device_output(host_output, host_expected_output)
    );

    std::cout << "Dynamic shared memory Kernel run was successful!" << std::endl;
}

template<class T>
void run_example_global_memory_storage(size_t size, hc::accelerator_view acc_view)
{
    constexpr unsigned int block_size = 256;
    // Make sure size is a multiple of block_size
    unsigned int number_of_tiles = (size + block_size - 1) / block_size;
    size = block_size * number_of_tiles;

    // Host memory allocation
    std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
    std::vector<T> host_output(size);
    std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size);

    // Device memory allocation
    hc::array_view<T> device_input(host_input.size(), host_input.data());
    hc::array_view<T> device_output(host_output.size(), host_output.data());

    // Using global memory as temporary storage
    using storage_type = typename rocprim::block_scan<T, block_size>::storage_type;
    std::vector<storage_type> host_global_storage(number_of_tiles);

    hc::array<storage_type, 1>
        device_global_storage(
            hc::extent<1>(host_global_storage.size()),
            host_global_storage.begin(),
            acc_view
        );

    hc::parallel_for_each(
        acc_view,
        hc::extent<1>(size).tile(block_size),
        [=, &device_global_storage](hc::tiled_index<1> thread_id) [[hc]]
        {
            int index = thread_id.global[0];

            // Specialising primitives
            using block_scan_type = rocprim::block_scan<T, block_size>;

            T input_value = device_input[index];
            T output_value;

            block_scan_type()
                .inclusive_scan(
                    input_value, output_value,
                    device_global_storage.data()[thread_id.tile[0]],
                    rocprim::plus<T>()
                );

            device_output[index] = output_value;
        }
    );

    // Getting output data from device
    device_output.synchronize();

    // Validating device output
    OUTPUT_VALIDATION_CHECK(
        validate_device_output(host_output, host_expected_output)
    );

    std::cout << "Global temporary memory Kernel run was successful!" << std::endl;
}

int main()
{
    hc::accelerator hc_accelerator;
    hc::accelerator_view acc_view = hc_accelerator.create_view();

    // Running each example kernel
    run_example_shared_memory<int>(1024, acc_view);
    run_example_union_storage_types<int>(1024, acc_view);
    run_example_dynamic_shared_memory<int>(1024, acc_view);
    run_example_global_memory_storage<int>(1024, acc_view);

}

