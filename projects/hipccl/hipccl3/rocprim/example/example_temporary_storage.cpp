// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <vector>
#include <random>

// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "example_utils.hpp"

// Example with allocating shared memory required as a temporary storage
// for a block-level parallel primitive inside a kernel
template<
    const unsigned int BlockSize,
    class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void example_shared_memory(const T *input, T *output)
{
    // Indexing for  this block
    unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;

    // Allocating storage in shared memory for the block
    using block_scan_type = rocprim::block_scan<T, BlockSize>;
    __shared__ typename block_scan_type::storage_type storage;

    // Variables required for performing a scan
    T input_value, output_value;

    // Execute inclusive plus scan
    input_value = input[index];

    block_scan_type()
        .inclusive_scan(
            input_value,
            output_value,
            storage,
            rocprim::plus<T>()
        );

    output[index] = output_value;
}

// Host function that runs example_shared_memory kernel
template<class T>
void run_example_shared_memory(size_t size)
{
    constexpr unsigned int block_size = 256;
    // Make sure size is a multiple of block_size
    unsigned int grid_size = (size + block_size - 1) / block_size;
    size = block_size * grid_size;

    // Generate input on host and copy it to device
    std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
    // Generating expected output for kernel
    std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size);
    // For reading device output
    std::vector<T> host_output(size);

    // Device memory allocation
    T * device_input;
    T * device_output;
    HIP_CHECK(hipMalloc(&device_input, host_input.size() * sizeof(typename decltype(host_input)::value_type)));
    HIP_CHECK(hipMalloc(&device_output, host_output.size() * sizeof(typename decltype(host_output)::value_type)));

    // Writing input data to device memory
    hip_write_device_memory<T>(device_input, host_input);

    // Launching kernel example_shared_memory
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(example_shared_memory<block_size, T>),
        dim3(grid_size), dim3(block_size),
        0, 0,
        device_input, device_output
    );

    // Reading output from device
    hip_read_device_memory<T>(host_output, device_output);

    // Validating output
    OUTPUT_VALIDATION_CHECK(
        validate_device_output(host_output, host_expected_output)
    );

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));

    std::cout << "Kernel run_example_shared_memory run was successful!" << std::endl;
}

// Kernel 2 - storage_type for one primitive union'ed with storage_type of other primitive
template<
    const unsigned int BlockSize,
    const unsigned int ItemsPerThread,
    class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void example_union_storage_types(const T *input, T *output)
{
    // Specialize primitives
    using block_scan_type = rocprim::block_scan<
        T, BlockSize, rocprim::block_scan_algorithm::using_warp_scan
    >;
    using block_load_type = rocprim::block_load<
        T, BlockSize, ItemsPerThread, rocprim::block_load_method::block_load_transpose
    >;
    using block_store_type = rocprim::block_store<
        T, BlockSize, ItemsPerThread, rocprim::block_store_method::block_store_transpose
    >;
    // Allocate storage in shared memory for both scan and sort operations

    __shared__ union
    {
        typename block_scan_type::storage_type scan;
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
    } storage;

    constexpr int items_per_block = BlockSize * ItemsPerThread;
    int block_offset = (hipBlockIdx_x * items_per_block);

    // Input/output array for block scan primitive
    T values[ItemsPerThread];

    // Loading data for this thread
    block_load_type().load(
        input + block_offset,
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
    block_store_type().store(
        output + block_offset,
        values,
        storage.store
    );
}

// Host function that runs example_union_storage_types kernel
template<class T>
void run_example_union_storage_types(size_t size)
{
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    // Make sure size is a multiple of block_size
    auto grid_size = (size + block_size - 1) / block_size;
    size = block_size * grid_size;

    // Generate input on host and copy it to device
    std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
    // Generating expected output for kernel
    std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size, items_per_thread);
    // For reading device output
    std::vector<T> host_output(size);

    // Device memory allocation
    T * device_input;
    T * device_output;
    HIP_CHECK(hipMalloc(&device_input, host_input.size() * sizeof(typename decltype(host_input)::value_type)));
    HIP_CHECK(hipMalloc(&device_output, host_output.size() * sizeof(typename decltype(host_output)::value_type)));

    // Writing input data to device memory
    hip_write_device_memory<T>(device_input, host_input);

    // Launching kernel example_union_storage_types
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(example_union_storage_types<block_size, items_per_thread, int>),
        dim3(grid_size), dim3(block_size),
        0, 0,
        device_input, device_output
    );

    // Reading output from device
    hip_read_device_memory<T>(host_output, device_output);

    // Validating output
    OUTPUT_VALIDATION_CHECK(
        validate_device_output(host_output, host_expected_output)
    );

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));

    std::cout << "Kernel run_example_union_storage_types run was successful!" << std::endl;
}

// Kernel 3 - Allocating shared memory in runtime
template<
    const unsigned int BlockSize,
    class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void example_dynamic_shared_memory(const T *input, T *output)
{
    // Indexing for  this block
    unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;

    // Initialize primitives
    using block_scan_type = rocprim::block_scan<T, BlockSize>;

    // Allocation done in runtime, for more information please visit:
    // https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/2_Cookbook/6_dynamic_shared
    HIP_DYNAMIC_SHARED(typename block_scan_type::storage_type, primitive_storage);

    // Variables required for performing a scan
    T input_value, output_value;

    // execute inclusive scan
    input_value = input[index];
    block_scan_type()
        .inclusive_scan(
            input_value, output_value,
            *primitive_storage,
            rocprim::plus<T>()
        );

    output[index] = output_value;
}

// Host function that runs example_dynamic_shared_memory kernel
template<class T>
void run_example_dynamic_shared_memory(size_t size)
{
    constexpr unsigned int block_size = 256;
    // Make sure size is a multiple of block_size
    auto grid_size = (size + block_size - 1) / block_size;
    size = block_size * grid_size;

    // Generate input on host and copy it to device
    std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
    // Generating expected output for kernel
    std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size);
    // For reading device output
    std::vector<T> host_output(size);

    // Device memory allocation
    T * device_input;
    T * device_output;
    HIP_CHECK(hipMalloc(&device_input, host_input.size() * sizeof(typename decltype(host_input)::value_type)));
    HIP_CHECK(hipMalloc(&device_output, host_output.size() * sizeof(typename decltype(host_output)::value_type)));

    // Writing input data to device memory
    hip_write_device_memory<T>(device_input, host_input);

    // Launching kernel example_shared_memory
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(example_dynamic_shared_memory<block_size, T>),
        dim3(grid_size), dim3(block_size),
        sizeof(typename rocprim::block_scan<T, block_size>::storage_type), 0,
        device_input, device_output
    );

    // Reading output from device
    hip_read_device_memory<T>(host_output, device_output);

    // Validating output
    OUTPUT_VALIDATION_CHECK(
        validate_device_output(host_output, host_expected_output)
    );

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));

    std::cout << "Kernel run_example_dynamic_shared_memory run was successful!" << std::endl;
}

// Kernel 4 - Using global memory for storage
template<
    const unsigned int BlockSize,
    class T
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void example_global_memory_storage(
        const T *input,
        T *output,
        typename rocprim::block_scan<T, BlockSize>::storage_type *global_storage)
{
    // Indexing for  this block
    unsigned int index = (hipBlockIdx_x * BlockSize) + hipThreadIdx_x;
    // specialize block_scan for type T and block of 256 threads
    using block_scan_type = rocprim::block_scan<T, BlockSize>;
    // Variables required for performing a scan
    T input_value, output_value;

    // execute inclusive scan
    input_value = input[index];

    block_scan_type()
        .inclusive_scan(
            input_value, output_value,
            global_storage[hipBlockIdx_x],
            rocprim::plus<T>()
        );

    output[index] = output_value;
}

// Host function that runs example_global_memory_storage kernel
template<class T>
void run_example_global_memory_storage(size_t size)
{
    constexpr unsigned int block_size = 256;
    // Make sure size is a multiple of block_size
    auto grid_size = (size + block_size - 1) / block_size;
    size = block_size * grid_size;

    // Generate input on host and copy it to device
    std::vector<T> host_input = get_random_data<T>(size, 0, 1000);
    // Generating expected output for kernel
    std::vector<T> host_expected_output = get_expected_output<T>(host_input, block_size);
    // For reading device output
    std::vector<T> host_output(size);

    // Device memory allocation
    T * device_input;
    T * device_output;
    HIP_CHECK(hipMalloc(&device_input, host_input.size() * sizeof(typename decltype(host_input)::value_type)));
    HIP_CHECK(hipMalloc(&device_output, host_output.size() * sizeof(typename decltype(host_output)::value_type)));

    // Writing input data to device memory
    hip_write_device_memory<T>(device_input, host_input);

    // Allocating temporary storage in global memory
    using storage_type = typename rocprim::block_scan<T, block_size>::storage_type;
    storage_type *global_storage;
    HIP_CHECK(hipMalloc(&global_storage, (grid_size * sizeof(storage_type))));

    // Launching kernel example_shared_memory
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(example_global_memory_storage<block_size, T>),
        dim3(grid_size), dim3(block_size),
        0, 0,
        device_input, device_output, global_storage
    );

    // Reading output from device
    hip_read_device_memory<T>(host_output, device_output);

    // Validating output
    OUTPUT_VALIDATION_CHECK(
        validate_device_output(host_output, host_expected_output)
    );

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(global_storage));

    std::cout << "Kernel run_example_global_memory_storage run was successful!" << std::endl;
}

int main()
{
    // Initializing HIP device
    hipDeviceProp_t device_properties;
    HIP_CHECK(hipGetDeviceProperties(&device_properties, 0));

    // Show device info
    printf("Selected device:         %s  \n", device_properties.name              );
    printf("Available global memory: %lu \n", device_properties.totalGlobalMem    );
    printf("Shared memory per block: %lu \n", device_properties.sharedMemPerBlock );
    printf("Warp size:               %d  \n", device_properties.warpSize          );
    printf("Max threads per block:   %d  \n", device_properties.maxThreadsPerBlock);

    // Running kernels
    run_example_global_memory_storage<int>(1024);
    run_example_shared_memory<int>(1024);
    run_example_union_storage_types<int>(1024);
    run_example_dynamic_shared_memory<int>(1024);
}
