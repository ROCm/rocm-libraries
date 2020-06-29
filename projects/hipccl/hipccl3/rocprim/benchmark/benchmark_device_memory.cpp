// MIT License
//
// Copyright (c) 2018-2019 Advanced Micro Devices, Inc. All rights reserved.
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
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>

// Google Benchmark
#include "benchmark/benchmark.h"
// CmdParser
#include "cmdparser.hpp"
// rocPRIM
#include <rocprim/rocprim.hpp>

#include "benchmark_utils.hpp"

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

enum memory_operation_method
{
    memcpy,
    block_primitives_transpose,
    striped,
    vectorized,
    block_primitive_direct,
};

enum kernel_operation
{
    no_operation,
    block_scan,
    custom_operation,
    atomics_no_collision,
    atomics_inter_block_collision,
    atomics_inter_warp_collision,
};

template<
    kernel_operation Operation,
    class T,
    unsigned int ItemsPerThread,
    unsigned int BlockSize = 0
>
struct operation;

// no operation
template<class T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<no_operation, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE inline
    void operator()(T (&)[ItemsPerThread], void* = nullptr, unsigned int = 0, T* = nullptr)
    {
        // No operation
    }
};

#define repeats 30

// custom operation
template<class T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<custom_operation, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE inline
    void operator()(T (&input)[ItemsPerThread],
                    void* shared_storage = nullptr, unsigned int shared_storage_size = 0,
                    T* global_mem_output = nullptr)
    {
        (void) shared_storage;
        (void) shared_storage_size;
        (void) global_mem_output;
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            input[i] = input[i] + 666;
            #pragma unroll
            for(unsigned int j = 0; j < repeats; j++)
            {
                input[i] = input[i] * (input[j % ItemsPerThread]);
            }
        }
    }
};

// block scan
template<class T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<block_scan, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE inline
    void operator()(T (&input)[ItemsPerThread],
                    void* shared_storage = nullptr, unsigned int shared_storage_size = 0,
                    T* global_mem_output = nullptr)
    {
        (void) global_mem_output;
        using block_scan_type = typename rocprim::block_scan<
            T, BlockSize, rocprim::block_scan_algorithm::using_warp_scan>;

        block_scan_type bscan;

        // when using vectorized or striped functions
        // NOTE: This is not safe but it is the easiest way to prevent code repetition
        if(shared_storage == nullptr ||
           shared_storage_size < sizeof(typename block_scan_type::storage_type))
        {
            __shared__ typename block_scan_type::storage_type storage;
            shared_storage = &storage;
        }

        bscan.inclusive_scan(
            input, input,
            *(reinterpret_cast<typename block_scan_type::storage_type*>(shared_storage))
        );
        __syncthreads();
    }
};

// atomics_no_collision
template<class T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<atomics_no_collision, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE inline
    void operator()(T (&input)[ItemsPerThread],
                    void* shared_storage = nullptr, unsigned int shared_storage_size = 0,
                    T* global_mem_output = nullptr)
    {
        (void) shared_storage;
        (void) shared_storage_size;
        (void) input;
        unsigned int index = hipThreadIdx_x * ItemsPerThread +
                             hipBlockIdx_x * hipBlockDim_x * ItemsPerThread;
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            atomicAdd(&global_mem_output[index + i], T(666));
        }
    }
};

// atomics_inter_block_collision
template<class T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<atomics_inter_warp_collision, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE inline
    void operator()(T (&input)[ItemsPerThread],
                    void* shared_storage = nullptr, unsigned int shared_storage_size = 0,
                    T* global_mem_output = nullptr)
    {
        (void) shared_storage;
        (void) shared_storage_size;
        (void) input;
        unsigned int index = (hipThreadIdx_x % warpSize) * ItemsPerThread +
                             hipBlockIdx_x * hipBlockDim_x * ItemsPerThread;
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            atomicAdd(&global_mem_output[index + i], T(666));
        }
    }
};

// atomics_inter_block_collision
template<class T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<atomics_inter_block_collision, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE inline
    void operator()(T (&input)[ItemsPerThread],
                    void* shared_storage = nullptr, unsigned int shared_storage_size = 0,
                    T* global_mem_output = nullptr)
    {
        (void) shared_storage;
        (void) shared_storage_size;
        (void) input;
        unsigned int index = hipThreadIdx_x * ItemsPerThread;
        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            atomicAdd(&global_mem_output[index + i], T(666));
        }
    }
};

// block_primitive_direct method base kernel
template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    memory_operation_method MemOp,
    class CustomOp =
        typename operation<no_operation, T, ItemsPerThread>::value_type,
    typename std::enable_if<MemOp == block_primitive_direct, int>::type = 0
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void operation_kernel(T* input, T* output, CustomOp op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using block_load_type = typename rocprim::block_load<
        T, BlockSize, ItemsPerThread, rocprim::block_load_method::block_load_direct>;
    using block_store_type = typename rocprim::block_store<
        T, BlockSize, ItemsPerThread, rocprim::block_store_method::block_store_direct>;

    block_load_type load;
    block_store_type store;

    __shared__ union
    {
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
    } storage;

    int offset = hipBlockIdx_x * items_per_block;

    T items[ItemsPerThread];
    load.load(input + offset, items, storage.load);
    __syncthreads();
    op(items, &storage, sizeof(storage), output);
    store.store(output + offset, items, storage.store);
}

// vectorized method base kernel
template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    memory_operation_method MemOp,
    class CustomOp =
        typename operation<no_operation, T, ItemsPerThread>::value_type,
    typename std::enable_if<MemOp == vectorized, int>::type = 0
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void operation_kernel(T* input, T* output, CustomOp op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    int offset = hipBlockIdx_x * items_per_block;
    T items[ItemsPerThread];

    rocprim::block_load_direct_blocked_vectorized<T, T, ItemsPerThread>
        (hipThreadIdx_x, input + offset, items);
    __syncthreads();

    op(items, nullptr, 0, output);

    rocprim::block_store_direct_blocked_vectorized<T, T, ItemsPerThread>
        (hipThreadIdx_x, output + offset, items);
}

// striped method base kernel
template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    memory_operation_method MemOp,
    class CustomOp =
        typename operation<no_operation, T, ItemsPerThread>::value_type,
    typename std::enable_if<MemOp == striped, int>::type = 0
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void operation_kernel(T* input, T* output, CustomOp op)
{
    const unsigned int lid = hipThreadIdx_x;
    const unsigned int block_offset = hipBlockIdx_x * ItemsPerThread * BlockSize;
    T items[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + block_offset, items);
    op(items, nullptr, 0, output);
    rocprim::block_store_direct_striped<BlockSize>(lid, output + block_offset, items);
}

// block_primitives_transpose method base kernel
template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    memory_operation_method MemOp,
    class CustomOp =
        typename operation<no_operation, T, ItemsPerThread>::value_type,
    typename std::enable_if<MemOp == block_primitives_transpose, int>::type = 0
>
__global__
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void operation_kernel(T* input, T* output, CustomOp op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using block_load_type = typename rocprim::block_load<
        T, BlockSize, ItemsPerThread, rocprim::block_load_method::block_load_transpose>;
    using block_store_type = typename rocprim::block_store<
        T, BlockSize, ItemsPerThread, rocprim::block_store_method::block_store_transpose>;

    block_load_type load;
    block_store_type store;

    __shared__ union
    {
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
    } storage;

    int offset = hipBlockIdx_x * items_per_block;

    T items[ItemsPerThread];
    load.load(input + offset, items, storage.load);
    __syncthreads();
    op(items, &storage, sizeof(storage), output);
    store.store(output + offset, items, storage.store);
}

template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    memory_operation_method MemOp,
    kernel_operation KernelOp = no_operation
>
void run_benchmark(benchmark::State& state,
                   size_t size,
                   const hipStream_t stream)
{
    const size_t grid_size = size / (BlockSize * ItemsPerThread);
    std::vector<T> input;
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, (T)-1000, (T)+1000);
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    HIP_CHECK(
        hipMemcpy(
            d_input, input.data(),
            size * sizeof(T),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    operation<KernelOp, T, ItemsPerThread, BlockSize> selected_operation;

    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(operation_kernel<T, BlockSize, ItemsPerThread, MemOp>),
            dim3(grid_size), dim3(BlockSize), 0, stream,
            d_input, d_output, selected_operation
        );
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(operation_kernel<T, BlockSize, ItemsPerThread, MemOp>),
                dim3(grid_size), dim3(BlockSize), 0, stream,
                d_input, d_output, selected_operation
            );
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

template<class T>
void run_benchmark_memcpy(benchmark::State& state,
                           size_t size,
                           const hipStream_t stream)
{
    std::vector<T> input;
    if(std::is_floating_point<T>::value)
    {
        input = get_random_data<T>(size, (T)-1000, (T)+1000);
    }
    else
    {
        input = get_random_data<T>(
            size,
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()
        );
    }
    T * d_input;
    T * d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(T)));
    // Warm-up
    for(size_t i = 0; i < 10; i++)
    {
        HIP_CHECK(hipMemcpy(d_output, d_input, size * sizeof(T), hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int batch_size = 10;
    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < batch_size; i++)
        {
            HIP_CHECK(hipMemcpy(d_output, d_input, size * sizeof(T), hipMemcpyDeviceToDevice));
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

#define CREATE_BENCHMARK(METHOD, OPERATION, T, SIZE, BLOCK_SIZE, IPT) \
benchmark::RegisterBenchmark( \
    (#METHOD "_" #OPERATION "<" #T "," #SIZE ",BS:" #BLOCK_SIZE ",IPT:" #IPT ">"), \
    run_benchmark<T, BLOCK_SIZE, IPT, METHOD, OPERATION \
    >, SIZE, stream \
)

#define CREATE_BENCHMARK_MEMCPY(T, SIZE) \
benchmark::RegisterBenchmark( \
    ("Memcpy<" #T "," #SIZE">"), run_benchmark_memcpy<T>, SIZE, stream \
)

template<class T>
constexpr unsigned int megabytes(unsigned int size)
{
    return(size * (1024 * 1024 / sizeof(T)));
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "Device name: " << devProp.name << std::endl;
    std::cout << "L2 Cache size: " << devProp.l2CacheSize << std::endl;
    std::cout << "Shared memory per block: " << devProp.sharedMemPerBlock << std::endl;

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks =
    {
        // simple memory copy not running kernel
        CREATE_BENCHMARK_MEMCPY(int, megabytes<int>(128)),

        // simple memory copy
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 1024, 4),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, megabytes<int>(128), 1024, 8),

        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 4),

        // simple memory copy using vector type
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 128, 1),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 128, 2),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 128, 4),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 128, 8),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 128, 16),

        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 1024, 4),
        CREATE_BENCHMARK(vectorized, no_operation, int, megabytes<int>(128), 1024, 8),

        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 1),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 2),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 4),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 8),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 16),

        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 1),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 2),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 4),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 8),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 16),

        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 1),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 2),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 4),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 8),

        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 1),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 2),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 4),
        CREATE_BENCHMARK(vectorized, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 8),

        // simple memory copy using striped
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 128, 1),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 128, 2),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 128, 4),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 128, 8),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 128, 16),

        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 1024, 4),
        CREATE_BENCHMARK(striped, no_operation, int, megabytes<int>(128), 1024, 8),

        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 1),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 2),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 4),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 8),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 128, 16),

        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 1),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 2),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 4),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 8),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 256, 16),

        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 1),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 2),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 4),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 512, 8),

        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 1),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 2),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 4),
        CREATE_BENCHMARK(striped, no_operation, uint64_t, megabytes<uint64_t>(128), 1024, 8),

        // block_scan
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 128, 16),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 128, 32),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 1024, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, megabytes<int>(128), 1024, 8),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 1024, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, megabytes<int>(128), 1024, 8),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, megabytes<uint64_t>(128), 1024, 4),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, megabytes<uint64_t>(128), 1024, 4),

        // vectorized - block_scan
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 128, 1),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 128, 2),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 128, 4),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 128, 8),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 128, 16),

        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 1024, 4),
        CREATE_BENCHMARK(vectorized, block_scan, int, megabytes<int>(128), 1024, 8),

        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 128, 1),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 128, 2),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 128, 4),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 128, 8),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 128, 16),

        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 256, 1),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 256, 2),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 256, 4),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 256, 8),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 256, 16),

        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 512, 1),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 512, 2),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 512, 4),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 512, 8),

        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 1024, 1),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 1024, 2),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 1024, 4),
        CREATE_BENCHMARK(vectorized, block_scan, float, megabytes<float>(128), 1024, 8),

        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 128, 1),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 128, 2),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 128, 4),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 128, 8),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 128, 16),

        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 256, 1),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 256, 2),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 256, 4),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 256, 8),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 256, 16),

        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 512, 1),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 512, 2),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 512, 4),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 512, 8),

        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 1024, 1),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 1024, 2),
        CREATE_BENCHMARK(vectorized, block_scan, double, megabytes<double>(128), 1024, 4),

        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 1),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 2),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 4),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 8),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 128, 16),

        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 1),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 2),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 4),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 8),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 256, 16),

        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 512, 1),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 512, 2),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 512, 4),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 512, 8),

        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 1024, 1),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 1024, 2),
        CREATE_BENCHMARK(vectorized, block_scan, uint64_t, megabytes<uint64_t>(128), 1024, 4),

        // custom_op
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, megabytes<int>(128), 1024, 4),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, megabytes<float>(128), 1024, 4),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, megabytes<double>(128), 1024, 2),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, megabytes<uint64_t>(128), 1024, 2),

        // block_primitives_transpose - atomics no collision
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 1024, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, megabytes<int>(128), 1024, 8),

        // block_primitives_transpose - atomics inter block collision
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 1024, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, megabytes<int>(128), 1024, 8),

        // block_primitives_transpose - atomics inter warp collision
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 128, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 128, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 128, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 128, 8),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 128, 16),

        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 256, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 256, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 256, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 256, 8),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 256, 16),

        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 512, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 512, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 512, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 512, 8),

        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 1024, 1),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 1024, 2),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 1024, 4),
        CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, megabytes<int>(128), 1024, 8),

    };

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
