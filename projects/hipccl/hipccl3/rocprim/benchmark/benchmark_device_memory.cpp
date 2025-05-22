// MIT License
//
// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_utils.hpp"

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

// rocPRIM
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_scan.hpp>
#include <rocprim/block/block_store.hpp>
#include <rocprim/config.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/types.hpp>

#include <cstddef>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <vector>

enum memory_operation_method
{
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

template<kernel_operation Operation,
         typename T,
         unsigned int ItemsPerThread,
         unsigned int BlockSize = 0>
struct operation;

// no operation
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<no_operation, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE
    inline void
        operator()(T (&)[ItemsPerThread], void* = nullptr, unsigned int = 0, T* = nullptr) const
    {
        // No operation
    }
};

#define repeats 30

// custom operation
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<custom_operation, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE
    inline void
        operator()(T (&input)[ItemsPerThread],
                   void*        shared_storage      = nullptr,
                   unsigned int shared_storage_size = 0,
                   T*           global_mem_output   = nullptr) const
    {
        (void)shared_storage;
        (void)shared_storage_size;
        (void)global_mem_output;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            input[i] = input[i] + 666;
            ROCPRIM_UNROLL
            for(unsigned int j = 0; j < repeats; ++j)
            {
                input[i] = input[i] * (input[j % ItemsPerThread]);
            }
        }
    }
};

// block scan
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<block_scan, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE
    inline void
        operator()(T (&input)[ItemsPerThread],
                   void*        shared_storage      = nullptr,
                   unsigned int shared_storage_size = 0,
                   T*           global_mem_output   = nullptr) const
    {
        (void)global_mem_output;
        using block_scan_type = typename rocprim::
            block_scan<T, BlockSize, rocprim::block_scan_algorithm::using_warp_scan>;

        block_scan_type bscan;

        // when using vectorized or striped functions
        // NOTE: This is not safe but it is the easiest way to prevent code repetition
        if(shared_storage == nullptr
           || shared_storage_size < sizeof(typename block_scan_type::storage_type))
        {
            __shared__ typename block_scan_type::storage_type storage;
            shared_storage = &storage;
        }

        bscan.inclusive_scan(
            input,
            input,
            *(reinterpret_cast<typename block_scan_type::storage_type*>(shared_storage)));
        __syncthreads();
    }
};

// atomics_no_collision
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<atomics_no_collision, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE
    inline void
        operator()(T (&input)[ItemsPerThread],
                   void*        shared_storage      = nullptr,
                   unsigned int shared_storage_size = 0,
                   T*           global_mem_output   = nullptr)
    {
        (void)shared_storage;
        (void)shared_storage_size;
        (void)input;
        unsigned int index
            = threadIdx.x * ItemsPerThread + blockIdx.x * blockDim.x * ItemsPerThread;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            atomicAdd(&global_mem_output[index + i], T(666));
        }
    }
};

// atomics_inter_block_collision
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<atomics_inter_warp_collision, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE
    inline void
        operator()(T (&input)[ItemsPerThread],
                   void*        shared_storage      = nullptr,
                   unsigned int shared_storage_size = 0,
                   T*           global_mem_output   = nullptr)
    {
        (void)shared_storage;
        (void)shared_storage_size;
        (void)input;
        unsigned int index = (threadIdx.x % rocprim::arch::wavefront::min_size()) * ItemsPerThread
                             + blockIdx.x * blockDim.x * ItemsPerThread;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            atomicAdd(&global_mem_output[index + i], T(666));
        }
    }
};

// atomics_inter_block_collision
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<atomics_inter_block_collision, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE
    inline void
        operator()(T (&input)[ItemsPerThread],
                   void*        shared_storage      = nullptr,
                   unsigned int shared_storage_size = 0,
                   T*           global_mem_output   = nullptr)
    {
        (void)shared_storage;
        (void)shared_storage_size;
        (void)input;
        unsigned int index = threadIdx.x * ItemsPerThread;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i)
        {
            atomicAdd(&global_mem_output[index + i], T(666));
        }
    }
};

// block_primitive_direct method base kernel
template<typename T,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         memory_operation_method MemOp,
         typename CustomOp = typename operation<no_operation, T, ItemsPerThread>::value_type,
         typename std::enable_if<MemOp == block_primitive_direct, int>::type = 0>
__global__ __launch_bounds__(BlockSize)
void operation_kernel(T* input, T* output, CustomOp op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using block_load_type = typename rocprim::
        block_load<T, BlockSize, ItemsPerThread, rocprim::block_load_method::block_load_direct>;
    using block_store_type = typename rocprim::
        block_store<T, BlockSize, ItemsPerThread, rocprim::block_store_method::block_store_direct>;

    block_load_type  load;
    block_store_type store;

    __shared__ union
    {
        typename block_load_type::storage_type  load;
        typename block_store_type::storage_type store;
    } storage;

    int offset = blockIdx.x * items_per_block;

    T items[ItemsPerThread];
    load.load(input + offset, items, storage.load);
    __syncthreads();
    op(items, &storage, sizeof(storage), output);
    store.store(output + offset, items, storage.store);
}

// vectorized method base kernel
template<typename T,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         memory_operation_method MemOp,
         typename CustomOp = typename operation<no_operation, T, ItemsPerThread>::value_type,
         typename std::enable_if<MemOp == vectorized, int>::type = 0>
__global__ __launch_bounds__(BlockSize)
void operation_kernel(T* input, T* output, CustomOp op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    int                    offset          = blockIdx.x * items_per_block;
    T                      items[ItemsPerThread];

    rocprim::block_load_direct_blocked_vectorized<T, T, ItemsPerThread>(threadIdx.x,
                                                                        input + offset,
                                                                        items);
    __syncthreads();

    op(items, nullptr, 0, output);

    rocprim::block_store_direct_blocked_vectorized<T, T, ItemsPerThread>(threadIdx.x,
                                                                         output + offset,
                                                                         items);
}

// striped method base kernel
template<typename T,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         memory_operation_method MemOp,
         typename CustomOp = typename operation<no_operation, T, ItemsPerThread>::value_type,
         typename std::enable_if<MemOp == striped, int>::type = 0>
__global__ __launch_bounds__(BlockSize)
void operation_kernel(T* input, T* output, CustomOp op)
{
    const unsigned int lid          = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerThread * BlockSize;
    T                  items[ItemsPerThread];
    rocprim::block_load_direct_striped<BlockSize>(lid, input + block_offset, items);
    op(items, nullptr, 0, output);
    rocprim::block_store_direct_striped<BlockSize>(lid, output + block_offset, items);
}

// block_primitives_transpose method base kernel
template<typename T,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         memory_operation_method MemOp,
         typename CustomOp = typename operation<no_operation, T, ItemsPerThread>::value_type,
         typename std::enable_if<MemOp == block_primitives_transpose, int>::type = 0>
__global__ __launch_bounds__(BlockSize)
void operation_kernel(T* input, T* output, CustomOp op)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using block_load_type = typename rocprim::
        block_load<T, BlockSize, ItemsPerThread, rocprim::block_load_method::block_load_transpose>;
    using block_store_type =
        typename rocprim::block_store<T,
                                      BlockSize,
                                      ItemsPerThread,
                                      rocprim::block_store_method::block_store_transpose>;

    block_load_type  load;
    block_store_type store;

    __shared__ union
    {
        typename block_load_type::storage_type  load;
        typename block_store_type::storage_type store;
    } storage;

    int offset = blockIdx.x * items_per_block;

    T items[ItemsPerThread];
    load.load(input + offset, items, storage.load);
    __syncthreads();
    op(items, &storage, sizeof(storage), output);
    store.store(output + offset, items, storage.store);
}

template<typename T,
         unsigned int            BlockSize,
         unsigned int            ItemsPerThread,
         memory_operation_method MemOp,
         kernel_operation        KernelOp = no_operation>
void run_benchmark(benchmark_utils::state&& state)
{
    const auto& stream = state.stream;
    const auto& bytes  = state.bytes;
    const auto& seed   = state.seed;

    const size_t size = bytes / sizeof(T);

    const size_t   grid_size = size / (BlockSize * ItemsPerThread);
    std::vector<T> input     = get_random_data<T>(size,
                                              common::generate_limits<T>::min(),
                                              common::generate_limits<T>::max(),
                                              seed.get_0());

    common::device_ptr<T> d_input(input);
    common::device_ptr<T> d_output(size);
    HIP_CHECK(hipDeviceSynchronize());

    operation<KernelOp, T, ItemsPerThread, BlockSize> selected_operation;

    state.run(
        [&]
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(operation_kernel<T, BlockSize, ItemsPerThread, MemOp>),
                dim3(grid_size),
                dim3(BlockSize),
                0,
                stream,
                d_input.get(),
                d_output.get(),
                selected_operation);
        });

    state.set_throughput(size, sizeof(T));
}

template<typename T>
void run_benchmark_memcpy(benchmark_utils::state&& state)
{
    const auto& bytes = state.bytes;

    const size_t size = bytes / sizeof(T);

    // Allocate device buffers
    // Note: since this benchmark only tests performance by memcpying between device buffers,
    // we don't really need to transfer data into these from the host - whatever happens
    // to be in device memory will do.
    common::device_ptr<T> d_input(size);
    common::device_ptr<T> d_output(size);

    state.run(
        [&]
        {
            HIP_CHECK(hipMemcpy(d_output.get(),
                                d_input.get(),
                                size * sizeof(T),
                                hipMemcpyDeviceToDevice));
        });

    state.set_throughput(size, sizeof(T));
}

#define CREATE_BENCHMARK(METHOD, OPERATION, T, BLOCK_SIZE, IPT)                            \
    executor.queue_fn(bench_naming::format_name("{lvl:device,algo:memory,subalgo:" #METHOD \
                                                ",operation:" #OPERATION ",key_type:" #T   \
                                                ",cfg:{bs:" #BLOCK_SIZE ",ipt:" #IPT "}}") \
                          .c_str(),                                                        \
                      run_benchmark<T, BLOCK_SIZE, IPT, METHOD, OPERATION>);

#define CREATE_BENCHMARK_MEMCPY(T)                                                    \
    executor.queue_fn(                                                                \
        bench_naming::format_name("{lvl:device,algo:memory,subalgo:copy,key_type:" #T \
                                  ",cfg:default_config}")                             \
            .c_str(),                                                                 \
        run_benchmark_memcpy<T>);

int main(int argc, char* argv[])
{
    benchmark_utils::executor executor(argc, argv, 128 * benchmark_utils::MiB, 10, 10);

    // simple memory copy not running kernel
    CREATE_BENCHMARK_MEMCPY(int)

    // simple memory copy
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, int, 1024, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, uint64_t, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::int128_t, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, no_operation, rocprim::uint128_t, 1024, 2)

    // simple memory copy using vector type
    CREATE_BENCHMARK(vectorized, no_operation, int, 128, 1)
    CREATE_BENCHMARK(vectorized, no_operation, int, 128, 2)
    CREATE_BENCHMARK(vectorized, no_operation, int, 128, 4)
    CREATE_BENCHMARK(vectorized, no_operation, int, 128, 8)
    CREATE_BENCHMARK(vectorized, no_operation, int, 128, 16)

    CREATE_BENCHMARK(vectorized, no_operation, int, 256, 1)
    CREATE_BENCHMARK(vectorized, no_operation, int, 256, 2)
    CREATE_BENCHMARK(vectorized, no_operation, int, 256, 4)
    CREATE_BENCHMARK(vectorized, no_operation, int, 256, 8)
    CREATE_BENCHMARK(vectorized, no_operation, int, 256, 16)

    CREATE_BENCHMARK(vectorized, no_operation, int, 512, 1)
    CREATE_BENCHMARK(vectorized, no_operation, int, 512, 2)
    CREATE_BENCHMARK(vectorized, no_operation, int, 512, 4)
    CREATE_BENCHMARK(vectorized, no_operation, int, 512, 8)

    CREATE_BENCHMARK(vectorized, no_operation, int, 1024, 1)
    CREATE_BENCHMARK(vectorized, no_operation, int, 1024, 2)
    CREATE_BENCHMARK(vectorized, no_operation, int, 1024, 4)
    CREATE_BENCHMARK(vectorized, no_operation, int, 1024, 8)

    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 1)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 2)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 4)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 8)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 128, 16)

    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 1)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 2)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 4)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 8)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 256, 16)

    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 512, 1)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 512, 2)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 512, 4)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 512, 8)

    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 1024, 4)
    CREATE_BENCHMARK(vectorized, no_operation, uint64_t, 1024, 8)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 8)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 256, 16)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 512, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 512, 8)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 1024, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::int128_t, 1024, 8)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 8)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 256, 16)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 512, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 512, 8)

    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 1024, 4)
    CREATE_BENCHMARK(vectorized, no_operation, rocprim::uint128_t, 1024, 8)

    // simple memory copy using striped
    CREATE_BENCHMARK(striped, no_operation, int, 128, 1)
    CREATE_BENCHMARK(striped, no_operation, int, 128, 2)
    CREATE_BENCHMARK(striped, no_operation, int, 128, 4)
    CREATE_BENCHMARK(striped, no_operation, int, 128, 8)
    CREATE_BENCHMARK(striped, no_operation, int, 128, 16)

    CREATE_BENCHMARK(striped, no_operation, int, 256, 1)
    CREATE_BENCHMARK(striped, no_operation, int, 256, 2)
    CREATE_BENCHMARK(striped, no_operation, int, 256, 4)
    CREATE_BENCHMARK(striped, no_operation, int, 256, 8)
    CREATE_BENCHMARK(striped, no_operation, int, 256, 16)

    CREATE_BENCHMARK(striped, no_operation, int, 512, 1)
    CREATE_BENCHMARK(striped, no_operation, int, 512, 2)
    CREATE_BENCHMARK(striped, no_operation, int, 512, 4)
    CREATE_BENCHMARK(striped, no_operation, int, 512, 8)

    CREATE_BENCHMARK(striped, no_operation, int, 1024, 1)
    CREATE_BENCHMARK(striped, no_operation, int, 1024, 2)
    CREATE_BENCHMARK(striped, no_operation, int, 1024, 4)
    CREATE_BENCHMARK(striped, no_operation, int, 1024, 8)

    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 1)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 2)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 4)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 8)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 128, 16)

    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 1)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 2)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 4)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 8)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 256, 16)

    CREATE_BENCHMARK(striped, no_operation, uint64_t, 512, 1)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 512, 2)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 512, 4)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 512, 8)

    CREATE_BENCHMARK(striped, no_operation, uint64_t, 1024, 1)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 1024, 2)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 1024, 4)
    CREATE_BENCHMARK(striped, no_operation, uint64_t, 1024, 8)

    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 8)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 256, 16)

    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 512, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 512, 8)

    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 1024, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 1024, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::int128_t, 1024, 8)

    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 8)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 256, 16)

    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 512, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 512, 8)

    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 1024, 2)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 1024, 4)
    CREATE_BENCHMARK(striped, no_operation, rocprim::uint128_t, 1024, 8)

    // block_scan
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 128, 16)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 128, 32)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, int, 1024, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, float, 1024, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, double, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, uint64_t, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::int128_t, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, block_scan, rocprim::uint128_t, 1024, 2)

    // vectorized - block_scan
    CREATE_BENCHMARK(vectorized, block_scan, int, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, int, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, int, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, int, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, int, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, int, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, int, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, int, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, int, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, int, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, int, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, int, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, int, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, int, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, int, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, int, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, int, 1024, 4)
    CREATE_BENCHMARK(vectorized, block_scan, int, 1024, 8)

    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, float, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, float, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, float, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, float, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, float, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, float, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, float, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, float, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, float, 1024, 4)
    CREATE_BENCHMARK(vectorized, block_scan, float, 1024, 8)

    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, double, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, double, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, double, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, double, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, double, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, double, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, double, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, double, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, double, 1024, 4)

    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, uint64_t, 1024, 4)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::int128_t, 1024, 4)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 8)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 256, 16)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 512, 4)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 512, 8)

    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 1024, 2)
    CREATE_BENCHMARK(vectorized, block_scan, rocprim::uint128_t, 1024, 4)

    // custom_op
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, int, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, float, 1024, 4)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, double, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, uint64_t, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::int128_t, 1024, 2)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 256, 8)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 512, 4)

    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, custom_operation, rocprim::uint128_t, 1024, 2)

    // block_primitives_transpose - atomics no collision
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_no_collision, int, 1024, 8)

    // block_primitives_transpose - atomics inter block collision
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_block_collision, int, 1024, 8)

    // block_primitives_transpose - atomics inter warp collision
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 128, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 128, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 128, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 128, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 128, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 256, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 256, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 256, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 256, 8)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 256, 16)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 512, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 512, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 512, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 512, 8)

    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 1024, 1)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 1024, 2)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 1024, 4)
    CREATE_BENCHMARK(block_primitives_transpose, atomics_inter_warp_collision, int, 1024, 8)

    executor.run();
}
