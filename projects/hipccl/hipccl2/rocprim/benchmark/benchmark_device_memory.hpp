// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "primbench.hpp"

#include "benchmark_utils.hpp"

#include "../common/utils_data_generation.hpp"
#include "../common/utils_device_ptr.hpp"

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
    copy,
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

// custom operation
template<typename T, unsigned int ItemsPerThread, unsigned int BlockSize>
struct operation<custom_operation, T, ItemsPerThread, BlockSize>
{
    ROCPRIM_HOST_DEVICE
    inline void operator()(T (&input)[ItemsPerThread],
                           void*        shared_storage      = nullptr,
                           unsigned int shared_storage_size = 0,
                           T*           global_mem_output   = nullptr) const
    {
        (void)shared_storage;
        (void)shared_storage_size;
        (void)global_mem_output;
        constexpr unsigned int repeats = 30;
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
    inline void operator()(T (&input)[ItemsPerThread],
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
            __shared__
            typename block_scan_type::storage_type storage;
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
    inline void operator()(T (&input)[ItemsPerThread],
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
    inline void operator()(T (&input)[ItemsPerThread],
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
    inline void operator()(T (&input)[ItemsPerThread],
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
         kernel_operation        KernelOp = no_operation,
         typename Config                  = rocprim::default_config>
struct device_memory_benchmark : public primbench::benchmark_interface
{
    primbench::json meta() const override
    {
        return primbench::json{}
            .add("lvl", "device")
            .add("algo", "device_memory")
            .add("subalgo", get_method_name(MemOp))
            .add("operation", get_operation_name(KernelOp))
            .add("key_type", primbench::name<T>())
            .add("cfg", primbench::json{}.add("bs", BlockSize).add("ipt", ItemsPerThread));
    }

    void run(primbench::state& state) override
    {
        const auto& stream = state.stream;
        const auto& bytes  = state.size;
        const auto& seed   = state.seed;

        const size_t items = bytes / sizeof(T);

        const size_t   grid_size = items / (BlockSize * ItemsPerThread);
        std::vector<T> input     = get_random_data<T>(items,
                                                  common::generate_limits<T>::min(),
                                                  common::generate_limits<T>::max(),
                                                  seed);

        common::device_ptr<T> d_input(input);
        common::device_ptr<T> d_output(items);

        operation<KernelOp, T, ItemsPerThread, BlockSize> selected_operation;

        state.set_items(items);
        state.add_reads<T>(items);

        state.run(
            [&]
            {
                if constexpr(MemOp == copy)
                {
                    // For copy benchmark, perform device-to-device memcpy
                    HIP_CHECK(hipMemcpy(d_output.get(),
                                        d_input.get(),
                                        items * sizeof(T),
                                        hipMemcpyDeviceToDevice));
                }
                else
                {
                    operation_kernel<T, BlockSize, ItemsPerThread, MemOp>
                        <<<dim3(grid_size), dim3(BlockSize), 0, stream>>>(d_input.get(),
                                                                          d_output.get(),
                                                                          selected_operation);
                }
            });
    }

private:
    std::string get_method_name(memory_operation_method method) const
    {
        switch(method)
        {
            case block_primitives_transpose: return "block_primitives_transpose";
            case striped: return "striped";
            case vectorized: return "vectorized";
            case block_primitive_direct: return "block_primitive_direct";
            case copy:
                return "copy";
                // Not using `default: ...` because it kills effectiveness of -Wswitch
        }
        return "unknown_method";
    }

    std::string get_operation_name(kernel_operation operation) const
    {
        switch(operation)
        {
            case no_operation: return "no_operation";
            case block_scan: return "block_scan";
            case custom_operation: return "custom_operation";
            case atomics_no_collision: return "atomics_no_collision";
            case atomics_inter_block_collision: return "atomics_inter_block_collision";
            case atomics_inter_warp_collision:
                return "atomics_inter_warp_collision";
                // Not using `default: ...` because it kills effectiveness of -Wswitch
        }
        return "unknown_operation";
    }
};
