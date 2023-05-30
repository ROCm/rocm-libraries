/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2023, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#include <thrust/system/hip/config.h>

#include <thrust/detail/cstdint.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/transform.h>
#include <thrust/system/hip/detail/util.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>

// rocprim include
#include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction>
__host__ __device__ OutputIterator
adjacent_difference(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                    InputIterator                                               first,
                    InputIterator                                               last,
                    OutputIterator                                              result,
                    BinaryFunction                                              binary_op);

namespace hip_rocprim
{
namespace __adjacent_difference
{
    template <unsigned int BlockSize, unsigned int ItemsPerThread>
    struct kernel_config
    {
        static constexpr unsigned int block_size       = BlockSize;
        static constexpr unsigned int items_per_thread = ItemsPerThread;
    };

    template <unsigned int BlockSize, unsigned int ItemsPerThread>
    using adjacent_difference_config = kernel_config<BlockSize, ItemsPerThread>;

    template <class Value>
    struct adjacent_difference_config_803
    {
        static constexpr unsigned int item_scale
            = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

        using type = adjacent_difference_config<256, ::rocprim::max(1u, 16u / item_scale)>;
    };

    template <class Value>
    struct adjacent_difference_config_900
    {
        static constexpr unsigned int item_scale
            = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

        using type = adjacent_difference_config<256, ::rocprim::max(1u, 16u / item_scale)>;
    };

    template <unsigned int TargetArch, class Value>
    struct default_adjacent_difference_config
        : rocprim::detail::select_arch<
              TargetArch,
              rocprim::detail::select_arch_case<803, adjacent_difference_config_803<Value>>,
              rocprim::detail::select_arch_case<900, adjacent_difference_config_900<Value>>,
              adjacent_difference_config_900<Value>>
    {
    };

    template <unsigned int BlockSize,
              unsigned int ItemsPerThread,
              unsigned int AdjacentDiffItemsPerBlock,
              class InputIt,
              class HeadsIt,
              class Size>
    __global__
    THRUST_HIP_LAUNCH_BOUNDS(BlockSize)
    void block_heads_fill(InputIt input, HeadsIt block_heads, Size input_size)
    {
        constexpr auto items_per_block  = BlockSize * ItemsPerThread;
        Size           tile_base        = blockIdx.x * items_per_block;
        Size           next_tile_base   = (blockIdx.x + 1) * items_per_block;
        Size           last_item_index  = next_tile_base - 1;
        Size           last_input_index = (last_item_index + 1) * AdjacentDiffItemsPerBlock - 1;

        const bool full_block = input_size > last_input_index;
        if(full_block)
        {
            #pragma unroll
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                Size idx         = tile_base + threadIdx.x * ItemsPerThread + i;
                Size input_index = (idx + 1) * AdjacentDiffItemsPerBlock - 1;
                block_heads[idx] = input[input_index];
            }
        }
        else
        {
            #pragma unroll
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                unsigned int idx         = tile_base + threadIdx.x * ItemsPerThread + i;
                Size         input_index = (idx + 1) * AdjacentDiffItemsPerBlock - 1;
                if(input_index < input_size)
                    block_heads[idx] = input[input_index];
            }
        }
    }

    template <unsigned int BlockSize,
              unsigned int ItemsPerThread,
              class InputIt,
              class HeadsIt,
              class OutputIt,
              class BinaryOp>
    __global__
    THRUST_HIP_LAUNCH_BOUNDS(BlockSize)
    void adjacent_difference_kernel(InputIt      input,
                                    HeadsIt      block_heads,
                                    const size_t input_size,
                                    OutputIt     output,
                                    BinaryOp     binary_op)
    {
        using input_type = typename std::iterator_traits<InputIt>::value_type;

        using block_load_type
            = ::rocprim::block_load<input_type,
                                    BlockSize,
                                    ItemsPerThread,
                                    ::rocprim::block_load_method::block_load_transpose>;
        using block_store_type
            = ::rocprim::block_store<input_type,
                                     BlockSize,
                                     ItemsPerThread,
                                     ::rocprim::block_store_method::block_store_transpose>;

        ROCPRIM_SHARED_MEMORY union
        {
            typename block_load_type::storage_type  load;
            typename block_store_type::storage_type store;
            input_type                              last_items[BlockSize] = {};
        } storage;

        constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

        const unsigned int flat_id       = ::rocprim::detail::block_thread_id<0>();
        const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
        const unsigned int block_offset  = flat_block_id * BlockSize * ItemsPerThread;
        const unsigned int number_of_blocks
            = (input_size + items_per_block - 1) / items_per_block;
        auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);

        input_type values[ItemsPerThread];

        // load input values into values
        if(flat_block_id == (number_of_blocks - 1)) // last block
        {
            block_load_type().load(input + block_offset,
                                   values,
                                   valid_in_last_block,
                                   *(input + block_offset),
                                   storage.load);
        }
        else
        {
            block_load_type().load(input + block_offset, values, storage.load);
        }
        ::rocprim::syncthreads(); // sync threads to reuse shared memory

        storage.last_items[flat_id] = values[ItemsPerThread - 1];
        ::rocprim::syncthreads();

        #pragma unroll
        for(unsigned int i = ItemsPerThread - 1; i > 0; i--)
        {
            values[i] = binary_op(values[i], values[i - 1]);
        }

        // calculate the first element of the thread
        if(!(flat_block_id == 0 && flat_id == 0))
        {
            // load previuos thread last value
            input_type input_prev;
            if(flat_id == 0) // first thread in block
            {
                input_prev = block_heads[flat_block_id - 1];
            }
            else
            {
                input_prev = storage.last_items[flat_id - 1];
            }
            values[0] = binary_op(values[0], input_prev);
        }
        ::rocprim::syncthreads();

        // Save values into output array
        if(flat_block_id == (number_of_blocks - 1)) // last block
        {
            block_store_type().store(
                output + block_offset, values, valid_in_last_block, storage.store);
        }
        else
        {
            block_store_type().store(output + block_offset, values, storage.store);
        }
    }

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                         \
    {                                                                                          \
        auto error = hipPeekAtLastError();                                                     \
        if(error != hipSuccess)                                                                \
            return error;                                                                      \
        if(debug_synchronous)                                                                  \
        {                                                                                      \
            std::cout << name << "(" << size << ")";                                           \
            auto error = hipStreamSynchronize(stream);                                         \
            if(error != hipSuccess)                                                            \
                return error;                                                                  \
            auto end = std::chrono::high_resolution_clock::now();                              \
            auto d   = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n';                             \
        }                                                                                      \
    }

    template <class InputIt, class OutputIt, class BinaryOp>
    hipError_t THRUST_HIP_RUNTIME_FUNCTION
    doit_step(void*       temporary_storage,
              size_t&     storage_size,
              InputIt     first,
              OutputIt    result,
              BinaryOp    binary_op,
              size_t      num_items,
              hipStream_t stream,
              bool        debug_synchronous)
    {
        using input_type  = typename std::iterator_traits<InputIt>::value_type;
        using result_type
            = typename ::rocprim::detail::match_result_type<input_type,BinaryOp>::type;

        // Get default config if Config is default_config
        using config = default_adjacent_difference_config<ROCPRIM_TARGET_ARCH, result_type>;

        constexpr unsigned int block_size       = config::block_size;
        constexpr unsigned int items_per_thread = config::items_per_thread;
        constexpr auto         items_per_block  = block_size * items_per_thread;

        const unsigned int heads      = (num_items + items_per_block - 1) / items_per_block;
        const size_t       head_bytes = (heads + 1) * sizeof(input_type);

        if(temporary_storage == nullptr)
        {
            // storage_size is never zero
            storage_size = head_bytes;
            return hipSuccess;
        }

        // Start point for time measurements
        std::chrono::high_resolution_clock::time_point start;

        auto number_of_blocks = heads;
        if(debug_synchronous)
        {
            std::cout << "block_size " << block_size << '\n';
            std::cout << "number of blocks " << number_of_blocks << '\n';
            std::cout << "items_per_block " << items_per_block << '\n';
        }

        input_type* block_heads = static_cast<input_type*>(temporary_storage);

        // The block heads fill kernel config
        using config_heads                            = kernel_config<256, 1>;
        constexpr unsigned int block_size_heads       = config_heads::block_size;
        constexpr unsigned int items_per_thread_heads = config_heads::items_per_thread;
        constexpr auto items_per_block_heads = block_size_heads * items_per_thread_heads;

        // Fill the block heads
        if(debug_synchronous)
            start = std::chrono::high_resolution_clock::now();
        auto number_of_blocks_heads
            = (heads + items_per_block_heads - 1) / items_per_block_heads;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_heads_fill<block_size_heads,
                                                            items_per_thread_heads,
                                                            items_per_block,
                                                            InputIt,
                                                            input_type*,
                                                            size_t>),
                           dim3(number_of_blocks_heads),
                           dim3(block_size_heads),
                           0,
                           stream,
                           first,
                           block_heads,
                           num_items);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_heads_fill", heads, start)

        // Adjacent difference
        if(debug_synchronous)
            start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(adjacent_difference_kernel<block_size,
                                                                      items_per_thread,
                                                                      InputIt,
                                                                      input_type*,
                                                                      OutputIt,
                                                                      BinaryOp>),
                           dim3(number_of_blocks),
                           dim3(block_size),
                           0,
                           stream,
                           first,
                           block_heads,
                           num_items,
                           result,
                           binary_op);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
            "adjacent_difference_kernel", num_items, start);

        return hipSuccess;
    }

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

    template <typename Derived,
              typename InputIt,
              typename OutputIt,
              typename BinaryOp>
    static OutputIt THRUST_HIP_RUNTIME_FUNCTION
    adjacent_difference(execution_policy<Derived>& policy,
                        InputIt                    first,
                        InputIt                    last,
                        OutputIt                   result,
                        BinaryOp                   binary_op)
    {
        typedef typename iterator_traits<InputIt>::difference_type size_type;

        size_type   num_items    = thrust::distance(first, last);
        size_t      storage_size = 0;
        hipStream_t stream       = hip_rocprim::stream(policy);
        bool        debug_sync   = THRUST_HIP_DEBUG_SYNC_FLAG;

        if(num_items == 0)
            return result;

        // Determine temporary device storage requirements.
        hip_rocprim::throw_on_error(doit_step(NULL,
                                              storage_size,
                                              first,
                                              result,
                                              binary_op,
                                              num_items,
                                              stream,
                                              debug_sync),
                                    "adjacent_difference failed on 1st step");

        // Allocate temporary storage.
        thrust::detail::temporary_array<thrust::detail::uint8_t, Derived>
            tmp(policy, storage_size);
        void *ptr = static_cast<void*>(tmp.data().get());

        hip_rocprim::throw_on_error(doit_step(ptr,
                                              storage_size,
                                              first,
                                              result,
                                              binary_op,
                                              num_items,
                                              stream,
                                              debug_sync),
                                    "adjacent_difference failed on 2nd step");

        hip_rocprim::throw_on_error(hip_rocprim::synchronize_optional(policy));

        return result + num_items;
    }

} // namespace __adjacent_difference

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived, class InputIt, class OutputIt, class BinaryOp>
OutputIt THRUST_HIP_FUNCTION
adjacent_difference(execution_policy<Derived>& policy,
                    InputIt                    first,
                    InputIt                    last,
                    OutputIt                   result,
                    BinaryOp                   binary_op)
{
    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__
        static void par(execution_policy<Derived>& policy,
                        InputIt                    first,
                        InputIt                    last,
                        OutputIt&                  result,
                        BinaryOp                   binary_op)
        {
            result = __adjacent_difference::adjacent_difference(
                policy, first, last, result, binary_op);
        }
        __device__
        static void seq(execution_policy<Derived>& policy,
                        InputIt                    first,
                        InputIt                    last,
                        OutputIt&                  result,
                        BinaryOp                   binary_op)
        {
            result = thrust::adjacent_difference(
               cvt_to_seq(derived_cast(policy)),
               first,
               last,
               result,
               binary_op
            );
        }
    };
    #if __THRUST_HAS_HIPRT__
    workaround::par(policy, first, last, result, binary_op);
    #else
    workaround::seq(policy, first, last, result, binary_op);
    #endif

    return result;
}

template <class Derived, class InputIt, class OutputIt>
OutputIt THRUST_HIP_FUNCTION
adjacent_difference(execution_policy<Derived>& policy,
                    InputIt                    first,
                    InputIt                    last,
                    OutputIt                   result)
{
    typedef typename iterator_traits<InputIt>::value_type input_type;
    return hip_rocprim::adjacent_difference(policy, first, last, result, minus<input_type>());
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END

//
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
