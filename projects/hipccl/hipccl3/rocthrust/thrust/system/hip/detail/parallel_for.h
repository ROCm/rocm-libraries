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

#include <thrust/detail/type_traits/result_of_adaptable_function.h>
#include <thrust/system/hip/config.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/util.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
namespace __parallel_for
{
    template <unsigned int BlockSize,
              unsigned int ItemsPerThread,
              unsigned int SizeLimit = THRUST_HIP_GRID_SIZE_LIMIT>
    struct kernel_config
    {
        /// \brief Number of threads in a block.
        static constexpr unsigned int block_size = BlockSize;
        /// \brief Number of items processed by each thread.
        static constexpr unsigned int items_per_thread = ItemsPerThread;
        /// \brief Number of items processed by a single kernel launch.
        static constexpr unsigned int size_limit = SizeLimit;
    };

    template <unsigned int BlockSize, class F, class Size, unsigned int ItemsPerThread>
    __global__
    THRUST_HIP_LAUNCH_BOUNDS(BlockSize)
    void kernel(F f, Size num_items, Size offset)
    {
        constexpr auto     items_per_block = BlockSize * ItemsPerThread;
        Size               tile_base       = blockIdx.x * items_per_block;
        Size               num_remaining   = num_items - offset - tile_base;
        const unsigned int items_in_tile   = static_cast<unsigned int>(
            num_remaining < (Size)items_per_block ? num_remaining : items_per_block);

        if(items_in_tile == items_per_block)
        {
            #pragma unroll
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                unsigned int idx = BlockSize * i + threadIdx.x;
                f(offset + tile_base + idx);
            }
        }
        else
        {
            #pragma unroll
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                unsigned int idx = BlockSize * i + threadIdx.x;
                if(idx < items_in_tile)
                    f(offset + tile_base + idx);
            }
        }
    }

    template <class F, class Size>
    hipError_t THRUST_HIP_RUNTIME_FUNCTION
    parallel_for(Size num_items, F f, hipStream_t stream)
    {
        using config    = kernel_config<256, 1>;
        bool debug_sync = THRUST_HIP_DEBUG_SYNC_FLAG;
        // Use debug_sync
        (void)debug_sync;

        constexpr unsigned int block_size       = config::block_size;
        constexpr unsigned int items_per_thread = config::items_per_thread;
        constexpr auto items_per_block          = block_size * items_per_thread;

        // Find maximum number of items per one step and number of steps
        constexpr size_t aligned_size_limit     = (config::size_limit / items_per_block) * items_per_block;
        const Size number_of_launch             = (num_items + aligned_size_limit - 1) / aligned_size_limit;

        for(Size i = 0, offset = 0; i < number_of_launch; ++i, offset += aligned_size_limit)
        {
            const size_t current_items = std::min<size_t>(num_items - offset, aligned_size_limit);
            const unsigned int number_of_blocks = (current_items + items_per_block - 1) / items_per_block;

            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<block_size, F, Size, items_per_thread>),
                               dim3(number_of_blocks),
                               dim3(block_size),
                               0,
                               stream,
                               f,
                               num_items,
                               offset);
        }

        auto error = hipPeekAtLastError();
        if(error != hipSuccess)
            return error;
        return hipSuccess;
    }
} // __parallel_for

__thrust_exec_check_disable__ template <class Derived, class F, class Size>
void THRUST_HIP_FUNCTION
parallel_for(execution_policy<Derived>& policy, F f, Size count)
{
    if(count == 0)
    {
        return;
    }

    // struct workaround is required for HIP-clang
    struct workaround
    {
        __host__
        static void par(execution_policy<Derived>& policy, F f, Size count)
        {
            hipStream_t stream = hip_rocprim::stream(policy);
            hipError_t  status = __parallel_for::parallel_for(count, f, stream);
            hip_rocprim::throw_on_error(status, "parallel_for failed");
            hip_rocprim::throw_on_error(hip_rocprim::synchronize_optional(policy),
                                        "parallel_for: failed to synchronize");
        }

        __device__
        static void seq(execution_policy<Derived>& policy, F f, Size count)
        {
            (void)policy;
            for(Size idx = 0; idx != count; ++idx)
                f(idx);
        }
    };

#if __THRUST_HAS_HIPRT__
    workaround::par(policy, f, count);
#else
    workaround::seq(policy, f, count);
#endif
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
