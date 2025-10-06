/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2025, Advanced Micro Devices, Inc.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "common_test_header.hpp"

#include <hipcub/block/block_reduce.hpp>
#include <hipcub/thread/thread_operators.hpp>

#include <hipcub/grid/grid_barrier.hpp>
#include <hipcub/grid/grid_even_share.hpp>
#include <hipcub/grid/grid_queue.hpp>

#if defined(__HIP_PLATFORM_NVIDIA__)
_CCCL_SUPPRESS_DEPRECATED_PUSH
#else
HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
#endif
__global__
void KernelGridBarrier(hipcub::GridBarrier global_barrier, int iterations)
#if defined(__HIP_PLATFORM_NVIDIA__)
    _CCCL_SUPPRESS_DEPRECATED_POP
#else
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
#endif
{
    for (int i = 0; i < iterations; i++)
    {
        global_barrier.Sync();
    }
}

TEST(HipcubGridTests, GridBarrier)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    constexpr int32_t block_size = 256;
    // NOTE increasing iterations will cause huge latency for tests
    constexpr int32_t iterations = 3;
    int32_t grid_size = -1;

    int32_t sm_count;
    int32_t max_block_threads;
    int32_t max_sm_occupancy;

    HIP_CHECK(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, device_id));
    HIP_CHECK(hipDeviceGetAttribute(&max_block_threads, hipDeviceAttributeMaxThreadsPerBlock, device_id));

    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_sm_occupancy,
        KernelGridBarrier,
        HIPCUB_HOST_WARP_THREADS,
        0));

    int32_t occupancy = std::min((max_block_threads / block_size), max_sm_occupancy);

    if (grid_size == -1)
    {
        grid_size = occupancy * sm_count;
    }
    else
    {
        occupancy = grid_size / sm_count;
    }
#if defined(__HIP_PLATFORM_NVIDIA__)
    _CCCL_SUPPRESS_DEPRECATED_PUSH
#else
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
#endif
    hipcub::GridBarrierLifetime global_barrier;
#if defined(__HIP_PLATFORM_NVIDIA__)
    _CCCL_SUPPRESS_DEPRECATED_POP
#else
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
#endif
    HIP_CHECK(global_barrier.Setup(grid_size));

    KernelGridBarrier<<<grid_size, block_size>>>(global_barrier, iterations);
}

template<
    int32_t BlockSize,
    class T,
    typename OffsetT
>
__global__ void KernelGridEvenShare(
    const T* device_output,
    T* device_output_reductions,
    hipcub::GridEvenShare<OffsetT>  even_share)
{
    using breduce_t = hipcub::BlockReduce<T, BlockSize>;
    __shared__ typename breduce_t::TempStorage temp_storage;

    even_share.template BlockInit<BlockSize, hipcub::GRID_MAPPING_RAKE>();

    const int32_t index = even_share.block_offset + hipThreadIdx_x;
    if(index > even_share.block_end)
    {
        return;
    }

    T value = device_output[index];

    value = breduce_t(temp_storage).Reduce(value, hipcub::Sum());
    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[hipBlockIdx_x] = value;
    }
}

TEST(HipcubGridTests, GridEvenShare)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using OffsetT = int32_t;
    using T = uint32_t;
    constexpr size_t block_size = 256;
    constexpr size_t size = block_size * 113;
    constexpr size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value += output[idx];
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        hipcub::GridEvenShare<OffsetT> even_share;
        even_share.DispatchInit(size, grid_size, block_size);

        KernelGridEvenShare<block_size, T, OffsetT>
            <<<grid_size, block_size>>>
                (device_output,
                 device_output_reductions,
                 even_share);

        // Reading results back
        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
    }
}

template<typename OffsetT>
__global__ void KernelGridQueueInit(hipcub::GridQueue<OffsetT> tile_queue)
{
    if ((threadIdx.x == 0) && (blockIdx.x == 0))
    {
        (void)tile_queue.ResetDrain();
    }
}

template<
    int32_t BlockSize,
    class T,
    typename OffsetT
>
__global__ void KernelGridQueue(
    const T* device_output,
    T* device_output_reductions,
    OffsetT num_tiles,
    hipcub::GridQueue<OffsetT> tile_queue)
{
    using breduce_t = hipcub::BlockReduce<T, BlockSize>;
    __shared__ typename breduce_t::TempStorage temp_storage;
    __shared__ int32_t block_tile_index;

    if(hipThreadIdx_x == 0)
    {
        block_tile_index = tile_queue.Drain(1);
    }
    __syncthreads();

    if(block_tile_index > num_tiles || block_tile_index < 0)
    {
        return;
    }

    int32_t index = block_tile_index * BlockSize + hipThreadIdx_x;
    T value = device_output[index];
    value = breduce_t(temp_storage).Reduce(value, hipcub::Sum());

    if(hipThreadIdx_x == 0)
    {
        device_output_reductions[block_tile_index] = value;
    }
}

TEST(HipcubGridTests, GridQueue)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using OffsetT = int32_t;
    using T = uint32_t;
    constexpr size_t block_size = 256;
    constexpr size_t size = block_size * 113;
    constexpr size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value += output[idx];
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        HIP_CHECK(hipMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        HIP_CHECK(
            hipMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                hipMemcpyHostToDevice
            )
        );

        OffsetT* queue_allocations;
        HIP_CHECK(hipMalloc(&queue_allocations, hipcub::GridQueue<OffsetT>().AllocationSize()));
        hipcub::GridQueue<OffsetT> tile_queue(queue_allocations);

        KernelGridQueueInit<OffsetT><<<1, 1>>>(tile_queue);

        KernelGridQueue<block_size, T, OffsetT>
            <<<grid_size, block_size>>>
                (device_output,
                 device_output_reductions,
                 113,
                 tile_queue);

        HIP_CHECK(
            hipMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                hipMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        HIP_CHECK(hipFree(device_output));
        HIP_CHECK(hipFree(device_output_reductions));
        HIP_CHECK(hipFree(queue_allocations));
    }
}
