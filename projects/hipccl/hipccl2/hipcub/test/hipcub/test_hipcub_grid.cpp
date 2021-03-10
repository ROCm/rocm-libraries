/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2020, Advanced Micro Devices, Inc.  All rights reserved.
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

#include "hipcub/grid/grid_barrier.hpp"
#include "hipcub/grid/grid_even_share.hpp"
#include "hipcub/grid/grid_queue.hpp"

__global__ void Kernel(
    hipcub::GridBarrier global_barrier,
    int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        global_barrier.Sync();
    }
}

TEST(HipcubGridTests, GridBarrier)
{
    constexpr int32_t block_size = 256;
    // NOTE increasing iterations will cause huge latency for tests
    constexpr int32_t iterations = 3;
    int32_t grid_size = -1;

    int32_t sm_count;
    int32_t max_block_threads;
    int32_t max_sm_occupancy;

    int32_t device_id;
    HIP_CHECK(hipGetDevice(&device_id));

    HIP_CHECK(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, device_id));
    HIP_CHECK(hipDeviceGetAttribute(&max_block_threads, hipDeviceAttributeMaxThreadsPerBlock, device_id));

    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_sm_occupancy,
        Kernel,
        HIPCUB_WARP_THREADS,
        0));

    // Compute grid size and occupancy
    int32_t occupancy = std::min((max_block_threads / block_size), max_sm_occupancy);

    if (grid_size == -1)
    {
        grid_size = occupancy * sm_count;
    }
    else
    {
        occupancy = grid_size / sm_count;
    }

    // Init global barrier
    hipcub::GridBarrierLifetime global_barrier;
    HIP_CHECK(global_barrier.Setup(grid_size));

    // Time kernel
    Kernel<<<grid_size, block_size>>>(global_barrier, iterations);
}
