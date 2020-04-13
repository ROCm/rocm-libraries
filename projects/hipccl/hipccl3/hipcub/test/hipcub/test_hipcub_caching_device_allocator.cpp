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

#include "hipcub/util_allocator.hpp"

__global__ void EmptyKernel() { }

// Hipified test/test_allocator.cu

TEST(HipcubCachingDeviceAllocatorTests, Test1)
{
    // Get number of GPUs and current GPU
    int num_gpus;
    int initial_gpu;

    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    HIP_CHECK(hipGetDevice(&initial_gpu));

    // Create default allocator (caches up to 6MB in device allocations per GPU)
    hipcub::CachingDeviceAllocator allocator;
    allocator.debug = true;

    //
    // Test0
    //

    // Create a new stream
    hipStream_t other_stream;
    HIP_CHECK(hipStreamCreate(&other_stream));

    // Allocate 999 bytes on the current gpu in stream0
    char *d_999B_stream0_a;
    char *d_999B_stream0_b;
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_999B_stream0_a, 999, 0));

    // Run some big kernel in stream 0
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(EmptyKernel),
        dim3(32000), dim3(256), 1024 * 8, 0
    );

    // Free d_999B_stream0_a
    HIP_CHECK(allocator.DeviceFree(d_999B_stream0_a));

    // Allocate another 999 bytes in stream 0
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_999B_stream0_b, 999, 0));

    // Check that that we have 1 live block on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 1u);

    // Check that that we have no cached block on the initial GPU
    ASSERT_EQ(allocator.cached_blocks.size(), 0u);

    // Run some big kernel in stream 0
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(EmptyKernel),
        dim3(32000), dim3(256), 1024 * 8, 0
    );

    // Free d_999B_stream0_b
    HIP_CHECK(allocator.DeviceFree(d_999B_stream0_b));

    // Allocate 999 bytes on the current gpu in other_stream
    char *d_999B_stream_other_a;
    char *d_999B_stream_other_b;
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_999B_stream_other_a, 999, other_stream));

    // Check that that we have 1 live blocks on the initial GPU (that we allocated a new one because d_999B_stream0_b is only available for stream 0 until it becomes idle)
    ASSERT_EQ(allocator.live_blocks.size(), 1u);

    // Check that that we have one cached block on the initial GPU
    ASSERT_EQ(allocator.cached_blocks.size(), 1u);

    // Run some big kernel in other_stream
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(EmptyKernel),
        dim3(32000), dim3(256), 1024 * 8, other_stream
    );

    // Free d_999B_stream_other
    HIP_CHECK(allocator.DeviceFree(d_999B_stream_other_a));

    // Check that we can now use both allocations in stream 0 after synchronizing the device
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_999B_stream0_a, 999, 0));
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_999B_stream0_b, 999, 0));

    // Check that that we have 2 live blocks on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 2u);

    // Check that that we have no cached block on the initial GPU
    ASSERT_EQ(allocator.cached_blocks.size(), 0u);

    // Free d_999B_stream0_a and d_999B_stream0_b
    HIP_CHECK(allocator.DeviceFree(d_999B_stream0_a));
    HIP_CHECK(allocator.DeviceFree(d_999B_stream0_b));

    // Check that we can now use both allocations in other_stream
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_999B_stream_other_a, 999, other_stream));
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_999B_stream_other_b, 999, other_stream));

    // Check that that we have 2 live blocks on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 2u);

    // Check that that we have no cached block on the initial GPU
    ASSERT_EQ(allocator.cached_blocks.size(), 0u);

    // Run some big kernel in other_stream
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(EmptyKernel),
        dim3(32000), dim3(256), 1024 * 8, other_stream
    );

    // Free d_999B_stream_other_a and d_999B_stream_other_b
    HIP_CHECK(allocator.DeviceFree(d_999B_stream_other_a));
    HIP_CHECK(allocator.DeviceFree(d_999B_stream_other_b));

    // Check that we can now use both allocations in stream 0 after synchronizing the device and destroying the other stream
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipStreamDestroy(other_stream));
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_999B_stream0_a, 999, 0));
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_999B_stream0_b, 999, 0));

    // Check that that we have 2 live blocks on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 2u);

    // Check that that we have no cached block on the initial GPU
    ASSERT_EQ(allocator.cached_blocks.size(), 0u);

    // Free d_999B_stream0_a and d_999B_stream0_b
    HIP_CHECK(allocator.DeviceFree(d_999B_stream0_a));
    HIP_CHECK(allocator.DeviceFree(d_999B_stream0_b));

    // Free all cached
    HIP_CHECK(allocator.FreeAllCached());

    //
    // Test1
    //

    // Allocate 5 bytes on the current gpu
    char *d_5B;
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_5B, 5));

    // Check that that we have zero free bytes cached on the initial GPU
    ASSERT_EQ(allocator.cached_bytes[initial_gpu].free, 0u);

    // Check that that we have 1 live block on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 1u);

    //
    // Test2
    //

    // Allocate 4096 bytes on the current gpu
    char *d_4096B;
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_4096B, 4096));

    // Check that that we have 2 live blocks on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 2u);

    //
    // Test3
    //

    // DeviceFree d_5B
    HIP_CHECK(allocator.DeviceFree(d_5B));

    // Check that that we have min_bin_bytes free bytes cached on the initial gpu
    ASSERT_EQ(allocator.cached_bytes[initial_gpu].free, allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 1u);

    // Check that that we have 1 cached block on the initial GPU
    ASSERT_EQ(allocator.cached_blocks.size(), 1u);

    //
    // Test4
    //

    // DeviceFree d_4096B
    HIP_CHECK(allocator.DeviceFree(d_4096B));

    // Check that that we have the 4096 + min_bin free bytes cached on the initial gpu
    ASSERT_EQ(allocator.cached_bytes[initial_gpu].free, allocator.min_bin_bytes + 4096);

    // Check that that we have 0 live block on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 0u);

    // Check that that we have 2 cached block on the initial GPU
    ASSERT_EQ(allocator.cached_blocks.size(), 2u);

    //
    // Test5
    //

    // Allocate 768 bytes on the current gpu
    char *d_768B;
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_768B, 768));

    // Check that that we have the min_bin free bytes cached on the initial gpu (4096 was reused)
    ASSERT_EQ(allocator.cached_bytes[initial_gpu].free, allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 1u);

    // Check that that we have 1 cached block on the initial GPU
    ASSERT_EQ(allocator.cached_blocks.size(), 1u);

    //
    // Test6
    //

    // Allocate max_cached_bytes on the current gpu
    char *d_max_cached;
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_max_cached, allocator.max_cached_bytes));

    // DeviceFree d_max_cached
    HIP_CHECK(allocator.DeviceFree(d_max_cached));

    // Check that that we have the min_bin free bytes cached on the initial gpu (max cached was not returned because we went over)
    ASSERT_EQ(allocator.cached_bytes[initial_gpu].free, allocator.min_bin_bytes);

    // Check that that we have 1 live block on the initial GPU
    ASSERT_EQ(allocator.live_blocks.size(), 1u);

    // Check that that we still have 1 cached block on the initial GPU
    ASSERT_EQ(allocator.cached_blocks.size(), 1u);

    //
    // Test7
    //

    // Free all cached blocks on all GPUs
    HIP_CHECK(allocator.FreeAllCached());

    // Check that that we have 0 bytes cached on the initial GPU
    ASSERT_EQ(allocator.cached_bytes[initial_gpu].free, 0u);

    // Check that that we have 0 cached blocks across all GPUs
    ASSERT_EQ(allocator.cached_blocks.size(), 0u);

    // Check that that still we have 1 live block across all GPUs
    ASSERT_EQ(allocator.live_blocks.size(), 1u);

    //
    // Test8
    //

    // Allocate max cached bytes + 1 on the current gpu
    char *d_max_cached_plus;
    HIP_CHECK(allocator.DeviceAllocate((void **) &d_max_cached_plus, allocator.max_cached_bytes + 1));

    // DeviceFree max cached bytes
    HIP_CHECK(allocator.DeviceFree(d_max_cached_plus));

    // DeviceFree d_768B
    HIP_CHECK(allocator.DeviceFree(d_768B));

    unsigned int power;
    size_t rounded_bytes;
    allocator.NearestPowerOf(power, rounded_bytes, allocator.bin_growth, 768);

    // Check that that we have 4096 free bytes cached on the initial gpu
    ASSERT_EQ(allocator.cached_bytes[initial_gpu].free, rounded_bytes);

    // Check that that we have 1 cached blocks across all GPUs
    ASSERT_EQ(allocator.cached_blocks.size(), 1u);

    // Check that that still we have 0 live block across all GPUs
    ASSERT_EQ(allocator.live_blocks.size(), 0u);

    if (num_gpus > 1)
    {
        //
        // Test9
        //

        // Allocate 768 bytes on the next gpu
        int next_gpu = (initial_gpu + 1) % num_gpus;
        char *d_768B_2;
        HIP_CHECK(allocator.DeviceAllocate(next_gpu, (void **) &d_768B_2, 768));

        // DeviceFree d_768B on the next gpu
        HIP_CHECK(allocator.DeviceFree(next_gpu, d_768B_2));

        // Re-allocate 768 bytes on the next gpu
        HIP_CHECK(allocator.DeviceAllocate(next_gpu, (void **) &d_768B_2, 768));

        // Re-free d_768B on the next gpu
        HIP_CHECK(allocator.DeviceFree(next_gpu, d_768B_2));

        // Check that that we have 4096 free bytes cached on the initial gpu
        ASSERT_EQ(allocator.cached_bytes[initial_gpu].free, rounded_bytes);

        // Check that that we have 4096 free bytes cached on the second gpu
        ASSERT_EQ(allocator.cached_bytes[next_gpu].free, rounded_bytes);

        // Check that that we have 2 cached blocks across all GPUs
        ASSERT_EQ(allocator.cached_blocks.size(), 2u);

        // Check that that still we have 0 live block across all GPUs
        ASSERT_EQ(allocator.live_blocks.size(), 0u);
    }
}
