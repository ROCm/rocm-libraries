// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_BENCHMARK_OCCUPANCY_HELPER_HPP__
#define ROCRAND_BENCHMARK_OCCUPANCY_HELPER_HPP__

#include "benchmark_utils.hpp"
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

struct launch_params
{
    int blocks                = 0;
    int threads               = 0;
    int occupancy             = 0;
    int multiprocessors       = 0;
    int max_threads_per_block = 0;
};

/// Compute the launch bounds (block size and grid size). By default we'll
/// launch a benchmark with a provisioning factor of 1x the grid size that would
/// achieve the maximum active blocks. The launch bounds can be overwritten by
/// setting `threads`, `blocks` and `provision`.
template<typename T>
inline launch_params get_benchmark_launch_parameters(T           kernel,
                                                     const char* kernel_name    = "unnamed_kernel",
                                                     int         user_threads   = 0,
                                                     int         user_blocks    = 0,
                                                     int         user_provision = 1)
{
    launch_params params{};

    int dev_id;
    HIP_CHECK(hipGetDevice(&dev_id));

    HIP_CHECK(hipDeviceGetAttribute(&params.multiprocessors,
                                    hipDeviceAttributeMultiprocessorCount,
                                    dev_id));

    HIP_CHECK(hipDeviceGetAttribute(&params.max_threads_per_block,
                                    hipDeviceAttributeMaxThreadsPerBlock,
                                    dev_id));

    if(user_threads > 0)
    {
        // Sanity check for user threads not exceeding device limit
        if(user_threads > params.max_threads_per_block)
        {
            fprintf(stderr,
                    "[Error] User threads (%d) exceed device limit (%d) for the current GPU\n",
                    user_threads,
                    params.max_threads_per_block);
            exit(EXIT_FAILURE);
        }

        params.threads = user_threads;
        HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&params.occupancy,
                                                               kernel,
                                                               params.threads,
                                                               0));

        // Sanity check for zero occupancy with user provided threads
        if(params.occupancy == 0)
        {
            fprintf(stderr,
                    "[Error] Kernel %s cannot be launched with %d threads due to zero occupancy.\n",
                    kernel_name,
                    params.threads);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        // Heuristic that picks thread count that maximizes occupancy
        hipFuncAttributes attr;
        HIP_CHECK(hipFuncGetAttributes(&attr, (const void*)kernel));
        for(int t = 32; t <= attr.maxThreadsPerBlock; t *= 2)
        {
            if(t > params.max_threads_per_block)
                continue;

            int current_occupancy = 0;
            HIP_CHECK(
                hipOccupancyMaxActiveBlocksPerMultiprocessor(&current_occupancy, kernel, t, 0));

            // Prefer configurations with higher occupancy.
            // If occupancy is equal, prefer larger block sizes to increase in-flight threads.
            if(current_occupancy > params.occupancy
               || (current_occupancy == params.occupancy && t > params.threads))
            {
                // Sanity check for threads not exceeding device limit
                if(t > params.max_threads_per_block)
                {
                    fprintf(stderr,
                            "[Error] Occupancy helper: computed threads (%d) exceed device limit "
                            "(%d) for the current GPU\n",
                            t,
                            params.max_threads_per_block);
                    exit(EXIT_FAILURE);
                }

                params.threads   = t;
                params.occupancy = current_occupancy;
            }
        }
    }

    // Check if user specified blocks beyond default 0
    if(user_blocks > 0)
    {
        params.blocks = user_blocks;
    }
    else
    {
        // Multiplied by provision factor
        params.blocks = params.occupancy * params.multiprocessors * user_provision;
    }

    //Sanity check for zero occupancy and zero threads
    if(params.threads == 0 || params.occupancy == 0)
    {
        fprintf(stderr, "[Error] Kernel %s: No valid thread configuration found.\n", kernel_name);
        exit(EXIT_FAILURE);
    }

    return params;
}

#endif
