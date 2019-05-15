/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_CONFIG_HPP_
#define HIPCUB_CONFIG_HPP_

#ifndef HIPCUB_MAIN_HEADER_INCLUDED
    // Unlike CUB, hipCUB does not support including particular headers,
    // because it is a wrapper of two backends
    #error "Include only the main header file: #include <hipcub/hipcub.hpp>"
#endif

#include <hip/hip_runtime.h>

#define HIPCUB_NAMESPACE hipcub

#define BEGIN_HIPCUB_NAMESPACE \
    namespace hipcub {

#define END_HIPCUB_NAMESPACE \
    } /* hipcub */

#ifdef __HIP_PLATFORM_HCC__
    #include <rocprim/rocprim.hpp>

    #define HIPCUB_ROCPRIM_API 1
    #define HIPCUB_DEVICE __device__
    #define HIPCUB_HOST __host__
    #define HIPCUB_HOST_DEVICE __host__ __device__
    #define HIPCUB_RUNTIME_FUNCTION __host__
    #define HIPCUB_SHARED_MEMORY __shared__
#elif defined(__HIP_PLATFORM_NVCC__)
    // Block
    #include <cub/block/block_histogram.cuh>
    #include <cub/block/block_discontinuity.cuh>
    #include <cub/block/block_exchange.cuh>
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_radix_rank.cuh>
    #include <cub/block/block_radix_sort.cuh>
    #include <cub/block/block_reduce.cuh>
    #include <cub/block/block_scan.cuh>
    #include <cub/block/block_store.cuh>

    // Thread
    #include <cub/thread/thread_load.cuh>
    #include <cub/thread/thread_operators.cuh>
    #include <cub/thread/thread_reduce.cuh>
    #include <cub/thread/thread_scan.cuh>
    #include <cub/thread/thread_store.cuh>

    // Warp
    #include <cub/warp/warp_reduce.cuh>
    #include <cub/warp/warp_scan.cuh>

    // Iterator
    #include <cub/iterator/arg_index_input_iterator.cuh>
    #include <cub/iterator/cache_modified_input_iterator.cuh>
    #include <cub/iterator/cache_modified_output_iterator.cuh>
    #include <cub/iterator/constant_input_iterator.cuh>
    #include <cub/iterator/counting_input_iterator.cuh>
    #include <cub/iterator/tex_obj_input_iterator.cuh>
    #include <cub/iterator/tex_ref_input_iterator.cuh>
    #include <cub/iterator/transform_input_iterator.cuh>

    // Util
    #include <cub/util_arch.cuh>
    #include <cub/util_debug.cuh>
    #include <cub/util_device.cuh>
    #include <cub/util_macro.cuh>
    #include <cub/util_ptx.cuh>
    #include <cub/util_type.cuh>

    #define HIPCUB_CUB_API 1
    #define HIPCUB_DEVICE __device__
    #define HIPCUB_HOST __host__
    #define HIPCUB_HOST_DEVICE __host__ __device__
    #define HIPCUB_RUNTIME_FUNCTION CUB_RUNTIME_FUNCTION
    #define HIPCUB_SHARED_MEMORY __shared__
#endif

#endif // HIPCUB_CONFIG_HPP_
