// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the >Software>), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED >AS IS>, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPCUB_CONFIG_HPP_
#define HIPCUB_CONFIG_HPP_

#include <hip/hip_runtime.h>

#define HIPCUB_NAMESPACE hipcub

#define BEGIN_HIPCUB_NAMESPACE \
    namespace hipcub {

#define END_HIPCUB_NAMESPACE \
    } /* hipcub */

#ifdef __HIP_PLATFORM_HCC__
    #ifndef ROCPRIM_HIP_API
        #define ROCPRIM_HIP_API
    #endif // ROCPRIM_HIP_API
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
