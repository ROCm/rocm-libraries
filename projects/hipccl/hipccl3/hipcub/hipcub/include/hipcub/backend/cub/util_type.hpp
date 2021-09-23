/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_CUB_UTIL_TYPE_HPP_
#define HIPCUB_CUB_UTIL_TYPE_HPP_

#include "../../config.hpp"

#include <cub/util_type.cuh>

#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !__NVCOMPILER_CUDA__
    #include <cuda_bf16.h>
#endif

//TODO: Remove this section after CUB update

/// Optional outer namespace(s)
CUB_NS_PREFIX

namespace cub {

#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !__NVCOMPILER_CUDA__

template <>
struct FpLimits<__nv_bfloat16>
{
    static __host__ __device__ __forceinline__ __nv_bfloat16 Max() {
        unsigned short max_word = 0x7F7F;
        return reinterpret_cast<__nv_bfloat16&>(max_word);
    }

    static __host__ __device__ __forceinline__ __nv_bfloat16 Lowest() {
        unsigned short lowest_word = 0xFF7F;
        return reinterpret_cast<__nv_bfloat16&>(lowest_word);
    }
};

#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000) && !__NVCOMPILER_CUDA__
    template <> struct NumericTraits<__nv_bfloat16> :   BaseTraits<FLOATING_POINT, true, false, unsigned short, __nv_bfloat16> {};
#endif

#endif

}
CUB_NS_POSTFIX  // Optional outer namespace(s)

#endif // HIPCUB_CUB_UTIL_TYPE_HPP_
