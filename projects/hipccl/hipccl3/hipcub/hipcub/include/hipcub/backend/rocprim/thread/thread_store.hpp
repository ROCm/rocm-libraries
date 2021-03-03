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

#ifndef HIPCUB_ROCPRIM_THREAD_THREAD_STORE_HPP_
#define HIPCUB_ROCPRIM_THREAD_THREAD_STORE_HPP_
BEGIN_HIPCUB_NAMESPACE

enum CacheStoreModifier
{
    STORE_DEFAULT,              ///< Default (no modifier)
    STORE_WB,                   ///< Cache write-back all coherent levels
    STORE_CG,                   ///< Cache at global level
    STORE_CS,                   ///< Cache streaming (likely to be accessed once)
    STORE_WT,                   ///< Cache write-through (to system memory)
    STORE_VOLATILE,             ///< Volatile shared (any memory space)
};

template <
    CacheStoreModifier MODIFIER = STORE_DEFAULT,
    typename OutputIteratorT,
    typename T
>
__device__ __forceinline__ void ThreadStore(
    OutputIteratorT itr,
    T               val)
{
    ThreadStore<MODIFIER>(&(*itr), val);
}

template <
    CacheStoreModifier MODIFIER = STORE_DEFAULT,
    typename T
>
__device__ __forceinline__ void ThreadStore(
    T *ptr,
    T val)
{
    __builtin_memcpy(ptr, &val, sizeof(T));
}

END_HIPCUB_NAMESPACE
#endif
