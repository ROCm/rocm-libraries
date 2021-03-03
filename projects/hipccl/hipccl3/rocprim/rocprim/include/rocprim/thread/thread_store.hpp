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

#ifndef ROCPRIM_THREAD_THREAD_STORE_HPP_
#define ROCPRIM_THREAD_THREAD_STORE_HPP_


#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

enum cache_store_modifier
{
    store_default,              ///< Default (no modifier)
    store_wb,                   ///< Cache write-back all coherent levels
    store_cg,                   ///< Cache at global level
    store_cs,                   ///< Cache streaming (likely to be accessed once)
    store_wt,                   ///< Cache write-through (to system memory)
    store_volatile,             ///< Volatile shared (any memory space)
};

template <
    cache_store_modifier MODIFIER = store_default,
    typename OutputIteratorT,
    typename T
>
ROCPRIM_DEVICE inline void thread_store(
    OutputIteratorT itr,
    T               val)
{
    thread_store<MODIFIER>(&(*itr), val);
}

template <
    cache_store_modifier MODIFIER = store_default,
    typename T
>
ROCPRIM_DEVICE inline void thread_store(
    T *ptr,
    T val)
{
    __builtin_memcpy(ptr, &val, sizeof(T));
}

END_ROCPRIM_NAMESPACE

#endif
