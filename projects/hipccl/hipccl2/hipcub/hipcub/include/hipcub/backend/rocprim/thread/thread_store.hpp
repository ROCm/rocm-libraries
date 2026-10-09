/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#include "../../../config.hpp"
#include "../util_type.hpp"

#include <rocprim/thread/thread_store.hpp> // IWYU pragma: export

#include <stdint.h>
#include <type_traits>

BEGIN_HIPCUB_NAMESPACE

enum CacheStoreModifier
{
    STORE_DEFAULT  = 0, ///< Default (no modifier)
    STORE_WB       = 1, ///< Cache write-back all coherent levels
    STORE_CG       = 2, ///< Cache at global level
    STORE_CS       = 3, ///< Cache streaming (likely to be accessed once)
    STORE_WT       = 4, ///< Cache write-through (to system memory)
    STORE_VOLATILE = 5 ///< Volatile shared (any memory space)
};

template<typename T, typename Fundamental>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void ThreadStoreVolatilePtr(T* ptr, T val, Fundamental /*is_fundamental*/)
{
    rocprim::thread_store<rocprim::store_volatile, T>(ptr, val);
}

template<int MODIFIER, typename T>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void ThreadStore(T* ptr,
                                    T  val,
                                    detail::int_constant_t<MODIFIER> /*modifier*/,
                                    ::std::true_type /*is_pointer*/)
{
    rocprim::thread_store<rocprim::cache_store_modifier(MODIFIER), T>(ptr, val);
}

template<int MODIFIER, typename OutputIteratorT, typename T>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void ThreadStore(OutputIteratorT itr,
                                    T               val,
                                    detail::int_constant_t<MODIFIER> /*modifier*/,
                                    ::std::false_type /*is_pointer*/)
{
    ThreadStore<MODIFIER>(&(*itr), val, detail::int_constant_t<MODIFIER>{}, ::std::true_type{});
}

template<CacheStoreModifier MODIFIER = STORE_DEFAULT, typename OutputIteratorT, typename T>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void ThreadStore(OutputIteratorT itr, T val)
{
    ThreadStore(itr,
                val,
                detail::int_constant_t<MODIFIER>{},
                ::std::bool_constant<::std::is_pointer<OutputIteratorT>::value>());
}

namespace detail
{

/// Helper structure for templated store iteration (inductive case)
template<int COUNT, int MAX>
struct iterate_thread_store
{
    template<CacheStoreModifier MODIFIER, typename T>
    static HIPCUB_DEVICE
    HIPCUB_FORCEINLINE void Store(T* ptr, T* vals)
    {
        ThreadStore<MODIFIER>(ptr + COUNT, vals[COUNT]);
        iterate_thread_store<COUNT + 1, MAX>::template Store<MODIFIER>(ptr, vals);
    }

    template<typename OutputIteratorT, typename T>
    static HIPCUB_DEVICE
    HIPCUB_FORCEINLINE void Dereference(OutputIteratorT ptr, T* vals)
    {
        ptr[COUNT] = vals[COUNT];
        iterate_thread_store<COUNT + 1, MAX>::Dereference(ptr, vals);
    }
};

/// Helper structure for templated store iteration (termination case)
template<int MAX>
struct iterate_thread_store<MAX, MAX>
{
    template<CacheStoreModifier MODIFIER, typename T>
    static HIPCUB_DEVICE
    HIPCUB_FORCEINLINE void Store(T* /*ptr*/, T* /*vals*/)
    {}

    template<typename OutputIteratorT, typename T>
    static HIPCUB_DEVICE
    HIPCUB_FORCEINLINE void Dereference(OutputIteratorT /*ptr*/, T* /*vals*/)
    {}
};

} // namespace detail

template<int COUNT, int MAX>
using IterateThreadStore HIPCUB_DEPRECATED = detail::iterate_thread_store<COUNT, MAX>;

END_HIPCUB_NAMESPACE
#endif
