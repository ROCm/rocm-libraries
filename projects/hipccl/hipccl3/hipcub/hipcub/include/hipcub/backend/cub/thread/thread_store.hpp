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

#ifndef HIPCUB_BACKEND_CUB_THREAD_STORE_HPP_
#define HIPCUB_BACKEND_CUB_THREAD_STORE_HPP_

#include "../../../config.hpp"
#include "../util_type.hpp"

#include <cub/thread/thread_store.cuh> // CUB thread store

#include <cstdint>
#include <type_traits>

BEGIN_HIPCUB_NAMESPACE

enum CacheStoreModifier
{
    STORE_DEFAULT  = 0,
    STORE_WB       = 1,
    STORE_CG       = 2,
    STORE_CS       = 3,
    STORE_WT       = 4,
    STORE_VOLATILE = 5
};

template<int MOD>
struct cub_cache_store_modifier_map
{
    static constexpr ::cub::CacheStoreModifier value = static_cast<::cub::CacheStoreModifier>(MOD);
};

template<typename T, typename Fundamental>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void ThreadStoreVolatilePtr(T* ptr, T val, Fundamental /*is_fundamental*/)
{
    ::cub::ThreadStore<::cub::STORE_VOLATILE>(ptr, val);
}

template<int MODIFIER, typename T>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void ThreadStore(T* ptr,
                                    T  val,
                                    ::std::integral_constant<int, MODIFIER> /*modifier*/,
                                    ::std::true_type /*is_pointer*/)
{
    ::cub::ThreadStore<cub_cache_store_modifier_map<MODIFIER>::value>(ptr, val);
}

template<int MODIFIER, typename OutputIteratorT, typename T>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void ThreadStore(OutputIteratorT itr,
                                    T               val,
                                    ::std::integral_constant<int, MODIFIER> /*modifier*/,
                                    ::std::false_type /*is_pointer*/)
{
    ThreadStore<MODIFIER>(&(*itr),
                          val,
                          ::std::integral_constant<int, MODIFIER>{},
                          ::std::true_type{});
}

template<CacheStoreModifier MODIFIER = STORE_DEFAULT, typename OutputIteratorT, typename T>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void ThreadStore(OutputIteratorT itr, T val)
{
    ThreadStore(itr,
                val,
                ::std::integral_constant<int, MODIFIER>{},
                ::std::bool_constant<_HIPCUB_STD::is_pointer<OutputIteratorT>::value>());
}

namespace detail
{

template<int COUNT, int MAX>
struct iterate_thread_store
{
    template<CacheStoreModifier MODIFIER, typename T>
    static HIPCUB_DEVICE
    HIPCUB_FORCEINLINE void Store(T* ptr, T* vals)
    {
        ThreadStore<MODIFIER>(ptr + COUNT,
                              vals[COUNT],
                              ::std::integral_constant<int, MODIFIER>{},
                              ::std::true_type{});
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

END_HIPCUB_NAMESPACE

#endif // HIPCUB_BACKEND_CUB_THREAD_STORE_HPP_
