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

#ifndef HIPCUB_ROCPRIM_THREAD_THREAD_LOAD_HPP_
#define HIPCUB_ROCPRIM_THREAD_THREAD_LOAD_HPP_

#include "../../../config.hpp"
#include "../util_type.hpp"

#include <rocprim/thread/thread_load.hpp> // IWYU pragma: export

#include <iterator>
#include <stdint.h>
#include <type_traits>

BEGIN_HIPCUB_NAMESPACE

enum CacheLoadModifier : int32_t
{
    LOAD_DEFAULT  = 0, ///< Default (no modifier)
    LOAD_CA       = 1, ///< Cache at all levels
    LOAD_CG       = 2, ///< Cache at global level
    LOAD_CS       = 3, ///< Cache streaming (likely to be accessed once)
    LOAD_CV       = 4, ///< Cache as volatile (including cached system lines)
    LOAD_LDG      = 5, ///< Cache as texture
    LOAD_VOLATILE = 6 ///< Volatile (any memory space)
};

template<int Count, CacheLoadModifier MODIFIER, typename T>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void UnrolledThreadLoad(T const* src, T* dst)
{
    rocprim::unrolled_thread_load<Count, rocprim::cache_load_modifier(MODIFIER)>(
        const_cast<T*>(src),
        dst);
}

template<int Count, typename InputIteratorT, typename T>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE void UnrolledCopy(InputIteratorT src, T* dst)
{
    rocprim::unrolled_copy<Count>(src, dst);
}

template<typename T, typename Fundamental>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE T ThreadLoadVolatilePointer(T* ptr, Fundamental /* is_fundamental*/)
{
    return rocprim::thread_load<rocprim::load_volatile>(ptr);
}

template<int MODIFIER, typename InputIteratorT>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE typename std::iterator_traits<InputIteratorT>::value_type
    ThreadLoad(InputIteratorT itr,
               detail::int_constant_t<MODIFIER> /*modifier*/,
               ::std::false_type /*is_pointer*/)
{
    return rocprim::thread_load<rocprim::cache_load_modifier(MODIFIER)>(itr);
}

template<int MODIFIER, typename T>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE T ThreadLoad(T* ptr,
                                detail::int_constant_t<MODIFIER> /*modifier*/,
                                ::std::true_type /*is_pointer*/)
{
    return rocprim::thread_load<rocprim::cache_load_modifier(MODIFIER)>(ptr);
}

template<CacheLoadModifier MODIFIER = LOAD_DEFAULT, typename InputIteratorT>
HIPCUB_DEVICE
HIPCUB_FORCEINLINE
    typename std::iterator_traits<InputIteratorT>::value_type ThreadLoad(InputIteratorT itr)
{
    return ThreadLoad(itr,
                      detail::int_constant_t<MODIFIER>(),
                      ::std::bool_constant<::std::is_pointer<InputIteratorT>::value>());
}

END_HIPCUB_NAMESPACE
#endif
