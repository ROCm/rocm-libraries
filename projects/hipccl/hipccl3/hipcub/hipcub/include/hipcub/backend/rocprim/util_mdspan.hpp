/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_UTIL_MDSPAN_HPP_
#define HIPCUB_ROCPRIM_UTIL_MDSPAN_HPP_

#include "../../config.hpp"
#include "util_type.hpp"

#include _HIPCUB_STD_INCLUDE(__mdspan/extents.h)

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
/**
 * \brief Different type comparison helper
 * This is a trait wrapper, in c++20, there is an equivalence
 */
template<class T, class U>
constexpr bool cmp_equal(T t, U u) noexcept
{
    if constexpr(rocprim::is_signed<T>::value == rocprim::is_signed<U>::value)
        return t == u;
    else if constexpr(rocprim::is_signed<T>::value)
        return t >= 0 && typename rocprim::make_unsigned<T>::type(t) == u;
    else
        return u >= 0 && typename rocprim::make_unsigned<U>::type(u) == t;
}

/**
 * \brief Different type comparison helper
 * This is a trait wrapper, in c++20, there is an equivalence
 */
template<class T, class U>
constexpr bool cmp_less(T t, U u) noexcept
{
    if constexpr(rocprim::is_signed<T>::value == rocprim::is_signed<U>::value)
        return t < u;
    else if constexpr(rocprim::is_signed<T>::value)
        return t < 0 || typename rocprim::make_unsigned<T>::type(t) < u;
    else
        return u >= 0 && t < typename rocprim::make_unsigned<U>::type(u);
}

/**
 * \brief Different type comparison helper
 */
template<class T, class U>
constexpr bool cmp_less_equal(T t, U u) noexcept
{
    return ::hipcub::detail::cmp_less(t, u) || ::hipcub::detail::cmp_equal(t, u);
}

/**
 * \brief Different type comparison helper
 */
template<class T, class U>
constexpr bool cmp_greater_equal(T t, U u) noexcept
{
    return !::hipcub::detail::cmp_less(t, u);
}

/**
 * \brief Different type comparison helper
 */
template<class T, class U>
constexpr bool in_range(U t) noexcept
{
    return ::hipcub::detail::cmp_greater_equal(t, ::rocprim::numeric_limits<T>::min())
           && ::hipcub::detail::cmp_less_equal(t, ::rocprim::numeric_limits<T>::max());
}

} // namespace detail

/**
 * \brief Helper template, which removes the first extent of `ExtentsType`.
 */
template<class ExtentsType>
struct extents_remove_first
{};

template<class IndexType, size_t First, size_t... Extents>
struct extents_remove_first<_HIPCUB_STD::extents<IndexType, First, Extents...>>
{
    using type = _HIPCUB_STD::extents<IndexType, Extents...>;
};

/**
 * \brief Helper template, to get the total size of `ExtentsType`
 * For example, if you have `_HIPCUB_STD::extents<int,4,3,2>`,
 * it returns 24 (4 * 3 * 2). 
 */
template<class ExtentsType>
struct extents_size
{
    static constexpr size_t value = 1;
};

template<class IndexType, size_t First, size_t... Extents>
struct extents_size<_HIPCUB_STD::extents<IndexType, First, Extents...>>
{
    static constexpr size_t value
        = First * extents_size<_HIPCUB_STD::extents<IndexType, Extents...>>::value;
};

/**
 * \brief Helper template, returns the size of the sub extents of 
 * `ExtentsType`, starting from rank number `StartRank`.
 */
template<size_t StartRank, class ExtentsType, class Enable = void>
struct extents_sub_size
{};

template<size_t StartRank, class ExtentsType>
struct extents_sub_size<StartRank, ExtentsType, typename std::enable_if<StartRank == 0>::type>
{
    constexpr static size_t value = extents_size<ExtentsType>::value;
};

template<size_t StartRank, class ExtentsType>
struct extents_sub_size<StartRank, ExtentsType, typename std::enable_if<StartRank != 0>::type>
{
    constexpr static size_t value
        = extents_sub_size<StartRank - 1, typename extents_remove_first<ExtentsType>::type>::value;
};

/** @} */ // end group UtilModule

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_UTIL_MDSPAN_HPP_
