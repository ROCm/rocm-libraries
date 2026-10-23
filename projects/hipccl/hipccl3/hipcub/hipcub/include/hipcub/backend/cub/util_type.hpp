/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.
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
#include "../../util_deprecated.hpp"

#include _HIPCUB_STD_INCLUDE(iterator)
#include _HIPCUB_STD_INCLUDE(type_traits)

#include <cub/util_type.cuh> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
// the following iterator helpers are not named iter_value_t etc, like the C++20 facilities, because they are defined in
// terms of C++17 iterator_traits and not the new C++20 indirectly_readable trait etc. This allows them to detect nested
// value_type, difference_type and reference aliases, which the new C+20 traits do not consider (they only consider
// specializations of iterator_traits). Also, a value_type of void remains supported (needed by some output iterators).

template<typename It, typename = void>
struct it_traits
{
    using value_type      = typename _HIPCUB_STD::iterator_traits<It>::value_type;
    using reference       = typename _HIPCUB_STD::iterator_traits<It>::reference;
    using difference_type = typename _HIPCUB_STD::iterator_traits<It>::difference_type;
    using pointer         = typename _HIPCUB_STD::iterator_traits<It>::pointer;
};
template<typename It>
struct it_traits<It,
                 std::void_t<typename It::value_type,
                             typename It::reference,
                             typename It::difference_type,
                             typename It::pointer>>
{
    using value_type      = typename It::value_type;
    using reference       = typename It::reference;
    using difference_type = typename It::difference_type;
    using pointer         = typename It::pointer;
};
template<typename It>
using it_value_t = typename it_traits<It>::value_type;
template<typename It>
using it_reference_t = typename it_traits<It>::reference;
template<typename It>
using it_difference_t = typename it_traits<It>::difference_type;
template<typename It>
using it_pointer_t = typename it_traits<It>::pointer;

// use this whenever you need to lazily evaluate a trait. E.g., as an alternative in replace_if_use_default.
template<template<typename...> typename Trait, typename... Args>
struct lazy_trait
{
    using type = Trait<Args...>;
};

template<int Value>
using int_constant_t = _HIPCUB_STD::integral_constant<int, Value>;

} // namespace detail

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CUB_UTIL_TYPE_HPP_
