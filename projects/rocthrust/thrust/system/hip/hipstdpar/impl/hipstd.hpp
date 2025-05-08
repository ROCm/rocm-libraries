/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file thrust/system/hip/hipstdpar/impl/hipstd.hpp
 *  \brief hipstd utilities implementation detail header for HIPSTDPAR.
 */

#ifndef THRUST_SYSTEM_HIP_HIPSTDPAR_HIPSTD_HPP
#define THRUST_SYSTEM_HIP_HIPSTDPAR_HIPSTD_HPP

#pragma once

#if defined(__HIPSTDPAR__)

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hipstd
{
template <typename... Cs>
inline constexpr bool is_offloadable_callable() noexcept
{
    return std::conjunction_v<std::negation<std::is_pointer<Cs>>...,
                              std::negation<std::is_member_function_pointer<Cs>>...>;
}

template <typename I, typename = void>
struct Is_offloadable_iterator : std::false_type
{
};
template <typename I>
struct Is_offloadable_iterator<
    I,
    std::void_t<decltype(std::declval<I>() < std::declval<I>()),
                decltype(std::declval<I&>() += std::declval<std::ptrdiff_t>()),
                decltype(std::declval<I>() + std::declval<std::ptrdiff_t>()),
                decltype(std::declval<I>()[std::declval<std::ptrdiff_t>()]),
                decltype(*std::declval<I>())>> : std::true_type
{
};

template <typename... Is>
inline constexpr bool is_offloadable_iterator() noexcept
{
#if defined(__cpp_lib_concepts)
    return (... && std::random_access_iterator<Is>);
#else
    return std::conjunction_v<Is_offloadable_iterator<Is>...>;
#endif
}

template <typename... Cs>
inline constexpr
    __attribute__((diagnose_if(true,
                               "HIP Standard Parallelism does not support passing pointers to "
                               "function as callable arguments, execution will not be "
                               "offloaded.",
                               "warning"))) void
    unsupported_callable_type() noexcept
{
}

template <typename... Is>
inline constexpr
    __attribute__((diagnose_if(true,
                               "HIP Standard Parallelism requires random access iterators, "
                               "execution will not be offloaded.",
                               "warning"))) void
    unsupported_iterator_category() noexcept
{
}
}
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__

#endif // THRUST_SYSTEM_HIP_HIPSTDPAR_HIPSTD_HPP

