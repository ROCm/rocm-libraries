/*
 *  Copyright 2026 NVIDIA Corporation
 *  Modifications Copyright© 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
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

#include <thrust/detail/type_traits/minimum_type.h>
#include <thrust/iterator/iterator_categories.h>

#include <cuda/std/type_traits>

#include <unittest/unittest.h>

template <typename... Ts>
using mt = thrust::detail::minimum_type<Ts...>;

template <typename, typename... Ts>
inline constexpr bool mt_fails_impl = true;

template <typename... Ts>
inline constexpr bool mt_fails_impl<::cuda::std::void_t<mt<Ts...>>, Ts...> = false;

template <typename... Ts>
inline constexpr bool mt_fails = mt_fails_impl<void, Ts...>;

struct A
{};
struct B : A
{};
struct C : B
{};
struct C2 : B
{};

void MinimumType()
{
  using ::cuda::std::is_same_v;

  ASSERT_EQUAL(true, (is_same_v<mt<int>, int>) );
  ASSERT_EQUAL(true, (is_same_v<mt<int, int>, int>) );
  ASSERT_EQUAL(true, (is_same_v<mt<int, int, int, int>, int>) );

  ASSERT_EQUAL(true, (is_same_v<mt<char, short, int>, char>) );
  ASSERT_EQUAL(true, (is_same_v<mt<int, short, char>, int>) );

  ASSERT_EQUAL(true, (is_same_v<mt<A, B, C>, A>) );
  ASSERT_EQUAL(true, (is_same_v<mt<C, B, A>, A>) );

  ASSERT_EQUAL(true, (is_same_v<mt<A, B, C>, A>) );
  ASSERT_EQUAL(true, (is_same_v<mt<C, B, A>, A>) );
  ASSERT_EQUAL(true, (is_same_v<mt<C, B, A, B, C>, A>) );

  ASSERT_EQUAL(
    true,
    (is_same_v<
      mt<::cuda::std::random_access_iterator_tag, ::cuda::std::input_iterator_tag, ::cuda::std::forward_iterator_tag>,
      ::cuda::std::input_iterator_tag>) );
  ASSERT_EQUAL(true,
               (is_same_v<mt<::cuda::std::random_access_iterator_tag,
                             ::cuda::std::random_access_iterator_tag,
                             ::cuda::std::random_access_iterator_tag>,
                          ::cuda::std::random_access_iterator_tag>) );
  ASSERT_EQUAL(true, (mt_fails<C, C2>) );
  ASSERT_EQUAL(true, (mt_fails<int, A>) );
  ASSERT_EQUAL(true, (mt_fails<int, A, B, C>) );
  ASSERT_EQUAL(true, (mt_fails<A, B, C, int>) );
}
DECLARE_UNITTEST(MinimumType);
