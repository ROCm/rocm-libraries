/*
 *  Copyright 2026 NVIDIA Corporation
 *  Modifications Copyright© 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"));
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

#include _THRUST_STD_INCLUDE(type_traits)

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

template <typename... Ts>
using mt = thrust::detail::minimum_type<Ts...>;

template <typename, typename... Ts>
inline constexpr bool mt_fails_impl = true;

template <typename... Ts>
inline constexpr bool mt_fails_impl<_THRUST_STD::void_t<mt<Ts...>>, Ts...> = false;

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

TEST(MinimumTypeTests, MinimumType)
{
  using _THRUST_STD::is_same_v;

  EXPECT_TRUE((is_same_v<mt<int>, int>) );
  EXPECT_TRUE((is_same_v<mt<int, int>, int>) );
  EXPECT_TRUE((is_same_v<mt<int, int, int, int>, int>) );

  EXPECT_TRUE((is_same_v<mt<char, short, int>, char>) );
  EXPECT_TRUE((is_same_v<mt<int, short, char>, int>) );

  EXPECT_TRUE((is_same_v<mt<A, B, C>, A>) );
  EXPECT_TRUE((is_same_v<mt<C, B, A>, A>) );

  EXPECT_TRUE((is_same_v<mt<A, B, C>, A>) );
  EXPECT_TRUE((is_same_v<mt<C, B, A>, A>) );
  EXPECT_TRUE((is_same_v<mt<C, B, A, B, C>, A>) );

  EXPECT_TRUE(
    (is_same_v<
      mt<_THRUST_STD::random_access_iterator_tag, _THRUST_STD::input_iterator_tag, _THRUST_STD::forward_iterator_tag>,
      _THRUST_STD::input_iterator_tag>) );
  EXPECT_TRUE((is_same_v<mt<_THRUST_STD::random_access_iterator_tag,
                            _THRUST_STD::random_access_iterator_tag,
                            _THRUST_STD::random_access_iterator_tag>,
                         _THRUST_STD::random_access_iterator_tag>) );

  EXPECT_TRUE((mt_fails<C, C2>) );
  EXPECT_TRUE((mt_fails<int, A>) );
  EXPECT_TRUE((mt_fails<int, A, B, C>) );
  EXPECT_TRUE((mt_fails<A, B, C, int>) );
}
