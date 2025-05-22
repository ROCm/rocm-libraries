/*
 *  Copyright© 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/optional.h>

#include <cstdint>
#include <type_traits>

#include "test_header.hpp"

TEST(OptionalTests, IsTriviallyCopyable)
{
  static_assert(std::is_trivially_copyable<uint64_t>::value == true,
                "type is not trivially copyable even though it should be!");
  static_assert(std::is_trivially_copyable<thrust::optional<uint64_t>>::value == true,
                "type is not trivially copyable even though it should be!");
}

TEST(OptionalTests, EmplaceReference)
{
  // See https://github.com/ROCm/rocThrust/issues/404
  {
    int a = 10;

    thrust::optional<int&> maybe(a);

    int b = 20;
    maybe.emplace(b);

    ASSERT_EQ(maybe.value(), 20);
    // Emplacing with b shouldn't change a
    ASSERT_EQ(a, 10);

    int c = 30;
    maybe.emplace(c);

    ASSERT_EQ(maybe.value(), 30);
    ASSERT_EQ(b, 20);
  }

  {
    thrust::optional<int&> maybe;

    int b = 21;
    maybe.emplace(b);

    ASSERT_EQ(maybe.value(), 21);

    int c = 31;
    maybe.emplace(c);

    ASSERT_EQ(maybe.value(), 31);
    ASSERT_EQ(b, 21);
  }
}
