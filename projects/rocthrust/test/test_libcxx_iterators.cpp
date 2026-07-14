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

#if _THRUST_HAS_DEVICE_SYSTEM_STD

#  include _THRUST_LIBCXX_INCLUDE(iterator)

#  include <vector>

#  include "test_utils.hpp"

TEST(LibcxxIteratorTests, TestLibcxxDiscardIterator)
{
  auto discard = _THRUST_LIBCXX::discard_iterator{};
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(vec.begin(), vec.end(), discard);
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(vec.begin(), vec.end(), discard);
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(vec.begin(), vec.end(), discard);
  }
}

TEST(LibcxxIteratorTests, TestLibcxxConstantIterator)
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(_THRUST_LIBCXX::constant_iterator{42, 0}, _THRUST_LIBCXX::constant_iterator{42, 4}, vec.begin());
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(_THRUST_LIBCXX::constant_iterator{42, 0}, _THRUST_LIBCXX::constant_iterator{42, 4}, vec.begin());
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(_THRUST_LIBCXX::constant_iterator{42, 0}, _THRUST_LIBCXX::constant_iterator{42, 4}, vec.begin());
  }
}

TEST(LibcxxIteratorTests, TestLibcxxCountingIterator)
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(_THRUST_LIBCXX::counting_iterator{0}, _THRUST_LIBCXX::counting_iterator{4}, vec.begin());
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(_THRUST_LIBCXX::counting_iterator{0}, _THRUST_LIBCXX::counting_iterator{4}, vec.begin());
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(_THRUST_LIBCXX::counting_iterator{0}, _THRUST_LIBCXX::counting_iterator{4}, vec.begin());
  }
}

TEST(LibcxxIteratorTests, TestLibcxxStridedIterator)
{
  auto discard = _THRUST_LIBCXX::discard_iterator{};
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4, 5, 6};
    thrust::copy(
      _THRUST_LIBCXX::strided_iterator{vec.begin(), 2}, _THRUST_LIBCXX::strided_iterator{vec.end(), 2}, discard);
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4, 5, 6};
    thrust::copy(
      _THRUST_LIBCXX::strided_iterator{vec.begin(), 2}, _THRUST_LIBCXX::strided_iterator{vec.end(), 2}, discard);
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4, 5, 6};
    thrust::copy(
      _THRUST_LIBCXX::strided_iterator{vec.begin(), 2}, _THRUST_LIBCXX::strided_iterator{vec.end(), 2}, discard);
  }
}

struct is_equal_index
{
  THRUST_HOST_DEVICE constexpr void
  operator()([[maybe_unused]] const int index, [[maybe_unused]] const int expected) const noexcept
  {
    _CCCL_VERIFY(index == expected, "should have right value");
  }
};

TEST(LibcxxIteratorTests, TestLibcxxTabulateOutputIterator)
{
  { // device system
    thrust::device_vector<int> vec{5, 6, 7, 8, 9};
    thrust::copy(vec.begin(), vec.end(), _THRUST_LIBCXX::make_tabulate_output_iterator(is_equal_index{}, 5));
  }

  { // host system
    thrust::host_vector<int> vec{5, 6, 7, 8, 9};
    thrust::copy(vec.begin(), vec.end(), _THRUST_LIBCXX::make_tabulate_output_iterator(is_equal_index{}, 5));
  }

  { // plain std::vector
    std::vector<int> vec{5, 6, 7, 8, 9};
    thrust::copy(vec.begin(), vec.end(), _THRUST_LIBCXX::make_tabulate_output_iterator(is_equal_index{}, 5));
  }
}

struct plus_one
{
  [[nodiscard]] THRUST_HOST_DEVICE constexpr int operator()(const int val) const noexcept
  {
    return val + 1;
  }
};

TEST(LibcxxIteratorTests, TestLibcxxTransformOutputIterator)
{
  { // device system
    thrust::device_vector<int> vec{-1, -1, -1, -1, -1};
    thrust::copy(_THRUST_LIBCXX::counting_iterator{0},
                 _THRUST_LIBCXX::counting_iterator{5},
                 _THRUST_LIBCXX::make_transform_output_iterator(vec.begin(), plus_one{}));
    thrust::device_vector<int> expected{1, 2, 3, 4, 5};
    ASSERT_TRUE(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{-1, -1, -1, -1, -1};
    thrust::copy(_THRUST_LIBCXX::counting_iterator{0},
                 _THRUST_LIBCXX::counting_iterator{5},
                 _THRUST_LIBCXX::make_transform_output_iterator(vec.begin(), plus_one{}));
    thrust::host_vector<int> expected{1, 2, 3, 4, 5};
    ASSERT_TRUE(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{-1, -1, -1, -1, -1};
    thrust::copy(_THRUST_LIBCXX::counting_iterator{0},
                 _THRUST_LIBCXX::counting_iterator{5},
                 _THRUST_LIBCXX::make_transform_output_iterator(vec.begin(), plus_one{}));
    std::vector<int> expected{1, 2, 3, 4, 5};
    ASSERT_TRUE(thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }
}

TEST(LibcxxIteratorTests, TestLibcxxTransformIterator)
{
  auto discard = _THRUST_LIBCXX::discard_iterator{};
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(_THRUST_LIBCXX::make_transform_iterator(vec.begin(), plus_one{}),
                 _THRUST_LIBCXX::make_transform_iterator(vec.end(), plus_one{}),
                 discard);
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(_THRUST_LIBCXX::make_transform_iterator(vec.begin(), plus_one{}),
                 _THRUST_LIBCXX::make_transform_iterator(vec.end(), plus_one{}),
                 discard);
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(_THRUST_LIBCXX::make_transform_iterator(vec.begin(), plus_one{}),
                 _THRUST_LIBCXX::make_transform_iterator(vec.end(), plus_one{}),
                 discard);
  }
}

#endif
