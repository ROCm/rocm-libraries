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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/iterator>

#include <vector>

#include <unittest/unittest.h>

void TestCUDADiscardIterator()
{
  auto discard = cuda::discard_iterator{};
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
DECLARE_UNITTEST(TestCUDADiscardIterator);

void TestCUDAConstantIterator()
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::constant_iterator{42, 0}, cuda::constant_iterator{42, 4}, vec.begin());
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::constant_iterator{42, 0}, cuda::constant_iterator{42, 4}, vec.begin());
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::constant_iterator{42, 0}, cuda::constant_iterator{42, 4}, vec.begin());
  }
}
DECLARE_UNITTEST(TestCUDAConstantIterator);

void TestCUDACountingIterator()
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::counting_iterator{0}, cuda::counting_iterator{4}, vec.begin());
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::counting_iterator{0}, cuda::counting_iterator{4}, vec.begin());
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::counting_iterator{0}, cuda::counting_iterator{4}, vec.begin());
  }
}
DECLARE_UNITTEST(TestCUDACountingIterator);

void TestCUDAPermutationIterator()
{
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    thrust::device_vector<int> off{5, 2, 7, 0};
    thrust::device_vector<int> res{-1, -1, -1, -1, -1};
    thrust::copy(cuda::permutation_iterator{vec.begin(), off.begin()},
                 cuda::permutation_iterator{vec.begin(), off.end()},
                 res.begin());
    ASSERT_EQUAL(res, (thrust::device_vector<int>{6, 3, 8, 1, -1}));
  }
  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    thrust::host_vector<int> off{5, 2, 7, 0};
    thrust::host_vector<int> res{-1, -1, -1, -1, -1};
    thrust::copy(cuda::permutation_iterator{vec.begin(), off.begin()},
                 cuda::permutation_iterator{vec.begin(), off.end()},
                 res.begin());
    ASSERT_EQUAL(res, (thrust::host_vector<int>{6, 3, 8, 1, -1}));
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> off{5, 2, 7, 0};
    std::vector<int> res{-1, -1, -1, -1, -1};
    thrust::copy(cuda::permutation_iterator{vec.begin(), off.begin()},
                 cuda::permutation_iterator{vec.begin(), off.end()},
                 res.begin());
    ASSERT_EQUAL(res, (std::vector<int>{6, 3, 8, 1, -1}));
  }
}
DECLARE_UNITTEST(TestCUDAPermutationIterator);

void TestCUDAStridedIterator()
{
  auto discard = cuda::discard_iterator{};
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4, 5, 6};
    thrust::copy(cuda::strided_iterator{vec.begin(), 2}, cuda::strided_iterator{vec.end(), 2}, discard);
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4, 5, 6};
    thrust::copy(cuda::strided_iterator{vec.begin(), 2}, cuda::strided_iterator{vec.end(), 2}, discard);
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4, 5, 6};
    thrust::copy(cuda::strided_iterator{vec.begin(), 2}, cuda::strided_iterator{vec.end(), 2}, discard);
  }
}
DECLARE_UNITTEST(TestCUDAStridedIterator);

struct is_equal_index
{
  _CCCL_HOST_DEVICE constexpr void
  operator()([[maybe_unused]] const int index, [[maybe_unused]] const int expected) const noexcept
  {
    _CCCL_VERIFY(index == expected, "should have right value");
  }
};

void TestCUDATabulateOutputIterator()
{
  { // device system
    thrust::device_vector<int> vec{5, 6, 7, 8, 9};
    thrust::copy(vec.begin(), vec.end(), cuda::make_tabulate_output_iterator(is_equal_index{}, 5));
  }

  { // host system
    thrust::host_vector<int> vec{5, 6, 7, 8, 9};
    thrust::copy(vec.begin(), vec.end(), cuda::make_tabulate_output_iterator(is_equal_index{}, 5));
  }

  { // plain std::vector
    std::vector<int> vec{5, 6, 7, 8, 9};
    thrust::copy(vec.begin(), vec.end(), cuda::make_tabulate_output_iterator(is_equal_index{}, 5));
  }
}
DECLARE_UNITTEST(TestCUDATabulateOutputIterator);

struct plus_one
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr int operator()(const int val) const noexcept
  {
    return val + 1;
  }
};

void TestCUDATransformOutputIterator()
{
  { // device system
    thrust::device_vector<int> vec{-1, -1, -1, -1, -1};
    thrust::copy(cuda::counting_iterator{0},
                 cuda::counting_iterator{5},
                 cuda::make_transform_output_iterator(vec.begin(), plus_one{}));
    thrust::device_vector<int> expected{1, 2, 3, 4, 5};
    ASSERT_EQUAL(true, thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // host system
    thrust::host_vector<int> vec{-1, -1, -1, -1, -1};
    thrust::copy(cuda::counting_iterator{0},
                 cuda::counting_iterator{5},
                 cuda::make_transform_output_iterator(vec.begin(), plus_one{}));
    thrust::host_vector<int> expected{1, 2, 3, 4, 5};
    ASSERT_EQUAL(true, thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }

  { // plain std::vector
    std::vector<int> vec{-1, -1, -1, -1, -1};
    thrust::copy(cuda::counting_iterator{0},
                 cuda::counting_iterator{5},
                 cuda::make_transform_output_iterator(vec.begin(), plus_one{}));
    std::vector<int> expected{1, 2, 3, 4, 5};
    ASSERT_EQUAL(true, thrust::equal(vec.begin(), vec.end(), expected.begin()));
  }
}
DECLARE_UNITTEST(TestCUDATransformOutputIterator);

void TestCUDATransformIterator()
{
  auto discard = cuda::discard_iterator{};
  { // device system
    thrust::device_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::make_transform_iterator(vec.begin(), plus_one{}),
                 cuda::make_transform_iterator(vec.end(), plus_one{}),
                 discard);
  }

  { // host system
    thrust::host_vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::make_transform_iterator(vec.begin(), plus_one{}),
                 cuda::make_transform_iterator(vec.end(), plus_one{}),
                 discard);
  }

  { // plain std::vector
    std::vector<int> vec{1, 2, 3, 4};
    thrust::copy(cuda::make_transform_iterator(vec.begin(), plus_one{}),
                 cuda::make_transform_iterator(vec.end(), plus_one{}),
                 discard);
  }
}
DECLARE_UNITTEST(TestCUDATransformIterator);
