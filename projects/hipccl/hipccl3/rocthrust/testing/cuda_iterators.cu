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

struct plus_one
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr int operator()(const int val) const noexcept
  {
    return val + 1;
  }
};

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
