/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>

#include <unittest/unittest.h>

void TestEqualRangeOnStream()
{ // Regression test for GH issue #921 (nvbug 2173437)
  using vector_t   = typename thrust::device_vector<int>;
  using iterator_t = typename vector_t::iterator;
  using result_t   = thrust::pair<iterator_t, iterator_t>;

  vector_t input(10);
  thrust::sequence(thrust::device, input.begin(), input.end(), 0);
  cudaStream_t stream = 0;
  result_t result     = thrust::equal_range(thrust::cuda::par.on(stream), input.begin(), input.end(), 5);

  ASSERT_EQUAL(5, thrust::distance(input.begin(), result.first));
  ASSERT_EQUAL(6, thrust::distance(input.begin(), result.second));
}
DECLARE_UNITTEST(TestEqualRangeOnStream);
