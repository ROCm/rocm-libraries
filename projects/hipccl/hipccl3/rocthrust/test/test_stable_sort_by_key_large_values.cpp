/*
 *  Copyright 2008-2023 NVIDIA Corporation
 *  Modifications Copyright© 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/functional.h>
#include <thrust/sort.h>

#include "test_header.hpp"

template <typename T>
struct greater_div_10
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const
  {
    return ((int)lhs) / 10 > ((int)rhs) / 10;
  }
};

template <unsigned int N>
void _TestStableSortByKeyWithLargeValues()
{
  size_t n = (128 * 1024) / sizeof(FixedVector<int, N>);

  thrust::host_vector<unsigned int> h_keys(n);
  thrust::host_vector<FixedVector<int, N>> h_vals(n);

  for (size_t i = 0; i < n; i++)
  {
    const auto uint_i   = static_cast<unsigned int>(i);
    const auto rand_int = unittest::generate_random_integer<unsigned int>()(uint_i);
    h_keys[i]           = rand_int;
    h_vals[i]           = FixedVector<int, N>(static_cast<int>(i));
  }

  thrust::device_vector<unsigned int> d_keys        = h_keys;
  thrust::device_vector<FixedVector<int, N>> d_vals = h_vals;

  thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_vals.begin());
  thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

  ASSERT_EQ_QUIET(h_keys, d_keys);
  ASSERT_EQ_QUIET(h_vals, d_vals);

  // so cuda::stable_merge_sort_by_key() is called
  thrust::stable_sort_by_key(h_keys.begin(),
                             h_keys.end(),
                             h_vals.begin(),
                             greater_div_10<unsigned int>());
  thrust::stable_sort_by_key(d_keys.begin(),
                             d_keys.end(),
                             d_vals.begin(),
                             greater_div_10<unsigned int>());

  ASSERT_EQ_QUIET(h_keys, d_keys);
  ASSERT_EQ_QUIET(h_vals, d_vals);
}

TEST(StableSortByKeyLargeTests, TestStableSortByKeyWithLargeValues)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  _TestStableSortByKeyWithLargeValues<4>();
  _TestStableSortByKeyWithLargeValues<8>();
  _TestStableSortByKeyWithLargeValues<16>();
}
