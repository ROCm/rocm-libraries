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

#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include "test_header.hpp"

TESTS_DEFINE(ZipIteratorReduceTests, IntegerTestsParams);

template <typename Tuple>
struct TuplePlus
{
  __host__ __device__ Tuple operator()(Tuple x, Tuple y) const
  {
    using namespace thrust;
    return make_tuple(get<0>(x) + get<0>(y), get<1>(x) + get<1>(y));
  }
}; // end SumTuple

TYPED_TEST(ZipIteratorReduceTests, TestZipIteratorReduce)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data0 =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::host_vector<T> h_data1 = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);

      thrust::device_vector<T> d_data0 = h_data0;
      thrust::device_vector<T> d_data1 = h_data1;

      using Tuple = thrust::tuple<T, T>;

      // run on host
      Tuple h_result = thrust::reduce(
        thrust::make_zip_iterator(thrust::make_tuple(h_data0.begin(), h_data1.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(h_data0.end(), h_data1.end())),
        thrust::make_tuple<T, T>(0, 0),
        TuplePlus<Tuple>());

      // run on device
      Tuple d_result = thrust::reduce(
        thrust::make_zip_iterator(thrust::make_tuple(d_data0.begin(), d_data1.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_data0.end(), d_data1.end())),
        thrust::make_tuple<T, T>(0, 0),
        TuplePlus<Tuple>());

      ASSERT_EQ(thrust::get<0>(h_result), thrust::get<0>(d_result));
      ASSERT_EQ(thrust::get<1>(h_result), thrust::get<1>(d_result));
    }
  }
}
