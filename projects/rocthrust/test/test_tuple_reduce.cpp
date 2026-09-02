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

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(TupleReduceTests, IntegerTestsParams);

struct SumTupleFunctor
{
  template <typename Tuple>
  THRUST_HOST_DEVICE Tuple operator()(const Tuple& lhs, const Tuple& rhs)
  {
    using thrust::get;

    return thrust::make_tuple(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
  }
};

struct MakeTupleFunctor
{
  template <typename T1, typename T2>
  THRUST_HOST_DEVICE thrust::tuple<T1, T2> operator()(T1& lhs, T2& rhs)
  {
    return thrust::make_tuple(lhs, rhs);
  }
};

TYPED_TEST(TupleReduceTests, TestTupleReduce)
{
  using T = typename TestFixture::input_type;
  using namespace thrust;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      host_vector<T> h_t1 = get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      host_vector<T> h_t2 = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);

      // zip up the data
      host_vector<tuple<T, T>> h_tuples(size);
      transform(h_t1.begin(), h_t1.end(), h_t2.begin(), h_tuples.begin(), MakeTupleFunctor());

      // copy to device
      device_vector<tuple<T, T>> d_tuples = h_tuples;

      tuple<T, T> zero(0, 0);

      // sum on host
      tuple<T, T> h_result = thrust::reduce(h_tuples.begin(), h_tuples.end(), zero, SumTupleFunctor());

      // sum on device
      tuple<T, T> d_result = thrust::reduce(d_tuples.begin(), d_tuples.end(), zero, SumTupleFunctor());

      ASSERT_EQ_QUIET(h_result, d_result);
    }
  }
}
