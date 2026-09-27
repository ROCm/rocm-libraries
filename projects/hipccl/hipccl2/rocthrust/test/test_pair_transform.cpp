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

#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <type_traits>
#endif

TESTS_DEFINE(PairTransformTests, NumericalTestsParams);

struct make_pair_functor
{
  template <typename T1, typename T2>
  THRUST_HOST_DEVICE thrust::pair<T1, T2> operator()(const T1& x, const T2& y)
  {
    return thrust::make_pair(x, y);
  } // end operator()()
}; // end make_pair_functor

struct add_pairs
{
  template <typename Pair1, typename Pair2>
  THRUST_HOST_DEVICE Pair1 operator()(const Pair1& x, const Pair2& y)
  {
    using T1 = typename _THRUST_STD::common_type<typename Pair1::first_type, typename Pair2::first_type>::type;
    using T2 = typename _THRUST_STD::common_type<typename Pair1::second_type, typename Pair2::second_type>::type;

    return thrust::make_pair(static_cast<T1>(x.first + y.first), static_cast<T2>(x.second + y.second));
  } // end operator()
}; // end add_pairs

TYPED_TEST(PairTransformTests, TestPairTransform)
{
  using T = typename TestFixture::input_type;
  using P = thrust::pair<T, T>;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_p1 =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::host_vector<T> h_p2 = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);
      thrust::host_vector<P> h_result(size);

      thrust::device_vector<T> d_p1 = h_p1;
      thrust::device_vector<T> d_p2 = h_p2;
      thrust::device_vector<P> d_result(size);

      // zip up pairs on the host
      thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_result.begin(), make_pair_functor());

      // zip up pairs on the device
      thrust::transform(d_p1.begin(), d_p1.end(), d_p2.begin(), d_result.begin(), make_pair_functor());

      ASSERT_EQ_QUIET(h_result, d_result);

      // add pairs on the host
      thrust::transform(h_result.begin(), h_result.end(), h_result.begin(), h_result.begin(), add_pairs());

      // add pairs on the device
      thrust::transform(d_result.begin(), d_result.end(), d_result.begin(), d_result.begin(), add_pairs());

      ASSERT_EQ_QUIET(h_result, d_result);
    }
  }
}; // end TestPairZip
