/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstdint>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

using PairScanByKeyTestParams = ::testing::Types<Params<int8_t>, Params<int16_t>, Params<int32_t>>;

TESTS_DEFINE(PairScanByKeyTests, PairScanByKeyTestParams);

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
    // Need cast to undo integer promotion, decltype(char{} + char{}) == int
    using P1T1 = typename Pair1::first_type;
    using P1T2 = typename Pair1::second_type;
    return thrust::make_pair(static_cast<P1T1>(x.first + y.first), static_cast<P1T2>(x.second + y.second));
  } // end operator()
}; // end add_pairs

TYPED_TEST(PairScanByKeyTests, TestPairScanByKey)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using T = typename TestFixture::input_type;
  using P = thrust::pair<T, T>;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_p1 = random_integers<T>(size);
    thrust::host_vector<T> h_p2 = random_integers<T>(size);
    thrust::host_vector<P> h_pairs(size);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    thrust::device_vector<T> d_p1    = h_p1;
    thrust::device_vector<T> d_p2    = h_p2;
    thrust::device_vector<P> d_pairs = h_pairs;

    thrust::host_vector<T> h_keys   = random_integers<bool>(size);
    thrust::device_vector<T> d_keys = h_keys;

    P init = thrust::make_pair(T{13}, T{13});

    // scan on the host
    thrust::exclusive_scan_by_key(
      h_keys.begin(), h_keys.end(), h_pairs.begin(), h_pairs.begin(), init, thrust::equal_to<T>(), add_pairs());

    // scan on the device
    thrust::exclusive_scan_by_key(
      d_keys.begin(), d_keys.end(), d_pairs.begin(), d_pairs.begin(), init, thrust::equal_to<T>(), add_pairs());

    ASSERT_EQ_QUIET(h_pairs, d_pairs);
  }
}
