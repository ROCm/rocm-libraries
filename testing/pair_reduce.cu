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

#include <unittest/unittest.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

struct make_pair_functor
{
  template <typename T1, typename T2>
  THRUST_HOST_DEVICE thrust::pair<T1, T2> operator()(const T1& x, const T2& y)
  {
    return thrust::make_pair(x,y);
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
    return thrust::make_pair(static_cast<P1T1>(x.first + y.first),
                             static_cast<P1T2>(x.second + y.second));
  } // end operator()
}; // end add_pairs


template <typename T>
  struct TestPairReduce
{
  void operator()(const size_t n)
  {
    using P = thrust::pair<T, T>;

    thrust::host_vector<T>   h_p1 = unittest::random_integers<T>(n);
    thrust::host_vector<T>   h_p2 = unittest::random_integers<T>(n);
    thrust::host_vector<P>   h_pairs(n);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    thrust::device_vector<T> d_p1 = h_p1;
    thrust::device_vector<T> d_p2 = h_p2;
    thrust::device_vector<P> d_pairs = h_pairs;

    P init = thrust::make_pair(T{13}, T{13});

    // reduce on the host
    P h_result = thrust::reduce(h_pairs.begin(), h_pairs.end(), init, add_pairs());

    // reduce on the device
    P d_result = thrust::reduce(d_pairs.begin(), d_pairs.end(), init, add_pairs());

    ASSERT_EQUAL_QUIET(h_result, d_result);
  }
}; // end TestPairReduce
VariableUnitTest<TestPairReduce, SignedIntegralTypes> TestPairReduceInstance;

