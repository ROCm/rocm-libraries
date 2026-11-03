/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/transform.h>

#include _THRUST_STD_INCLUDE(tuple)

#include <unittest/unittest.h>

using namespace unittest;

struct MakeTupleFunctor
{
  template <typename T1, typename T2>
  THRUST_HOST_DEVICE _THRUST_STD::tuple<T1, T2> operator()(T1& lhs, T2& rhs)
  {
    return _THRUST_STD::tuple(lhs, rhs);
  }
};

template <int N>
struct GetFunctor
{
  template <typename Tuple>
  THRUST_HOST_DEVICE typename _THRUST_STD::tuple_element<N, Tuple>::type operator()(const Tuple& t)
  {
    return _THRUST_STD::get<N>(t);
  }
};

template <typename T>
struct TestTupleTransform
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_t1 = random_integers<T>(n);
    thrust::host_vector<T> h_t2 = random_integers<T>(n);

    // zip up the data
    thrust::host_vector<_THRUST_STD::tuple<T, T>> h_tuples(n);
    thrust::transform(h_t1.begin(), h_t1.end(), h_t2.begin(), h_tuples.begin(), MakeTupleFunctor());

    // copy to device
    thrust::device_vector<_THRUST_STD::tuple<T, T>> d_tuples = h_tuples;

    thrust::device_vector<T> d_t1(n), d_t2(n);

    // select 0th
    thrust::transform(d_tuples.begin(), d_tuples.end(), d_t1.begin(), GetFunctor<0>());

    // select 1st
    thrust::transform(d_tuples.begin(), d_tuples.end(), d_t2.begin(), GetFunctor<1>());

    ASSERT_ALMOST_EQUAL(h_t1, d_t1);
    ASSERT_ALMOST_EQUAL(h_t2, d_t2);

    ASSERT_EQUAL_QUIET(h_tuples, d_tuples);
  }
};
VariableUnitTest<TestTupleTransform, SignedIntegralTypes> TestTupleTransformInstance;
