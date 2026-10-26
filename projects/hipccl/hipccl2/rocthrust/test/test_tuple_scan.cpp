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

#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

using IntegralTypes = ::testing::Types<
  Params<char>,
  Params<signed char>,
  Params<unsigned char>,
  Params<short>,
  Params<unsigned short>,
  Params<int>,
  Params<unsigned int>,
  Params<long>,
  Params<unsigned long>,
  Params<long long>,
  Params<unsigned long long>>;

TESTS_DEFINE(TupleScanTests, IntegralTypes);

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

TYPED_TEST(TupleScanTests, TestTupleScan)
{
  using T = typename TestFixture::input_type;
  using namespace thrust;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    host_vector<T> h_t1 = random_integers<T>(size);
    host_vector<T> h_t2 = random_integers<T>(size);

    // initialize input
    host_vector<tuple<T, T>> h_input(size);
    transform(h_t1.begin(), h_t1.end(), h_t2.begin(), h_input.begin(), MakeTupleFunctor());
    device_vector<tuple<T, T>> d_input = h_input;

    // allocate output
    tuple<T, T> zero(0, 0);
    host_vector<tuple<T, T>> h_output(size, zero);
    device_vector<tuple<T, T>> d_output(size, zero);

    // inclusive_scan
    thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), SumTupleFunctor());
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), SumTupleFunctor());
    ASSERT_EQ_QUIET(h_output, d_output);

    // exclusive_scan
    tuple<T, T> init(13, 17);
    thrust::exclusive_scan(h_input.begin(), h_input.end(), h_output.begin(), init, SumTupleFunctor());
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), init, SumTupleFunctor());

    ASSERT_EQ_QUIET(h_output, d_output);
  }
}
