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

#include <thrust/functional.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

using VariableTestParams =
  ::testing::Types<Params<signed char>,
                   Params<unsigned char>,
                   Params<short>,
                   Params<unsigned short>,
                   Params<int>,
                   Params<unsigned int>,
                   Params<float>>;

TESTS_DEFINE(SetDifferenceKeyValueTest, VariableTestParams);

TYPED_TEST(SetDifferenceKeyValueTest, TestSetDifferenceKeyValue)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using U = typename TestFixture::input_type;
  using T = key_value<U, U>;

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<U> h_keys_a   = random_integers<U>(size);
    thrust::host_vector<U> h_values_a = random_integers<U>(size);

    thrust::host_vector<U> h_keys_b   = random_integers<U>(size);
    thrust::host_vector<U> h_values_b = random_integers<U>(size);

    thrust::host_vector<T> h_a(size), h_b(size);
    for (size_t i = 0; i < size; ++i)
    {
      h_a[i] = T(h_keys_a[i], h_values_a[i]);
      h_b[i] = T(h_keys_b[i], h_values_b[i]);
    }

    thrust::stable_sort(h_a.begin(), h_a.end());
    thrust::stable_sort(h_b.begin(), h_b.end());

    thrust::device_vector<T> d_a = h_a;
    thrust::device_vector<T> d_b = h_b;

    thrust::host_vector<T> h_result(size);
    thrust::device_vector<T> d_result(size);

    typename thrust::host_vector<T>::iterator h_end;
    typename thrust::device_vector<T>::iterator d_end;

    h_end = thrust::set_difference(h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
    h_result.resize(h_end - h_result.begin());

    d_end = thrust::set_difference(d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

    d_result.resize(d_end - d_result.begin());

    ASSERT_EQ_QUIET(h_result, d_result);
  }
}
