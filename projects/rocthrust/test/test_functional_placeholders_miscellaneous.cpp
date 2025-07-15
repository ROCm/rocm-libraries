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
#include <thrust/transform.h>

#include <utility>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

using ThirtyTwoBitVectorTypes =
  ::testing::Types<Params<thrust::host_vector<int>>,
                   Params<thrust::host_vector<unsigned int>>,
                   Params<thrust::host_vector<float>>,
                   Params<thrust::device_vector<int>>,
                   Params<thrust::device_vector<unsigned int>>,
                   Params<thrust::device_vector<float>>>;

TESTS_DEFINE(FunctionalPlaceholdersMiscellaneousTests, ThirtyTwoBitVectorTypes);

template <typename T>
struct saxpy_reference
{
  THRUST_HOST_DEVICE saxpy_reference(const T& aa)
      : a(aa)
  {}

  THRUST_HOST_DEVICE T operator()(const T& x, const T& y) const
  {
    return a * x + y;
  }

  T a;
};

TYPED_TEST(FunctionalPlaceholdersMiscellaneousTests, TestFunctionalPlaceholdersValue)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t n = 10000;
  using Vector   = typename TestFixture::input_type;
  using T        = typename Vector::value_type;

  T a(13);

  Vector x = random_integers<T>(n);
  Vector y = random_integers<T>(n);
  Vector result(n), reference(n);

  thrust::transform(x.begin(), x.end(), y.begin(), reference.begin(), saxpy_reference<T>(a));

  using namespace thrust::placeholders;
  thrust::transform(x.begin(), x.end(), y.begin(), result.begin(), a * _1 + _2);

  ASSERT_EQ(reference, result);
}

TYPED_TEST(FunctionalPlaceholdersMiscellaneousTests, TestFunctionalPlaceholdersTransformIterator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const size_t n = 10000;
  using Vector   = typename TestFixture::input_type;
  using T        = typename Vector::value_type;

  T a(13);

  Vector x = random_integers<T>(n);
  Vector y = random_integers<T>(n);
  Vector result(n), reference(n);

  thrust::transform(x.begin(), x.end(), y.begin(), reference.begin(), saxpy_reference<T>(a));

  using namespace thrust::placeholders;
  thrust::transform(
    thrust::make_transform_iterator(x.begin(), a * _1),
    thrust::make_transform_iterator(x.end(), a * _1),
    y.begin(),
    result.begin(),
    _1 + _2);

  ASSERT_EQ(reference, result);
}

TEST(FunctionalPlaceholdersMiscellaneousTests, TestFunctionalPlaceholdersArgumentValueCategories)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust::placeholders;
  auto expr = _1 * _1 + _2 * _2;
  int a     = 2;
  int b     = 3;
  ASSERT_EQ(expr(2, 3), 13); // pass pr-value
  ASSERT_EQ(expr(a, b), 13); // pass l-value
  ASSERT_EQ(expr(::std::move(a), ::std::move(b)), 13); // pass x-value
}

TEST(FunctionalPlaceholdersMiscellaneousTests, TestFunctionalPlaceholdersSemiRegular)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust::placeholders;
  using Expr = decltype(_1 * _1 + _2 * _2);
  Expr expr; // default-constructible
  ASSERT_EQ(expr(2, 3), 13);
  Expr expr2 = expr; // copy-constructible
  ASSERT_EQ(expr2(2, 3), 13);
  Expr expr3;
  expr3 = expr; // copy-assignable
  ASSERT_EQ(expr3(2, 3), 13);
}
