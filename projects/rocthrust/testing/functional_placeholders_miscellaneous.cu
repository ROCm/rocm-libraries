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

#include <thrust/functional.h>
#include <thrust/transform.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <cuda/std/utility>
#elif defined(__has_include)
#  if __has_include(<cuda/std/utility>)
#    include <cuda/std/utility>
#  endif // __has_include(<cuda/std/utility>)
#endif // THRUST_DEVICE_SYSTEM
#include <utility>

#include <unittest/unittest.h>

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

template <typename Vector>
struct TestFunctionalPlaceholdersValue
{
  void operator()(const size_t)
  {
    const size_t n = 10000;
    using T        = typename Vector::value_type;

    T a(13);

    Vector x = unittest::random_integers<T>(n);
    Vector y = unittest::random_integers<T>(n);
    Vector result(n), reference(n);

    thrust::transform(x.begin(), x.end(), y.begin(), reference.begin(), saxpy_reference<T>(a));

    using namespace thrust::placeholders;
    thrust::transform(x.begin(), x.end(), y.begin(), result.begin(), a * _1 + _2);

    ASSERT_ALMOST_EQUAL(reference, result);
  }
};
VectorUnitTest<TestFunctionalPlaceholdersValue, ThirtyTwoBitTypes, thrust::device_vector, thrust::device_allocator>
  TestFunctionalPlaceholdersValueDevice;
VectorUnitTest<TestFunctionalPlaceholdersValue, ThirtyTwoBitTypes, thrust::host_vector, std::allocator>
  TestFunctionalPlaceholdersValueHost;

template <typename Vector>
struct TestFunctionalPlaceholdersTransformIterator
{
  void operator()(const size_t)
  {
    const size_t n = 10000;
    using T        = typename Vector::value_type;

    T a(13);

    Vector x = unittest::random_integers<T>(n);
    Vector y = unittest::random_integers<T>(n);
    Vector result(n), reference(n);

    thrust::transform(x.begin(), x.end(), y.begin(), reference.begin(), saxpy_reference<T>(a));

    using namespace thrust::placeholders;
    thrust::transform(
      thrust::make_transform_iterator(x.begin(), a * _1),
      thrust::make_transform_iterator(x.end(), a * _1),
      y.begin(),
      result.begin(),
      _1 + _2);

    ASSERT_ALMOST_EQUAL(reference, result);
  }
};
VectorUnitTest<TestFunctionalPlaceholdersTransformIterator,
               ThirtyTwoBitTypes,
               thrust::device_vector,
               thrust::device_allocator>
  TestFunctionalPlaceholdersTransformIteratorInstanceDevice;
VectorUnitTest<TestFunctionalPlaceholdersTransformIterator, ThirtyTwoBitTypes, thrust::host_vector, std::allocator>
  TestFunctionalPlaceholdersTransformIteratorInstanceHost;

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
void TestFunctionalPlaceholdersArgumentValueCategories()
{
  using namespace thrust::placeholders;
  auto expr = _1 * _1 + _2 * _2;
  int a     = 2;
  int b     = 3;
  ASSERT_EQUAL(expr(2, 3), 13); // pass pr-value
  ASSERT_EQUAL(expr(a, b), 13); // pass l-value
  ASSERT_EQUAL(expr(::cuda::std::move(a), ::cuda::std::move(b)), 13); // pass x-value
}
DECLARE_UNITTEST(TestFunctionalPlaceholdersArgumentValueCategories);

void TestFunctionalPlaceholdersSemiRegular()
{
  using namespace thrust::placeholders;
  using Expr = decltype(_1 * _1 + _2 * _2);
  Expr expr; // default-constructible
  ASSERT_EQUAL(expr(2, 3), 13);
  Expr expr2 = expr; // copy-constructible
  ASSERT_EQUAL(expr2(2, 3), 13);
  Expr expr3;
  expr3 = expr; // copy-assignable
  ASSERT_EQUAL(expr3(2, 3), 13);

#  if _CCCL_STD_VER >= 2014
  static_assert(::cuda::std::semiregular<Expr>, "");
#  endif // _CCCL_STD_VER >= 2014
}
DECLARE_UNITTEST(TestFunctionalPlaceholdersSemiRegular);
#endif
