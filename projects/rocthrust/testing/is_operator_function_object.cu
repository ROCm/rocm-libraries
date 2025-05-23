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

#include <thrust/detail/static_assert.h>
#include <thrust/type_traits/is_operator_less_or_greater_function_object.h>
#include <thrust/type_traits/is_operator_plus_function_object.h>

#include <unittest/unittest.h>

#if THRUST_CPP_DIALECT >= 2017
THRUST_STATIC_ASSERT((thrust::is_operator_less_function_object<std::less<>>::value));

THRUST_STATIC_ASSERT((thrust::is_operator_greater_function_object<std::greater<>>::value));

THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<std::less<>>::value));

THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<std::greater<>>::value));

THRUST_STATIC_ASSERT((thrust::is_operator_plus_function_object<std::plus<>>::value));
#endif

template <typename T>
THRUST_HOST void test_is_operator_less_function_object()
{
  THRUST_STATIC_ASSERT((thrust::is_operator_less_function_object<thrust::less<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<thrust::greater<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<thrust::less_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<thrust::greater_equal<T>>::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_less_function_object<std::less<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<std::greater<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<std::less_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<std::greater_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_function_object<T>::value));
}
DECLARE_GENERIC_UNITTEST(test_is_operator_less_function_object);

template <typename T>
THRUST_HOST void test_is_operator_greater_function_object()
{
  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<thrust::less<T>>::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_greater_function_object<thrust::greater<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<thrust::less_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<thrust::greater_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<std::less<T>>::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_greater_function_object<std::greater<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<std::less_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<std::greater_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_greater_function_object<T>::value));
}
DECLARE_GENERIC_UNITTEST(test_is_operator_greater_function_object);

template <typename T>
THRUST_HOST void test_is_operator_less_or_greater_function_object()
{
  THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<thrust::less<T>>::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<thrust::greater<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<thrust::less_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<thrust::greater_equal<T>>::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<std::less<T>>::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_less_or_greater_function_object<std::greater<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<std::less_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<std::greater_equal<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_less_or_greater_function_object<T>::value));
}
DECLARE_GENERIC_UNITTEST(test_is_operator_less_or_greater_function_object);

template <typename T>
THRUST_HOST void test_is_operator_plus_function_object()
{
  THRUST_STATIC_ASSERT((thrust::is_operator_plus_function_object<thrust::plus<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<thrust::minus<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<thrust::less<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<thrust::greater<T>>::value));

  THRUST_STATIC_ASSERT((thrust::is_operator_plus_function_object<std::plus<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<std::minus<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<std::less<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<std::greater<T>>::value));

  THRUST_STATIC_ASSERT((!thrust::is_operator_plus_function_object<T>::value));
}
DECLARE_GENERIC_UNITTEST(test_is_operator_plus_function_object);
