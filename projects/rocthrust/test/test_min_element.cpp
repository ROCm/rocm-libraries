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

#include <thrust/extrema.h>
#include <thrust/iterator/retag.h>
#include <thrust/universal_vector.h>

#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

using VectorTestsParams = ::testing::Types<
  Params<thrust::host_vector<signed char>>,
  Params<thrust::host_vector<short>>,
  Params<thrust::host_vector<int>>,
  Params<thrust::host_vector<float>>,
  Params<thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>>,
  Params<thrust::device_vector<signed char>>,
  Params<thrust::device_vector<short>>,
  Params<thrust::device_vector<int>>,
  Params<thrust::device_vector<float>>,
  Params<thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>>,
  Params<thrust::universal_vector<int>>,
  Params<thrust::universal_host_pinned_vector<int>>>;

TESTS_DEFINE(MinElementTests, FullTestsParams);
TESTS_DEFINE(MinElementPrimitiveTests, NumericalTestsParams);
TESTS_DEFINE(MinElementVectorUnitTests, VectorTestsParams);

TYPED_TEST(MinElementTests, TestMinElementSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{3, 5, 1, 2, 5, 1};

  ASSERT_EQ(*thrust::min_element(data.begin(), data.end()), 1);
  ASSERT_EQ(thrust::min_element(data.begin(), data.end()) - data.begin(), 2);

  ASSERT_EQ(*thrust::min_element(data.begin(), data.end(), thrust::greater<T>()), 5);
  ASSERT_EQ(thrust::min_element(data.begin(), data.end(), thrust::greater<T>()) - data.begin(), 1);
}

TYPED_TEST(MinElementVectorUnitTests, TestMinElementWithTransform)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{3, 5, 1, 2, 5, 1};

  ASSERT_EQ(*thrust::min_element(thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
                                 thrust::make_transform_iterator(data.end(), thrust::negate<T>())),
            -5);
  ASSERT_EQ(*thrust::min_element(thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
                                 thrust::make_transform_iterator(data.end(), thrust::negate<T>()),
                                 thrust::greater<T>()),
            -1);
}

TYPED_TEST(MinElementPrimitiveTests, TestMinElement)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> h_data =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_data = h_data;

      typename thrust::host_vector<T>::iterator h_min   = thrust::min_element(h_data.begin(), h_data.end());
      typename thrust::device_vector<T>::iterator d_min = thrust::min_element(d_data.begin(), d_data.end());

      ASSERT_EQ(h_min - h_data.begin(), d_min - d_data.begin());

      typename thrust::host_vector<T>::iterator h_max =
        thrust::min_element(h_data.begin(), h_data.end(), thrust::greater<T>());
      typename thrust::device_vector<T>::iterator d_max =
        thrust::min_element(d_data.begin(), d_data.end(), thrust::greater<T>());

      ASSERT_EQ(h_max - h_data.begin(), d_max - d_data.begin());
    }
  }
}

template <typename ForwardIterator>
ForwardIterator min_element(my_system& system, ForwardIterator first, ForwardIterator)
{
  system.validate_dispatch();
  return first;
}

TEST(MinElementTests, TestMinElementDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::min_element(sys, vec.begin(), vec.end());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator>
ForwardIterator min_element(my_tag, ForwardIterator first, ForwardIterator)
{
  *first = 13;
  return first;
}

TEST(MinElementTests, TestMinElementDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::min_element(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

  ASSERT_EQ(13, vec.front());
}

void TestMinElementWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(1);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQ(thrust::distance(begin, end), 1ll << magnitude);

  ASSERT_EQ(*thrust::min_element(thrust::device, begin, end, thrust::greater<long long>()), (1ll << magnitude));
}

TEST(MinElementTests, TestMinElementWithBigIndexes)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  TestMinElementWithBigIndexesHelper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestMinElementWithBigIndexesHelper(31);
  TestMinElementWithBigIndexesHelper(32);
  TestMinElementWithBigIndexesHelper(33);
#endif
}
