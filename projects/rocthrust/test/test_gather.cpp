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

#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/sequence.h>

#include <algorithm>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(GatherTests, FullTestsParams);
TESTS_DEFINE(PrimitiveGatherTests, NumericalTestsParams);

TEST(GatherTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

THRUST_DIAG_PUSH
THRUST_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

TYPED_TEST(GatherTests, TestGatherSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector map{6, 2, 1, 7, 2}; // gather indices
  Vector src{0, 1, 2, 3, 4, 5, 6, 7}; // source vector
  Vector dst(5, 0); // destination vector

  thrust::gather(map.begin(), map.end(), src.begin(), dst.begin());

  Vector ref{6, 2, 1, 7, 2};
  ASSERT_EQ(dst, ref);
}

template <typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
OutputIterator gather(my_system& system, InputIterator, InputIterator, RandomAccessIterator, OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(GatherTests, TestGatherDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::gather(sys, vec.begin(), vec.end(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename RandomAccessIterator, typename OutputIterator>
OutputIterator gather(my_tag, InputIterator, InputIterator, RandomAccessIterator, OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(GatherTests, TestGatherDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::gather(thrust::retag<my_tag>(vec.begin()),
                 thrust::retag<my_tag>(vec.end()),
                 thrust::retag<my_tag>(vec.begin()),
                 thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveGatherTests, TestGather)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    const size_t source_size = std::min((size_t) 10, 2 * size);
    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      // source vectors to gather from
      thrust::host_vector<T> h_source =
        get_random_data<T>(source_size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_source = h_source;

      // gather indices
      thrust::host_vector<unsigned int> h_map = get_random_data<unsigned int>(
        size,
        get_default_limits<unsigned int>::min(),
        get_default_limits<unsigned int>::max(),
        seed + seed_value_addition);

      for (size_t i = 0; i < size; i++)
      {
        h_map[i] = h_map[i] % source_size;
      }

      thrust::device_vector<unsigned int> d_map = h_map;

      // gather destination
      thrust::host_vector<T> h_output(size);
      thrust::device_vector<T> d_output(size);

      thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), h_output.begin());
      thrust::gather(d_map.begin(), d_map.end(), d_source.begin(), d_output.begin());

      ASSERT_EQ(h_output, d_output);
    }
  }
}

TYPED_TEST(PrimitiveGatherTests, TestGatherToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    const size_t source_size = std::min((size_t) 10, 2 * size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      // source vectors to gather from
      thrust::host_vector<T> h_source =
        get_random_data<T>(source_size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_source = h_source;

      // gather indices
      thrust::host_vector<unsigned int> h_map = get_random_data<unsigned int>(
        size,
        get_default_limits<unsigned int>::min(),
        get_default_limits<unsigned int>::max(),
        seed + seed_value_addition);

      for (size_t i = 0; i < size; i++)
      {
        h_map[i] = h_map[i] % source_size;
      }

      thrust::device_vector<unsigned int> d_map = h_map;

      thrust::discard_iterator<> h_result =
        thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), thrust::make_discard_iterator());

      thrust::discard_iterator<> d_result =
        thrust::gather(d_map.begin(), d_map.end(), d_source.begin(), thrust::make_discard_iterator());

      thrust::discard_iterator<> reference(size);

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);
    }
  }
}

TYPED_TEST(GatherTests, TestGatherIfSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector flg{0, 1, 0, 1, 0}; // predicate array
  Vector map{6, 2, 1, 7, 2}; // gather indices
  Vector src{0, 1, 2, 3, 4, 5, 6, 7}; // source vector
  Vector dst(5, 0); // destination vector

  thrust::gather_if(map.begin(), map.end(), flg.begin(), src.begin(), dst.begin());

  Vector ref{0, 2, 0, 7, 0};
  ASSERT_EQ(dst, ref);
}

template <typename T>
struct is_even_gather_if
{
  THRUST_HOST_DEVICE bool operator()(const T i) const
  {
    return (i % 2) == 0;
  }
};

template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator, typename OutputIterator>
OutputIterator gather_if(
  my_system& system,
  InputIterator1, //       map_first,
  InputIterator1, //       map_last,
  InputIterator2, //       stencil,
  RandomAccessIterator, // input_first,
  OutputIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST(GatherTests, TestGatherIfDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::gather_if(sys, vec.begin(), vec.end(), vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename RandomAccessIterator, typename OutputIterator>
OutputIterator gather_if(
  my_tag,
  InputIterator1, //       map_first,
  InputIterator1, //       map_last,
  InputIterator2, //       stencil,
  RandomAccessIterator, // input_first,
  OutputIterator result)
{
  *result = 13;
  return result;
}

TEST(GatherTests, TestGatherIfDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::gather_if(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.end()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveGatherTests, TestGatherIf)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    const size_t source_size = std::min((size_t) 10, 2 * size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      // source vectors to gather from
      thrust::host_vector<T> h_source =
        get_random_data<T>(source_size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_source = h_source;

      // gather indices
      thrust::host_vector<unsigned int> h_map = get_random_data<unsigned int>(
        size,
        get_default_limits<unsigned int>::min(),
        get_default_limits<unsigned int>::max(),
        seed + seed_value_addition);

      for (size_t i = 0; i < size; i++)
      {
        h_map[i] = h_map[i] % source_size;
      }

      thrust::device_vector<unsigned int> d_map = h_map;

      // gather stencil
      thrust::host_vector<unsigned int> h_stencil = get_random_data<unsigned int>(
        size,
        get_default_limits<unsigned int>::min(),
        get_default_limits<unsigned int>::max(),
        seed + 2 * seed_value_addition);

      for (size_t i = 0; i < size; i++)
      {
        h_stencil[i] = h_stencil[i] % 2;
      }

      thrust::device_vector<unsigned int> d_stencil = h_stencil;

      // gather destination
      thrust::host_vector<T> h_output(size);
      thrust::device_vector<T> d_output(size);

      thrust::gather_if(
        h_map.begin(),
        h_map.end(),
        h_stencil.begin(),
        h_source.begin(),
        h_output.begin(),
        is_even_gather_if<unsigned int>());
      thrust::gather_if(
        d_map.begin(),
        d_map.end(),
        d_stencil.begin(),
        d_source.begin(),
        d_output.begin(),
        is_even_gather_if<unsigned int>());

      ASSERT_EQ(h_output, d_output);
    }
  }
}

TYPED_TEST(PrimitiveGatherTests, TestGatherIfToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    const size_t source_size = std::min((size_t) 10, 2 * size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      // source vectors to gather from
      thrust::host_vector<T> h_source =
        get_random_data<T>(source_size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::device_vector<T> d_source = h_source;

      // gather indices
      thrust::host_vector<unsigned int> h_map = get_random_data<unsigned int>(
        size,
        get_default_limits<unsigned int>::min(),
        get_default_limits<unsigned int>::max(),
        seed + seed_value_addition);

      for (size_t i = 0; i < size; i++)
      {
        h_map[i] = h_map[i] % source_size;
      }

      thrust::device_vector<unsigned int> d_map = h_map;

      // gather stencil
      thrust::host_vector<unsigned int> h_stencil = get_random_data<unsigned int>(
        size,
        std::numeric_limits<unsigned int>::min(),
        std::numeric_limits<unsigned int>::max(),
        seed + 2 * seed_value_addition);

      for (size_t i = 0; i < size; i++)
      {
        h_stencil[i] = h_stencil[i] % 2;
      }

      thrust::device_vector<unsigned int> d_stencil = h_stencil;

      thrust::discard_iterator<> h_result = thrust::gather_if(
        h_map.begin(),
        h_map.end(),
        h_stencil.begin(),
        h_source.begin(),
        thrust::make_discard_iterator(),
        is_even_gather_if<unsigned int>());

      thrust::discard_iterator<> d_result = thrust::gather_if(
        d_map.begin(),
        d_map.end(),
        d_stencil.begin(),
        d_source.begin(),
        thrust::make_discard_iterator(),
        is_even_gather_if<unsigned int>());

      thrust::discard_iterator<> reference(size);

      ASSERT_EQ_QUIET(reference, h_result);
      ASSERT_EQ_QUIET(reference, d_result);
    }
  }
}

TYPED_TEST(GatherTests, TestGatherCountingIterator)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector source(10);
  thrust::sequence(source.begin(), source.end(), 0);

  Vector map(10);
  thrust::sequence(map.begin(), map.end(), 0);

  Vector output(10);

  // source has any_system_tag
  thrust::fill(output.begin(), output.end(), 0);
  thrust::gather(map.begin(), map.end(), thrust::make_counting_iterator(0), output.begin());

  ASSERT_EQ(output, map);

  // map has any_system_tag
  thrust::fill(output.begin(), output.end(), 0);
  thrust::gather(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator((int) source.size()),
                 source.begin(),
                 output.begin());

  ASSERT_EQ(output, map);

  // source and map have any_system_tag
  thrust::fill(output.begin(), output.end(), 0);
  thrust::gather(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator((int) output.size()),
                 thrust::make_counting_iterator(0),
                 output.begin());

  ASSERT_EQ(output, map);
}

THRUST_DIAG_POP
