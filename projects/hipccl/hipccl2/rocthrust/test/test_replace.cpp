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

#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/replace.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

// There is a unfortunate miscompilation of the gcc-11 vectorizer leading to OOB writes
// Adding this attribute suffices that this miscompilation does not appear anymore
#if defined(THRUST_HOST_COMPILER_GCC) && __GNUC__ >= 11
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER __attribute__((optimize("no-tree-vectorize")))
#else
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER
#endif

// GCC 12 + omp + c++11 miscompiles some test cases and emits spurious warnings.
#if defined(THRUST_HOST_COMPILER_GCC) && __GNUC__ == 12 && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP \
  && THRUST_CPP_DIALECT == 2011
#  define THRUST_GCC12_OMP_MISCOMPILE
#endif

// New GCC, new miscompile. 13 + TBB this time.
#if defined(THRUST_HOST_COMPILER_GCC) && __GNUC__ == 13 && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
#  define THRUST_GCC13_TBB_MISCOMPILE
#endif

TESTS_DEFINE(ReplaceTests, FullTestsParams);

TESTS_DEFINE(PrimitiveReplaceTests, NumericalTestsParams);

TEST(ReplaceTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(ReplaceTests, TestReplaceSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 2, 1, 3, 2};

  thrust::replace(data.begin(), data.end(), (T) 1, (T) 4);
  thrust::replace(data.begin(), data.end(), (T) 2, (T) 5);

  Vector result{4, 5, 4, 3, 5};

  ASSERT_EQ(data, result);
}

template <typename ForwardIterator, typename T>
void replace(my_system& system, ForwardIterator, ForwardIterator, const T&, const T&)
{
  system.validate_dispatch();
}

TEST(ReplaceTests, TestReplaceDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace(sys, vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename T>
void replace(my_tag, ForwardIterator first, ForwardIterator, const T&, const T&)
{
  *first = 13;
}

TEST(ReplaceTests, TestReplaceDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::replace(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveReplaceTests, TestReplace)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    T old_value = 0;
    T new_value = 1;

    thrust::replace(h_data.begin(), h_data.end(), old_value, new_value);
    thrust::replace(d_data.begin(), d_data.end(), old_value, new_value);

    thrust::host_vector<T> h_data_d(d_data);
    for (size_t i = 0; i < size; ++i)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
    }
  }
}

#ifndef THRUST_GCC13_TBB_MISCOMPILE
#  ifndef THRUST_GCC12_OMP_MISCOMPILE
TYPED_TEST(ReplaceTests, TestReplaceCopySimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 2, 1, 3, 2};

  Vector dest(5);

  thrust::replace_copy(data.begin(), data.end(), dest.begin(), (T) 1, (T) 4);
  thrust::replace_copy(dest.begin(), dest.end(), dest.begin(), (T) 2, (T) 5);

  Vector result{4, 5, 4, 3, 5};
  ASSERT_EQ(dest, result);
}
#  endif
#endif

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(my_system& system, InputIterator, InputIterator, OutputIterator result, const T&, const T&)
{
  system.validate_dispatch();
  return result;
}

TEST(ReplaceTests, TestReplaceCopyDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_copy(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator replace_copy(my_tag, InputIterator, InputIterator, OutputIterator result, const T&, const T&)
{
  *result = 13;
  return result;
}

TEST(ReplaceTests, TestReplaceCopyDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::replace_copy(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveReplaceTests, TestReplaceCopy)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    T old_value = 0;
    T new_value = 1;

    thrust::host_vector<T> h_dest(size);
    thrust::device_vector<T> d_dest(size);

    thrust::replace_copy(h_data.begin(), h_data.end(), h_dest.begin(), old_value, new_value);
    thrust::replace_copy(d_data.begin(), d_data.end(), d_dest.begin(), old_value, new_value);

    thrust::host_vector<T> h_data_d(d_data);
    thrust::host_vector<T> h_dest_d(d_dest);
    for (size_t i = 0; i < size; ++i)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
      ASSERT_NEAR(h_dest[i], h_dest_d[i], T(0.1));
    }
  }
}

TYPED_TEST(PrimitiveReplaceTests, TestReplaceCopyToDiscardIterator)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    T old_value = 0;
    T new_value = 1;

    thrust::discard_iterator<> h_result =
      thrust::replace_copy(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), old_value, new_value);

    thrust::discard_iterator<> d_result =
      thrust::replace_copy(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), old_value, new_value);

    thrust::discard_iterator<> reference(size);

    ASSERT_EQ_QUIET(reference, h_result);
    ASSERT_EQ_QUIET(reference, d_result);
  }
}

template <typename T>
struct less_than_five
{
  THRUST_HOST_DEVICE bool operator()(const T& val) const
  {
    return val < 5;
  }
};

TYPED_TEST(ReplaceTests, TestReplaceIfSimple)
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 3, 4, 6, 5};

  thrust::replace_if(data.begin(), data.end(), less_than_five<T>(), (T) 0);

  Vector result{0, 0, 0, 6, 5};

  ASSERT_EQ(data, result);
}

template <typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_system& system, ForwardIterator, ForwardIterator, Predicate, const T&)
{
  system.validate_dispatch();
}

TEST(ReplaceTests, TestReplaceIfDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_if(sys, vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename Predicate, typename T>
void replace_if(my_tag, ForwardIterator first, ForwardIterator, Predicate, const T&)
{
  *first = 13;
}

TEST(ReplaceTests, TestReplaceIfDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::replace_if(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(ReplaceTests, TestReplaceIfStencilSimple) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 3, 4, 6, 5};

  Vector stencil{5, 4, 6, 3, 7};
  thrust::replace_if(data.begin(), data.end(), stencil.begin(), less_than_five<T>(), (T) 0);

  Vector result{1, 0, 4, 0, 5};

  ASSERT_EQ(data, result);
}

template <typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(my_system& system, ForwardIterator, ForwardIterator, InputIterator, Predicate, const T&)
{
  system.validate_dispatch();
}

TEST(ReplaceTests, TestReplaceIfStencilDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_if(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
void replace_if(my_tag, ForwardIterator first, ForwardIterator, InputIterator, Predicate, const T&)
{
  *first = 13;
}

TEST(ReplaceTests, TestReplaceIfStencilDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::replace_if(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveReplaceTests, TestReplaceIf) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::replace_if(h_data.begin(), h_data.end(), less_than_five<T>(), (T) 0);
    thrust::replace_if(d_data.begin(), d_data.end(), less_than_five<T>(), (T) 0);

    thrust::host_vector<T> h_data_d(d_data);
    for (size_t i = 0; i < size; ++i)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
    }
  }
}

TYPED_TEST(PrimitiveReplaceTests, TestReplaceIfStencil) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_stencil   = random_samples<T>(size);
    thrust::device_vector<T> d_stencil = h_stencil;

    thrust::replace_if(h_data.begin(), h_data.end(), h_stencil.begin(), less_than_five<T>(), (T) 0);
    thrust::replace_if(d_data.begin(), d_data.end(), d_stencil.begin(), less_than_five<T>(), (T) 0);

    thrust::host_vector<T> h_data_d(d_data);
    for (size_t i = 0; i < size; ++i)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
    }
  }
}

TYPED_TEST(ReplaceTests, TestReplaceCopyIfSimple) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 3, 4, 6, 5};

  Vector dest(5);

  thrust::replace_copy_if(data.begin(), data.end(), dest.begin(), less_than_five<T>(), (T) 0);

  Vector result{0, 0, 0, 6, 5};
  ASSERT_EQ(dest, result);
}

template <typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator
replace_copy_if(my_system& system, InputIterator, InputIterator, OutputIterator result, Predicate, const T&)
{
  system.validate_dispatch();
  return result;
}

TEST(ReplaceTests, TestReplaceCopyIfDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_copy_if(sys, vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(my_tag, InputIterator, InputIterator, OutputIterator result, Predicate, const T&)
{
  *result = 13;
  return result;
}

TEST(ReplaceTests, TestReplaceCopyIfDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::replace_copy_if(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0, 0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(ReplaceTests, TestReplaceCopyIfStencilSimple) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  using Vector = typename TestFixture::input_type;
  using T      = typename Vector::value_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector data{1, 3, 4, 6, 5};
  Vector stencil{1, 5, 4, 7, 8};

  Vector dest(5);

  thrust::replace_copy_if(data.begin(), data.end(), stencil.begin(), dest.begin(), less_than_five<T>(), (T) 0);

  Vector result{0, 3, 0, 6, 5};

  ASSERT_EQ(dest, result);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator replace_copy_if(
  my_system& system, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate, const T&)
{
  system.validate_dispatch();
  return result;
}

TEST(ReplaceTests, TestReplaceCopyIfStencilDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::replace_copy_if(sys, vec.begin(), vec.begin(), vec.begin(), vec.begin(), 0, 0);

  ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
OutputIterator
replace_copy_if(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputIterator result, Predicate, const T&)
{
  *result = 13;
  return result;
}

TEST(ReplaceTests, TestReplaceCopyIfStencilDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::replace_copy_if(
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    thrust::retag<my_tag>(vec.begin()),
    0,
    0);

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(PrimitiveReplaceTests, TestReplaceCopyIf) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_dest(size);
    thrust::device_vector<T> d_dest(size);

    thrust::replace_copy_if(h_data.begin(), h_data.end(), h_dest.begin(), less_than_five<T>(), T{0});
    thrust::replace_copy_if(d_data.begin(), d_data.end(), d_dest.begin(), less_than_five<T>(), T{0});

    thrust::host_vector<T> h_data_d(d_data);
    thrust::host_vector<T> h_dest_d(d_dest);
    for (size_t i = 0; i < size; ++i)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
      ASSERT_NEAR(h_dest[i], h_dest_d[i], T(0.1));
    }
  }
}

TYPED_TEST(PrimitiveReplaceTests, TestReplaceCopyIfToDiscardIterator) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::discard_iterator<> h_result =
      thrust::replace_copy_if(h_data.begin(), h_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

    thrust::discard_iterator<> d_result =
      thrust::replace_copy_if(d_data.begin(), d_data.end(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

    thrust::discard_iterator<> reference(size);

    ASSERT_EQ_QUIET(reference, h_result);
    ASSERT_EQ_QUIET(reference, d_result);
  }
}

TYPED_TEST(PrimitiveReplaceTests, TestReplaceCopyIfStencil) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_stencil   = random_samples<T>(size);
    thrust::device_vector<T> d_stencil = h_stencil;

    thrust::host_vector<T> h_dest(size);
    thrust::device_vector<T> d_dest(size);

    thrust::replace_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_dest.begin(), less_than_five<T>(), T{0});
    thrust::replace_copy_if(d_data.begin(), d_data.end(), d_stencil.begin(), d_dest.begin(), less_than_five<T>(), T{0});

    thrust::host_vector<T> h_data_d(d_data);
    thrust::host_vector<T> h_dest_d(d_dest);
    for (size_t i = 0; i < size; ++i)
    {
      ASSERT_NEAR(h_data[i], h_data_d[i], T(0.1));
      ASSERT_NEAR(h_dest[i], h_dest_d[i], T(0.1));
    }
  }
}

TYPED_TEST(PrimitiveReplaceTests, TestReplaceCopyIfStencilToDiscardIterator) THRUST_DISABLE_BROKEN_GCC_VECTORIZER
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    thrust::host_vector<T> h_data   = random_samples<T>(size);
    thrust::device_vector<T> d_data = h_data;

    thrust::host_vector<T> h_stencil   = random_samples<T>(size);
    thrust::device_vector<T> d_stencil = h_stencil;

    thrust::discard_iterator<> h_result = thrust::replace_copy_if(
      h_data.begin(), h_data.end(), h_stencil.begin(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

    thrust::discard_iterator<> d_result = thrust::replace_copy_if(
      d_data.begin(), d_data.end(), d_stencil.begin(), thrust::make_discard_iterator(), less_than_five<T>(), T{0});

    thrust::discard_iterator<> reference(size);

    ASSERT_EQ_QUIET(reference, h_result);
    ASSERT_EQ_QUIET(reference, d_result);
  }
}

__global__ THRUST_HIP_LAUNCH_BOUNDS_DEFAULT void ReplaceKernel(int const N, int* array, int old_value, int new_value)
{
  if (threadIdx.x == 0)
  {
    thrust::device_ptr<int> begin(array);
    thrust::device_ptr<int> end(array + N);
    thrust::replace(thrust::hip::par, begin, end, old_value, new_value);
  }
}

TEST(ReplaceTests, TestReplaceDevice)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<int> h_data   = get_random_data<int>(size, 0, size, seed);
      thrust::device_vector<int> d_data = h_data;

      int old_value = get_random_data<int>(1, 0, size, seed)[0];
      int new_value = get_random_data<int>(1, -size, size, seed)[0];

      SCOPED_TRACE(testing::Message() << "with old_value= " << old_value);

      thrust::replace(h_data.begin(), h_data.end(), old_value, new_value);
      hipLaunchKernelGGL(
        ReplaceKernel,
        dim3(1, 1, 1),
        dim3(128, 1, 1),
        0,
        0,
        size,
        thrust::raw_pointer_cast(&d_data[0]),
        old_value,
        new_value);

      ASSERT_EQ(h_data, d_data);
    }
  }
}
