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

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>
#include <thrust/swap.h>
#include <thrust/system/cpp/memory.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

TESTS_DEFINE(SwapRangesTests, FullTestsParams);
TESTS_DEFINE(PrimitiveSwapRangesTests, NumericalTestsParams);

TEST(SwapRangesTests, UsingHip)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator2 swap_ranges(my_system& system, ForwardIterator1, ForwardIterator1, ForwardIterator2 first2)
{
  system.validate_dispatch();
  return first2;
}

TEST(SwapRangesTests, TestSwapRangesDispatchExplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::swap_ranges(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator2 swap_ranges(my_tag, ForwardIterator1, ForwardIterator1, ForwardIterator2 first2)
{
  *first2 = 13;
  return first2;
}

TEST(SwapRangesTests, TestSwapRangesDispatchImplicit)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> vec(1);

  thrust::swap_ranges(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQ(13, vec.front());
}

TYPED_TEST(SwapRangesTests, TestSwapRangesSimple)
{
  using Vector = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  Vector v1{0, 1, 2, 3, 4};
  Vector v2{5, 6, 7, 8, 9};

  thrust::swap_ranges(v1.begin(), v1.end(), v2.begin());

  Vector ref1{5, 6, 7, 8, 9};
  ASSERT_EQ(v1, ref1);

  Vector ref2{0, 1, 2, 3, 4};
  ASSERT_EQ(v2, ref2);
}

TYPED_TEST(PrimitiveSwapRangesTests, TestSwapRanges)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  for (auto size : get_sizes())
  {
    SCOPED_TRACE(testing::Message() << "with size= " << size);

    for (auto seed : get_seeds())
    {
      SCOPED_TRACE(testing::Message() << "with seed= " << seed);

      thrust::host_vector<T> a1 =
        get_random_data<T>(size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
      thrust::host_vector<T> a2 = get_random_data<T>(
        size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed + seed_value_addition);

      thrust::host_vector<T> h1   = a1;
      thrust::host_vector<T> h2   = a2;
      thrust::device_vector<T> d1 = a1;
      thrust::device_vector<T> d2 = a2;

      thrust::swap_ranges(h1.begin(), h1.end(), h2.begin());
      thrust::swap_ranges(d1.begin(), d1.end(), d2.begin());

      ASSERT_EQ(h1, a2);
      ASSERT_EQ(d1, a2);
      ASSERT_EQ(h2, a1);
      ASSERT_EQ(d2, a1);
    }
  }
}

#if (THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP)
TEST(SwapRangesTests, TestSwapRangesForcedIterator)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::device_vector<int> A(3, 0);
  thrust::device_vector<int> B(3, 1);

  thrust::swap_ranges(thrust::retag<thrust::cpp::tag>(A.begin()),
                      thrust::retag<thrust::cpp::tag>(A.end()),
                      thrust::retag<thrust::cpp::tag>(B.begin()));

  ASSERT_EQ(A[0], 1);
  ASSERT_EQ(A[1], 1);
  ASSERT_EQ(A[2], 1);
  ASSERT_EQ(B[0], 0);
  ASSERT_EQ(B[1], 0);
  ASSERT_EQ(B[2], 0);
}
#endif

struct type_with_swap
{
  inline THRUST_HOST_DEVICE type_with_swap()
      : m_x()
      , m_swapped(false)
  {}

  inline THRUST_HOST_DEVICE type_with_swap(int x)
      : m_x(x)
      , m_swapped(false)
  {}

  inline THRUST_HOST_DEVICE type_with_swap(int x, bool s)
      : m_x(x)
      , m_swapped(s)
  {}

  inline THRUST_HOST_DEVICE type_with_swap(const type_with_swap& other)
      : m_x(other.m_x)
      , m_swapped(other.m_swapped)
  {}

  inline THRUST_HOST_DEVICE bool operator==(const type_with_swap& other) const
  {
    return m_x == other.m_x && m_swapped == other.m_swapped;
  }

  type_with_swap& operator=(const type_with_swap&) = default;

  int m_x;
  bool m_swapped;
};

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
namespace detail
{
THRUST_EXEC_CHECK_DISABLE
template <typename Assignable1, typename Assignable2>
THRUST_HOST_DEVICE inline void swap(Assignable1& a, Assignable2& b)
{
  Assignable1 temp = a;
  a                = b;
  b                = temp;
} // end swap()
} // namespace detail
#endif

inline THRUST_HOST_DEVICE void swap(type_with_swap& a, type_with_swap& b)
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  using _THRUST_STD::swap;
#else
  using ::detail::swap;
#endif
  swap(a.m_x, b.m_x);
  a.m_swapped = true;
  b.m_swapped = true;
}

TEST(SwapRangesTests, TestSwapRangesUserSwap)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<type_with_swap> h_A(3, type_with_swap(0));
  thrust::host_vector<type_with_swap> h_B(3, type_with_swap(1));

  thrust::device_vector<type_with_swap> d_A = h_A;
  thrust::device_vector<type_with_swap> d_B = h_B;

  // check that nothing is yet swapped
  type_with_swap ref = type_with_swap(0, false);

  ASSERT_EQ_QUIET(ref, h_A[0]);
  ASSERT_EQ_QUIET(ref, h_A[1]);
  ASSERT_EQ_QUIET(ref, h_A[2]);

  ASSERT_EQ_QUIET(ref, d_A[0]);
  ASSERT_EQ_QUIET(ref, d_A[1]);
  ASSERT_EQ_QUIET(ref, d_A[2]);

  ref = type_with_swap(1, false);

  ASSERT_EQ_QUIET(ref, h_B[0]);
  ASSERT_EQ_QUIET(ref, h_B[1]);
  ASSERT_EQ_QUIET(ref, h_B[2]);

  ASSERT_EQ_QUIET(ref, d_B[0]);
  ASSERT_EQ_QUIET(ref, d_B[1]);
  ASSERT_EQ_QUIET(ref, d_B[2]);

  // swap the ranges

  thrust::swap_ranges(h_A.begin(), h_A.end(), h_B.begin());
  thrust::swap_ranges(d_A.begin(), d_A.end(), d_B.begin());

  // check that things were swapped
  ref = type_with_swap(1, true);

  ASSERT_EQ_QUIET(ref, h_A[0]);
  ASSERT_EQ_QUIET(ref, h_A[1]);
  ASSERT_EQ_QUIET(ref, h_A[2]);

  ASSERT_EQ_QUIET(ref, d_A[0]);
  ASSERT_EQ_QUIET(ref, d_A[1]);
  ASSERT_EQ_QUIET(ref, d_A[2]);

  ref = type_with_swap(0, true);

  ASSERT_EQ_QUIET(ref, h_B[0]);
  ASSERT_EQ_QUIET(ref, h_B[1]);
  ASSERT_EQ_QUIET(ref, h_B[2]);

  ASSERT_EQ_QUIET(ref, d_B[0]);
  ASSERT_EQ_QUIET(ref, d_B[1]);
  ASSERT_EQ_QUIET(ref, d_B[2]);
}
