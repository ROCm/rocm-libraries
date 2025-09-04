/*
 *  Copyright 2008-2024 NVIDIA Corporation
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

#include <thrust/generate.h>
#include <thrust/swap.h>
#include <thrust/tuple.h>

#include "test_param_fixtures.hpp"
#include "test_real_assertions.hpp"
#include "test_utils.hpp"

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <utility>
#endif

TESTS_DEFINE(TupleTests, NumericalTestsParams);

// Yes we're using 'thrust::null_type', I don't care >:(
#if !_THRUST_HAS_DEVICE_SYSTEM_STD
THRUST_SUPPRESS_DEPRECATED_PUSH
#endif

TYPED_TEST(TupleTests, TestTupleConstructor)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    host_vector<T> data = get_random_data<T>(10, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

    tuple<T> t1(data[0]);
    ASSERT_EQ(data[0], get<0>(t1));

    tuple<T, T> t2(data[0], data[1]);
    ASSERT_EQ(data[0], get<0>(t2));
    ASSERT_EQ(data[1], get<1>(t2));

    tuple<T, T, T> t3(data[0], data[1], data[2]);
    ASSERT_EQ(data[0], get<0>(t3));
    ASSERT_EQ(data[1], get<1>(t3));
    ASSERT_EQ(data[2], get<2>(t3));

    tuple<T, T, T, T> t4(data[0], data[1], data[2], data[3]);
    ASSERT_EQ(data[0], get<0>(t4));
    ASSERT_EQ(data[1], get<1>(t4));
    ASSERT_EQ(data[2], get<2>(t4));
    ASSERT_EQ(data[3], get<3>(t4));

    tuple<T, T, T, T, T> t5(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQ(data[0], get<0>(t5));
    ASSERT_EQ(data[1], get<1>(t5));
    ASSERT_EQ(data[2], get<2>(t5));
    ASSERT_EQ(data[3], get<3>(t5));
    ASSERT_EQ(data[4], get<4>(t5));

    tuple<T, T, T, T, T, T> t6(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQ(data[0], get<0>(t6));
    ASSERT_EQ(data[1], get<1>(t6));
    ASSERT_EQ(data[2], get<2>(t6));
    ASSERT_EQ(data[3], get<3>(t6));
    ASSERT_EQ(data[4], get<4>(t6));
    ASSERT_EQ(data[5], get<5>(t6));

    tuple<T, T, T, T, T, T, T> t7(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQ(data[0], get<0>(t7));
    ASSERT_EQ(data[1], get<1>(t7));
    ASSERT_EQ(data[2], get<2>(t7));
    ASSERT_EQ(data[3], get<3>(t7));
    ASSERT_EQ(data[4], get<4>(t7));
    ASSERT_EQ(data[5], get<5>(t7));
    ASSERT_EQ(data[6], get<6>(t7));

    tuple<T, T, T, T, T, T, T, T> t8(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQ(data[0], get<0>(t8));
    ASSERT_EQ(data[1], get<1>(t8));
    ASSERT_EQ(data[2], get<2>(t8));
    ASSERT_EQ(data[3], get<3>(t8));
    ASSERT_EQ(data[4], get<4>(t8));
    ASSERT_EQ(data[5], get<5>(t8));
    ASSERT_EQ(data[6], get<6>(t8));
    ASSERT_EQ(data[7], get<7>(t8));

    tuple<T, T, T, T, T, T, T, T, T> t9(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQ(data[0], get<0>(t9));
    ASSERT_EQ(data[1], get<1>(t9));
    ASSERT_EQ(data[2], get<2>(t9));
    ASSERT_EQ(data[3], get<3>(t9));
    ASSERT_EQ(data[4], get<4>(t9));
    ASSERT_EQ(data[5], get<5>(t9));
    ASSERT_EQ(data[6], get<6>(t9));
    ASSERT_EQ(data[7], get<7>(t9));
    ASSERT_EQ(data[8], get<8>(t9));

    tuple<T, T, T, T, T, T, T, T, T, T> t10(
      data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    ASSERT_EQ(data[0], get<0>(t10));
    ASSERT_EQ(data[1], get<1>(t10));
    ASSERT_EQ(data[2], get<2>(t10));
    ASSERT_EQ(data[3], get<3>(t10));
    ASSERT_EQ(data[4], get<4>(t10));
    ASSERT_EQ(data[5], get<5>(t10));
    ASSERT_EQ(data[6], get<6>(t10));
    ASSERT_EQ(data[7], get<7>(t10));
    ASSERT_EQ(data[8], get<8>(t10));
    ASSERT_EQ(data[9], get<9>(t10));
  }
}

TYPED_TEST(TupleTests, TestMakeTuple)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    host_vector<T> data = get_random_data<T>(10, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

    tuple<T> t1 = make_tuple(data[0]);
    ASSERT_EQ(data[0], get<0>(t1));

    tuple<T, T> t2 = make_tuple(data[0], data[1]);
    ASSERT_EQ(data[0], get<0>(t2));
    ASSERT_EQ(data[1], get<1>(t2));

    tuple<T, T, T> t3 = make_tuple(data[0], data[1], data[2]);
    ASSERT_EQ(data[0], get<0>(t3));
    ASSERT_EQ(data[1], get<1>(t3));
    ASSERT_EQ(data[2], get<2>(t3));

    tuple<T, T, T, T> t4 = make_tuple(data[0], data[1], data[2], data[3]);
    ASSERT_EQ(data[0], get<0>(t4));
    ASSERT_EQ(data[1], get<1>(t4));
    ASSERT_EQ(data[2], get<2>(t4));
    ASSERT_EQ(data[3], get<3>(t4));

    tuple<T, T, T, T, T> t5 = make_tuple(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQ(data[0], get<0>(t5));
    ASSERT_EQ(data[1], get<1>(t5));
    ASSERT_EQ(data[2], get<2>(t5));
    ASSERT_EQ(data[3], get<3>(t5));
    ASSERT_EQ(data[4], get<4>(t5));

    tuple<T, T, T, T, T, T> t6 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQ(data[0], get<0>(t6));
    ASSERT_EQ(data[1], get<1>(t6));
    ASSERT_EQ(data[2], get<2>(t6));
    ASSERT_EQ(data[3], get<3>(t6));
    ASSERT_EQ(data[4], get<4>(t6));
    ASSERT_EQ(data[5], get<5>(t6));

    tuple<T, T, T, T, T, T, T> t7 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQ(data[0], get<0>(t7));
    ASSERT_EQ(data[1], get<1>(t7));
    ASSERT_EQ(data[2], get<2>(t7));
    ASSERT_EQ(data[3], get<3>(t7));
    ASSERT_EQ(data[4], get<4>(t7));
    ASSERT_EQ(data[5], get<5>(t7));
    ASSERT_EQ(data[6], get<6>(t7));

    tuple<T, T, T, T, T, T, T, T> t8 =
      make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQ(data[0], get<0>(t8));
    ASSERT_EQ(data[1], get<1>(t8));
    ASSERT_EQ(data[2], get<2>(t8));
    ASSERT_EQ(data[3], get<3>(t8));
    ASSERT_EQ(data[4], get<4>(t8));
    ASSERT_EQ(data[5], get<5>(t8));
    ASSERT_EQ(data[6], get<6>(t8));
    ASSERT_EQ(data[7], get<7>(t8));

    tuple<T, T, T, T, T, T, T, T, T> t9 =
      make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQ(data[0], get<0>(t9));
    ASSERT_EQ(data[1], get<1>(t9));
    ASSERT_EQ(data[2], get<2>(t9));
    ASSERT_EQ(data[3], get<3>(t9));
    ASSERT_EQ(data[4], get<4>(t9));
    ASSERT_EQ(data[5], get<5>(t9));
    ASSERT_EQ(data[6], get<6>(t9));
    ASSERT_EQ(data[7], get<7>(t9));
    ASSERT_EQ(data[8], get<8>(t9));

    tuple<T, T, T, T, T, T, T, T, T, T> t10 =
      make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    ASSERT_EQ(data[0], get<0>(t10));
    ASSERT_EQ(data[1], get<1>(t10));
    ASSERT_EQ(data[2], get<2>(t10));
    ASSERT_EQ(data[3], get<3>(t10));
    ASSERT_EQ(data[4], get<4>(t10));
    ASSERT_EQ(data[5], get<5>(t10));
    ASSERT_EQ(data[6], get<6>(t10));
    ASSERT_EQ(data[7], get<7>(t10));
    ASSERT_EQ(data[8], get<8>(t10));
    ASSERT_EQ(data[9], get<9>(t10));
  }
}

TYPED_TEST(TupleTests, TestTupleGet)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  for (auto seed : get_seeds())
  {
    SCOPED_TRACE(testing::Message() << "with seed= " << seed);

    host_vector<T> data = get_random_data<T>(10, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);

    tuple<T> t1(data[0]);
    ASSERT_EQ(data[0], thrust::get<0>(t1));

    tuple<T, T> t2(data[0], data[1]);
    ASSERT_EQ(data[0], thrust::get<0>(t2));
    ASSERT_EQ(data[1], thrust::get<1>(t2));

    tuple<T, T, T> t3 = make_tuple(data[0], data[1], data[2]);
    ASSERT_EQ(data[0], thrust::get<0>(t3));
    ASSERT_EQ(data[1], thrust::get<1>(t3));
    ASSERT_EQ(data[2], thrust::get<2>(t3));

    tuple<T, T, T, T> t4 = make_tuple(data[0], data[1], data[2], data[3]);
    ASSERT_EQ(data[0], thrust::get<0>(t4));
    ASSERT_EQ(data[1], thrust::get<1>(t4));
    ASSERT_EQ(data[2], thrust::get<2>(t4));
    ASSERT_EQ(data[3], thrust::get<3>(t4));

    tuple<T, T, T, T, T> t5 = make_tuple(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQ(data[0], thrust::get<0>(t5));
    ASSERT_EQ(data[1], thrust::get<1>(t5));
    ASSERT_EQ(data[2], thrust::get<2>(t5));
    ASSERT_EQ(data[3], thrust::get<3>(t5));
    ASSERT_EQ(data[4], thrust::get<4>(t5));

    tuple<T, T, T, T, T, T> t6 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQ(data[0], thrust::get<0>(t6));
    ASSERT_EQ(data[1], thrust::get<1>(t6));
    ASSERT_EQ(data[2], thrust::get<2>(t6));
    ASSERT_EQ(data[3], thrust::get<3>(t6));
    ASSERT_EQ(data[4], thrust::get<4>(t6));
    ASSERT_EQ(data[5], thrust::get<5>(t6));

    tuple<T, T, T, T, T, T, T> t7 = make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQ(data[0], thrust::get<0>(t7));
    ASSERT_EQ(data[1], thrust::get<1>(t7));
    ASSERT_EQ(data[2], thrust::get<2>(t7));
    ASSERT_EQ(data[3], thrust::get<3>(t7));
    ASSERT_EQ(data[4], thrust::get<4>(t7));
    ASSERT_EQ(data[5], thrust::get<5>(t7));
    ASSERT_EQ(data[6], thrust::get<6>(t7));

    tuple<T, T, T, T, T, T, T, T> t8 =
      make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQ(data[0], thrust::get<0>(t8));
    ASSERT_EQ(data[1], thrust::get<1>(t8));
    ASSERT_EQ(data[2], thrust::get<2>(t8));
    ASSERT_EQ(data[3], thrust::get<3>(t8));
    ASSERT_EQ(data[4], thrust::get<4>(t8));
    ASSERT_EQ(data[5], thrust::get<5>(t8));
    ASSERT_EQ(data[6], thrust::get<6>(t8));
    ASSERT_EQ(data[7], thrust::get<7>(t8));

    tuple<T, T, T, T, T, T, T, T, T> t9 =
      make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQ(data[0], thrust::get<0>(t9));
    ASSERT_EQ(data[1], thrust::get<1>(t9));
    ASSERT_EQ(data[2], thrust::get<2>(t9));
    ASSERT_EQ(data[3], thrust::get<3>(t9));
    ASSERT_EQ(data[4], thrust::get<4>(t9));
    ASSERT_EQ(data[5], thrust::get<5>(t9));
    ASSERT_EQ(data[6], thrust::get<6>(t9));
    ASSERT_EQ(data[7], thrust::get<7>(t9));
    ASSERT_EQ(data[8], thrust::get<8>(t9));

    tuple<T, T, T, T, T, T, T, T, T, T> t10 =
      make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    ASSERT_EQ(data[0], thrust::get<0>(t10));
    ASSERT_EQ(data[1], thrust::get<1>(t10));
    ASSERT_EQ(data[2], thrust::get<2>(t10));
    ASSERT_EQ(data[3], thrust::get<3>(t10));
    ASSERT_EQ(data[4], thrust::get<4>(t10));
    ASSERT_EQ(data[5], thrust::get<5>(t10));
    ASSERT_EQ(data[6], thrust::get<6>(t10));
    ASSERT_EQ(data[7], thrust::get<7>(t10));
    ASSERT_EQ(data[8], thrust::get<8>(t10));
    ASSERT_EQ(data[9], thrust::get<9>(t10));
  }
}

TYPED_TEST(TupleTests, TestTupleComparison)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  using namespace thrust;

  tuple<T, T, T, T, T> lhs(0, 0, 0, 0, 0), rhs(0, 0, 0, 0, 0);

  // equality
  ASSERT_EQ(true, lhs == rhs);
  get<0>(rhs) = 1;
  ASSERT_EQ(false, lhs == rhs);

  // inequality
  ASSERT_EQ(true, lhs != rhs);
  lhs = rhs;
  ASSERT_EQ(false, lhs != rhs);

  // less than
  lhs = make_tuple(0, 0, 0, 0, 0);
  rhs = make_tuple(0, 0, 1, 0, 0);
  ASSERT_EQ(true, lhs < rhs);
  get<0>(lhs) = 2;
  ASSERT_EQ(false, lhs < rhs);

  // less than equal
  lhs = make_tuple(0, 0, 0, 0, 0);
  rhs = lhs;
  ASSERT_EQ(true, lhs <= rhs); // equal
  get<2>(rhs) = 1;
  ASSERT_EQ(true, lhs <= rhs); // less than
  get<2>(lhs) = 2;
  ASSERT_EQ(false, lhs <= rhs);

  // greater than
  lhs = make_tuple(1, 0, 0, 0, 0);
  rhs = make_tuple(0, 1, 1, 1, 1);
  ASSERT_EQ(true, lhs > rhs);
  get<0>(rhs) = 2;
  ASSERT_EQ(false, lhs > rhs);

  // greater than equal
  lhs = make_tuple(0, 0, 0, 0, 0);
  rhs = lhs;
  ASSERT_EQ(true, lhs >= rhs); // equal
  get<4>(lhs) = 1;
  ASSERT_EQ(true, lhs >= rhs); // greater than
  get<3>(rhs) = 1;
  ASSERT_EQ(false, lhs >= rhs);
}

template <typename T>
struct TestTupleTieFunctor
{
  THRUST_HOST_DEVICE void clear(T* data) const
  {
    for (int i = 0; i < 10; ++i)
    {
      data[i] = 13;
    }
  }

  THRUST_HOST_DEVICE bool operator()() const
  {
    using namespace thrust;

    bool result = true;

    T data[10];
    clear(data);

    // 17 and not 0 to avoid triggering custom_numeric's `operator void *` and a comparison with a null pointer
    // TODO: get this back from 17 to 0 once C++11 is on everywhere and that operator on custom_numeric is changed
    // to an explicit operator bool
    tie(data[0]) = make_tuple(17);
    result &= data[0] == 17;
    clear(data);

    tie(data[0], data[1]) = make_tuple(17, 1);
    result &= data[0] == 17;
    result &= data[1] == 1;
    clear(data);

    tie(data[0], data[1], data[2]) = make_tuple(17, 1, 2);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    clear(data);

    tie(data[0], data[1], data[2], data[3]) = make_tuple(17, 1, 2, 3);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4]) = make_tuple(17, 1, 2, 3, 4);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5]) = make_tuple(17, 1, 2, 3, 4, 5);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6]) = make_tuple(17, 1, 2, 3, 4, 5, 6);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]) = make_tuple(17, 1, 2, 3, 4, 5, 6, 7);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]) =
      make_tuple(17, 1, 2, 3, 4, 5, 6, 7, 8);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    result &= data[8] == 8;
    clear(data);

    tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]) =
      make_tuple(17, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    result &= data[0] == 17;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    result &= data[8] == 8;
    result &= data[9] == 9;
    clear(data);

    return result;
  }
};

TYPED_TEST(TupleTests, TestTupleTie)
{
  using T = typename TestFixture::input_type;

  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  thrust::host_vector<bool> h_result(1);
  thrust::generate(h_result.begin(), h_result.end(), TestTupleTieFunctor<T>());

  thrust::device_vector<bool> d_result(1);
  thrust::generate(d_result.begin(), d_result.end(), TestTupleTieFunctor<T>());

  ASSERT_EQ(true, h_result[0]);
  ASSERT_EQ(true, d_result[0]);
}

TEST(TupleTests, TestTupleSwap)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  int a = 7;
  int b = 13;
  int c = 42;

  int x = 77;
  int y = 1313;
  int z = 4242;

  thrust::tuple<int, int, int> t1(a, b, c);
  thrust::tuple<int, int, int> t2(x, y, z);

  using _THRUST_STD::swap;
  swap(t1, t2);

  ASSERT_EQ(x, thrust::get<0>(t1));
  ASSERT_EQ(y, thrust::get<1>(t1));
  ASSERT_EQ(z, thrust::get<2>(t1));
  ASSERT_EQ(a, thrust::get<0>(t2));
  ASSERT_EQ(b, thrust::get<1>(t2));
  ASSERT_EQ(c, thrust::get<2>(t2));

  using swappable_tuple = thrust::tuple<user_swappable, user_swappable, user_swappable, user_swappable>;

  thrust::host_vector<swappable_tuple> h_v1(1), h_v2(1);
  thrust::device_vector<swappable_tuple> d_v1(1), d_v2(1);

  thrust::swap_ranges(h_v1.begin(), h_v1.end(), h_v2.begin());
  thrust::swap_ranges(d_v1.begin(), d_v1.end(), d_v2.begin());

  swappable_tuple ref(user_swappable(true), user_swappable(true), user_swappable(true), user_swappable(true));

  ASSERT_EQ_QUIET(ref, h_v1[0]);
  ASSERT_EQ_QUIET(ref, h_v1[0]);
  ASSERT_EQ_QUIET(ref, (swappable_tuple) d_v1[0]);
  ASSERT_EQ_QUIET(ref, (swappable_tuple) d_v1[0]);
}

#if THRUST_CPP_DIALECT >= 2017
TEST(TupleTests, TestTupleStructuredBindings)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const int a = 0;
  const int b = 42;
  const int c = 1337;
  thrust::tuple<int, int, int> t(a, b, c);

  auto [a2, b2, c2] = t;
  ASSERT_EQ(a, a2);
  ASSERT_EQ(b, b2);
  ASSERT_EQ(c, c2);
}

TEST(TupleTests, TestTupleCTAD)
{
  SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

  const int a   = 0;
  const char b  = 42;
  const short c = 1337;
  thrust::tuple t(a, b, c);

  auto [a2, b2, c2] = t;
  ASSERT_EQ(a, a2);
  ASSERT_EQ(b, b2);
  ASSERT_EQ(c, c2);
}
#endif // THRUST_CPP_DIALECT >= 2017

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
THRUST_SUPPRESS_DEPRECATED_POP
#endif

// Ensure that we are backwards compatible with the old thrust::tuple implementation
THRUST_SUPPRESS_DEPRECATED_PUSH
static_assert(
  thrust::tuple_size<thrust::tuple<thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type>>::value
    == 0,
  "");
static_assert(
  thrust::tuple_size<thrust::tuple<int,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type>>::value
    == 1,
  "");
static_assert(
  thrust::tuple_size<thrust::tuple<int,
                                   int,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type>>::value
    == 2,
  "");
static_assert(
  thrust::tuple_size<thrust::tuple<int,
                                   int,
                                   int,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type>>::value
    == 3,
  "");
static_assert(
  thrust::tuple_size<thrust::tuple<int,
                                   int,
                                   int,
                                   int,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type>>::value
    == 4,
  "");
static_assert(
  thrust::tuple_size<thrust::tuple<int,
                                   int,
                                   int,
                                   int,
                                   int,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type,
                                   thrust::null_type>>::value
    == 5,
  "");
static_assert(
  thrust::tuple_size<
    thrust::
      tuple<int, int, int, int, int, int, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>>::value
    == 6,
  "");
static_assert(
  thrust::tuple_size<
    thrust::tuple<int, int, int, int, int, int, int, thrust::null_type, thrust::null_type, thrust::null_type>>::value
    == 7,
  "");
static_assert(
  thrust::tuple_size<thrust::tuple<int, int, int, int, int, int, int, int, thrust::null_type, thrust::null_type>>::value
    == 8,
  "");
static_assert(
  thrust::tuple_size<thrust::tuple<int, int, int, int, int, int, int, int, int, thrust::null_type>>::value == 9, "");
static_assert(thrust::tuple_size<thrust::tuple<int, int, int, int, int, int, int, int, int, int>>::value == 10, "");
THRUST_SUPPRESS_DEPRECATED_POP
