/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/detail/static_map.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/swap.h>
#include <thrust/tuple.h>

#include "test_header.hpp"

TESTS_DEFINE(TupleTests, NumericalTestsParams);

TYPED_TEST(TupleTests, TestTupleConstructor)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data
        = get_random_data<T>(10, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    thrust::tuple<T> t1(data[0]);
    ASSERT_EQ(data[0], thrust::get<0>(t1));

    thrust::tuple<T, T> t2(data[0], data[1]);
    ASSERT_EQ(data[0], thrust::get<0>(t2));
    ASSERT_EQ(data[1], thrust::get<1>(t2));

    thrust::tuple<T, T, T> t3(data[0], data[1], data[2]);
    ASSERT_EQ(data[0], thrust::get<0>(t3));
    ASSERT_EQ(data[1], thrust::get<1>(t3));
    ASSERT_EQ(data[2], thrust::get<2>(t3));

    thrust::tuple<T, T, T, T> t4(data[0], data[1], data[2], data[3]);
    ASSERT_EQ(data[0], thrust::get<0>(t4));
    ASSERT_EQ(data[1], thrust::get<1>(t4));
    ASSERT_EQ(data[2], thrust::get<2>(t4));
    ASSERT_EQ(data[3], thrust::get<3>(t4));

    thrust::tuple<T, T, T, T, T> t5(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQ(data[0], thrust::get<0>(t5));
    ASSERT_EQ(data[1], thrust::get<1>(t5));
    ASSERT_EQ(data[2], thrust::get<2>(t5));
    ASSERT_EQ(data[3], thrust::get<3>(t5));
    ASSERT_EQ(data[4], thrust::get<4>(t5));

    thrust::tuple<T, T, T, T, T, T> t6(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQ(data[0], thrust::get<0>(t6));
    ASSERT_EQ(data[1], thrust::get<1>(t6));
    ASSERT_EQ(data[2], thrust::get<2>(t6));
    ASSERT_EQ(data[3], thrust::get<3>(t6));
    ASSERT_EQ(data[4], thrust::get<4>(t6));
    ASSERT_EQ(data[5], thrust::get<5>(t6));

    thrust::tuple<T, T, T, T, T, T, T> t7(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQ(data[0], thrust::get<0>(t7));
    ASSERT_EQ(data[1], thrust::get<1>(t7));
    ASSERT_EQ(data[2], thrust::get<2>(t7));
    ASSERT_EQ(data[3], thrust::get<3>(t7));
    ASSERT_EQ(data[4], thrust::get<4>(t7));
    ASSERT_EQ(data[5], thrust::get<5>(t7));
    ASSERT_EQ(data[6], thrust::get<6>(t7));

    thrust::tuple<T, T, T, T, T, T, T, T> t8(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQ(data[0], thrust::get<0>(t8));
    ASSERT_EQ(data[1], thrust::get<1>(t8));
    ASSERT_EQ(data[2], thrust::get<2>(t8));
    ASSERT_EQ(data[3], thrust::get<3>(t8));
    ASSERT_EQ(data[4], thrust::get<4>(t8));
    ASSERT_EQ(data[5], thrust::get<5>(t8));
    ASSERT_EQ(data[6], thrust::get<6>(t8));
    ASSERT_EQ(data[7], thrust::get<7>(t8));

    thrust::tuple<T, T, T, T, T, T, T, T, T> t9(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQ(data[0], thrust::get<0>(t9));
    ASSERT_EQ(data[1], thrust::get<1>(t9));
    ASSERT_EQ(data[2], thrust::get<2>(t9));
    ASSERT_EQ(data[3], thrust::get<3>(t9));
    ASSERT_EQ(data[4], thrust::get<4>(t9));
    ASSERT_EQ(data[5], thrust::get<5>(t9));
    ASSERT_EQ(data[6], thrust::get<6>(t9));
    ASSERT_EQ(data[7], thrust::get<7>(t9));
    ASSERT_EQ(data[8], thrust::get<8>(t9));

    // TODO: tuple cannot handle 10 element
    /*thrust::tuple<T,T,T,T,T,T,T,T,T,T> t10(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
  ASSERT_EQ(data[0], thrust::get<0>(t10));
  ASSERT_EQ(data[1], thrust::get<1>(t10));
  ASSERT_EQ(data[2], thrust::get<2>(t10));
  ASSERT_EQ(data[3], thrust::get<3>(t10));
  ASSERT_EQ(data[4], thrust::get<4>(t10));
  ASSERT_EQ(data[5], thrust::get<5>(t10));
  ASSERT_EQ(data[6], thrust::get<6>(t10));
  ASSERT_EQ(data[7], thrust::get<7>(t10));
  ASSERT_EQ(data[8], thrust::get<8>(t10));
  ASSERT_EQ(data[9], thrust::get<9>(t10));*/
}

TYPED_TEST(TupleTests, TestMakeTuple)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data
        = get_random_data<T>(10, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    thrust::tuple<T> t1 = thrust::make_tuple(data[0]);
    ASSERT_EQ(data[0], thrust::get<0>(t1));

    thrust::tuple<T, T> t2 = thrust::make_tuple(data[0], data[1]);
    ASSERT_EQ(data[0], thrust::get<0>(t2));
    ASSERT_EQ(data[1], thrust::get<1>(t2));

    thrust::tuple<T, T, T> t3 = thrust::make_tuple(data[0], data[1], data[2]);
    ASSERT_EQ(data[0], thrust::get<0>(t3));
    ASSERT_EQ(data[1], thrust::get<1>(t3));
    ASSERT_EQ(data[2], thrust::get<2>(t3));

    thrust::tuple<T, T, T, T> t4 = thrust::make_tuple(data[0], data[1], data[2], data[3]);
    ASSERT_EQ(data[0], thrust::get<0>(t4));
    ASSERT_EQ(data[1], thrust::get<1>(t4));
    ASSERT_EQ(data[2], thrust::get<2>(t4));
    ASSERT_EQ(data[3], thrust::get<3>(t4));

    thrust::tuple<T, T, T, T, T> t5
        = thrust::make_tuple(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQ(data[0], thrust::get<0>(t5));
    ASSERT_EQ(data[1], thrust::get<1>(t5));
    ASSERT_EQ(data[2], thrust::get<2>(t5));
    ASSERT_EQ(data[3], thrust::get<3>(t5));
    ASSERT_EQ(data[4], thrust::get<4>(t5));

    thrust::tuple<T, T, T, T, T, T> t6
        = thrust::make_tuple(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQ(data[0], thrust::get<0>(t6));
    ASSERT_EQ(data[1], thrust::get<1>(t6));
    ASSERT_EQ(data[2], thrust::get<2>(t6));
    ASSERT_EQ(data[3], thrust::get<3>(t6));
    ASSERT_EQ(data[4], thrust::get<4>(t6));
    ASSERT_EQ(data[5], thrust::get<5>(t6));

    thrust::tuple<T, T, T, T, T, T, T> t7
        = thrust::make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQ(data[0], thrust::get<0>(t7));
    ASSERT_EQ(data[1], thrust::get<1>(t7));
    ASSERT_EQ(data[2], thrust::get<2>(t7));
    ASSERT_EQ(data[3], thrust::get<3>(t7));
    ASSERT_EQ(data[4], thrust::get<4>(t7));
    ASSERT_EQ(data[5], thrust::get<5>(t7));
    ASSERT_EQ(data[6], thrust::get<6>(t7));

    thrust::tuple<T, T, T, T, T, T, T, T> t8 = thrust::make_tuple(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQ(data[0], thrust::get<0>(t8));
    ASSERT_EQ(data[1], thrust::get<1>(t8));
    ASSERT_EQ(data[2], thrust::get<2>(t8));
    ASSERT_EQ(data[3], thrust::get<3>(t8));
    ASSERT_EQ(data[4], thrust::get<4>(t8));
    ASSERT_EQ(data[5], thrust::get<5>(t8));
    ASSERT_EQ(data[6], thrust::get<6>(t8));
    ASSERT_EQ(data[7], thrust::get<7>(t8));

    thrust::tuple<T, T, T, T, T, T, T, T, T> t9 = thrust::make_tuple(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQ(data[0], thrust::get<0>(t9));
    ASSERT_EQ(data[1], thrust::get<1>(t9));
    ASSERT_EQ(data[2], thrust::get<2>(t9));
    ASSERT_EQ(data[3], thrust::get<3>(t9));
    ASSERT_EQ(data[4], thrust::get<4>(t9));
    ASSERT_EQ(data[5], thrust::get<5>(t9));
    ASSERT_EQ(data[6], thrust::get<6>(t9));
    ASSERT_EQ(data[7], thrust::get<7>(t9));
    ASSERT_EQ(data[8], thrust::get<8>(t9));

    // TODO: tuple cannot handle 10 element
    /*thrust::tuple<T,T,T,T,T,T,T,T,T,T> t10 = thrust::make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
  ASSERT_EQ(data[0], thrust::get<0>(t10));
  ASSERT_EQ(data[1], thrust::get<1>(t10));
  ASSERT_EQ(data[2], thrust::get<2>(t10));
  ASSERT_EQ(data[3], thrust::get<3>(t10));
  ASSERT_EQ(data[4], thrust::get<4>(t10));
  ASSERT_EQ(data[5], thrust::get<5>(t10));
  ASSERT_EQ(data[6], thrust::get<6>(t10));
  ASSERT_EQ(data[7], thrust::get<7>(t10));
  ASSERT_EQ(data[8], thrust::get<8>(t10));
  ASSERT_EQ(data[9], thrust::get<9>(t10));*/
}

TYPED_TEST(TupleTests, TestTupleGet)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<T> data
        = get_random_data<T>(10, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

    thrust::tuple<T> t1(data[0]);
    ASSERT_EQ(data[0], thrust::get<0>(t1));

    thrust::tuple<T, T> t2(data[0], data[1]);
    ASSERT_EQ(data[0], thrust::get<0>(t2));
    ASSERT_EQ(data[1], thrust::get<1>(t2));

    thrust::tuple<T, T, T> t3 = thrust::make_tuple(data[0], data[1], data[2]);
    ASSERT_EQ(data[0], thrust::get<0>(t3));
    ASSERT_EQ(data[1], thrust::get<1>(t3));
    ASSERT_EQ(data[2], thrust::get<2>(t3));

    thrust::tuple<T, T, T, T> t4 = thrust::make_tuple(data[0], data[1], data[2], data[3]);
    ASSERT_EQ(data[0], thrust::get<0>(t4));
    ASSERT_EQ(data[1], thrust::get<1>(t4));
    ASSERT_EQ(data[2], thrust::get<2>(t4));
    ASSERT_EQ(data[3], thrust::get<3>(t4));

    thrust::tuple<T, T, T, T, T> t5
        = thrust::make_tuple(data[0], data[1], data[2], data[3], data[4]);
    ASSERT_EQ(data[0], thrust::get<0>(t5));
    ASSERT_EQ(data[1], thrust::get<1>(t5));
    ASSERT_EQ(data[2], thrust::get<2>(t5));
    ASSERT_EQ(data[3], thrust::get<3>(t5));
    ASSERT_EQ(data[4], thrust::get<4>(t5));

    thrust::tuple<T, T, T, T, T, T> t6
        = thrust::make_tuple(data[0], data[1], data[2], data[3], data[4], data[5]);
    ASSERT_EQ(data[0], thrust::get<0>(t6));
    ASSERT_EQ(data[1], thrust::get<1>(t6));
    ASSERT_EQ(data[2], thrust::get<2>(t6));
    ASSERT_EQ(data[3], thrust::get<3>(t6));
    ASSERT_EQ(data[4], thrust::get<4>(t6));
    ASSERT_EQ(data[5], thrust::get<5>(t6));

    thrust::tuple<T, T, T, T, T, T, T> t7
        = thrust::make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
    ASSERT_EQ(data[0], thrust::get<0>(t7));
    ASSERT_EQ(data[1], thrust::get<1>(t7));
    ASSERT_EQ(data[2], thrust::get<2>(t7));
    ASSERT_EQ(data[3], thrust::get<3>(t7));
    ASSERT_EQ(data[4], thrust::get<4>(t7));
    ASSERT_EQ(data[5], thrust::get<5>(t7));
    ASSERT_EQ(data[6], thrust::get<6>(t7));

    thrust::tuple<T, T, T, T, T, T, T, T> t8 = thrust::make_tuple(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
    ASSERT_EQ(data[0], thrust::get<0>(t8));
    ASSERT_EQ(data[1], thrust::get<1>(t8));
    ASSERT_EQ(data[2], thrust::get<2>(t8));
    ASSERT_EQ(data[3], thrust::get<3>(t8));
    ASSERT_EQ(data[4], thrust::get<4>(t8));
    ASSERT_EQ(data[5], thrust::get<5>(t8));
    ASSERT_EQ(data[6], thrust::get<6>(t8));
    ASSERT_EQ(data[7], thrust::get<7>(t8));

    thrust::tuple<T, T, T, T, T, T, T, T, T> t9 = thrust::make_tuple(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
    ASSERT_EQ(data[0], thrust::get<0>(t9));
    ASSERT_EQ(data[1], thrust::get<1>(t9));
    ASSERT_EQ(data[2], thrust::get<2>(t9));
    ASSERT_EQ(data[3], thrust::get<3>(t9));
    ASSERT_EQ(data[4], thrust::get<4>(t9));
    ASSERT_EQ(data[5], thrust::get<5>(t9));
    ASSERT_EQ(data[6], thrust::get<6>(t9));
    ASSERT_EQ(data[7], thrust::get<7>(t9));
    ASSERT_EQ(data[8], thrust::get<8>(t9));

    // TODO: tuple cannot handle 10 element
    /*thrust::tuple<T,T,T,T,T,T,T,T,T,T> t10 = thrust::make_tuple(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
  ASSERT_EQ(data[0], thrust::get<0>(t10));
  ASSERT_EQ(data[1], thrust::get<1>(t10));
  ASSERT_EQ(data[2], thrust::get<2>(t10));
  ASSERT_EQ(data[3], thrust::get<3>(t10));
  ASSERT_EQ(data[4], thrust::get<4>(t10));
  ASSERT_EQ(data[5], thrust::get<5>(t10));
  ASSERT_EQ(data[6], thrust::get<6>(t10));
  ASSERT_EQ(data[7], thrust::get<7>(t10));
  ASSERT_EQ(data[8], thrust::get<8>(t10));
  ASSERT_EQ(data[9], thrust::get<9>(t10));*/
}

TYPED_TEST(TupleTests, TestTupleComparison)
{
    using T = typename TestFixture::input_type;

    thrust::tuple<T, T, T, T, T> lhs(0, 0, 0, 0, 0), rhs(0, 0, 0, 0, 0);

    // equality
    ASSERT_EQ(true, lhs == rhs);
    thrust::get<0>(rhs) = 1;
    ASSERT_EQ(false, lhs == rhs);

    // inequality
    ASSERT_EQ(true, lhs != rhs);
    lhs = rhs;
    ASSERT_EQ(false, lhs != rhs);

    // less than
    lhs = thrust::make_tuple(0, 0, 0, 0, 0);
    rhs = thrust::make_tuple(0, 0, 1, 0, 0);
    ASSERT_EQ(true, lhs < rhs);
    thrust::get<0>(lhs) = 2;
    ASSERT_EQ(false, lhs < rhs);

    // less than equal
    lhs = thrust::make_tuple(0, 0, 0, 0, 0);
    rhs = lhs;
    ASSERT_EQ(true, lhs <= rhs); // equal
    thrust::get<2>(rhs) = 1;
    ASSERT_EQ(true, lhs <= rhs); // less than
    thrust::get<2>(lhs) = 2;
    ASSERT_EQ(false, lhs <= rhs);

    // greater than
    lhs = thrust::make_tuple(1, 0, 0, 0, 0);
    rhs = thrust::make_tuple(0, 1, 1, 1, 1);
    ASSERT_EQ(true, lhs > rhs);
    thrust::get<0>(rhs) = 2;
    ASSERT_EQ(false, lhs > rhs);

    // greater than equal
    lhs = thrust::make_tuple(0, 0, 0, 0, 0);
    rhs = lhs;
    ASSERT_EQ(true, lhs >= rhs); // equal
    thrust::get<4>(lhs) = 1;
    ASSERT_EQ(true, lhs >= rhs); // greater than
    thrust::get<3>(rhs) = 1;
    ASSERT_EQ(false, lhs >= rhs);
}

template <typename T>
struct TestTupleTieFunctor
{
    __host__ __device__ void clear(T* data) const
    {
        for(int i = 0; i < 10; ++i)
            data[i] = 13;
    }

    __host__ __device__ bool operator()() const
    {
        using namespace thrust;

        bool result = true;

        T data[9];
        clear(data);

        tie(data[0]) = thrust::make_tuple(0);
        ;
        result &= data[0] == 0;
        clear(data);

        tie(data[0], data[1]) = thrust::make_tuple(0, 1);
        result &= data[0] == 0;
        result &= data[1] == 1;
        clear(data);

        tie(data[0], data[1], data[2]) = thrust::make_tuple(0, 1, 2);
        result &= data[0] == 0;
        result &= data[1] == 1;
        result &= data[2] == 2;
        clear(data);

        tie(data[0], data[1], data[2], data[3]) = thrust::make_tuple(0, 1, 2, 3);
        result &= data[0] == 0;
        result &= data[1] == 1;
        result &= data[2] == 2;
        result &= data[3] == 3;
        clear(data);

        tie(data[0], data[1], data[2], data[3], data[4]) = thrust::make_tuple(0, 1, 2, 3, 4);
        result &= data[0] == 0;
        result &= data[1] == 1;
        result &= data[2] == 2;
        result &= data[3] == 3;
        result &= data[4] == 4;
        clear(data);

        tie(data[0], data[1], data[2], data[3], data[4], data[5])
            = thrust::make_tuple(0, 1, 2, 3, 4, 5);
        result &= data[0] == 0;
        result &= data[1] == 1;
        result &= data[2] == 2;
        result &= data[3] == 3;
        result &= data[4] == 4;
        result &= data[5] == 5;
        clear(data);

        tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6])
            = thrust::make_tuple(0, 1, 2, 3, 4, 5, 6);
        result &= data[0] == 0;
        result &= data[1] == 1;
        result &= data[2] == 2;
        result &= data[3] == 3;
        result &= data[4] == 4;
        result &= data[5] == 5;
        result &= data[6] == 6;
        clear(data);

        tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
            = thrust::make_tuple(0, 1, 2, 3, 4, 5, 6, 7);
        result &= data[0] == 0;
        result &= data[1] == 1;
        result &= data[2] == 2;
        result &= data[3] == 3;
        result &= data[4] == 4;
        result &= data[5] == 5;
        result &= data[6] == 6;
        result &= data[7] == 7;
        clear(data);

        tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8])
            = thrust::make_tuple(0, 1, 2, 3, 4, 5, 6, 7, 8);
        result &= data[0] == 0;
        result &= data[1] == 1;
        result &= data[2] == 2;
        result &= data[3] == 3;
        result &= data[4] == 4;
        result &= data[5] == 5;
        result &= data[6] == 6;
        result &= data[7] == 7;
        result &= data[8] == 8;
        clear(data);

        // TODO: tuple cannot handle 10 element
        /*tie(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]) = thrust::make_tuple(0,1,2,3,4,5,6,7,8,9);
    result &= data[0] == 0;
    result &= data[1] == 1;
    result &= data[2] == 2;
    result &= data[3] == 3;
    result &= data[4] == 4;
    result &= data[5] == 5;
    result &= data[6] == 6;
    result &= data[7] == 7;
    result &= data[8] == 8;
    result &= data[9] == 9;
    clear(data);*/

        return result;
    }
};

TYPED_TEST(TupleTests, TestTupleTie)
{
    using T = typename TestFixture::input_type;

    thrust::host_vector<bool> h_result(1);
    thrust::generate(h_result.begin(), h_result.end(), TestTupleTieFunctor<T>());

    thrust::device_vector<bool> d_result(1);
    thrust::generate(d_result.begin(), d_result.end(), TestTupleTieFunctor<T>());

    ASSERT_EQ(true, h_result[0]);
    ASSERT_EQ(true, d_result[0]);
}

TEST(TupleTests, TestTupleSwap)
{
    int a = 7;
    int b = 13;
    int c = 42;

    int x = 77;
    int y = 1313;
    int z = 4242;

    thrust::tuple<int, int, int> t1(a, b, c);
    thrust::tuple<int, int, int> t2(x, y, z);

    thrust::swap(t1, t2);

    ASSERT_EQ(x, thrust::get<0>(t1));
    ASSERT_EQ(y, thrust::get<1>(t1));
    ASSERT_EQ(z, thrust::get<2>(t1));
    ASSERT_EQ(a, thrust::get<0>(t2));
    ASSERT_EQ(b, thrust::get<1>(t2));
    ASSERT_EQ(c, thrust::get<2>(t2));

    typedef thrust::tuple<user_swappable, user_swappable, user_swappable, user_swappable>
        swappable_tuple;

    thrust::host_vector<swappable_tuple>   h_v1(1), h_v2(1);
    thrust::device_vector<swappable_tuple> d_v1(1), d_v2(1);

    thrust::swap_ranges(h_v1.begin(), h_v1.end(), h_v2.begin());
    thrust::swap_ranges(d_v1.begin(), d_v1.end(), d_v2.begin());

    swappable_tuple ref(
        user_swappable(true), user_swappable(true), user_swappable(true), user_swappable(true));

    ASSERT_EQ(ref, h_v1[0]);
    ASSERT_EQ(ref, h_v1[0]);
    ASSERT_EQ(ref, (swappable_tuple)d_v1[0]);
    ASSERT_EQ(ref, (swappable_tuple)d_v1[0]);
}
