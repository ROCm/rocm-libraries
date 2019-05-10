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

#include <thrust/pair.h>
#include <thrust/swap.h>
#include <thrust/tuple.h>

#include <utility>

#include "test_header.hpp"

TESTS_DEFINE(PairTests, NumericalTestsParams);

TYPED_TEST(PairTests, TestPairManipulation)
{
    using T = typename TestFixture::input_type;
    using P = thrust::pair<T, T>;

    // test null constructor
    P p1;
    ASSERT_EQ(T(0), p1.first);
    ASSERT_EQ(T(0), p1.second);

    // test individual value manipulation
    p1.first  = T(1);
    p1.second = T(2);
    ASSERT_EQ(T(1), p1.first);
    ASSERT_EQ(T(2), p1.second);

    // test copy constructor
    P p2(p1);
    ASSERT_EQ(p1.first, p2.first);
    ASSERT_EQ(p1.second, p2.second);

    // test copy from std::pair constructor
    std::pair<T, T> sp(p1.first, p1.second);
    ASSERT_EQ(p1.first, sp.first);
    ASSERT_EQ(p1.second, sp.second);

    // test initialization
    P p3 = p2;
    ASSERT_EQ(p2.first, p3.first);
    ASSERT_EQ(p2.second, p3.second);

    // test initialization from std::pair
    P p4 = sp;
    ASSERT_EQ(sp.first, p4.first);
    ASSERT_EQ(sp.second, p4.second);

    // test copy from pair
    p4.first  = T(2);
    p4.second = T(3);

    P p5;
    p5 = p4;
    ASSERT_EQ(p4.first, p5.first);
    ASSERT_EQ(p4.second, p5.second);

    // test copy from std::pair
    sp.first  = T(4);
    sp.second = T(5);

    P p6;
    p6 = sp;
    ASSERT_EQ(sp.first, p6.first);
    ASSERT_EQ(sp.second, p6.second);

    // test initialization from make_pair
    P p7 = thrust::make_pair(T(6), T(7));
    ASSERT_EQ(T(6), p7.first);
    ASSERT_EQ(T(7), p7.second);

    // test copy from make_pair
    p7 = thrust::make_pair(T(8), T(9));
    ASSERT_EQ(T(8), p7.first);
    ASSERT_EQ(T(9), p7.second);
}

TYPED_TEST(PairTests, TestPairComparison)
{
    using T = typename TestFixture::input_type;
    using P = thrust::pair<T, T>;

    P x, y;

    // test operator ==
    x.first = x.second = y.first = y.second = T(0);
    ASSERT_EQ(true, x == y);
    ASSERT_EQ(true, y == x);

    x.first = y.first = y.second = T(0);
    x.second                     = T(1);
    ASSERT_EQ(false, x == y);
    ASSERT_EQ(false, y == x);

    // test operator<
    x.first  = T(0);
    x.second = T(0);
    y.first  = T(0);
    y.second = T(0);
    ASSERT_EQ(false, x < y);
    ASSERT_EQ(false, y < x);

    x.first  = T(0);
    x.second = T(1);
    y.first  = T(2);
    y.second = T(3);
    ASSERT_EQ(true, x < y);
    ASSERT_EQ(false, y < x);

    x.first  = T(0);
    x.second = T(0);
    y.first  = T(0);
    y.second = T(1);
    ASSERT_EQ(true, x < y);
    ASSERT_EQ(false, y < x);

    x.first  = T(0);
    x.second = T(1);
    y.first  = T(0);
    y.second = T(2);
    ASSERT_EQ(true, x < y);
    ASSERT_EQ(false, y < x);

    // test operator!=
    x.first = y.first = y.second = T(0);
    x.second                     = T(1);
    ASSERT_EQ(true, x != y);
    ASSERT_EQ(true, y != x);

    x.first = x.second = y.first = y.second = T(0);
    ASSERT_EQ(false, x != y);
    ASSERT_EQ(false, y != x);

    // test operator>
    x.first  = T(0);
    x.second = T(0);
    y.first  = T(0);
    y.second = T(0);
    ASSERT_EQ(false, x > y);
    ASSERT_EQ(false, y > x);

    x.first  = T(2);
    x.second = T(3);
    y.first  = T(0);
    y.second = T(1);
    ASSERT_EQ(true, x > y);
    ASSERT_EQ(false, y > x);

    x.first  = T(0);
    x.second = T(1);
    y.first  = T(0);
    y.second = T(0);
    ASSERT_EQ(true, x > y);
    ASSERT_EQ(false, y > x);

    x.first  = T(0);
    x.second = T(2);
    y.first  = T(0);
    y.second = T(1);
    ASSERT_EQ(true, x > y);
    ASSERT_EQ(false, y > x);

    // test operator <=
    x.first = x.second = y.first = y.second = T(0);
    ASSERT_EQ(true, x <= y);
    ASSERT_EQ(true, y <= x);

    x.first = y.first = y.second = T(0);
    x.second                     = T(1);
    ASSERT_EQ(false, x <= y);

    x.first  = T(0);
    x.second = T(1);
    y.first  = T(2);
    y.second = T(3);
    ASSERT_EQ(true, x <= y);
    ASSERT_EQ(false, y <= x);

    x.first  = T(0);
    x.second = T(0);
    y.first  = T(0);
    y.second = T(1);
    ASSERT_EQ(true, x <= y);
    ASSERT_EQ(false, y <= x);

    x.first  = T(0);
    x.second = T(1);
    y.first  = T(0);
    y.second = T(2);
    ASSERT_EQ(true, x <= y);
    ASSERT_EQ(false, y <= x);

    // test operator >=
    x.first = x.second = y.first = y.second = T(0);
    ASSERT_EQ(true, x >= y);
    ASSERT_EQ(true, y >= x);

    x.first = x.second = y.first = T(0);
    y.second                     = T(1);
    ASSERT_EQ(false, x >= y);

    x.first  = T(2);
    x.second = T(3);
    y.first  = T(0);
    y.second = T(1);
    ASSERT_EQ(true, x >= y);
    ASSERT_EQ(false, y >= x);

    x.first  = T(0);
    x.second = T(1);
    y.first  = T(0);
    y.second = T(0);
    ASSERT_EQ(true, x >= y);
    ASSERT_EQ(false, y >= x);

    x.first  = T(0);
    x.second = T(2);
    y.first  = T(0);
    y.second = T(1);
    ASSERT_EQ(true, x >= y);
    ASSERT_EQ(false, y >= x);
}

TYPED_TEST(PairTests, TestPairGet)
{
    using T = typename TestFixture::input_type;
    thrust::host_vector<T> data
        = get_random_data<T>(2, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    ;

    thrust::pair<T, T> p(data[0], data[1]);

    ASSERT_EQ(data[0], thrust::get<0>(p));
    ASSERT_EQ(data[1], thrust::get<1>(p));
}

TEST(PairTests, TestPairTupleSize)
{
    int result = thrust::tuple_size<thrust::pair<int, int>>::value;
    ASSERT_EQ(2, result);
}

TEST(PairTests, TestPairTupleElement)
{
    using type0 = thrust::tuple_element<0, thrust::pair<int, float>>::type;
    using type1 = thrust::tuple_element<1, thrust::pair<int, float>>::type;

    ASSERT_EQ(typeid(int), typeid(type0));
    ASSERT_EQ(typeid(float), typeid(type1));
}

TEST(PairTests, TestPairSwap)
{
    int x = 7;
    int y = 13;

    int z = 42;
    int w = 0;

    thrust::pair<int, int> a(x, y);
    thrust::pair<int, int> b(z, w);

    thrust::swap(a, b);

    ASSERT_EQ(z, a.first);
    ASSERT_EQ(w, a.second);
    ASSERT_EQ(x, b.first);
    ASSERT_EQ(y, b.second);

    using swappable_pair = thrust::pair<user_swappable, user_swappable>;

    thrust::host_vector<swappable_pair>   h_v1(1), h_v2(1);
    thrust::device_vector<swappable_pair> d_v1(1), d_v2(1);

    thrust::swap_ranges(h_v1.begin(), h_v1.end(), h_v2.begin());
    thrust::swap_ranges(d_v1.begin(), d_v1.end(), d_v2.begin());

    swappable_pair ref(user_swappable(true), user_swappable(true));

    ASSERT_EQ(ref, h_v1[0]);
    ASSERT_EQ(ref, h_v1[0]);
    ASSERT_EQ(ref, (swappable_pair)d_v1[0]);
    ASSERT_EQ(ref, (swappable_pair)d_v1[0]);
}
