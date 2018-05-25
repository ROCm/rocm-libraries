// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <type_traits>
#include <iterator>
#include <tuple>

// Google Test
#include <gtest/gtest.h>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

template<class T>
struct non_trivial
{
    non_trivial() { x = 1; }
    non_trivial(T x) : x(x) {}
    ~non_trivial() {}

    non_trivial& operator=(const non_trivial& other)
    {
        x = other.x;
        return *this;
    }

    non_trivial operator+(const non_trivial& other) const
    {
        return non_trivial(x + other.x);
    }

    non_trivial operator-(const non_trivial& other) const
    {
        return non_trivial(x - other.x);
    }

    bool operator==(const non_trivial& other) const
    {
        return (x == other.x);
    }

    bool operator!=(const non_trivial& other) const
    {
        return !(*this == other);
    }

    bool operator==(const T& value) const
    {
        return (x == value);
    }

    bool operator!=(const T& value) const
    {
        return !(*this == value);
    }

    T x;
};

using non_trivial_int = non_trivial<int>;

TEST(RocprimTupleTests, IntegerSequence)
{
    ASSERT_TRUE((
        std::is_same<
            rocprim::make_index_sequence<4>,
            rocprim::index_sequence<0, 1, 2, 3>
        >::value
    ));
    ASSERT_TRUE((
        !std::is_same<
            rocprim::make_index_sequence<4>,
            rocprim::index_sequence<0, 1, 2, 3, 4>
        >::value
    ));
    ASSERT_TRUE((
        !std::is_same<
            rocprim::make_index_sequence<3>,
            rocprim::integer_sequence<char, 0, 1, 2>
        >::value
    ));
    ASSERT_TRUE((
        std::is_same<
            rocprim::make_integer_sequence<char, 2>,
            rocprim::integer_sequence<char, 0, 1>
        >::value
    ));
}

TEST(RocprimTupleTests, TupleSize)
{
    using t3 = rocprim::tuple<char, int, double>;
    ASSERT_EQ(rocprim::tuple_size<t3>::value, 3);

    using t1 = rocprim::tuple<char>;
    ASSERT_EQ(rocprim::tuple_size<t1>::value, 1);

    using t4 = rocprim::tuple<char, int, double, int>;
    ASSERT_EQ(rocprim::tuple_size<t4>::value, 4);

    using t0 = rocprim::tuple<>;
    ASSERT_EQ(rocprim::tuple_size<t0>::value, 0);
}

TEST(RocprimTupleTests, TupleElement)
{
    using t0 = rocprim::tuple<char, int, double>;
    ASSERT_TRUE((
        std::is_same<
            rocprim::tuple_element<0, t0>::type,
            char
        >::value
    ));
    ASSERT_TRUE((
        std::is_same<
            rocprim::tuple_element<1, t0>::type,
            int
        >::value
    ));
    ASSERT_TRUE((
        std::is_same<
            rocprim::tuple_element_t<2, t0>,
            double
        >::value
    ));
}

TEST(RocprimTupleTests, TupleConstructors)
{
    rocprim::tuple<> t1; (void) t1;
    rocprim::tuple<int, char, double>t2(1, 2, 3); (void) t2;
    rocprim::tuple<int, int, int> t3(4, 5, 6); (void) t3;
    auto t4(t3); (void) t4;
    rocprim::tuple<int, char, double> t5(2, 4, 1U); (void) t5;
    rocprim::tuple<double, float> t6(5, 2); (void) t6;
    rocprim::tuple<double, float> t7; (void) t7;
    const int i = 5;
    rocprim::tuple<double, float> t8(i, 2); (void) t8;
    rocprim::tuple<double, float> t9(i, i); (void) t9;
    rocprim::tuple<int, int> t10(i, i); (void) t10;

    const rocprim::tuple<float, unsigned int> ct1(1.0f, 2U);
    rocprim::tuple<double, unsigned int> ct2(ct1);

    rocprim::tuple<non_trivial_int> nt_ct1; (void) nt_ct1;
    rocprim::tuple<non_trivial_int> nt_ct2(non_trivial_int(1)); (void) nt_ct2;
}

struct empty1
{
};

struct empty2
{
};

TEST(RocprimTupleTests, TupleConstructorsEBO)
{
    rocprim::tuple<empty1> t1; (void) t1;
    rocprim::tuple<empty1, int> t2(empty1(), 2.0); (void) t2;
    rocprim::tuple<empty1, empty2> t3; (void) t3;
    rocprim::tuple<empty1, int, empty2> t4(empty1(), 2, empty2()); (void) t4;
    auto t5(t4);  (void) t5;
}

TEST(RocprimTupleTests, TupleGetIndex)
{
    rocprim::tuple<int, char, double> t1(1, 2, 3U);
    rocprim::tuple<double, float, int> t2(4, 5, 6);

    // Read value
    ASSERT_EQ(rocprim::get<0>(t1), 1);
    ASSERT_EQ(rocprim::get<1>(t1), char(2));
    ASSERT_EQ(rocprim::get<2>(t1), 3);

    ASSERT_EQ(rocprim::get<0>(t2), 4);
    ASSERT_EQ(rocprim::get<1>(t2), 5);
    ASSERT_EQ(rocprim::get<2>(t2), 6);

    // Write value
    rocprim::get<0>(t2) = 7;
    rocprim::get<1>(t2) = 8;
    rocprim::get<2>(t2) = 9;
    ASSERT_EQ(rocprim::get<0>(t2), 7);
    ASSERT_EQ(rocprim::get<1>(t2), 8);
    ASSERT_EQ(rocprim::get<2>(t2), 9);

    // rvalue
    using type = rocprim::tuple<int, double>;
    ASSERT_EQ(rocprim::get<0>(type()), 0);
    ASSERT_EQ(rocprim::get<1>(type(1, 2)), 2);

    // non-trivial
    rocprim::tuple<non_trivial_int, non_trivial_int> nt1(
        non_trivial_int(1), non_trivial_int(2)
    );
    ASSERT_EQ(rocprim::get<0>(nt1), 1);
    ASSERT_EQ(rocprim::get<1>(nt1), 2);
}

TEST(RocprimTupleTests, TupleAssignOperator)
{
    rocprim::tuple<int, char, double> t1(2, 4, 2U);
    rocprim::tuple<int, char, double> t2(3, 5, 1U);
    rocprim::tuple<double, float, int> t3(1, 2, 3);

    ASSERT_EQ(rocprim::get<0>(t2), 3);
    ASSERT_EQ(rocprim::get<1>(t2), char(5));
    ASSERT_EQ(rocprim::get<2>(t2), 1);
    ASSERT_EQ(rocprim::get<0>(t3), 1);
    ASSERT_EQ(rocprim::get<1>(t3), 2);
    ASSERT_EQ(rocprim::get<2>(t3), 3);

    // Same tuple types
    auto t4 = t2;
    ASSERT_EQ(rocprim::get<0>(t4), 3);
    ASSERT_EQ(rocprim::get<1>(t4), char(5));
    ASSERT_EQ(rocprim::get<2>(t4), 1);

    t2 = t1;
    ASSERT_EQ(rocprim::get<0>(t2), 2);
    ASSERT_EQ(rocprim::get<1>(t2), char(4));
    ASSERT_EQ(rocprim::get<2>(t2), 2);

    // Different tuple types (same number of elements)
    t3 = t2;
    ASSERT_EQ(rocprim::get<0>(t3), 2);
    ASSERT_EQ(rocprim::get<1>(t3), 4);
    ASSERT_EQ(rocprim::get<2>(t3), 2);

    // Move (same tuple types)
    t3 = decltype(t3)();
    ASSERT_EQ(rocprim::get<0>(t3), 0);
    ASSERT_EQ(rocprim::get<1>(t3), 0);
    ASSERT_EQ(rocprim::get<2>(t3), 0);

    // Move (different types)
    t3 = decltype(t2)(1, 1, 1);
    ASSERT_EQ(rocprim::get<0>(t3), 1);
    ASSERT_EQ(rocprim::get<1>(t3), 1);
    ASSERT_EQ(rocprim::get<2>(t3), 1);

    // Empty tuple
    rocprim::tuple<> t6;
    rocprim::tuple<> t7;

    t6 = decltype(t7)();
    t7 = t6;

    rocprim::tuple<int, int> t10(1, 2);
    const rocprim::tuple<int, int>& t10_ref = t10;
    rocprim::tuple<int, int> t11(10, 20);
    t11 = t10_ref;
    ASSERT_EQ(rocprim::get<0>(t11), 1);
    ASSERT_EQ(rocprim::get<1>(t11), 2);

    rocprim::tuple<int, int> t12(11, 12);
    t12 = std::move(t10);
    ASSERT_EQ(rocprim::get<0>(t12), 1);
    ASSERT_EQ(rocprim::get<1>(t12), 2);

    rocprim::tuple<float, double> t13(11, 12);
    t13 = std::move(t10);
    ASSERT_EQ(rocprim::get<0>(t13), float(1));
    ASSERT_EQ(rocprim::get<1>(t13), double(2));

    // non-trivial
    rocprim::tuple<non_trivial_int, non_trivial_int> nt1(
        non_trivial_int(1), non_trivial_int(2)
    );
    rocprim::tuple<non_trivial_int, non_trivial_int> nt2;
    nt2 = nt1;
    ASSERT_EQ(rocprim::get<0>(nt2), 1);
    ASSERT_EQ(rocprim::get<1>(nt2), 2);
}

TEST(RocprimTupleTests, TupleComparisonOperators)
{
    rocprim::tuple<int, char, double> t1(2, 4, 2);
    rocprim::tuple<int, char, double> t2(3, 5, 1);
    rocprim::tuple<double, float, char> t3(3, 5, 1);
    rocprim::tuple<double, float, char> t4(2, 4, 2);

    // ==
    ASSERT_EQ(t1, t1);
    ASSERT_EQ(t2, t2);
    ASSERT_EQ(t3, t3);
    ASSERT_EQ(t2, t3);
    ASSERT_EQ(t1, t4);
    ASSERT_EQ(rocprim::tuple<>(), rocprim::tuple<>());

    // !=
    ASSERT_NE(t1, t2);
    ASSERT_NE(t3, t4);
    ASSERT_NE(t1, t3);
    ASSERT_NE(t2, t4);

    rocprim::tuple<float, float, float> t5(1, 2, 3);
    rocprim::tuple<float, float, float> t6(1, 2, 4);
    rocprim::tuple<int, int, int> t7(1, 4, 2);
    rocprim::tuple<int, int, int> t8(1, 4, 2);

    // <
    ASSERT_LT(t5, t6);
    ASSERT_LT(t5, t7);

    // <=
    ASSERT_LE(t5, t6);
    ASSERT_LE(t5, t7);
    ASSERT_LE(t5, t5);
    ASSERT_LE(t6, t6);
    ASSERT_LE(t7, t8);

    // >=
    ASSERT_GE(t5, t5);
    ASSERT_GE(t6, t6);
    ASSERT_GE(t7, t8);
    ASSERT_GE(t6, t5);
    ASSERT_GE(t7, t5);

    // >
    ASSERT_GT(t6, t5);
    ASSERT_GT(t7, t5);
}

TEST(RocprimTupleTests, TupleSwap)
{
    rocprim::tuple<int, char, double> t1(1, 2, 3);
    rocprim::tuple<int, char, double> t2(4, 5, 6);

    t1.swap(t2);
    ASSERT_EQ(rocprim::get<0>(t1), 4);
    ASSERT_EQ(rocprim::get<1>(t1), 5);
    ASSERT_EQ(rocprim::get<2>(t1), 6);
    ASSERT_EQ(rocprim::get<0>(t2), 1);
    ASSERT_EQ(rocprim::get<1>(t2), 2);
    ASSERT_EQ(rocprim::get<2>(t2), 3);

    rocprim::swap(t1, t2);
    ASSERT_EQ(rocprim::get<0>(t1), 1);
    ASSERT_EQ(rocprim::get<1>(t1), 2);
    ASSERT_EQ(rocprim::get<2>(t1), 3);
    ASSERT_EQ(rocprim::get<0>(t2), 4);
    ASSERT_EQ(rocprim::get<1>(t2), 5);
    ASSERT_EQ(rocprim::get<2>(t2), 6);
}

TEST(RocprimTupleTests, TupleMakeTuple)
{
    rocprim::tuple<int, double> t1(1, 2);
    auto t2 = rocprim::make_tuple(1, 2.0);
    ASSERT_EQ(t1, t2);
    ASSERT_TRUE((
        std::is_same<
            rocprim::tuple<int, double>,
            decltype(t2)
        >::value
    ));
    ASSERT_TRUE((
        std::is_same<
            rocprim::tuple<int, int, double>,
            decltype(rocprim::make_tuple(1, 6, 1.0))
        >::value
    ));
}

TEST(RocprimTupleTests, TupleMakeTie)
{
    int a = 0;
    double b = 0.0;

    rocprim::tuple<int, double> t1(1, 2);
    ASSERT_EQ(rocprim::get<0>(t1), 1);
    ASSERT_EQ(rocprim::get<1>(t1), 2);

    rocprim::tie(a, b) = t1;
    ASSERT_EQ(a, 1);
    ASSERT_EQ(b, 2);

    rocprim::tie(rocprim::ignore, a) = t1;
    ASSERT_EQ(a, 2);
    ASSERT_EQ(b, 2);
}

TEST(RocprimTupleTests, Conversions)
{
    ASSERT_EQ(
        (std::is_convertible<
            std::tuple<int&, int&>,
            std::tuple<int, int>
        >::value),
        (std::is_convertible<
            rocprim::tuple<int&, int&>,
            rocprim::tuple<int, int>
        >::value)
    );
    using T1R = std::iterator_traits<std::vector<int>::iterator>::reference;
    using T1V = std::iterator_traits<std::vector<int>::iterator>::value_type;
    ASSERT_EQ(
        (std::is_convertible<
            std::tuple<T1R, T1R>,
            std::tuple<T1V, T1V>
        >::value),
        (std::is_convertible<
            rocprim::tuple<T1R, T1R>,
            rocprim::tuple<T1V, T1V>
        >::value)
    );

    int x = 1;
    std::tuple<int&, int&> tx = std::tie(x, x);
    std::tuple<int, int> ux(tx); (void) ux;
    rocprim::tuple<int, int> rtx = rocprim::tie(x, x);
    rocprim::tuple<int, int> rux(rtx); (void) rux;
}