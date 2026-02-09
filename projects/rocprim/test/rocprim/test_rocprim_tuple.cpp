// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../common_test_header.hpp"

#include "test_utils.hpp"

//tuple_size : check size for various tuples and their cv‑qualified variants
TEST(TupleSize, BasicAndCV)
{
    using Empty  = rocprim::tuple<>;
    using Single = rocprim::tuple<int>;
    using Mixed  = rocprim::tuple<int, double, char>;

    // Compile time assertions
    static_assert(rocprim::tuple_size<Empty>::value == 0, "");
    static_assert(rocprim::tuple_size<Single>::value == 1, "");
    static_assert(rocprim::tuple_size<Mixed>::value == 3, "");

    // const/ volatile specific, cv‑qualified tuples must report the same size
    static_assert(rocprim::tuple_size<const Mixed>::value == 3, "");
    static_assert(rocprim::tuple_size<volatile Mixed>::value == 3, "");
    static_assert(rocprim::tuple_size<const volatile Mixed>::value == 3, "");

    // Runtime assertions
    EXPECT_EQ(rocprim::tuple_size<Mixed>::value, 3u);
}

// get<I> : read / write / cv‑correctness
TEST(Get, ValueAndReference)
{
    rocprim::tuple<int, double> t{1, 2.0};

    // Runtime value access
    EXPECT_EQ(rocprim::get<0>(t), 1);
    EXPECT_DOUBLE_EQ(rocprim::get<1>(t), 2.0);

    // Lvalue reference must be writable
    rocprim::get<0>(t) = 10;
    EXPECT_EQ(rocprim::get<0>(t), 10);

    // Reading through const tuple returns const&
    const auto& ct = t;
    static_assert(std::is_same_v<decltype(rocprim::get<0>(ct)), const int&>);
}

// tuple_element : index -> type mapping (+ alias tuple_element_t)
TEST(TupleElement, Types)
{
    using T0 = rocprim::tuple<int>;
    static_assert(std::is_same_v<rocprim::tuple_element<0, T0>::type, int>, "");

    using T3 = rocprim::tuple<int, double, char>;
    static_assert(std::is_same_v<rocprim::tuple_element<0, T3>::type, int>, "");
    static_assert(std::is_same_v<rocprim::tuple_element<1, T3>::type, double>, "");
    static_assert(std::is_same_v<rocprim::tuple_element<2, T3>::type, char>, "");

    // alias helper
    static_assert(std::is_same_v<rocprim::tuple_element_t<1, T3>, double>, "");

    // cv‑qualified tuple_element
    using T = rocprim::tuple<int, double>;

    static_assert(std::is_same_v<rocprim::tuple_element<0, const T>::type, const int>, "");
    static_assert(std::is_same_v<rocprim::tuple_element<1, volatile T>::type, volatile double>, "");
    static_assert(
        std::is_same_v<rocprim::tuple_element<0, const volatile T>::type, const volatile int>,
        "");

    // Verify rocprim::get<I>() pairs with tuple_element_t
    rocprim::tuple<int, char> t{42, 'x'};
    static_assert(std::is_same_v<decltype(rocprim::get<0>(t)), int&>, "");
    EXPECT_EQ(rocprim::get<0>(t), 42);
    EXPECT_EQ(rocprim::get<1>(t), 'x');
}

// custom_forward : perfect‑forwarding category check
TEST(CustomForward, ValueCategoryAndType)
{
    int  x    = 1;
    int& lref = x;

    // Should return int& when parameter is lvalue
    static_assert(std::is_same_v<decltype(rocprim::detail::custom_forward<int&>(lref)), int&>);

    // Should return int&& when parameter is rvalue
    static_assert(
        std::is_same_v<decltype(rocprim::detail::custom_forward<int&&>(std::move(x))), int&&>);
}

//is_final trait : detects final classes
namespace
{
struct NonFinal
{};
struct Final final
{};
} // namespace

TEST(IsFinalTrait, DetectsCorrectly)
{
    static_assert(!rocprim::detail::is_final<NonFinal>::value, "NonFinal should be false");
    static_assert(rocprim::detail::is_final<Final>::value, "Final should be true");
}

// tuple_value : one element storage + EBO, tested with Typed-Test
namespace
{
struct Empty // Empty type to trigger EBO
{};
struct EmptyFinal final // Final empty type
{};
struct NonEmpty // Normal type
{
    int x;
};
using Builtin = int; // Built-in type

using Params = ::testing::Types<Empty, EmptyFinal, NonEmpty, Builtin>;
} // namespace

template<class T>
class TupleValueTyped : public ::testing::Test
{};
TYPED_TEST_SUITE(TupleValueTyped, Params);

// Compile‑time checks (size, get(), constructibility)
TYPED_TEST(TupleValueTyped, CompileTimeTraits)
{
    using T  = TypeParam;
    using TV = rocprim::detail::tuple_value<0, T>;

    if constexpr(std::is_empty_v<T> && !rocprim::detail::is_final<T>::value)
        static_assert(sizeof(TV) == 1, "EBO failed");
    else
        static_assert(sizeof(TV) >= sizeof(T), "size too small?");

    TV tv;
    static_assert(std::is_same_v<decltype(tv.get()), T&>);
    static_assert(std::is_same_v<decltype(std::as_const(tv).get()), const T&>);

    static_assert(std::is_constructible_v<TV, T&&>);
    static_assert(std::is_assignable_v<TV&, T&&>);
}

// Runtime swap() behaviour
TEST(TupleValueRuntime, SwapAndGet)
{
    using TV = rocprim::detail::tuple_value<0, int>;

    TV a{1}, b{2};
    a.swap(b);

    EXPECT_EQ(a.get(), 2);
    EXPECT_EQ(b.get(), 1);
}

template<class T>
class TupleImplTyped : public ::testing::Test
{};
TYPED_TEST_SUITE(TupleImplTyped, Params);

// tuple_impl : internal aggregate of N tuple_value bases
TYPED_TEST(TupleImplTyped, CompileTimeTraits)
{
    using T = TypeParam;

    using Impl = rocprim::detail::tuple_impl<rocprim::make_index_sequence<3>, T, char, double>;

    // size check : removing an empty non‑final element should not change size
    if constexpr(std::is_empty_v<T> && !rocprim::detail::is_final<T>::value)
    {
        using ImplNoEmpty
            = rocprim::detail::tuple_impl<rocprim::make_index_sequence<2>, char, double>;

        static_assert(sizeof(Impl) == sizeof(ImplNoEmpty), "EBO failed: empty element adds size");
    }

    static_assert(std::is_constructible_v<Impl, T, char, double>);
    static_assert(std::is_assignable_v<Impl&, Impl const&>);
}

// tuple<Types...> : public API
//  a) Compile‑time: Typed‑Tests for constructors & assignment
//  b) Runtime behaviour: Unit TEST for Ctor/Assign/Swap/Get
template<class T>
class TupleTyped : public ::testing::Test
{};
TYPED_TEST_SUITE(TupleTyped, Params);

// a) Compile‑time
TYPED_TEST(TupleTyped, CompileTimeConstructAssign)
{
    using T    = TypeParam;
    using Tup1 = rocprim::tuple<T>;
    using Tup3 = rocprim::tuple<T, char, double>;

    // Default‑constructible when each element is default‑constructible
    static_assert(std::is_default_constructible_v<Tup1>);
    static_assert(std::is_default_constructible_v<Tup3>);

    // Direct value constructor
    static_assert(std::is_constructible_v<Tup3, T, char, double>);

    // Converting copy / move ctor
    static_assert(std::is_constructible_v<Tup3, const Tup3&>);
    static_assert(std::is_constructible_v<Tup3, Tup3&&>);

    // Assignment operators
    static_assert(std::is_assignable_v<Tup3&, const Tup3&>);
    static_assert(std::is_assignable_v<Tup3&, Tup3&&>);
}

// b) Runtime behaviour
TEST(TupleRuntime, CtorAssignSwapGet)
{
    rocprim::tuple<int, double, char> t1{1, 2.5, 'a'};
    rocprim::tuple<int, double, char> t2{10, 20.0, 'z'};

    // Verify construction & access
    EXPECT_EQ(rocprim::get<0>(t1), 1);
    EXPECT_DOUBLE_EQ(rocprim::get<1>(t1), 2.5);
    EXPECT_EQ(rocprim::get<2>(t1), 'a');

    // Copy assignment
    t2 = t1;
    EXPECT_EQ(rocprim::get<0>(t2), 1);
    EXPECT_EQ(rocprim::get<2>(t2), 'a');

    // Move assignment
    t1 = rocprim::tuple<int, double, char>{5, 7.7, 'b'};
    EXPECT_EQ(rocprim::get<0>(t1), 5);

    // swap
    t1.swap(t2);
    EXPECT_EQ(rocprim::get<0>(t1), 1);
    EXPECT_EQ(rocprim::get<0>(t2), 5);
}

// Empty tuple<> specialisation — compile‑time + runtime
TEST(EmptyTuple, CompileTimeAndRuntime)
{
    using E = rocprim::tuple<>;

    // Compile‑time
    static_assert(std::is_default_constructible_v<E>);
    static_assert(std::is_trivially_destructible_v<E>);

    // Runtime: swap is a no‑op but still callable
    E e1, e2;
    e1.swap(e2);
    SUCCEED();
}

// tuple comparisons: ==, !=, <, >, <=, >=, free swap

// operator== and operator!= on empty tuples
TEST(TupleComparison, EmptyTuples)
{
    rocprim::tuple<> a, b;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
}

// operator== and operator!= on non-empty tuples
TEST(TupleComparison, EqualAndNotEqual)
{
    rocprim::tuple<int, char> t1{1, 'a'};
    rocprim::tuple<int, char> t2{1, 'a'};
    rocprim::tuple<int, char> t3{2, 'b'};

    EXPECT_TRUE(t1 == t2);
    EXPECT_FALSE(t1 != t2);

    EXPECT_FALSE(t1 == t3);
    EXPECT_TRUE(t1 != t3);
}

// Lexicographical operator<, >, <=, >=
TEST(TupleComparison, LexicographicalOrder)
{
    rocprim::tuple<int, int> a{1, 2};
    rocprim::tuple<int, int> b{1, 3};
    rocprim::tuple<int, int> c{2, 0};

    // a < b because 2 < 3 at index 1
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(a > b);
    EXPECT_TRUE(a <= b);
    EXPECT_FALSE(a >= b);

    // b < c because 1 < 2 at index 0
    EXPECT_TRUE(b < c);
    EXPECT_FALSE(b > c);

    // transitive: a < c
    EXPECT_TRUE(a < c);
}

// Free swap calls member swap
TEST(TupleSwapFree, FreeSwap)
{
    rocprim::tuple<int, char> t1{5, 'x'};
    rocprim::tuple<int, char> t2{6, 'y'};

    swap(t1, t2);
    EXPECT_EQ(rocprim::get<0>(t1), 6);
    EXPECT_EQ(rocprim::get<1>(t1), 'y');
    EXPECT_EQ(rocprim::get<0>(t2), 5);
    EXPECT_EQ(rocprim::get<1>(t2), 'x');
}

// get<I> rvalue overload
TEST(GetRvalue, RvalueReferenceReturn)
{
    rocprim::tuple<int, double, char> t{1, 2.0, 'c'};
    // rvalue get should return T&& and preserve value
    using R0 = decltype(rocprim::get<0>(std::move(t)));
    static_assert(std::is_same_v<R0, int&&>);
    // example use: move out char
    char c = rocprim::get<2>(rocprim::tuple<int, double, char>{9, 9.9, 'z'});
    EXPECT_EQ(c, 'z');
}

// make_tuple: deduce types and handle reference_wrapper
TEST(MakeTuple, TypeDeductionAndValues)
{
    auto t1 = rocprim::make_tuple(10, 3.14, 'x');
    static_assert(std::is_same_v<decltype(t1), rocprim::tuple<int, double, char>>);
    EXPECT_EQ(rocprim::get<0>(t1), 10);
    EXPECT_DOUBLE_EQ(rocprim::get<1>(t1), 3.14);
    EXPECT_EQ(rocprim::get<2>(t1), 'x');

    int  a  = 5;
    auto t2 = rocprim::make_tuple(std::ref(a), 7);
    static_assert(std::is_same_v<decltype(t2), rocprim::tuple<int&, int>>);
    EXPECT_EQ(rocprim::get<0>(t2), 5);
    rocprim::get<0>(t2) = 20;
    EXPECT_EQ(a, 20);
}

// rocprim::tie and ignore: unpacking and placeholder behavior
TEST(TieIgnore, TieAssignAndIgnore)
{
    int  x = 0;
    char y = ' ';

    auto tied = rocprim::tie(x, y);
    static_assert(std::is_same_v<decltype(tied), rocprim::tuple<int&, char&>>);
    rocprim::tuple<int, char> src{42, 'A'};
    tied = src; // assigns x=42, y='A'
    EXPECT_EQ(x, 42);
    EXPECT_EQ(y, 'A');

    // ignore first element
    x = 1;
    y = 'b';

    auto tied2 = rocprim::tie(rocprim::ignore, y);
    tied2      = rocprim::tuple<int, char>{99, 'Z'};
    EXPECT_EQ(x, 1); // unchanged
    EXPECT_EQ(y, 'Z'); // updated
}
