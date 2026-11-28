// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <type_traits>

#include <hip/hip_complex.h>

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/dtype_vector.hpp"
#include "ck/utility/dtype_complex.hpp"
#include "ck/utility/functional2.hpp"
#include "ck/utility/type_convert.hpp"
#include "complex_test_data.hpp"

using ck::complex_t;

template <typename T>
class ComplexType : public ::testing::Test
{
};

using ComponentTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ComplexType, ComponentTypes);

TYPED_TEST(ComplexType, Layout)
{
    using C = complex_t<TypeParam>;
    EXPECT_EQ(sizeof(C), 2 * sizeof(TypeParam));
    EXPECT_EQ(alignof(C), 2 * sizeof(TypeParam));

    // Real part first, as in hipFloatComplex / hipDoubleComplex.
    const C c{TypeParam{3}, TypeParam{-4}};
    TypeParam parts[2];
    std::memcpy(parts, &c, sizeof(c));
    EXPECT_EQ(parts[0], TypeParam{3});
    EXPECT_EQ(parts[1], TypeParam{-4});
}

// The HIP complex type of the same precision has the same size, alignment and order of parts, so
// a buffer of either type can be read as the other.
TYPED_TEST(ComplexType, SameBitsAsHipComplex)
{
    using C = complex_t<TypeParam>;
    using H =
        std::conditional_t<std::is_same_v<TypeParam, float>, hipFloatComplex, hipDoubleComplex>;
    EXPECT_EQ(sizeof(C), sizeof(H));
    EXPECT_EQ(alignof(C), alignof(H));

    H h;
    if constexpr(std::is_same_v<TypeParam, float>)
    {
        h = make_hipFloatComplex(3.0f, -4.0f);
    }
    else
    {
        h = make_hipDoubleComplex(3.0, -4.0);
    }
    C c;
    std::memcpy(&c, &h, sizeof(c));
    EXPECT_EQ(c.real, TypeParam{3});
    EXPECT_EQ(c.imag, TypeParam{-4});
}

TYPED_TEST(ComplexType, ConstructFromZero)
{
    using C = complex_t<TypeParam>;
    const C zero{0};
    EXPECT_EQ(zero.real, TypeParam{0});
    EXPECT_EQ(zero.imag, TypeParam{0});

    // CK builds identity and "invalid element" values with type_convert<T>(0.0f).
    const C converted = ck::type_convert<C>(0.0f);
    EXPECT_EQ(converted.real, TypeParam{0});
    EXPECT_EQ(converted.imag, TypeParam{0});
}

// A real number becomes complex only when written as complex_t{r}; an implicit conversion would
// hide where a real factor is used as the complex number (r, 0).
TYPED_TEST(ComplexType, RealDoesNotConvertImplicitly)
{
    static_assert(!std::is_convertible_v<TypeParam, complex_t<TypeParam>>,
                  "complex_t(T) must stay explicit");
    static_assert(std::is_constructible_v<complex_t<TypeParam>, TypeParam>,
                  "complex_t{r} must still work");
}

TYPED_TEST(ComplexType, CountsAsOneElement)
{
    using C = complex_t<TypeParam>;
    EXPECT_TRUE(ck::is_two_component_type<C>::value);
    EXPECT_FALSE(ck::is_two_component_type<TypeParam>::value);

    EXPECT_EQ((ck::scalar_type<typename ck::vector_type<C, 1>::type>::vector_size), 1);
    EXPECT_EQ((ck::scalar_type<typename ck::vector_type<C, 2>::type>::vector_size), 2);
    EXPECT_EQ((ck::scalar_type<typename ck::vector_type<C, 4>::type>::vector_size), 4);
}

TYPED_TEST(ComplexType, AsTypeReadBack)
{
    using C = complex_t<TypeParam>;
    ck::vector_type<C, 4> v;
    ck::static_for<0, 4, 1>{}(
        [&](auto i) { v.template AsType<C>()(i) = C{TypeParam(i.value), TypeParam(-i.value)}; });
    ck::static_for<0, 4, 1>{}([&](auto i) {
        const C c = v.template AsType<C>()[i];
        EXPECT_EQ(c.real, TypeParam(i.value));
        EXPECT_EQ(c.imag, TypeParam(-i.value));
    });
}

TYPED_TEST(ComplexType, Arithmetic)
{
    using C = complex_t<TypeParam>;
    const C a{TypeParam{1}, TypeParam{2}};
    const C b{TypeParam{3}, TypeParam{4}};

    EXPECT_TRUE((a + b == C{TypeParam{4}, TypeParam{6}}));
    EXPECT_TRUE((a - b == C{TypeParam{-2}, TypeParam{-2}}));
    EXPECT_TRUE((-a == C{TypeParam{-1}, TypeParam{-2}}));
    EXPECT_TRUE((a * b == C{TypeParam{-5}, TypeParam{10}}));
    EXPECT_TRUE((ck::conj(a) == C{TypeParam{1}, TypeParam{-2}}));

    C acc = a;
    acc += b;
    acc *= b;
    EXPECT_TRUE((acc == C{TypeParam{-12}, TypeParam{34}}));
    EXPECT_TRUE(acc != a);
}

// Real part of (1 + e, 1) * (1 - e, 1) is (1 + e)(1 - e) - 1 * 1 = -e^2. With e small enough,
// the product (1 + e)(1 - e) rounds to 1, so an unfused formula returns 0. Kahan's form
// returns -e^2 exactly.
TYPED_TEST(ComplexType, MultiplyKeepsCancellingProducts)
{
    using C = complex_t<TypeParam>;
    // e^2 is below half an ulp of 1: 2^-26 for float, 2^-60 for double.
    const TypeParam e = ck::is_same_v<TypeParam, float> ? TypeParam{1} / TypeParam{1 << 13}
                                                        : TypeParam{1} / TypeParam{1 << 30};
    const C a{TypeParam{1} + e, TypeParam{1}};
    const C b{TypeParam{1} - e, TypeParam{1}};

    const C p = a * b;
    EXPECT_EQ(p.real, -e * e);
    EXPECT_EQ(p.imag, TypeParam{2});
}

// Zero parts carry the sign IEEE arithmetic gives a * c - b * d and a * d + b * c.
TYPED_TEST(ComplexType, MultiplyKeepsSignOfZero)
{
    using T = TypeParam;
    using C = complex_t<T>;

    // (-0 - 0, 0 + -0) = (-0, +0)
    const C p = C{T{0}, T{0}} * C{T{-1}, T{0}};
    EXPECT_TRUE(std::signbit(p.real));
    EXPECT_FALSE(std::signbit(p.imag));

    // (+0 - 0, -0 + -0) = (+0, -0)
    const C q = C{-T{0}, T{0}} * C{-T{0}, T{0}};
    EXPECT_FALSE(std::signbit(q.real));
    EXPECT_TRUE(std::signbit(q.imag));
}

// Finite factors give an infinity only in a part of the exact product that is too large for the
// type; every other part keeps its exact value here, even when a product inside overflows.
TYPED_TEST(ComplexType, MultiplyOverflowOnlyWhenTooLarge)
{
    using T     = TypeParam;
    using C     = complex_t<T>;
    using L     = std::numeric_limits<T>;
    const T big = std::sqrt(L::max()) * T{2}; // big * big overflows

    // (1, big)^2 = (1 - big^2, 2 big): only the real part is too large.
    const C p1 = C{T{1}, big} * C{T{1}, big};
    EXPECT_EQ(p1.real, -L::infinity());
    EXPECT_EQ(p1.imag, T{2} * big);

    // (big, big)^2 = (0, 2 big^2): both real products overflow, yet the real part is 0.
    const C p2 = C{big, big} * C{big, big};
    EXPECT_EQ(p2.real, T{0});
    EXPECT_EQ(p2.imag, L::infinity());

    // (0, -1.5) * (max, m): the real part comes from the small product alone.
    const T m  = L::min() * T{4};
    const C p3 = C{T{0}, T{-1.5}} * C{L::max(), m};
    EXPECT_EQ(p3.real, T{1.5} * m);
    EXPECT_EQ(p3.imag, -L::infinity());

    // (1, big) * (big, -1) = (2 big, big^2 - 1): the real part is finite, only the imaginary
    // part overflows.
    const C p4 = C{T{1}, big} * C{big, T{-1}};
    EXPECT_EQ(p4.real, T{2} * big);
    EXPECT_EQ(p4.imag, L::infinity());
}

// C11 Annex G.5.1: an infinity times a nonzero value is an infinity (a complex value with an
// infinite part), never NaN + iNaN. NaN times a finite value stays NaN. The reference is the host
// compiler's own complex multiply (std::complex, which calls __mulsc3 / __muldc3): every pair of
// special values with an infinite or NaN part must give the same bits in each part, or NaN in
// both. Pairs of finite factors are not compared: there CK rescales a product that overflows,
// where Annex G's code may return an infinity (see MultiplyOverflowOnlyWhenTooLarge).
TYPED_TEST(ComplexType, MultiplyNonFiniteMatchesStdComplex)
{
    using T       = TypeParam;
    using C       = complex_t<T>;
    using S       = std::complex<T>;
    const auto v  = ck_complex_test::special_values<T>();
    const auto m  = v.size();
    const auto mm = m * m;
    long bad      = 0;
    for(std::size_t i = 0; i < mm * mm; ++i)
    {
        const std::size_t x = i % mm, y = i / mm;
        const T a = v[x % m], b = v[x / m], c = v[y % m], d = v[y / m];
        const bool finite =
            std::isfinite(a) && std::isfinite(b) && std::isfinite(c) && std::isfinite(d);
        const C p = C{a, b} * C{c, d};
        const S q = S{a, b} * S{c, d};
        if(!finite && !(ck_complex_test::same_part(p.real, q.real()) &&
                        ck_complex_test::same_part(p.imag, q.imag())))
        {
            if(bad < 5)
            {
                ADD_FAILURE() << "(" << a << ", " << b << ") * (" << c << ", " << d << "): ck ("
                              << p.real << ", " << p.imag << "), std::complex (" << q.real() << ", "
                              << q.imag() << ")";
            }
            ++bad;
        }
    }
    EXPECT_EQ(bad, 0);
}
