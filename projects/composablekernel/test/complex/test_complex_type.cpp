// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstring>
#include <type_traits>

#include <hip/hip_complex.h>

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/dtype_complex.hpp"
#include "ck/utility/type_convert.hpp"

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
