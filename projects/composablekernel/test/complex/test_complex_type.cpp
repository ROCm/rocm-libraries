// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
