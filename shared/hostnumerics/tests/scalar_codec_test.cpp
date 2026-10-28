// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <roc/hostnumerics/tensor.hpp>

#include "_scalar_codec_test_support.hpp"

TEST(ScalarCodecs, ScalarTypeInfoContract) {
    scalar_codec_test::testScalarTypeInfoContract();
}

TEST(ScalarCodecs, IntegerConversionPrimitives) {
    scalar_codec_test::testIntegerConversionPrimitives();
}

TEST(ScalarCodecsDeathTest, IntegerConversionAssertsInputRequirements) {
#ifndef NDEBUG
    using namespace roc::hostnumerics;
    using enum IntegerRounding;
    using enum IntegerOverflow;
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Identity requires the rounded value to fit.
    EXPECT_DEATH(convertToInteger<int8_t>(int32_t{300}, TowardZero, Identity), "");
    EXPECT_DEATH(convertToInteger<int8_t>(127.5, NearestEven, Identity), "");
    // ModuloWrap requires a finite input.
    EXPECT_DEATH(convertToInteger<int8_t>(infinity, TowardZero, ModuloWrap), "");
    // NaN is outside the domain of every policy.
    for (const auto overflow : {Identity, Saturate, ModuloWrap})
        EXPECT_DEATH(convertToInteger<int8_t>(nan, NearestEven, overflow), "");
#else
    GTEST_SKIP() << "Input requirements are checked only when assertions are enabled.";
#endif
}

TEST(ScalarCodecs, NearestEvenIgnoresHostRoundingMode) {
    scalar_codec_test::testNearestEvenIgnoresHostRoundingMode();
}

TEST(ScalarCodecs, IntegerCodecPolicies) {
    scalar_codec_test::testIntegerCodecPolicies();
}

TEST(ScalarCodecs, ScalarEncodings) {
    scalar_codec_test::testScalarEncodings();
}

TEST(ScalarCodecs, DirectDoubleAndExplicitFloatIntermediate) {
    using namespace roc::hostnumerics;
    ScalarConversionOptions options{IntegerRounding::NearestEven, IntegerOverflow::Saturate,
                                    BFloat16Rounding::NearestEven, FloatingSourcePrecision::Direct};
    const auto encode = [&](ScalarType type, double value) {
        Tensor result(type, Shape{1});
        result.storeFrom({0}, value, options);
        return result.loadAs<double>({0});
    };
    EXPECT_EQ(encode(ScalarType::BFloat16, 1.0 + 0x1p-8 + 0x1p-30), 1.0 + 0x1p-7);
    options.floatingSourcePrecision = FloatingSourcePrecision::Float32;
    EXPECT_EQ(encode(ScalarType::BFloat16, 1.0 + 0x1p-8 + 0x1p-30), 1.0);
    options.floatingSourcePrecision = FloatingSourcePrecision::Direct;
    EXPECT_EQ(encode(ScalarType::Float16, 1.0 + 0x1p-11 + 0x1p-30), 1.0 + 0x1p-10);
    EXPECT_EQ(encode(ScalarType::Float8E4M3, 1.0 + 0x1p-4 + 0x1p-30), 1.125);
    EXPECT_EQ(encode(ScalarType::E8M0, 1.5 - 0x1p-30), 1.0);
}
