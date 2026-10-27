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
