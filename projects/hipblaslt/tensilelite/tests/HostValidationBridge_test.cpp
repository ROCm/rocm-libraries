// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/adapters/tensilelite/HostValidationBridge.hpp>

#include <Tensile/DataTypes.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

namespace
{
    template <typename T>
    bool productValuesClose(rocisa::DataType type, T observed, T expected, double threshold = -1.0)
    {
        using namespace roc::host_validation;

        const std::array<T, 1> observedStorage{observed};
        const std::array<T, 1> expectedStorage{expected};
        ComparisonOptions      options
            = TensileLite::Client::validationComparisonOptions(type, threshold);
        options.computePointwiseStatistics = false;
        options.computeFrobenius           = false;
        const ComparisonResult report
            = TensileLite::Client::compareHostBuffers(type,
                                                      observedStorage.data(),
                                                      expectedStorage.data(),
                                                      Layout::contiguous(Shape{1}),
                                                      options);
        return report.passed();
    }

    template <typename T>
    T fromRawByte(uint8_t byte)
    {
        static_assert(sizeof(T) == 1);
        T value{};
        std::memcpy(&value, &byte, sizeof(byte));
        return value;
    }
}

TEST(HostValidationComparisonAdapter, PreservesFloatOnlyPositiveThresholdOverride)
{
    EXPECT_FALSE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.02f));
    EXPECT_TRUE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.02f, 0.01));
    EXPECT_FALSE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.02f, 0.0));
    EXPECT_FALSE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.02f, -0.01));

    EXPECT_FALSE(productValuesClose<double>(rocisa::DataType::Double, 1.0, 1.02, 0.1));
}

TEST(HostValidationComparisonAdapter, UsesTensileLowPrecisionConversions)
{
    using namespace TensileLite;

    EXPECT_TRUE(productValuesClose<Half>(rocisa::DataType::Half, Half(1.0f), Half(1.02f)));
    EXPECT_FALSE(productValuesClose<Half>(rocisa::DataType::Half, Half(1.0f), Half(1.1f)));

    EXPECT_TRUE(
        productValuesClose<BFloat16>(rocisa::DataType::BFloat16, BFloat16(2.0f), BFloat16(2.25f)));
    EXPECT_FALSE(
        productValuesClose<BFloat16>(rocisa::DataType::BFloat16, BFloat16(1.0f), BFloat16(2.0f)));

    EXPECT_TRUE(productValuesClose<Float8>(rocisa::DataType::Float8, Float8(2.0f), Float8(2.5f)));
    EXPECT_FALSE(productValuesClose<Float8>(rocisa::DataType::Float8, Float8(1.0f), Float8(2.0f)));

    EXPECT_TRUE(
        productValuesClose<BFloat8>(rocisa::DataType::BFloat8, BFloat8(1.0f), BFloat8(1.5f)));
    EXPECT_FALSE(
        productValuesClose<BFloat8>(rocisa::DataType::BFloat8, BFloat8(1.0f), BFloat8(3.0f)));

    EXPECT_TRUE(productValuesClose<Float8_fnuz>(
        rocisa::DataType::Float8_fnuz, Float8_fnuz(2.0f), Float8_fnuz(2.5f)));
    EXPECT_FALSE(productValuesClose<Float8_fnuz>(
        rocisa::DataType::Float8_fnuz, Float8_fnuz(1.0f), Float8_fnuz(2.0f)));

    EXPECT_TRUE(productValuesClose<BFloat8_fnuz>(
        rocisa::DataType::BFloat8_fnuz, BFloat8_fnuz(1.0f), BFloat8_fnuz(1.5f)));
    EXPECT_FALSE(productValuesClose<BFloat8_fnuz>(
        rocisa::DataType::BFloat8_fnuz, BFloat8_fnuz(1.0f), BFloat8_fnuz(3.0f)));
}

TEST(HostValidationComparisonAdapter, PreservesTensileRawFp8SpecialValues)
{
    using namespace TensileLite;

    const Float8 float8Nan = fromRawByte<Float8>(0x7f);
    EXPECT_FALSE(productValuesClose<Float8>(rocisa::DataType::Float8, float8Nan, float8Nan));
    const Float8 float8Maximum = fromRawByte<Float8>(0x7e);
    EXPECT_TRUE(productValuesClose<Float8>(rocisa::DataType::Float8, float8Maximum, float8Maximum));

    const BFloat8 bfloat8Nan = fromRawByte<BFloat8>(0x7f);
    EXPECT_FALSE(productValuesClose<BFloat8>(rocisa::DataType::BFloat8, bfloat8Nan, bfloat8Nan));
    const BFloat8 bfloat8Infinity         = fromRawByte<BFloat8>(0x7c);
    const BFloat8 bfloat8NegativeInfinity = fromRawByte<BFloat8>(0xfc);
    EXPECT_TRUE(
        productValuesClose<BFloat8>(rocisa::DataType::BFloat8, bfloat8Infinity, bfloat8Infinity));
    EXPECT_FALSE(productValuesClose<BFloat8>(
        rocisa::DataType::BFloat8, bfloat8Infinity, bfloat8NegativeInfinity));

    const Float8_fnuz float8FnuzNan = fromRawByte<Float8_fnuz>(0x80);
    EXPECT_FALSE(productValuesClose<Float8_fnuz>(
        rocisa::DataType::Float8_fnuz, float8FnuzNan, float8FnuzNan));
    const Float8_fnuz float8FnuzMaximum = fromRawByte<Float8_fnuz>(0x7f);
    EXPECT_TRUE(productValuesClose<Float8_fnuz>(
        rocisa::DataType::Float8_fnuz, float8FnuzMaximum, float8FnuzMaximum));

    const BFloat8_fnuz bfloat8FnuzNan = fromRawByte<BFloat8_fnuz>(0x80);
    EXPECT_FALSE(productValuesClose<BFloat8_fnuz>(
        rocisa::DataType::BFloat8_fnuz, bfloat8FnuzNan, bfloat8FnuzNan));
    const BFloat8_fnuz bfloat8FnuzMaximum = fromRawByte<BFloat8_fnuz>(0x7f);
    EXPECT_TRUE(productValuesClose<BFloat8_fnuz>(
        rocisa::DataType::BFloat8_fnuz, bfloat8FnuzMaximum, bfloat8FnuzMaximum));
}
