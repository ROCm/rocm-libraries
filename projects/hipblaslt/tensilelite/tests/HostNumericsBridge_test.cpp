// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <TensileLite/Client/hostnumerics/HostNumericsBridge.hpp>

#include <Tensile/DataTypes.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>

namespace
{
    template <typename T>
    bool productValuesClose(rocisa::DataType type, T observed, T expected, double threshold = -1.0)
    {
        using roc::hostnumerics::ComparisonOptions;
        using roc::hostnumerics::Layout;
        using roc::hostnumerics::Shape;
        using roc::hostnumerics::Tensor;

        const std::array<T, 1> observedStorage{observed};
        const std::array<T, 1> expectedStorage{expected};
        ComparisonOptions      options
            = TensileLite::Client::validationComparisonOptions(type, threshold);
        options.computeElementwiseStatistics = false;
        options.computeFrobenius             = false;
        const auto scalarType                = TensileLite::Client::toHostNumericsScalarType(type);
        const auto layout                    = Layout::contiguousLastDimensionFastest(Shape{1});
        return roc::hostnumerics::compare(
                   Tensor::copyEncodedBackingStorage(
                       scalarType, layout, std::as_bytes(std::span<const T>(observedStorage))),
                   Tensor::copyEncodedBackingStorage(
                       scalarType, layout, std::as_bytes(std::span<const T>(expectedStorage))),
                   options)
            .passed();
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

TEST(HostNumericsComparisonAdapter, PreservesFloatOnlyPositiveThresholdOverride)
{
    EXPECT_FALSE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.02f));
    EXPECT_TRUE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.02f, 0.01));
    EXPECT_FALSE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.02f, 0.0));
    EXPECT_FALSE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.02f, -0.01));

    EXPECT_FALSE(productValuesClose<double>(rocisa::DataType::Double, 1.0, 1.02, 0.1));
    EXPECT_THROW(TensileLite::Client::validationComparisonOptions(rocisa::DataType::None, -1.0),
                 std::runtime_error);
}

TEST(HostNumericsComparisonAdapter, UsesStrictSymmetricTolerance)
{
    using TensileLite::BFloat16;
    const auto bfloat16Close = [](float observed, float expected) {
        return productValuesClose<BFloat16>(
            rocisa::DataType::BFloat16, BFloat16(observed), BFloat16(expected));
    };
    EXPECT_TRUE(bfloat16Close(1.3125f, 1.0f));
    EXPECT_TRUE(bfloat16Close(1.0f, 1.3125f));
    EXPECT_FALSE(bfloat16Close(0.71875f, 1.0f));
    EXPECT_FALSE(bfloat16Close(1.0f, 0.71875f));

    EXPECT_FALSE(productValuesClose<float>(rocisa::DataType::Float, 0.0f, 1.0f, 0.5));
    EXPECT_TRUE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.0002f));
    EXPECT_FALSE(productValuesClose<float>(rocisa::DataType::Float, 1.0f, 1.001f));
}

TEST(HostNumericsComparisonAdapter, MatchesBFloat16ScalarRuleAcrossBoundaryValues)
{
    using TensileLite::BFloat16;
    const std::array<float, 8> values{0.0f,
                                      0.71875f,
                                      1.0f,
                                      1.3125f,
                                      -1.0f,
                                      -1.3125f,
                                      std::numeric_limits<float>::infinity(),
                                      std::numeric_limits<float>::quiet_NaN()};
    for(float observed : values)
        for(float expected : values)
        {
            const BFloat16 encodedObserved(observed);
            const BFloat16 encodedExpected(expected);
            const float    a = static_cast<float>(encodedObserved);
            const float    b = static_cast<float>(encodedExpected);
            const bool     legacy
                = a == b || std::fabs(a - b) < 0.1f * (std::fabs(a) + std::fabs(b) + 1.0f);
            EXPECT_EQ(productValuesClose<BFloat16>(
                          rocisa::DataType::BFloat16, encodedObserved, encodedExpected),
                      legacy)
                << "observed=" << observed << ", expected=" << expected;
        }
}

TEST(HostNumericsComparisonAdapter, UsesTensileLowPrecisionConversions)
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

TEST(HostNumericsComparisonAdapter, PreservesTensileRawFp8SpecialValues)
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

TEST(HostNumericsBridge, SamplingAndLayoutRejectUnrepresentableShapeAsAResult)
{
    const size_t maximum = std::numeric_limits<size_t>::max();
    for(const auto& sizes : {std::vector<size_t>{maximum, 2}, std::vector<size_t>{2, maximum}})
    {
        const std::vector<size_t>           strides(sizes.size(), 0);
        const TensileLite::TensorDescriptor descriptor("oversized",
                                                       rocisa::DataType::Float,
                                                       sizes.begin(),
                                                       sizes.end(),
                                                       strides.begin(),
                                                       strides.end());
        const auto selection = TensileLite::Client::getReferenceIndexRange(descriptor, 1);
        ASSERT_FALSE(selection.has_value());
        EXPECT_NE(std::string(TensileLite::Client::hostnumerics::detail::message(selection.error()))
                      .find("element count exceeds size_t"),
                  std::string::npos);
        const auto layout = TensileLite::Client::getHostnumericsLayout(descriptor);
        ASSERT_FALSE(layout.has_value());
        EXPECT_NE(std::string(TensileLite::Client::hostnumerics::detail::message(layout.error()))
                      .find("element count exceeds size_t"),
                  std::string::npos);
        EXPECT_THROW(TensileLite::Client::referenceIndexRange(descriptor, 1), std::runtime_error);
    }
}

TEST(HostNumericsBridge, SamplingAndLayoutAcceptOversizedEmptyShapes)
{
    const size_t maximum = std::numeric_limits<size_t>::max();
    for(const auto& sizes : {std::vector<size_t>{maximum, 2, 0},
                             std::vector<size_t>{0, maximum, 2},
                             std::vector<size_t>{maximum, 0, 2}})
    {
        const std::vector<size_t>           strides(sizes.size(), 0);
        const TensileLite::TensorDescriptor descriptor("empty",
                                                       rocisa::DataType::Float,
                                                       sizes.begin(),
                                                       sizes.end(),
                                                       strides.begin(),
                                                       strides.end());
        const auto selection = TensileLite::Client::getReferenceIndexRange(descriptor, 1);
        ASSERT_TRUE(selection.has_value());
        const auto layout = TensileLite::Client::getHostnumericsLayout(descriptor);
        ASSERT_TRUE(layout.has_value());
        EXPECT_EQ(layout->shape().elementCount(), 0);
        EXPECT_NO_THROW(TensileLite::Client::referenceIndexRange(descriptor, 1));
    }
}
