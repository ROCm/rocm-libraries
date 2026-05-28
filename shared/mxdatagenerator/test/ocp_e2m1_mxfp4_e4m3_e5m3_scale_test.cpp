// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <mxDataGenerator/DataGenerator.hpp>
#include <mxDataGenerator/dataTypeInfo.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace DGen;

template <typename DT>
constexpr ScaleType scaleTypeFor()
{
    if constexpr(DT::scaleInfo.exponentBits == 5 && DT::scaleInfo.mantissaBits == 3)
        return ScaleType::E5M3;
    else if constexpr(DT::scaleInfo.exponentBits == 4 && DT::scaleInfo.mantissaBits == 3)
        return ScaleType::E4M3;
    else
        static_assert(std::is_same_v<DT, void>, "Unexpected scale format for DT.");
}

template <typename DT>
uint8_t getScaleOneByte()
{
    constexpr auto st = scaleTypeFor<DT>();
    if constexpr(st == ScaleType::E5M3)
        return static_cast<uint8_t>(getScaleOne<ScaleType::E5M3>());
    else
        return static_cast<uint8_t>(getScaleOne<ScaleType::E4M3>());
}

template <typename DT>
uint8_t getScaleTwoByte()
{
    constexpr auto st = scaleTypeFor<DT>();
    if constexpr(st == ScaleType::E5M3)
        return static_cast<uint8_t>(getScaleTwo<ScaleType::E5M3>());
    else
        return static_cast<uint8_t>(getScaleTwo<ScaleType::E4M3>());
}

template <typename DT>
uint8_t getScaleNanByte()
{
    constexpr auto st = scaleTypeFor<DT>();
    if constexpr(st == ScaleType::E5M3)
        return static_cast<uint8_t>(getScaleNan<ScaleType::E5M3>());
    else
        return static_cast<uint8_t>(getScaleNan<ScaleType::E4M3>());
}

template <typename DT>
double scaleFactorFromByte(uint8_t scaleByte)
{
    constexpr auto st = scaleTypeFor<DT>();
    if constexpr(st == ScaleType::E5M3)
        return getScaleValue<ScaleInfo<ScaleType::E5M3>>(scaleByte);
    else
        return getScaleValue<ScaleInfo<ScaleType::E4M3>>(scaleByte);
}

template <typename DT>
bool hasNonzeroScaleMantissa(const std::vector<uint8_t>& scales)
{
    return std::any_of(scales.begin(), scales.end(), [](uint8_t scale) {
        const uint8_t mantissaMask = (1 << DT::scaleInfo.mantissaBits) - 1;
        return (scale & mantissaMask) != 0;
    });
}

template <typename DT>
bool hasDenormalScale(const std::vector<uint8_t>& scales)
{
    return std::any_of(scales.begin(), scales.end(), [](uint8_t scale) {
        const uint8_t mantissaMask = (1 << DT::scaleInfo.mantissaBits) - 1;
        const uint8_t exponent
            = getExponentValue<uint8_t>(scale, DT::scaleInfo.mantissaBits, DT::scaleInfo.exponentBits);
        return exponent == 0 && (scale & mantissaMask) != 0;
    });
}

template <typename DT>
double denormalOnlyGenerationMax()
{
    constexpr auto st = scaleTypeFor<DT>();
    if constexpr(st == ScaleType::E5M3)
        return std::ldexp(1.0, -16);
    else
        return std::ldexp(1.0, -8);
}

template <typename DT>
class OcpE2M1MxFp4AltScaleTest : public ::testing::Test
{
};

using OcpE2M1AltScaleTypes
    = ::testing::Types<DGen::ocp_e2m1_mxfp4_e5m3, DGen::ocp_e2m1_mxfp4_e4m3>;
TYPED_TEST_SUITE(OcpE2M1MxFp4AltScaleTest, OcpE2M1AltScaleTypes);

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, ScaleOneTwoHaveExpectedExponent)
{
    const uint8_t one = getScaleOneByte<TypeParam>();
    const uint8_t two = getScaleTwoByte<TypeParam>();

    const int oneExp = getExponentValue<uint8_t>(
        one, TypeParam::scaleInfo.mantissaBits, TypeParam::scaleInfo.exponentBits);
    const int twoExp = getExponentValue<uint8_t>(
        two, TypeParam::scaleInfo.mantissaBits, TypeParam::scaleInfo.exponentBits);

    EXPECT_EQ(oneExp, static_cast<int>(TypeParam::scaleInfo.bias));
    EXPECT_EQ(twoExp, static_cast<int>(TypeParam::scaleInfo.bias) + 1);
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, SetOneCreatesOneInDouble)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {0};

    setOne<TypeParam>(scale, data, 0, 0, false);
    EXPECT_TRUE(isOne<TypeParam>(scale, data, 0, 0));
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 1.0);

    setOne<TypeParam>(scale, data, 0, 0, true);
    EXPECT_TRUE(isOne<TypeParam>(scale, data, 0, 0));
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 1.0);
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, SetNaNSetsScaleNaNAndPropagates)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {DGen::ocp_e2m1_mxfp4::oneMask};

    setNaN<TypeParam>(scale, data, 0, 0);
    EXPECT_TRUE(isNaN<TypeParam>(scale, data, 0, 0));
    EXPECT_TRUE(std::isnan(toDouble<TypeParam>(scale, data, 0, 0)));
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, ScaleNaNUsesExpectedByte)
{
    constexpr auto st = scaleTypeFor<TypeParam>();
    if constexpr(st == ScaleType::E5M3)
    {
        EXPECT_EQ(getScaleNanByte<TypeParam>(), 0xff);
    }
    else
    {
        EXPECT_EQ(getScaleNanByte<TypeParam>(), 0x7f);

        uint8_t scale[1] = {0xff};
        uint8_t data[1]  = {DGen::ocp_e2m1_mxfp4::oneMask};
        EXPECT_TRUE(isNaN<TypeParam>(scale, data, 0, 0));
        EXPECT_TRUE(std::isnan(toDouble<TypeParam>(scale, data, 0, 0)));
    }
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, ScaleZeroForcesZero)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {DGen::ocp_e2m1_mxfp4::oneMask};

    EXPECT_TRUE(isZero<TypeParam>(scale, data, 0, 0));
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 0.0);
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, ScaleValuesMatchEncodingTable)
{
    std::vector<std::pair<uint8_t, double>> expectedScales;
    constexpr auto                          st = scaleTypeFor<TypeParam>();
    if constexpr(st == ScaleType::E5M3)
    {
        expectedScales = {
            {0x00, 0.0},
            {0x01, std::pow(2.0, -17)},
            {0x07, 0.875 * std::pow(2.0, -14)},
            {0x08, std::pow(2.0, -14)},
            {0x78, 1.0},
            {0x80, 2.0},
            {0xfe, 114688.0},
        };
    }
    else
    {
        expectedScales = {
            {0x00, 0.0},
            {0x01, std::pow(2.0, -9)},
            {0x07, 0.875 * std::pow(2.0, -6)},
            {0x08, std::pow(2.0, -6)},
            {0x38, 1.0},
            {0x40, 2.0},
            {0x7e, 448.0},
        };
    }

    const uint8_t data[1]       = {DGen::ocp_e2m1_mxfp4::oneMask};
    const uint8_t packedData[1] = {DGen::ocp_e2m1_mxfp4::oneMask};
    for(const auto& [scaleByte, expected] : expectedScales)
    {
        const uint8_t scale[1] = {scaleByte};

        EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), expected)
            << "scaleByte=" << int(scaleByte);
        EXPECT_DOUBLE_EQ(toDoublePacked<TypeParam>(scale, packedData, 0, 0), expected)
            << "scaleByte=" << int(scaleByte);
        EXPECT_FLOAT_EQ(toFloat<TypeParam>(scale, data, 0, 0), static_cast<float>(expected))
            << "scaleByte=" << int(scaleByte);
        EXPECT_FLOAT_EQ(toFloatPacked<TypeParam>(scale, packedData, 0, 0),
                        static_cast<float>(expected))
            << "scaleByte=" << int(scaleByte);
    }
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, ToDoubleMatchesBaseTypeWithDecodedScaling)
{
    const uint8_t baseScale[1] = {DGen::Constants::E8M0_1};

    const std::vector<uint8_t> dataBytes = {
        DGen::ocp_e2m1_mxfp4::positiveZeroMask,
        DGen::ocp_e2m1_mxfp4::oneMask,
        DGen::ocp_e2m1_mxfp4::dataMaxPositiveNormalMask,
        DGen::ocp_e2m1_mxfp4::dataMaxNegativeNormalMask,
        DGen::ocp_e2m1_mxfp4::dataMaxPositiveSubNormalMask,
        DGen::ocp_e2m1_mxfp4::dataMaxNegativeSubNormalMask,
    };

    for(int s = 0; s < 256; ++s)
    {
        const uint8_t scaleByte = static_cast<uint8_t>(s);
        const uint8_t scale[1]  = {scaleByte};
        const double  scaleValue = scaleFactorFromByte<TypeParam>(scaleByte);

        for(const uint8_t dataByte : dataBytes)
        {
            const uint8_t data[1] = {dataByte};
            const double  actual  = toDouble<TypeParam>(scale, data, 0, 0);

            if(std::isnan(scaleValue))
            {
                EXPECT_TRUE(std::isnan(actual));
                continue;
            }

            if(scaleByte == 0 || dataByte == DGen::ocp_e2m1_mxfp4::positiveZeroMask
               || dataByte == DGen::ocp_e2m1_mxfp4::negativeZeroMask)
            {
                EXPECT_DOUBLE_EQ(actual, 0.0);
                continue;
            }

            const double base = toDouble<DGen::ocp_e2m1_mxfp4>(baseScale, data, 0, 0);
            const double expected = base * scaleValue;
            EXPECT_DOUBLE_EQ(actual, expected)
                << "scaleByte=" << int(scaleByte) << " dataByte=" << int(dataByte);
        }
    }
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, BoundedGenerationUsesNonzeroScaleMantissas)
{
    DataGeneratorOptions opts;
    opts.initMode     = DataInitMode(Bounded{});
    opts.min          = -1.0;
    opts.max          = 1.0;
    opts.blockScaling = 1;
    opts.forceDenorm  = false;

    auto dgen = DataGenerator<TypeParam>();
    dgen.setSeed(12345);
    dgen.generate({4096}, {1}, opts);

    const auto scales = dgen.getScaleBytes();
    EXPECT_TRUE(hasNonzeroScaleMantissa<TypeParam>(scales));
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, UnboundedGenerationUsesNonzeroScaleMantissas)
{
    DataGeneratorOptions opts;
    opts.initMode     = DataInitMode(Unbounded{});
    opts.blockScaling = 1;
    opts.forceDenorm  = false;

    auto dgen = DataGenerator<TypeParam>();
    dgen.setSeed(67890);
    dgen.generate({4096}, {1}, opts);

    const auto scales = dgen.getScaleBytes();
    EXPECT_TRUE(hasNonzeroScaleMantissa<TypeParam>(scales));
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, SmallRangeBoundedGenerationUsesDenormalScales)
{
    DataGeneratorOptions opts;
    opts.initMode     = DataInitMode(Bounded{});
    opts.min          = 0.0;
    opts.max          = denormalOnlyGenerationMax<TypeParam>();
    opts.blockScaling = 1;
    opts.forceDenorm  = false;

    auto dgen = DataGenerator<TypeParam>();
    dgen.setSeed(13579);
    dgen.generate({1024}, {1}, opts);

    const auto scales = dgen.getScaleBytes();
    EXPECT_TRUE(hasDenormalScale<TypeParam>(scales));
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, GeneratedReferencesUseDecodedScaleValues)
{
    DataGeneratorOptions opts;
    opts.initMode     = DataInitMode(Bounded{});
    opts.min          = -1.0;
    opts.max          = 1.0;
    opts.blockScaling = 1;
    opts.forceDenorm  = false;

    auto dgen = DataGenerator<TypeParam>();
    dgen.setSeed(24680);
    dgen.generate({512}, {1}, opts);

    const auto data      = dgen.getDataBytes();
    const auto scales    = dgen.getScaleBytes();
    const auto reference = dgen.getReferenceDouble();

    const uint8_t baseScale[1] = {DGen::Constants::E8M0_1};
    for(index_t i = 0; i < reference.size(); i++)
    {
        const double dataValue = toDoublePacked<DGen::ocp_e2m1_mxfp4>(baseScale, data.data(), 0, i);
        const double expected  = scaleFactorFromByte<TypeParam>(scales[i]) * dataValue;
        EXPECT_DOUBLE_EQ(reference[i], expected) << "i=" << i << " scale=" << int(scales[i]);
    }
}

TEST(OcpE2M1MxFp4E8M0ScaleGenerationTest, BoundedGenerationKeepsExponentOnlyScaleFormat)
{
    DataGeneratorOptions opts;
    opts.initMode     = DataInitMode(Bounded{});
    opts.min          = -1.0;
    opts.max          = 1.0;
    opts.blockScaling = 1;
    opts.forceDenorm  = false;

    auto dgen = DataGenerator<DGen::ocp_e2m1_mxfp4>();
    dgen.setSeed(12345);
    dgen.generate({512}, {1}, opts);

    const auto scales = dgen.getScaleBytes();
    EXPECT_TRUE(std::none_of(scales.begin(), scales.end(), [](uint8_t scale) {
        return scale == DGen::Constants::E8M0_NAN;
    }));
}
} // namespace

