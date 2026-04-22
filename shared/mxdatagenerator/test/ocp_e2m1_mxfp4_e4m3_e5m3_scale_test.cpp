// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <mxDataGenerator/dataTypeInfo.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
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
    const int scaleExp
        = getExponentValue<uint8_t>(scaleByte, DT::scaleInfo.mantissaBits, DT::scaleInfo.exponentBits);
    return std::pow(2.0, static_cast<double>(scaleExp - static_cast<int>(DT::scaleInfo.bias)));
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

    const int oneExp
        = getExponentValue<uint8_t>(one, TypeParam::scaleInfo.mantissaBits, TypeParam::scaleInfo.exponentBits);
    const int twoExp
        = getExponentValue<uint8_t>(two, TypeParam::scaleInfo.mantissaBits, TypeParam::scaleInfo.exponentBits);

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

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, ScaleZeroForcesZero)
{
    uint8_t scale[1] = {0};
    uint8_t data[1]  = {DGen::ocp_e2m1_mxfp4::oneMask};

    EXPECT_TRUE(isZero<TypeParam>(scale, data, 0, 0));
    EXPECT_DOUBLE_EQ(toDouble<TypeParam>(scale, data, 0, 0), 0.0);
}

TYPED_TEST(OcpE2M1MxFp4AltScaleTest, ToDoubleMatchesBaseTypeWithPowerOfTwoScaling)
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

    const uint8_t nanScale = getScaleNanByte<TypeParam>();
    for(int s = 0; s < 256; ++s)
    {
        const uint8_t scaleByte = static_cast<uint8_t>(s);
        const uint8_t scale[1]  = {scaleByte};

        for(const uint8_t dataByte : dataBytes)
        {
            const uint8_t data[1] = {dataByte};
            const double  actual  = toDouble<TypeParam>(scale, data, 0, 0);

            if(scaleByte == nanScale)
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
            const double expected = base * scaleFactorFromByte<TypeParam>(scaleByte);
            EXPECT_DOUBLE_EQ(actual, expected) << "scaleByte=" << int(scaleByte) << " dataByte=" << int(dataByte);
        }
    }
}
} // namespace

