// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "dataTypeInfo.hpp"

inline uint8_t signedNaNMaskOcpE5m2Mxfp8(uint sign)
{
    return sign ? 0b11111111 : 0b01111111;
}

template <>
inline bool isNaN<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes,
                                  index_t         scaleIndex,
                                  index_t         dataIndex)
{
    uint8_t data  = *(dataBytes + dataIndex);
    uint8_t scale = *(scaleBytes + scaleIndex);

    if(scale == getScaleNan<ScaleType::E8M0>())
        return true;

    // set sign bit to 1
    data |= ocp_e5m2_mxfp8::signBitMask;

    auto& nanValues{ocp_e5m2_mxfp8::dataNaNMasks};
    return std::count(nanValues.begin(), nanValues.end(), data) > 0;
}

template <>
inline bool isNaNPacked<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        index_t         scaleIndex,
                                        index_t         dataIndex)
{
    return isNaN<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isInf<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes,
                                  index_t         scaleIndex,
                                  index_t         dataIndex)
{
    if(scaleBytes[scaleIndex] == getScaleNan<ScaleType::E8M0>())
        return false;

    uint8_t data = *(dataBytes + dataIndex);

    //set the sign bit to zero as it does not matter
    data &= ~ocp_e5m2_mxfp8::signBitMask;

    return data == ocp_e5m2_mxfp8::positiveInfMask;
}

template <>
inline bool isInfPacked<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        index_t         scaleIndex,
                                        index_t         dataIndex)
{
    return isInf<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isZero<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                   uint8_t const* dataBytes,
                                   index_t         scaleIndex,
                                   index_t         dataIndex)
{

    if(isNaN<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    //keep bits 0-7 same, set sign bit to 0
    //no need to check for scale as it doesn't have a zero representation
    uint8_t data = *(dataBytes + dataIndex) & (~ocp_e5m2_mxfp8::signBitMask);

    return data == ocp_e5m2_mxfp8::positiveZeroMask;
}

template <>
inline bool isZeroPacked<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                         uint8_t const* dataBytes,
                                         index_t         scaleIndex,
                                         index_t         dataIndex)
{
    return isZero<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline double toDouble<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                       uint8_t const* dataBytes,
                                       index_t         scaleIndex,
                                       index_t         dataIndex)
{
    if(isNaN<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
    {
        uint8_t data = *(dataBytes + dataIndex);
        return std::copysign(0.0, data & ocp_e5m2_mxfp8::signBitMask ? -1.0 : 1.0);
    }

    if(isInf<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
    {
        return std::numeric_limits<double>::infinity()
               * ((*(dataBytes + dataIndex) >> (ocp_e5m2_mxfp8::dataInfo.exponentBits
                                                + ocp_e5m2_mxfp8::dataInfo.mantissaBits))
                      ? -1
                      : 1);
    }

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e5m2_mxfp8::scaleInfo.mantissaBits,
                                             ocp_e5m2_mxfp8::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E5M2_MXFP8_DATA, ScaleInfo<ScaleType::E8M0>>(data, scaleExp);
}

template <>
inline double toDoublePacked<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             index_t         scaleIndex,
                                             index_t         dataIndex)
{
    return toDouble<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline float toFloat<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                     uint8_t const* dataBytes,
                                     index_t         scaleIndex,
                                     index_t         dataIndex)
{
    if(isNaN<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
    {
        uint8_t data = *(dataBytes + dataIndex);
        return std::copysign(0.0f, data & ocp_e5m2_mxfp8::signBitMask ? -1.0f : 1.0f);
    }

    if(isInf<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
    {
        return std::numeric_limits<float>::infinity()
               * ((*(dataBytes + dataIndex) >> (ocp_e5m2_mxfp8::dataInfo.exponentBits
                                                + ocp_e5m2_mxfp8::dataInfo.mantissaBits))
                      ? -1
                      : 1);
    }

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e5m2_mxfp8::scaleInfo.mantissaBits,
                                             ocp_e5m2_mxfp8::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E5M2_MXFP8_DATA, ScaleInfo<ScaleType::E8M0>>(data, scaleExp);
}

template <>
inline float toFloatPacked<ocp_e5m2_mxfp8>(uint8_t const* scaleBytes,
                                           uint8_t const* dataBytes,
                                           index_t         scaleIndex,
                                           index_t         dataIndex)
{
    return toFloat<ocp_e5m2_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isSubnorm<ocp_e5m2_mxfp8>(uint8_t const* dataBytes, index_t dataIndex)
{
    uint8_t data     = *(dataBytes + dataIndex);
    uint8_t exponent = getExponentValue<uint8_t>(
        data, ocp_e5m2_mxfp8::dataInfo.mantissaBits, ocp_e5m2_mxfp8::dataInfo.exponentBits);
    uint8_t mantissa = data & ((1 << ocp_e5m2_mxfp8::dataInfo.mantissaBits) - 1);
    return exponent == 0 && mantissa != 0;
}

template <>
inline bool isSubnormPacked<ocp_e5m2_mxfp8>(uint8_t const* dataBytes, index_t dataIndex)
{
    return isSubnorm<ocp_e5m2_mxfp8>(dataBytes, dataIndex);
}

template <>
inline void setOne<ocp_e5m2_mxfp8>(
    uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex, bool subNormal)
{
    *(scaleBytes + scaleIndex) = subNormal ? Constants::E8M0_142 : Constants::E8M0_1;
    *(dataBytes + dataIndex)
        = subNormal ? ocp_e5m2_mxfp8::dataSubNormalOneMask : ocp_e5m2_mxfp8::oneMask;
}

template <>
inline void setZero<ocp_e5m2_mxfp8>(uint8_t* scaleBytes [[maybe_unused]],
                                    uint8_t* dataBytes,
                                    index_t   scaleIndex [[maybe_unused]],
                                    index_t   dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e5m2_mxfp8::positiveZeroMask;
}

template <>
inline void setNaN<ocp_e5m2_mxfp8>(uint8_t* scaleBytes [[maybe_unused]],
                                   uint8_t* dataBytes,
                                   index_t   scaleIndex [[maybe_unused]],
                                   index_t   dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e5m2_mxfp8::dataNaNMasks[0];
}

template <>
inline void setInf<ocp_e5m2_mxfp8>(uint8_t* scaleBytes [[maybe_unused]],
                                   uint8_t* dataBytes,
                                   index_t   scaleIndex [[maybe_unused]],
                                   index_t   dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e5m2_mxfp8::positiveInfMask;
}

template <>
inline void
    setDataMax<ocp_e5m2_mxfp8>(uint8_t* dataBytes, index_t dataIndex, bool subNormal, bool positive)
{
    if(subNormal)
        *(dataBytes + dataIndex) = positive ? ocp_e5m2_mxfp8::dataMaxPositiveSubNormalMask
                                            : ocp_e5m2_mxfp8::dataMaxNegativeSubNormalMask;
    else
        *(dataBytes + dataIndex) = positive ? ocp_e5m2_mxfp8::dataMaxPositiveNormalMask
                                            : ocp_e5m2_mxfp8::dataMaxNegativeNormalMask;
}

template <>
inline uint64_t satConvertToType<ocp_e5m2_mxfp8>(float value)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return signedNaNMaskOcpE5m2Mxfp8(sign);
    }

    uint8_t res = convertToType<uint8_t, ocp_e5m2_mxfp8>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e5m2_mxfp8>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e5m2_mxfp8::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? ocp_e5m2_mxfp8::dataMaxNegativeNormalMask
                         : ocp_e5m2_mxfp8::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e5m2_mxfp8::dataMinSubNormalNumber)
        return sign ? ocp_e5m2_mxfp8::negativeZeroMask : ocp_e5m2_mxfp8::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToType<ocp_e5m2_mxfp8>(float value)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return signedNaNMaskOcpE5m2Mxfp8(sign);
    }

    uint8_t res = convertToType<uint8_t, ocp_e5m2_mxfp8>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e5m2_mxfp8>(tScale, tData, 0, 0);
    if(std::abs(resVal) > ocp_e5m2_mxfp8::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? ocp_e5m2_mxfp8::negativeInfMask : ocp_e5m2_mxfp8::positiveInfMask;

    if(std::abs(resVal) < ocp_e5m2_mxfp8::dataMinSubNormalNumber)
        return sign ? ocp_e5m2_mxfp8::negativeZeroMask : ocp_e5m2_mxfp8::positiveZeroMask;

    return res;
}

template <>
inline uint64_t satConvertToTypeSR<ocp_e5m2_mxfp8>(float value, uint seed)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;
    if(std::isnan(value))
        return signedNaNMaskOcpE5m2Mxfp8(sign);
    else if(value > ocp_e5m2_mxfp8::dataMaxRoundedRange)
        return ocp_e5m2_mxfp8::dataMaxPositiveNormalMask;
    else if(value < -ocp_e5m2_mxfp8::dataMaxRoundedRange)
        return ocp_e5m2_mxfp8::dataMaxNegativeNormalMask;

    uint8_t res = convertToTypeSR<uint8_t, ocp_e5m2_mxfp8>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e5m2_mxfp8>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e5m2_mxfp8::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? ocp_e5m2_mxfp8::dataMaxNegativeNormalMask
                         : ocp_e5m2_mxfp8::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e5m2_mxfp8::dataMinSubNormalNumber)
        return sign ? ocp_e5m2_mxfp8::negativeZeroMask : ocp_e5m2_mxfp8::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToTypeSR<ocp_e5m2_mxfp8>(float value, uint seed)
{
    cvt t;

    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return signedNaNMaskOcpE5m2Mxfp8(sign);
    else if(value > ocp_e5m2_mxfp8::dataMaxRoundedRange)
        return ocp_e5m2_mxfp8::positiveInfMask;
    else if(value < -ocp_e5m2_mxfp8::dataMaxRoundedRange)
        return ocp_e5m2_mxfp8::negativeInfMask;

    uint8_t res = convertToTypeSR<uint8_t, ocp_e5m2_mxfp8>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e5m2_mxfp8>(tScale, tData, 0, 0);
    if(std::abs(resVal) > ocp_e5m2_mxfp8::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? ocp_e5m2_mxfp8::negativeInfMask : ocp_e5m2_mxfp8::positiveInfMask;

    if(std::abs(resVal) < ocp_e5m2_mxfp8::dataMinSubNormalNumber)
        return sign ? ocp_e5m2_mxfp8::negativeZeroMask : ocp_e5m2_mxfp8::positiveZeroMask;

    return res;
}
