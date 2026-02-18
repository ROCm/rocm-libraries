/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include "dataTypeInfo.hpp"

//return true iff XN = NAN
template <>
inline bool isNaN<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes,
                                  index_t         scaleIndex,
                                  index_t         dataIndex)
{
    uint8_t data  = *(dataBytes + dataIndex);
    uint8_t scale = *(scaleBytes + scaleIndex);

    if(scale == Constants::E8M0_NAN)
        return true;

    // set sign bit to 0
    data &= (~ocp_e4m3_mxfp8::signBitMask);

    auto& nanValues{ocp_e4m3_mxfp8::dataNaNMasks};
    return std::count(nanValues.begin(), nanValues.end(), data) > 0;
}

template <>
inline bool isNaNPacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        index_t         scaleIndex,
                                        index_t         dataIndex)
{
    return isNaN<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

// return true iff XN = 0
template <>
inline bool isZero<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                   uint8_t const* dataBytes,
                                   index_t         scaleIndex,
                                   index_t         dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    // keep bits 0-7 same, set sign bit to 0
    // no need to check for scale as it doesn't have a zero representation
    uint8_t data = *(dataBytes + dataIndex) & (~ocp_e4m3_mxfp8::signBitMask);

    return data == ocp_e4m3_mxfp8::positiveZeroMask;
}

template <>
inline bool isZeroPacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                         uint8_t const* dataBytes,
                                         index_t         scaleIndex,
                                         index_t         dataIndex)
{
    return isZero<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

// no infinity representation in ocp_e4m3_mxfp8 will always return false
template <>
inline bool isInf<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes [[maybe_unused]],
                                  uint8_t const* dataBytes [[maybe_unused]],
                                  index_t         scaleIndex [[maybe_unused]],
                                  index_t         dataIndex [[maybe_unused]])
{
    return false;
}

template <>
inline bool isInfPacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        index_t         scaleIndex,
                                        index_t         dataIndex)
{
    return isInf<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline double toDouble<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                       uint8_t const* dataBytes,
                                       index_t         scaleIndex,
                                       index_t         dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e4m3_mxfp8::scaleInfo.mantissaBits,
                                             ocp_e4m3_mxfp8::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E4M3_MXFP8_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline double toDoublePacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             index_t         scaleIndex,
                                             index_t         dataIndex)
{
    return toDouble<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline float toFloat<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                     uint8_t const* dataBytes,
                                     index_t         scaleIndex,
                                     index_t         dataIndex)
{
    if(isNaN<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data     = *(dataBytes + dataIndex);
    int     scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e4m3_mxfp8::scaleInfo.mantissaBits,
                                             ocp_e4m3_mxfp8::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E4M3_MXFP8_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline float toFloatPacked<ocp_e4m3_mxfp8>(uint8_t const* scaleBytes,
                                           uint8_t const* dataBytes,
                                           index_t         scaleIndex,
                                           index_t         dataIndex)
{
    return toFloat<ocp_e4m3_mxfp8>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isSubnorm<ocp_e4m3_mxfp8>(uint8_t const* dataBytes, index_t dataIndex)
{
    uint8_t data = *(dataBytes + dataIndex);
    return isSubNormal<uint16_t>(
        data, ocp_e4m3_mxfp8::dataInfo.mantissaBits, ocp_e4m3_mxfp8::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<ocp_e4m3_mxfp8>(uint8_t const* dataBytes, index_t dataIndex)
{
    return isSubnorm<ocp_e4m3_mxfp8>(dataBytes, dataIndex);
}

template <>
inline void setOne<ocp_e4m3_mxfp8>(
    uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex, bool subNormal)
{
    //if its sub normal, 0b00000010 * E8M0_135 will equal to 1
    *(scaleBytes + scaleIndex) = subNormal ? Constants::E8M0_135 : Constants::E8M0_1;
    *(dataBytes + dataIndex)
        = subNormal ? ocp_e4m3_mxfp8::dataSubNormalOneMask : ocp_e4m3_mxfp8::oneMask;
}

//set XN = 0, scale X will not be changed
template <>
inline void setZero<ocp_e4m3_mxfp8>(uint8_t* scaleBytes [[maybe_unused]],
                                    uint8_t* dataBytes,
                                    index_t   scaleIndex [[maybe_unused]],
                                    index_t   dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e4m3_mxfp8::positiveZeroMask;
}

//set XN = NAN, scale X will not be changed
template <>
inline void setNaN<ocp_e4m3_mxfp8>(uint8_t* scaleBytes [[maybe_unused]],
                                   uint8_t* dataBytes,
                                   index_t   scaleIndex [[maybe_unused]],
                                   index_t   dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e4m3_mxfp8::dataNaNMasks[0];
}

//ocp_e4m3_mxfp8 does not have an infinity representation, method will just return
template <>
inline void setInf<ocp_e4m3_mxfp8>(uint8_t* scaleBytes [[maybe_unused]],
                                   uint8_t* dataBytes [[maybe_unused]],
                                   index_t   scaleIndex [[maybe_unused]],
                                   index_t   dataIndex [[maybe_unused]])
{
    return;
}

template <>
inline void
    setDataMax<ocp_e4m3_mxfp8>(uint8_t* dataBytes, index_t dataIndex, bool subNormal, bool positive)
{
    if(subNormal)
        *(dataBytes + dataIndex) = positive ? ocp_e4m3_mxfp8::dataMaxPositiveSubNormalMask
                                            : ocp_e4m3_mxfp8::dataMaxNegativeSubNormalMask;
    else
        *(dataBytes + dataIndex) = positive ? ocp_e4m3_mxfp8::dataMaxPositiveNormalMask
                                            : ocp_e4m3_mxfp8::dataMaxNegativeNormalMask;
}

template <>
inline uint64_t satConvertToType<ocp_e4m3_mxfp8>(float value)
{

    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return sign << 15 | ocp_e4m3_mxfp8::dataNaNMasks[0];
    }

    uint8_t res = convertToType<uint8_t, ocp_e4m3_mxfp8>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e4m3_mxfp8>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8::dataMaxNormalNumber
       || std::isnan(resVal)) // covers inf case too
        return value < 0 ? ocp_e4m3_mxfp8::dataMaxNegativeNormalMask
                         : ocp_e4m3_mxfp8::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e4m3_mxfp8::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8::negativeZeroMask : ocp_e4m3_mxfp8::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToType<ocp_e4m3_mxfp8>(float value)
{

    uint8_t res = convertToType<uint8_t, ocp_e4m3_mxfp8>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e4m3_mxfp8>(tScale, tData, 0, 0);

    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    //std::abs(value) > dataMaxNornal covers inf case as well
    if(std::abs(resVal) > ocp_e4m3_mxfp8::dataMaxNormalNumber || std::isnan(value))
        return sign << 15 | ocp_e4m3_mxfp8::dataNaNMasks[0];

    if(std::abs(resVal) < ocp_e4m3_mxfp8::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8::negativeZeroMask : ocp_e4m3_mxfp8::positiveZeroMask;

    return res;
}

template <>
inline uint64_t satConvertToTypeSR<ocp_e4m3_mxfp8>(float value, uint seed)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 7 | ocp_e4m3_mxfp8::dataNaNMasks[0];
    else if(value > ocp_e4m3_mxfp8::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8::dataMaxPositiveNormalMask;
    else if(value < -ocp_e4m3_mxfp8::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8::dataMaxNegativeNormalMask;

    uint8_t res = convertToTypeSR<uint8_t, ocp_e4m3_mxfp8>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e4m3_mxfp8>(tScale, tData, 0, 0);

    if(std::abs(resVal) > ocp_e4m3_mxfp8::dataMaxNormalNumber
       || std::isnan(resVal)) // covers inf case too
        return value < 0 ? ocp_e4m3_mxfp8::dataMaxNegativeNormalMask
                         : ocp_e4m3_mxfp8::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < ocp_e4m3_mxfp8::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8::negativeZeroMask : ocp_e4m3_mxfp8::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToTypeSR<ocp_e4m3_mxfp8>(float value, uint seed)
{

    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 7 | ocp_e4m3_mxfp8::dataNaNMasks[0];
    else if(value > ocp_e4m3_mxfp8::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8::dataNaNMasks[0];
    else if(value < -ocp_e4m3_mxfp8::dataMaxRoundedRange)
        return ocp_e4m3_mxfp8::dataNaNMasks[0];

    uint8_t res = convertToTypeSR<uint8_t, ocp_e4m3_mxfp8>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    float resVal = toFloat<ocp_e4m3_mxfp8>(tScale, tData, 0, 0);

    //std::abs(value) > dataMaxNornal covers inf case as well
    if(std::abs(resVal) > ocp_e4m3_mxfp8::dataMaxNormalNumber)
        return sign << 7 | ocp_e4m3_mxfp8::dataNaNMasks[0];

    if(std::abs(resVal) < ocp_e4m3_mxfp8::dataMinSubnormalNumber)
        return value < 0 ? ocp_e4m3_mxfp8::negativeZeroMask : ocp_e4m3_mxfp8::positiveZeroMask;

    return res;
}
