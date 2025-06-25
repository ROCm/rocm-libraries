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
#include "fp16.hpp"
#include "packing.hpp"

//return true iff XN = NAN
template <>
inline bool isNaN<fp16>(uint8_t const* scaleBytes [[maybe_unused]],
                        uint8_t const* dataBytes,
                        index_t         scaleIndex [[maybe_unused]],
                        index_t         dataIndex)
{
    uint16_t data = getDataFP16(dataBytes, dataIndex);

    uint16_t mantissa = data & fp16::dataMantissaMask;
    uint16_t exponent = (data >> fp16::dataInfo.mantissaBits) & 0b11111;

    return (exponent == 0b11111 && mantissa != 0b0);
}

// return true iff XN = 0
template <>
inline bool isZero<fp16>(uint8_t const* scaleBytes,
                         uint8_t const* dataBytes,
                         index_t         scaleIndex,
                         index_t         dataIndex)
{

    if(isNaN<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    uint16_t data = getDataFP16(dataBytes, dataIndex);

    return (data & fp16::setSignMask) == fp16::positiveZeroMask;
}

template <>
inline bool isInf<fp16>(uint8_t const* scaleBytes [[maybe_unused]],
                        uint8_t const* dataBytes,
                        index_t         scaleIndex [[maybe_unused]],
                        index_t         dataIndex)
{
    uint16_t data = getDataFP16(dataBytes, dataIndex);
    return (data & fp16::setSignMask) == fp16::dataInfMask;
}

// // return the double value of XN
template <>
inline double toDouble<fp16>(uint8_t const* scaleBytes,
                             uint8_t const* dataBytes,
                             index_t         scaleIndex,
                             index_t         dataIndex)
{

    if(isNaN<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint16_t data = getDataFP16(dataBytes, dataIndex);

    uint16_t signBit = data >> (fp16::dataInfo.exponentBits + fp16::dataInfo.mantissaBits);

    if(isInf<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::infinity() * (signBit ? -1 : 1);

    double d_s = std::pow(-1, static_cast<double>(signBit));

    double d_e;
    if(isSubNormal<uint16_t>(data, fp16::dataInfo.mantissaBits, fp16::dataInfo.exponentBits))
        d_e = std::pow(2, 1 - static_cast<int>(fp16::dataInfo.bias));
    else
        d_e = std::pow(2,
                       getExponentValue<uint16_t>(
                           data, fp16::dataInfo.mantissaBits, fp16::dataInfo.exponentBits)
                           - static_cast<int>(fp16::dataInfo.bias));

    double d_m = getMantissaValue<uint16_t>(
        data, fp16::dataInfo.mantissaBits, fp16::dataInfo.exponentBits);

    return d_s * d_e * d_m;
}

template <>
inline float toFloat<fp16>(uint8_t const* scaleBytes,
                           uint8_t const* dataBytes,
                           index_t         scaleIndex,
                           index_t         dataIndex)
{
    if(isNaN<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint16_t data = getDataFP16(dataBytes, dataIndex);

    uint16_t signBit = data >> (fp16::dataInfo.exponentBits + fp16::dataInfo.mantissaBits);

    if(isInf<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::infinity() * (signBit ? -1 : 1);

    float d_s = std::pow(-1, static_cast<float>(signBit));

    float d_e;
    if(isSubNormal<uint16_t>(data, fp16::dataInfo.mantissaBits, fp16::dataInfo.exponentBits))
        d_e = std::pow(2, 1 - static_cast<int>(fp16::dataInfo.bias));
    else
        d_e = std::pow(2,
                       getExponentValue<uint16_t>(
                           data, fp16::dataInfo.mantissaBits, fp16::dataInfo.exponentBits)
                           - static_cast<int>(fp16::dataInfo.bias));
    float d_m = getMantissaValue<uint16_t>(
        data, fp16::dataInfo.mantissaBits, fp16::dataInfo.exponentBits);

    return d_s * d_e * d_m;
}

//return true iff XN = NAN
template <>
inline bool isZeroPacked<fp16>(uint8_t const* scaleBytes,
                               uint8_t const* dataBytes,
                               index_t         scaleIndex,
                               index_t         dataIndex)
{
    return isZero<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isNaNPacked<fp16>(uint8_t const* scaleBytes,
                              uint8_t const* dataBytes,
                              index_t         scaleIndex,
                              index_t         dataIndex)
{
    return isNaN<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isInfPacked<fp16>(uint8_t const* scaleBytes,
                              uint8_t const* dataBytes,
                              index_t         scaleIndex,
                              index_t         dataIndex)
{
    return isInf<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//return true iff XN < val
template <>
inline bool isLessPacked<fp16>(double         val,
                               uint8_t const* scaleBytes,
                               uint8_t const* dataBytes,
                               index_t         scaleIndex,
                               index_t         dataIndex)
{
    return isLess<fp16>(val, scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//return true iff XN > val
template <>
inline bool isGreaterPacked<fp16>(double         val,
                                  uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes,
                                  index_t         scaleIndex,
                                  index_t         dataIndex)
{
    return isGreater<fp16>(val, scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isSubnorm<fp16>(uint8_t const* dataBytes, index_t dataIndex)
{
    uint16_t data = getDataFP16(dataBytes, dataIndex);
    return isSubNormal<uint16_t>(data, fp16::dataInfo.mantissaBits, fp16::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<fp16>(uint8_t const* dataBytes, index_t dataIndex)
{
    return isSubnorm<fp16>(dataBytes, dataIndex);
}

// return the normal double value of XN
template <>
inline double toDoublePacked<fp16>(uint8_t const* scaleBytes,
                                   uint8_t const* dataBytes,
                                   index_t         scaleIndex,
                                   index_t         dataIndex)
{
    return toDouble<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline float toFloatPacked<fp16>(uint8_t const* scaleBytes,
                                 uint8_t const* dataBytes,
                                 index_t         scaleIndex,
                                 index_t         dataIndex)
{
    return toFloat<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//set XN = 1
template <>
inline void setOne<fp16>(uint8_t* scaleBytes [[maybe_unused]],
                         uint8_t* dataBytes,
                         index_t   scaleIndex [[maybe_unused]],
                         index_t   dataIndex,
                         bool     subNormal [[maybe_unused]])
{
    setDataFP16(dataBytes, dataIndex, fp16::oneMask);
}

//set X = 0
template <>
inline void setZero<fp16>(uint8_t* scaleBytes [[maybe_unused]],
                          uint8_t* dataBytes,
                          index_t   scaleIndex [[maybe_unused]],
                          index_t   dataIndex)
{
    setDataFP16(dataBytes, dataIndex, fp16::positiveZeroMask);
}

template <>
inline void setNaN<fp16>(uint8_t* scaleBytes [[maybe_unused]],
                         uint8_t* dataBytes,
                         index_t   scaleIndex [[maybe_unused]],
                         index_t   dataIndex)
{
    setDataFP16(dataBytes, dataIndex, fp16::dataNanMask);
}

template <>
inline void setInf<fp16>(uint8_t* scaleBytes [[maybe_unused]],
                         uint8_t* dataBytes,
                         index_t   scaleIndex [[maybe_unused]],
                         index_t   dataIndex)
{

    setDataFP16(dataBytes, dataIndex, fp16::dataInfMask);
}

template <>
inline void setDataMax<fp16>(uint8_t* dataBytes, index_t dataIndex, bool subNormal, bool positive)
{
    if(subNormal)
        setDataFP16(dataBytes,
                    dataIndex,
                    positive ? fp16::dataMaxPositiveSubNormalMask
                             : fp16::dataMaxNegativeSubNormalMask);

    else
        setDataFP16(dataBytes,
                    dataIndex,
                    positive ? fp16::dataMaxPositiveNormalMask : fp16::dataMaxNegativeNormalMask);
}

template <>
inline void setOnePacked<fp16>(uint8_t* scaleBytes,
                               uint8_t* dataBytes,
                               index_t   scaleIndex,
                               index_t   dataIndex,
                               bool     subNormal [[maybe_unused]])
{
    setOne<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//set XN = 0, scale X will not be changed
template <>
inline void setZeroPacked<fp16>(uint8_t* scaleBytes,
                                uint8_t* dataBytes,
                                index_t   scaleIndex,
                                index_t   dataIndex)
{
    setZero<fp16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline void setNaNPacked<fp16>(uint8_t* scaleBytes [[maybe_unused]],
                               uint8_t* dataBytes,
                               index_t   scaleIndex [[maybe_unused]],
                               index_t   dataIndex)
{
    setDataFP16(dataBytes, dataIndex, fp16::dataNanMask);
}

template <>
inline void setInfPacked<fp16>(uint8_t* scaleBytes [[maybe_unused]],
                               uint8_t* dataBytes,
                               index_t   scaleIndex [[maybe_unused]],
                               index_t   dataIndex)
{
    setDataFP16(dataBytes, dataIndex, fp16::dataInfMask);
}

template <>
inline void
    setDataMaxPacked<fp16>(uint8_t* dataBytes, index_t dataIndex, bool subNormal, bool positive)
{
    setDataMax<fp16>(dataBytes, dataIndex, subNormal, positive);
}

template <>
inline uint64_t satConvertToType<fp16>(float value)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return sign << 15 | fp16::dataNanMask;
    }

    uint16_t res = convertToType<uint16_t, fp16>(value);

    uint8_t tData[] = {0b0, 0b0};

    setDataFP16(tData, 0, res);

    float resVal = toFloat<fp16>(tData, tData, 0, 0);

    if(std::abs(resVal) > fp16::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? fp16::dataMaxNegativeNormalMask : fp16::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < fp16::dataMinSubNormalNumber)
        return value < 0 ? fp16::negativeZeroMask : fp16::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToType<fp16>(float value)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return sign << 15 | fp16::dataNanMask;
    }
    uint16_t res = convertToType<uint16_t, fp16>(value);

    uint8_t tData[] = {0b0, 0b0};

    setDataFP16(tData, 0, res);

    float resVal = toFloat<fp16>(tData, tData, 0, 0);

    if(std::abs(resVal) > fp16::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? fp16::dataNegativeInfMask : fp16::dataInfMask;

    if(std::abs(resVal) < fp16::dataMinSubNormalNumber)
        return value < 0 ? fp16::negativeZeroMask : fp16::positiveZeroMask;

    return res;
}

template <>
inline uint64_t satConvertToTypeSR<fp16>(float value, uint seed)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 15 | fp16::dataNanMask;
    else if(std::isinf(value))
        return value > 0 ? fp16::dataMaxPositiveNormalMask : fp16::dataMaxNegativeNormalMask;

    uint16_t res = convertToTypeSR<uint16_t, fp16>(value, seed);

    uint8_t tData[] = {0b0, 0b0};

    setDataFP16(tData, 0, res);

    float resVal = toFloat<fp16>(tData, tData, 0, 0);

    if(std::abs(resVal) > fp16::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? fp16::dataMaxNegativeNormalMask : fp16::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < fp16::dataMinSubNormalNumber)
        return value < 0 ? fp16::negativeZeroMask : fp16::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToTypeSR<fp16>(float value, uint seed)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign << 15 | fp16::dataNanMask;

    uint16_t res = convertToTypeSR<uint16_t, fp16>(value, seed);

    uint8_t tData[] = {0b0, 0b0};

    setDataFP16(tData, 0, res);

    float resVal = toFloat<fp16>(tData, tData, 0, 0);

    if(std::abs(resVal) > fp16::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? fp16::dataNegativeInfMask : fp16::dataInfMask;

    if(std::abs(resVal) < fp16::dataMinSubNormalNumber)
        return value < 0 ? fp16::negativeZeroMask : fp16::positiveZeroMask;

    return res;
}
