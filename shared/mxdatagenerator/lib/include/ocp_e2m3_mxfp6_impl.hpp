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
#include "fp6.hpp"

template <>
inline uint8_t oneMask<ocp_e2m3_mxfp6>()
{
    return 0b001000;
}
template <>
inline uint8_t setSignMaskPositive<ocp_e2m3_mxfp6>()
{
    return 0b011111;
}
template <>
inline uint8_t dataMaxPositiveNormalMask<ocp_e2m3_mxfp6>()
{
    return 0b011111;
}
template <>
inline uint8_t dataMaxNegativeNormalMask<ocp_e2m3_mxfp6>()
{
    return 0b111111;
}
template <>
inline uint8_t dataMaxPositiveSubNormalMask<ocp_e2m3_mxfp6>()
{
    return 0b000111;
}
template <>
inline uint8_t dataMaxNegativeSubNormalMask<ocp_e2m3_mxfp6>()
{
    return 0b100111;
}
template <>
inline uint8_t dataSubNormalOneMask<ocp_e2m3_mxfp6>()
{
    return 0b000100;
}
template <>
inline float dataMaxNormalNumber<ocp_e2m3_mxfp6>()
{
    return 7.5;
}
template <>
inline float dataMinSubNormalNumber<ocp_e2m3_mxfp6>()
{
    return 0.125;
}
template <>
inline uint8_t positiveZeroMask<ocp_e2m3_mxfp6>()
{
    return 0b000000;
}
template <>
inline uint8_t negativeZeroMask<ocp_e2m3_mxfp6>()
{
    return 0b100000;
}
template <>
inline uint8_t scaleSubNormalOne<ocp_e2m3_mxfp6>()
{
    return Constants::E8M0_2;
}
template <>
inline uint8_t scaleOne<ocp_e2m3_mxfp6>()
{
    return Constants::E8M0_1;
}

template <>
inline bool isNaN<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes [[maybe_unused]],
                                  index_t         scaleIndex,
                                  index_t         dataIndex [[maybe_unused]])
{
    // no need to check for NAN in dataBytes since there's no NAN representation
    return *(scaleBytes + scaleIndex) == Constants::E8M0_NAN;
}

template <>
inline bool isNaNPacked<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        index_t         scaleIndex,
                                        index_t         dataIndex)
{
    // since the scale is e8m0 and is 8 bits its packed is the same as unpacked
    // as well as there are no NAN for ocp_e3m2_mxfp6
    return isNaN<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isInf<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes [[maybe_unused]],
                                  uint8_t const* dataBytes [[maybe_unused]],
                                  index_t         scaleIndex [[maybe_unused]],
                                  index_t         dataIndex [[maybe_unused]])
{
    // no infinity representation in ocp_e3m2_mxfp6 will always return false
    return false;
}

// no infinity representation in ocp_e3m2_mxfp6 will always return false
template <>
inline bool isInfPacked<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes [[maybe_unused]],
                                        uint8_t const* dataBytes [[maybe_unused]],
                                        index_t         scaleIndex [[maybe_unused]],
                                        index_t         dataIndex [[maybe_unused]])
{
    return false;
}

template <>
inline bool isZero<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes,
                                   uint8_t const* dataBytes,
                                   index_t         scaleIndex,
                                   index_t         dataIndex)
{
    if(isNaN<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;
    // no need to check for exponent, since e8m0 does not have zero representation
    return (*(dataBytes + dataIndex) & setSignMaskPositive<ocp_e2m3_mxfp6>())
           == positiveZeroMask<ocp_e2m3_mxfp6>();
}

template <>
inline bool isZeroPacked<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes,
                                         uint8_t const* dataBytes,
                                         index_t         scaleIndex,
                                         index_t         dataIndex)
{
    if(isNaNPacked<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    return (getDataFromPackedF6(dataBytes, dataIndex) & setSignMaskPositive<ocp_e2m3_mxfp6>())
           == positiveZeroMask<ocp_e2m3_mxfp6>();
}

template <>
inline bool isSubnorm<ocp_e2m3_mxfp6>(uint8_t const* dataBytes, index_t dataIndex)
{
    // XXX 6 bit
    uint8_t data = *(dataBytes + dataIndex) & 0b00111111;
    return isSubNormal<uint16_t>(
        data, ocp_e2m3_mxfp6::dataInfo.mantissaBits, ocp_e2m3_mxfp6::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<ocp_e2m3_mxfp6>(uint8_t const* dataBytes, index_t dataIndex)
{
    uint8_t data = getDataFromPackedF6(dataBytes, dataIndex);
    return isSubNormal<uint16_t>(
        data, ocp_e2m3_mxfp6::dataInfo.mantissaBits, ocp_e2m3_mxfp6::dataInfo.exponentBits);
}

template <>
inline double toDouble<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes,
                                       uint8_t const* dataBytes,
                                       index_t         scaleIndex,
                                       index_t         dataIndex)
{
    if(isNaN<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0;

    uint8_t data = *(dataBytes + dataIndex) & 0b00111111;

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e2m3_mxfp6::scaleInfo.mantissaBits,
                                             ocp_e2m3_mxfp6::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E2M3_MXFP6_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline double toDoublePacked<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             index_t         scaleIndex,
                                             index_t         dataIndex)
{
    if(isNaNPacked<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZeroPacked<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = getDataFromPackedF6(dataBytes, dataIndex);

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e2m3_mxfp6::scaleInfo.mantissaBits,
                                             ocp_e2m3_mxfp6::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E2M3_MXFP6_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline float toFloat<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes,
                                     uint8_t const* dataBytes,
                                     index_t         scaleIndex,
                                     index_t         dataIndex)
{
    if(isNaN<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = *(dataBytes + dataIndex) & 0b00111111;

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e2m3_mxfp6::scaleInfo.mantissaBits,
                                             ocp_e2m3_mxfp6::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E2M3_MXFP6_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline float toFloatPacked<ocp_e2m3_mxfp6>(uint8_t const* scaleBytes,
                                           uint8_t const* dataBytes,
                                           index_t         scaleIndex,
                                           index_t         dataIndex)
{
    if(isNaNPacked<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZeroPacked<ocp_e2m3_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = getDataFromPackedF6(dataBytes, dataIndex);

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e2m3_mxfp6::scaleInfo.mantissaBits,
                                             ocp_e2m3_mxfp6::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E2M3_MXFP6_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline void setOne<ocp_e2m3_mxfp6>(
    uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex, bool subNormal)
{
    *(scaleBytes + scaleIndex)
        = subNormal ? scaleSubNormalOne<ocp_e2m3_mxfp6>() : scaleOne<ocp_e2m3_mxfp6>();
    *(dataBytes + dataIndex)
        = subNormal ? dataSubNormalOneMask<ocp_e2m3_mxfp6>() : oneMask<ocp_e2m3_mxfp6>();
}

//set XN = 0, scale X will not be changed
template <>
inline void setZero<ocp_e2m3_mxfp6>(uint8_t* scaleBytes [[maybe_unused]],
                                    uint8_t* dataBytes,
                                    index_t   scaleIndex [[maybe_unused]],
                                    index_t   dataIndex)
{
    *(dataBytes + dataIndex) = positiveZeroMask<ocp_e2m3_mxfp6>();
}

//ocp_e3m2_mxfp6 does not have NAN representation, method will just return
template <>
inline void setNaN<ocp_e2m3_mxfp6>(uint8_t* scaleBytes,
                                   uint8_t* dataBytes [[maybe_unused]],
                                   index_t   scaleIndex,
                                   index_t   dataIndex [[maybe_unused]])
{
    *(scaleBytes + scaleIndex) = Constants::E8M0_NAN;
}

//ocp_e3m2_mxfp6 does not have an infinity representation, method will just return
template <>
inline void setInf<ocp_e2m3_mxfp6>(uint8_t* scaleBytes [[maybe_unused]],
                                   uint8_t* dataBytes [[maybe_unused]],
                                   index_t   scaleIndex [[maybe_unused]],
                                   index_t   dataIndex [[maybe_unused]])
{
    return;
}

template <>
inline void
    setDataMax<ocp_e2m3_mxfp6>(uint8_t* dataBytes, index_t dataIndex, bool subNormal, bool positive)
{
    if(subNormal)
        *(dataBytes + dataIndex) = positive ? dataMaxPositiveSubNormalMask<ocp_e2m3_mxfp6>()
                                            : dataMaxNegativeSubNormalMask<ocp_e2m3_mxfp6>();
    else
        *(dataBytes + dataIndex) = positive ? dataMaxPositiveNormalMask<ocp_e2m3_mxfp6>()
                                            : dataMaxNegativeNormalMask<ocp_e2m3_mxfp6>();
}

template <>
inline void setOnePacked<ocp_e2m3_mxfp6>(
    uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex, bool subNormal)
{
    *(scaleBytes + scaleIndex) = subNormal ? Constants::E8M0_3 : Constants::E8M0_1;
    uint8_t mask = subNormal ? dataSubNormalOneMask<ocp_e2m3_mxfp6>() : oneMask<ocp_e2m3_mxfp6>();

    setDataPackedF6(dataBytes, dataIndex, mask);
}

//set XN = 0, scale X will not be changed
template <>
inline void setZeroPacked<ocp_e2m3_mxfp6>(uint8_t* scaleBytes [[maybe_unused]],
                                          uint8_t* dataBytes,
                                          index_t   scaleIndex [[maybe_unused]],
                                          index_t   dataIndex)
{
    setDataPackedF6(dataBytes, dataIndex, positiveZeroMask<ocp_e2m3_mxfp6>());
}

template <>
inline void setNaNPacked<ocp_e2m3_mxfp6>(uint8_t* scaleBytes,
                                         uint8_t* dataBytes [[maybe_unused]],
                                         index_t   scaleIndex,
                                         index_t   dataIndex [[maybe_unused]])
{
    *(scaleBytes + scaleIndex) = Constants::E8M0_NAN;
}

//ocp_e3m2_mxfp6 does not have an infinity representation, method will just return
template <>
inline void setInfPacked<ocp_e2m3_mxfp6>(uint8_t* scaleBytes [[maybe_unused]],
                                         uint8_t* dataBytes [[maybe_unused]],
                                         index_t   scaleIndex [[maybe_unused]],
                                         index_t   dataIndex [[maybe_unused]])
{
    return;
}

template <>
inline void setDataMaxPacked<ocp_e2m3_mxfp6>(uint8_t* dataBytes,
                                             index_t   dataIndex,
                                             bool     subNormal,
                                             bool     positive)
{
    uint8_t mask;
    if(subNormal)
        mask = positive ? dataMaxPositiveSubNormalMask<ocp_e2m3_mxfp6>()
                        : dataMaxNegativeSubNormalMask<ocp_e2m3_mxfp6>();
    else
        mask = positive ? dataMaxPositiveNormalMask<ocp_e2m3_mxfp6>()
                        : dataMaxNegativeNormalMask<ocp_e2m3_mxfp6>();

    setDataPackedF6(dataBytes, dataIndex, mask);
}

template <>
inline uint64_t satConvertToType<ocp_e2m3_mxfp6>(float value)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return (sign << 5) | dataMaxPositiveNormalMask<ocp_e2m3_mxfp6>();
    }

    if(std::abs(value) > dataMaxNormalNumber<ocp_e2m3_mxfp6>()) //covers inf case as well
        return value < 0 ? dataMaxNegativeNormalMask<ocp_e2m3_mxfp6>()
                         : dataMaxPositiveNormalMask<ocp_e2m3_mxfp6>();

    uint8_t res = convertToType<uint8_t, ocp_e2m3_mxfp6>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    if(std::abs(toFloat<ocp_e2m3_mxfp6>(tScale, tData, 0, 0))
       < dataMinSubNormalNumber<ocp_e2m3_mxfp6>())
        return value < 0 ? negativeZeroMask<ocp_e2m3_mxfp6>() : positiveZeroMask<ocp_e2m3_mxfp6>();

    return res;
}

template <>
inline uint64_t nonSatConvertToType<ocp_e2m3_mxfp6>(float value [[maybe_unused]])
{
    return 0b0;
}

template <>
inline uint64_t satConvertToTypeSR<ocp_e2m3_mxfp6>(float value, uint seed)
{
    cvt t;
    t.num = value;

    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return (sign << 5) | ocp_e2m3_mxfp6::dataMaxPositiveNormalMask;
    }

    if(std::abs(value) > ocp_e2m3_mxfp6::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? ocp_e2m3_mxfp6::dataMaxNegativeNormalMask
                         : ocp_e2m3_mxfp6::dataMaxPositiveNormalMask;

    uint8_t res = convertToTypeSR<uint8_t, ocp_e2m3_mxfp6>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    if(std::abs(toFloat<ocp_e2m3_mxfp6>(tScale, tData, 0, 0))
       < ocp_e2m3_mxfp6::dataMinSubNormalNumber)
        return value < 0 ? ocp_e2m3_mxfp6::negativeZeroMask : ocp_e2m3_mxfp6::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToTypeSR<ocp_e2m3_mxfp6>(float value [[maybe_unused]],
                                                      uint  seed [[maybe_unused]])
{
    return 0b0;
}
