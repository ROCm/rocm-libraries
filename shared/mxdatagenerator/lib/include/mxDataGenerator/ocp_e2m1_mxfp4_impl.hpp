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

/**
 * Get the data representation from a packed
 * dataByte buffer
 *
 * @param dataBytes
 *      The packed dataByte buffer to read from
 *
 * @param index
 *      The index to the bit representation
 *      to read
 */
static uint8_t getDataFromPackedF4(uint8_t const* dataBytes, size_t index)
{
    size_t cellIndex = index / 2;

    // odd index -> first half of cell
    if(index % 2)
        return (*(dataBytes + cellIndex) & 0b11110000) >> 4;
    // even index -> second half of cell
    else
        return *(dataBytes + cellIndex) & 0b00001111;
}

/**
 *  Set the bit representation at the
 *  specified index to the mask provided
 *
 *  @param dataBytes
 *      The packed dataByte buffer to write to
 *
 *  @param index
 *      The index to the bit representation
 *      to write to
 *
 *  @param mask
 *      The mask to set the bit representation
 *      to
 */
static void setDataPackedF4(uint8_t* dataBytes, size_t index, uint8_t mask)
{
    size_t cellIndex = index / 2;

    // odd index -> first half of cell
    if(index % 2)
    {
        *(dataBytes + cellIndex) &= 0b00001111; //clear first half
        *(dataBytes + cellIndex) |= (mask << 4); // set first half
    }
    else
    {
        *(dataBytes + cellIndex) &= 0b11110000; //clear second half
        *(dataBytes + cellIndex) |= mask; // set second half
    }
}

template <>
inline bool isNaN<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes [[maybe_unused]],
                                  index_t         scaleIndex,
                                  index_t         dataIndex [[maybe_unused]])
{
    // no need to check for data as it does not have representation
    uint8_t scale = *(scaleBytes + scaleIndex);
    return scale == Constants::E8M0_NAN;
}

template <>
inline bool isZero<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes,
                                   uint8_t const* dataBytes,
                                   index_t         scaleIndex,
                                   index_t         dataIndex)
{
    if(isNaN<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    // no need to check for scale as it does not have a 0 representation
    uint8_t data = (*(dataBytes + dataIndex) & 0b00001111) & ocp_e2m1_mxfp4::setSignMask;

    return data == 0b0;
}

template <>
inline double toDouble<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes,
                                       uint8_t const* dataBytes,
                                       index_t         scaleIndex,
                                       index_t         dataIndex)
{
    if(isNaN<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = *(dataBytes + dataIndex) & 0b00001111;

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e2m1_mxfp4::scaleInfo.mantissaBits,
                                             ocp_e2m1_mxfp4::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E2M1_MXFP4_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline float toFloat<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes,
                                     uint8_t const* dataBytes,
                                     index_t         scaleIndex,
                                     index_t         dataIndex)
{
    if(isNaN<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = *(dataBytes + dataIndex) & 0b00001111;

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e2m1_mxfp4::scaleInfo.mantissaBits,
                                             ocp_e2m1_mxfp4::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E2M1_MXFP4_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline bool isNaNPacked<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes [[maybe_unused]],
                                        index_t         scaleIndex,
                                        index_t         dataIndex [[maybe_unused]])
{
    // no need to check for data as it does not have representation
    uint8_t scale = *(scaleBytes + scaleIndex);
    return scale == Constants::E8M0_NAN;
}

template <>
inline bool isInfPacked<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes [[maybe_unused]],
                                        uint8_t const* dataBytes [[maybe_unused]],
                                        index_t         scaleIndex [[maybe_unused]],
                                        index_t         dataIndex [[maybe_unused]])
{
    // no infinity representation in ocp_e2m1_mxfp4 will always return false
    return false;
}

// return true iff XN = 0
template <>
inline bool isZeroPacked<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes,
                                         uint8_t const* dataBytes,
                                         index_t         scaleIndex,
                                         index_t         dataIndex)
{
    if(isNaNPacked<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;
    // no need to check for sign as it does not have a 0 representation
    uint8_t data = getDataFromPackedF4(dataBytes, dataIndex) & ocp_e2m1_mxfp4::setSignMask;

    return data == 0b0;
}

template <>
inline double toDoublePacked<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             index_t         scaleIndex,
                                             index_t         dataIndex)
{

    if(isNaNPacked<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZeroPacked<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = getDataFromPackedF4(dataBytes, dataIndex);

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e2m1_mxfp4::scaleInfo.mantissaBits,
                                             ocp_e2m1_mxfp4::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E2M1_MXFP4_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline float toFloatPacked<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes,
                                           uint8_t const* dataBytes,
                                           index_t         scaleIndex,
                                           index_t         dataIndex)
{
    if(isNaNPacked<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZeroPacked<ocp_e2m1_mxfp4>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = getDataFromPackedF4(dataBytes, dataIndex);

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e2m1_mxfp4::scaleInfo.mantissaBits,
                                             ocp_e2m1_mxfp4::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E2M1_MXFP4_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

// no infinity representation in ocp_e2m1_mxfp4 will always return false
template <>
inline bool isInf<ocp_e2m1_mxfp4>(uint8_t const* scaleBytes [[maybe_unused]],
                                  uint8_t const* dataBytes [[maybe_unused]],
                                  index_t         scaleIndex [[maybe_unused]],
                                  index_t         dataIndex [[maybe_unused]])
{
    // no inf representation for ocp_e2m1_mxfp4
    return false;
}

template <>
inline bool isSubnorm<ocp_e2m1_mxfp4>(uint8_t const* dataBytes, index_t dataIndex)
{
    uint8_t data = *(dataBytes + dataIndex) & 0b00001111;
    return isSubNormal<uint16_t>(
        data, ocp_e2m1_mxfp4::dataInfo.mantissaBits, ocp_e2m1_mxfp4::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<ocp_e2m1_mxfp4>(uint8_t const* dataBytes, index_t dataIndex)
{
    uint8_t data = getDataFromPackedF4(dataBytes, dataIndex);
    return isSubNormal<uint16_t>(
        data, ocp_e2m1_mxfp4::dataInfo.mantissaBits, ocp_e2m1_mxfp4::dataInfo.exponentBits);
}

//return the sub normal double value of XN
//set XN = 1
template <>
inline void setOne<ocp_e2m1_mxfp4>(
    uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex, bool subNormal)
{
    *(scaleBytes + scaleIndex) = subNormal ? Constants::E8M0_2 : Constants::E8M0_1;
    *(dataBytes + dataIndex)
        = subNormal ? ocp_e2m1_mxfp4::dataSubNormalOneMask : ocp_e2m1_mxfp4::oneMask;
}

//set XN = 0, scale X will not be changed
template <>
inline void setZero<ocp_e2m1_mxfp4>(uint8_t* scaleBytes [[maybe_unused]],
                                    uint8_t* dataBytes,
                                    index_t   scaleIndex [[maybe_unused]],
                                    index_t   dataIndex)
{
    *(dataBytes + dataIndex) = ocp_e2m1_mxfp4::positiveZeroMask;
}

template <>
inline void setNaN<ocp_e2m1_mxfp4>(uint8_t* scaleBytes,
                                   uint8_t* dataBytes [[maybe_unused]],
                                   index_t   scaleIndex,
                                   index_t   dataIndex [[maybe_unused]])
{
    *(scaleBytes + scaleIndex) = Constants::E8M0_NAN;
}

//ocp_e2m1_mxfp4 does not have an infinity representation, method will just return
template <>
inline void setInf<ocp_e2m1_mxfp4>(uint8_t* scaleBytes [[maybe_unused]],
                                   uint8_t* dataBytes [[maybe_unused]],
                                   index_t   scaleIndex [[maybe_unused]],
                                   index_t   dataIndex [[maybe_unused]])
{
    return;
}

template <>
inline void
    setDataMax<ocp_e2m1_mxfp4>(uint8_t* dataBytes, index_t dataIndex, bool subNormal, bool positive)
{
    if(subNormal)
        *(dataBytes + dataIndex) = positive ? ocp_e2m1_mxfp4::dataMaxPositiveSubNormalMask
                                            : ocp_e2m1_mxfp4::dataMaxNegativeSubNormalMask;
    else
        *(dataBytes + dataIndex) = positive ? ocp_e2m1_mxfp4::dataMaxPositiveNormalMask
                                            : ocp_e2m1_mxfp4::dataMaxNegativeNormalMask;
}

template <>
inline void setDataMaxPacked<ocp_e2m1_mxfp4>(uint8_t* dataBytes,
                                             index_t   dataIndex,
                                             bool     subNormal,
                                             bool     positive)
{
    uint8_t mask = 0b0;
    if(subNormal)
        mask = positive ? ocp_e2m1_mxfp4::dataMaxPositiveSubNormalMask
                        : ocp_e2m1_mxfp4::dataMaxNegativeSubNormalMask;
    else
        mask = positive ? ocp_e2m1_mxfp4::dataMaxPositiveNormalMask
                        : ocp_e2m1_mxfp4::dataMaxNegativeNormalMask;

    setDataPackedF4(dataBytes, dataIndex, mask);
}

template <>
inline void setOnePacked<ocp_e2m1_mxfp4>(
    uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex, bool subNormal)
{
    *(scaleBytes + scaleIndex) = subNormal ? Constants::E8M0_2 : Constants::E8M0_1;
    uint8_t dataMask = subNormal ? ocp_e2m1_mxfp4::dataSubNormalOneMask : ocp_e2m1_mxfp4::oneMask;

    setDataPackedF4(dataBytes, dataIndex, dataMask);
}

//set XN = 0, scale X will not be changed
template <>
inline void setZeroPacked<ocp_e2m1_mxfp4>(uint8_t* scaleBytes [[maybe_unused]],
                                          uint8_t* dataBytes,
                                          index_t   scaleIndex [[maybe_unused]],
                                          index_t   dataIndex)
{
    setDataPackedF4(dataBytes, dataIndex, ocp_e2m1_mxfp4::positiveZeroMask);
}

template <>
inline void setNaNPacked<ocp_e2m1_mxfp4>(uint8_t* scaleBytes,
                                         uint8_t* dataBytes [[maybe_unused]],
                                         index_t   scaleIndex,
                                         index_t   dataIndex [[maybe_unused]])
{
    *(scaleBytes + scaleIndex) = Constants::E8M0_NAN;
}

template <>
inline uint64_t satConvertToType<ocp_e2m1_mxfp4>(float value)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return sign ? ocp_e2m1_mxfp4::dataMaxNegativeNormalMask
                    : ocp_e2m1_mxfp4::dataMaxPositiveNormalMask;
    }

    if(std::abs(value) > ocp_e2m1_mxfp4::dataMaxNormalNumber) //covers inf case as well
        return sign ? ocp_e2m1_mxfp4::dataMaxNegativeNormalMask
                    : ocp_e2m1_mxfp4::dataMaxPositiveNormalMask;

    uint8_t res = convertToType<uint8_t, ocp_e2m1_mxfp4>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    if(std::abs(toFloat<ocp_e2m1_mxfp4>(tScale, tData, 0, 0))
       < ocp_e2m1_mxfp4::dataMinSubNormalNumber)
        return value < 0 ? ocp_e2m1_mxfp4::negativeZeroMask : ocp_e2m1_mxfp4::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToType<ocp_e2m1_mxfp4>(float value [[maybe_unused]])
{
    return 0b0;
}

template <>
inline uint64_t satConvertToTypeSR<ocp_e2m1_mxfp4>(float value, uint seed)
{
    cvt t;
    t.num      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
        return sign ? ocp_e2m1_mxfp4::dataMaxNegativeNormalMask
                    : ocp_e2m1_mxfp4::dataMaxPositiveNormalMask;

    if(std::abs(value) > ocp_e2m1_mxfp4::dataMaxNormalNumber) //covers inf case as well
        return sign ? ocp_e2m1_mxfp4::dataMaxNegativeNormalMask
                    : ocp_e2m1_mxfp4::dataMaxPositiveNormalMask;

    uint8_t res = convertToTypeSR<uint8_t, ocp_e2m1_mxfp4>(value, seed);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    if(std::abs(toFloat<ocp_e2m1_mxfp4>(tScale, tData, 0, 0))
       < ocp_e2m1_mxfp4::dataMinSubNormalNumber)
        return value < 0 ? ocp_e2m1_mxfp4::negativeZeroMask : ocp_e2m1_mxfp4::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToTypeSR<ocp_e2m1_mxfp4>(float value [[maybe_unused]],
                                                      uint  seed [[maybe_unused]])
{
    return 0b0;
}
