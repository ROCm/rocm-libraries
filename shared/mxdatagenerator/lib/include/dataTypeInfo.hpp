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

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <data_generation_utils.hpp>

#if defined(__clang__)
#include <type_traits>
namespace my_math
{

    template <typename T>
        requires(std::is_integral_v<T>)
    constexpr float pow(T base, T exp)
    {
        if(exp == 0)
            return base < 0 ? -1.0f : 1.0f;
        T sign = 1;
        if(exp < 0)
        {
            sign = -1;
            exp  = -exp;
        }
        float ret = float(base);
        while(--exp)
            ret *= float(base);
        return sign > 0 ? ret : 1.0f / ret;
    }
}
#define constexpr_pow(base, exp) my_math::pow(base, exp)
#else
#define constexpr_pow(base, exp) std::pow(base, exp)
#endif

namespace DGen
{
    namespace Constants
    {
        constexpr uint8_t E8M0_1   = 0b01111111;
        constexpr uint8_t E8M0_2   = 0b10000000;
        constexpr uint8_t E8M0_3   = 0b10000010;
        constexpr uint8_t E8M0_135 = 0b10000111;
        constexpr uint8_t E8M0_142 = 0b10001110;
        constexpr uint8_t E8M0_MIN = 0b00000000;
        constexpr uint8_t E8M0_MAX = 0b11111110;
        constexpr uint8_t E8M0_NAN = 0b11111111;

        constexpr int32_t F32EXPONENTBITS = 8;
        constexpr int32_t F32MANTISSABITS = 23;
        constexpr int32_t F32SIGNBITS     = 1;
        constexpr int32_t F32BIAS         = 127;
        constexpr int32_t F32MINEXP       = -126;
        constexpr int32_t F32MAXEXP       = 127;
        constexpr int32_t F64EXPONENTBITS = 11;
        constexpr int32_t F64MANTISSABITS = 52;
        constexpr int32_t F64SIGNBITS     = 1;
        constexpr int32_t F64BIAS         = 1023;

        constexpr double QNaN   = std::numeric_limits<double>::quiet_NaN();
        constexpr double Inf    = std::numeric_limits<double>::infinity();
        constexpr double NegInf = -std::numeric_limits<double>::infinity();
    }

    struct E8M0_SCALE_INFO
    {
        static constexpr uint signBits     = 0;
        static constexpr uint exponentBits = 8;
        static constexpr uint mantissaBits = 0;
        static constexpr uint bias         = 127;

        static constexpr int unBiasedEMin = -127;
        static constexpr int unBiasedEMax = 127;
        static constexpr int biasedEMin   = 0;
        static constexpr int biasedEMax   = 254;
    };
    
    union cvt
    {
        float num;
        uint  bRep;
    };

    /**
     * Get the unbiased exponent value from a bit representation
     *
     * @param x
     *     The bit representation, can be any type
     *
     * @param mantissaBits
     *      How many mantissa bits x has
     *
     * @param exponentBits
     *      How many exponent bits x has
     *
     * @return
     *      An integer of the unbiased exponent
     *      value of x
     */
    template <typename T>
    inline int getExponentValue(T x, uint mantissaBits, uint exponentBits)
    {
        x >>= mantissaBits;

        x &= ((1 << exponentBits) - 1);

        return static_cast<int>(x);
    }

    template <typename T>
    inline bool isSubNormal(T x, uint mantissaBits, uint exponentBits)
    {
        return getExponentValue<T>(x, mantissaBits, exponentBits) == 0;
    }

    /**
     * Get the mantissa value from a bit representation
     *
     * @param x
     *     The bit representation, can be any type
     *
     * @param mantissaBits
     *      How many mantissa bits x has
     *
     * @param exponentBits
     *      How many exponent bits x has
     *
     * @return
     *      A double of the mantissa value
     *      of x
     */
    template <typename T>
    inline double getMantissaValue(T x, uint mantissaBits, uint exponentBits)
    {
        double mantissa = isSubNormal<T>(x, mantissaBits, exponentBits) ? 0.0f : 1.0f;

        for(uint i = 0; i < mantissaBits; i++)
        {

            mantissa += std::pow(2, -int32_t((mantissaBits - i))) * (x & 0b1);

            x >>= 1;
        }

        return mantissa;
    }

    /**
     * Check if the product of the scale and data
     * at a index is POSITIVE one or not from
     * an unpacked byte buffer
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is a
     *      POSITIVE one
     */
    template <typename DTYPE>
    inline bool isOne(uint8_t const* scaleBytes,
                      uint8_t const* dataBytes,
                      index_t         scaleIndex,
                      index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is zero or not from an unpacked
     * byte buffer
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is zero
     */
    template <typename DTYPE>
    inline bool isZero(uint8_t const* scaleBytes,
                       uint8_t const* dataBytes,
                       index_t         scaleIndex,
                       index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is NaN or not from an unpacked
     * byte buffer
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is NaN
     */
    template <typename DTYPE>
    inline bool isNaN(uint8_t const* scaleBytes,
                      uint8_t const* dataBytes,
                      index_t         scaleIndex,
                      index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is infinite from an unpacked byte
     * buffer
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is infinite
     */
    template <typename DTYPE>
    inline bool isInf(uint8_t const* scaleBytes,
                      uint8_t const* dataBytes,
                      index_t         scaleIndex,
                      index_t         dataIndex);

    /**
     * XXX
     */
    template <typename DTYPE>
    inline bool isSubnorm(uint8_t const* dataBytes, index_t dataIndex);

    /**
     * XXX
     */
    template <typename DTYPE>
    inline bool isSubnormPacked(uint8_t const* dataBytes, index_t dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is less than a value from an unpacked
     * byte buffer
     *
     * @param val
     *      The value to compare the product to
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is less
     *      than the value passed in
     */
    template <typename DTYPE>
    inline bool isLess(double         val,
                       uint8_t const* scaleBytes,
                       uint8_t const* dataBytes,
                       index_t         scaleIndex,
                       index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is greater than a value from an
     * unpacked byte buffer
     *
     * @param val
     *      The value to compare the product to
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is greater
     *      than the value passed in
     */
    template <typename DTYPE>
    inline bool isGreater(double         val,
                          uint8_t const* scaleBytes,
                          uint8_t const* dataBytes,
                          index_t         scaleIndex,
                          index_t         dataIndex);

    /**
     * Convert the product of the scale and data
     * to a double. This number can exceed the
     * limit of the databyte.
     * The databyte buffer is unpacked
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A double, the value of the product of the
     *      scale and data
     */
    template <typename DTYPE>
    inline double toDouble(uint8_t const* scaleBytes,
                           uint8_t const* dataBytes,
                           index_t         scaleIndex,
                           index_t         dataIndex);

    /**
     * Convert the product of the scale and data
     * to a float. This number can exceed the
     * limit of the databyte.
     * The databyte buffer is unpacked
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A float, the value of the product of the
     *      scale and data
     */
    template <typename DTYPE>
    inline float toFloat(uint8_t const* scaleBytes,
                         uint8_t const* dataBytes,
                         index_t         scaleIndex,
                         index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is POSITIVE one or not from
     * a packed byte buffer
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is a
     *      POSITIVE one
     */
    template <typename DTYPE>
    inline bool isOnePacked(uint8_t const* scaleBytes,
                            uint8_t const* dataBytes,
                            index_t         scaleIndex,
                            index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is zero or not from a packed
     * byte buffer
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is zero
     */
    template <typename DTYPE>
    inline bool isZeroPacked(uint8_t const* scaleBytes,
                             uint8_t const* dataBytes,
                             index_t         scaleIndex,
                             index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is NaN or not from a packed
     * byte buffer
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is NaN
     */
    template <typename DTYPE>
    inline bool isNaNPacked(uint8_t const* scaleBytes,
                            uint8_t const* dataBytes,
                            index_t         scaleIndex,
                            index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is infinite from a packed byte
     * buffer
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is infinite
     */
    template <typename DTYPE>
    inline bool isInfPacked(uint8_t const* scaleBytes,
                            uint8_t const* dataBytes,
                            index_t         scaleIndex,
                            index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is less than a value from a packed
     * byte buffer
     *
     * @param val
     *      The value to compare the product to
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is less
     *      than the value passed in
     */
    template <typename DTYPE>
    inline bool isLessPacked(double         val,
                             uint8_t const* scaleBytes,
                             uint8_t const* dataBytes,
                             index_t         scaleIndex,
                             index_t         dataIndex);

    /**
     * Check if the product of the scale and data
     * at a index is greater than a value from a
     * packed byte buffer
     *
     * @param val
     *      The value to compare the product to
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A boolean signaling whether or not the
     *      product of the scale and data is greater
     *      than the value passed in
     */
    template <typename DTYPE>
    inline bool isGreaterPacked(double         val,
                                uint8_t const* scaleBytes,
                                uint8_t const* dataBytes,
                                index_t         scaleIndex,
                                index_t         dataIndex);

    /**
     * Convert the product of the scale and data
     * to a double. This number can exceed the
     * limit of the databyte.
     * The databyte buffer is packed
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A double, the value of the product of the
     *      scale and data
     */
    template <typename DTYPE>
    inline double toDoublePacked(uint8_t const* scaleBytes,
                                 uint8_t const* dataBytes,
                                 index_t         scaleIndex,
                                 index_t         dataIndex);

    /**
     * Convert the product of the scale and data
     * to a float. This number can exceed the
     * limit of the databyte.
     * The databyte buffer is packed
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @return
     *      A float, the value of the product of the
     *      scale and data
     */
    template <typename DTYPE>
    inline float toFloatPacked(uint8_t const* scaleBytes,
                               uint8_t const* dataBytes,
                               index_t         scaleIndex,
                               index_t         dataIndex);

    /**
     * Set the product of the scale and data to be 1
     * The dataByte buffer should be unpacked
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @param subNormal
     *      A boolean to flag if the data should be a
     *      subnormal number. The default is false
     *
     */
    template <typename DTYPE>
    void setOne(uint8_t* scaleBytes,
                uint8_t* dataBytes,
                index_t   scaleIndex,
                index_t   dataIndex,
                bool     subNormal = false);

    /**
     * Set the product of the scale and data to be 0,
     * the scale will remain unchanged
     * The dataByte buffer should be unpacked
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     */
    template <typename DTYPE>
    void setZero(uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex);

    /**
     * Set the product of the scale and data to be NaN,
     * the scale will remain unchanged
     * The dataByte buffer should be unpacked
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     */
    template <typename DTYPE>
    void setNaN(uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex);

    /**
     * Set the product of the scale and data to be Inf,
     * the scale will remain unchanged
     * The dataByte buffer should be unpacked
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     */
    template <typename DTYPE>
    void setInf(uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex);

    /**
     * Set the element at the specified index
     * to be the max for that datatype.
     * This method is for unpacked dataByte buffer
     *
     * @param dataBytes
     *      The byte buffer to set an element
     *      to max
     *
     * @param dataIndex
     *      The index to the element to set
     *
     * @param subNormal
     *      An optional flag to set if the max
     *      will be subNormal or not.
     *      Default is false (normal)
     *
     * @param positive
     *      An optional flag to set if the max
     *      will be max positive or max negative.
     *      Default is true (positive)
     */
    template <typename DTYPE>
    void setDataMax(uint8_t* dataBytes,
                    index_t   dataIndex,
                    bool     subNormal = false,
                    bool     positive  = true);

    /**
     * Set the product of the scale and data to be 1
     * The dataByte buffer should be packed
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     *
     * @param subNormal
     *      A boolean to flag if the data should be a
     *      subnormal number. The default is false
     *
     */
    template <typename DTYPE>
    void setOnePacked(uint8_t* scaleBytes,
                      uint8_t* dataBytes,
                      index_t   scaleIndex,
                      index_t   dataIndex,
                      bool     subNormal = false);

    /**
     * Set the product of the scale and data to be 0,
     * the scale will remain unchanged
     * The dataByte buffer should be packed
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     */
    template <typename DTYPE>
    void
        setZeroPacked(uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex);

    /**
     * Set the product of the scale and data to be NaN,
     * the scale will remain unchanged
     * The dataByte buffer should be packed
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     */
    template <typename DTYPE>
    void setNaNPacked(uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex);

    /**
     * Set the product of the scale and data to be Inf,
     * the scale will remain unchanged
     * The dataByte buffer should be packed
     *
     * @param scaleBytes
     *      The byte buffer containing all the
     *      scale representation
     *
     * @param dataBytes
     *      The byte buffer containing all the
     *      data representation
     *
     * @param scaleIndex
     *      The index to the scale bit representation
     *
     * @param dataIndex
     *      The index to the data bit representation
     */
    template <typename DTYPE>
    void setInfPacked(uint8_t* scaleBytes, uint8_t* dataBytes, index_t scaleIndex, index_t dataIndex);

    /**
     * Set the element at the specified index
     * to be the max for that datatype.
     * This method is for packed dataByte buffer
     *
     * @param dataBytes
     *      The byte buffer to set an element
     *      to max
     *
     * @param dataIndex
     *      The index to the element to set
     *
     * @param subNormal
     *      An optional flag to set if the max
     *      will be subNormal or not.
     *      Default is false (normal)
     *
     * @param positive
     *      An optional flag to set if the max
     *      will be max positive or max negative.
     *      Default is true (positive)
     */
    template <typename DTYPE>
    void setDataMaxPacked(uint8_t* dataBytes,
                          index_t   dataIndex,
                          bool     subNormal = false,
                          bool     positive  = true);

    /**
     * SAT conversion from a wider format to the data type
     *
     * @param value
     *      The value to convert to the bit representation
     *
     * @return
     *      The bit representation of the value.
     *      If the datatype is less than 8 bit
     *      the MSB will be padded with 0s
     *
     */
    template <typename DTYPE>
    uint64_t satConvertToType(float value);

    /**
     * NON-SAT conversion from a wider format to the data type
     *
     * @param value
     *      The value to convert to the bit representation
     *
     * @return
     *      The bit representation of the value.
     *      If the datatype is less than 8 bit
     *      the MSB will be padded with 0s
     *
     */
    template <typename DTYPE>
    uint64_t nonSatConvertToType(float value);

    /**
     * SAT stochastic rounding from a wider format to the data type
     *
     * @param value
     *      The value to convert to the bit representation
     * 
     * @param seed
     *      The seed used for rounding
     * 
     * @return
     *      The bit representation of the value.
     *      If the datatype is less than 8 bit
     *      the MSB will be padded with 0s
     *
     */
    template <typename DTYPE>
    uint64_t satConvertToTypeSR(float value, uint seed);

    /**
     * NON-SAT stochastic rounding from a wider format to the data type
     *
     * @param value
     *      The value to convert to the bit representation
     * 
     * @param seed
     *      The seed used for rounding
     * 
     * @return
     *      The bit representation of the value.
     *      If the datatype is less than 8 bit
     *      the MSB will be padded with 0s
     *
     */
    template <typename DTYPE>
    uint64_t nonSatConvertToTypeSR(float value, uint seed);

    /**
     * Convert a float32 value to type T representation
     * DOES NOT CHECK FOR OUT OF RANGE/NAN/INF
     *      Should be done before calling this method
     *
     * @param value
     *      The float32 value to be converted to type T
     */
    template <typename T, typename DTYPE>
    T convertToType(float value);

    /**
     * Performs stochastic rounding on a float32 value to 
     * type T representation
     * DOES NOT CHECK FOR OUT OF RANGE/NAN/INF
     *      Should be done before calling this method
     * 
     * @param value
     *      The float32 value to be converted to type T
     * 
     * @param seed
     *      The seed used for rounding
     */
    template <typename T, typename DTYPE>
    T convertToTypeSR(float value, uint seed);

    template <typename T, typename dataInfo, typename scaleInfo>
    double convertToDouble(T data, int scaleExp)
    {
        double d_s = std::pow(
            -1, static_cast<double>(data >> (dataInfo::exponentBits + dataInfo::mantissaBits)));

        double d_e;
        if(isSubNormal<uint8_t>(data, dataInfo::mantissaBits, dataInfo::exponentBits))
            d_e = std::pow(2, 1 - static_cast<int>(dataInfo::bias));
        else
            d_e = std::pow(
                2,
                getExponentValue<uint8_t>(data, dataInfo::mantissaBits, dataInfo::exponentBits)
                    - static_cast<int>(dataInfo::bias));
        double d_m
            = getMantissaValue<uint8_t>(data, dataInfo::mantissaBits, dataInfo::exponentBits);

        double dataValue = d_s * d_e * d_m;
        double scaleValue
            = std::pow(2, static_cast<double>((scaleExp - static_cast<int>(scaleInfo::bias))));

        return dataValue * scaleValue;
    }

    template <typename T, typename dataInfo, typename scaleInfo>
    float convertToFloat(T data, int scaleExp)
    {
        float d_s = std::pow(
            -1, static_cast<float>(data >> (dataInfo::exponentBits + dataInfo::mantissaBits)));

        float d_e;
        if(isSubNormal<uint8_t>(data, dataInfo::mantissaBits, dataInfo::exponentBits))
            d_e = std::pow(2, 1 - static_cast<int>(dataInfo::bias));
        else
            d_e = std::pow(
                2,
                getExponentValue<uint8_t>(data, dataInfo::mantissaBits, dataInfo::exponentBits)
                    - static_cast<int>(dataInfo::bias));
        float d_m = getMantissaValue<uint8_t>(data, dataInfo::mantissaBits, dataInfo::exponentBits);

        float dataValue = d_s * d_e * d_m;
        float scaleValue
            = std::pow(2, static_cast<float>((scaleExp - static_cast<int>(scaleInfo::bias))));

        return dataValue * scaleValue;
    }
}

//
// Specializations
//

#include "bf16.hpp"
#include "f32.hpp"
#include "fp16.hpp"
#include "ocp_e2m1_mxfp4.hpp"
#include "ocp_e2m3_mxfp6.hpp"
#include "ocp_e3m2_mxfp6.hpp"
#include "ocp_e4m3_mxfp8.hpp"
#include "ocp_e5m2_mxfp8.hpp"

//
// Generic implementations
//
namespace DGen
{
    template <typename DataType>
    constexpr bool isScaled()
    {
        auto isF32  = std::is_same_v<DataType, f32>;
        auto isFP16 = std::is_same_v<DataType, fp16>;
        auto isBF16 = std::is_same_v<DataType, bf16>;
        return !(isF32 || isFP16 || isBF16);
    }
#include "dataTypeInfo_impl.hpp"
}
