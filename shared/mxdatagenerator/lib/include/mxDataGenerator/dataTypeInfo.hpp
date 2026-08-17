// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <limits.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "data_generation_utils.hpp"

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

    enum class ScaleType
    {
        None,
        E8M0,
        E5M3,
        E4M3,
    };

    template <uint SignBits, uint ExponentBits, uint MantissaBits, bool HasZero, bool HasInf, bool HasNaN>
    struct ScaleFmt
    {
        static constexpr uint signBits     = SignBits;
        static constexpr uint exponentBits = ExponentBits;
        static constexpr uint mantissaBits = MantissaBits;
        static constexpr int  bias         = (1 << (ExponentBits - 1)) - 1;
        static constexpr int  biasedEMin   = HasZero ? 1 : 0;
        static constexpr int  biasedEMax   = (1 << ExponentBits) - (HasInf ? 2 : 1);
        static constexpr int  unBiasedEMin = biasedEMin - bias;
        static constexpr int  unBiasedEMax = biasedEMax - bias;
        static constexpr bool hasZero = HasZero;
        static constexpr bool hasInf = HasInf;
        static constexpr bool hasNan = HasNaN;
    };

    template <ScaleType T>
    struct ScaleInfoFor;

    template <>
    struct ScaleInfoFor<ScaleType::E8M0>
    {
        using info = ScaleFmt<0, 8, 0, false, true, true>;
    };

    template <>
    struct ScaleInfoFor<ScaleType::E5M3>
    {
        using info = ScaleFmt<0, 5, 3, true, false, true>;
    };

    template <>
    struct ScaleInfoFor<ScaleType::E4M3>
    {
        // The scale doesn't have a sign bit, but it is needed
        // here for size computations.
        using info = ScaleFmt<1, 4, 3, true, false, true>;
    };

    template <ScaleType T>
    using ScaleInfo = typename ScaleInfoFor<T>::info;

    template <ScaleType T>
    inline constexpr bool isConcreteScaleType()
    {
        return T != ScaleType::None;
    };

    template<ScaleType T>
    inline constexpr uint getScaleNan();

    template<>
    inline constexpr uint getScaleNan<ScaleType::E8M0>()
    {
        return Constants::E8M0_NAN;
    }

    template<>
    inline constexpr uint getScaleNan<ScaleType::E5M3>()
    {
        return 0b11111111;
    }

    template<>
    inline constexpr uint getScaleNan<ScaleType::E4M3>()
    {
        return 0b01111111;
    }

    template<ScaleType T>
    inline constexpr uint getScaleOne();

    template<>
    inline constexpr uint getScaleOne<ScaleType::E8M0>()
    {
        return Constants::E8M0_1;
    }

    template<>
    inline constexpr uint getScaleOne<ScaleType::E5M3>()
    {
        return 0b01111000;
    }

    template<>
    inline constexpr uint getScaleOne<ScaleType::E4M3>()
    {
        return 0b00111000;
    }

    template<ScaleType T>
    inline constexpr uint getScaleTwo();

    template<>
    inline constexpr uint getScaleTwo<ScaleType::E8M0>()
    {
        return Constants::E8M0_2;
    }

    template<>
    inline constexpr uint getScaleTwo<ScaleType::E5M3>()
    {
        return 0b10000000;
    }

    template<>
    inline constexpr uint getScaleTwo<ScaleType::E4M3>()
    {
        return 0b01000000;
    }

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

    template <typename scaleInfo>
    inline constexpr uint8_t getScalePayloadMask()
    {
        constexpr uint scaleBits = scaleInfo::exponentBits + scaleInfo::mantissaBits;
        if constexpr(scaleBits >= 8)
            return 0xff;
        else
            return (1 << scaleBits) - 1;
    }

    template <typename scaleInfo>
    inline uint8_t getScalePayload(uint8_t scale)
    {
        return scale & getScalePayloadMask<scaleInfo>();
    }

    template <typename scaleInfo>
    inline bool isScaleNaN(uint8_t scale)
    {
        if constexpr(!scaleInfo::hasNan)
            return false;
        else
            return getScalePayload<scaleInfo>(scale) == getScalePayloadMask<scaleInfo>();
    }

    template <typename scaleInfo>
    inline double getScaleValue(uint8_t scale)
    {
        if(isScaleNaN<scaleInfo>(scale))
            return std::numeric_limits<double>::quiet_NaN();

        const uint8_t payload = getScalePayload<scaleInfo>(scale);
        if constexpr(scaleInfo::mantissaBits == 0)
        {
            const int exponent = getExponentValue<uint8_t>(
                payload, scaleInfo::mantissaBits, scaleInfo::exponentBits);
            return std::pow(2.0,
                            static_cast<double>(exponent - static_cast<int>(scaleInfo::bias)));
        }

        if constexpr(scaleInfo::hasZero)
        {
            if(payload == 0)
                return 0.0;
        }

        const int exponent
            = getExponentValue<uint8_t>(payload, scaleInfo::mantissaBits, scaleInfo::exponentBits);
        const int unbiasedExponent
            = exponent == 0 ? 1 - static_cast<int>(scaleInfo::bias)
                            : exponent - static_cast<int>(scaleInfo::bias);
        const double mantissa
            = getMantissaValue<uint8_t>(payload, scaleInfo::mantissaBits, scaleInfo::exponentBits);

        return mantissa * std::pow(2.0, static_cast<double>(unbiasedExponent));
    }

    template <typename scaleInfo>
    inline float getScaleValueFloat(uint8_t scale)
    {
        return static_cast<float>(getScaleValue<scaleInfo>(scale));
    }

    template <typename scaleInfo>
    inline bool isCanonicalScaleByte(uint8_t scale)
    {
        return scale == getScalePayload<scaleInfo>(scale);
    }

    template <typename scaleInfo>
    inline bool isFiniteScaleByte(uint8_t scale)
    {
        return isCanonicalScaleByte<scaleInfo>(scale) && std::isfinite(getScaleValue<scaleInfo>(scale));
    }

    template <typename scaleInfo>
    inline bool isFiniteNonzeroScaleByte(uint8_t scale)
    {
        const auto value = getScaleValue<scaleInfo>(scale);
        return isCanonicalScaleByte<scaleInfo>(scale) && std::isfinite(value) && value != 0.0;
    }

    template <typename scaleInfo>
    inline std::vector<uint8_t> enumerateFiniteScaleBytes(
        double minValue = 0.0, double maxValue = std::numeric_limits<double>::infinity())
    {
        std::vector<uint8_t> candidates;
        for(uint32_t raw = 0; raw <= getScalePayloadMask<scaleInfo>(); raw++)
        {
            const auto scale = static_cast<uint8_t>(raw);
            const auto value = getScaleValue<scaleInfo>(scale);
            if(std::isfinite(value) && value >= minValue && value <= maxValue)
                candidates.push_back(scale);
        }
        return candidates;
    }

    template <typename scaleInfo>
    inline std::vector<uint8_t> enumerateFiniteNonzeroScaleBytes(
        double minValue = 0.0, double maxValue = std::numeric_limits<double>::infinity())
    {
        std::vector<uint8_t> candidates;
        for(const auto scale : enumerateFiniteScaleBytes<scaleInfo>(minValue, maxValue))
        {
            if(getScaleValue<scaleInfo>(scale) != 0.0)
                candidates.push_back(scale);
        }
        return candidates;
    }

    template <typename scaleInfo>
    inline uint8_t nearestFiniteScaleByte(double target, const std::vector<uint8_t>& candidates)
    {
        assert(!candidates.empty());

        auto best     = candidates.front();
        auto bestDiff = std::abs(getScaleValue<scaleInfo>(best) - target);
        for(const auto candidate : candidates)
        {
            const auto diff = std::abs(getScaleValue<scaleInfo>(candidate) - target);
            if(diff < bestDiff)
            {
                best     = candidate;
                bestDiff = diff;
            }
        }

        return best;
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

    template <typename T, typename dataInfo>
    double convertToDoubleWithScale(T data, double scaleValue)
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

        return dataValue * scaleValue;
    }

    template <typename T, typename dataInfo, typename scaleInfo>
    double convertToDouble(T data, int scaleExp)
    {
        double scaleValue
            = std::pow(2, static_cast<double>((scaleExp - static_cast<int>(scaleInfo::bias))));
        return convertToDoubleWithScale<T, dataInfo>(data, scaleValue);
    }

    template <typename T, typename dataInfo>
    float convertToFloatWithScale(T data, float scaleValue)
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

        return dataValue * scaleValue;
    }

    template <typename T, typename dataInfo, typename scaleInfo>
    float convertToFloat(T data, int scaleExp)
    {
        float scaleValue
            = std::pow(2, static_cast<float>((scaleExp - static_cast<int>(scaleInfo::bias))));
        return convertToFloatWithScale<T, dataInfo>(data, scaleValue);
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
