// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace rocRoller
{
    struct E8M0;
    struct E5M3;
    struct E4M3;
    template <typename T>
    concept CScaleType
        = std::is_same_v<T, E8M0> || std::is_same_v<T, E5M3> || std::is_same_v<T, E4M3>;

    template <CScaleType T>
    inline float scaleToFloat(T scale)
    {
        return static_cast<float>(scale);
    }

    template <CScaleType T>
    inline T floatToScale(float value)
    {
        return T(value);
    }

    constexpr uint8_t floatToE5M3(float value)
    {
        constexpr int     floatMantBits = 23;
        constexpr int     floatExpBits  = 8;
        constexpr int     floatExpBias  = 127;
        constexpr int     e5m3MantBits  = 3;
        constexpr int     e5m3ExpBits   = 5;
        constexpr int     e5m3ExpBias   = 15;
        constexpr uint8_t nanEncoding   = 0xFF;

        if(std::isnan(value) || std::isinf(value))
            return nanEncoding;

        float abs_val = std::fabs(value);

        if(abs_val == 0.0f)
            return 0x00;

        uint32_t bits;
        std::memcpy(&bits, &abs_val, sizeof(bits));

        // Remove float bias
        int32_t exponent = ((bits >> floatMantBits) & ((1u << floatExpBits) - 1)) - floatExpBias;
        // Get the bottom 23 bits
        uint32_t mantissa = bits & ((1u << floatMantBits) - 1);
        // Add implicit 1
        mantissa |= 1 << floatMantBits;

        // Convert raw exponent to E5 with bias
        int32_t e5_exp = exponent + e5m3ExpBias;

        // Exponent is too large, NaN
        if(e5_exp >= (1 << e5m3ExpBits) - 1)
            return nanEncoding;

        if(e5_exp <= 0)
        {
            // Denormal case: exponent too small for normalized range
            int shift = (1 - e5_exp) + floatMantBits - e5m3MantBits;
            // Clamp to 0 if the shift exceeds mantissa size
            if(shift > floatMantBits)
                return 0x00;

            uint32_t mant = mantissa >> shift;

            // Round to Nearest Even
            uint32_t rounding_bit  = 1 << (shift - 1);
            uint32_t rounding_mask = (1 << shift) - 1;
            uint32_t rem           = mantissa & rounding_mask;
            if((rem > rounding_bit) || (rem == rounding_bit && (mant & 1)))
                mant++;

            if(mant == 0)
                return 0x00;

            // 3-bits mantissa -> max 7, clamp
            if(mant > (1u << e5m3MantBits) - 1)
                return (1u << e5m3MantBits) - 1;

            return mant;
        }
        else
        {
            // Normal case
            int      shift = floatMantBits - e5m3MantBits;
            uint32_t mant  = mantissa >> shift;

            // Round to Nearest Even
            uint32_t rounding_bit  = 1 << (shift - 1);
            uint32_t rounding_mask = (1 << shift) - 1;
            uint32_t rem           = mantissa & rounding_mask;
            if((rem > rounding_bit) || (rem == rounding_bit && (mant & 1)))
                mant++;

            if(mant > (1u << e5m3MantBits) - 1)
            {
                mant = 0;
                e5_exp++;
                if(e5_exp >= (1 << e5m3ExpBits) - 1)
                    return nanEncoding;
            }

            return (e5_exp << e5m3MantBits) | (mant & ((1u << e5m3MantBits) - 1));
        }
    }

    constexpr float E5M3ToFloat(uint8_t scale)
    {
        constexpr int     floatMantBits = 23;
        constexpr int     floatExpBias  = 127;
        constexpr int     e5m3MantBits  = 3;
        constexpr int     e5m3ExpBits   = 5;
        constexpr int     e5m3ExpBias   = 15;
        constexpr uint8_t nan_encoding  = 0xFF;

        if(scale == 0x00)
            return 0.0f;
        if(scale == nan_encoding)
            return std::numeric_limits<float>::quiet_NaN(); // Quiet NaN as per spec

        uint32_t exponent = (scale >> e5m3MantBits) & ((1u << e5m3ExpBits) - 1);
        uint32_t mantissa = scale & ((1u << e5m3MantBits) - 1);
        float    result   = 0.0f;
        float    baseFrac = mantissa / std::ldexp(1.0f, e5m3MantBits);
        if(exponent == 0)
        {
            result = std::ldexp(baseFrac, -e5m3ExpBias + 1);
        }
        else
        {
            float frac = 1.0f + baseFrac;
            result     = std::ldexp(frac, exponent - e5m3ExpBias);
        }

        return result;
    }
} // namespace rocRoller
