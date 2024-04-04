/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2024 Advanced Micro Devices, Inc.
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

#include "DataTypes_Half.hpp"
#include <cinttypes>
#include <cmath>
#include <iostream>

#define ROCROLLER_USE_FP8_NANOO

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

namespace rocRoller
{
    inline int clz(uint32_t x)
    {
        return __builtin_clz(x);
    }

    template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
    uint8_t cast_to_f8(T _x, bool stoch, uint32_t rng)
    {
        constexpr bool is_half  = std::is_same<T, Half>::value;
        constexpr bool is_float = std::is_same<T, float>::value;
        static_assert(wm + we == 7, "wm+we==7");
        static_assert(is_half || is_float, "Only half and float can be cast to f8");

        //if(sizeof(T)==2 && we==5 && !negative_zero_nan)
        //return cast_to_f8_no_range_reduce<2, 5, Half>(_x, stoch, rng);

        const int mfmt = (sizeof(T) == 4) ? 23 : 10;
        uint32_t  x;
        if(sizeof(T) == 4)
            x = reinterpret_cast<uint32_t&>(_x);
        else
            x = reinterpret_cast<uint16_t&>(_x);

        uint32_t y, head, mantissa;
        int      exponent, bias;
        uint32_t sign;

        if(sizeof(T) == 4)
        {
            head     = x & 0xFF800000;
            mantissa = x & 0x7FFFFF;
            exponent = (head >> 23) & 0xFF;
            sign     = head >> 31;
            bias     = 127;
        }
        else
        {
            head     = x & 0xFC00;
            mantissa = x & 0x3FF;
            exponent = (head >> 10) & 0x1F;
            sign     = head >> 15;
            bias     = 15;
        }

        uint32_t signed_inf = (sign << 7) + (((1 << we) - 1) << wm);

        // Deal with inf and NaNs
        if(negative_zero_nan)
        {
            if(sizeof(T) == 4)
            {
                if((x & 0x7F800000) == 0x7F800000)
                    return 0x80;
            }
            else
            {
                //if(__hisinf(x) || __hisnan(x))
                if((x & 0x7C00) == 0x7C00)
                    return 0x80;
            }
        }
        else
        {
            if(sizeof(T) == 4)
            {
                if((x & 0x7F800000) == 0x7F800000)
                    return signed_inf + (mantissa != 0 ? 1 : 0);
            }
            else
            {
                if((x & 0x7C00) == 0x7C00)
                    return signed_inf + (mantissa != 0 ? 1 : 0);
            }
        }
        if(x == 0)
            return 0;

        // First need to check if it is normal or denorm as there is a difference of implict 1
        // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
        // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for
        // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
        // exponent and mantissa again

        // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
        const int f8_bias                  = (1 << (we - 1)) - 1 + (negative_zero_nan ? 1 : 0);
        const int f8_denormal_act_exponent = 1 - f8_bias; //actual exponent of f8 denormal
        // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
        // f8_exponent is the converted f8 exponent with bias encoding
        // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
        // the difference needs to be adjusted and mantissa shifted
        int act_exponent, f8_exponent, exponent_diff;

        if(exponent == 0)
        { // fp32/fp16 is in denormal.
            /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16 here.
   In this case, f8 is usually in denormal. But there could be exceptions.
   fp16 denormal has exponent bias 15 while bf8 with NANOO has exponent bias 16.
   It means that there are some numbers in fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15.
   fp16 numbers where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal.
   In this case, the fp16 mantissa should be shift left by 1  */
            act_exponent  = exponent - bias + 1;
            exponent_diff = f8_denormal_act_exponent
                            - act_exponent; // actual exponent is exponent-bias+1 as it is denormal
        }
        else
        { // fp32/fp16 is normal with implicit 1
            act_exponent = exponent - bias;
            if(act_exponent <= f8_denormal_act_exponent)
            {
                /* This is the case where fp32/fp16 is normal but it is in f8 denormal range.
       For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
       actual exponent is -7, it is actually larger due to the implict 1,
       Therefore it needs to be adjust to -6 and mantissa shift right by 1.
       So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
                exponent_diff = f8_denormal_act_exponent - act_exponent;
            }
            else
            { //both fp32/fp16 and f8 are in normal range
                exponent_diff
                    = 0; // exponent_diff=0 does not mean there is no difference for this case,
                //act_exponent could be larger. Just that it does not need shift mantissa
            }
            mantissa += (1 << mfmt); //Add the implicit 1 into mantissa
        }

        bool midpoint = (mantissa & ((1 << (mfmt - wm + exponent_diff)) - 1))
                        == (1 << (mfmt - wm + exponent_diff - 1));
        /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we shift right
     as shift right could rip off some residual part and make something not midpoint look like midpoint.
     For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than midpoint,
     but after shift right by 4 bits, it would look like midpoint.
  */

        if(exponent_diff > 0)
            mantissa >>= exponent_diff;
        else if(exponent_diff == -1)
            mantissa <<= -exponent_diff;
        bool implicit_one = mantissa & (1 << mfmt);
        //if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
        f8_exponent = (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias
                      - (implicit_one ? 0 : 1);

        //Now we have the exponent and mantissa adjusted
        uint32_t drop_mask = (1 << (mfmt - wm)) - 1;
        //bool midpoint = (mantissa & drop_mask) == ( 1 << (mfmt-wm-1) );
        bool odd = mantissa
                   & (1 << (mfmt - wm)); // if the least significant bit that is not truncated is 1
        mantissa
            += (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa)) & drop_mask;

        //Now we deal with overflow
        if(f8_exponent == 0)
        {
            if((1 << mfmt) & mantissa)
            {
                f8_exponent = 1; //denormal overflow to become normal, promote exponent
                //mantissa &=  (1<<mfmt) -1 ; //No need to make 1 implicit now as it will be addressed later
            }
        }
        else
        {
            if((1 << (mfmt + 1)) & mantissa)
            {
                mantissa >>= 1;
                f8_exponent++;
                //mantissa &=  (1<<mfmt) -1 ; // No need to make 1 implicit now as it will be addressed later
            }
        }

        mantissa >>= (mfmt - wm);

        // above range: quantize to maximum possible float of the same sign
        const int max_exp = (1 << we) - (negative_zero_nan ? 1 : 2);
        if(f8_exponent > max_exp)
        {
            if(clip)
            {
                mantissa    = (1 << wm) - 1;
                f8_exponent = max_exp;
            }
            else
            {
                return signed_inf;
            }
        }

        if(f8_exponent == 0 && mantissa == 0)
            return negative_zero_nan ? 0 : (sign << 7);
        mantissa &= (1 << wm) - 1;
        return (sign << 7) | (f8_exponent << wm) | mantissa;
    }

    template <int wm, int we, typename T, bool negative_zero_nan>
    T cast_from_f8(uint8_t x)
    {
        constexpr bool is_half  = std::is_same<T, Half>::value;
        constexpr bool is_float = std::is_same<T, float>::value;
        //constexpr bool is_bf16 = std::is_same<T,hip_bfloat16>::value;
        static_assert(is_half || is_float, "only half and float are supported");

        constexpr int weo = is_half ? 5 : 8;
        constexpr int wmo = is_half ? 10 : (is_float ? 23 : 7);

        T fInf, fNegInf, fNaN, fNeg0;
        if(is_half)
        {
            const uint16_t ihInf    = 0x7C00;
            const uint16_t ihNegInf = 0xFC00;
            const uint16_t ihNaN    = 0x7C01;
            const uint16_t ihNeg0   = 0x8000;
            fInf                    = reinterpret_cast<const Half&>(ihInf);
            fNegInf                 = reinterpret_cast<const Half&>(ihNegInf);
            fNaN                    = reinterpret_cast<const Half&>(ihNaN);
            fNeg0                   = reinterpret_cast<const Half&>(ihNeg0);
        }
        else if(is_float)
        {
            const uint32_t ifInf    = 0x7F800000;
            const uint32_t ifNegInf = 0xFF800000;
            const uint32_t ifNaN    = 0x7F800001;
            const uint32_t ifNeg0   = 0x80000000;
            fInf                    = reinterpret_cast<const float&>(ifInf);
            fNegInf                 = reinterpret_cast<const float&>(ifNegInf);
            fNaN                    = reinterpret_cast<const float&>(ifNaN);
            fNeg0                   = reinterpret_cast<const float&>(ifNeg0);
        }

        if(x == 0)
            return 0;

        uint32_t sign     = x >> 7;
        uint32_t mantissa = x & ((1 << wm) - 1);
        int      exponent = (x & 0x7F) >> wm;
        if(negative_zero_nan)
        {
            if(x == 0x80)
                return fNaN;
        }
        else
        {
            if(x == 0x80)
                return fNeg0;
            if(exponent == ((1 << we) - 1))
                return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
        }
        typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type retval;
        if(we == 5 && is_half && !negative_zero_nan)
        {
            retval = x << 8;
            return reinterpret_cast<const T&>(retval);
        }

        const int exp_low_cutoff
            = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

        //subnormal input
        if(exponent == 0)
        {
            //guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
            int sh = 1 + clz(mantissa) - (32 - wm);
            mantissa <<= sh;
            exponent += 1 - sh;
            /*
    exponent++;
    while(mantissa<(1<<wm)) {
      mantissa <<= 1;
      exponent--;
    }
    */
            mantissa &= ((1 << wm) - 1);
        }
        exponent += exp_low_cutoff - 1;
        mantissa <<= wmo - wm;

        // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
        if(exponent <= 0)
        {
            mantissa |= 1 << wmo;
            mantissa >>= 1 - exponent;
            exponent = 0;
        }

        if(sizeof(T) == 2)
            retval = (sign << 15) | (exponent << 10) | mantissa;
        else
            retval = (sign << 31) | (exponent << 23) | mantissa;
        return reinterpret_cast<const T&>(retval);
    }

    /**
 * \ingroup DataTypes
 * @{
 */

    struct FP8_NANOO
    {
        FP8_NANOO()
            : data(FP8_NANOO_ZERO_VALUE)
        {
        }

        FP8_NANOO(FP8_NANOO const& other) = default;

        template <typename T,
                  typename
                  = typename std::enable_if<(!std::is_same<T, FP8_NANOO>::value)
                                            && std::is_convertible<T, double>::value>::type>
        explicit FP8_NANOO(T const& value)
            : data(float_to_fp8(static_cast<double>(value)).data)
        {
        }

        explicit operator float() const
        {
            return fp8_to_float(*this);
        }

        operator double() const
        {
            return static_cast<double>(float(*this));
        }

        explicit operator int() const
        {
            return static_cast<int>(float(*this));
        }

        explicit operator uint32_t() const
        {
            return static_cast<uint32_t>(float(*this));
        }

        explicit operator uint64_t() const
        {
            return static_cast<uint64_t>(float(*this));
        }

        uint8_t data;

    private:
        static const int8_t FP8_NANOO_ZERO_VALUE = 0x0;

        static float fp8_to_float(const FP8_NANOO v)
        {
            return cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(v.data);
        }

        static FP8_NANOO float_to_fp8(const float v)
        {
            FP8_NANOO fp8;
            fp8.data = cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
                v, 0 /*stochastic*/, 0 /*rng*/);
            return fp8;
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const FP8_NANOO& obj)
    {
        os << static_cast<float>(obj);
        return os;
    }

    inline FP8_NANOO operator+(FP8_NANOO a, FP8_NANOO b)
    {
        return static_cast<FP8_NANOO>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP8_NANOO operator+(int a, FP8_NANOO b)
    {
        return static_cast<FP8_NANOO>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP8_NANOO operator+(FP8_NANOO a, int b)
    {
        return static_cast<FP8_NANOO>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP8_NANOO operator-(FP8_NANOO a, FP8_NANOO b)
    {
        return static_cast<FP8_NANOO>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline FP8_NANOO operator*(FP8_NANOO a, FP8_NANOO b)
    {
        return static_cast<FP8_NANOO>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline FP8_NANOO operator/(FP8_NANOO a, FP8_NANOO b)
    {
        return static_cast<FP8_NANOO>(static_cast<float>(a) / static_cast<float>(b));
    }

    inline FP8_NANOO operator-(FP8_NANOO const& a)
    {
        return static_cast<FP8_NANOO>(-static_cast<float>(a));
    }

    inline bool operator!(FP8_NANOO const& a)
    {
        return !static_cast<float>(a);
    }

    template <typename T, typename = typename std::enable_if_t<std::is_convertible_v<T, float>>>
    inline auto operator<=>(FP8_NANOO const& a, T const& b)
    {
        return static_cast<float>(a) <=> static_cast<float>(b);
    }

    template <typename T, typename = typename std::enable_if_t<std::is_convertible_v<T, float>>>
    inline bool operator==(FP8_NANOO const& a, T const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline bool operator==(FP8_NANOO const& a, FP8_NANOO const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline FP8_NANOO& operator+=(FP8_NANOO& a, FP8_NANOO b)
    {
        a = a + b;
        return a;
    }
    inline FP8_NANOO& operator-=(FP8_NANOO& a, FP8_NANOO b)
    {
        a = a - b;
        return a;
    }
    inline FP8_NANOO& operator*=(FP8_NANOO& a, FP8_NANOO b)
    {
        a = a * b;
        return a;
    }
    inline FP8_NANOO& operator/=(FP8_NANOO& a, FP8_NANOO b)
    {
        a = a / b;
        return a;
    }

    inline FP8_NANOO operator++(FP8_NANOO& a)
    {
        a += FP8_NANOO(1);
        return a;
    }
    inline FP8_NANOO operator++(FP8_NANOO& a, int)
    {
        FP8_NANOO original_value = a;
        ++a;
        return original_value;
    }

    /**
 * @}
 */
} // namespace rocRoller

namespace std
{
    inline bool isinf(const rocRoller::FP8_NANOO& a)
    {
        return std::isinf(static_cast<float>(a));
    }
    inline bool isnan(const rocRoller::FP8_NANOO& a)
    {
        return std::isnan(static_cast<float>(a));
    }
    inline bool iszero(const rocRoller::FP8_NANOO& a)
    {
        return (a.data & 0x7FFF) == 0;
    }

    inline rocRoller::FP8_NANOO abs(const rocRoller::FP8_NANOO& a)
    {
        return static_cast<rocRoller::FP8_NANOO>(std::abs(static_cast<float>(a)));
    }
    inline rocRoller::FP8_NANOO sin(const rocRoller::FP8_NANOO& a)
    {
        return static_cast<rocRoller::FP8_NANOO>(std::sin(static_cast<float>(a)));
    }
    inline rocRoller::FP8_NANOO cos(const rocRoller::FP8_NANOO& a)
    {
        return static_cast<rocRoller::FP8_NANOO>(std::cos(static_cast<float>(a)));
    }
    inline rocRoller::FP8_NANOO exp2(const rocRoller::FP8_NANOO& a)
    {
        return static_cast<rocRoller::FP8_NANOO>(std::exp2(static_cast<float>(a)));
    }
    inline rocRoller::FP8_NANOO exp(const rocRoller::FP8_NANOO& a)
    {
        return static_cast<rocRoller::FP8_NANOO>(std::exp(static_cast<float>(a)));
    }

    template <>
    struct is_floating_point<rocRoller::FP8_NANOO> : true_type
    {
    };

    template <>
    struct hash<rocRoller::FP8_NANOO>
    {
        size_t operator()(const rocRoller::FP8_NANOO& a) const
        {
            return hash<uint8_t>()(a.data);
        }
    };
} // namespace std
