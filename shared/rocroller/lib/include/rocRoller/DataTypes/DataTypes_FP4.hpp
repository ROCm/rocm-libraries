/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
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

#include <cinttypes>
#include <cmath>
#include <iostream>

#define ROCROLLER_USE_FP4

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

namespace rocRoller
{
    typedef struct
    {
        uint32_t val : 4;
    } uint4_t;

    template <typename T>
    uint4_t f32_to_fp4(T        _x,
                       uint32_t scale_exp_f32    = 127,
                       bool     stochastic_round = false,
                       uint32_t in1              = 0)
    {
        uint32_t in = reinterpret_cast<uint32_t&>(_x);

        uint32_t sign_f32                 = (in >> 31);
        uint32_t trailing_significand_f32 = (in & 0x7fffff);
        int32_t  exp_f32                  = ((in & 0x7f800000) >> 23);
        int32_t  unbiased_exp_f32         = exp_f32 - 127;
        bool     is_f32_pre_scale_inf     = (exp_f32 == 0xff) && (trailing_significand_f32 == 0);
        bool     is_f32_pre_scale_nan     = (exp_f32 == 0xff) && (trailing_significand_f32 != 0);
        bool     is_f32_pre_scale_zero    = ((in & 0x7fffffff) == 0);
        bool     is_f32_pre_scale_denorm  = (exp_f32 == 0x00) && (trailing_significand_f32 != 0);
        // stochastic rounding
        // copied from existing f8_math.cpp
        if(stochastic_round)
        {
            trailing_significand_f32 += ((in1 & 0xfffff000) >> 12);
        }

        // normalize subnormal number
        if(is_f32_pre_scale_denorm)
        {
            unbiased_exp_f32 = -126;
            for(int mB = 22; mB >= 0; mB--)
            {
                if((trailing_significand_f32 >> mB) != 0)
                {
                    trailing_significand_f32 = (trailing_significand_f32 << (23 - mB)) & 0x7fffff;
                    unbiased_exp_f32         = unbiased_exp_f32 - (23 - mB);
                    break;
                }
            }
        }
        // at this point, leading significand bit is always 1 for non-zero input

        // apply scale
        unbiased_exp_f32 -= (scale_exp_f32 - 127);

        // at this point the exponent is the output exponent range

        uint4_t fp4                      = {0};
        bool    is_sig_ovf               = false;
        auto    round_f4_significand_rne = [&is_sig_ovf](uint32_t trail_sig_f4) {
            is_sig_ovf = false;
            // trail_sig_f4 is of the form 1.31
            uint32_t trail_significand = (trail_sig_f4 >> 30) & 0x1;
            uint32_t ulp_half_ulp      = (trail_sig_f4 >> 29) & 0x3; // 1.31 >> (31-1-1)
            uint32_t or_remain         = (trail_sig_f4 >> 0) & ((1 << 29) - 1);
            switch(ulp_half_ulp)
            {
            case 0:
            case 2:
                break;
            case 1:
                if(or_remain)
                {
                    trail_significand += 1;
                }
                break;
            case 3:
                trail_significand += 1;
                break;
            default:
                break;
            }
            is_sig_ovf = (((trail_significand >> 1) & 0x1) == 0x1);
            // trail_significand is of the form .1
            return (trail_significand & 0x1);
        };

        if(is_f32_pre_scale_inf || is_f32_pre_scale_nan || (scale_exp_f32 == 0xff))
        {
            fp4.val = (sign_f32 << 3) | 0x7;
        }
        else if(is_f32_pre_scale_zero)
        {
            fp4.val = (sign_f32 << 3);
        }
        else
        {
            int32_t  min_subnorm_uexp_f4 = -1;
            int32_t  max_subnorm_uexp_f4 = 0;
            int32_t  max_norm_uexp_f4    = +2;
            uint32_t mantissa_bits_f4    = 1;
            uint32_t exponent_bits_f4    = 2;
            if(unbiased_exp_f32 < min_subnorm_uexp_f4)
            {
                // scaled number is less than f4 min subnorm; output 0
                fp4.val = (sign_f32 << 3);
            }
            else if(unbiased_exp_f32 < max_subnorm_uexp_f4)
            {
                // scaled number is in f4 subnorm range,
                //  adjust mantissa such that unbiased_exp_f32 is
                //  max_subnorm_uexp_f4 and apply rne
                int32_t  exp_shift       = max_subnorm_uexp_f4 - unbiased_exp_f32;
                int32_t  unbiased_exp_f4 = unbiased_exp_f32 + exp_shift;
                uint32_t trail_sig_f4    = (1u << 31) | (trailing_significand_f32 << 8);
                trail_sig_f4 >>= exp_shift;
                trail_sig_f4 = round_f4_significand_rne(trail_sig_f4);
                fp4.val      = (sign_f32 << 3)
                          | ((uint8_t)((is_sig_ovf ? 0x01 : 0x00) << mantissa_bits_f4))
                          | (trail_sig_f4 & ((1 << mantissa_bits_f4) - 1));
            }
            else if(unbiased_exp_f32 <= max_norm_uexp_f4)
            {
                // scaled number is in f4 normal range
                //  apply rne
                uint32_t biased_exp_f4 = unbiased_exp_f32 + 1;
                uint32_t trail_sig_f4  = (1u << 31) | (trailing_significand_f32 << 8);
                trail_sig_f4           = round_f4_significand_rne(trail_sig_f4);
                biased_exp_f4 += (is_sig_ovf ? 1 : 0);
                if(biased_exp_f4 == (uint32_t)(max_norm_uexp_f4 + 1 + 1))
                {
                    fp4.val = (sign_f32 << 3) | 0x7;
                }
                else
                {
                    fp4.val
                        = (sign_f32 << 3)
                          | ((biased_exp_f4 & ((1 << exponent_bits_f4) - 1)) << mantissa_bits_f4)
                          | (trail_sig_f4 & ((1 << mantissa_bits_f4) - 1));
                }
            }
            else
            {
                // scaled number is greater than f4 max normal output
                //  clamp to f4 flt_max
                fp4.val = (sign_f32 << 3) | 0x7;
            }
        }

        return fp4;
    }

    template <typename T>
    T fp4_to_f32(uint4_t in, uint32_t scale_exp_f32 = 127)
    {
        uint32_t sign_fp4                 = ((in.val >> 3) & 1);
        uint32_t trailing_significand_fp4 = (in.val & 0x1);
        int32_t  unbiased_exp_fp4         = ((in.val & 0x6) >> 1) - 1;
        bool     is_fp4_pre_scale_zero    = ((in.val & 0x7) == 0x0);
        bool     is_fp4_pre_scale_dnrm    = (sign_fp4 == 0) && (trailing_significand_fp4 != 0);

        // normalize subnormal number
        if(is_fp4_pre_scale_dnrm)
        {
            trailing_significand_fp4 = 0;
            unbiased_exp_fp4         = -1;
        }
        // at this point, leading significand bit is always 1 for non-zero input

        // apply scale
        unbiased_exp_fp4 += (scale_exp_f32 - 127);

        // at this point the exponent range is the output exponent range

        uint32_t f32 = 0;

        bool is_sig_ovf                    = false;
        auto round_fp32_f4_significand_rne = [&is_sig_ovf](uint32_t trail_sig_fp32) {
            is_sig_ovf = false;
            // trail_sig_fp32 is of the form 1.31
            uint32_t trail_significand = (trail_sig_fp32 >> 8) & 0x7fffff;
            uint32_t ulp_half_ulp      = (trail_sig_fp32 >> 7) & 0x3; // 1.31 >> 7 = 1.24
            uint32_t or_remain         = (trail_sig_fp32 >> 0) & 0x7f;
            switch(ulp_half_ulp)
            {
            case 0:
            case 2:
                break;
            case 1:
                if(or_remain)
                {
                    trail_significand += 1;
                }
                break;
            case 3:
                trail_significand += 1;
                break;
            default:
                break;
            }
            is_sig_ovf = (((trail_significand >> 23) & 0x1) == 0x1);
            return (trail_significand & 0x7fffff); // trail_significand is of the form .23
        };

        if(scale_exp_f32 == 0xff)
        {
            f32 = (sign_fp4 << 31) | 0x7fc00000 | (trailing_significand_fp4 << 22);
        }
        else if(scale_exp_f32 == 0x7f)
        {
            // Scale is 1.0; Direct conversion
            switch(in.val & 0x7)
            {
            case 0:
                f32 = 0x00000000;
                break; // +-0.0
            case 1:
                f32 = 0x3f000000;
                break; // +-0.5
            case 2:
                f32 = 0x3f800000;
                break; // +-1.0
            case 3:
                f32 = 0x3fc00000;
                break; // +-1.5
            case 4:
                f32 = 0x40000000;
                break; // +-2.0
            case 5:
                f32 = 0x40400000;
                break; // +-3.0
            case 6:
                f32 = 0x40800000;
                break; // +-4.0
            case 7:
                f32 = 0x40c00000;
                break; // +-6.0
            default:
                f32 = 0;
                break;
            }
            f32 |= (sign_fp4 << 31);
        }
        else if(is_fp4_pre_scale_zero)
        {
            f32 = (sign_fp4 << 31);
        }
        else
        {
            if(unbiased_exp_fp4 < -149)
            {
                // scaled number is less than f32 min subnorm; output 0
                f32 = (sign_fp4 << 31);
            }
            else if(unbiased_exp_fp4 < -126)
            {
                // scaled number is in f32 subnorm range,
                //  adjust mantissa such that unbiased_exp_fp4 is -126 and apply rne
                int32_t  exp_shift        = -126 - unbiased_exp_fp4;
                int32_t  unbiased_exp_f32 = unbiased_exp_fp4 + exp_shift;
                uint32_t trail_sig_fp32   = (1u << 31) | (trailing_significand_fp4 << 30);
                trail_sig_fp32 >>= exp_shift;
                trail_sig_fp32 = round_fp32_f4_significand_rne(trail_sig_fp32);
                f32            = (sign_fp4 << 31) | ((is_sig_ovf ? 0x01 : 0x00) << 23)
                      | (trail_sig_fp32 & 0x7fffff);
            }
            else if(unbiased_exp_fp4 < +128)
            {
                // scaled number is in f32 normal range
                //  apply rne
                uint32_t biased_exp_f32 = unbiased_exp_fp4 + 127;
                uint32_t trail_sig_fp32 = (1u << 31) | (trailing_significand_fp4 << 30);
                trail_sig_fp32          = round_fp32_f4_significand_rne(trail_sig_fp32);
                biased_exp_f32 += (is_sig_ovf ? 1 : 0);
                if(biased_exp_f32 == +255)
                {
                    f32 = (sign_fp4 << 31) | 0x7f800000;
                }
                else
                {
                    f32 = (sign_fp4 << 31) | ((biased_exp_f32 & 0xff) << 23)
                          | (trail_sig_fp32 & 0x7fffff);
                }
            }
            else
            {
                // scaled number is greater than f32 max normL output +/- inf
                f32 = (sign_fp4 << 31) | 0x7f800000;
            }
        }

        //return f32;
        return reinterpret_cast<const T&>(f32);
    }

    /**
     * \ingroup DataTypes
     * @{
     */

    struct FP4
    {
        constexpr FP4()
            : data(FP4_ZERO_VALUE)
        {
        }

        FP4(FP4 const& other) = default;

        template <typename T>
        requires(!std::is_same_v<T, FP4> && std::is_convertible_v<T, float>) explicit FP4(
            T const& value)
            : data(float_to_fp4(static_cast<float>(value)).data)
        {
        }

        template <typename T>
        requires(std::is_convertible_v<T, float>) void operator=(T const& value)
        {
            data = float_to_fp4(static_cast<float>(value)).data;
        }

        explicit operator float() const
        {
            return fp4_to_float(*this);
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
        static const int8_t FP4_ZERO_VALUE = 0x0;

        static float fp4_to_float(const FP4 v)
        {
            uint4_t in;
            in.val = v.data;
            return fp4_to_f32<float>(in);
        }

        static FP4 float_to_fp4(const float v)
        {
            FP4     fp4;
            uint4_t fp4_tmp = f32_to_fp4<float>(v);
            fp4.data        = fp4_tmp.val;
            return fp4;
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const FP4& obj)
    {
        os << static_cast<float>(obj);
        return os;
    }

    inline FP4 operator+(FP4 a, FP4 b)
    {
        return static_cast<FP4>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP4 operator+(int a, FP4 b)
    {
        return static_cast<FP4>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP4 operator+(FP4 a, int b)
    {
        return static_cast<FP4>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP4 operator-(FP4 a, FP4 b)
    {
        return static_cast<FP4>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline FP4 operator*(FP4 a, FP4 b)
    {
        return static_cast<FP4>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline FP4 operator/(FP4 a, FP4 b)
    {
        return static_cast<FP4>(static_cast<float>(a) / static_cast<float>(b));
    }

    inline FP4 operator-(FP4 const& a)
    {
        return static_cast<FP4>(-static_cast<float>(a));
    }

    inline bool operator!(FP4 const& a)
    {
        return !static_cast<float>(a);
    }

    template <typename T>
    requires(std::is_convertible_v<T, float>) inline auto operator<=>(FP4 const& a, T const& b)
    {
        return static_cast<float>(a) <=> static_cast<float>(b);
    }

    template <typename T>
    requires(std::is_convertible_v<T, float>) inline bool operator==(FP4 const& a, T const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline bool operator==(FP4 const& a, FP4 const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline FP4& operator+=(FP4& a, FP4 b)
    {
        a = a + b;
        return a;
    }
    inline FP4& operator-=(FP4& a, FP4 b)
    {
        a = a - b;
        return a;
    }
    inline FP4& operator*=(FP4& a, FP4 b)
    {
        a = a * b;
        return a;
    }
    inline FP4& operator/=(FP4& a, FP4 b)
    {
        a = a / b;
        return a;
    }

    inline FP4 operator++(FP4& a)
    {
        a += FP4(1);
        return a;
    }
    inline FP4 operator++(FP4& a, int)
    {
        FP4 original_value = a;
        ++a;
        return original_value;
    }

    /**
     * @}
     */
} // namespace rocRoller

namespace std
{
    inline bool isinf(const rocRoller::FP4& a)
    {
        return std::isinf(static_cast<float>(a));
    }
    inline bool isnan(const rocRoller::FP4& a)
    {
        return std::isnan(static_cast<float>(a));
    }
    inline bool iszero(const rocRoller::FP4& a)
    {
        return (a.data & 0x1f) == 0x0;
    }

    inline rocRoller::FP4 abs(const rocRoller::FP4& a)
    {
        return static_cast<rocRoller::FP4>(std::abs(static_cast<float>(a)));
    }
    inline rocRoller::FP4 sin(const rocRoller::FP4& a)
    {
        return static_cast<rocRoller::FP4>(std::sin(static_cast<float>(a)));
    }
    inline rocRoller::FP4 cos(const rocRoller::FP4& a)
    {
        return static_cast<rocRoller::FP4>(std::cos(static_cast<float>(a)));
    }
    inline rocRoller::FP4 exp2(const rocRoller::FP4& a)
    {
        return static_cast<rocRoller::FP4>(std::exp2(static_cast<float>(a)));
    }
    inline rocRoller::FP4 exp(const rocRoller::FP4& a)
    {
        return static_cast<rocRoller::FP4>(std::exp(static_cast<float>(a)));
    }

    template <>
    struct is_floating_point<rocRoller::FP4> : true_type
    {
    };

    template <>
    struct hash<rocRoller::FP4>
    {
        size_t operator()(const rocRoller::FP4& a) const
        {
            return hash<uint8_t>()(a.data);
        }
    };
} // namespace std
