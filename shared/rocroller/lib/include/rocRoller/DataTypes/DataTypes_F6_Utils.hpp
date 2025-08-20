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
j* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <cassert>
#include <cinttypes>
#include <cstdint>

namespace rocRoller
{
    namespace DataTypes
    {
        enum //F6 formats
        {
            BF6_FMT = 0,
            FP6_FMT
        };

        /**
         *  @brief Cast a half or single precision number to 6-bit floating point number (FP6 or BF6)
         *
         *                Sign    Exponent     Mantissa        Bias
         *  FP6(E2M3)      1          2           3              1
         *  BF6(E3M2)      1          3           2              3
         *
         *  Special values:
         *                 0   INF/-INF  NaN/-NaN  Max Norm   Min Norm   Max SubNorm   Min SubNorm
         *  FP6(E2M3)     0x00    N/A       N/A     +/- 7.5   +/- 1.0    +/- 0.875     +/- 0.125
         *  BF6(E3M2)     0x00    N/A       N/A     +/- 28    +/- 0.25   +/- 0.1875    +/- 0.0625
         *
         *  @tparam T Type (half or single precision) to be cast to f6
         *
         *  @param _x Floating number to be cast to f8
         *  @param f6_src_fmt Format (FP6 or BF6) of F6
         *  @param stochastic_round Stochastic rounding or not
         */
        template <typename T>
        uint8_t cast_to_f6(T        _x,
                           uint32_t f6_src_fmt,
                           uint32_t scale_exp_f32    = 127,
                           bool     stochastic_round = false,
                           uint32_t in1              = 0)
        {
            // cast input to uint32
            uint32_t in;
            if constexpr(sizeof(T) == 4)
                in = reinterpret_cast<uint32_t&>(_x);
            else
                in = reinterpret_cast<uint16_t&>(_x);

            // read the sign, significand, and exponent from input
            uint32_t sign_f32                 = (in >> 31);
            uint32_t trailing_significand_f32 = (in & 0x7fffff);
            int      exp_f32                  = ((in & 0x7f800000) >> 23);
            int      unbiased_exp_f32         = exp_f32 - 127;

            // check if input is inf, nan, zero, or denorm
            bool is_f32_pre_scale_inf    = (exp_f32 == 0xff) && (trailing_significand_f32 == 0);
            bool is_f32_pre_scale_nan    = (exp_f32 == 0xff) && (trailing_significand_f32 != 0);
            bool is_f32_pre_scale_zero   = ((in & 0x7fffffff) == 0);
            bool is_f32_pre_scale_denorm = (exp_f32 == 0x00) && (trailing_significand_f32 != 0);

            // stochastic rounding
            // copied from existing f8_math.cpp
            if(stochastic_round)
            {
                trailing_significand_f32 += ((f6_src_fmt == BF6_FMT) ? ((in1 & 0xfffff800) >> 11)
                                                                     : ((in1 & 0xfffff000) >> 12));
            }

            // normalize subnormal number
            if(is_f32_pre_scale_denorm)
            {
                unbiased_exp_f32 = -126;
                for(int32_t mB = 22; mB >= 0; mB--)
                {
                    if((trailing_significand_f32 >> mB) != 0)
                    {
                        trailing_significand_f32
                            = (trailing_significand_f32 << (23 - mB)) & 0x7fffff;
                        unbiased_exp_f32 = unbiased_exp_f32 - (23 - mB);
                        break;
                    }
                }
            }
            // at this point, leading significand bit is always 1 for non-zero input

            // apply scale
            unbiased_exp_f32 -= (scale_exp_f32 - 127);

            // at this point the exponent is the output exponent range

            uint8_t fp6 = 0x0;

            bool is_sig_ovf               = false;
            auto round_f6_significand_rne = [&is_sig_ovf](uint32_t f6_src_fmt,
                                                          uint32_t trail_sig_f6) {
                is_sig_ovf = false;
                // trail_sig_f6 is of the form 1.31
                uint32_t mantissa_bits_f6 = (f6_src_fmt == BF6_FMT) ? 2 : 3;
                uint32_t trail_significand
                    = (trail_sig_f6 >> (31 - mantissa_bits_f6)) & ((1 << mantissa_bits_f6) - 1);
                uint32_t ulp_half_ulp = (trail_sig_f6 >> (31 - mantissa_bits_f6 - 1))
                                        & 0x3; // 1.31 >> (31-mantissa_bits_f6-1)
                uint32_t or_remain = (trail_sig_f6 >> 0) & ((1 << (31 - mantissa_bits_f6 - 1)) - 1);
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
                is_sig_ovf = (((trail_significand >> mantissa_bits_f6) & 0x1) == 0x1);
                // trail_significand is of the form .mantissa_bits_f6
                return (trail_significand & ((1 << mantissa_bits_f6) - 1));
            };

            if(is_f32_pre_scale_inf || is_f32_pre_scale_nan || (scale_exp_f32 == 0xff))
            {
                fp6 = (sign_f32 << 5) | 0x1f;
            }
            else if(is_f32_pre_scale_zero)
            {
                fp6 = (sign_f32 << 5);
            }
            else
            {
                int32_t  min_subnorm_uexp_f6 = (f6_src_fmt == BF6_FMT) ? -4 : -3;
                int32_t  max_subnorm_uexp_f6 = (f6_src_fmt == BF6_FMT) ? -2 : 0;
                int32_t  max_norm_uexp_f6    = (f6_src_fmt == BF6_FMT) ? +4 : +2;
                uint32_t mantissa_bits_f6    = (f6_src_fmt == BF6_FMT) ? 2 : 3;
                uint32_t exponent_bits_f6    = (f6_src_fmt == BF6_FMT) ? 3 : 2;
                if(unbiased_exp_f32 < min_subnorm_uexp_f6)
                {
                    // scaled number is less than f6 min subnorm; output 0
                    fp6 = (sign_f32 << 5);
                }
                else if(unbiased_exp_f32 < max_subnorm_uexp_f6)
                {
                    // scaled number is in f6 subnorm range,
                    //  adjust mantissa such that unbiased_exp_f32 is
                    //  max_subnorm_uexp_f6 and apply rne
                    int32_t exp_shift       = max_subnorm_uexp_f6 - unbiased_exp_f32;
                    int32_t unbiased_exp_f6 = unbiased_exp_f32 + exp_shift;
                    assert(unbiased_exp_f6 == max_subnorm_uexp_f6);
                    uint32_t trail_sig_f6 = (1u << 31) | (trailing_significand_f32 << 8);
                    trail_sig_f6 >>= exp_shift;

                    trail_sig_f6 = round_f6_significand_rne(f6_src_fmt, trail_sig_f6);
                    fp6          = (sign_f32 << 5)
                          | ((uint8_t)((is_sig_ovf ? 0x01 : 0x00) << mantissa_bits_f6))
                          | (trail_sig_f6 & ((1 << mantissa_bits_f6) - 1));
                }
                else if(unbiased_exp_f32 <= max_norm_uexp_f6)
                {
                    // scaled number is in f6 normal range
                    //  apply rne
                    int32_t  biased_exp_f6 = unbiased_exp_f32 + ((f6_src_fmt == BF6_FMT) ? 3 : 1);
                    uint32_t trail_sig_f6  = (1u << 31) | (trailing_significand_f32 << 8);
                    trail_sig_f6           = round_f6_significand_rne(f6_src_fmt, trail_sig_f6);
                    biased_exp_f6 += (is_sig_ovf ? 1 : 0);
                    if(biased_exp_f6 == (max_norm_uexp_f6 + ((f6_src_fmt == BF6_FMT) ? 3 : 1) + 1))
                    {
                        fp6 = (sign_f32 << 5) | 0x1f;
                    }
                    else
                    {
                        fp6 = (sign_f32 << 5)
                              | ((biased_exp_f6 & ((1 << exponent_bits_f6) - 1))
                                 << mantissa_bits_f6)
                              | (trail_sig_f6 & ((1 << mantissa_bits_f6) - 1));
                    }
                }
                else
                {
                    // scaled number is greater than f6 max normal output
                    //  clamp to f6 flt_max
                    fp6 = (sign_f32 << 5) | 0x1f;
                }
            }

            return fp6;
        }

        template <typename T>
        T cast_from_f6(uint8_t in, uint32_t f6_src_fmt, uint32_t scale_exp_f32 = 127)
        {
            // read the sign, mantissa, and exponent from input
            uint32_t sign_fp6                 = ((in >> 5) & 0x1);
            uint32_t trailing_significand_fp6 = (f6_src_fmt == BF6_FMT) ? (in & 0x3) : (in & 0x7);
            int32_t  exp_fp6 = (f6_src_fmt == BF6_FMT) ? ((in & 0x1c) >> 2) : ((in & 0x18) >> 3);
            uint32_t exp_bias_fp6     = (f6_src_fmt == BF6_FMT) ? 3 : 1;
            int32_t  unbiased_exp_fp6 = exp_fp6 - exp_bias_fp6;

            // check if the input is zero or denormal
            bool     is_fp6_pre_scale_zero = ((in & 0x1f) == 0x0);
            bool     is_fp6_pre_scale_dnrm = ((exp_fp6 == 0) && (trailing_significand_fp6 != 0));
            uint32_t mantissa_bits         = (f6_src_fmt == BF6_FMT) ? 2 : 3;

            // normalize subnormal number
            if(is_fp6_pre_scale_dnrm)
            {
                unbiased_exp_fp6 = (f6_src_fmt == BF6_FMT) ? -2 : 0;
                for(int32_t mB = (mantissa_bits - 1); mB >= 0; mB--)
                {
                    if((trailing_significand_fp6 >> mB) != 0)
                    {
                        trailing_significand_fp6
                            = (trailing_significand_fp6 << (mantissa_bits - mB))
                              & ((1 << mantissa_bits) - 1);
                        unbiased_exp_fp6 = unbiased_exp_fp6 - (mantissa_bits - mB);
                        break;
                    }
                }
            }
            // at this point, leading significand bit is always 1 for non-zero input

            // apply scale
            unbiased_exp_fp6 += (scale_exp_f32 - 127);

            // at this point the exponent range is the output exponent range

            uint32_t f32                           = 0;
            bool     is_sig_ovf                    = false;
            auto     round_fp32_f6_significand_rne = [&is_sig_ovf](uint32_t trail_sig_fp32) {
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
                f32 = (sign_fp6 << 31) | 0x7f8c0000
                      | (trailing_significand_fp6 << (23 - mantissa_bits));
            }
            else if(is_fp6_pre_scale_zero)
            {
                f32 = (sign_fp6 << 31);
            }
            else
            {
                if(unbiased_exp_fp6 < -149)
                {
                    // scaled number is less than f32 min subnorm; output 0
                    f32 = (sign_fp6 << 31);
                }
                else if(unbiased_exp_fp6 < -126)
                {
                    // scaled number is in f32 subnorm range,
                    //  adjust mantissa such that unbiased_exp_fp6 is -126 and apply rne
                    int32_t exp_shift        = -126 - unbiased_exp_fp6;
                    int32_t unbiased_exp_f32 = unbiased_exp_fp6 + exp_shift;
                    assert(unbiased_exp_f32 == -126);
                    uint32_t trail_sig_fp32
                        = (1u << 31) | (trailing_significand_fp6 << (31 - mantissa_bits));
                    trail_sig_fp32 >>= exp_shift;
                    trail_sig_fp32 = round_fp32_f6_significand_rne(trail_sig_fp32);
                    f32            = (sign_fp6 << 31) | ((is_sig_ovf ? 0x01 : 0x00) << 23)
                          | (trail_sig_fp32 & 0x7fffff);
                }
                else if(unbiased_exp_fp6 < +128)
                {
                    // scaled number is in f32 normal range
                    //  apply rne
                    uint32_t biased_exp_f32 = unbiased_exp_fp6 + 127;
                    uint32_t trail_sig_fp32
                        = (1u << 31) | (trailing_significand_fp6 << (31 - mantissa_bits));
                    trail_sig_fp32 = round_fp32_f6_significand_rne(trail_sig_fp32);
                    biased_exp_f32 += (is_sig_ovf ? 1 : 0);
                    if(biased_exp_f32 == +255)
                    {
                        f32 = (sign_fp6 << 31) | 0x7f800000;
                    }
                    else
                    {
                        f32 = (sign_fp6 << 31) | ((biased_exp_f32 & 0xff) << 23)
                              | (trail_sig_fp32 & 0x7fffff);
                    }
                }
                else
                {
                    // scaled number is greater than f32 max normL output +/- inf
                    f32 = (sign_fp6 << 31) | 0x7f800000;
                }
            }

            return reinterpret_cast<const T&>(f32);
        }

    }

    inline constexpr int8_t F6_ZERO_VALUE = 0x0;

    struct FP6;
    float fp6_to_float(const FP6 v);
    FP6   float_to_fp6(const float v);

    struct BF6;
    float bf6_to_float(const BF6 v);
    BF6   float_to_bf6(const float v);
}
