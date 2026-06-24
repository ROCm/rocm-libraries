/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef ROCM_GUARD_ROCM_LIMITS_HPP
#define ROCM_GUARD_ROCM_LIMITS_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

enum float_round_style
{
    round_indeterminate       = -1,
    round_toward_zero         = 0,
    round_to_nearest          = 1,
    round_toward_infinity     = 2,
    round_toward_neg_infinity = 3
};

namespace detail {

constexpr unsigned long int_max(unsigned long n)
{
    // Note, left shift cannot be used to get the maximum value of int64_type or
    // uint64_type because it is undefined behavior to left shift 64 bits for
    // these types
    if(n == 8)
        return -1;
    return (1ul << (n * 8)) - 1;
}

template <class T>
struct numeric_limits_integer
{
    static constexpr const bool is_specialized = true;

    static constexpr const bool is_signed = T(-1) < T(0);
    static constexpr const int digits =
        static_cast<int>(sizeof(T) * 8 - static_cast<unsigned long>(is_signed));
    static constexpr const int digits10     = digits * 3 / 10;
    static constexpr const int max_digits10 = 0;
    static constexpr T min() noexcept
    {
        if constexpr(is_signed)
            return -max() - 1;
        return 0;
    }
    static constexpr T max() noexcept
    {
        if constexpr(is_signed)
            return int_max(sizeof(T)) / 2;
        return int_max(sizeof(T));
    }
    static constexpr T lowest() noexcept { return min(); }

    static constexpr const bool is_integer = true;
    static constexpr const bool is_exact   = true;
    static constexpr const int radix       = 2;
    static constexpr T epsilon() noexcept { return T(0); }
    static constexpr T round_error() noexcept { return T(0); }

    static constexpr const int min_exponent   = 0;
    static constexpr const int min_exponent10 = 0;
    static constexpr const int max_exponent   = 0;
    static constexpr const int max_exponent10 = 0;

    static constexpr const bool has_infinity = false;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const bool has_quiet_NaN = false;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const bool has_signaling_NaN = false;
    static constexpr T infinity() noexcept { return T(0); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr T quiet_NaN() noexcept { return T(0); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr T signaling_NaN() noexcept { return T(0); }
    static constexpr T denorm_min() noexcept { return T(0); }

    static constexpr const bool is_iec559  = false;
    static constexpr const bool is_bounded = true;
    static constexpr const bool is_modulo  = not is_signed;

    static constexpr const bool traps                    = false;
    static constexpr const bool tinyness_before          = false;
    static constexpr const float_round_style round_style = round_toward_zero;
};

template <class Base>
struct numeric_limits_fp_mixin : Base
{
    static constexpr const bool is_specialized = true;

    static constexpr const bool is_signed   = true;
    static constexpr const int max_digits10 = 2 + (Base::digits * 30103L) / 100000L;
    static constexpr const bool is_integer  = false;
    static constexpr const bool is_exact    = false;
    static constexpr const int radix        = __FLT_RADIX__;
    static constexpr typename Base::type round_error() noexcept { return 0.5F; }
    static constexpr typename Base::type lowest() noexcept { return -Base::max(); }
    static constexpr const bool has_infinity = true;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const bool has_quiet_NaN = true;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const bool has_signaling_NaN = true;
    static constexpr const bool is_iec559         = true;
    static constexpr const bool is_bounded        = true;
    static constexpr const bool is_modulo         = false;

    static constexpr const bool traps                    = false;
    static constexpr const bool tinyness_before          = false;
    static constexpr const float_round_style round_style = round_to_nearest;
};

struct numeric_limits_float
{
    using type                          = float;
    static constexpr const int digits   = __FLT_MANT_DIG__;
    static constexpr const int digits10 = __FLT_DIG__;
    static constexpr type min() noexcept { return __FLT_MIN__; }
    static constexpr type max() noexcept { return __FLT_MAX__; }

    static constexpr type epsilon() noexcept { return __FLT_EPSILON__; }

    static constexpr const int min_exponent   = __FLT_MIN_EXP__;
    static constexpr const int min_exponent10 = __FLT_MIN_10_EXP__;
    static constexpr const int max_exponent   = __FLT_MAX_EXP__;
    static constexpr const int max_exponent10 = __FLT_MAX_10_EXP__;

    static constexpr type infinity() noexcept { return __builtin_huge_valf(); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr type quiet_NaN() noexcept { return __builtin_nanf(""); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr type signaling_NaN() noexcept { return __builtin_nansf(""); }
    static constexpr type denorm_min() noexcept { return __FLT_DENORM_MIN__; }
};

struct numeric_limits_double
{
    using type = double;

    static constexpr const int digits   = __DBL_MANT_DIG__;
    static constexpr const int digits10 = __DBL_DIG__;
    static constexpr type min() noexcept { return __DBL_MIN__; }
    static constexpr type max() noexcept { return __DBL_MAX__; }

    static constexpr const int radix = __FLT_RADIX__;
    static constexpr type epsilon() noexcept { return __DBL_EPSILON__; }

    static constexpr const int min_exponent   = __DBL_MIN_EXP__;
    static constexpr const int min_exponent10 = __DBL_MIN_10_EXP__;
    static constexpr const int max_exponent   = __DBL_MAX_EXP__;
    static constexpr const int max_exponent10 = __DBL_MAX_10_EXP__;

    static constexpr type infinity() noexcept { return __builtin_huge_val(); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr type quiet_NaN() noexcept { return __builtin_nan(""); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr type signaling_NaN() noexcept { return __builtin_nans(""); }
    static constexpr type denorm_min() noexcept { return __DBL_DENORM_MIN__; }
};

#ifdef __FLT16_MAX__
struct numeric_limits_fp16
{
    using type = _Float16;

    static constexpr const int digits   = __FLT16_MANT_DIG__;
    static constexpr const int digits10 = __FLT16_DIG__;
    static constexpr type min() noexcept { return __FLT16_MIN__; }
    static constexpr type max() noexcept { return __FLT16_MAX__; }

    static constexpr const int radix = __FLT_RADIX__;
    static constexpr type epsilon() noexcept { return __FLT16_EPSILON__; }

    static constexpr const int min_exponent   = __FLT16_MIN_EXP__;
    static constexpr const int min_exponent10 = __FLT16_MIN_10_EXP__;
    static constexpr const int max_exponent   = __FLT16_MAX_EXP__;
    static constexpr const int max_exponent10 = __FLT16_MAX_10_EXP__;

    static constexpr type infinity() noexcept { return __builtin_huge_valf16(); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr type quiet_NaN() noexcept { return __builtin_nanf16(""); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr type signaling_NaN() noexcept { return __builtin_nansf16(""); }
    static constexpr type denorm_min() noexcept { return __FLT16_DENORM_MIN__; }
};
#endif

struct numeric_limits_bf16
{
    using type = __bf16;

    static constexpr type from_bits(unsigned short bits) noexcept
    {
        return __builtin_bit_cast(type, bits);
    }

    static constexpr const int digits   = 8;
    static constexpr const int digits10 = 2;
    static constexpr type min() noexcept { return from_bits(0x0080); }
    static constexpr type max() noexcept { return from_bits(0x7f7f); }

    static constexpr const int radix = __FLT_RADIX__;
    static constexpr type epsilon() noexcept { return from_bits(0x3c00); }

    static constexpr const int min_exponent   = -125;
    static constexpr const int min_exponent10 = -37;
    static constexpr const int max_exponent   = 128;
    static constexpr const int max_exponent10 = 38;

    static constexpr type infinity() noexcept { return from_bits(0x7f80); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr type quiet_NaN() noexcept { return from_bits(0x7fc0); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr type signaling_NaN() noexcept { return from_bits(0x7fa0); }
    static constexpr type denorm_min() noexcept { return from_bits(0x0001); }
};

} // namespace detail

template <class T>
struct numeric_limits
{
    static constexpr const bool is_specialized = false;
    static constexpr T min() noexcept { return T(); }
    static constexpr T max() noexcept { return T(); }
    static constexpr T lowest() noexcept { return T(); }

    static constexpr const int digits       = 0;
    static constexpr const int digits10     = 0;
    static constexpr const int max_digits10 = 0;
    static constexpr const bool is_signed   = false;
    static constexpr const bool is_integer  = false;
    static constexpr const bool is_exact    = false;
    static constexpr const int radix        = 0;
    static constexpr T epsilon() noexcept { return T(); }
    static constexpr T round_error() noexcept { return T(); }

    static constexpr const int min_exponent   = 0;
    static constexpr const int min_exponent10 = 0;
    static constexpr const int max_exponent   = 0;
    static constexpr const int max_exponent10 = 0;

    static constexpr const bool has_infinity = false;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const bool has_quiet_NaN = false;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr const bool has_signaling_NaN = false;
    static constexpr T infinity() noexcept { return T(); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr T quiet_NaN() noexcept { return T(); }
    // NOLINTNEXTLINE(readability-identifier-naming)
    static constexpr T signaling_NaN() noexcept { return T(); }
    static constexpr T denorm_min() noexcept { return T(); }

    static constexpr const bool is_iec559  = false;
    static constexpr const bool is_bounded = false;
    static constexpr const bool is_modulo  = false;

    static constexpr const bool traps                    = false;
    static constexpr const bool tinyness_before          = false;
    static constexpr const float_round_style round_style = round_toward_zero;
};

template <class T>
struct numeric_limits<const T> : numeric_limits<T>
{
};

template <class T>
struct numeric_limits<volatile T> : numeric_limits<T>
{
};

template <class T>
struct numeric_limits<const volatile T> : numeric_limits<T>
{
};

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ROCM_DEFINE_NUMERIC_LIMITS_INT(T)                        \
    template <>                                                  \
    struct numeric_limits<T> : detail::numeric_limits_integer<T> \
    {                                                            \
    }

ROCM_DEFINE_NUMERIC_LIMITS_INT(char);
ROCM_DEFINE_NUMERIC_LIMITS_INT(signed char);
ROCM_DEFINE_NUMERIC_LIMITS_INT(unsigned char);
ROCM_DEFINE_NUMERIC_LIMITS_INT(wchar_t);
ROCM_DEFINE_NUMERIC_LIMITS_INT(char16_t);
ROCM_DEFINE_NUMERIC_LIMITS_INT(char32_t);
ROCM_DEFINE_NUMERIC_LIMITS_INT(short);
ROCM_DEFINE_NUMERIC_LIMITS_INT(unsigned short);
ROCM_DEFINE_NUMERIC_LIMITS_INT(int);
ROCM_DEFINE_NUMERIC_LIMITS_INT(unsigned int);
ROCM_DEFINE_NUMERIC_LIMITS_INT(long);
ROCM_DEFINE_NUMERIC_LIMITS_INT(unsigned long);
ROCM_DEFINE_NUMERIC_LIMITS_INT(long long);
ROCM_DEFINE_NUMERIC_LIMITS_INT(unsigned long long);

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ROCM_DEFINE_NUMERIC_LIMITS_FLOAT(T, base)                            \
    template <>                                                              \
    struct numeric_limits<T> : detail::numeric_limits_fp_mixin<detail::base> \
    {                                                                        \
    }

ROCM_DEFINE_NUMERIC_LIMITS_FLOAT(float, numeric_limits_float);
ROCM_DEFINE_NUMERIC_LIMITS_FLOAT(double, numeric_limits_double);
#ifdef __FLT16_MAX__
ROCM_DEFINE_NUMERIC_LIMITS_FLOAT(_Float16, numeric_limits_fp16);
#endif
ROCM_DEFINE_NUMERIC_LIMITS_FLOAT(__bf16, numeric_limits_bf16);

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_LIMITS_HPP
