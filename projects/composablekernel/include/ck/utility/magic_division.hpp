// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "numeric_limits.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "type.hpp"
#include "tuple.hpp"

#ifdef CK_CODE_GEN_RTC
#define INT32_MAX 2147483647
#endif

namespace ck {

// magic number division
// Caution:
//   1. For uint32_t as dividend: magic number division implementation being used would produce
//   correct result if the dividend is uint32_t and its value is within 31-bit value range.
//   2. For int32_t as dividendd: magic number division for int32_t dividened has not been
//   implemented, the int32_t dividend would be bit-wise interpreted as uint32_t and magic number
//   division implementation for uint32_t is then used. Therefore, dividend value need to be
//   non-negative.
// TODO:
//   1. Implement magic number divison for int32_t
//   2. Implement magic number divison for unit32_t with 32-bit value range
struct MagicDivision
{
    // uint32_t
    __host__ __device__ static constexpr auto CalculateMagicNumbers(uint32_t divisor)
    {
        // WARNING: magic division is only applicable for division inside this range.
        // You should use the return value of CalculateMagicNumbers, if division is not inside this
        // range. The "else" logic below is to quiet down run-time error.
        if(divisor >= 1 && divisor <= ck::NumericLimits<int32_t>::Max())
        {
            uint32_t shift = 0;
            for(shift = 0; shift < 32; ++shift)
            {
                if((1U << shift) >= divisor)
                {
                    break;
                }
            }

            uint64_t one        = 1;
            uint64_t multiplier = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
            // assert(multiplier <= 0xffffffffUL);

            return make_tuple(uint32_t(multiplier), shift);
        }
        else
        {
            return make_tuple(uint32_t(0), uint32_t(0));
        }
    }

    __host__ __device__ static constexpr uint32_t CalculateMagicMultiplier(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers(divisor);

        return tmp[Number<0>{}];
    }

    __host__ __device__ static constexpr uint32_t CalculateMagicShift(uint32_t divisor)
    {
        auto tmp = CalculateMagicNumbers(divisor);

        return tmp[Number<1>{}];
    }

    // integral_constant<uint32_t, .>
    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<uint32_t, Divisor>)
    {
        constexpr auto tmp = CalculateMagicNumbers(uint32_t{Divisor});

        constexpr uint32_t multiplier = tmp[Number<0>{}];
        constexpr uint32_t shift      = tmp[Number<1>{}];

        return make_tuple(integral_constant<uint32_t, multiplier>{},
                          integral_constant<uint32_t, shift>{});
    }

    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicMultiplier(integral_constant<uint32_t, Divisor>)
    {
        constexpr uint32_t multiplier = CalculateMagicMultiplier(uint32_t{Divisor});

        return integral_constant<uint32_t, multiplier>{};
    }

    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicShift(integral_constant<uint32_t, Divisor>)
    {
        constexpr uint32_t shift = CalculateMagicShift(uint32_t{Divisor});

        return integral_constant<uint32_t, shift>{};
    }

    // integral_constant<int32_t, .>
    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicNumbers(integral_constant<uint32_t, Divisor>{});
    }

    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicMultiplier(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicMultiplier(integral_constant<uint32_t, Divisor>{});
    }

    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicShift(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicShift(integral_constant<uint32_t, Divisor>{});
    }

    // magic division for uint32_t
    __device__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = __umulhi(dividend, multiplier);
        return (tmp + dividend) >> shift;
    }

    __host__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = static_cast<uint64_t>(dividend) * multiplier >> 32;
        return (tmp + dividend) >> shift;
    }

    // magic division for int32_t
    // HACK: use dividend_i32 as if it's uint32_t, dividend_i32 need to be
    // non-negative for result to be correct
    // TODO: figure out how to do magic number divison for int32_t as dividended
    __device__ static constexpr int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = __umulhi(dividend_u32, multiplier);
        return (tmp + dividend_u32) >> shift;
    }

    __host__ static constexpr int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = static_cast<uint64_t>(dividend_u32) * multiplier >> 32;
        return (tmp + dividend_u32) >> shift;
    }
};

struct MDiv
{
    // 1 dword -> 3 dword storage
    uint32_t divisor;
    uint32_t multiplier;
    uint32_t shift; // TODO: 8 bit is enough

    // prefer construct on host
    __host__ __device__ MDiv(uint32_t divisor_) : divisor(divisor_)
    {
        auto tmp = MagicDivision::CalculateMagicNumbers(divisor_);

        multiplier = tmp[Number<0>{}];
        shift      = tmp[Number<1>{}];
    }

    __host__ __device__ MDiv() : divisor(0), multiplier(0), shift(0) {}

    __host__ __device__ void update(uint32_t divisor_)
    {
        divisor  = divisor_;
        auto tmp = MagicDivision::CalculateMagicNumbers(divisor_);

        multiplier = tmp[Number<0>{}];
        shift      = tmp[Number<1>{}];
    }

    __host__ __device__ uint32_t div(uint32_t dividend_) const
    {
        return MagicDivision::DoMagicDivision(dividend_, multiplier, shift);
    }

    __host__ __device__ void
    divmod(uint32_t dividend_, uint32_t& quotient_, uint32_t& remainder_) const
    {
        quotient_  = div(dividend_);
        remainder_ = dividend_ - (quotient_ * divisor);
    }

    __host__ __device__ uint32_t get() const { return divisor; }
};

struct MDiv2
{
    // 1 dword -> 2 dword storage, divisor need compute from runtime
    uint32_t multiplier;
    uint32_t shift; // TODO: 8 bit is enough

    // prefer construct on host
    __host__ __device__ MDiv2(uint32_t divisor_)
    {
        auto tmp = MagicDivision::CalculateMagicNumbers(divisor_);

        multiplier = tmp[Number<0>{}];
        shift      = tmp[Number<1>{}];
    }

    __host__ __device__ MDiv2() : multiplier(0), shift(0) {}

    __host__ __device__ uint32_t div(uint32_t dividend_) const
    {
        return MagicDivision::DoMagicDivision(dividend_, multiplier, shift);
    }

    __host__ __device__ void
    divmod(uint32_t dividend_, uint32_t divisor_, uint32_t& quotient_, uint32_t& remainder_) const
    {
        quotient_  = div(dividend_);
        remainder_ = dividend_ - (quotient_ * divisor_);
    }
};

} // namespace ck
