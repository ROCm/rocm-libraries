// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

// Reference: https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/philox.cuh
class philox
{
    public:
    CK_TILE_HOST_DEVICE philox(unsigned long long seed_, unsigned long long offset_)
        : seed(reinterpret_cast<const uint2&>(seed_))
    {

        ull2* tmp = reinterpret_cast<ull2*>(&counter);
        tmp->x    = offset_;
    }

    CK_TILE_HOST_DEVICE uint4 get_philox_4x32(const unsigned long long subsequence) const
    {

        uint4 counter_ = counter;
        ull2* tmp      = reinterpret_cast<ull2*>(&counter_);
        tmp->y         = subsequence;

        uint2 key_ = seed;
// 7-round philox
#pragma unroll
        for(int i = 0; i < 6; i++)
        {
            counter_ = philox_single_round(counter_, key_);
            key_.x += kPhilox10A;
            key_.y += kPhilox10B;
        }
        uint4 output = philox_single_round(counter_, key_);
        return output;
    }

    CK_TILE_HOST_DEVICE void get_random_16x8(uint8_t* out,
                                             const unsigned long long subsequence) const
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);

        out_tmp[0] = tmp_ph.x;
        out_tmp[1] = tmp_ph.y;
        out_tmp[2] = tmp_ph.z;
        out_tmp[3] = tmp_ph.w;
    }

    CK_TILE_HOST_DEVICE void get_random_8x8(uint8_t* out,
                                            const unsigned long long subsequence,
                                            const index_t idx0,
                                            const index_t idx1) const
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32x4_t tmp;
        tmp[0]            = tmp_ph.x;
        tmp[1]            = tmp_ph.y;
        tmp[2]            = tmp_ph.z;
        tmp[3]            = tmp_ph.w;
        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);
        out_tmp[0]        = tmp[idx0];
        out_tmp[1]        = tmp[idx1];
    }

    CK_TILE_HOST_DEVICE void
    get_random_4x8(uint8_t* out, const unsigned long long subsequence, const index_t idx) const
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32x4_t tmp;
        tmp[0]            = tmp_ph.x;
        tmp[1]            = tmp_ph.y;
        tmp[2]            = tmp_ph.z;
        tmp[3]            = tmp_ph.w;
        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);
        out_tmp[0]        = tmp[idx];
    }

    private:
    struct ull2
    {
        uint64_t x;
        uint64_t y;
    };
    uint4 counter;
    const uint2 seed;

    CK_TILE_HOST_DEVICE uint2 mulhilo32(const unsigned int a, const unsigned int b) const
    {
#if defined(__gfx1250__)
        // gfx1250 lowers the 64-bit product below to v_mul_u64, which is several times
        // slower than a v_mul_lo_u32 + v_mul_hi_u32 pair. Same result.
        return make_uint2(a * b, __umulhi(a, b));
#else
        uint2* res;
        unsigned long long tmp;
        tmp = static_cast<unsigned long long>(a) * b;
        res = reinterpret_cast<uint2*>(&tmp);
        return *res;
#endif
    }

#if defined(__gfx1250__)
    // gfx1250: v_mad_u64_u32 returns the lo and the hi word of the 32x32 product in one
    // instruction, at about the cost of a single v_mul_hi_u32. The compiler does not select it
    // (a 64-bit product lowers to v_mul_u64, a * b + __umulhi(a, b) to v_mul_lo_u32 +
    // v_mul_hi_u32), hence the inline asm. The multiplier is an immediate operand, so there is no
    // extra move. Not volatile, so the compiler can still schedule/CSE it. The result is a 2 x
    // 32-bit vector instead of a 64-bit integer so that the xors in philox_single_round stay
    // 32-bit and fuse into v_xor3_b32. Same result as mulhilo32(M, b).
    template <unsigned int M>
    CK_TILE_DEVICE uint2 mulhilo32_const(const unsigned int b) const
    {
        uint32x2_t p;
        asm("v_mad_u64_u32 %0, null, %1, %2, 0" : "=v"(p) : "C"(M), "v"(b));
        return make_uint2(p[0], p[1]);
    }
#endif

    CK_TILE_HOST_DEVICE uint4 philox_single_round(const uint4 ctr, const uint2 key) const
    {
#if defined(__gfx1250__)
        uint2 res0 = mulhilo32_const<kPhiloxSA>(ctr.x);
        uint2 res1 = mulhilo32_const<kPhiloxSB>(ctr.z);
#else
        uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
        uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
#endif
        uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
        return ret;
    }

    static const unsigned long kPhilox10A = 0x9E3779B9;
    static const unsigned long kPhilox10B = 0xBB67AE85;
    static const unsigned long kPhiloxSA  = 0xD2511F53;
    static const unsigned long kPhiloxSB  = 0xCD9E8D57;
};

} // namespace ck_tile
