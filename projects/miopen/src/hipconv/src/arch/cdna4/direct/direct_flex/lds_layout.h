#pragma once

// Half-tile LDS swizzle and the reads of a wave's MFMA operands from it (Operand staging).
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "config.h"
#include "traits.h"
#include "bunnies.hpp"
#include "detail.h"

#include <hip/hip_runtime.h>

namespace hipconv::cdna4::direct_flex
{

namespace detail
{

// A DTL load's LDS slot follows the lane, so the swizzle moves to the global address.
__device__ inline int granule_for_slot(int row, int slot_g)
{
    return slot_g ^ (row & (Config::granules() - 1));
}

// Group g's place in a dgrad weight row of `groups` groups, conflict-free for ds_read_b64_tr_b16
// (Weights in KRSC).
__device__ inline int dgrad_group(int g, int row, int groups = 8)
{
    // Rows of four groups are 128 bytes, so row parity already alternates halves of the banks.
    if(groups == 4)
        return g ^ (((row >> 1) & 1) | ((row >> 2) & 2));
    return g ^ ((row & 3) | ((row >> 1) & 4));
}

} // namespace detail

template <class T>
class OperandReader
{
    using arch    = bunnies::arch_cdna4;
    using rt_in   = typename T::rt_in;
    using rt_wgt  = typename T::rt_wgt;
    using tr_read = arch::ds_read_b64_tr_b16;

    static constexpr Config cfg      = T::cfg;
    static constexpr int K_STEPS     = Config::k_steps();
    static constexpr int ROW_BYTES   = C_CHUNK * T::elem_bytes;
    static constexpr int BLOCK_BYTES = MFMA_M * ROW_BYTES;
    // A dgrad weight row holds the wider of an X half and a K half.
    static constexpr int DG_ROW_BYTES = (T::HALF > T::HALF_K ? T::HALF : T::HALF_K) * T::elem_bytes;
    static constexpr int DG_GROUP     = MFMA_N * T::elem_bytes;
    static constexpr int DG_GROUPS    = DG_ROW_BYTES / T::elem_bytes / MFMA_N;
    // A csplit wave reads only its C(32) half.
    static constexpr int K_READ = T::CSPLIT ? 1 : K_STEPS;

public:
    // Lane offsets shared by all LDS buffers.
    __device__ OperandReader(const int& xi, const int& ki) : xi_(xi), ki_(ki)
    {
        const int lane = __lane_id();
#pragma unroll
        for(int cs = 0; cs < K_STEPS; ++cs)
            lane_off_[cs] = (lane % 16) * ROW_BYTES +
                            ((cs * 4 + lane / 16) ^ (lane & 7)) * GRANULE * T::elem_bytes;
        if constexpr(T::DGRAD)
        {
            const int row = 8 * (lane / 16) + (lane / 4) % 4;
            wgt_lane_     = row * DG_ROW_BYTES + detail::dgrad_group(0, row, DG_GROUPS) * DG_GROUP +
                        (lane % 4) * 8;
        }
        // A csplit wave's memory phases are issue-bound, so its offsets are formed here, once.
        if constexpr(T::CSPLIT)
        {
            // Four ways, the waves of a group take alternate C(32) halves; two ways, each group
            // one.
            const int ks = T::cfg.csplit == 4 ? bunnies::wave_id() % 4 / 2 : bunnies::wave_id() / 4;
            const int lo = ks ? lane_off_[1] : lane_off_[0];
            in_cs_       = lo + xi * cfg.wave_x() * ROW_BYTES;
            wgt_cs_      = lo + ki * cfg.wave_k() * ROW_BYTES;
#pragma unroll
            for(int kb = 0; kb < T::BLOCK_K16; ++kb)
                wgt_kb_[kb] = (wgt_lane_ ^ ((ki * T::BLOCK_K16 + kb) * DG_GROUP)) +
                              ks * MFMA_K * DG_ROW_BYTES;
        }
    }

    // A volatile row band avoids a VGPR per buffer.
    __device__ __forceinline__ void read_in(const char* buf, rt_in (&dst)[K_STEPS]) const
    {
        if constexpr(T::CSPLIT)
        {
            bunnies::load_tile<arch::ds_load_b128>(
                dst[0], buf + in_cs_, [](int, int xb, int, int) { return xb * BLOCK_BYTES; });
            return;
        }
        const int band = bunnies::pin_sgpr(xi_ * cfg.wave_x() * ROW_BYTES);
        static_for<K_STEPS>([&]<int CS>() {
            bunnies::load_tile<arch::ds_load_b128>(
                dst[CS], buf + band + lane_off_[CS], [](int, int xb, int, int) {
                return xb * BLOCK_BYTES;
            });
        });
    }

    __device__ __forceinline__ void read_wgt(const char* buf, rt_wgt (&dst)[K_STEPS]) const
    {
        if constexpr(T::DGRAD)
        {
            // Two ds_read_b64_tr_b16 per C(32) half of a K(16) block, group swizzle XORed (Dgrad).
#pragma unroll
            for(int kb = 0; kb < T::BLOCK_K16; ++kb)
            {
                const char* p = buf;
                if constexpr(T::CSPLIT)
                    p += wgt_kb_[kb];
                else
                    p += wgt_lane_ ^ bunnies::pin_sgpr((ki_ * T::BLOCK_K16 + kb) * DG_GROUP);
                static_for<K_READ>([&]<int CS>() {
#pragma unroll
                    for(int rnd = 0; rnd < 2; ++rnd)
                        tr_read::load(const_cast<char*>(p + (CS * MFMA_K + 4 * rnd) * DG_ROW_BYTES),
                                      reinterpret_cast<tr_read::type*>(&dst[CS].block(kb, 0).data) +
                                          rnd);
                });
            }
            return;
        }
        if constexpr(T::CSPLIT)
        {
            bunnies::load_tile<arch::ds_load_b128>(
                dst[0], buf + wgt_cs_, [](int kb, int, int, int) { return kb * BLOCK_BYTES; });
            return;
        }
        const int band = bunnies::pin_sgpr(ki_ * cfg.wave_k() * ROW_BYTES);
        static_for<K_STEPS>([&]<int CS>() {
            bunnies::load_tile<arch::ds_load_b128>(
                dst[CS], buf + band + lane_off_[CS], [](int kb, int, int, int) {
                return kb * BLOCK_BYTES;
            });
        });
    }

private:
    const int& xi_;
    const int& ki_;
    int lane_off_[K_STEPS];
    int wgt_lane_ = 0;
    int in_cs_    = 0;
    int wgt_cs_   = 0;
    int wgt_kb_[T::BLOCK_K16];
};

} // namespace hipconv::cdna4::direct_flex
