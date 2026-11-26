// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @brief The model's final norm as a megakernel phase: y = bf16(x * (mean(x^2) + eps)^-0.5 * w),
/// in fp32, one row per work item (HF RMSNorm with scale). x, w, y are bf16; the
/// hidden size must be a multiple of 8 (16-byte accesses). Counter slots are rows.
///
/// TODO: the epilogue workstream (megakernel/epilogue) has the same reductions; share them once
/// that header settles. A row per workgroup leaves most of the GPU idle for this short phase;
/// fusing it into the producer of x (the last layer's residual gate) would remove it.
/// NumWaves_ is the workgroup's wave count; the thread count follows the compile pass's wave size
/// (the host pass sees 64-lane waves, gfx1250 runs 32), so the type is the same in both passes.
template <index_t NumWaves_>
struct HeadNormPhase
{
    struct Args
    {
        const bf16_t* x; // [M, ld_x]
        const bf16_t* w; // [hidden]
        bf16_t* y;       // [M, ld_y]
        index_t M;
        index_t hidden;
        index_t ld_x;
        index_t ld_y;
        float eps;
    };

    using barrier_pipeline = void;

    static constexpr index_t kBlockSize = NumWaves_ * get_warp_size();
    static constexpr index_t kSlotRows  = 1;
    static constexpr index_t kVec       = 8;
    static constexpr index_t kWaves     = NumWaves_;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kWaves * sizeof(float); }

    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return a.M; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a) { return a.M; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args&, index_t) { return 1; }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t work) { return work; }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args&, index_t work)
    {
        return make_tuple(work, work + 1);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        return a.x != nullptr && a.w != nullptr && a.y != nullptr && a.M > 0 && a.hidden > 0 &&
               a.hidden % kVec == 0 && a.ld_x % kVec == 0 && a.ld_y % kVec == 0 &&
               a.ld_x >= a.hidden && a.ld_y >= a.hidden;
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* smem, const Token&)
    {
        using vec_t       = thread_buffer<bf16_t, kVec>;
        const index_t row = work;
        const bf16_t* x   = a.x + static_cast<long_index_t>(row) * a.ld_x;
        bf16_t* y         = a.y + static_cast<long_index_t>(row) * a.ld_y;
        const index_t tid = static_cast<index_t>(threadIdx.x);

        float ss = 0.f;
        for(index_t i = tid * kVec; i < a.hidden; i += kBlockSize * kVec)
        {
            const vec_t v = *reinterpret_cast<const vec_t*>(x + i);
            static_for<0, kVec, 1>{}([&](auto j) {
                const float f = type_convert<float>(v[j]);
                ss += f * f;
            });
        }

        // Workgroup sum: within the wave by xor shuffles, then across waves through LDS.
        for(index_t m = get_warp_size() / 2; m > 0; m /= 2)
            ss += __shfl_xor(ss, m);
        float* s_part = static_cast<float*>(smem);
        if(tid % get_warp_size() == 0)
            s_part[tid / get_warp_size()] = ss;
        block_sync_lds();
        float total = 0.f;
        static_for<0, kWaves, 1>{}([&](auto i) { total += s_part[i]; });

        // HF: pow(mean + eps, -0.5); 1 / sqrt is within an ulp of it.
        const float rstd = 1.f / sqrtf(total / static_cast<float>(a.hidden) + a.eps);

        for(index_t i = tid * kVec; i < a.hidden; i += kBlockSize * kVec)
        {
            const vec_t v  = *reinterpret_cast<const vec_t*>(x + i);
            const vec_t wv = *reinterpret_cast<const vec_t*>(a.w + i);
            vec_t o;
            static_for<0, kVec, 1>{}([&](auto j) {
                // Round to nearest even; ck_tile's default float -> bf16 truncates.
                o[j] = float_to_bf16<bf16_rounding_mode::standard>(
                    type_convert<float>(v[j]) * rstd * type_convert<float>(wv[j]));
            });
            *reinterpret_cast<vec_t*>(y + i) = o;
        }
    }
};

} // namespace ck_tile
