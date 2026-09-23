#pragma once

#include "config.h"
#include "input_lds_layout.h"
#include "matrix_layout.h"
#include "types.h"

#include <hip/hip_runtime.h>

#include <type_traits>

namespace hipconv::cdna4
{

using direct_l1::Config;

// Loads MFMA input-operand tiles (16 Q x 32 C) from the double-buffered LDS tile.
//
// One instance per wave; the per-lane swizzle seeds are recomputed from
// __lane_id() at each use, so the loader holds no per-lane state.
template <Config cfg, typename datatype_t>
class InputLoaderLds
{
public:
    static_assert(sizeof(datatype_t) == 2);

    using datatypex8_t = std::conditional_t<std::is_same_v<datatype_t, bf16_t>, bf16x8_t, fp16x8_t>;

    using Layout        = InputLdsLayout<cfg>;
    using OperandLayout = MatrixLayout<16, 32, 1, datatype_t>;

    __device__ InputLoaderLds(const datatype_t* lds)
        : lds_uint4_(reinterpret_cast<const uint4*>(lds))
    {
    }

    // Per-lane LDS column this lane reads (before the wave-uniform q base).
    //
    // lane_w = the MFMA N index in [0,16). Standard tile: lane_w is the column.
    // Unfold: the 16 lanes cover packed images of w_unfold columns each with a
    // per-image w_per_image stride, so the column is (lane_w / w_unfold)*w_per_image
    // + lane_w % w_unfold.
    __device__ static int lane_column()
    {
        const int lane_w = OperandLayout::outer(__lane_id());
        if constexpr(Layout::n_unfold == 1)
            return lane_w;
        else
            return (lane_w / Layout::w_unfold) * Layout::w_per_image + (lane_w % Layout::w_unfold);
    }

    // Per-lane swizzle seeds, recomputed from __lane_id() at each use.
    //
    // lane_base is the linear term lane_column()*w_stride; resid_seed is the per-lane
    // part of the (x + c8) mod C8 bank residue. Both are ~2 ALU ops off __lane_id(),
    // so they rematerialize; held as members their kernel-long live range spills, and
    // each reload's s_waitcnt vmcnt would drain the in-flight prefetches.
    __device__ static int lane_base() { return lane_column() * Layout::w_stride; }
    __device__ static int resid_seed()
    {
        const int lane     = __lane_id();
        const int c8_strip = OperandLayout::inner(lane) / 8;
        // Unsigned % reduces to a mask; signed % emits a 5-op correction that spills.
        return static_cast<int>(static_cast<unsigned>(lane_column() + c8_strip) %
                                static_cast<unsigned>(Layout::block_size_c8));
    }

    // Compile-time guard that load()'s decomposition equals Layout::Swizzle.
    //
    // load() avoids offset_uint4 (whose fat per-lane offset spills across the step
    // loop) for an equivalent decomposition of two rematerializable seeds. That
    // equality holds only for the additive (X_SHIFT=0) swizzle the input tile uses;
    // this enumerates the tile and fails the build if it ever stops matching.
    static constexpr bool decomposition_matches_swizzle()
    {
        constexpr int C8 = Layout::block_size_c8;
        for(int x = 0; x < Layout::block_size_w; ++x)
            for(int c8 = 0; c8 < C8; ++c8)
            {
                const int decomposed = x * C8 + static_cast<int>(static_cast<unsigned>(x + c8) %
                                                                 static_cast<unsigned>(C8));
                if(decomposed != Layout::Swizzle::offset_uint4(x, c8))
                    return false;
            }
        return true;
    }
    static_assert(decomposition_matches_swizzle(),
                  "InputLoaderLds offset decomposition desyncs from Layout::Swizzle; the input "
                  "tile must use an additive (X_SHIFT = 0) swizzle, i.e. "
                  "SwizzleReadModel::Wave64B128.");

    // Load one W(16) x C(32) matrix.
    //   toc:      which double-buffer half.
    //   h:        LDS row of the tile.
    //   q16:      LDS column of lane 0; the wave's 16 lanes read q16 + 0..15.
    //   c32_half: which C(32) half of the C(64) tile (shifts the c8 strip by 4).
    __device__ datatypex8_t load(int toc, int h, int q16, int c32_half) const
    {
        // Spill-free decomposition of Swizzle::offset_uint4 into wave-uniform + seeds.
        //
        //   offset = x*C8 + (x + c8) mod C8, with x = q16 + lane_column(),
        //            c8 = 4*c32_half + c8_strip
        // so the step loop carries just lane_base and resid_seed. Equal to the write
        // side's swizzle for every x (decomposition_matches_swizzle above).
        const int wave_uniform = Layout().tic(toc).h(h).offset + q16 * Layout::w_stride;
        const int residue =
            static_cast<int>(static_cast<unsigned>(resid_seed() + q16 + 4 * c32_half) %
                             static_cast<unsigned>(Layout::block_size_c8));
        const int off = wave_uniform + lane_base() + residue;
        return *reinterpret_cast<const datatypex8_t*>(&lds_uint4_[off]);
    }

    // Load a tile of W(16) x C(32) input matrices for one Kw-step.
    //
    // Fills input[h][q16] from (toc, h_base+h, q16_base + q16*16 + kw_shift,
    // c32_half). kw_shift is the filter-column offset the outer Kw loop applies.
    template <int H, int Q16>
    __device__ void load_step(int toc,
                              int h_base,
                              int q16_base,
                              int kw_shift,
                              int c32_half,
                              datatypex8_t (&input)[H][Q16]) const
    {
#pragma unroll
        for(int q16 = 0; q16 < Q16; ++q16)
        {
            const int q_lds = q16_base + q16 * 16 + kw_shift;
#pragma unroll
            for(int h = 0; h < H; ++h)
            {
                input[h][q16] = load(toc, h_base + h, q_lds, c32_half);
            }
        }
    }

private:
    static constexpr int WAVE_SIZE = direct_l1::WAVE_SIZE;
    const uint4* lds_uint4_;
};

} // namespace hipconv::cdna4
