#pragma once

// A kernel variant's template parameters and the constants and types they fix.
// Parenthesized section names refer to docs/algorithms/direct/direct-flex-cdna4.md.

#include "config.h"
#include "bunnies_cdna4.hpp"
#include "types.h"

namespace hipconv::cdna4::direct_flex
{

// Chunks per filter position: one, even, or odd >= 3; OneStep for one-step tiles.
enum class Chunks
{
    One,
    Even,
    Odd,
    OneStep
};

// One kernel variant's flags; each flag's code compiles only into its own kernels.
//
// N: positions per filter tile with positions inner, 0 for outer, where CHUNKS applies.
// ONE_FTILE: the filter is one filter tile. SPLIT: one piece of a split tile (Split-K).
// DGRAD: KRSC weights read transposed (Dgrad). DEPTH: filter rows span depth planes (Conv3d).
template <Config cfg_,
          DataType DT_,
          bool DGRAD_,
          bool NARROW_K_,
          bool C_TAIL_,
          Chunks CHUNKS_,
          int N_,
          bool ONE_FTILE_,
          bool SPLIT_,
          bool DEPTH_>
struct Traits
{
    static constexpr Config cfg     = cfg_;
    static constexpr DataType DT    = DT_;
    static constexpr bool DGRAD     = DGRAD_;
    static constexpr bool NARROW_K  = NARROW_K_;
    static constexpr bool C_TAIL    = C_TAIL_;
    static constexpr Chunks CHUNKS  = CHUNKS_;
    static constexpr int N          = N_;
    static constexpr bool ONE_FTILE = ONE_FTILE_;
    static constexpr bool SPLIT     = SPLIT_;
    static constexpr bool DEPTH     = DEPTH_;
    // The reduction split two or four ways in the workgroup (csplit_kernel.h).
    static constexpr bool CSPLIT = cfg.csplit > 1;
    static_assert((cfg.x_tile() == 256 || CSPLIT) && cfg.waves() == 8,
                  "the ping-pong schedule is written for an X(256) tile on 8 waves");
    static_assert(N == 0 || (N >= 3 && N <= TAB_ROWS),
                  "a sweep's loads look three positions ahead within a table buffer");
    static_assert(N > 0 || !ONE_FTILE, "only positions inner fixes the filter tiles");
    static_assert(!DGRAD || !NARROW_K, "dgrad's output channels are its granule-aligned C");
    // Dgrad on positions outer runs as phases, its tiles each a segment's (Strided dgrad).
    static constexpr bool PHASED = DGRAD && N == 0;
    static_assert(CHUNKS != Chunks::OneStep || (N == 0 && !SPLIT),
                  "one-step tiles run on positions outer, unsplit");
    static constexpr bool ONE_STEP  = CHUNKS == Chunks::OneStep;
    static constexpr bool ONE_CHUNK = CHUNKS == Chunks::One || ONE_STEP;
    // The current tile and the 2 (one-step: 3) tiles its loads and table builds reach.
    static constexpr int RING = ONE_STEP ? 4 : 3;

    using dtype = ToType<DT>;
    using arch  = bunnies::arch_cdna4;

    // A wave owns 2 x 2 X(wave_x) x K(wave_k) blocks, one per pair of input and weight halves.
    static constexpr int BLOCK_X16 = cfg.wave_x() / MFMA_M;
    static constexpr int BLOCK_K16 = cfg.wave_k() / MFMA_N;
    static constexpr int HALF      = cfg.half_x(); // rows of a half-tile's LDS buffer
    static constexpr int HALF_K    = cfg.half_k(); // K rows a K half holds; zeros fill its buffer

    // Weights take the A role, so a lane's accumulators hold four consecutive output channels.
    using mat_wgt = arch::matrix<bunnies::fpfmt_of<DT>, MFMA_N, MFMA_K, bunnies::use::A>;
    using mat_in  = arch::matrix<bunnies::fpfmt_of<DT>, MFMA_K, MFMA_M, bunnies::use::B>;
    using mat_acc = arch::matrix<bunnies::fpfmt::e8m23, MFMA_N, MFMA_M, bunnies::use::Acc>;
    using mat_out = arch::matrix<bunnies::fpfmt_of<DT>, MFMA_N, MFMA_M, bunnies::use::Acc>;
    using rt_wgt  = bunnies::reg_tile<mat_wgt, BLOCK_K16, 1>;
    using rt_in   = bunnies::reg_tile<mat_in, 1, BLOCK_X16>;
    using rt_acc  = bunnies::reg_tile<mat_acc, BLOCK_K16, BLOCK_X16>;
    using rt_out  = bunnies::reg_tile<mat_out, BLOCK_K16, BLOCK_X16>;

    static constexpr int granules   = Config::granules();
    static constexpr int elem_bytes = sizeof(dtype);
};

} // namespace hipconv::cdna4::direct_flex
