#pragma once

#include "config.h"

#include <array>

// The compiled set of direct_wgrad configs.
//
// Intentionally free of device code: the autoshard generator includes this header on the host
// to count the configs it must instantiate.
//
// docs/algorithms/direct/direct-wgrad-config-table.md documents the wave tilings, which
// spreads the machine admits, which ones the corpus removed, and the order the table keeps.

namespace hipconv::cdna4::direct_wgrad
{

// A wave tile: the gradient one wave holds, and how far ahead its ring prefetches.
//
// The tile shrinks as the filter grows, against TILE_VGPR_BUDGET; prefetch_rows answers to the
// main-loop unroll and to MFMA cover. See direct-wgrad-config-table.md.
struct WaveTile
{
    int kh;
    int wave_c16;
    int wave_k16;
    int prefetch_rows;
};

constexpr WaveTile TILE_2X2 = {.kh = 2, .wave_c16 = 2, .wave_k16 = 2, .prefetch_rows = 3};
constexpr WaveTile TILE_3X3 = {.kh = 3, .wave_c16 = 2, .wave_k16 = 2, .prefetch_rows = 2};
constexpr WaveTile TILE_4X4 = {.kh = 4, .wave_c16 = 2, .wave_k16 = 1, .prefetch_rows = 3};
constexpr WaveTile TILE_5X5 = {.kh = 5, .wave_c16 = 1, .wave_k16 = 1, .prefetch_rows = 4};

// A second tile at 2x2 and 3x3, carrying half the K of the one above.
//
// It serves exactly one channel block, C(32) x K(32), which the wide tile cannot reach at these
// two filter sizes; every other block it could reach the wide tile already serves.
constexpr WaveTile TILE_2X2_NARROW_K = {.kh = 2, .wave_c16 = 2, .wave_k16 = 1, .prefetch_rows = 3};
constexpr WaveTile TILE_3X3_NARROW_K = {.kh = 3, .wave_c16 = 2, .wave_k16 = 1, .prefetch_rows = 2};

// The tile a 16-channel group fits, at the three filter sizes that need one of their own.
//
// A grouped entry covers its group exactly on both axes, so 16 channels per group admits one
// wave shape: C(16) x K(16), one whole group per wave. 5x5's own tile is that shape already.
constexpr WaveTile TILE_2X2_GROUP16 = {.kh = 2, .wave_c16 = 1, .wave_k16 = 1, .prefetch_rows = 3};
constexpr WaveTile TILE_3X3_GROUP16 = {.kh = 3, .wave_c16 = 1, .wave_k16 = 1, .prefetch_rows = 2};
constexpr WaveTile TILE_4X4_GROUP16 = {.kh = 4, .wave_c16 = 1, .wave_k16 = 1, .prefetch_rows = 3};

// One entry of the table before the packing crosses in: a wave tile and its wave spread.
//
// The tile is named here rather than looked up by filter size, because a filter size can carry
// more than one and the spread has to say which.
struct Spread
{
    WaveTile tile;
    Arrangement waves;

    // Ring depth, where the spread cannot afford the tile's. Zero takes the tile's.
    //
    // Unused since the block that needed it was pruned; the LDS budget against a widening block
    // will ask for it again. See direct-wgrad-config-table.md.
    int prefetch_rows = 0;

    constexpr int ring_depth() const
    {
        return prefetch_rows != 0 ? prefetch_rows : tile.prefetch_rows;
    }
};

// Every arrangement the register file, the LDS budget, the swizzle and the corpus admit.
//
// Widest tile first, 32 blocks in all. The order is load-bearing at both tie-breaks:
// preferred_unfold_n takes the smaller packing, and the ranking's stable sort takes the first
// entry of equal index, which on a layer with C == K is common. direct-wgrad-config-table.md has
// the blocks each filter size gave up and why a K-narrow entry precedes the wide twin it ties.
constexpr auto spreads = std::array{
    // ---- 2x2, on the C(32) x K(32) wave tile ----
    Spread{TILE_2X2, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(64)  K(128)
    Spread{TILE_2X2, {.waves_c = 8, .waves_k = 1, .waves_q = 1}}, // C(256) K(32)
    Spread{TILE_2X2, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(64)  K(64)
    Spread{TILE_2X2, {.waves_c = 1, .waves_k = 4, .waves_q = 2}}, // C(32)  K(128)
    // The one block the wide tile cannot reach here; see TILE_2X2_NARROW_K.
    Spread{TILE_2X2_NARROW_K, {.waves_c = 1, .waves_k = 2, .waves_q = 4}}, // C(32) K(32)

    Spread{TILE_2X2, {.waves_c = 2, .waves_k = 1, .waves_q = 4}}, // C(64)  K(32)

    // ---- 2x2 at 32 channels per group, on the same K-narrow tile ----
    Spread{TILE_2X2_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 1, .waves_g = 4}}, // C(128) K(128), 4 groups
    Spread{TILE_2X2_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 2, .waves_g = 2}}, // C(64) K(64), 2 groups

    // ---- 2x2 at 16 channels per group, one whole group to a wave ----
    Spread{TILE_2X2_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 1, .waves_g = 8}}, // C(128) K(128), 8 groups
    Spread{TILE_2X2_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 2, .waves_g = 4}}, // C(64) K(64), 4 groups

    // ---- 3x3, on the C(32) x K(32) wave tile ----
    Spread{TILE_3X3, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(64)  K(128)
    Spread{TILE_3X3, {.waves_c = 8, .waves_k = 1, .waves_q = 1}}, // C(256) K(32)
    Spread{TILE_3X3, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(64)  K(64)
    Spread{TILE_3X3, {.waves_c = 4, .waves_k = 1, .waves_q = 2}}, // C(128) K(32)
    // The one block the wide tile cannot reach here; see TILE_3X3_NARROW_K. It precedes the
    // C(32) x K(64) it ties on every term, that order being the whole decision between them.
    Spread{TILE_3X3_NARROW_K, {.waves_c = 1, .waves_k = 2, .waves_q = 4}}, // C(32) K(32)

    Spread{TILE_3X3, {.waves_c = 1, .waves_k = 2, .waves_q = 4}}, // C(32)  K(64)
    Spread{TILE_3X3, {.waves_c = 2, .waves_k = 1, .waves_q = 4}}, // C(64)  K(32)

    // ---- 3x3 at 32 channels per group, on the same K-narrow tile ----
    //
    // Halving the wave is not optional here: the full tile puts one group on one wave, and the
    // waves_g * waves_q of 8 that follows is over the LDS budget however the waves split.
    Spread{TILE_3X3_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 1, .waves_g = 4}}, // C(128) K(128), 4 groups
    Spread{TILE_3X3_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 2, .waves_g = 2}}, // C(64) K(64), 2 groups

    // ---- 3x3 at 16 channels per group, one whole group to a wave ----
    //
    // The pair repeats at every filter size and differs in nothing but reduction depth, so only
    // the atomic term separates them and it always prefers the shallower split.
    Spread{TILE_3X3_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 1, .waves_g = 8}}, // C(128) K(128), 8 groups
    Spread{TILE_3X3_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 2, .waves_g = 4}}, // C(64) K(64), 4 groups

    // ---- 4x4, on the C(32) x K(16) wave tile ----
    //
    // The wave gives up K rather than C so that the epilogue's drain stays a 128-byte run of
    // channels, at the cost of every arrangement leaving one wave on the K axis.
    Spread{TILE_4X4, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(64)  K(64)
    Spread{TILE_4X4, {.waves_c = 4, .waves_k = 2, .waves_q = 1}}, // C(128) K(32)
    // Kept on measurement rather than on the model, which picks them almost never: they are the
    // measured best on 24 corpus layers over the block it picks instead.
    Spread{TILE_4X4, {.waves_c = 1, .waves_k = 8, .waves_q = 1}}, // C(32)  K(128)
    Spread{TILE_4X4, {.waves_c = 1, .waves_k = 4, .waves_q = 2}}, // C(32)  K(64)
    Spread{TILE_4X4, {.waves_c = 1, .waves_k = 2, .waves_q = 4}}, // C(32)  K(32)

    // ---- 4x4 at 32 channels per group, on the tile it already had ----
    //
    // The four-group C(128) x K(128) block is absent: over the widest wave any grouped entry
    // runs on, a C(128) block spills its epilogue.
    Spread{TILE_4X4,
           {.waves_c = 1, .waves_k = 2, .waves_q = 2, .waves_g = 2}}, // C(64) K(64), 2 groups

    // ---- 4x4 at 16 channels per group, one whole group to a wave ----
    Spread{TILE_4X4_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 1, .waves_g = 8}}, // C(128) K(128), 8 groups
    Spread{TILE_4X4_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 2, .waves_g = 4}}, // C(64) K(64), 4 groups

    // ---- 5x5, on the C(16) x K(16) wave tile ----
    //
    // Both axes sit at the swizzle's floor on a single wave, so only the arrangements leaving
    // two waves on each survive: three of the ten. The third, C(64) x K(32), spills its row
    // loop under every packing, only a C(32) S row folding a run of ladder rungs onto one
    // address.
    Spread{TILE_5X5, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(32) K(64)
    Spread{TILE_5X5, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(32) K(32)

    // ---- 5x5 at 16 channels per group, two groups to a workgroup ----
    //
    // The one grouped shape 5x5 can hold, and the only entry in the table under the cache line
    // the others fill: every wider window spills the row loop, for the reason above. A
    // 32-channel group needs no entry here, the ungrouped C(32) x K(32) block covering it
    // exactly. See direct-wgrad-config-table.md and the grouped section of direct-wgrad.md.
    Spread{TILE_5X5,
           {.waves_c = 1, .waves_k = 1, .waves_q = 4, .waves_g = 2}}, // C(32) K(32), 2 groups
};

// Images packed into one column block of the MFMA reduction, in increasing order.
//
// An axis of its own: packing costs no LDS at any value and moves neither the tile nor the wave
// count, so preferred_unfold_n decides it on the output width alone.
constexpr auto packings = std::array{1, 2, 4};

// Every spread crossed with every packing, in the order the two tie-breaks need.
//
// The spreads are filter-major already, so crossing the packings innermost keeps one filter
// size's entries contiguous and leaves both rules reading the order they describe.
constexpr auto make_configs()
{
    std::array<Config, spreads.size() * packings.size()> configs{};

    std::size_t out = 0;
    for(const Spread& spread : spreads)
        for(const int unfold_n : packings)
            configs[out++] = Config{.kh            = spread.tile.kh,
                                    .kw            = spread.tile.kh,
                                    .wave_c16      = spread.tile.wave_c16,
                                    .wave_k16      = spread.tile.wave_k16,
                                    .waves_c       = spread.waves.waves_c,
                                    .waves_k       = spread.waves.waves_k,
                                    .waves_q       = spread.waves.waves_q,
                                    .waves_g       = spread.waves.waves_g,
                                    .unfold_n      = unfold_n,
                                    .prefetch_rows = spread.ring_depth()};
    return configs;
}

constexpr auto configs    = make_configs();
constexpr int num_configs = static_cast<int>(configs.size());

// Every entry is one whole workgroup of waves, on a tile the register file and swizzle admit.
//
// Checked here rather than at the kernel template so it fires in every translation unit that
// reads the table, the autoshard generator's host build included, where no kernel is
// instantiated and nothing else would look. The wave count is checked first because a spread
// naming a filter with no wave tile leaves default-constructed configs at the end of the array.
constexpr bool table_is_well_formed()
{
    for(const Config& cfg : configs)
    {
        if(cfg.waves() != WAVES_PER_WORKGROUP)
            return false;
        if(cfg.unfold_n < 1 || MFMA_K % cfg.unfold_n != 0)
            return false;
        if(cfg.acc_vgprs() + cfg.operand_vgprs() > TILE_VGPR_BUDGET)
            return false;
        if(cfg.block_c() < MIN_SWIZZLED_CHANS || cfg.block_k() < MIN_SWIZZLED_CHANS)
            return false;

        // A grouped entry exists to fill a cache line, so a window under one line defeats it.
        //
        // 64 channels of fp16 is 128 bytes on each operand. 5x5 is exempt, every window that
        // would fill the line spilling its row loop.
        if(cfg.waves_g > 1 && cfg.kh != 5 && (cfg.block_c() < 64 || cfg.block_k() < 64))
            return false;
    }
    return true;
}

static_assert(table_is_well_formed(),
              "a direct_wgrad config is malformed: waves_c * waves_k * waves_q * waves_g must "
              "be WAVES_PER_WORKGROUP, so a wave given to the spatial axis or to a second "
              "group has to come off C or K; unfold_n must divide the MFMA's column count; the "
              "gradient and the operands must fit TILE_VGPR_BUDGET, which the wave tile has to "
              "shrink for as the filter grows; both channel blocks must reach "
              "MIN_SWIZZLED_CHANS; and a grouped entry's window must reach a whole cache line");

} // namespace hipconv::cdna4::direct_wgrad
