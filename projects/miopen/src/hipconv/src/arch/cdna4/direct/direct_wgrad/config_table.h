#pragma once

#include "config.h"

#include <array>

// The compiled set of direct_wgrad configs.
//
// Intentionally free of device code: the autoshard generator includes this header on the host
// to count the configs it must instantiate.
//
// docs/algorithms/details/direct/direct-wgrad-config-table.md documents the wave tilings, which
// spreads the machine admits, which ones the corpus removed, and the order the table keeps.

namespace hipconv::cdna4::direct_wgrad
{

// A wave tile: the gradient one wave holds, and how far ahead its ring prefetches.
//
// The tile shrinks as the filter grows, against TILE_VGPR_BUDGET; prefetch_rows answers to the
// main-loop unroll and to MFMA cover. See direct-wgrad-config-table.md.
//
// kw is named rather than taken from kh: a wave holds kh * kw accumulator tiles, so an oblong
// filter frees the registers a wider tile spends. 3×1 is the one that uses it.
struct WaveTile
{
    int kh;
    int kw;
    int wave_c16;
    int wave_k16;
    int prefetch_rows;
    int elem_bytes = 2;
};

constexpr WaveTile TILE_2X2 = {.kh = 2, .kw = 2, .wave_c16 = 2, .wave_k16 = 2, .prefetch_rows = 3};
constexpr WaveTile TILE_3X3 = {.kh = 3, .kw = 3, .wave_c16 = 2, .wave_k16 = 2, .prefetch_rows = 2};
constexpr WaveTile TILE_4X4 = {.kh = 4, .kw = 4, .wave_c16 = 2, .wave_k16 = 1, .prefetch_rows = 3};
constexpr WaveTile TILE_5X5 = {.kh = 5, .kw = 5, .wave_c16 = 1, .wave_k16 = 1, .prefetch_rows = 4};

// A second tile at 2×2 and 3×3, carrying half the K of the one above.
//
// It serves exactly one channel block, C(32) × K(32), which the wide tile cannot reach at these
// two filter sizes; every other block it could reach the wide tile already serves.
constexpr WaveTile TILE_2X2_NARROW_K = {.kh            = 2,
                                        .kw            = 2,
                                        .wave_c16      = 2,
                                        .wave_k16      = 1,
                                        .prefetch_rows = 3};
constexpr WaveTile TILE_3X3_NARROW_K = {.kh            = 3,
                                        .kw            = 3,
                                        .wave_c16      = 2,
                                        .wave_k16      = 1,
                                        .prefetch_rows = 2};

// The tile a 16-channel group fits, at 2×2, 3×3 and 4×4. TILE_3X1_GROUP16 is with the 3×1 tiles.
//
// A grouped entry covers its group exactly on both axes, so 16 channels per group admits one
// wave shape: C(16) × K(16), one whole group per wave. 5×5's own tile is that shape already.
constexpr WaveTile TILE_2X2_GROUP16 = {.kh            = 2,
                                       .kw            = 2,
                                       .wave_c16      = 1,
                                       .wave_k16      = 1,
                                       .prefetch_rows = 3};
constexpr WaveTile TILE_3X3_GROUP16 = {.kh            = 3,
                                       .kw            = 3,
                                       .wave_c16      = 1,
                                       .wave_k16      = 1,
                                       .prefetch_rows = 2};
constexpr WaveTile TILE_4X4_GROUP16 = {.kh            = 4,
                                       .kw            = 4,
                                       .wave_c16      = 1,
                                       .wave_k16      = 1,
                                       .prefetch_rows = 3};

// The tiles at tf32, one per filter size.
//
// A split operand block is the (big, small) bf16 pair and costs two registers where a 16-bit one
// costs one, so each filter takes the smallest tile it admits: C(16) × K(16), except 3×1, which
// keeps the wider C its oblong filter pays for as it does at 16 bits. The wider C(32) × K(16) at
// 3×3 clears TILE_VGPR_BUDGET and still spills. The measurements are in
// direct-wgrad-config-table.md.
constexpr WaveTile TILE_2X2_TF32 =
    {.kh = 2, .kw = 2, .wave_c16 = 1, .wave_k16 = 1, .prefetch_rows = 3, .elem_bytes = 4};
constexpr WaveTile TILE_3X3_TF32 =
    {.kh = 3, .kw = 3, .wave_c16 = 1, .wave_k16 = 1, .prefetch_rows = 2, .elem_bytes = 4};
constexpr WaveTile TILE_3X1_TF32 =
    {.kh = 3, .kw = 1, .wave_c16 = 2, .wave_k16 = 1, .prefetch_rows = 2, .elem_bytes = 4};
// 4×4 keeps the deep ring although it spills: the spill is in the item body, and the shallow ring
// that spills none drains the row loop instead and measures slower.
constexpr WaveTile TILE_4X4_TF32 =
    {.kh = 4, .kw = 4, .wave_c16 = 1, .wave_k16 = 1, .prefetch_rows = 2, .elem_bytes = 4};
// 5×5 is on the C(16) × K(16) floor already, so it has nothing to give back and spills its row
// loop at every ring depth. The row loop has no registers to spare, and each extra ring slot
// stages another row in registers, so the shallowest ring spills least. It also measures fastest,
// although its row loop drains several times a row.
constexpr WaveTile TILE_5X5_TF32 =
    {.kh = 5, .kw = 5, .wave_c16 = 1, .wave_k16 = 1, .prefetch_rows = 1, .elem_bytes = 4};

// The tiles at 3×1, the one oblong filter the table serves.
//
// A third of the filter area frees the registers for twice the input channels: C(64) × K(32)
// retires 24 MFMA per column block against 3×3's 36. C(64) × K(64) costs 256 VGPRs of a 192
// budget, and C(96) × K(32) fits it but gives the swizzle a C4 of 24.
//
// The wider wave is paid for on the spatial axis, every waves_q = 4 spread over it reaching
// 192 KiB against a 160 KiB budget. The narrow tiles carry those and the grouped entries.
constexpr WaveTile TILE_3X1 = {.kh = 3, .kw = 1, .wave_c16 = 4, .wave_k16 = 2, .prefetch_rows = 2};
constexpr WaveTile TILE_3X1_NARROW_C = {.kh            = 3,
                                        .kw            = 1,
                                        .wave_c16      = 2,
                                        .wave_k16      = 2,
                                        .prefetch_rows = 2};
constexpr WaveTile TILE_3X1_NARROW_K = {.kh            = 3,
                                        .kw            = 1,
                                        .wave_c16      = 2,
                                        .wave_k16      = 1,
                                        .prefetch_rows = 2};
constexpr WaveTile TILE_3X1_GROUP16  = {.kh            = 3,
                                        .kw            = 1,
                                        .wave_c16      = 1,
                                        .wave_k16      = 1,
                                        .prefetch_rows = 2};

// The tile at 5×1, which also runs 1×5 layers through the H/W exchange.
//
// See "Entries at 5×1" in docs/algorithms/details/direct/direct-wgrad-config-table.md.
constexpr WaveTile TILE_5X1 = {.kh = 5, .kw = 1, .wave_c16 = 2, .wave_k16 = 2, .prefetch_rows = 4};

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

    // Rows one buffer descriptor spans, or 0 for the whole image. See config.h.
    int rows_per_tile = 0;

    constexpr int ring_depth() const
    {
        return prefetch_rows != 0 ? prefetch_rows : tile.prefetch_rows;
    }
};

// Every arrangement the register file, the LDS budget, the swizzle and the corpus admit.
//
// Grouped by filter shape, with the tf32 entries last. The order is load-bearing at both
// tie-breaks: preferred_unfold_n takes the smaller packing, and the ranking's stable sort takes the
// first entry of equal index, which on a layer with C == K is common. direct-wgrad-config-table.md
// has the blocks each filter size gave up and why a K-narrow entry precedes the wide twin it ties.
constexpr auto spreads = std::array{
    // ---- 2×2, on the C(32) × K(32) wave tile ----
    Spread{TILE_2X2, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(64)  K(128)
    Spread{TILE_2X2, {.waves_c = 8, .waves_k = 1, .waves_q = 1}}, // C(256) K(32)
    Spread{TILE_2X2, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(64)  K(64)
    Spread{TILE_2X2, {.waves_c = 1, .waves_k = 4, .waves_q = 2}}, // C(32)  K(128)
    // The one block the wide tile cannot reach here; see TILE_2X2_NARROW_K.
    Spread{TILE_2X2_NARROW_K, {.waves_c = 1, .waves_k = 2, .waves_q = 4}}, // C(32) K(32)

    Spread{TILE_2X2, {.waves_c = 2, .waves_k = 1, .waves_q = 4}}, // C(64)  K(32)

    // ---- 2×2 at 32 channels per group, on the same K-narrow tile ----
    Spread{TILE_2X2_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 1, .waves_g = 4}}, // C(128) K(128), 4 groups
    Spread{TILE_2X2_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 2, .waves_g = 2}}, // C(64) K(64), 2 groups

    // ---- 2×2 at 16 channels per group, one whole group to a wave ----
    Spread{TILE_2X2_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 1, .waves_g = 8}}, // C(128) K(128), 8 groups
    Spread{TILE_2X2_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 2, .waves_g = 4}}, // C(64) K(64), 4 groups

    // ---- 3×3, on the C(32) × K(32) wave tile ----
    Spread{TILE_3X3, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(64)  K(128)
    Spread{TILE_3X3, {.waves_c = 8, .waves_k = 1, .waves_q = 1}}, // C(256) K(32)
    Spread{TILE_3X3, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(64)  K(64)
    Spread{TILE_3X3, {.waves_c = 4, .waves_k = 1, .waves_q = 2}}, // C(128) K(32)
    // The one block the wide tile cannot reach here; see TILE_3X3_NARROW_K. It precedes the
    // C(32) × K(64) it ties on every term, that order being the whole decision between them.
    Spread{TILE_3X3_NARROW_K, {.waves_c = 1, .waves_k = 2, .waves_q = 4}}, // C(32) K(32)

    Spread{TILE_3X3, {.waves_c = 1, .waves_k = 2, .waves_q = 4}}, // C(32)  K(64)
    Spread{TILE_3X3, {.waves_c = 2, .waves_k = 1, .waves_q = 4}}, // C(64)  K(32)

    // ---- 3×3 at 32 channels per group, on the same K-narrow tile ----
    //
    // Halving the wave is not optional here: the full tile puts one group on one wave, and the
    // waves_g * waves_q of 8 that follows is over the LDS budget however the waves split.
    Spread{TILE_3X3_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 1, .waves_g = 4}}, // C(128) K(128), 4 groups
    Spread{TILE_3X3_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 2, .waves_g = 2}}, // C(64) K(64), 2 groups

    // ---- 3×3 at 16 channels per group, one whole group to a wave ----
    //
    // The pair repeats at every filter size and differs in nothing but reduction depth, so only
    // the write term separates them and it always prefers the shallower split.
    Spread{TILE_3X3_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 1, .waves_g = 8}}, // C(128) K(128), 8 groups
    Spread{TILE_3X3_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 2, .waves_g = 4}}, // C(64) K(64), 4 groups

    // ---- 3×1, on the C(64) × K(32) wave tile ----
    //
    // Three more blocks are over the LDS budget: C(64) × K(128) at two spatial items is 168 KiB,
    // C(256) × K(32) at two is 185 KiB, and every four-item spread is 192 KiB, against 160 KiB.
    // A fourth, C(256) × K(64), fits at 137 KiB and the prune sweep dropped it: the model sends
    // it no layer and it wins on none.
    Spread{TILE_3X1, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(128) K(128)
    Spread{TILE_3X1, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(128) K(64)

    // ---- 3×1 row-tiled, for an image past 2 GiB ----
    //
    // Same two blocks as above with the tile's row origin folded into the base, so the 32-bit
    // offset spans 18 rows rather than the whole image. They are complementary to the untiled
    // pair -- addressing_fits gives a shape to one or the other, never both.
    //
    // 18 rather than 16: rows_per_tile has to be a multiple of unroll(), which is 3 here, or a
    // block would straddle a boundary and address half its rows through the wrong base. At 18
    // the window is 20 S rows and 21 delta rows, which on the widest shape this serves --
    // a 57600-column row of 512 channels, 56.2 MiB -- is 1.15 GiB against the 2 GiB limit.
    Spread{TILE_3X1, {.waves_c = 2, .waves_k = 4, .waves_q = 1}, 0, 18}, // C(128) K(128)
    Spread{TILE_3X1, {.waves_c = 2, .waves_k = 2, .waves_q = 2}, 0, 18}, // C(128) K(64)

    // ---- 3×1's one narrow-tile entry, the four-item spread ----
    //
    // Halving C halves the block a spatial item costs, which is what buys waves_q = 4.
    Spread{TILE_3X1_NARROW_C, {.waves_c = 1, .waves_k = 2, .waves_q = 4}}, // C(32) K(64)

    // ---- 3×1 at 32 channels per group, on the K-narrow tile ----
    Spread{TILE_3X1_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 1, .waves_g = 4}}, // C(128) K(128), 4 groups
    Spread{TILE_3X1_NARROW_K,
           {.waves_c = 1, .waves_k = 2, .waves_q = 2, .waves_g = 2}}, // C(64) K(64), 2 groups

    // ---- 3×1 at 16 channels per group, one whole group to a wave ----
    Spread{TILE_3X1_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 1, .waves_g = 8}}, // C(128) K(128), 8 groups

    // ---- 5×1, on the C(32) × K(32) wave tile ----
    Spread{TILE_5X1, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(64) K(128)
    Spread{TILE_5X1, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(64) K(64)

    // ---- 4×4, on the C(32) × K(16) wave tile ----
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

    // ---- 4×4 at 32 channels per group, on the tile it already had ----
    //
    // The four-group C(128) × K(128) block is absent: over the widest wave any grouped entry
    // runs on, a C(128) block spills its epilogue.
    Spread{TILE_4X4,
           {.waves_c = 1, .waves_k = 2, .waves_q = 2, .waves_g = 2}}, // C(64) K(64), 2 groups

    // ---- 4×4 at 16 channels per group, one whole group to a wave ----
    Spread{TILE_4X4_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 1, .waves_g = 8}}, // C(128) K(128), 8 groups
    Spread{TILE_4X4_GROUP16,
           {.waves_c = 1, .waves_k = 1, .waves_q = 2, .waves_g = 4}}, // C(64) K(64), 4 groups

    // ---- 5×5, on the C(16) × K(16) wave tile ----
    //
    // Both axes sit at the swizzle's floor on a single wave, so only the arrangements leaving
    // two waves on each survive: three of the ten. The third, C(64) × K(32), spills its row
    // loop under every packing, only a C(32) S row folding a run of ladder rungs onto one
    // address.
    Spread{TILE_5X5, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(32) K(64)
    Spread{TILE_5X5, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(32) K(32)

    // ---- 5×5 at 16 channels per group, two groups to a workgroup ----
    //
    // The one grouped shape 5×5 can hold, and the only entry in the table under the cache line
    // the others fill: every wider window spills the row loop, for the reason above. A
    // 32-channel group needs no entry here, the ungrouped C(32) × K(32) block covering it
    // exactly. See direct-wgrad-config-table.md and the grouped section of direct-wgrad.md.
    Spread{TILE_5X5,
           {.waves_c = 1, .waves_k = 1, .waves_q = 4, .waves_g = 2}}, // C(32) K(32), 2 groups

    // ---- the tf32 entries ----
    //
    // A separate run rather than blocks interleaved with the 16-bit ones: fits_the_layer
    // partitions the table on the layer's data type, so no layer ever ranks an entry from both
    // and the order only has to hold within this run.
    //
    // Every filter but 5×5 takes the same three blocks, and any one of them covers every
    // ungrouped shape -- a layer narrower than a block pads, and the epilogue's bounds drop what
    // the padding computed. A C(16) × K(16) wave tile caps the widest block eight waves reach at
    // half a 16-bit one's area, so the list is the three shapes that area admits. Grouped entries
    // wait on the same budget.

    // ---- 3×3 at tf32, on the C(16) × K(16) wave tile ----
    Spread{TILE_3X3_TF32, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(32) K(64)
    Spread{TILE_3X3_TF32, {.waves_c = 4, .waves_k = 2, .waves_q = 1}}, // C(64) K(32)
    Spread{TILE_3X3_TF32, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(32) K(32)

    // ---- 2×2 at tf32, on the C(16) × K(16) wave tile ----
    Spread{TILE_2X2_TF32, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(32) K(64)
    Spread{TILE_2X2_TF32, {.waves_c = 4, .waves_k = 2, .waves_q = 1}}, // C(64) K(32)
    Spread{TILE_2X2_TF32, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(32) K(32)

    // ---- 3×1 at tf32, on the C(32) × K(16) wave tile the oblong filter affords ----
    Spread{TILE_3X1_TF32, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(64)  K(64)
    Spread{TILE_3X1_TF32, {.waves_c = 4, .waves_k = 2, .waves_q = 1}}, // C(128) K(32)
    Spread{TILE_3X1_TF32, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(64)  K(32)

    // ---- 3×1 at tf32, the same three blocks with the tile's row origin in the base ----
    //
    // tf32 needs this sooner than 16 bits does: window_bytes measures on the operand's
    // own width, so the same layer reaches the 2 GiB buffer window at half the image, and the
    // video stack's folded-depth shapes are what 3×1 is here for.
    //
    // 6 rather than 18: rows_per_tile has to be a multiple of unroll(), which is 3 here, and a
    // tf32 row is twice the bytes. 6 is the largest multiple whose window reaches as long a row as
    // the 16-bit tile18 entry's, so no row width that entry serves is out of this one's reach.
    Spread{TILE_3X1_TF32, {.waves_c = 2, .waves_k = 4, .waves_q = 1}, 0, 6}, // C(64)  K(64)
    Spread{TILE_3X1_TF32, {.waves_c = 4, .waves_k = 2, .waves_q = 1}, 0, 6}, // C(128) K(32)
    Spread{TILE_3X1_TF32, {.waves_c = 2, .waves_k = 2, .waves_q = 2}, 0, 6}, // C(64)  K(32)

    // ---- 4×4 at tf32, on the C(16) × K(16) wave tile ----
    Spread{TILE_4X4_TF32, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(32) K(64)
    Spread{TILE_4X4_TF32, {.waves_c = 4, .waves_k = 2, .waves_q = 1}}, // C(64) K(32)
    Spread{TILE_4X4_TF32, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(32) K(32)

    // ---- 5×5 at tf32, on the C(16) × K(16) wave tile ----
    //
    // Two of the three blocks, as at 16 bits: C(64) × K(32) measured slowest on every layer tried.
    Spread{TILE_5X5_TF32, {.waves_c = 2, .waves_k = 4, .waves_q = 1}}, // C(32) K(64)
    Spread{TILE_5X5_TF32, {.waves_c = 2, .waves_k = 2, .waves_q = 2}}, // C(32) K(32)
};

// Images packed into one column block of the MFMA reduction, in increasing order.
//
// An axis of its own: packing moves neither the tile nor the wave count, and every packing of
// every spread fits the LDS budget.
//
// preferred_unfold_n therefore decides among packings on padded pixels alone: the packed output
// columns × N rounded up to a multiple of the packing.
constexpr auto packings = std::array{1, 2, 4};

// Packings of the exchanged entries.
//
// 16 and 32 images (w_unfold 2 and 1) fill the MFMA's 32 reduction columns on the view of a 1×n
// layer one source row high, which is one column wide. preferred_unfold_n takes a multi-image
// packing here only where its window fits 32 bits, then decides on the padded pixels.
constexpr auto exchanged_packings = std::array{1, 2, 4, 16, 32};

// Configs a tiled spread contributes per write: one, because it packs no images.
//
// A tiled loader leaves the packed-image term out of the address, that term being the int which
// overflows on the shapes tiling exists for, so the entry is correct at a packing of one alone.
// The dispatch gate says the same, and kernel_test runs a config without consulting it.
constexpr std::size_t count_configs_per_write()
{
    std::size_t n = 0;
    for(const Spread& spread : spreads)
        n += spread.rows_per_tile > 0 ? 1 : packings.size();
    return n;
}

// Whether a spread has row-segmented configs, one for each packing.
//
// The untiled ungrouped spreads at 3×3, 3×1 and 5×5, 16-bit and tf32. Only these filter sizes
// have workload layers with too few column blocks to occupy every partition. A row-segmented
// config always splits, so only the split write has them. Computed from the fields because
// is_valid_config calls it for every entry.
//
// The 16-bit 3×3 spread with a C(64) × K(128) block on the C(32) × K(32) wave tile has none: its
// row-segmented partials kernel spills registers in the row loop, and its unsegmented one spills
// none. A row-segmented kernel that spills only where its unsegmented one does is kept, as the
// 5×5 tf32 kernels are.
constexpr bool takes_row_segments(int elem_bytes,
                                  int kh,
                                  int kw,
                                  int wave_c16,
                                  int wave_k16,
                                  const Arrangement& waves,
                                  int rows_per_tile)
{
    const bool filter = (kh == 3 && (kw == 3 || kw == 1)) || (kh == 5 && kw == 5);
    const bool spills = kh == 3 && kw == 3 && elem_bytes == 2 && wave_c16 == 2 && wave_k16 == 2 &&
                        waves.waves_c == 2 && waves.waves_k == 4 && waves.waves_q == 1;
    return filter && waves.waves_g == 1 && rows_per_tile == 0 && !spills;
}

constexpr bool takes_row_segments(const Spread& spread)
{
    return takes_row_segments(spread.tile.elem_bytes,
                              spread.tile.kh,
                              spread.tile.kw,
                              spread.tile.wave_c16,
                              spread.tile.wave_k16,
                              spread.waves,
                              spread.rows_per_tile);
}

constexpr bool has_row_segmented_config(const Config& cfg)
{
    const Arrangement waves{.waves_c = cfg.waves_c,
                            .waves_k = cfg.waves_k,
                            .waves_q = cfg.waves_q,
                            .waves_g = cfg.waves_g};
    return !cfg.no_splits && takes_row_segments(cfg.elem_bytes,
                                                cfg.kh,
                                                cfg.kw,
                                                cfg.wave_c16,
                                                cfg.wave_k16,
                                                waves,
                                                cfg.rows_per_tile);
}

constexpr std::size_t count_row_segmented_configs()
{
    std::size_t n = 0;
    for(const Spread& spread : spreads)
        n += takes_row_segments(spread) ? packings.size() : 0;
    return n;
}

// Whether a spread has H/W-exchanged configs, which run 1×N layers.
//
// Only the ungrouped, untiled N×1 spreads take it, row-segmented configs included. A tiled
// loader's window is a run of rows, which the exchange scatters across the image.
constexpr bool takes_exchange(const Spread& spread)
{
    return spread.tile.kw == 1 && spread.waves.waves_g == 1 && spread.rows_per_tile == 0;
}

constexpr std::size_t count_exchanged_configs()
{
    std::size_t n = 0;
    for(const Spread& spread : spreads)
        if(takes_exchange(spread))
            n += (takes_row_segments(spread) ? 3 : 2) * exchanged_packings.size();
    return n;
}

constexpr std::size_t count_configs()
{
    return 2 * count_configs_per_write() + count_row_segmented_configs() +
           count_exchanged_configs();
}

// Every spread crossed with the packings it takes, in the order the two tie-breaks need, then
// the same run again on the store epilogue, then the row-segmented configs.
//
// The spreads are filter-major already, so crossing the packings innermost keeps one filter
// shape's entries contiguous and leaves both rules reading the order they describe. The store
// run follows the whole split one rather than interleaving, which keeps every split entry at
// the index it had before the store entries existed; writes_fit_the_split gives a layer one
// twin of each pair, so no tie-break ever reads across the two runs.
constexpr auto make_configs()
{
    std::array<Config, count_configs()> configs{};

    const auto entry = [](const Spread& spread, int unfold_n, bool no_splits) {
        return Config{.kh            = spread.tile.kh,
                      .kw            = spread.tile.kw,
                      .elem_bytes    = spread.tile.elem_bytes,
                      .wave_c16      = spread.tile.wave_c16,
                      .wave_k16      = spread.tile.wave_k16,
                      .waves_c       = spread.waves.waves_c,
                      .waves_k       = spread.waves.waves_k,
                      .waves_q       = spread.waves.waves_q,
                      .waves_g       = spread.waves.waves_g,
                      .unfold_n      = unfold_n,
                      .prefetch_rows = spread.ring_depth(),
                      .rows_per_tile = spread.rows_per_tile,
                      .no_splits     = no_splits};
    };

    std::size_t out = 0;
    for(const bool no_splits : {false, true})
        for(const Spread& spread : spreads)
            for(const int unfold_n : packings)
            {
                if(spread.rows_per_tile > 0 && unfold_n != 1)
                    continue;
                configs[out++] = entry(spread, unfold_n, no_splits);
            }

    // After both runs, so every unsegmented config keeps its index. rows_fit_the_split admits one
    // config of each pair for a layer, so no tie-break compares the two.
    for(const Spread& spread : spreads)
    {
        if(!takes_row_segments(spread))
            continue;
        for(const int unfold_n : packings)
        {
            Config cfg       = entry(spread, unfold_n, false);
            cfg.row_segments = true;
            configs[out++]   = cfg;
        }
    }

    // Appended last, so every other config keeps its index. fits_the_layer gives a layer either
    // these or the rest, so no tie-break compares across them.
    for(const bool no_splits : {false, true})
        for(const Spread& spread : spreads)
        {
            if(!takes_exchange(spread))
                continue;
            for(const int unfold_n : exchanged_packings)
            {
                Config cfg      = entry(spread, unfold_n, no_splits);
                cfg.exchange_hw = true;
                configs[out++]  = cfg;
            }
        }
    for(const Spread& spread : spreads)
    {
        if(!takes_exchange(spread) || !takes_row_segments(spread))
            continue;
        for(const int unfold_n : exchanged_packings)
        {
            Config cfg       = entry(spread, unfold_n, false);
            cfg.row_segments = true;
            cfg.exchange_hw  = true;
            configs[out++]   = cfg;
        }
    }
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
        if(cfg.elem_bytes != 2 && cfg.elem_bytes != 4)
            return false;
        // The prologue primes the delta register ring from kh - 1 scratch rows, so a 1-row
        // filter leaves it nothing to declare: ScratchStage, DeltaScratch and prime_delta_ring
        // all size on kh - 1. Serving a 1×N entry means making the prologue degenerate at kh = 1.
        if(cfg.kh < 2)
            return false;
        if(cfg.acc_vgprs() + cfg.operand_vgprs() > TILE_VGPR_BUDGET)
            return false;
        if(cfg.block_c() < MIN_SWIZZLED_CHANS || cfg.block_k() < MIN_SWIZZLED_CHANS)
            return false;

        // A grouped entry exists to fill a cache line, so a window under one line defeats it.
        //
        // 128 bytes on each operand, which is 64 channels of fp16 and 32 of tf32. 5×5 is exempt,
        // every window that would fill the line spilling its row loop.
        if(cfg.waves_g > 1 && cfg.kh != 5 &&
           (cfg.block_c() * cfg.elem_bytes < 128 || cfg.block_k() * cfg.elem_bytes < 128))
            return false;

        // A tile has to be a whole number of unrolled blocks, or one would straddle a boundary
        // and address half its rows through the wrong base.
        if(cfg.rows_per_tile > 0 && cfg.rows_per_tile % cfg.unroll() != 0)
            return false;

        // And it packs no images; see count_configs.
        if(cfg.rows_per_tile > 0 && cfg.unfold_n != 1)
            return false;

        // An exchanged entry is N×1, and untiled because the exchange needs a window that reaches
        // the end of the image; see takes_exchange.
        if(cfg.exchange_hw && (cfg.kw != 1 || cfg.rows_per_tile > 0))
            return false;
    }
    return true;
}

static_assert(table_is_well_formed(),
              "a direct_wgrad config is malformed: waves_c * waves_k * waves_q * waves_g must "
              "be WAVES_PER_WORKGROUP, so a wave given to the spatial axis or to a second "
              "group has to come off C or K; unfold_n must divide the MFMA's column count; "
              "elem_bytes must be 2 or 4; kh must be at least 2; the gradient and the operands "
              "must fit TILE_VGPR_BUDGET, which the wave tile has to shrink for as the filter "
              "grows; both channel blocks must reach MIN_SWIZZLED_CHANS; a grouped entry's "
              "window must reach a whole cache line; a row-tiled entry must span a whole "
              "number of unrolled blocks and pack no images; and an exchanged entry must be "
              "Nx1 and untiled");

} // namespace hipconv::cdna4::direct_wgrad
