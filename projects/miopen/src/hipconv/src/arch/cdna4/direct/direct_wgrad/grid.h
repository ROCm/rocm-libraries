#pragma once

// The work partition: which cells exist, and which workgroup runs each.
//
// A cell is one channel tile, (group, C block, K block), crossed with a segment of the flattened
// (image, column block) spatial axis. The channel tile is the outer axis and the spatial segment
// the inner one, because the accumulators are the tile's output and have to live across the whole
// reduction.
//
// FlatGrid defines the cells, CellWalk the traversal that decides the cache locality. See the
// grid and grid traversal sections of docs/algorithms/direct/direct-wgrad.md.
//
// Host arithmetic on the layer's shape, with no device code, so the partition is testable
// without a launch.

#include "mathutil.h"
#include "persistent_grid.h"

namespace hipconv::cdna4::direct_wgrad
{

// The gradient tile a cell accumulates.
//
// c_origin is measured inside the group, because dW's contiguous extent is one group's channels;
// k_origin across every group, because the filter index spans them. group is the first of the
// groups_per_block consecutive groups the tile covers, so its channel window is one contiguous
// run that the waves of each group slice.
struct ChannelTile
{
    int group;
    int c_origin;
    int k_origin;
};

// A half-open run of the spatial axis, [begin, end).
struct ItemRange
{
    int begin;
    int end;

    constexpr bool empty() const { return begin >= end; }
};

// One entry of the spatial axis: a whole image height at one column window.
// Under the batch unfold it covers unfold_n consecutive images, and `image` names the first.
struct SpatialItem
{
    int image;
    int col_block;
};

struct FlatGrid
{
    // ---- the layer ----

    int groups;
    int c_per_group;
    int k_per_group;
    int images;

    // Columns of the output gradient, the extent the spatial axis blocks up.
    int out_cols;

    // ---- the tile the config defines ----
    //
    // Both blocks are per group: a workgroup covering several groups blocks each group's own
    // axis up by these and reads groups_per_block times as many channels.

    int block_c;
    int block_k;

    // Groups one channel tile covers, 1 unless the layer's group is too narrow to fill a cache
    // line on its own.
    int groups_per_block = 1;

    // Output columns one packed image contributes to a block, the whole MFMA K without the
    // unfold, and the extent the spatial axis blocks the width up by.
    int block_cols;

    // Images one entry of the spatial axis packs into a block.
    int unfold_n = 1;

    // Workgroups the launch provides, which bounds how wide the reduction splits.
    int workgroups;

    // ---- the partition ----

    // Channel blocks per group. The last one may run past the real channel count, and the
    // epilogue's guards keep those columns out of dW.
    constexpr int c_blocks() const { return divup(c_per_group, block_c); }
    constexpr int k_blocks() const { return divup(k_per_group, block_k); }

    // Group blocks. A layer whose groups the block does not divide leaves the last one short,
    // and the waves holding the groups past the end are bounded out of dW like the channel
    // padding.
    constexpr int group_blocks() const { return divup(groups, groups_per_block); }

    // Channel tiles, and entries on the spatial axis.
    //
    // An entry names unfold_n images, so a batch the packing does not divide leaves the last
    // entry short and the loader reads its missing images as zero.
    constexpr int tiles() const { return group_blocks() * c_blocks() * k_blocks(); }
    constexpr int col_blocks() const { return divup(out_cols, block_cols); }
    constexpr int image_groups() const { return divup(images, unfold_n); }
    constexpr int items() const { return image_groups() * col_blocks(); }

    // Workgroups sharing one channel tile: enough to give every workgroup a tile, capped at the
    // number of entries to hand out, and 1 once the tiles alone outnumber the workgroups.
    constexpr int splits() const { return minimum(maximum(1, workgroups / tiles()), items()); }

    constexpr int cells() const { return tiles() * splits(); }

    // Entries per segment, rounded up, so the last segments can come out short or empty.
    constexpr int items_per_split() const { return divup(items(), splits()); }

    // fp32 roundings on the longest path from a product to one dW element.
    //
    // The tolerance's accumulation depth, which the blocking puts orders of magnitude below the
    // N*P*Q products summed. `iterations` is RowSchedule::iterations(); rows the padding leaves
    // empty add zero and cannot round, so this is an upper bound. Derivation in
    // docs/algorithms/direct/direct-wgrad-tolerance.md.
    constexpr size_t accumulation_depth(int iterations, int mfma_k, int exact_block) const
    {
        // srcC is the accumulator's own step, so `chain` counts it, once per mma.
        const size_t within_mfma = static_cast<size_t>(mfma_k / exact_block) - 1;
        const size_t chain       = static_cast<size_t>(items_per_split()) * iterations;
        const size_t staging     = 1;
        const size_t atomics     = static_cast<size_t>(splits()) - 1;
        return within_mfma + chain + staging + atomics;
    }

    // ---- the decode ----
    //
    // Cell c takes tile c % tiles() and segment c / tiles(), so the workgroups co-active at any
    // moment differ in channel tile and share the spatial region, which an L2 blocking schedule
    // then refines.

    constexpr ChannelTile tile(int cell) const
    {
        const int t    = cell % tiles();
        const int k    = t % k_blocks();
        const int rest = t / k_blocks();
        const int c    = rest % c_blocks();
        const int g    = rest / c_blocks() * groups_per_block;
        return {g, c * block_c, g * k_per_group + k * block_k};
    }

    constexpr ItemRange segment(int cell) const
    {
        const int per   = items_per_split();
        const int begin = minimum((cell / tiles()) * per, items());
        return {begin, minimum(begin + per, items())};
    }

    // Where entry x sits.
    //
    // The main loop walks a segment by incrementing, so this runs once per cell to seed it. The
    // image is the entry's first, so it steps by unfold_n.
    constexpr SpatialItem item(int x) const
    {
        return {x / col_blocks() * unfold_n, x % col_blocks()};
    }
};

// The cells one workgroup runs, in the order it runs them.
//
// Cells are laid out tile-fastest, so a contiguous run of them is a run of channel tiles at one
// spatial segment: the same rows of S and delta, read by every tile in the run. One contiguous
// run per XCD therefore puts its workgroups on shared rows and keeps the reuse inside one L2.
//
// The run is a permutation of the cell space, so coverage holds however it is ordered. Which XCD
// a block lands on is an assumption (see persistent_grid.h) whose failure costs hit rate only.
struct CellWalk
{
    int begin;
    int end;
    int stride;
};

// Assign `block` of `workgroups` its run over `cells`.
//
// Each XCD takes a contiguous run of ceil(cells / NUM_XCD) and its own workgroups stride through
// it, so a run longer than one round advances the whole XCD together and the next round is
// another contiguous rectangle. A workgroup whose first cell is already past its XCD's run has
// nothing to do, which is how a cell space narrower than the launch spreads over all the XCDs.
constexpr CellWalk plan_cell_walk(int cells, int workgroups, int block)
{
    // Stripe when the workgroups do not divide evenly among the XCDs, since there is no grouping
    // to be had. Only a test launches such a grid; the persistent grid is 8 x 32.
    if(workgroups % persistent::NUM_XCD != 0)
        return {block, cells, workgroups};

    const int per_xcd = divup(cells, persistent::NUM_XCD);
    const int xcd     = block % persistent::NUM_XCD;
    const int cu      = block / persistent::NUM_XCD;

    return {
        xcd * per_xcd + cu, minimum(cells, (xcd + 1) * per_xcd), workgroups / persistent::NUM_XCD};
}

} // namespace hipconv::cdna4::direct_wgrad
