#pragma once

// The work partition: which cells exist, and which workgroup runs each.
//
// A cell is one channel tile, (group, C block, K block), crossed with a row segment of the image
// and a column split, a run of the flattened (image, column block) spatial axis. The channel tile
// is the outer axis and the (row segment, column split) the inner one, because the accumulators
// are the tile's output and have to live across the whole reduction.
//
// FlatGrid defines the cells, CellWalk the traversal that decides the cache locality. See the
// grid and grid traversal sections of docs/algorithms/direct/direct-wgrad.md.
//
// Host arithmetic on the layer's shape, with no device code, so the partition is testable
// without a launch.

#include "config.h"
#include "mathutil.h"
#include "persistent_grid.h"

#include "hipconv/conv_params.hpp"

#include <cstdint>

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

// One item of the spatial axis: the cell's row segment at one column window.
// Under the batch unfold it covers unfold_n consecutive images, and `image` names the first.
struct SpatialItem
{
    int image;
    int col_block;
};

// The shortest row segment the grid cuts an image into.
//
// Each segment repeats the prologue and the kh - 1 halo delta rows per item, and a shorter segment
// amortizes them over fewer rows.
inline constexpr int MIN_SEG_ROWS = 32;

// The most row segments an image of `rows` rows takes.
//
// Each holds at least MIN_SEG_ROWS rows, and rows / s >= s - 1 keeps rounding seg_rows() up from
// putting the last segment's first row past the image. The second bound holds up to about
// sqrt(rows) + 1, so the count starts there and steps down. See
// docs/algorithms/details/direct/direct-wgrad-row-segments.md.
constexpr int most_row_segs(int rows)
{
    int s = 1;
    while(int64_t{s + 1} * (s + 1) <= rows)
        s *= 2;
    for(int step = s / 2; step > 0; step /= 2)
        if(int64_t{s - step} * (s - step) > rows)
            s -= step;
    // s is now within one of sqrt(rows); try one above it, then step down to the bound.
    s += 1;
    while(s > 1 && rows / s < s - 1)
        --s;
    return maximum(1, minimum(s, rows / MIN_SEG_ROWS));
}

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

    // S rows, which row segments divide.
    int rows = 0;

    // Partitions per workgroup, each running its own item.
    int waves_q = 1;

    // Config::row_segments. When false, every function below computes the unsegmented partition.
    bool row_split = false;

    // Delta rows each segment past the first loads again above its first row: kh - 1.
    int halo_rows = 0;

    // The segment count wanted_row_segs() picks: flat_grid works it out on the host and the kernel
    // receives it, so no device code runs the search.
    int row_seg_count = 1;

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

    // Column splits at `segs` row segments, the same rule as col_splits().
    constexpr int col_splits_at(int segs) const
    {
        return minimum(maximum(1, workgroups / (tiles() * segs)), items());
    }

    // Rows one partition runs at `segs` row segments: the rounds of cells a workgroup takes, times
    // the rounds of items a cell's partitions take, times one segment's rows and halo.
    //
    // 64-bit because a tall image's rows times the rounds, or the search's products with 9 and 10,
    // can pass INT32_MAX.
    constexpr int64_t partition_rows(int segs) const
    {
        const int cols   = col_splits_at(segs);
        const int cells  = tiles() * segs * cols;
        const int rounds = divup(divup(items(), cols), waves_q);
        return int64_t{divup(cells, workgroups)} * rounds *
               (divup(rows, segs) + (segs > 1 ? halo_rows : 0));
    }

    // Row segments per image: the count that gives a partition the fewest rows or, with one column
    // split, the fewest that give every partition an item. See
    // docs/algorithms/details/direct/direct-wgrad-row-segments.md.
    constexpr int wanted_row_segs() const
    {
        const int most = most_row_segs(rows);
        if(col_splits_at(1) <= 1)
        {
            const int want = maximum(1, workgroups / tiles()) * waves_q;
            if(items() >= want)
                return 1;
            return maximum(1, minimum(divup(want, items()), most));
        }
        // Past workgroups / tiles segments the cells no longer fit one round.
        int best = 1;
        for(int segs = 2; segs <= most && tiles() * segs <= workgroups; ++segs)
        {
            if(partition_rows(segs) < partition_rows(best))
                best = segs;
        }
        return partition_rows(best) * 10 <= partition_rows(1) * 9 ? best : 1;
    }

    constexpr int row_segs() const { return row_split ? row_seg_count : 1; }

    // Rows per segment. The last segment's rows past the image read zero.
    //
    // Uniform across segments because the kernel builds its row schedule once per launch, outside
    // the cell loop.
    constexpr int seg_rows() const { return divup(rows, row_segs()); }

    // Workgroups sharing one channel tile at one row segment: enough to give every workgroup a
    // cell, capped at the number of items to hand out, and 1 once tiles() × row_segs() alone
    // outnumber half the workgroups.
    constexpr int col_splits() const { return col_splits_at(row_segs()); }

    // Workgroups sharing one channel tile.
    constexpr int splits() const
    {
        if(!row_split)
            return minimum(maximum(1, workgroups / tiles()), items());
        return row_segs() * col_splits();
    }

    constexpr int cells() const { return tiles() * splits(); }

    // Whether several workgroups hold partial gradients of one dW element.
    //
    // Unsplit, each element has one writer, which is what a no_splits entry needs; see
    // writes_fit_the_split.
    constexpr bool reduces() const { return splits() > 1; }

    // Entries per segment, rounded up, so the last segments can come out short or empty.
    constexpr int items_per_split() const
    {
        return divup(items(), row_split ? col_splits() : splits());
    }

    // Column splits holding at least one item, per row segment.
    constexpr int live_col_splits() const { return divup(items(), items_per_split()); }

    // Segments holding at least one entry: the ones a cell runs, and so the ones that write a
    // partial gradient.
    constexpr int live_splits() const
    {
        if(!row_split)
            return divup(items(), items_per_split());
        return row_segs() * live_col_splits();
    }

    // fp32 roundings on the longest path from a product to one dW element.
    //
    // The tolerance's accumulation depth, which the blocking puts orders of magnitude below the
    // N*P*Q products summed. `iterations` is RowSchedule::iterations(); rows the padding leaves
    // empty add zero and cannot round, so this is an upper bound. Derivation in
    // docs/algorithms/details/direct/direct-wgrad-tolerance.md.
    constexpr size_t
    accumulation_depth(int iterations, int mfma_k, int exact_block, int waves_q) const
    {
        // srcC is the accumulator's own step, so `chain` counts it, once per mma.
        const size_t within_mfma = static_cast<size_t>(mfma_k / exact_block) - 1;
        const size_t chain       = static_cast<size_t>(items_per_split()) * iterations;
        const size_t staging     = static_cast<size_t>(waves_q) - 1;
        // live_splits() - 1 bounds the roundings of reduce_partials_cdna4, which may sum
        // interleaved groups and combine them through LDS; a no_splits entry has one partial, so
        // this is 0.
        const size_t reduction = static_cast<size_t>(live_splits()) - 1;
        return within_mfma + chain + staging + reduction;
    }

    // ---- the decode ----
    //
    // Cell c takes tile c % tiles() and split c / tiles(), so the workgroups co-active at any
    // moment differ in channel tile and share the spatial region, which an L2 blocking schedule
    // then refines. Split s is column split s % col_splits() of row segment s / col_splits().

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
        const int split = row_split ? cell / tiles() % col_splits() : cell / tiles();
        const int begin = minimum(split * per, items());
        return {begin, minimum(begin + per, items())};
    }

    constexpr int row_seg(int cell) const { return row_split ? cell / tiles() / col_splits() : 0; }

    // Where a live cell's partial goes: dense over the live column splits of every row segment, so
    // the reduction sums exactly live_splits() of them.
    constexpr int partial_index(int cell) const
    {
        const int split = cell / tiles();
        if(!row_split)
            return split;
        return split / col_splits() * live_col_splits() + split % col_splits();
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

// The FlatGrid for a launch of `workgroups` workgroups, with row_seg_count left at 1.
//
// The kernel calls this and takes row_seg_count from the host; flat_grid calls it and runs the
// search, so both compute the same partition.
constexpr FlatGrid grid_shape(const Config& cfg,
                              int groups,
                              int c_per_group,
                              int k_per_group,
                              int images,
                              int rows,
                              int out_cols,
                              int workgroups)
{
    FlatGrid grid{.groups           = groups,
                  .c_per_group      = c_per_group,
                  .k_per_group      = k_per_group,
                  .images           = images,
                  .out_cols         = out_cols,
                  .block_c          = cfg.group_c(),
                  .block_k          = cfg.group_k(),
                  .groups_per_block = cfg.waves_g,
                  .block_cols       = cfg.w_unfold(),
                  .unfold_n         = cfg.unfold_n,
                  .workgroups       = workgroups,
                  .rows             = rows,
                  .waves_q          = cfg.waves_q,
                  .row_split        = cfg.row_segments,
                  .halo_rows        = cfg.kh - 1};
    return grid;
}

// The grid with its row segment count, which the host works out once and passes to the kernel:
// the search is a chain of divisions the kernel would otherwise run in every workgroup.
constexpr FlatGrid flat_grid(const Config& cfg,
                             int groups,
                             int c_per_group,
                             int k_per_group,
                             int images,
                             int rows,
                             int out_cols,
                             int workgroups)
{
    FlatGrid grid =
        grid_shape(cfg, groups, c_per_group, k_per_group, images, rows, out_cols, workgroups);
    if(grid.row_split)
        grid.row_seg_count = grid.wanted_row_segs();
    return grid;
}

// The grid of `par` for a launch of `workgroups` workgroups.
inline FlatGrid flat_grid(const ConvParams& par, const Config& cfg, int workgroups)
{
    return flat_grid(cfg,
                     par.groups,
                     par.channels_per_group(),
                     par.filters_per_group(),
                     par.n,
                     par.h,
                     par.q,
                     workgroups);
}

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
    // to be had. Only a test launches such a grid; the persistent grid is 8 × 32.
    if(workgroups % persistent::NUM_XCD != 0)
        return {block, cells, workgroups};

    const int per_xcd = divup(cells, persistent::NUM_XCD);
    const int xcd     = block % persistent::NUM_XCD;
    const int cu      = block / persistent::NUM_XCD;

    return {
        xcd * per_xcd + cu, minimum(cells, (xcd + 1) * per_xcd), workgroups / persistent::NUM_XCD};
}

} // namespace hipconv::cdna4::direct_wgrad
