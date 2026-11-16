#pragma once

// The direct_wgrad device kernel, and the host selection that gives a layer one config.
//
// docs/algorithms/direct/direct-wgrad.md is the algorithm, and
// docs/algorithms/details/direct/direct-wgrad-estimated-cost.md the model the selection ranks by.

#include "config.h"
#include "config_desc.h"
#include "config_table.h"
#include "epilogue.h"
#include "grid.h"
#include "lds_layout.h"
#include "main_loop.h"
#include "operand_loader.h"
#include "partials.h"
#include "perf_model.h"
#include "prologue.h"
#include "row_loader.h"
#include "row_schedule.h"
#include "workgroup_tiles.h"
#include "bunnies.hpp"
#include "bunnies_cdna4.hpp"
#include "conv_kernel.h"
#include "direct_conv_kernel.h"
#include "hip_util.h"
#include "launch_params.h"
#include "persistent_grid.h"
#include "types.h"
#include "hipconv/conv_params.hpp"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdlib>

namespace hipconv::cdna4::direct_wgrad
{
// The tensor shape, in a form the compiler cannot hoist out of the loop that reads it.
__device__ inline RowTensorPars volatile_pars(const RowTensorPars& pars)
{
    return {bunnies::pin_sgpr(pars.images),
            bunnies::pin_sgpr(pars.rows),
            bunnies::pin_sgpr(pars.cols),
            bunnies::pin_sgpr(pars.chans)};
}

// A copy of the grid whose decode cannot be hoisted out of the cell loop.
__device__ inline FlatGrid volatile_grid(const FlatGrid& g)
{
    FlatGrid out    = g;
    out.groups      = bunnies::pin_sgpr(g.groups);
    out.c_per_group = bunnies::pin_sgpr(g.c_per_group);
    out.k_per_group = bunnies::pin_sgpr(g.k_per_group);
    out.images      = bunnies::pin_sgpr(g.images);
    out.out_cols    = bunnies::pin_sgpr(g.out_cols);
    if(g.row_split)
    {
        out.rows          = bunnies::pin_sgpr(g.rows);
        out.row_seg_count = bunnies::pin_sgpr(g.row_seg_count);
    }
    return out;
}

// The four knobs the sweep steers the dispatch with, zero meaning unpinned.
//
// Each filters the candidates rather than naming an entry, so the ranking still runs over
// whatever survives and the four compose. See the re-fitting section of
// direct-wgrad-estimated-cost.md.
//
// One instance for the process, read on first use: is_valid_config asks for the knobs once per
// table entry, so reading the environment on each would put four getenv calls on the dispatch of
// every entry of every layer, to answer a question a launched process cannot change the answer to.
class Pins
{
public:
    static Pins& instance()
    {
        static Pins pins;
        return pins;
    }

    Pins(const Pins&)            = delete;
    Pins& operator=(const Pins&) = delete;

    int unfold_n() const { return unfold_n_; }

    // Whether this entry carries the channel block and group count pinned.
    bool match(const Config& cfg) const
    {
        return (block_c_ == 0 || cfg.block_c() == block_c_) &&
               (block_k_ == 0 || cfg.block_k() == block_k_) &&
               (waves_g_ == 0 || cfg.waves_g == waves_g_);
    }

    // Read the environment again, for a test that moves it under a running process.
    //
    // Not safe to race against a dispatch on another thread, which is why nothing outside a test
    // calls it.
    void reload() { read(); }

private:
    Pins() { read(); }

    void read()
    {
        block_c_  = knob("HIPCONV_WGRAD_BLOCK_C");
        block_k_  = knob("HIPCONV_WGRAD_BLOCK_K");
        waves_g_  = knob("HIPCONV_WGRAD_WAVES_G");
        unfold_n_ = knob("HIPCONV_WGRAD_UNFOLD_N");
    }

    // An integer pinned in the environment, or 0 for absent and unparseable alike.
    // Zero is never a valid setting for any of the knobs, so it doubles as "no override".
    static int knob(const char* name)
    {
        const char* value = std::getenv(name);
        return value ? std::atoi(value) : 0;
    }

    int block_c_  = 0;
    int block_k_  = 0;
    int waves_g_  = 0;
    int unfold_n_ = 0;
};

// Bytes one loader's window spans, per operand, at the given row count.
//
// A row count of zero asks for the whole image, which is what an untiled config addresses from
// one base.
struct WindowBytes
{
    int64_t s;
    int64_t delta;
};

// The window at `images` packed images; see window_bytes.
inline WindowBytes
window_bytes_at(const ConvParams& par, int64_t images, int s_rows, int delta_rows, int lead_rows)
{
    // The operands' own width, which is 4 for tf32: what has to fit 32 bits is the buffer
    // descriptor's window, and that is measured on the fp32 the loader fetches.
    const int64_t elem_bytes = static_cast<int64_t>(sizeof_data_type(par.input_type));
    const int64_t s_row      = int64_t{par.w} * par.groups * par.channels_per_group() * elem_bytes;
    const int64_t d_row      = int64_t{par.q} * par.groups * par.filters_per_group() * elem_bytes;
    const int64_t s_count    = s_rows > 0 ? s_rows : images * par.h;
    const int64_t d_count    = delta_rows > 0 ? delta_rows : images * par.p + lead_rows;
    return {s_count * s_row, d_count * d_row};
}

// How many images this layer should pack into a column block.
//
// Of the packings an entry may take, the one that runs the fewest padded output pixels, ties to
// the smaller.
//
// divup(n, u) * u images run where n exist, which outweighs the column padding on a small
// batch. padded_pixels less the row factor, common to every packing.
//
// HIPCONV_WGRAD_UNFOLD_N pins the answer when it names a packing the entry may take, and is
// ignored otherwise. Over the packings rather than the table: every untiled spread carries all of
// `packings`, every spread that takes_exchange carries all of exchanged_packings in its
// exchange_hw entries, and the column block's width follows the packing alone, so the table has
// nothing to add and is_valid_config asks this once per entry. An exchanged entry may take a
// multi-image packing from exchanged_packings only where its window fits 32 bits, counting the
// kh - 1 halo rows a row-segmented config adds, because exchanged entries have no tiled fallback.
inline int preferred_unfold_n(const ConvParams& par, bool exchanged = false)
{
    const auto fits = [&](int unfold_n) {
        if(!exchanged || unfold_n == 1)
            return true;
        constexpr int64_t int32_max = 0x7fffffff;
        const WindowBytes w         = window_bytes_at(par, unfold_n, 0, 0, par.kh - 1);
        return w.s <= int32_max && w.delta <= int32_max;
    };
    const auto choose = [&](const auto& candidates) {
        const int pinned = Pins::instance().unfold_n();
        for(const int unfold_n : candidates)
            if(unfold_n == pinned && fits(unfold_n))
                return pinned;

        int chosen     = 0;
        int64_t fewest = 0;
        for(const int unfold_n : candidates)
        {
            if(!fits(unfold_n))
                continue;
            // Strictly fewer, so a tie leaves the smaller packing in place.
            // The table lists the packings in increasing order.
            //
            // Rows are the same factor under every packing, so they cancel and this is
            // padded_pixels without them.
            const int64_t pixels =
                int64_t{divup(par.n, unfold_n)} * unfold_n * packed_columns(par, unfold_n);
            if(chosen == 0 || pixels < fewest)
            {
                fewest = pixels;
                chosen = unfold_n;
            }
        }
        return chosen;
    };
    return exchanged ? choose(exchanged_packings) : choose(packings);
}

// Whether this entry may run this layer at all.
//
// Every entry with waves_g == 1 may: a block wider than the tensor pads, and the epilogue's bounds
// drop what the padding computed. A grouped entry reads waves_g whole groups as one contiguous
// run, so it admits no padding on either channel axis, and the group count has to divide or a
// short last block leaves waves writing past dW. The same tests keep it off a dense layer, where
// it would compute the block diagonal of a tile with no block structure.
inline bool serves(const ConvParams& par, const Config& cfg)
{
    if(cfg.waves_g == 1)
        return true;
    return par.groups % cfg.waves_g == 0 && par.channels_per_group() == cfg.group_c() &&
           par.filters_per_group() == cfg.group_k();
}

// Whether an exchange_hw entry runs this layer: a 1×N filter.
inline bool exchanges(const ConvParams& par)
{
    return par.dims == 2 && par.kh == 1 && par.kw > 1;
}

// The N×1 layer over a 1×N layer's H/W-exchanged view.
//
// See docs/algorithms/details/direct/direct-wgrad-exchange-hw.md.
inline ConvParams exchanged(const ConvParams& par)
{
    ConvParams view = par;
    std::swap(view.h, view.w);
    std::swap(view.p, view.q);
    std::swap(view.kh, view.kw);
    std::swap(view.pad_h, view.pad_w);
    std::swap(view.stride_h, view.stride_w);
    std::swap(view.dilation_h, view.dilation_w);
    return view;
}

// The layer an entry computes: the exchanged view for an exchange_hw entry, the layer otherwise.
inline ConvParams layer_view(const ConvParams& par, const Config& cfg)
{
    return cfg.exchange_hw ? exchanged(par) : par;
}

// Whether this entry's data type, filter and group shape fit the layer, before the packing
// narrows further.
//
// Field compares only. is_valid_config leads with these because they reject most of the table,
// and what follows them is per-layer work that would otherwise run once per entry.
inline bool fits_the_layer(const ConvParams& par, const Config& cfg)
{
    // The two tables are partitioned on the operand's type, since sizeof is 4 for fp32 as well
    // as tf32.
    if((par.input_type == DataType::tf32) != cfg.is_tf32())
        return false;
    if(cfg.exchange_hw != exchanges(par))
        return false;
    const ConvParams view = layer_view(par, cfg);
    return cfg.kh == view.kh && cfg.kw == view.kw && serves(view, cfg);
}

// Whether this entry is one of the layer's candidates, before the pins narrow them.
inline bool is_a_candidate(const ConvParams& par, const Config& cfg, int unfold_n)
{
    return fits_the_layer(par, cfg) && cfg.unfold_n == unfold_n;
}

// Whether the pins name a candidate of this layer at all.
//
// When they name none, every caller below ignores them and proceeds as if none were set, so a
// mistyped knob does not look like a missing kernel.
inline bool the_pins_name_a_candidate(const ConvParams& par, int unfold_n)
{
    for(const Config& c : configs)
        if(is_a_candidate(par, c, unfold_n) && Pins::instance().match(c))
            return true;
    return false;
}

// lead_rows: the kh - 1 halo rows that a row-segmented config's delta base sits above the image;
// see RowLoader.
//
// Counts all unfold_n image slots, past N included: RowLoader forms a slot's offset in a 32-bit
// int, and a slot past N reads zero only if that offset does not wrap into the window.
inline WindowBytes window_bytes(const ConvParams& par,
                                int s_rows,
                                int delta_rows,
                                int lead_rows  = 0,
                                bool exchanged = false)
{
    return window_bytes_at(par, preferred_unfold_n(par, exchanged), s_rows, delta_rows, lead_rows);
}

// Whether the images one loader spans fit in 32 bits, which is what an untiled config needs.
inline bool window_fits_a_buffer(const ConvParams& par, int lead_rows = 0, bool exchanged = false)
{
    constexpr int64_t int32_max = 0x7fffffff;
    const WindowBytes w         = window_bytes(par, 0, 0, lead_rows, exchanged);
    return w.s <= int32_max && w.delta <= int32_max;
}

// Whether one tile fits instead, for a shape whose whole image does not.
//
// Rejects every shape window_fits_a_buffer accepts, and every shape whose preferred_unfold_n is
// above 1, so a shape whose window at that packing exceeds 2 GiB fits neither. A tiled config runs
// at unfold_n 1, which is what keeps the packed-image term, and the int it would overflow, out of
// the address; see RowLoader.
inline bool tile_fits_a_buffer(const ConvParams& par, const Config& cfg)
{
    constexpr int64_t int32_max = 0x7fffffff;
    if(window_fits_a_buffer(par))
        return false;
    if(preferred_unfold_n(par) != 1)
        return false;
    const WindowBytes w = window_bytes(par,
                                       cfg.rows_per_tile + cfg.prefetch_rows,
                                       cfg.rows_per_tile + par.pad_h + cfg.prefetch_rows);
    return w.s <= int32_max && w.delta <= int32_max;
}

// Whether this entry can address the layer at all, tiled or not.
inline bool addressing_fits(const ConvParams& par, const Config& cfg)
{
    if(cfg.rows_per_tile > 0)
        return tile_fits_a_buffer(par, cfg);
    return window_fits_a_buffer(par, cfg.row_segments ? cfg.kh - 1 : 0, cfg.exchange_hw);
}

// One workgroup per CU, so the grid shape does not encode the problem dimensions.
inline LaunchParams get_launch_params(const Config& cfg)
{
    LaunchParams launch;
    launch.grid                 = dim3(persistent::PERSISTENT_GRID_SIZE, 1, 1);
    launch.block_size           = dim3(cfg.threads(), 1, 1);
    launch.dynamic_shared_bytes = 0;
    return launch;
}

// Builds on the host the FlatGrid that the kernel builds at launch.
//
// The width comes from get_launch_params, so a change to the launch width reaches
// writes_fit_the_split and the tolerance too.
inline FlatGrid flat_grid(const ConvParams& par, const Config& cfg)
{
    return flat_grid(par, cfg, static_cast<int>(get_launch_params(cfg).grid.x));
}

// Whether this entry's write suits the layer's partition.
//
// A no_splits entry stores, which is correct only where each dW element has one writer. Its
// split twin, which stores partials for a reduction, is correct everywhere, and is turned away
// where nothing splits because the direct store is faster. So a layer ranks one twin of each pair.
// Like `serves`, a rule rather than a preference, and kernel_test applies it too.
inline bool writes_fit_the_split(const ConvParams& par, const Config& cfg)
{
    return cfg.no_splits == !flat_grid(par, cfg).reduces();
}

// Whether this entry's rows suit the layer's partition.
//
// A row-segmented config runs the layers whose column blocks leave waves_q partitions without an
// item, and its unsegmented config runs the rest. Like writes_fit_the_split, a hard constraint, so
// only one config of each pair reaches a layer's ranking.
//
// The row-segmented config's delta window also spans the kh - 1 halo rows above the image, so a
// layer whose window fits 32 bits only without them keeps the unsegmented config.
inline bool rows_fit_the_split(const ConvParams& par, const Config& cfg)
{
    if(!cfg.row_segments && !has_row_segmented_config(cfg))
        return true;
    const bool segmented = flat_grid(par, cfg).wanted_row_segs() > 1 &&
                           window_fits_a_buffer(par, cfg.kh - 1, cfg.exchange_hw);
    return cfg.row_segments == segmented;
}

// Workspace bytes: a split entry's partial gradients, which reduce_partials_cdna4 sums into dW.
//
// Sized at the persistent grid's width, which bounds the partials of any narrower launch.
inline size_t get_workspace_size(const ConvParams& par, const Config& cfg)
{
    if(cfg.no_splits)
        return 0;
    const FlatGrid grid = flat_grid(par, cfg);
    return partial_layout(grid, cfg.kh, cfg.kw).part_vec4() * sizeof(fp32x4_t) *
           partial_count(grid);
}

// Whether this entry may run this layer, which several entries of a layer may.
//
// Which of them the dispatcher prefers is the ranking's answer, not this one. A launch reaches
// this through ConvLaunch::make, which admits any entry the ranking offered.
inline bool is_valid_config(const ConvParams& par, const Config& cfg)
{
    if(!fits_the_layer(par, cfg))
        return false;
    const ConvParams view = layer_view(par, cfg);
    if(!addressing_fits(view, cfg))
        return false;
    const int unfold_n = preferred_unfold_n(view, cfg.exchange_hw);
    if(cfg.unfold_n != unfold_n)
        return false;
    if(!writes_fit_the_split(view, cfg))
        return false;
    if(!rows_fit_the_split(view, cfg))
        return false;
    return Pins::instance().match(cfg) || !the_pins_name_a_candidate(par, unfold_n);
}

// fp32 roundings on the longest path from a product to one dW element.
//
// See docs/algorithms/details/direct/direct-wgrad-tolerance.md. The main loop runs over the S
// rows, so RowSchedule::iterations() is the chain length.
inline size_t accumulation_depth(const ConvParams& par, const Config& cfg)
{
    const FlatGrid grid = flat_grid(par, cfg);
    const RowSchedule sched{.kh            = cfg.kh,
                            .prefetch_rows = cfg.prefetch_rows,
                            .pad_h         = par.pad_h,
                            .s_rows        = grid.seg_rows(),
                            .row_buffers   = cfg.row_buffers()};
    return grid.accumulation_depth(
        sched.iterations(), MFMA_K, bunnies::arch_cdna4::mfma_f16_f16_f32_exact_block, cfg.waves_q);
}

// Partials has each cell store its tile of its segment's partial gradient for
// reduce_partials_cdna4 to sum; wgrad then points at the partials.
template <Config cfg, DataType DT, typename dw_t, bool Partials = false>
__device__ void conv2d_direct_wgrad_impl(const ToType<DT>* __restrict__ in,
                                         const ToType<DT>* __restrict__ delta,
                                         dw_t* __restrict__ wgrad,
                                         int N,
                                         int groups,
                                         int c_per_group,
                                         int k_per_group,
                                         int hi,
                                         int wi,
                                         int ho,
                                         int wo,
                                         int py,
                                         int px,
                                         int row_segs)
{
    const int thread = static_cast<int>(threadIdx.x);
    const int wave   = __builtin_amdgcn_readfirstlane(thread / WAVE_SIZE);
    const int lane   = thread % WAVE_SIZE;

    // Where this wave sits in the workgroup tile.
    //
    // slot.item and slot.load_wave already derive from the readfirstlane'd `wave`, and
    // readfirstlaning them again lands a v_readfirstlane the scheduler cannot move off the chain
    // the loads issue from.
    const WaveSlot slot = wave_slot(cfg, wave);
    const int c_base    = __builtin_amdgcn_readfirstlane(slot.c_base);
    const int k_base    = __builtin_amdgcn_readfirstlane(slot.k_base);

    // Subtracted from the uniform c_base rather than read out of the slot.
    // It inherits that uniformity instead of costing a second readfirstlane.
    const int c_local    = c_base - slot.group * cfg.group_c();
    const int wave_group = __builtin_amdgcn_readfirstlane(thread / ((cfg.waves() / 2) * WAVE_SIZE));

    // Both source tensors carry every group's channels.
    // dW's contiguous extent is one group's, so only the epilogue takes c_per_group.
    const int chans   = groups * c_per_group;
    const int filters = groups * k_per_group;

    const RowTensorPars s_pars{.images = N, .rows = hi, .cols = wi, .chans = chans};
    const RowTensorPars delta_pars{.images = N, .rows = ho, .cols = wo, .chans = filters};

    // The launch's own width rather than the persistent constant.
    // A launch narrower than the cell space still covers it, one cell per stride.
    const int workgroups = static_cast<int>(gridDim.x);
    FlatGrid grid        = grid_shape(cfg, groups, c_per_group, k_per_group, N, hi, wo, workgroups);
    grid.row_seg_count   = row_segs;

    // seg_rows() divides on the VALU; readfirstlane keeps the uniform result in an SGPR.
    int seg_rows = hi;
    if constexpr(cfg.row_segments)
        seg_rows = __builtin_amdgcn_readfirstlane(grid.seg_rows());

    const RowSchedule sched{.kh            = cfg.kh,
                            .prefetch_rows = cfg.prefetch_rows,
                            .pad_h         = py,
                            .s_rows        = seg_rows,
                            .row_buffers   = cfg.row_buffers(),
                            .rows_per_tile = cfg.rows_per_tile};

    const int col_blocks = grid.col_blocks();

    // The XCD grouping lives here and nowhere else.
    const CellWalk walk = plan_cell_walk(grid.cells(), workgroups, static_cast<int>(blockIdx.x));

    with_lds_rings<cfg, DT>([&](const SRing<cfg, DT>& s_ring,
                                const DeltaRing<cfg, DT>& delta_ring,
                                const DeltaScratch<cfg, DT>& scratch) {
        for(int cell = walk.begin; cell < walk.end; cell += walk.stride)
        {
            const FlatGrid cell_grid = volatile_grid(grid);

            // A short spatial axis leaves the last segments empty.
            // Skipping is workgroup-uniform, so no wave arrives at a barrier its partners skipped.
            const ItemRange seg = cell_grid.segment(cell);
            if(seg.empty())
                continue;

            const ChannelTile tile = cell_grid.tile(cell);
            const int s_chan0      = tile.group * c_per_group + tile.c_origin;

            // Decoded before the accumulators are live: each is a chain of VALU divisions.
            int seg_origin = 0;
            int part       = 0;
            if constexpr(cfg.row_segments)
            {
                seg_origin = __builtin_amdgcn_readfirstlane(cell_grid.row_seg(cell) * seg_rows);
                part       = __builtin_amdgcn_readfirstlane(cell_grid.partial_index(cell));
            }

            // The wave writes no further than the end of its own group; see WgradTensorPars.
            // The channel axis needs no such bound, both origins being measured inside the group.
            const WgradTensorPars wgrad_pars{
                .filter_end = (tile.group + slot.group + 1) * k_per_group, .chans = c_per_group};

            DeltaRegRing<cfg, DT> delta_regs;

            // The rows in flight, on the tf32 path: fetched one iteration, split and written
            // the next. Empty on the 16-bit path, whose DMA writes LDS itself.
            RowStage<cfg, DT> stage;

            // Zeroed rather than folded into a three-operand mma.
            //
            // Folding needs the segment's first entry peeled, which instantiates the row loop
            // twice: 432 MFMA in the cell body against 216, and 160 bytes of scratch against 24.
            // See docs/algorithms/details/direct/direct-wgrad-main-loop-blocks.md.
            Accumulators<cfg> acc;
            for(int r = 0; r < cfg.kh; ++r)
                for(int s = 0; s < cfg.kw; ++s)
                    zero_tile(acc.tile[r][s]);

            // Each wave walks its own entry of the round by incrementing.
            // That keeps the division out of the loop holding the accumulators.
            SpatialItem item = grid.item(seg.begin + slot.item);

            auto run_item = [&](bool valid) {
                // The S window starts pad_w columns left of the output tile.
                //
                // That lines the read at filter shift x up with source column q + x - pad_w. One
                // column origin serves every packed image.
                const SRowLoader<cfg, DT> s_loader(volatile_pars(s_pars),
                                                   in,
                                                   slot.item,
                                                   valid,
                                                   item.image,
                                                   item.col_block * cfg.w_unfold() - px,
                                                   s_chan0,
                                                   sched.tile_origin(0),
                                                   sched.s_tile_rows(),
                                                   seg_origin);
                const DeltaRowLoader<cfg, DT> delta_loader(volatile_pars(delta_pars),
                                                           delta,
                                                           slot.item,
                                                           valid,
                                                           item.image,
                                                           item.col_block * cfg.w_unfold(),
                                                           tile.k_origin,
                                                           sched.tile_origin(0),
                                                           sched.delta_tile_rows(),
                                                           seg_origin);
                // Built only for a row-segmented config: the compiler cannot drop volatile_pars's
                // volatile loads, so an unsegmented config would keep them unused.
                const auto halo_loader = [&] {
                    if constexpr(cfg.row_segments)
                        return DeltaRowLoader<cfg, DT>(volatile_pars(delta_pars),
                                                       delta,
                                                       slot.item,
                                                       valid,
                                                       item.image,
                                                       item.col_block * cfg.w_unfold(),
                                                       tile.k_origin,
                                                       sched.tile_origin(0),
                                                       sched.delta_tile_rows(),
                                                       seg_origin,
                                                       cfg.kh - 1);
                    else
                        return 0;
                }();

                run_prologue<cfg, DT>(wave_group,
                                      slot.load_wave,
                                      slot.item,
                                      sched,
                                      s_loader,
                                      delta_loader,
                                      halo_loader,
                                      s_ring,
                                      delta_ring,
                                      scratch,
                                      k_base,
                                      stage,
                                      delta_regs);

                run_main_loop<cfg, DT, /*FirstTouch=*/false>(wave_group,
                                                             slot.load_wave,
                                                             slot.item,
                                                             sched,
                                                             s_loader,
                                                             delta_loader,
                                                             s_ring,
                                                             delta_ring,
                                                             c_base,
                                                             k_base,
                                                             delta_regs,
                                                             stage,
                                                             acc);

                // On to this wave's entry of the next round.
                //
                // The step is waves_q, so the walk crosses at most that many column blocks and
                // the carry unrolls rather than dividing.
                item.col_block += cfg.waves_q;
                bunnies::static_unroll<cfg.waves_q>([&](auto) {
                    if(item.col_block >= col_blocks)
                    {
                        item.col_block -= col_blocks;
                        item.image += cfg.unfold_n;
                    }
                });
            };

            // A round is waves_q entries, one per wave of each channel slice.
            //
            // The waves a short last round leaves over run it with an idle loader: they issue
            // every load, hit every barrier, and add zeros, so the vmcnt waits and the barrier
            // counts stay uniform across the workgroup.
            for(int x = seg.begin; x < seg.end; x += cfg.waves_q)
                run_item(x + slot.item < seg.end);

            if constexpr(Partials)
            {
                // run_partial_epilogue takes K within the wave's group. It is derived from
                // k_origin, which the delta loads keep live, so tile.group need not stay live
                // through the loop.
                const int tile_group = tile.k_origin / k_per_group;
                const int k_in_group =
                    tile.k_origin % k_per_group + k_base - slot.group * cfg.group_k();
                run_partial_epilogue<cfg>(wave,
                                          slot.item,
                                          cfg.row_segments ? part : cell / cell_grid.tiles(),
                                          tile_group + slot.group,
                                          k_in_group / MFMA_N,
                                          (tile.c_origin + c_local) / MFMA_M,
                                          partial_layout(cell_grid, cfg.kh, cfg.kw),
                                          wgrad,
                                          acc);
            }
            else
            {
                run_epilogue<cfg, dw_t>(wave,
                                        slot.item,
                                        lane,
                                        c_local,
                                        k_base,
                                        wgrad,
                                        wgrad_pars,
                                        tile.c_origin,
                                        tile.k_origin,
                                        acc);
            }
        }
    });
}

template <Config cfg, DataType DT>
__global__ __launch_bounds__(cfg.threads(),
                             1) void conv2d_direct_wgrad_cdna4(const ToType<DT>* __restrict__ in,
                                                               const ToType<DT>* __restrict__ delta,
                                                               float* __restrict__ wgrad,
                                                               int N,
                                                               int groups,
                                                               int c_per_group,
                                                               int k_per_group,
                                                               int hi,
                                                               int wi,
                                                               int ho,
                                                               int wo,
                                                               int py,
                                                               int px,
                                                               int row_segs)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_read_tr16_b64_v4i16))
    {
        conv2d_direct_wgrad_impl<cfg, DT, float>(in,
                                                 delta,
                                                 wgrad,
                                                 N,
                                                 groups,
                                                 c_per_group,
                                                 k_per_group,
                                                 hi,
                                                 wi,
                                                 ho,
                                                 wo,
                                                 py,
                                                 px,
                                                 row_segs);
    }
}

// A split entry's kernel: it stores per-segment partials, which reduce_partials_cdna4 sums into
// dW.
template <Config cfg, DataType DT>
__global__
__launch_bounds__(cfg.threads(),
                  1) void conv2d_direct_wgrad_partials_cdna4(const ToType<DT>* __restrict__ in,
                                                             const ToType<DT>* __restrict__ delta,
                                                             float* __restrict__ partials,
                                                             int N,
                                                             int groups,
                                                             int c_per_group,
                                                             int k_per_group,
                                                             int hi,
                                                             int wi,
                                                             int ho,
                                                             int wo,
                                                             int py,
                                                             int px,
                                                             int row_segs)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_read_tr16_b64_v4i16))
    {
        conv2d_direct_wgrad_impl<cfg, DT, float, /*Partials=*/true>(in,
                                                                    delta,
                                                                    partials,
                                                                    N,
                                                                    groups,
                                                                    c_per_group,
                                                                    k_per_group,
                                                                    hi,
                                                                    wi,
                                                                    ho,
                                                                    wo,
                                                                    py,
                                                                    px,
                                                                    row_segs);
    }
}

// A no_splits entry's twin that stores dW in the operand type.
//
// Its own entry point rather than a parameter of the fp32 one, which keeps every fp32 kernel's
// symbol and code as they were.
template <Config cfg, DataType DT>
__global__
__launch_bounds__(cfg.threads(),
                  1) void conv2d_direct_wgrad_narrow_dw_cdna4(const ToType<DT>* __restrict__ in,
                                                              const ToType<DT>* __restrict__ delta,
                                                              ToType<DT>* __restrict__ wgrad,
                                                              int N,
                                                              int groups,
                                                              int c_per_group,
                                                              int k_per_group,
                                                              int hi,
                                                              int wi,
                                                              int ho,
                                                              int wo,
                                                              int py,
                                                              int px,
                                                              int row_segs)
{
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_f16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_mfma_f32_16x16x32_bf16) &&
       __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_read_tr16_b64_v4i16))
    {
        conv2d_direct_wgrad_impl<cfg, DT, ToType<DT>>(in,
                                                      delta,
                                                      wgrad,
                                                      N,
                                                      groups,
                                                      c_per_group,
                                                      k_per_group,
                                                      hi,
                                                      wi,
                                                      ho,
                                                      wo,
                                                      py,
                                                      px,
                                                      row_segs);
    }
}

// The launch's operands are not the tensors the LaunchFn signature names.
//
// `wei` carries the output gradient Delta (dY) and `out` the weight gradient dW, fp32 or the
// operand type. A no_splits entry stores every element of dW. A split entry stores per-segment
// partials in the workspace, and a reduction kernel sums them into dW.
template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const Device&,
                 const ConvParams& layer,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream)
{
    const ConvParams par = layer_view(layer, cfg);

    // The kernel builds its grid at the launch's width, so this one must too.
    const FlatGrid grid = flat_grid(par, cfg, static_cast<int>(lp.grid.x));
    const int row_segs  = grid.row_segs();

    auto typed_launch = [&]<DataType DT>() {
        using dtype = ToType<DT>;

        auto conv = [&](auto kernel, auto* dst) {
            kernel<<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const dtype*>(in),
                static_cast<const dtype*>(wei),
                dst,
                par.n,
                par.groups,
                par.channels_per_group(),
                par.filters_per_group(),
                par.h,
                par.w,
                par.p,
                par.q,
                par.pad_h,
                par.pad_w,
                row_segs);
        };

        if constexpr(cfg.no_splits)
        {
            if constexpr(!cfg.is_tf32())
            {
                if(par.weight_grad_type != DataType::fp32)
                {
                    conv(conv2d_direct_wgrad_narrow_dw_cdna4<cfg, DT>, static_cast<dtype*>(out));
                    return;
                }
            }
            conv(conv2d_direct_wgrad_cdna4<cfg, DT>, static_cast<float*>(out));
        }
        else
        {
            if(workspace == nullptr)
                throw HipError(hipErrorInvalidValue,
                               "direct_wgrad's split partials need a workspace");

            const PartialLayout layout = partial_layout(grid, cfg.kh, cfg.kw);
            auto* partials             = static_cast<float*>(workspace);
            conv(conv2d_direct_wgrad_partials_cdna4<cfg, DT>, partials);

            const int parts = partial_count(grid);
            const dim3 reduce_grid(layout.groups * layout.k16s * layout.positions *
                                   divup(layout.c16s, REDUCE_WAVES));
            auto reduce = [&]<typename dw_t>(dw_t* dst) {
                auto run = [&]<int Groups>() {
                    reduce_partials_cdna4<dw_t, Groups>
                        <<<reduce_grid, REDUCE_WAVES * Groups * WAVE_SIZE, 0, stream>>>(
                            reinterpret_cast<const fp32x4_t*>(partials),
                            dst,
                            layout,
                            parts,
                            par.filters_per_group(),
                            par.channels_per_group());
                };
                if(parts >= REDUCE_GROUPS_FROM_PARTS)
                    run.template operator()<REDUCE_GROUPS>();
                else
                    run.template operator()<1>();
            };
            if constexpr(!cfg.is_tf32())
            {
                if(par.weight_grad_type != DataType::fp32)
                {
                    reduce(static_cast<dtype*>(out));
                    return;
                }
            }
            reduce(static_cast<float*>(out));
        }
    };
    if constexpr(cfg.is_tf32())
        typed_launch.template operator()<DataType::tf32>();
    else if(par.input_type == DataType::bf16)
        typed_launch.template operator()<DataType::bf16>();
    else
        typed_launch.template operator()<DataType::fp16>();
}

class DirectWgrad_ConvKernel : public DirectConvKernel
{
public:
    constexpr DirectWgrad_ConvKernel(const Config& cfg, LaunchFn launch_fn)
        : DirectConvKernel(launch_fn)
        , cfg_(cfg)
    {
    }

    std::string_view name() const override { return "direct_wgrad"; }

    KVDescriptor config_descriptor() const override { return ConfigMatcher(cfg_); }

    // Does not chain to DirectConvKernel::is_applicable.
    // That base serves the fprop/dgrad families and rejects Wgrad outright.
    bool is_applicable(const Device& /*dev*/, const ConvParams& par) const override
    {
        const bool ok_16bit =
            (par.input_type == DataType::fp16 || par.input_type == DataType::bf16) &&
            par.weight_type == par.input_type && par.output_grad_type() == par.input_type;
        const bool ok_tf32 = par.input_type == DataType::tf32 &&
                             par.weight_type == DataType::tf32 &&
                             par.output_grad_type() == DataType::fp32;
        if(!ok_16bit && !ok_tf32)
            return false;
        if(par.weight_grad_type != DataType::fp32 &&
           !(ok_16bit && par.weight_grad_type == par.input_type))
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        if(par.direction != Direction::Wgrad)
            return false;
        // The global reader requires a 4-byte multiple on the last dimension.
        // At 16 bits per element that makes both channel counts even; a 4-byte element meets it
        // whatever the count.
        if(sizeof_data_type(par.input_type) == 2 &&
           (par.channels_per_group() % 2 != 0 || par.filters_per_group() % 2 != 0))
            return false;
        if(par.stride_h != 1 || par.stride_w != 1)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        // Whether the layer can be addressed is per-entry, like the filter shape: a baseline
        // entry spans the image from one base and a tiled one spans a tile. See
        // addressing_fits.
        return true;
    }

    bool is_valid_config(const Device& /*dev*/, const ConvParams& par) const override
    {
        return direct_wgrad::is_valid_config(par, cfg_);
    }

    float get_weighted_throughput_index(const Device& /*dev*/, const ConvParams& par) const override
    {
        return throughput_index(layer_view(par, cfg_), cfg_);
    }

    LaunchParams get_launch_params(const Device& /*dev*/, const ConvParams&) const override
    {
        return direct_wgrad::get_launch_params(cfg_);
    }

    size_t get_workspace_size(const Device& /*dev*/, const ConvParams& par) const override
    {
        return direct_wgrad::get_workspace_size(layer_view(par, cfg_), cfg_);
    }

    // Supplies the blocked accumulation depth; the default would use all N*P*Q products.
    void get_tolerance(const Device& /*dev*/,
                       const ConvParams& par,
                       float& atol,
                       float& rtol) const override
    {
        get_mixed_precision_tolerance(
            par, accumulation_depth(layer_view(par, cfg_), cfg_), atol, rtol);
    }

private:
    const Config& cfg_;
};

} // namespace hipconv::cdna4::direct_wgrad
