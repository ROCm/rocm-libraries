#pragma once

// The direct_wgrad device kernel, and the host selection that gives a layer one config.
//
// docs/algorithms/direct/direct-wgrad.md is the algorithm, and
// docs/algorithms/direct/direct-wgrad-estimated-cost.md the model the selection ranks by.

#include "config.h"
#include "config_desc.h"
#include "config_table.h"
#include "epilogue.h"
#include "grid.h"
#include "lds_layout.h"
#include "main_loop.h"
#include "operand_loader.h"
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
#include "hipconv/conv2d_params.hpp"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdlib>

namespace hipconv::cdna4::direct_wgrad
{
// Force variable x to be an SGPR and prevent it from being hoisted by LICM.
__device__ inline int volatile_scalar(int x)
{
    asm volatile("" : "+s"(x));
    return x;
}

// The tensor shape, in a form the compiler cannot hoist out of the loop that reads it.
__device__ inline RowTensorPars volatile_pars(const RowTensorPars& pars)
{
    return {volatile_scalar(pars.images),
            volatile_scalar(pars.rows),
            volatile_scalar(pars.cols),
            volatile_scalar(pars.chans)};
}

// A copy of the grid whose decode cannot be hoisted out of the cell loop.
__device__ inline FlatGrid volatile_grid(const FlatGrid& g)
{
    FlatGrid out    = g;
    out.groups      = volatile_scalar(g.groups);
    out.c_per_group = volatile_scalar(g.c_per_group);
    out.k_per_group = volatile_scalar(g.k_per_group);
    out.images      = volatile_scalar(g.images);
    out.out_cols    = volatile_scalar(g.out_cols);
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

// How many images this layer should pack into a column block.
//
// The packing that runs the fewest padded output columns, ties to the smaller. Width alone
// decides it, the packing and the arrangement trading against nothing in common; see
// direct-wgrad-estimated-cost.md.
//
// HIPCONV_WGRAD_UNFOLD_N pins the answer, on the same terms as the arrangement knobs.
// Over the packings rather than the table: every spread carries all three, and the column block's
// width follows the packing alone, so the table has nothing to add and is_valid_config asks this
// once per entry.
inline int preferred_unfold_n(const Conv2dParams& par)
{
    const int pinned = Pins::instance().unfold_n();

    int chosen = 0;
    int fewest = 0;
    for(const int unfold_n : packings)
    {
        if(pinned != 0)
        {
            if(unfold_n == pinned)
                return pinned;
            continue;
        }

        // Strictly fewer, so a tie leaves the smaller packing in place.
        // The table lists the packings in increasing order.
        const int columns = packed_columns(par, unfold_n);
        if(chosen == 0 || columns < fewest)
        {
            fewest = columns;
            chosen = unfold_n;
        }
    }
    return chosen == 0 ? 1 : chosen;
}

// Whether this entry may run this layer at all.
//
// Every ungrouped entry may: a block wider than the tensor pads, and the epilogue's bounds drop
// what the padding computed. A grouped entry reads waves_g whole groups as one contiguous run, so
// it admits no padding on either channel axis, and the group count has to divide or a short last
// block leaves waves writing past dW. The same tests keep it off an ungrouped layer, where it
// would compute the block diagonal of a tile with no block structure.
inline bool serves(const Conv2dParams& par, const Config& cfg)
{
    if(cfg.waves_g == 1)
        return true;
    return par.groups % cfg.waves_g == 0 && par.channels_per_group() == cfg.group_c() &&
           par.filters_per_group() == cfg.group_k();
}

// Whether this entry's filter and group shape fit the layer, before the packing narrows further.
//
// Field compares only. is_valid_config leads with these because they reject most of the table,
// and what follows them is per-layer work that would otherwise run once per entry.
inline bool fits_the_layer(const Conv2dParams& par, const Config& cfg)
{
    return cfg.kh == par.kh && cfg.kw == par.kw && serves(par, cfg);
}

// Whether this entry is one of the layer's candidates, before the pins narrow them.
inline bool is_a_candidate(const Conv2dParams& par, const Config& cfg, int unfold_n)
{
    return fits_the_layer(par, cfg) && cfg.unfold_n == unfold_n;
}

// Whether the pins name a candidate of this layer at all.
//
// When they name none, every caller below ignores them and proceeds as if none were set, so a
// mistyped knob does not look like a missing kernel.
inline bool the_pins_name_a_candidate(const Conv2dParams& par, int unfold_n)
{
    for(const Config& c : configs)
        if(is_a_candidate(par, c, unfold_n) && Pins::instance().match(c))
            return true;
    return false;
}

// Whether the images one loader spans fit in 32 bits.
inline bool window_fits_a_buffer(const Conv2dParams& par)
{
    const int64_t images = preferred_unfold_n(par);
    const int64_t s_bytes =
        images * par.h * par.w * (int64_t{par.groups} * par.channels_per_group()) * 2;
    const int64_t delta_bytes =
        images * par.p * par.q * (int64_t{par.groups} * par.filters_per_group()) * 2;
    constexpr int64_t int32_max = 0x7fffffff;
    return s_bytes <= int32_max && delta_bytes <= int32_max;
}

// Whether this entry may run this layer, which several entries of a layer may.
//
// Which of them the dispatcher prefers is the ranking's answer, not this one. A launch reaches
// this through ConvLaunch::make, which admits any entry the ranking offered.
inline bool is_valid_config(const Conv2dParams& par, const Config& cfg)
{
    if(!fits_the_layer(par, cfg))
        return false;
    const int unfold_n = preferred_unfold_n(par);
    if(cfg.unfold_n != unfold_n)
        return false;
    return Pins::instance().match(cfg) || !the_pins_name_a_candidate(par, unfold_n);
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

template <Config cfg, DataType DT>
__device__ void conv2d_direct_wgrad_impl(const ToType<DT>* __restrict__ in,
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
                                         int px)
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

    const RowSchedule sched{.kh            = cfg.kh,
                            .prefetch_rows = cfg.prefetch_rows,
                            .pad_h         = py,
                            .s_rows        = hi,
                            .row_buffers   = cfg.row_buffers()};

    // The launch's own width rather than the persistent constant.
    // A launch narrower than the cell space still covers it, one cell per stride.
    const int workgroups = static_cast<int>(gridDim.x);
    const FlatGrid grid{.groups           = groups,
                        .c_per_group      = c_per_group,
                        .k_per_group      = k_per_group,
                        .images           = N,
                        .out_cols         = wo,
                        .block_c          = cfg.group_c(),
                        .block_k          = cfg.group_k(),
                        .groups_per_block = cfg.waves_g,
                        .block_cols       = cfg.w_unfold(),
                        .unfold_n         = cfg.unfold_n,
                        .workgroups       = workgroups};

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

            // The wave writes no further than the end of its own group; see WgradTensorPars.
            // The channel axis needs no such bound, both origins being measured inside the group.
            const WgradTensorPars wgrad_pars{
                .filter_end = (tile.group + slot.group + 1) * k_per_group, .chans = c_per_group};

            DeltaRegRing<cfg, DT> delta_regs;

            // Zeroed rather than folded into a three-operand mma.
            //
            // Folding needs the segment's first entry peeled, which instantiates the row loop
            // twice: 432 MFMA in the cell body against 216, and 160 bytes of scratch against 24.
            // See docs/algorithms/direct/direct-wgrad-main-loop-blocks.md.
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
                                                   s_chan0);
                const DeltaRowLoader<cfg, DT> delta_loader(volatile_pars(delta_pars),
                                                           delta,
                                                           slot.item,
                                                           valid,
                                                           item.image,
                                                           item.col_block * cfg.w_unfold(),
                                                           tile.k_origin);

                run_prologue<cfg, DT>(slot.load_wave,
                                      slot.item,
                                      sched,
                                      s_loader,
                                      delta_loader,
                                      s_ring,
                                      delta_ring,
                                      scratch,
                                      k_base,
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

            run_epilogue<cfg>(wave,
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
                                                               int px)
{
    conv2d_direct_wgrad_impl<cfg, DT>(
        in, delta, wgrad, N, groups, c_per_group, k_per_group, hi, wi, ho, wo, py, px);
}

// The launch's operands are not the tensors the LaunchFn signature names.
//
// `wei` carries the output gradient Delta (dY) and `out` the fp32 weight gradient dW, which the
// epilogue reduces into with atomicAdd, so the launch zeroes it first.
template <Config cfg>
void launch_impl(const LaunchParams& lp,
                 const Conv2dParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* /*workspace*/,
                 hipStream_t stream)
{
    auto typed_launch = [&]<DataType DT>() {
        using dtype = ToType<DT>;

        HIP_CHECK(hipMemsetAsync(out, 0, Conv2dSize(par).weight_grad_bytes(), stream));

        conv2d_direct_wgrad_cdna4<cfg, DT>
            <<<lp.grid, lp.block_size, lp.dynamic_shared_bytes, stream>>>(
                static_cast<const dtype*>(in),
                static_cast<const dtype*>(wei),
                static_cast<float*>(out),
                par.n,
                par.groups,
                par.channels_per_group(),
                par.filters_per_group(),
                par.h,
                par.w,
                par.p,
                par.q,
                par.pad_h,
                par.pad_w);
    };
    if(par.input_type == DataType::bf16)
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

    std::string describe_config() const override { return ConfigMatcher(cfg_).describe(); }

    bool matches_descriptor(std::string_view spec, std::string* error) const override
    {
        ConfigMatcher matcher(cfg_);
        if(matcher.match(spec))
            return true;
        if(error)
            *error = matcher.error();
        return false;
    }

    // Does not chain to DirectConvKernel::is_applicable.
    // That base serves the fprop/dgrad families and rejects Wgrad outright.
    bool is_applicable(const Conv2dParams& par) const override
    {
        // S and Delta are fp16 or bf16 and share a type; dW is fp32.
        if(par.input_type != DataType::fp16 && par.input_type != DataType::bf16)
            return false;
        if(par.weight_type != par.input_type || par.output_grad_type() != par.input_type)
            return false;
        if(par.weight_grad_type != DataType::fp32)
            return false;
        if(par.order != TensorOrder::NHWC)
            return false;
        if(par.direction != Direction::Wgrad)
            return false;
        // The global reader requires a 4-byte multiple on the last dimension.
        // At 16 bits per element that makes both channel counts even.
        if(par.channels_per_group() % 2 != 0 || par.filters_per_group() % 2 != 0)
            return false;
        if(par.stride_h != 1 || par.stride_w != 1)
            return false;
        if(par.dilation_h != 1 || par.dilation_w != 1)
            return false;
        return window_fits_a_buffer(par);
    }

    bool is_valid_config(const Conv2dParams& par) const override
    {
        return direct_wgrad::is_valid_config(par, cfg_);
    }

    float get_weighted_throughput_index(const Conv2dParams& par) const override
    {
        return throughput_index(par, cfg_);
    }

    LaunchParams get_launch_params(const Conv2dParams&) const override
    {
        return direct_wgrad::get_launch_params(cfg_);
    }

private:
    const Config& cfg_;
};

} // namespace hipconv::cdna4::direct_wgrad
