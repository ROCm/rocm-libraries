#pragma once

#include "hipconv/conv_params.hpp"

#include <array>
#include <cstddef>

// The compiled set of depthwise 1D column-Toeplitz configs.
//
// Intentionally free of device code so the autoshard generator can read
// `num_configs` (and tests can name a config) without compiling the kernel; the
// kernel body lives in kernel.h.

namespace hipconv::cdna4::depthwise_1d_toeplitz
{
using namespace hipconv;

constexpr int WAVE_SIZE = 64;
constexpr int BLOCK_Q   = 32; // 16 tiles x 2 outputs per tile

// A CU's 4 SIMDs each hold a 512-register file per lane, arch VGPRs and AGPRs together, so a
// w-wave workgroup puts ceil(w/4) waves on every SIMD and leaves each lane 512/ceil(w/4)
// registers: 512 up to w 4, then 256, 170 and 128 at 8, 12 and 16. The kernel carries
// __launch_bounds__(block_size()), so a rung over its budget does not quietly lose occupancy,
// it spills; wave_ok and tf32_wave_ok drop the rungs that would.
//
// Peak use rises with kh, and tf32 is the heavier width on both counts -- its operands are twice
// as wide and the split holds a bf16 pair of each, and its LDS ring is sized in uint4s of 4 fp32
// channels rather than 8 halves. Which rungs that leaves was read off the compiled asm, a rung
// admitted only where it carries no scratch.
//
// Only the 16 rung is ever short of budget for 16-bit, and only from kh 9 up.
constexpr bool wave_ok(int kh, int w)
{
    return !(kh >= 9 && w == 16);
}
// The tf32 cap, deliberately tighter than the register budget alone asks for.
//
// The budget leaves three rungs this gate still drops: w 16 at kh 3 and 5, w 12 at kh 7. All
// three compile with no scratch and then lose on all eight shapes measured, by 1.05x to 1.78x
// against the best rung already on the ladder. The doubled LDS ring is what costs them: against
// the kernel's 64 KiB static bound it pushes the widest tf32 rungs to two buffers where the rung
// below keeps three (at kh 3, 51200 bytes at w 16 against 51456 at 12), and a wide tile is not
// worth the prefetch depth it costs.
//
// What remains still covers every C: a shape needs only one w with c % (8*w) == 0, and w == 1
// divides every channel count.
constexpr bool tf32_wave_ok(int kh, int w)
{
    // Also the swizzle's bound, not just the ring's: block_c 128 is C4 = 32, whose MASK = 31
    // reaches bit 4 and collides with the `a >> 4` it is xored against, costing DwSwizzleTf32's
    // fold its involution (B = 5 > S = 4). C4 = 24 takes the MASK = 7 branch at B = 3.
    if(w >= 16)
        return false;
    if(kh >= 9)
        return w <= 4;
    if(kh >= 7)
        return w <= 8;
    return w <= 12;
}

// The order auto-pick walks, which is what makes the ladder a heuristic rather than an
// enumeration: enumerate_configs emits it as written and the dispatcher takes the first rung
// is_valid_config admits, so listing it by descending width means "the widest workgroup that
// divides C". That is the wrong end. A 4-wave workgroup is 256 threads, one wave per SIMD --
// the narrowest that spans the CU, and narrow enough that the scheduler still has several
// workgroups per CU to interleave. Fallbacks step outward from it, wider first, a wider
// block_c reading longer contiguous channel runs.
//
// Measured over 24 depthwise shapes x {n, dtype} (88 points, gfx950): the first valid rung
// costs 5.6% against the per-shape best, against 15.1% for descending width, beating it on 46
// of the 88 and losing on 6, four of those by more than 5%. Derived on 14 shapes and confirmed
// on 10 held out, kh 9 and n 64 among them. tf32 keeps the same preference -- its heavier
// register footprint is already spent by tf32_wave_ok capping the ladder, and starting it at 2
// instead bought 0.6% of the mean for four times the regressions.
//
// Where the shape wants something else the gap survives (up to 1.6x): the best rung ranges
// over the whole ladder and no fixed order can track that. Closing it needs an index that
// reads the shape, which is what get_weighted_throughput_index is for; the family reports a
// flat 1 today, so this order is the whole selection story.
constexpr auto wave_order = std::array{4, 8, 2, 12, 16, 1};

struct Config
{
    int waves_per_wg;
    int kh         = 3;
    int kw         = 3; // filter width (== kh for the shipped square filters)
    int group_size = 8;
    int n_fold     = 8;
    int stride     = 1;
    // Input-upsample factor: 1 for Fprop; for Dgrad it carries the forward stride
    // (dgrad = stride-1 rot180 correlation over dY upsampled by s).
    int dilation        = 1;
    Direction direction = Direction::Fprop;
    // Narrow-channel path (C % 8 != 0): no coalesced uint4 load/store, so one
    // 8-channel group per workgroup (waves_per_wg == 1), one channel at a time (b16)
    // and masked past C. LDS layout and MFMA match the wide path.
    bool narrow_c = false;
    // Batch-folded W tiling: pack w_fold images into one 32-wide MFMA tile, each
    // contributing BLOCK_Q/w_fold columns of the same w-window. Fills the tile when a
    // single image leaves trailing tile-columns idle. w_fold == 1 is the one-image tile.
    int w_fold = 1;
    // Input LDS ring depth: 3 == triple-buffered (2 rows in flight to hide load
    // latency), 2 == double-buffered (33% less input LDS -> higher occupancy).
    // Must be >= 2.
    int lds_buffers = 3;
    // The dtype width, and so which type the entry compiles for: 2 covers fp16/bf16 (picked
    // apart at launch), 4 is tf32. A config knob rather than a launch branch because the LDS
    // geometry and the register budget both scale with it.
    int elem_bytes = 2;

    constexpr int block_c() const { return group_size * waves_per_wg; }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
    // Outputs per sub-image, and MFMA tile-columns per sub-image (2 outputs each).
    constexpr int w_sub() const { return BLOCK_Q / w_fold; }
    constexpr int subtiles() const { return 16 / w_fold; }

    // Shared memory the kernel statically allocates: the input ring (w_fold segments, each
    // w_sub columns plus a kw-1 halo, lds_buffers deep) and the output staging tile, both
    // counted in uint4s of 16/elem_bytes channels. Mirrors the kernel's IO_LDS_SIZE, which
    // asserts against the same bound -- so a rung the table lets through cannot overflow.
    //
    // Note folding *raises* this: each of the w_fold segments carries its own halo, so at
    // kw 5 an F=4 tile spans 48 input columns against F=1's 36.
    constexpr int shared_bytes() const
    {
        const int block_c_uint4 = block_c() / (16 / elem_bytes);
        const int seg_uint4     = block_c_uint4 * (w_sub() + kw - 1);
        const int input         = lds_buffers * w_fold * seg_uint4;
        const int output        = block_c_uint4 * (BLOCK_Q / stride);
        return (input + output) * 16;
    }
};

// The kernel allocates its ring with a static __shared__ array, so the 64 KiB static bound
// applies rather than the 160 KiB a dynamic allocation could opt into.
constexpr int kMaxSharedBytes = 64 * 1024;

// Generated config set: for every filter size, element width and (stride, dilation,
// direction) family, the wide (C % 8 == 0) wave ladder plus one narrow entry.
//
// Order contract: auto-pick takes the first valid config. is_valid_config fixes
// kh/stride/direction/elem_bytes and gates waves by c % block_c(), so only the matching
// family's ladder competes, in wave_order's preference order rather than by descending
// width; numeric --config indices are not a stable ABI.
//
// The enumeration is written once and driven by `emit`, so the table and its size come from
// the same loops -- a hand-maintained count would have to be re-derived every time a gate
// changes, and a short one silently leaves default-constructed entries for the shard
// generator to compile and the matcher to consider.
template <typename Emit>
constexpr void enumerate_configs(Emit&& emit)
{
    // One (stride, dilation, direction) family. Fprop varies stride at dilation 1;
    // Dgrad stays stride 1 and carries the forward stride in the upsample dilation.
    struct Variant
    {
        int stride;
        int dilation;
        Direction direction;
    };

    constexpr auto filters  = std::array{3, 5, 7, 9, 11};
    constexpr auto variants = std::array<Variant, 4>{{
        {1, 1, Direction::Fprop}, // stride-1 fprop
        {2, 1, Direction::Fprop}, // stride-2 fprop
        {1, 1, Direction::Dgrad}, // stride-1 dgrad
        {1, 2, Direction::Dgrad}, // stride-2 dgrad
    }};

    // Batch-fold factors (wide, same compute path), stamped as extra families. F=2
    // (w_sub=16) reclaims a half-empty tail tile; F=4 (w_sub=8) fills sub-tile feature
    // maps. is_valid_config / preferred_wfold gate and auto-pick them per shape.
    constexpr auto wfolds         = std::array{2, 4};
    constexpr auto wfold_variants = std::array<Variant, 4>{{
        {1, 1, Direction::Fprop}, // stride-1 fprop
        {2, 1, Direction::Fprop}, // stride-2 fprop
        {1, 1, Direction::Dgrad}, // stride-1 dgrad
        {1, 2, Direction::Dgrad}, // stride-2 dgrad (input upsampled 2x)
    }};

    // Element widths, 16-bit families first so the 2-byte order is exactly what it was.
    constexpr auto widths = std::array{2, 4};

    // A tf32 ring can outgrow the static shared bound where its 16-bit twin fits, most often
    // on a folded tile (F segments, F halos). Halving the ring depth recovers the rung at the
    // cost of one less row in flight, which beats dropping it and leaving the shape to a
    // kernel with no Toeplitz; only if that still overflows is the rung skipped.
    const auto with_depth = [](Config c) {
        for(int depth : {c.lds_buffers, 2})
        {
            c.lds_buffers = depth;
            if(c.shared_bytes() <= kMaxSharedBytes)
                return c;
        }
        c.lds_buffers = 0; // sentinel: no depth fits, caller drops the rung
        return c;
    };

    // Wide: capped wave ladder per family. Narrow: one waves==1 entry per family.
    // Folded: the wide stride-1 wave ladder once per (fold variant, w_fold). Large
    // filters shorten the ladder to skip the spilling top rungs.
    for(int eb : widths)
    {
        const auto ok = [eb](int kh, int w) {
            return eb == 4 ? tf32_wave_ok(kh, w) : wave_ok(kh, w);
        };
        // Wide (C % 8 == 0): coalesced uint4 path, full wave ladder.
        for(int kh : filters)
            for(const auto& v : variants)
                for(int w : wave_order)
                {
                    if(!ok(kh, w))
                        continue;
                    const auto c = with_depth(Config{.waves_per_wg = w,
                                                     .kh           = kh,
                                                     .kw           = kh,
                                                     .stride       = v.stride,
                                                     .dilation     = v.dilation,
                                                     .direction    = v.direction,
                                                     .elem_bytes   = eb});
                    if(c.lds_buffers > 0)
                        emit(c);
                }
        // Narrow (C % 8 != 0): one 8-channel group per workgroup, per-channel transfers,
        // masked past C.
        for(int kh : filters)
            for(const auto& v : variants)
                emit(Config{.waves_per_wg = 1,
                            .kh           = kh,
                            .kw           = kh,
                            .stride       = v.stride,
                            .dilation     = v.dilation,
                            .direction    = v.direction,
                            .narrow_c     = true,
                            // Single-wave (64-thread) fallback caps at 256 VGPRs; large
                            // filters need double- not triple-buffering to keep the input
                            // ring's in-flight rows out of scratch. tf32 reaches that sooner
                            // still, its operands being twice as wide.
                            .lds_buffers = (kh >= 5 || eb == 4) ? 2 : 3,
                            .elem_bytes  = eb});
        // Batch-folded (F in wfolds) wide stride-1 ladders (fprop + dgrad).
        for(int kh : filters)
            for(const auto& v : wfold_variants)
                for(int f : wfolds)
                    for(int w : wave_order)
                    {
                        if(!ok(kh, w))
                            continue;
                        const auto c = with_depth(Config{.waves_per_wg = w,
                                                         .kh           = kh,
                                                         .kw           = kh,
                                                         .stride       = v.stride,
                                                         .dilation     = v.dilation,
                                                         .direction    = v.direction,
                                                         .w_fold       = f,
                                                         .elem_bytes   = eb});
                        if(c.lds_buffers > 0)
                            emit(c);
                    }
    }
}

constexpr std::size_t num_dw_configs()
{
    std::size_t n = 0;
    enumerate_configs([&](const Config&) { ++n; });
    return n;
}

constexpr auto make_configs()
{
    std::array<Config, num_dw_configs()> configs{};
    std::size_t cfg = 0;
    enumerate_configs([&](const Config& c) { configs[cfg++] = c; });
    return configs;
}

constexpr auto configs = make_configs();

constexpr int num_configs = static_cast<int>(configs.size());

} // namespace hipconv::cdna4::depthwise_1d_toeplitz
