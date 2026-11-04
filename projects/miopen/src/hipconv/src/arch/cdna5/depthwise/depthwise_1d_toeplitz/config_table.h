#pragma once

// The compiled set of depthwise 1D column-Toeplitz configs for CDNA5. Free of device code so the
// autoshard generator can read `num_configs` (and tests can name a config) without compiling the
// kernel; the kernel body lives in kernel.h.

#include "hipconv/conv_params.hpp"

#include <array>
#include <cstddef>

namespace hipconv::cdna5
{
namespace depthwise_1d_toeplitz
{

using namespace hipconv;

// 32 output columns per tile-column set == 16 WMMA tile-columns x 2 q; 8 channels per wave.
constexpr int WAVE_SIZE  = 32;
constexpr int BLOCK_Q    = 32;
constexpr int N_TILE     = 16;
constexpr int GROUP_SIZE = 8;

// A compiled kernel variant. Dgrad expresses stride 2 as `dilation` with `stride` 1, narrow_c owns
// C % 8 != 0, prefetch_depth is the input ring depth, n_fold only reshapes the grid, and w_fold
// packs that many images into one 32-column tile (needs N % w_fold == 0 and dilation 1).
//
// elem_bytes is the dtype width, and so which type the entry compiles for: 2 covers fp16/bf16
// (picked apart at launch), 4 is tf32. A config knob rather than a launch branch because the LDS
// geometry and the register budget scale with it.
struct Config
{
    int waves_per_wg    = 1;
    int kh              = 3;
    int kw              = 3;
    int stride          = 1;
    int dilation        = 1;
    Direction direction = Direction::Fprop;
    bool narrow_c       = false;
    int prefetch_depth  = 3;
    int n_fold          = 8;
    int w_fold          = 1;
    int elem_bytes      = 2;

    constexpr int block_c() const { return GROUP_SIZE * waves_per_wg; }
    constexpr int block_size() const { return waves_per_wg * WAVE_SIZE; }
    constexpr int w_sub() const { return BLOCK_Q / w_fold; }
    constexpr int subtiles() const { return N_TILE / w_fold; }
};

// One (stride, dilation, direction) family per filter: the wide wave ladder plus a single narrow
// entry. Auto-pick takes the first valid config and is_valid_config fixes the family and routes
// C % 8, so only the matching ladder competes and the head of the ladder is the pick.
constexpr auto make_configs()
{
    struct Variant
    {
        int stride;
        int dilation;
        int prefetch_depth;
        Direction direction;
    };

    constexpr auto filters  = std::array{3, 5, 7, 9, 11};
    constexpr auto variants = std::array<Variant, 4>{{
        {1, 1, 3, Direction::Fprop}, // stride-1 fprop
        {2, 1, 3, Direction::Fprop}, // stride-2 fprop
        {1, 1, 3, Direction::Dgrad}, // stride-1 dgrad
        {1, 2, 3, Direction::Dgrad}, // stride-2 dgrad (dilation path)
    }};

    // Batch-fold factors as extra wide families, gated per shape by preferred_wfold. UPS==1 only,
    // so no dgrad stride-2 fold variant.
    constexpr auto wfolds         = std::array{2, 4};
    constexpr auto wfold_variants = std::array<Variant, 3>{{
        {1, 1, 3, Direction::Fprop}, // stride-1 fprop
        {2, 1, 3, Direction::Fprop}, // stride-2 fprop
        {1, 1, 3, Direction::Dgrad}, // stride-1 dgrad
    }};

    // Waves per group, preferred first. The ladder never reaches 16: a 128-channel block wants that
    // many, but no filter has the VGPR headroom for one, and the group would own the CU alone. The
    // WMMA's M/K split pins GROUP_SIZE at 8.
    //
    // Between 8 and 4 the VGPR budget decides. A group of 8 spreads 2 waves over each of the CU's 4
    // SIMDs, so it only packs where the budget leaves an even number of wave slots per SIMD; the
    // families past 256 VGPRs get 3 slots, where one 8-wave group claims two and strands the third.
    // "The wave ladder" in docs/algorithms/toeplitz/depthwise-1d-toeplitz-cdna5.md has the budgets
    // and what each arrangement measured.
    // Every tf32 family is past 256 too: its operands are twice as wide as the 16-bit ones.
    constexpr auto wave_order = [](int kh, const Variant& v, int elem_bytes) {
        const bool over_256 = elem_bytes == 4 || kh == 11 || (kh == 9 && v.dilation == 2);
        return over_256 ? std::array{4, 8, 2, 1} : std::array{8, 4, 2, 1};
    };

    // A deeper ring than the variant default hides more cold-start TDM latency, at one LDS slot and
    // a runtime (no longer constant-folded) slot index. Only worthwhile on the 3x3 wide stride-1
    // fprop ladder; filters >= 5 must stay at the default, where a deeper ring reads out of bounds.
    constexpr auto depth_of = [](int kh, const Variant& v) {
        const bool s1_fprop = v.stride == 1 && v.dilation == 1 && v.direction == Direction::Fprop;
        return (s1_fprop && kh == 3) ? 5 : v.prefetch_depth;
    };

    constexpr std::size_t nwaves          = wave_order(3, variants[0], 2).size();
    constexpr std::size_t wide_and_narrow = filters.size() * variants.size() * (nwaves + 1);
    constexpr std::size_t folded = filters.size() * wfold_variants.size() * wfolds.size() * nwaves;
    // 16-bit wide+narrow, its folded ladders, then the same pair again for tf32.
    std::array<Config, 2 * (wide_and_narrow + folded)> configs{};
    std::size_t cfg = 0;
    for(int kh : filters)
        for(const auto& v : variants)
            for(int w : wave_order(kh, v, 2))
                configs[cfg++] = Config{.waves_per_wg   = w,
                                        .kh             = kh,
                                        .kw             = kh,
                                        .stride         = v.stride,
                                        .dilation       = v.dilation,
                                        .direction      = v.direction,
                                        .narrow_c       = false,
                                        .prefetch_depth = depth_of(kh, v)};
    for(int kh : filters)
        for(const auto& v : variants)
            configs[cfg++] = Config{.waves_per_wg   = 1,
                                    .kh             = kh,
                                    .kw             = kh,
                                    .stride         = v.stride,
                                    .dilation       = v.dilation,
                                    .direction      = v.direction,
                                    .narrow_c       = true,
                                    .prefetch_depth = v.prefetch_depth};
    // Batch-folded (F in wfolds) wide ladders (fprop s1/s2 + dgrad s1).
    for(int kh : filters)
        for(const auto& v : wfold_variants)
            for(int f : wfolds)
                for(int w : wave_order(kh, v, 2))
                    configs[cfg++] = Config{.waves_per_wg   = w,
                                            .kh             = kh,
                                            .kw             = kh,
                                            .stride         = v.stride,
                                            .dilation       = v.dilation,
                                            .direction      = v.direction,
                                            .narrow_c       = false,
                                            .prefetch_depth = v.prefetch_depth,
                                            .w_fold         = f};
    // tf32: the wide ladder plus a narrow entry, mirroring the 16-bit families above (its folded
    // ladders follow, so the whole 16-bit family set carries over). Two waves per
    // SIMD cap each at 512 VGPR, so the heaviest filters (569..595) spill 268..340 B at 8 waves --
    // little enough to keep the rung, and the ladder's other entries cover those shapes anyway.
    for(int kh : filters)
        for(const auto& v : variants)
        {
            for(int w : wave_order(kh, v, 4))
                configs[cfg++] = Config{.waves_per_wg   = w,
                                        .kh             = kh,
                                        .kw             = kh,
                                        .stride         = v.stride,
                                        .dilation       = v.dilation,
                                        .direction      = v.direction,
                                        .narrow_c       = false,
                                        .prefetch_depth = v.prefetch_depth,
                                        .elem_bytes     = 4};
            configs[cfg++] = Config{.waves_per_wg   = 1,
                                    .kh             = kh,
                                    .kw             = kh,
                                    .stride         = v.stride,
                                    .dilation       = v.dilation,
                                    .direction      = v.direction,
                                    .narrow_c       = true,
                                    .prefetch_depth = v.prefetch_depth,
                                    .elem_bytes     = 4};
        }
    // The tf32 folded ladders. Nothing about the fold depends on the element width: it only
    // reshapes the 32-column tile, and at 4 bytes the widest of them holds the ring and staging
    // tile in 27 KiB. Without them a narrow map leaves over half the tile padding.
    for(int kh : filters)
        for(const auto& v : wfold_variants)
            for(int f : wfolds)
                for(int w : wave_order(kh, v, 4))
                    configs[cfg++] = Config{.waves_per_wg   = w,
                                            .kh             = kh,
                                            .kw             = kh,
                                            .stride         = v.stride,
                                            .dilation       = v.dilation,
                                            .direction      = v.direction,
                                            .narrow_c       = false,
                                            .prefetch_depth = v.prefetch_depth,
                                            .w_fold         = f,
                                            .elem_bytes     = 4};
    // A short count would leave default-constructed entries in the tail for the shard generator to
    // compile and the matcher to consider, so make the mismatch a compile error instead.
    if(cfg != configs.size())
        throw "depthwise_1d_toeplitz config table is not exactly filled";
    return configs;
}

constexpr auto configs = make_configs();

constexpr int num_configs = static_cast<int>(configs.size());

} // namespace depthwise_1d_toeplitz
} // namespace hipconv::cdna5
