#pragma once

#include "config.h"
#include "hipconv/conv_params.hpp"

#include <array>

// The compiled configs; host-only, as the autoshard generator counts them on the host.

namespace hipconv::cdna4::direct_flex
{

constexpr Config tiles[] = {
    {.x_tile16 = 16, .k_tile16 = 16, .waves_x = 2, .waves_k = 4},
    {.x_tile16 = 16, .k_tile16 = 12, .waves_x = 4, .waves_k = 2},
    {.x_tile16 = 16, .k_tile16 = 8, .waves_x = 4, .waves_k = 2},
};

// Tiles whose reduction is split within the workgroup (csplit_kernel.h). Each compiles
// only variants for nine-position filters in one filter tile, without a channel tail.
constexpr Config csplit_tiles[] = {
    {.x_tile16 = 8, .k_tile16 = 8, .waves_x = 2, .waves_k = 1, .csplit = 4},
    {.x_tile16 = 8, .k_tile16 = 16, .waves_x = 2, .waves_k = 2, .csplit = 2},
};
constexpr int CSPLIT_SWEEP = 9;

// A tile and the variant a layer fixes, so a layer has one config per tile.
// launch_impl picks the chunk parity and the split, which depend on the device and workspace.
struct Variant
{
    Config tile;
    DataType type;
    Direction direction;
    bool narrow_k;      // fprop with K % GRANULE != 0, stored element-wise
    bool c_tail;        // the reduction's channels are not a multiple of C_CHUNK
    int sweep;          // positions per filter tile with positions inner, 0 for positions outer
    bool one_ftile;     // positions inner, the whole filter one filter tile
    bool depth = false; // conv3d

    constexpr bool operator==(const Variant&) const = default;
};

constexpr int DEPTH_SWEEP = 9;

// Whether a positions-inner variant is compiled; a layer whose variant is not runs positions outer.
//
// docs/algorithms/details/direct/direct-flex-launch.md, "Configs and kernels".
constexpr bool inner_compiled(bool dgrad, bool narrow, bool tail, int sweep, bool one_ftile)
{
    if(sweep == 9)
        return one_ftile;
    if(one_ftile)
        return tail || !narrow;
    return !narrow && !(dgrad && tail);
}

// Positions outer in every variant, then the compiled positions-inner variants; depth last.
template <class F>
constexpr void for_each_config(F&& add)
{
    for(const bool depth : {false, true})
        for(const Config& tile : tiles)
            for(const DataType type : {DataType::fp16, DataType::bf16})
                for(const Direction dir : {Direction::Fprop, Direction::Dgrad})
                    for(const bool narrow : {false, true})
                        for(const bool tail : {false, true})
                        {
                            // A dgrad's output channels are its granule-aligned C.
                            const bool dgrad = dir == Direction::Dgrad;
                            if(dgrad && narrow)
                                continue;
                            add(Variant{tile, type, dir, narrow, tail, 0, false, depth});
                            if(depth)
                                add(Variant{
                                    tile, type, dir, narrow, tail, DEPTH_SWEEP, false, true});
                            else
                                for(const int n : SWEEPS)
                                    for(const bool one : {false, true})
                                        if(inner_compiled(dgrad, narrow, tail, n, one))
                                            add(Variant{tile, type, dir, narrow, tail, n, one});
                        }
    for(const Config& tile : csplit_tiles)
        for(const DataType type : {DataType::fp16, DataType::bf16})
            for(const Direction dir : {Direction::Fprop, Direction::Dgrad})
                add(Variant{tile, type, dir, false, false, CSPLIT_SWEEP, true});
}

constexpr int count_configs()
{
    int n = 0;
    for_each_config([&](const Variant&) { ++n; });
    return n;
}
constexpr int num_configs = count_configs();

constexpr std::array<Variant, num_configs> make_configs()
{
    std::array<Variant, num_configs> v{};
    int i = 0;
    for_each_config([&](const Variant& x) { v[i++] = x; });
    return v;
}

constexpr auto configs = make_configs();

constexpr bool sweep_compiled(int n)
{
    for(const int s : SWEEPS)
        if(s == n)
            return true;
    return false;
}
static_assert(sweep_compiled(DEPTH_SWEEP), "the depth variants sweep a compiled filter-tile size");

// Whether every entry names a kernel the compute and staging phases can build.
constexpr bool table_is_well_formed()
{
    for(const Config& cfg : tiles)
    {
        // The ping-pong schedule runs two wave groups of one wave per SIMD over X(256).
        if(cfg.x_tile() != 256 || cfg.waves() != 8)
            return false;
        if(cfg.half_x() % (cfg.waves_x * MFMA_M) != 0 || cfg.half_k() % (cfg.waves_k * MFMA_N) != 0)
            return false;
        // A K half stages into a buffer of an X half's rows.
        if(cfg.half_k() > cfg.half_x())
            return false;
        // A staging pass keeps a thread on one granule column, so it covers whole rows.
        if(cfg.in_rounds() * cfg.threads() != cfg.x_tile() * Config::granules())
            return false;
        if(cfg.threads() % Config::granules() != 0)
            return false;
        if(cfg.lds_bytes() > 160 * 1024)
            return false;
    }
    // csplit_kernel.h is written for X(128) × K(128) split four ways and X(128) × K(256) split
    // two ways, each on 8 waves of X(64) × K(128).
    for(const Config& cfg : csplit_tiles)
    {
        const bool four = cfg.csplit == 4 && cfg.k_tile() == 128 && cfg.waves_k == 1;
        const bool two  = cfg.csplit == 2 && cfg.k_tile() == 256 && cfg.waves_k == 2;
        if(!(four || two) || cfg.x_tile() != 128 || cfg.waves_x != 2 || cfg.waves() != 8)
            return false;
    }
    return true;
}

static_assert(table_is_well_formed(),
              "a direct_flex config is malformed: X(256) on 8 waves or a split X(128) × K(128) or "
              "X(128) × K(256), "
              "the wave grid dividing both "
              "half-tiles into 16-row blocks, a K half no larger than an X half, staging in whole "
              "passes over the workgroup, and both tiles within LDS");

} // namespace hipconv::cdna4::direct_flex
