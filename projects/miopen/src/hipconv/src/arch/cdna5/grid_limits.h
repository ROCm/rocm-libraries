#pragma once

#include "launch_params.h"

#include <algorithm>
#include <cstdint>

namespace hipconv::cdna5
{

// HIP's advertised limit on grid y and z (hipDeviceProp_t::maxGridSize).
//
// A launch with grid z of 98,304 failed on gfx1250; see docs/gpu-reference.md.
constexpr unsigned max_grid_yz = 65535;

// A grid extent computed in 64 bits, saturated so that a count past 32 bits fails grid_fits.
inline unsigned grid_extent(int64_t blocks)
{
    return static_cast<unsigned>(std::min<int64_t>(blocks, UINT32_MAX));
}

// Whether HIP accepts a launch: grid y and z within max_grid_yz, and fewer than 2^32 work items
// in x.
inline bool grid_fits(const LaunchParams& launch)
{
    return launch.grid.y <= max_grid_yz && launch.grid.z <= max_grid_yz &&
           uint64_t(launch.grid.x) * launch.block_size.x <= UINT32_MAX;
}

} // namespace hipconv::cdna5
