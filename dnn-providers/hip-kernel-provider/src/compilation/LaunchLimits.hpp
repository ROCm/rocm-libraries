// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

/**
 * @file LaunchLimits.hpp
 * @brief The limit HIP puts on one launch axis, shared by the matchers that decline a grid
 *        past it and the launch that refuses one.
 */
namespace hip_kernel_provider::compilation
{

/// The most work-items one launch axis holds: HIP rejects a launch whose workgroups times
/// workgroup size on any axis passes the dispatch packet's 32-bit grid size.
inline constexpr int64_t LAUNCH_AXIS_ITEM_LIMIT = 4294967295LL;

/// Whether @p workgroups workgroups of @p threads work-items each fit one launch axis.
inline bool launchAxisFits(int64_t workgroups, int64_t threads)
{
    return workgroups >= 0 && threads > 0 && workgroups <= LAUNCH_AXIS_ITEM_LIMIT / threads;
}

} // namespace hip_kernel_provider::compilation
