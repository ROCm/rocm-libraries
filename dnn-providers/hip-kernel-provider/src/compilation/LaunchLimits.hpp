// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

/**
 * @file LaunchLimits.hpp
 * @brief The limits HIP puts on one launch axis, shared by the matchers that decline a grid
 *        past them and the launch that refuses one.
 */
namespace hip_kernel_provider::compilation
{

/// The most work-items one launch axis holds: HIP rejects a launch whose workgroups times
/// workgroup size on any axis passes the dispatch packet's 32-bit grid size.
inline constexpr int64_t LAUNCH_AXIS_ITEM_LIMIT = 4294967295LL;

/// The most workgroups the x axis of a launch grid holds: `hipModuleLaunchKernel` (ROCm/clr
/// `hipamd/src/hip_module.cpp`) refuses a gridDimX past INT32_MAX.
inline constexpr int64_t LAUNCH_GRID_X_LIMIT = 2147483647LL;

/// The most workgroups the y or z axis of a launch grid holds: `hipModuleLaunchKernel` refuses
/// a gridDimY or gridDimZ past 65536.
inline constexpr int64_t LAUNCH_GRID_YZ_LIMIT = 65536;

/// Whether @p workgroups workgroups of @p threads work-items each fit one launch axis.
inline bool launchAxisFits(int64_t workgroups, int64_t threads)
{
    return workgroups >= 0 && threads > 0 && workgroups <= LAUNCH_AXIS_ITEM_LIMIT / threads;
}

} // namespace hip_kernel_provider::compilation
