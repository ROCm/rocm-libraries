// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string_view>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

/**
 * @file DeviceUnits.hpp
 * @brief What HIP's multiProcessorCount counts, for the conv planners.
 *
 * On gfx9 (GCN, CDNA) multiProcessorCount is the compute-unit (CU) count. On gfx10, gfx11 and
 * gfx12 HIP reports work-group processors (WGPs), two CUs each: gfx1151 20 (rocminfo: 40 CUs),
 * gfx1100 W7900 48 (96), gfx1201 R9700 32 (64).
 *
 * Both are the unit a workgroup is dispatched onto (gfx10+ kernels run in WGP mode), and both
 * hold four SIMDs, so a planner that sizes a grid in blocks per unit reads dispatchUnits(). A
 * heuristic that counts SIMD lanes needs the physical CUs: computeUnits().
 *
 * Both return 0 when the device does not report a count; the planners that size their launch
 * from it then decline rather than guess.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::device_units
{

/// Whether multiProcessorCount counts WGPs rather than CUs: gfx10, gfx11 and gfx12 (suffixes
/// and generic targets such as gfx11-generic included).
inline bool reportsWorkgroupProcessors(std::string_view gcnArchName)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    return arch.size() >= 6 && arch.substr(0, 4) == "gfx1" && arch[4] >= '0' && arch[4] <= '9';
}

/// The units workgroups are dispatched onto (CUs on gfx9, WGPs on gfx10+); 0 when unknown.
inline int64_t dispatchUnits(const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    return device.multiProcessorCount > 0 ? static_cast<int64_t>(device.multiProcessorCount) : 0;
}

/// Physical compute units (two per WGP on gfx10+); 0 when unknown.
inline int64_t computeUnits(const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    const int64_t units = dispatchUnits(device);
    return reportsWorkgroupProcessors(device.gcnArchName) ? 2 * units : units;
}

/// The XCDs (accelerator dies, each with its own L2) the device deals workgroups to round-robin
/// in launch order: gfx942 dies hold 38 CUs (MI300X / MI325X 8, MI300A 6) or 20 (MI308X 4),
/// gfx950 dies 32 (MI350X / MI355X 8). 1 elsewhere, and for a partition of one die.
inline int xcdCount(const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(device.gcnArchName);
    const int cus = device.multiProcessorCount;
    if(cus <= 0)
    {
        return 1;
    }
    if(arch == "gfx942" && cus % 38 == 0)
    {
        return cus / 38;
    }
    if(arch == "gfx942" && cus % 20 == 0)
    {
        return cus / 20;
    }
    if(arch == "gfx950")
    {
        return cus % 32 == 0 ? cus / 32 : 1;
    }
    return 1;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::device_units
