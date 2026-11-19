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
 * On gfx9 (GCN, CDNA) and gfx1250, multiProcessorCount is the compute-unit (CU) count.
 * Legacy gfx10, gfx11 and gfx1200/gfx1201 targets report work-group processors (WGPs),
 * two CUs each. Architecture family prefixes alone do not establish that convention.
 *
 * A planner sizing blocks per reported dispatch unit reads dispatchUnits(). A heuristic
 * counting physical SIMD lanes uses computeUnits().
 *
 * Both return 0 when the device does not report a count; the planners that size their launch
 * from it then decline rather than guess.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::device_units
{

/// Legacy WGP-reporting families, excluding the observed CU-reporting gfx1250 target.
/// Normalize feature suffixes before applying that architecture-specific correction.
inline bool reportsWorkgroupProcessors(std::string_view gcnArchName)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    return arch != "gfx1250" && arch.size() >= 6 && arch.substr(0, 4) == "gfx1" && arch[4] >= '0'
           && arch[4] <= '9';
}

/// HIP's reported launch units (including CUs on gfx1250); 0 when unknown.
inline int64_t dispatchUnits(const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    return device.multiProcessorCount > 0 ? static_cast<int64_t>(device.multiProcessorCount) : 0;
}

/// Physical CUs, correcting the legacy WGP-reporting targets; 0 when unknown.
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
