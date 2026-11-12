// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Per-device loading of the Flash-Attention 2 V7 kernel. A plan selects its
// variant for an architecture at build time; the module that variant is loaded
// into belongs to one device. This keeps what is needed to load the same
// variant on any device of that architecture, so execute() can launch on the
// device the executing handle's stream names. See device::PerDeviceResolver
// for the admission rule.

#pragma once

#include "HipFlash2KernelUtils.hpp"
#include "device/PerDeviceResolver.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace hip_flash2_engine
{

using Flash2Module = std::shared_ptr<const HipModuleGuard>;
using Flash2Kernels = hip_kernel_provider::device::PerDeviceResolver<Flash2Module>;

/// Loads @p funcName from @p coPath on @p deviceOrdinal. When @p expectedBlockDim is
/// non-zero, also verifies the object was built with exactly that many threads per
/// block.
///
/// The file probe that chose @p coPath only proves a file exists at that path -- it
/// says nothing about how the object was compiled. A mismatch is not benign in EITHER
/// direction: an object built for fewer threads than the table claims fails every
/// launch with hipError 719, and one built for more threads launches fine and computes
/// silently wrong results.
///
/// The comparison is therefore exact, not an upper bound. An earlier revision used
/// `<`, which caught only the loud direction; S. Reeder demonstrated the gap by copying
/// the 512-thread w8q2k4 object over the w4q1k4 filename (table: 256 threads) and
/// running the integration suite: `<` threw 0 times and produced 4 wrong results, `!=`
/// threw 4 times and produced 0. Exact comparison cannot false-positive here because
/// every variant carries __launch_bounds__(F2_BLOCK, 1), so max_flat_workgroup_size
/// equals the compiled block size exactly on all five (measured 256/512/512/512/512
/// against a table of the same).
///
/// @throws std::runtime_error when the load or the check fails.
inline Flash2Module loadFlash2Module(const std::string& coPath,
                                     const std::string& funcName,
                                     int deviceOrdinal,
                                     unsigned int expectedBlockDim,
                                     const std::string& variantTag)
{
    HIPDNN_PLUGIN_LOG_INFO("HipFlash2 -- loading " << coPath << " fn=" << funcName << " on device "
                                                   << deviceOrdinal);

    auto kernelOpt = loadKernelModule(coPath, funcName.c_str(), deviceOrdinal);
    if(!kernelOpt)
    {
        const std::string msg = "HipFlash2 -- failed to load kernel from: " + coPath + " on device "
                                + std::to_string(deviceOrdinal);
        HIPDNN_PLUGIN_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }

    if(expectedBlockDim != 0)
    {
        int maxThreads = 0;
        const hipError_t attrErr = hipFuncGetAttribute(
            &maxThreads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernelOpt->function());
        if(attrErr != hipSuccess)
        {
            const std::string msg = "HipFlash2 -- hipFuncGetAttribute failed for " + coPath + ": "
                                    + std::string(hipGetErrorString(attrErr));
            HIPDNN_PLUGIN_LOG_ERROR(msg);
            throw std::runtime_error(msg);
        }
        if(maxThreads != static_cast<int>(expectedBlockDim))
        {
            const std::string msg
                = "HipFlash2 -- variant '" + variantTag + "' geometry mismatch: " + coPath
                  + " was built for " + std::to_string(maxThreads)
                  + " threads/block but the variant table claims "
                  + std::to_string(expectedBlockDim)
                  + ". The installed kernel object does not match the variant it is named for.";
            HIPDNN_PLUGIN_LOG_ERROR(msg);
            throw std::runtime_error(msg);
        }
    }

    return std::make_shared<const HipModuleGuard>(std::move(*kernelOpt));
}

/// Loads the plan's selected variant on each device it executes on. The variant was
/// chosen at build from the build device's CU count; it is kept for every device of the
/// architecture, since the choice is a performance decision and the object is the same
/// file on each.
class Flash2KernelResolver : public Flash2Kernels
{
public:
    /// @param expectedBlockDim Threads per block the object must have been built with,
    ///                         or 0 to skip the check (the legacy object).
    Flash2KernelResolver(const std::string& arch,
                         std::string coPath,
                         std::string funcName,
                         unsigned int expectedBlockDim,
                         std::string variantTag)
        : Flash2Kernels(arch, "HipFlash2 kernel '" + funcName + "' from " + coPath)
        , _coPath(std::move(coPath))
        , _funcName(std::move(funcName))
        , _expectedBlockDim(expectedBlockDim)
        , _variantTag(std::move(variantTag))
    {
    }

protected:
    Flash2Module resolveForDevice(int deviceOrdinal,
                                  const std::string& /*reportedArch*/) const override
    {
        return loadModule(deviceOrdinal);
    }

    /// Loads the variant on @p deviceOrdinal.
    virtual Flash2Module loadModule(int deviceOrdinal) const
    {
        return loadFlash2Module(_coPath, _funcName, deviceOrdinal, _expectedBlockDim, _variantTag);
    }

private:
    std::string _coPath;
    std::string _funcName;
    unsigned int _expectedBlockDim;
    std::string _variantTag;
};

} // namespace hip_flash2_engine
