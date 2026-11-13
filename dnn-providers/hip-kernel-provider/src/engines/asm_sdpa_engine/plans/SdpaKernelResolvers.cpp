// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaKernelResolvers.hpp"
#include "asm/AsmKernelPath.hpp"

#include <utility>

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace asm_sdpa_engine
{

namespace
{

CachedModule loadOrThrow(SdpaModuleCache& cache,
                         const std::string& tocKey,
                         const std::string& arch,
                         const std::string& knlName,
                         int deviceOrdinal)
{
    auto module = cache.getOrLoad(tocKey, arch, deviceOrdinal, knlName.c_str());
    if(!module)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "failed to load SDPA kernel tocKey='" + tocKey + "' arch='" + arch + "' on device "
                + std::to_string(deviceOrdinal));
    }
    return module;
}

} // namespace

bool isMi308ChipId(int chipId)
{
    return chipId == 0x74a2 || chipId == 0x74a8 || chipId == 0x74b6 || chipId == 0x74bc;
}

std::string fwdKernelTocKey(std::string coName, const std::string& archId, bool isMi308)
{
    if(archId == "gfx942")
    {
        const auto pos = coName.rfind('/');
        if(isMi308)
        {
            coName = coName.substr(0, pos + 1) + "MI308/" + coName.substr(pos + 1);
        }
        else
        {
            coName = coName.substr(0, pos + 1) + "MI300/" + coName.substr(pos + 1);
        }
    }
    return asm_kernels::getAsmKernelTocKey(coName);
}

SdpaFwdKernelResolver::SdpaFwdKernelResolver(SdpaModuleCache& cache,
                                             const std::string& arch,
                                             std::string coName,
                                             std::string knlName)
    : SdpaFwdKernels(arch, "SDPA forward kernel '" + knlName + "'")
    , _cache(&cache)
    , _coName(std::move(coName))
    , _knlName(std::move(knlName))
{
}

CachedModule SdpaFwdKernelResolver::resolveForDevice(int deviceOrdinal,
                                                     const std::string& /*reportedArch*/) const
{
    // Only gfx942 splits its binaries by board, so only gfx942 pays for the query.
    const bool isMi308 = builtArch() == "gfx942" && queryIsMi308(deviceOrdinal);
    const std::string tocKey = fwdKernelTocKey(_coName, builtArch(), isMi308);

    HIPDNN_PLUGIN_LOG_INFO("Using kernel with tocKey: " << tocKey << " arch: " << builtArch()
                                                        << " device: " << deviceOrdinal);
    return loadModule(tocKey, deviceOrdinal);
}

bool SdpaFwdKernelResolver::queryIsMi308(int deviceOrdinal) const
{
    int chipId = 0;
    const hipError_t status
        = hipDeviceGetAttribute(&chipId, hipDeviceAttributePciChipId, deviceOrdinal);
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            subject() + ": hipDeviceGetAttribute failed for device " + std::to_string(deviceOrdinal)
                + " with error code: " + std::to_string(status));
    }

    HIPDNN_PLUGIN_LOG_INFO("pciDeviceID  = " << std::hex << std::to_string(chipId));
    return isMi308ChipId(chipId);
}

CachedModule SdpaFwdKernelResolver::loadModule(const std::string& tocKey, int deviceOrdinal) const
{
    return loadOrThrow(*_cache, tocKey, builtArch(), _knlName, deviceOrdinal);
}

SdpaBwdKernelResolver::SdpaBwdKernelResolver(SdpaModuleCache& cache,
                                             const std::string& arch,
                                             SdpaKernelRef odo,
                                             SdpaKernelRef dqdkdv,
                                             std::optional<SdpaKernelRef> post)
    : SdpaBwdKernels(arch, "SDPA backward kernels '" + dqdkdv.knlName + "'")
    , _cache(&cache)
    , _odo(std::move(odo))
    , _dqdkdv(std::move(dqdkdv))
    , _post(std::move(post))
{
}

SdpaBwdModules SdpaBwdKernelResolver::resolveForDevice(int deviceOrdinal,
                                                       const std::string& /*reportedArch*/) const
{
    SdpaBwdModules modules;
    modules.odo = loadModule(_odo, deviceOrdinal);
    modules.dqdkdv = loadModule(_dqdkdv, deviceOrdinal);
    if(_post)
    {
        modules.post = loadModule(*_post, deviceOrdinal);
    }
    return modules;
}

CachedModule SdpaBwdKernelResolver::loadModule(const SdpaKernelRef& kernel, int deviceOrdinal) const
{
    return loadOrThrow(*_cache, kernel.tocKey, builtArch(), kernel.knlName, deviceOrdinal);
}

} // namespace asm_sdpa_engine
