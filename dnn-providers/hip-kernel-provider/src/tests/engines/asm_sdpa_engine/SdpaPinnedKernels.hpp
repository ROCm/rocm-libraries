// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/asm_sdpa_engine/plans/SdpaKernelResolvers.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace asm_sdpa_engine
{

/// Kernels answered for device 0 only. For plan tests that fail before any launch, where
/// what the plan would launch is never reached.
template <typename Value>
class PinnedKernels : public hip_kernel_provider::device::PerDeviceResolver<Value>
{
public:
    explicit PinnedKernels(Value value)
        : hip_kernel_provider::device::PerDeviceResolver<Value>("gfx942", "pinned test kernels")
    {
        this->seed(0, std::move(value));
    }

protected:
    Value resolveForDevice(int /*deviceOrdinal*/,
                           const std::string& /*reportedArch*/) const override
    {
        throw std::logic_error("PinnedKernels: only device 0 is pinned");
    }
};

inline std::unique_ptr<SdpaFwdKernels> pinnedFwdKernels()
{
    return std::make_unique<PinnedKernels<CachedModule>>(std::make_shared<HipModuleGuard>());
}

inline std::unique_ptr<SdpaBwdKernels> pinnedBwdKernels()
{
    return std::make_unique<PinnedKernels<SdpaBwdModules>>(
        SdpaBwdModules{std::make_shared<HipModuleGuard>(),
                       std::make_shared<HipModuleGuard>(),
                       std::make_shared<HipModuleGuard>()});
}

} // namespace asm_sdpa_engine
