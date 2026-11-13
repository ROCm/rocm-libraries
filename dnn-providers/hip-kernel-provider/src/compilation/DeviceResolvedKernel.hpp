// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "IRunnableKernel.hpp"
#include "Program.hpp"
#include "device/PerDeviceResolver.hpp"

#include <functional>
#include <memory>
#include <string>

#include <hip/hip_runtime_api.h>

namespace hip_kernel_provider::compilation
{

/// Loads a compiled program on a device and returns the program loaded there.
using ProgramLoader = std::function<std::shared_ptr<Program>(int deviceOrdinal)>;

/// A HIPRTC kernel that launches on the device of the stream it is launched on.
///
/// A HIPRTC binary is compiled for an architecture, but the module it is loaded into
/// belongs to one device. A plan holding this kernel can therefore be executed under a
/// handle on any device of that architecture: each launch resolves the device from its
/// stream, and the first launch on a device loads that device's module (see
/// device::PerDeviceResolver for the admission rule). A launch on a device already seen
/// queries and loads nothing.
///
/// This keeps the IRunnableKernel interface unchanged: launchImpl already receives the
/// stream, which is all per-device resolution needs, so the plans holding a
/// unique_ptr<IRunnableKernel> do not change.
class DeviceResolvedKernel : public IRunnableKernel
{
public:
    /// One device's program and the function resolved out of it. The program is held
    /// because a hipFunction_t does not own the module it came from.
    struct Resolved
    {
        std::shared_ptr<Program> program;
        hipFunction_t function = nullptr;
    };

    using Resolver = device::PerDeviceResolver<Resolved>;

    DeviceResolvedKernel(std::string kernelName, std::unique_ptr<Resolver> resolver);

    void setBlockSize(unsigned int x, unsigned int y = 1, unsigned int z = 1) override;
    void setGridSize(unsigned int x, unsigned int y = 1, unsigned int z = 1) override;
    void setSharedMemBytes(unsigned int bytes) override;

    /// What a launch on @p deviceOrdinal runs. Exposed so a test can observe resolution
    /// without launching.
    Resolved resolvedFor(int deviceOrdinal) const;

protected:
    void launchImpl(hipStream_t stream, void** kernelParams) const override;

private:
    std::string _kernelName;
    std::unique_ptr<Resolver> _resolver;
    unsigned int _blockX = 1;
    unsigned int _blockY = 1;
    unsigned int _blockZ = 1;
    unsigned int _gridX = 1;
    unsigned int _gridY = 1;
    unsigned int _gridZ = 1;
    unsigned int _sharedMemBytes = 0;
};

/// Resolves one kernel of a compiled program on each device, through a loader that
/// returns the program loaded on that device.
class ProgramKernelResolver : public DeviceResolvedKernel::Resolver
{
public:
    /// @param builtArch The architecture the program was compiled for.
    ProgramKernelResolver(const std::string& builtArch,
                          std::string kernelName,
                          ProgramLoader loader);

protected:
    DeviceResolvedKernel::Resolved resolveForDevice(int deviceOrdinal,
                                                    const std::string& reportedArch) const override;

private:
    std::string _kernelName;
    ProgramLoader _loader;
};

} // namespace hip_kernel_provider::compilation
