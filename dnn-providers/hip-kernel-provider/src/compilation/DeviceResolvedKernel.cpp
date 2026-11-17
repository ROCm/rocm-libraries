// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "DeviceResolvedKernel.hpp"
#include "Utils.hpp"
#include "device/LaunchDeviceOrdinal.hpp"
#include "device/ScopedDevice.hpp"

#include <utility>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::compilation
{

DeviceResolvedKernel::DeviceResolvedKernel(std::string kernelName,
                                           std::unique_ptr<Resolver> resolver)
    : _kernelName(std::move(kernelName))
    , _resolver(std::move(resolver))
{
}

void DeviceResolvedKernel::setBlockSize(unsigned int x, unsigned int y, unsigned int z)
{
    _blockX = x;
    _blockY = y;
    _blockZ = z;
}

void DeviceResolvedKernel::setGridSize(unsigned int x, unsigned int y, unsigned int z)
{
    _gridX = x;
    _gridY = y;
    _gridZ = z;
}

void DeviceResolvedKernel::setSharedMemBytes(unsigned int bytes)
{
    _sharedMemBytes = bytes;
}

DeviceResolvedKernel::Resolved DeviceResolvedKernel::resolvedFor(int deviceOrdinal) const
{
    return _resolver->forOrdinal(deviceOrdinal);
}

void DeviceResolvedKernel::launchImpl(hipStream_t stream, void** kernelParams) const
{
    // The device the stream names, by the rule plan build used; the module launched is the
    // one loaded there. No stream-versus-module check is needed: the module is chosen from
    // the stream.
    const int deviceOrdinal = device::launchDeviceOrdinal(stream);
    const Resolved resolved = _resolver->forOrdinal(deviceOrdinal);

    // A module has to be launched from the device it was loaded on, and the stream's
    // device need not be the current one.
    const device::ScopedDevice binding(deviceOrdinal);
    if(!binding.bound())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "cannot make device " + std::to_string(deviceOrdinal) + " current to launch kernel '"
                + _kernelName + "'");
    }

    HIP_CHECK(hipModuleLaunchKernel(resolved.function,
                                    _gridX,
                                    _gridY,
                                    _gridZ,
                                    _blockX,
                                    _blockY,
                                    _blockZ,
                                    _sharedMemBytes,
                                    stream,
                                    kernelParams,
                                    nullptr));
}

ProgramKernelResolver::ProgramKernelResolver(const std::string& builtArch,
                                             std::string kernelName,
                                             ProgramLoader loader)
    : DeviceResolvedKernel::Resolver(builtArch, "kernel '" + kernelName + "'")
    , _kernelName(std::move(kernelName))
    , _loader(std::move(loader))
{
}

DeviceResolvedKernel::Resolved
    ProgramKernelResolver::resolveForDevice(int deviceOrdinal,
                                            const std::string& /*reportedArch*/) const
{
    auto program = _loader(deviceOrdinal);
    if(program == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            subject() + ": no program was loaded on device " + std::to_string(deviceOrdinal));
    }
    auto* const function = program->getKernel(_kernelName);
    return DeviceResolvedKernel::Resolved{std::move(program), function};
}

} // namespace hip_kernel_provider::compilation
