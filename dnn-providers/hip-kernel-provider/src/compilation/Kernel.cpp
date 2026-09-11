// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Kernel.hpp"
#include "Program.hpp"
#include "Utils.hpp"
#include "device/ScopedDevice.hpp"

#include <optional>
#include <string>
#include <utility>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::compilation
{

Kernel::Kernel(const Program& program, const std::string& kernelName)
    : _kernelName(kernelName)
    , _kernel(program.getKernel(kernelName))
{
}

Kernel::Kernel(hipFunction_t kernel, std::string kernelName, int deviceOrdinal)
    : _kernelName(std::move(kernelName))
    , _kernel(kernel)
    , _deviceOrdinal(deviceOrdinal)
{
}

void Kernel::setBlockSize(unsigned int x, unsigned int y, unsigned int z)
{
    _blockX = x;
    _blockY = y;
    _blockZ = z;
}

void Kernel::setGridSize(unsigned int x, unsigned int y, unsigned int z)
{
    _gridX = x;
    _gridY = y;
    _gridZ = z;
}

void Kernel::setSharedMemBytes(unsigned int bytes)
{
    _sharedMemBytes = bytes;
}

void Kernel::launchImpl(hipStream_t stream, void** kernelParams) const
{
    // A module belongs to the device that was current at hipModuleLoadData, and the launch
    // has to be made from there. Measured on a two-GPU MI300X: with the module on device 0
    // and device 1 current, the launch is refused with "invalid resource handle" even though
    // the stream is device 0's own. The cache entry outlives the dispatch that filled it, so
    // by the next dispatch the current device is whatever the application last set.
    std::optional<device::ScopedDevice> binding;
    if(_deviceOrdinal != NO_DEVICE)
    {
        // The same measurement refused a stream belonging to a THIRD device while the
        // module's own device was correctly current -- binding cannot rescue that. HIP
        // already refuses it, so what this check buys is the diagnosis rather than the
        // correctness. The null stream is exempt: it follows the current device, which the
        // bind below is about to make the module's own.
        if(stream != nullptr)
        {
            // Seeded to -1, not 0: a runtime that returns hipSuccess without writing the
            // out-parameter would otherwise go unseen. Mirrors HandleDeviceResolver::deviceId.
            int streamDevice = -1;
            if(hipStreamGetDevice(stream, &streamDevice) == hipSuccess && streamDevice >= 0
               && streamDevice != _deviceOrdinal)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "kernel '" + _kernelName + "' was loaded on device "
                        + std::to_string(_deviceOrdinal) + " but is being launched on a stream "
                        + "belonging to device " + std::to_string(streamDevice)
                        + "; a plan is being executed under a handle from another device");
            }
        }

        binding.emplace(_deviceOrdinal);
        if(!binding->bound())
        {
            // Throwing matches KpackModuleCache::load, which refuses a load it cannot bind.
            // Launching anyway would run on whatever device happens to be current and
            // silently write the wrong buffers.
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "cannot make device " + std::to_string(_deviceOrdinal)
                    + " current to launch kernel '" + _kernelName + "'");
        }
    }

    HIP_CHECK(hipModuleLaunchKernel(_kernel,
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

} // namespace hip_kernel_provider::compilation
