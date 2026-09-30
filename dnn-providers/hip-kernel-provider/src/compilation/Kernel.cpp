// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Kernel.hpp"
#include "Program.hpp"
#include "Utils.hpp"
#include "device/ScopedDevice.hpp"

#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::compilation
{

namespace
{

// The hardware limits on a workgroup cluster. The messages match rocke's, which checks
// the same limits when it compiles a clustered kernel.
constexpr unsigned int MAX_CLUSTER_DIM = 15;
constexpr unsigned int MAX_CLUSTER_WORKGROUPS = 16;

std::string triple(unsigned int x, unsigned int y, unsigned int z)
{
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
}

#ifdef HIP_KERNEL_PROVIDER_HAS_CLUSTER_LAUNCH
/// Whether `deviceOrdinal` can launch clusters. hipDeviceProp_t is large and this sits on
/// the dispatch path, so each device is asked once.
bool deviceSupportsClusterLaunch(int deviceOrdinal)
{
    static std::mutex mutex;
    static std::map<int, bool> supported;

    const std::lock_guard<std::mutex> lock(mutex);
    const auto found = supported.find(deviceOrdinal);
    if(found != supported.end())
    {
        return found->second;
    }
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, deviceOrdinal));
    const bool answer = properties.clusterLaunch != 0;
    supported.emplace(deviceOrdinal, answer);
    return answer;
}
#endif

} // namespace

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

void Kernel::setClusterDims(unsigned int x, unsigned int y, unsigned int z)
{
    if(x < 1 || y < 1 || z < 1 || x > MAX_CLUSTER_DIM || y > MAX_CLUSTER_DIM || z > MAX_CLUSTER_DIM)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "kernel '" + _kernelName + "': cluster_dims " + triple(x, y, z)
                + ": each dimension must be in 1.." + std::to_string(MAX_CLUSTER_DIM));
    }
    const unsigned int workgroups = x * y * z;
    if(workgroups > MAX_CLUSTER_WORKGROUPS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "kernel '" + _kernelName + "': cluster_dims " + triple(x, y, z) + ": "
                + std::to_string(workgroups) + " workgroups exceeds the cluster limit of "
                + std::to_string(MAX_CLUSTER_WORKGROUPS));
    }
    _clusterDims = std::array<unsigned int, 3>{x, y, z};
}

void Kernel::launchImpl(hipStream_t stream, void** kernelParams) const
{
    // Checked before any HIP call, so a refused launch leaves no device state behind. A
    // partial cluster at the edge of the grid is not something the hardware can launch.
    if(_clusterDims)
    {
        const std::array<unsigned int, 3> grid{_gridX, _gridY, _gridZ};
        const auto& cluster = *_clusterDims;
        for(size_t axis = 0; axis < 3; ++axis)
        {
            if(grid[axis] % cluster[axis] != 0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                    "kernel '" + _kernelName + "': grid " + triple(_gridX, _gridY, _gridZ)
                        + " is not a multiple of cluster "
                        + triple(cluster[0], cluster[1], cluster[2]) + " in "
                        + std::string(1, "xyz"[axis]));
            }
        }
    }

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
        // correctness. Default stream tokens are device-relative, so they are exempt:
        // the bind below makes their current device the module's own.
        if(!hipdnn_plugin_sdk::isDefaultStream(stream))
        {
            // Seeded to -1, not 0: a runtime that returns hipSuccess without writing the
            // out-parameter would otherwise go unseen. Mirrors HandleDeviceResolver::deviceId.
            int streamDevice = -1;
            if(hipdnn_plugin_sdk::getDeviceFromStream(stream, &streamDevice) == hipSuccess
               && streamDevice >= 0 && streamDevice != _deviceOrdinal)
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

    if(_clusterDims)
    {
        launchClustered(stream, kernelParams);
        return;
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

void Kernel::launchClustered(hipStream_t stream, void** kernelParams) const
{
#ifdef HIP_KERNEL_PROVIDER_HAS_CLUSTER_LAUNCH
    // launchImpl has already made the module's device current, so for a device-bound
    // kernel the current device and _deviceOrdinal agree; asking HIP covers both.
    int deviceOrdinal = _deviceOrdinal;
    if(deviceOrdinal == NO_DEVICE)
    {
        HIP_CHECK(hipGetDevice(&deviceOrdinal));
    }
    const auto& cluster = *_clusterDims;
    if(!deviceSupportsClusterLaunch(deviceOrdinal))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "kernel '" + _kernelName + "': device " + std::to_string(deviceOrdinal)
                + " does not support cluster launch (cluster "
                + triple(cluster[0], cluster[1], cluster[2]) + ")");
    }

    hipLaunchAttribute attribute{};
    attribute.id = hipLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = cluster[0];
    attribute.val.clusterDim.y = cluster[1];
    attribute.val.clusterDim.z = cluster[2];

    HIP_LAUNCH_CONFIG config{};
    config.gridDimX = _gridX;
    config.gridDimY = _gridY;
    config.gridDimZ = _gridZ;
    config.blockDimX = _blockX;
    config.blockDimY = _blockY;
    config.blockDimZ = _blockZ;
    config.sharedMemBytes = _sharedMemBytes;
    config.hStream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;

    HIP_CHECK(hipDrvLaunchKernelEx(&config, _kernel, kernelParams, nullptr));
#else
    static_cast<void>(stream);
    static_cast<void>(kernelParams);
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "kernel '" + _kernelName
            + "' asks for a cluster launch, but this build's HIP headers do not provide "
              "hipDrvLaunchKernelEx with hipLaunchAttributeClusterDimension");
#endif
}

} // namespace hip_kernel_provider::compilation
