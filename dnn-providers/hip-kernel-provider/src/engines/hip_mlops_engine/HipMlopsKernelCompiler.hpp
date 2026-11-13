// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipMlopsModuleCache.hpp"
#include "compilation/CompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "device/ScopedDevice.hpp"

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hip_kernel_provider
{

class HipMlopsKernelCompiler : public compilation::IKernelCompiler
{
public:
    /// Compiles @p kernelFileName and loads it on the current device -- the one the caller
    /// read the device properties behind @p options from. The kernels of the result
    /// launch on the device of the stream they are given, loading the program there on
    /// first use, so a plan built under one handle can execute under another on a device
    /// of the same architecture.
    std::unique_ptr<compilation::ICompiledProgram>
        compile(const std::string& kernelFileName,
                const std::vector<std::string>& options) const override
    {
        auto program = loadOn(kernelFileName, options, device::currentDeviceOrdinal());
        auto arch = targetArch(options, program->deviceOrdinal());
        return std::make_unique<compilation::CompiledProgram>(
            std::move(program), std::move(arch), [kernelFileName, options](int deviceOrdinal) {
                return loadOn(kernelFileName, options, deviceOrdinal);
            });
    }

    static HipMlopsModuleCache& moduleCache()
    {
        static HipMlopsModuleCache s_cache;
        return s_cache;
    }

private:
    /// The program compiled from @p kernelFileName with @p options and loaded on
    /// @p deviceOrdinal, from the cache when present.
    static std::shared_ptr<compilation::Program> loadOn(const std::string& kernelFileName,
                                                        const std::vector<std::string>& options,
                                                        int deviceOrdinal)
    {
        // The cache returns nullptr when it fails to create the program, in which case
        // create the compilation objects directly, which will generate a more helpful
        // exception.
        auto program = moduleCache().getOrLoad(kernelFileName, options, deviceOrdinal);
        if(program)
        {
            return program;
        }
        return std::make_shared<compilation::Program>(
            kernelFileName,
            compilation::Program::compileBinary(kernelFileName, options),
            deviceOrdinal);
    }

    /// The architecture a binary compiled with @p options targets: the --offload-arch it
    /// names, or, when it names none, the architecture HIPRTC defaults to -- that of the
    /// device it was loaded on.
    static std::string targetArch(const std::vector<std::string>& options, int deviceOrdinal)
    {
        constexpr std::string_view K_OFFLOAD_ARCH = "--offload-arch=";
        for(const auto& option : options)
        {
            if(option.compare(0, K_OFFLOAD_ARCH.size(), K_OFFLOAD_ARCH) == 0)
            {
                return option.substr(K_OFFLOAD_ARCH.size());
            }
        }

        hipDeviceProp_t properties{};
        const hipError_t status = hipGetDeviceProperties(&properties, deviceOrdinal);
        if(status != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "cannot query device " + std::to_string(deviceOrdinal)
                    + " for the architecture its program was compiled for: "
                    + hipGetErrorString(status));
        }
        return properties.gcnArchName;
    }
};

} // namespace hip_kernel_provider
