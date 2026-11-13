// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "compilation/ModuleCache.hpp"
#include "compilation/Program.hpp"
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <memory>
#include <string>
#include <vector>

namespace hip_kernel_provider
{

/// HIPRTC binaries by (kernel file, compile options). A binary is compiled for the
/// architecture its options name and touches no device, so one entry serves every
/// device of that architecture.
class HipMlopsBinaryCache
    : public hip_kernel_provider::compilation::ModuleCache<HipMlopsBinaryCache,
                                                           compilation::Program::Binary,
                                                           const std::string&,
                                                           const std::vector<std::string>&>
{
public:
    static std::string makeKey(const std::string& kernelFileName,
                               const std::vector<std::string>& options)
    {
        std::string key(kernelFileName);

        for(const std::string& option : options)
        {
            key.append("::");
            key.append(option);
        }

        return key;
    }

    static compilation::Program::Binary load(const std::string& kernelFileName,
                                             const std::vector<std::string>& options)
    {
        try
        {
            return compilation::Program::compileBinary(kernelFileName, options);
        }
        catch(hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return nullptr;
        }
    }
};

/// Loaded programs by (kernel file, compile options, device ordinal).
///
/// The ordinal is in the key because a module belongs to the device it was loaded on:
/// every device has its own HIP context, so two devices of one architecture must not
/// share an entry. Each load compiles nothing new -- the binary comes from
/// binaryCache(), shared across devices.
class HipMlopsModuleCache
    : public hip_kernel_provider::compilation::ModuleCache<HipMlopsModuleCache,
                                                           std::shared_ptr<compilation::Program>,
                                                           const std::string&,
                                                           const std::vector<std::string>&,
                                                           int>
{
public:
    static std::string makeKey(const std::string& kernelFileName,
                               const std::vector<std::string>& options,
                               int deviceOrdinal)
    {
        return HipMlopsBinaryCache::makeKey(kernelFileName, options) + "@"
               + std::to_string(deviceOrdinal);
    }

    static std::shared_ptr<compilation::Program> load(const std::string& kernelFileName,
                                                      const std::vector<std::string>& options,
                                                      int deviceOrdinal)
    {
        const auto binary = binaryCache().getOrLoad(kernelFileName, options);
        if(binary == nullptr)
        {
            return nullptr;
        }

        try
        {
            return std::make_shared<compilation::Program>(kernelFileName, binary, deviceOrdinal);
        }
        catch(hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return nullptr;
        }
    }

    /// Process-lifetime: binaries are device-independent, so every module cache shares one.
    static HipMlopsBinaryCache& binaryCache()
    {
        static HipMlopsBinaryCache s_cache;
        return s_cache;
    }
};

} // namespace hip_kernel_provider
