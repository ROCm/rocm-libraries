// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "compilation/ModuleCache.hpp"
#include "compilation/Program.hpp"
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hip_kernel_provider
{

class HipMlopsModuleCache
    : public hip_kernel_provider::compilation::ModuleCache<HipMlopsModuleCache,
                                                           std::shared_ptr<compilation::Program>,
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

    static std::shared_ptr<compilation::Program> load(const std::string& kernelFileName,
                                                      const std::vector<std::string>& options)
    {
        try
        {
            return std::make_shared<compilation::Program>(kernelFileName, options);
        }
        catch(hipdnn_plugin_sdk::HipdnnPluginException& error)
        {
            lastLoadFailure() = error;
            return nullptr;
        }
    }

    /// Why the last load() on this thread returned null, so a caller can report the failure
    /// without building the program a second time. load() runs on the thread that called
    /// getOrLoad(), so this is that call's failure.
    static std::optional<hipdnn_plugin_sdk::HipdnnPluginException>& lastLoadFailure()
    {
        thread_local std::optional<hipdnn_plugin_sdk::HipdnnPluginException> s_failure;
        return s_failure;
    }
};

} // namespace hip_kernel_provider
