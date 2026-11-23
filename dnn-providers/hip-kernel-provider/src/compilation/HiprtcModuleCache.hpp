// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ModuleCache.hpp"
#include "Program.hpp"
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <memory>
#include <string>
#include <vector>

namespace hip_kernel_provider::compilation
{

class HiprtcModuleCache : public ModuleCache<HiprtcModuleCache,
                                             std::shared_ptr<Program>,
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

    static std::shared_ptr<Program> load(const std::string& kernelFileName,
                                         const std::vector<std::string>& options)
    {
        try
        {
            return std::make_shared<Program>(kernelFileName, options);
        }
        catch(hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return nullptr;
        }
    }
};

} // namespace hip_kernel_provider::compilation
