// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "compilation/IKernelCompiler.hpp"
#include "compilation/ModuleCache.hpp"
#include "compilation/Program.hpp"
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <memory>
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
        catch(hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return nullptr;
        }
    }
};

/// The same cache for kernels whose source is owned text rather than a key into the
/// embedded table -- a hiprtc_file kernel read out of a drop-in bundle.
///
/// Separate class rather than a second entry point on HipMlopsModuleCache because the
/// two load from different places and CRTP fixes one `load` per cache. The key is
/// (programName, options), the same shape as the embedded path's (filename, options):
/// callers pass the resolved bundle path as programName, so two bundles each holding
/// `attention.hip` with identical defines stay distinct entries. The source text is
/// deliberately not in the key -- the resolved path names it, exactly as the embedded
/// path's filename names its table entry.
class HipMlopsSourceModuleCache : public hip_kernel_provider::compilation::ModuleCache<
                                      HipMlopsSourceModuleCache,
                                      std::shared_ptr<compilation::Program>,
                                      const std::string&,
                                      const std::string&,
                                      const std::vector<compilation::KernelHeader>&,
                                      const std::vector<std::string>&>
{
public:
    static std::string makeKey(const std::string& /*sourceText*/,
                               const std::string& programName,
                               const std::vector<compilation::KernelHeader>& /*headers*/,
                               const std::vector<std::string>& options)
    {
        std::string key(programName);

        for(const std::string& option : options)
        {
            key.append("::");
            key.append(option);
        }

        return key;
    }

    static std::shared_ptr<compilation::Program>
        load(const std::string& sourceText,
             const std::string& programName,
             const std::vector<compilation::KernelHeader>& headers,
             const std::vector<std::string>& options)
    {
        try
        {
            return std::make_shared<compilation::Program>(
                sourceText, programName, headers, options);
        }
        catch(hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return nullptr;
        }
    }
};

} // namespace hip_kernel_provider
