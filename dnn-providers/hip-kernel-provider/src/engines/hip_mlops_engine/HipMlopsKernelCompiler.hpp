// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipMlopsModuleCache.hpp"
#include "compilation/CompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/ProgramFetchRecording.hpp"

#include <chrono>
#include <memory>

namespace hip_kernel_provider
{

class HipMlopsKernelCompiler : public compilation::IKernelCompiler
{
public:
    std::unique_ptr<compilation::ICompiledProgram>
        compile(const std::string& kernelFileName,
                const std::vector<std::string>& options) const override
    {
        // A compilation::Program cached on a key of file name and compile options. A program
        // that fails to build is not cached; its own exception is rethrown, rather than the
        // program built a second time to produce it.
        HipMlopsModuleCache::lastLoadFailure().reset();
        const auto started = std::chrono::steady_clock::now();
        const auto recorded = compilation::ProgramFetchRecording::recordedOnThisThread();
        auto program = moduleCache().getOrLoad(kernelFileName, options);
        if(program)
        {
            // A lookup that constructed the program recorded where it came from; one that
            // did not was served by a module this process already holds.
            if(compilation::ProgramFetchRecording::recordedOnThisThread() == recorded)
            {
                compilation::ProgramFetchRecording::record(
                    {kernelFileName,
                     compilation::CodeSource::IN_PROCESS,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()
                                                               - started)
                         .count()});
            }
            return std::make_unique<compilation::CompiledProgram>(program);
        }
        if(auto& failure = HipMlopsModuleCache::lastLoadFailure(); failure.has_value())
        {
            const auto status = failure->getStatus();
            auto message = failure->getMessage();
            failure.reset();
            throw hipdnn_plugin_sdk::HipdnnPluginException(status, std::move(message));
        }
        return std::make_unique<compilation::CompiledProgram>(
            std::make_shared<compilation::Program>(kernelFileName, options));
    }

    static HipMlopsModuleCache& moduleCache()
    {
        static HipMlopsModuleCache s_cache;
        return s_cache;
    }
};

} // namespace hip_kernel_provider
