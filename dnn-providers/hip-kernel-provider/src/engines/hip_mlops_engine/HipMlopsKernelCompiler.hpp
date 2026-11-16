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
        // Try to use a cached compilation::Program object based on a key of filename and
        // compilation options. A nullptr will be returned if this fails, in which case
        // create the compilation objects directly, which will generate a more helpful
        // exception if the program cannot be created.
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
