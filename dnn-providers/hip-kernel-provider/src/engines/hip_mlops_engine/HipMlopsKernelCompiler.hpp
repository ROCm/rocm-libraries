// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipMlopsModuleCache.hpp"
#include "compilation/CompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"

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
        auto program = moduleCache().getOrLoad(kernelFileName, options);
        if(program)
        {
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
