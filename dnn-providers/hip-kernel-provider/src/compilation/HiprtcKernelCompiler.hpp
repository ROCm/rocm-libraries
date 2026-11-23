// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "CompiledProgram.hpp"
#include "HiprtcModuleCache.hpp"
#include "IKernelCompiler.hpp"

#include <memory>

namespace hip_kernel_provider::compilation
{

class HiprtcKernelCompiler : public IKernelCompiler
{
public:
    std::unique_ptr<ICompiledProgram>
        compile(const std::string& kernelFileName,
                const std::vector<std::string>& options) const override
    {
        // Try to use a cached Program object based on a key of filename and compilation
        // options. A nullptr will be returned if this fails, in which case create the
        // compilation objects directly, which will generate a more helpful exception if the
        // program cannot be created.
        auto program = moduleCache().getOrLoad(kernelFileName, options);
        if(program)
        {
            return std::make_unique<CompiledProgram>(program);
        }
        return std::make_unique<CompiledProgram>(
            std::make_shared<Program>(kernelFileName, options));
    }

    static HiprtcModuleCache& moduleCache()
    {
        static HiprtcModuleCache s_cache;
        return s_cache;
    }
};

} // namespace hip_kernel_provider::compilation
