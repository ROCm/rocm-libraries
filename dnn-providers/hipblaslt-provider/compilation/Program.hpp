// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

namespace hipblaslt_plugin::compilation
{

// Compiles a kernel source embedded in the plugin (cmake/KernelEmbedding.cmake) with
// HIPRTC and returns its code object; needs no device. Throws HipdnnPluginException for
// an unknown source name and, with the compile log, when HIPRTC rejects the source.
std::vector<char> compileKernelSource(const std::string& sourceName,
                                      const std::vector<std::string>& options);

// A kernel source embedded in the plugin (cmake/KernelEmbedding.cmake), compiled with
// HIPRTC and loaded as a module on the current device. Ported from hip-kernel-provider's
// compilation/Program.
class Program
{
public:
    // Throws as compileKernelSource() does, and when the module does not load.
    Program(std::string sourceName, const std::vector<std::string>& options);
    ~Program();

    // A Program owns its module and is shared, not moved: a moved-from Program would
    // unload the module a second time.
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&) = delete;
    Program& operator=(Program&&) = delete;

    // The entry point `kernelName` (an extern "C" __global__ function); throws when the
    // module has none.
    hipFunction_t getKernel(const std::string& kernelName) const;

private:
    std::string _programName;
    std::vector<char> _binary;
    hipModule_t _module = nullptr;
};

} // namespace hipblaslt_plugin::compilation
