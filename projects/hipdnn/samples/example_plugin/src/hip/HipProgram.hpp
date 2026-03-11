// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime_api.h>

#include <string>
#include <vector>

namespace example_plugin
{

/// Wraps HIPRTC compilation and HIP module loading.
///
/// Compiles embedded kernel source at runtime using hiprtcCompileProgram(),
/// extracts the compiled binary, and loads it as a HIP module. The module
/// remains loaded until the HipProgram is destroyed.
class HipProgram
{
public:
    /// Compile the specified kernel source file with the given compiler options.
    /// @param kernelFileName The filename key used to look up the embedded source
    /// @param compilerOptions HIPRTC compiler options (e.g., "--offload-arch=gfx90a")
    HipProgram(const std::string& kernelFileName, const std::vector<std::string>& compilerOptions);

    ~HipProgram();

    HipProgram(const HipProgram&) = delete;
    HipProgram& operator=(const HipProgram&) = delete;
    HipProgram(HipProgram&&) = delete;
    HipProgram& operator=(HipProgram&&) = delete;

    /// Get a kernel function handle from the loaded module.
    /// @param kernelName The name of the kernel function (must match extern "C" name)
    hipFunction_t getKernel(const std::string& kernelName) const;

private:
    hipModule_t _module = nullptr;
};

} // namespace example_plugin
