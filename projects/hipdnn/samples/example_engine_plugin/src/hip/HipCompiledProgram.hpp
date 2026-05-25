// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ICompiledProgram.hpp"

#include <hip/hip_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace example_provider
{

/// Concrete ICompiledProgram that compiles kernels via HIPRTC.
///
/// Compiles embedded kernel source at runtime using hiprtcCompileProgram(),
/// extracts the compiled binary, and loads it as a HIP module. The module
/// remains loaded until the HipCompiledProgram is destroyed. Kernel functions can
/// be extracted by name via the ICompiledProgram interface.
class HipCompiledProgram : public ICompiledProgram
{
public:
    /// Compile the specified kernel source file with the given compiler options.
    /// @param kernelFileName The filename key used to look up the embedded source
    /// @param compilerOptions HIPRTC compiler options (e.g., "--offload-arch=gfx90a")
    HipCompiledProgram(const std::string& kernelFileName,
                       const std::vector<std::string>& compilerOptions);

    ~HipCompiledProgram() noexcept override;

    HipCompiledProgram(const HipCompiledProgram&) = delete;
    HipCompiledProgram& operator=(const HipCompiledProgram&) = delete;
    HipCompiledProgram(HipCompiledProgram&&) = delete;
    HipCompiledProgram& operator=(HipCompiledProgram&&) = delete;

    /// Get a runnable kernel from the loaded module (ICompiledProgram interface).
    /// @param kernelFunctionName The name of the kernel function (must match extern "C" name)
    std::unique_ptr<IRunnableKernel>
        getRunnableKernel(const std::string& kernelFunctionName) const override;

private:
    hipModule_t _module = nullptr;
};

} // namespace example_provider
