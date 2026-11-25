// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <string>
#include <vector>

namespace hip_kernel_provider::compilation
{

/// A test hook: a program whose name contains this variable's value fails to compile, with a
/// real HIPRTC error, and is never read from or written to the disk cache. Read at every
/// compile. Unset in production.
inline constexpr const char* TEST_FAIL_COMPILE_ENV = "HIP_KERNEL_PROVIDER_TEST_FAIL_COMPILE";

/// Set to 1/true/on/yes: the provider never compiles with HIPRTC. A program the disk cache
/// holds still loads; any other fails to build, so a plan that needs it declines at plan build.
/// Read at every compile.
inline constexpr const char* DISABLE_HIPRTC_ENV = "HIP_KERNEL_PROVIDER_DISABLE_HIPRTC";

class Program
{
public:
    Program(std::string kernelFileName, const std::vector<std::string>& options);

    /// The key the code object of @p kernelFileName compiled with @p options is cached under:
    /// a digest of the compiler version, the source, every embedded include and every option.
    /// Two compiles share a code object exactly when their keys are equal.
    static std::string codeObjectKey(const std::string& kernelFileName,
                                     const std::vector<std::string>& options);
    hipFunction_t getKernel(const std::string& kernelName) const;

    ~Program();

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&) = default;
    Program& operator=(Program&&) = default;

private:
    std::string _programName;
    hipModule_t _module = nullptr;
    std::vector<char> _binary;
};

} // namespace hip_kernel_provider::compilation
