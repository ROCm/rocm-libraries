// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace hip_kernel_provider::compilation
{

/// A HIPRTC-compiled code object loaded as a module on one device.
///
/// Compiling targets an architecture and touches no device; loading binds the module to
/// one device's HIP context. The two are separate so one compiled binary can be loaded on
/// every device of its architecture without compiling it again.
class Program
{
public:
    /// HIPRTC-compiled code object bytes. Shared, so the loads of one binary on several
    /// devices keep a single copy.
    using Binary = std::shared_ptr<const std::vector<char>>;

    /// Compiles @p kernelFileName and loads it on the current device.
    Program(const std::string& kernelFileName, const std::vector<std::string>& options);

    /// Loads an already-compiled @p binary on @p deviceOrdinal.
    ///
    /// @param programName Names the program in failure messages.
    Program(std::string programName, Binary binary, int deviceOrdinal);

    /// Compiles @p kernelFileName with HIPRTC. Loads nothing and touches no device.
    ///
    /// @throws HipdnnPluginException when the source is unknown or does not compile.
    static Binary compileBinary(const std::string& kernelFileName,
                                const std::vector<std::string>& options);

    /// Resolves @p kernelName from the module, on the device the module was loaded on.
    hipFunction_t getKernel(const std::string& kernelName) const;

    /// The device the module was loaded on.
    int deviceOrdinal() const
    {
        return _deviceOrdinal;
    }

    ~Program();

    // Not movable: a moved-from Program would still name the module and unload it a
    // second time. Programs are shared through shared_ptr instead.
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&) = delete;
    Program& operator=(Program&&) = delete;

private:
    std::string _programName;
    hipModule_t _module = nullptr;
    Binary _binary;
    int _deviceOrdinal = 0;
};

} // namespace hip_kernel_provider::compilation
