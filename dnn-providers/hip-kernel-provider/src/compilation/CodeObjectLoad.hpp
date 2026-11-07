// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "ICompiledProgram.hpp"
#include "IRunnableKernel.hpp"

namespace hip_kernel_provider::compilation
{

/// The step of loadCodeObjectOnDevice() that failed.
enum class CodeObjectLoadStage
{
    BIND_DEVICE,
    MODULE_LOAD,
    SYMBOL_LOOKUP,
};

/// A failure of loadCodeObjectOnDevice(). The message names the step, the device and the
/// label. hipStatus() is the HIP error of MODULE_LOAD and SYMBOL_LOOKUP.
class CodeObjectLoadFailure : public std::runtime_error
{
public:
    CodeObjectLoadFailure(CodeObjectLoadStage stage,
                          hipError_t hipStatus,
                          const std::string& message)
        : std::runtime_error(message)
        , _stage(stage)
        , _hipStatus(hipStatus)
    {
    }

    CodeObjectLoadStage stage() const
    {
        return _stage;
    }

    hipError_t hipStatus() const
    {
        return _hipStatus;
    }

private:
    CodeObjectLoadStage _stage;
    hipError_t _hipStatus;
};

/// A loaded module and one kernel resolved from it. The program keeps the module loaded
/// while the kernel exists.
struct LoadedCodeObject
{
    std::unique_ptr<ICompiledProgram> program;
    std::unique_ptr<IRunnableKernel> kernel;
};

/// Loads @p bytes as a module on @p deviceOrdinal and resolves @p symbol from it. Works
/// for a code object of any source kind. Uses no module cache, so each call loads its own
/// module.
///
/// @param label Names the kernel in the failure messages.
/// @throws CodeObjectLoadFailure when the device cannot be made current, when the driver
///         refuses the bytes, or when the module has no @p symbol.
LoadedCodeObject loadCodeObjectOnDevice(const std::vector<uint8_t>& bytes,
                                        const std::string& symbol,
                                        int deviceOrdinal,
                                        const std::string& label);

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
