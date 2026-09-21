// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "ICompiledProgram.hpp"
#include "KpackModule.hpp"

#include <memory>
#include <string>

namespace hip_kernel_provider::compilation
{

/// An ICompiledProgram backed by a module loaded from a kpack archive.
///
/// Holds the module by shared_ptr because one module serves every kernel in the archive
/// that shares its (toc_key, arch) -- see KpackModuleCache. Holds `descriptorLabel`
/// because ICompiledProgram::getKernel takes only the kernel name, and this is the sole
/// site that can raise the missing-symbol error; without the label that message could
/// name the symbol but not the descriptor that asked for it.
///
/// getKernel returns a plain compilation::Kernel holding the resolved hipFunction_t, so
/// launch goes through the existing hipModuleLaunchKernel path unchanged. There is no
/// kpack-specific launch code.
class KpackProgram : public ICompiledProgram
{
public:
    KpackProgram(std::shared_ptr<const KpackModule> module, std::string descriptorLabel);

    /// @throws HipdnnPluginException naming the descriptor and the symbol when
    ///         hipModuleGetFunction cannot resolve `kernelName` in the module.
    std::unique_ptr<IRunnableKernel> getKernel(const std::string& kernelName) const override;

    /// The module this program resolves symbols against. Exposed so a test can observe
    /// that two programs differing only by symbol were handed the same hipModule_t,
    /// which is otherwise invisible from outside the cache.
    hipModule_t module() const
    {
        return _module->module();
    }

private:
    std::shared_ptr<const KpackModule> _module;
    std::string _descriptorLabel;
};

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
