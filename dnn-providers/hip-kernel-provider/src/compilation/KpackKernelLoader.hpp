// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "ICompiledProgram.hpp"
#include "KpackModuleCache.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace hip_kernel_provider::compilation
{

/// Turns a (kpack archive, toc_key, device arch, symbol) tuple into a runnable program.
///
/// Concrete rather than an IKernelCompiler implementation: compile(fileName, options)
/// is HIPRTC-shaped and cannot express (library, tocKey), and nothing substitutes this
/// loader -- the missing-archive test uses a genuinely nonexistent path.
///
/// The module cache is injected rather than reached as a singleton, so two loaders in
/// one process do not silently share state. It must outlive the loader.
class KpackKernelLoader
{
public:
    explicit KpackKernelLoader(KpackModuleCache& moduleCache)
        : _moduleCache(moduleCache)
    {
    }

    /// Loads (or reuses) the module holding `tocKey` for `deviceArch` and returns a
    /// program over it.
    ///
    /// `symbol` is here for the message text only and is deliberately NOT part of the
    /// cache key. Missing archive, unreadable archive, and arch mismatch are all raised
    /// before any symbol lookup, yet every message must name both the descriptor and the
    /// symbol. Folding `symbol` into KpackModuleCache::makeKey would defeat the sharing
    /// of one module across kernels differing only by entry point.
    ///
    /// @throws HipdnnPluginException, one distinct message per failing stage: archive
    ///         missing, archive unreadable, arch mismatch, toc_key absent, decompress
    ///         or module-load failure. The sixth, a missing symbol, is raised by
    ///         KpackProgram::getKernel, which is the only site that can see it.
    std::unique_ptr<ICompiledProgram> load(const std::filesystem::path& archive,
                                           const std::string& tocKey,
                                           const std::string& deviceArch,
                                           const std::string& symbol,
                                           const std::string& descriptorLabel) const;

private:
    KpackModuleCache& _moduleCache;
};

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
