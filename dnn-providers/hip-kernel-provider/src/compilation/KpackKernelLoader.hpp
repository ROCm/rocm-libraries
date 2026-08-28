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
    /// `deviceOrdinal` both keys the cache entry and is made current across the load. A
    /// device that cannot be made current fails the load rather than yielding a foreign module.
    ///
    /// `expectedSha256` is the descriptor's declared digest of the decompressed code
    /// object, 64 lowercase hex. A TOC entry pointing at the wrong offset still
    /// decompresses cleanly, so without this the wrong kernel launches and nothing reports
    /// an error. KpackModuleCache both checks it and keys on it.
    ///
    /// @throws HipdnnPluginException, one distinct message per failing stage: archive
    ///         missing, archive unreadable, arch mismatch, toc_key absent, decompress,
    ///         digest mismatch, or module-load failure. The last, a missing symbol, is
    ///         raised by KpackProgram::getKernel, which is the only site that can see it.
    std::unique_ptr<ICompiledProgram> load(const std::filesystem::path& archive,
                                           const std::string& tocKey,
                                           const std::string& deviceArch,
                                           int deviceOrdinal,
                                           const std::string& symbol,
                                           const std::string& expectedSha256,
                                           const std::string& descriptorLabel) const;

private:
    KpackModuleCache& _moduleCache;
};

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
