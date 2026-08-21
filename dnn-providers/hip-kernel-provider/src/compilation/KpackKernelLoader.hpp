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
/// A narrow counterpart to IKernelCompiler in role only, not a subtype of it:
/// compile(fileName, options) is HIPRTC-shaped and cannot express (library, tocKey).
/// IKernelCompiler is an interface because the compiler genuinely is substituted in
/// tests; this loader is not substituted anywhere -- §5.2 passes a real one everywhere,
/// and the missing-archive test deliberately uses a genuinely nonexistent path so it
/// exercises the real archive-open path rather than a mock's. That is why it is
/// concrete: a virtual base with one implementation and no substitution would be
/// abstraction for its own sake.
///
/// The module cache is injected rather than reached as a singleton, so a test can
/// observe its size (AC #6) and so two loaders in one process do not silently share
/// state. The referenced cache must outlive the loader.
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
    /// cache key. Three of the failures below -- missing archive, unreadable archive,
    /// arch mismatch -- are raised before any symbol lookup happens, yet AC #7 requires
    /// every message to name the descriptor *and* the symbol. Folding `symbol` into
    /// KpackModuleCache::makeKey because this function now receives one would defeat
    /// AC #6's sharing of a single module across kernels that differ only by entry
    /// point.
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
