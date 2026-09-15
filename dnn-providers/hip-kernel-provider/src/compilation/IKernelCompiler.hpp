// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ICompiledProgram.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hip_kernel_provider::compilation
{

/// One hipRTC virtual header: the name an `#include` in the kernel names it by, and its
/// text. Owned, unlike the configure-time embedded include table, because a drop-in
/// bundle's headers are read off disk at prepare() and nothing else holds them.
using KernelHeader = std::pair<std::string, std::string>;

class IKernelCompiler
{
public:
    virtual ~IKernelCompiler() = default;
    /// Compiles a kernel named by key into the provider's configure-time embedded source
    /// table. The embedded include list is supplied by the implementation.
    virtual std::unique_ptr<ICompiledProgram> compile(const std::string& kernelFileName,
                                                      const std::vector<std::string>& options) const
        = 0;

    /// Compiles owned source text that no embedded table holds -- a hiprtc_file kernel
    /// read out of a drop-in bundle.
    ///
    /// @param sourceText  The kernel's full source.
    /// @param programName Identity, not a lookup key: it names the program in hipRTC
    ///                    diagnostics and is the compile cache's key alongside
    ///                    @p options, so callers pass the RESOLVED path rather than the
    ///                    bare file name. Two bundles each holding `attention.hip` with
    ///                    identical defines would otherwise share one cache entry and the
    ///                    second would silently be served the first's binary.
    /// @param headers     The complete virtual-header list, embedded and bundle alike.
    ///                    Complete rather than additive so the collision between the two
    ///                    is decided by the caller, which can name the offending file.
    virtual std::unique_ptr<ICompiledProgram>
        compileSource(const std::string& sourceText,
                      const std::string& programName,
                      const std::vector<KernelHeader>& headers,
                      const std::vector<std::string>& options) const
        = 0;
};

} // namespace hip_kernel_provider::compilation
