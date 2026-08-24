// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/IRunnableKernel.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The kpack module cache the pointwise packs' dispatch handler loads through,
/// process-lifetime. Declared here rather than in IngestorPacks.hpp because it belongs
/// to the kernel-code path, and exposed at all so a test can assert that two dispatches
/// over one (archive, toc_key, arch) produced a single hipModule_t -- the direct
/// otherwise unobservable. Defined in PointwiseNative.cpp beside the handler it serves.
compilation::KpackModuleCache& pointwiseKpackModuleCache();

/// The program plus the kernel resolved out of it, in the shape every pack's
/// PreparedDispatch already holds. Returned together because the kernel is a
/// non-owning view into the program and the two must be stored side by side.
struct IngestorKernelCode
{
    std::unique_ptr<compilation::ICompiledProgram> program;
    std::unique_ptr<compilation::IRunnableKernel> kernel;
};

/// The single place a KernelSource's `kind` decides where the code object comes from.
///
/// One helper rather than a branch copied into each pack handler: ConvNative is then a
/// two-line follow-up rather than a second copy of this logic.
///
/// @param compiler   Used only on the EMBEDDED_SOURCE path.
/// @param kpackLoader Used only on the KPACK path.
/// @param options    HIPRTC build options. Deliberately not consulted on the KPACK
///                   path: a kpack blob's build defines were baked at pack time, so
///                   there is nothing left for them to affect. Silently ignoring them
///                   is the correct behaviour, not an oversight.
inline IngestorKernelCode
    buildIngestorKernelCode(const compilation::IKernelCompiler& compiler,
                            const compilation::KpackKernelLoader& kpackLoader,
                            const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                            const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                            const compilation::KernelCompileOptions& options)
{
    using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

    switch(kernel.source.kind)
    {
    case KernelSourceKind::EMBEDDED_SOURCE:
    {
        auto program = compiler.compile(kernel.source.sourceFile, options);
        auto runnableKernel = program->getKernel(kernel.source.entryPoint);
        return IngestorKernelCode{std::move(program), std::move(runnableKernel)};
    }
    case KernelSourceKind::KPACK:
    {
        // `library` is authored relative to the descriptor that declared it;
        // originDirectory is the loader-supplied anchor that makes it nameable.
        // weakly_canonical because the target need not exist -- when it does not, the
        // archive-open failure below is the diagnostic, not a filesystem exception.
        std::error_code ignored;
        const std::filesystem::path origin
            = std::filesystem::weakly_canonical(kernel.originDirectory, ignored);
        const std::filesystem::path resolved
            = std::filesystem::weakly_canonical(origin / kernel.source.library, ignored);

        const std::string label = hipdnn_plugin_sdk::ingestor::describeDescriptor(
            "kernel", kernel.name, kernel.kernelId);

        // A descriptor names an archive shipped inside the tree it was loaded from, never
        // one elsewhere on the filesystem. weakly_canonical normalises `..` and absolute
        // paths rather than rejecting them, so without this a descriptor could name any
        // readable file and have it loaded as executable code. Compare canonical forms:
        // the lexical check alone would miss a symlink out of the tree.
        //
        // The boundary is the TREE, not the descriptor's own directory. One archive ships
        // per arch shard, at the shard root, so a descriptor authored in a child folder --
        // which is every production layout, since packing preserves the authored subpath --
        // has to climb out of its own directory to reach it. Anchoring on originDirectory
        // rejected exactly those, which made every production-packaged kernel unloadable
        // while flat fixture trees stayed green.
        //
        // treeRoot rather than a derived arch-shard root: it is what the loader actually
        // walked, so it needs no filesystem probing and assumes nothing about how deep a
        // shard sits under it. A kernel built in memory carries neither path and is not
        // reachable here -- KPACK requires a file -- but an empty treeRoot would degrade
        // to the old behaviour rather than open a hole, so fall back to origin.
        const std::filesystem::path boundary
            = kernel.treeRoot.empty() ? origin
                                      : std::filesystem::weakly_canonical(kernel.treeRoot, ignored);
        const std::string relative = resolved.lexically_relative(boundary).generic_string();
        if(resolved != boundary && (relative.empty() || relative.rfind("..", 0) == 0))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "kpack kernel source for " + label + ": library '" + kernel.source.library
                    + "' resolves to '" + resolved.string()
                    + "', which is outside the descriptor tree '" + boundary.string() + "'");
        }

        auto program = kpackLoader.load(resolved,
                                        kernel.source.tocKey,
                                        context.deviceProperties.gcnArchName,
                                        kernel.source.symbol,
                                        label);
        auto runnableKernel = program->getKernel(kernel.source.symbol);
        return IngestorKernelCode{std::move(program), std::move(runnableKernel)};
    }
    case KernelSourceKind::HSACO_FILE:
    case KernelSourceKind::ROCKE_BUILDER:
    // A kind added after this adapter was written lands here too, and gets the same
    // named diagnostic rather than falling off the end of the function.
    default:
        break;
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        "no kernel source adapter for "
            + hipdnn_plugin_sdk::ingestor::describeDescriptor(
                "kernel", kernel.name, kernel.kernelId)
            + ": its source kind is not one this provider can load");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
