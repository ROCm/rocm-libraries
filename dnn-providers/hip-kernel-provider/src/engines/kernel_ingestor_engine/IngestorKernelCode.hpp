// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

/// Fails unless the argument list a descriptor records matches the one its pack marshals.
///
/// A prebuilt archive is compiled out of band from the pack that launches it, and
/// `hipModuleGetFunction` confirms only that a symbol of that name exists, so a drifted
/// signature is otherwise undefined behaviour at launch rather than a diagnostic. Embedded
/// source needs none -- HIPRTC compiles it against the declaration the host marshals.
///
/// `kind` and `size` always. `name` only where both sides carry one: it is the one thing
/// that catches an operand permutation, but clang omits it for HIP `extern "C" __global__`
/// kernels, and requiring it would make every such kernel undispatchable. `offset` is
/// printed and never compared -- the kernarg layout is the driver's, not the pack's to
/// assert.
///
/// Throwing rather than warning because every way this fires is a static disagreement
/// between two authored artifacts: no configuration reaches it with a correct launch.
inline void
    requireSignatureMatch(const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& recorded,
                          const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& expected,
                          const std::string& symbol,
                          const std::string& label)
{
    const auto agrees = [](const hipdnn_plugin_sdk::ingestor::KernelArgument& lhs,
                           const hipdnn_plugin_sdk::ingestor::KernelArgument& rhs) {
        if(lhs.kind != rhs.kind || lhs.size != rhs.size)
        {
            return false;
        }
        return lhs.name.empty() || rhs.name.empty() || lhs.name == rhs.name;
    };

    if(recorded.size() != expected.size()
       || !std::equal(recorded.begin(), recorded.end(), expected.begin(), agrees))
    {
        // Both sides printed: a mismatch diagnostic naming only one of them sends the
        // reader to the archive by hand to find out what the other was.
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "kpack kernel source for " + label + ": symbol '" + symbol
                + "' is packaged with arguments "
                + hipdnn_plugin_sdk::ingestor::describeKernelSignature(recorded)
                + ", but this pack launches it with "
                + hipdnn_plugin_sdk::ingestor::describeKernelSignature(expected));
    }
}

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
/// @param expectedSignature The list the calling pack declares beside its own launch.
///                   Used only on the KPACK path, and deliberately without a default:
///                   a pack that omits it should not compile into one that silently
///                   skips the check.
inline IngestorKernelCode buildIngestorKernelCode(
    const compilation::IKernelCompiler& compiler,
    const compilation::KpackKernelLoader& kpackLoader,
    const hipdnn_plugin_sdk::ingestor::MatchContext& context,
    const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
    const compilation::KernelCompileOptions& options,
    const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& expectedSignature)
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

        // Reject a `library` that reaches its archive through a link -- a POSIX symlink or
        // a Windows junction -- inside the tree.
        //
        // Walked over the UN-canonicalised join: weakly_canonical has already resolved the
        // symlinks in whatever prefix exists, so testing `resolved` would find none. Scoped
        // strictly below `boundary` because a tree under a symlinked prefix is ordinary.
        //
        // Each component must BE an ordinary file or directory rather than merely not be one
        // named kind. MSVC reports a junction as file_type::junction, so is_symlink answers
        // false for one -- and mklink /J needs no privilege, unlike a symlink.
        //
        // This refuses a path that IS a link at validation time and does not close the
        // time-of-check/time-of-use race: kpack_open is path-only, with no fd or handle
        // overload anywhere in the kpack C API.
        std::filesystem::path prefix;
        for(const auto& component : origin / kernel.source.library)
        {
            prefix /= component;
            if(component == "." || component == "..")
            {
                continue;
            }
            const std::string below = prefix.lexically_relative(boundary).generic_string();
            if(below.empty() || below == "." || below.rfind("..", 0) == 0)
            {
                continue;
            }
            // symlink_status, not status: the latter follows the link and reports the
            // target's kind, which is exactly the answer that must not be trusted here.
            const std::filesystem::file_type kind
                = std::filesystem::symlink_status(prefix, ignored).type();

            // not_found is allowed through so the archive's own absence is reported by the
            // loader, which names it, rather than as a link that is not there either.
            if(kind != std::filesystem::file_type::regular
               && kind != std::filesystem::file_type::directory
               && kind != std::filesystem::file_type::not_found)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                    "kpack kernel source for " + label + ": library '" + kernel.source.library
                        + "' reaches its archive through the link '" + prefix.string()
                        + "' inside the descriptor tree '" + boundary.string() + "'");
            }
        }

        // Ahead of the load: the descriptor and the pack already disagree, and opening an
        // archive to confirm it would only delay the same diagnostic.
        requireSignatureMatch(
            kernel.source.signature, expectedSignature, kernel.source.symbol, label);

        auto program = kpackLoader.load(resolved,
                                        kernel.source.tocKey,
                                        context.deviceProperties.gcnArchName,
                                        context.deviceId,
                                        kernel.source.symbol,
                                        kernel.source.sha256,
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
