// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefineSubstitution.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/IRunnableKernel.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "kernel_includes.hpp"

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

namespace detail
{

/// Resolves @p relative against the directory of the descriptor that declared @p kernel,
/// and reports whether it stayed inside the descriptor tree.
///
/// A descriptor names a file shipped inside the tree it was loaded from, never one
/// elsewhere on the filesystem. weakly_canonical normalises `..` and absolute paths
/// rather than rejecting them, so without this a descriptor could name any readable file
/// and have it loaded or compiled as code. Canonical forms are compared: the lexical
/// check alone would miss a symlink out of the tree. weakly_canonical rather than
/// canonical because the target need not exist -- when it does not, the caller's
/// open failure is the diagnostic, not a filesystem exception.
///
/// The boundary is the TREE, not the descriptor's own directory. One kpack archive ships
/// per arch shard at the shard root, so a descriptor authored in a child folder -- which
/// is every production layout, since packing preserves the authored subpath -- has to
/// climb out of its own directory to reach it. Anchoring on originDirectory rejected
/// exactly those and made every production-packaged kernel unloadable while flat fixture
/// trees stayed green. A drop-in bundle inherits the same rule for free.
///
/// treeRoot rather than a derived arch-shard root: it is what the loader actually walked,
/// so it needs no filesystem probing and assumes nothing about how deep a shard sits
/// under it. A kernel built in memory carries neither path and reaches neither adapter --
/// both require a file -- but an empty treeRoot degrades to the origin rather than
/// opening a hole.
inline bool resolveInsideDescriptorTree(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                                        const std::filesystem::path& relative,
                                        std::filesystem::path& resolved,
                                        std::filesystem::path& boundary)
{
    std::error_code ignored;
    const std::filesystem::path origin
        = std::filesystem::weakly_canonical(kernel.originDirectory, ignored);
    resolved = std::filesystem::weakly_canonical(origin / relative, ignored);
    boundary = kernel.treeRoot.empty()
                   ? origin
                   : std::filesystem::weakly_canonical(kernel.treeRoot, ignored);

    const std::string within = resolved.lexically_relative(boundary).generic_string();
    return resolved == boundary || (!within.empty() && within.rfind("..", 0) != 0);
}

/// Reads a whole file the descriptor named, or says which one could not be opened.
inline std::string readDescriptorFile(const std::filesystem::path& path,
                                      const std::string& what,
                                      const std::string& label)
{
    std::ifstream input(path, std::ios::binary);
    if(!input.good())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                       "hiprtc_file kernel source for " + label
                                                           + ": cannot open " + what + " '"
                                                           + path.string() + "'");
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

/// The complete virtual-header list a bundle-sourced kernel compiles against: the
/// provider's embedded includes first, then every header shipped in the bundle.
///
/// Embedded first so a drop-in kernel can `#include` the same helpers a built-in one
/// does. A bundle header whose name collides with an embedded one is a load error rather
/// than a silent shadow: hipRTC resolves the first match, so whichever way the collision
/// were broken it would be invisible, and the author's mental model -- "my file wins" or
/// "theirs does" -- would be right only by luck.
///
/// The bundle is read one level deep and sorted by name, so the list a kernel compiles
/// against does not depend on directory-iteration order.
///
/// Every bundle header goes through resolveInsideDescriptorTree before it is read, by the
/// same rule and with the same message shape as `bundle` and `source_file`: is_regular_file
/// follows a symlink, so a link out of an otherwise contained bundle would otherwise be
/// read whole and handed to hipRTC as a virtual header. Canonical comparison rather than a
/// blanket symlink refusal, so a bundle legitimately assembled by symlink INSIDE the tree
/// still loads and only an escape is refused.
///
/// @param embeddedNames Names of the provider's embedded includes, parallel to
///                      @p embeddedTexts. Passed in rather than read from
///                      hip_plugin::getKernelIncList here so the collision branch is
///                      reachable from a test binary, which embeds no headers of its own.
inline std::vector<compilation::KernelHeader>
    collectKernelHeaders(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                         const std::filesystem::path& bundleDirectory,
                         const std::string& label,
                         const std::vector<const char*>& embeddedNames,
                         const std::vector<std::string_view>& embeddedTexts)
{
    std::vector<compilation::KernelHeader> headers;
    headers.reserve(embeddedTexts.size());
    for(size_t index = 0; index < embeddedNames.size(); ++index)
    {
        headers.emplace_back(embeddedNames[index], std::string(embeddedTexts[index]));
    }

    // Checked rather than ignored: an unreadable or non-directory bundle would otherwise
    // degrade to an empty header list and surface as a hipRTC "file not found" inside the
    // kernel's own source, naming neither the bundle nor the real cause.
    std::error_code walkError;
    const std::filesystem::directory_iterator walk(bundleDirectory, walkError);
    if(walkError)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "hiprtc_file kernel source for " + label + ": cannot read bundle directory '"
                + bundleDirectory.string() + "': " + walkError.message());
    }

    std::map<std::string, std::filesystem::path> bundleHeaders;
    for(const auto& entry : walk)
    {
        if(!entry.is_regular_file())
        {
            continue;
        }
        const std::string extension = entry.path().extension().string();
        if(extension != ".h" && extension != ".hpp" && extension != ".cuh")
        {
            continue;
        }

        const std::string name = entry.path().filename().string();
        std::filesystem::path resolved;
        std::filesystem::path boundary;
        if(!resolveInsideDescriptorTree(
               kernel, std::filesystem::path(kernel.source.bundle) / name, resolved, boundary))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "hiprtc_file kernel source for " + label + ": bundle header '" + name
                    + "' resolves to '" + resolved.string()
                    + "', which is outside the descriptor tree '" + boundary.string() + "'");
        }
        bundleHeaders.emplace(name, resolved);
    }

    for(const auto& bundleHeader : bundleHeaders)
    {
        const std::string& name = bundleHeader.first;
        const bool collides
            = std::any_of(headers.begin(), headers.end(), [&name](const auto& header) {
                  return header.first == name;
              });
        if(collides)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "hiprtc_file kernel source for " + label + ": bundle header '" + name
                    + "' has the same name as one of the provider's embedded headers."
                      " Rename it: which one an #include resolves to would otherwise be"
                      " invisible");
        }
        headers.emplace_back(name, readDescriptorFile(bundleHeader.second, "bundle header", label));
    }
    return headers;
}

} // namespace detail

/// The single place a KernelSource's `kind` decides where the code object comes from.
///
/// One helper rather than a branch copied into each pack handler: ConvNative is then a
/// two-line follow-up rather than a second copy of this logic.
///
/// @param compiler   Used on the EMBEDDED_SOURCE and HIPRTC_FILE paths.
/// @param kpackLoader Used only on the KPACK path.
/// @param options    HIPRTC build options. Deliberately not consulted on the KPACK
///                   path: a kpack blob's build defines were baked at pack time, so
///                   there is nothing left for them to affect. Silently ignoring them
///                   is the correct behaviour, not an oversight. Taken by mutable
///                   reference because HIPRTC_FILE appends this kernel's own bound
///                   defines to them, which is the whole point of that kind; the
///                   handler's own defines are already in place by then and a bound one
///                   naming the same macro deliberately wins, since it is the more
///                   specific statement. `const&` plus an internal copy is not merely
///                   unattractive, it is unavailable: KernelCompileOptions' copy
///                   constructor is deleted (KernelCompileOptions.hpp:52-53), so a
///                   mutable reference is the only shape this can take.
inline IngestorKernelCode
    buildIngestorKernelCode(const compilation::IKernelCompiler& compiler,
                            const compilation::KpackKernelLoader& kpackLoader,
                            const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                            const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                            compilation::KernelCompileOptions& options)
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
        // originDirectory is the loader-supplied anchor that makes it nameable, and the
        // descriptor tree is the boundary it may not cross.
        std::filesystem::path resolved;
        std::filesystem::path boundary;
        const bool contained = detail::resolveInsideDescriptorTree(
            kernel, kernel.source.library, resolved, boundary);

        const std::string label = hipdnn_plugin_sdk::ingestor::describeDescriptor(
            "kernel", kernel.name, kernel.kernelId);

        if(!contained)
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
    case KernelSourceKind::HIPRTC_FILE:
    {
        const std::string label = hipdnn_plugin_sdk::ingestor::describeDescriptor(
            "kernel", kernel.name, kernel.kernelId);

        // Same rule as KPACK above: the bundle is authored relative to the descriptor and
        // may not resolve outside the tree the descriptor was loaded from.
        std::filesystem::path bundleDirectory;
        std::filesystem::path boundary;
        if(!detail::resolveInsideDescriptorTree(
               kernel, kernel.source.bundle, bundleDirectory, boundary))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "hiprtc_file kernel source for " + label + ": bundle '" + kernel.source.bundle
                    + "' resolves to '" + bundleDirectory.string()
                    + "', which is outside the descriptor tree '" + boundary.string() + "'");
        }

        // Checked separately rather than trusting the bundle: `source_file` is authored
        // too, so `../../etc/passwd` inside a contained bundle would otherwise escape.
        std::filesystem::path resolved;
        if(!detail::resolveInsideDescriptorTree(kernel,
                                                std::filesystem::path(kernel.source.bundle)
                                                    / kernel.source.sourceFile,
                                                resolved,
                                                boundary))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "hiprtc_file kernel source for " + label + ": source_file '"
                    + kernel.source.sourceFile + "' in bundle '" + kernel.source.bundle
                    + "' resolves to '" + resolved.string()
                    + "', which is outside the descriptor tree '" + boundary.string() + "'");
        }

        // Bound against the kernel's COMPLETED metadata -- the state manager filled the
        // KMD's defaults before this definition was built -- so a descriptor that omits a
        // defaulted field still resolves. Every token was already checked against the KMD
        // at set resolution, so a failure here is a value this build cannot render rather
        // than an authoring typo, and it costs one plan rather than the engine.
        for(const auto& [name, templateText] : kernel.source.defines)
        {
            std::string value;
            std::string error;
            if(!hipdnn_plugin_sdk::ingestor::substituteKernelDefine(
                   templateText, kernel.metadata, value, error))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                    "hiprtc_file kernel source for " + label + ": cannot bind define '" + name
                        + "': " + error);
            }
            options.add(name, value);
        }

        const std::string sourceText = detail::readDescriptorFile(resolved, "source_file", label);

        std::vector<std::string_view> embeddedTexts;
        std::vector<const char*> embeddedNames;
        hip_plugin::getKernelIncList(embeddedTexts, embeddedNames);
        const auto headers = detail::collectKernelHeaders(
            kernel, bundleDirectory, label, embeddedNames, embeddedTexts);

        // The RESOLVED path, not the bare source_file: it is the compile cache's key
        // alongside the options, and two bundles each holding `attention.hip` with
        // identical defines would otherwise share one entry and the second silently get
        // the first's binary.
        auto program = compiler.compileSource(sourceText, resolved.string(), headers, options);
        auto runnableKernel = program->getKernel(kernel.source.entryPoint);
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
