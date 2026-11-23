// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

/**
 * @file LaunchKernelDefinition.hpp
 * @brief The definition of one launch of a multi-launch kernel.
 *
 * A UKD with a `sources` map (RFC 0017 §15.1) names each of its launches by launch name and
 * carries a complete kpack source for each. A native dispatch handler that launches such a
 * kernel declares its launch names (IKernelDispatchHandler::launchNames), so every kernel
 * that reaches it carries exactly those. To launch one it hands buildIngestorKernelCode the
 * definition this returns: the kernel's own, with `source` set to that launch's entry. The
 * code object, symbol, digest and recorded signature are the entry's, so the launch resolves
 * through the ordinary kpack path, and requireSignatureMatch compares the native's
 * marshalled list against the packaged one exactly as it does for a single-launch kernel.
 * Launches that one rocKE module produced share a library and toc key, so they resolve out
 * of one loaded module.
 *
 * Launch order, workspace and the hand-off between launches stay with the native handler:
 * the descriptor only says what each launch is.
 *
 * entryPointKernelDefinition is the `embedded_source` counterpart: one translation unit,
 * several entry points, selected by the handler rather than listed in the descriptor.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The definition of launch @p launch of @p kernel.
///
/// The copy has `source` set to that launch's entry and an empty `sources`: it is one
/// kernel to load, not a kernel with launches of its own. Everything else is @p kernel's.
///
/// Throws INVALID_VALUE when @p kernel carries no `sources` (a single-launch kernel has no
/// launch names), or when it has no launch called @p launch. The latter names the launches
/// it does have.
inline hipdnn_plugin_sdk::ingestor::KernelDefinition
    launchKernelDefinition(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                           const std::string& launch)
{
    const std::string label
        = hipdnn_plugin_sdk::ingestor::describeDescriptor("kernel", kernel.name, kernel.kernelId);
    if(kernel.sources.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            label + " is a single-launch kernel, so it has no launch '" + launch + "'");
    }

    const auto found = kernel.sources.find(launch);
    if(found == kernel.sources.end())
    {
        std::string available;
        for(const auto& entry : kernel.sources)
        {
            available += (available.empty() ? "'" : ", '") + entry.first + "'";
        }
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                       label + " has no launch '" + launch
                                                           + "'; available launches: " + available);
    }

    // Positional, as KernelDefinition is elsewhere (its field count is pinned by a
    // static_assert), so the launches map is never copied just to be cleared.
    return hipdnn_plugin_sdk::ingestor::KernelDefinition{kernel.kernelId,
                                                         kernel.packId,
                                                         kernel.dispatchId,
                                                         found->second,
                                                         kernel.metadata,
                                                         kernel.priority,
                                                         kernel.arch,
                                                         kernel.originDirectory,
                                                         kernel.name,
                                                         kernel.treeRoot,
                                                         {},
                                                         kernel.prepared,
                                                         kernel.preparedBy};
}

/// The definition of entry point @p entryPoint of the `embedded_source` kernel @p kernel.
///
/// A single-launch UKD whose native handler launches further kernels by symbol (RFC 0021 §3)
/// names one entry point of its translation unit; the handler reaches the others through
/// this. The copy has `source.entryPoint` set to @p entryPoint and is otherwise @p kernel's,
/// so buildIngestorKernelCode compiles the same file with the same options. The compiler's
/// module cache keys on (file, options), so every entry point after the first resolves out
/// of the program the first one compiled. `embedded_source` cannot appear in a `sources`
/// map, which is why this exists beside launchKernelDefinition rather than through it.
///
/// Throws INVALID_VALUE when @p kernel carries `sources`, when its source kind is not
/// `embedded_source` (a kpack entry's symbol is bound to a recorded signature and digest), or
/// when @p entryPoint is empty.
inline hipdnn_plugin_sdk::ingestor::KernelDefinition
    entryPointKernelDefinition(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                               const std::string& entryPoint)
{
    const std::string label
        = hipdnn_plugin_sdk::ingestor::describeDescriptor("kernel", kernel.name, kernel.kernelId);
    if(!kernel.sources.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            label + " carries launches ('sources'); select one with launchKernelDefinition");
    }
    if(kernel.source.kind != hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            label + " is not an 'embedded_source' kernel, so it has no entry point '" + entryPoint
                + "' to select");
    }
    if(entryPoint.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE, label + " was asked for an empty entry point");
    }

    hipdnn_plugin_sdk::ingestor::KernelDefinition definition = kernel;
    definition.source.entryPoint = entryPoint;
    return definition;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
