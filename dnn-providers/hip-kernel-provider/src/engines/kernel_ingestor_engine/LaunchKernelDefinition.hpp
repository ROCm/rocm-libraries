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

    hipdnn_plugin_sdk::ingestor::KernelDefinition definition = kernel;
    definition.source = found->second;
    definition.sources.clear();
    return definition;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
