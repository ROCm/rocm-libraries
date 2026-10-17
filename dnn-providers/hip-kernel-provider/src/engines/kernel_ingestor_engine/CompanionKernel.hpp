// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <utility>
#include <variant>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

/**
 * @file CompanionKernel.hpp
 * @brief The descriptor of a kernel shipped in another kernel's code object.
 *
 * A rocKE builder may emit several kernels into ONE code object (a split-KV decode and the
 * kernel merging its splits): the descriptor names the primary in `kernel_source.symbol`,
 * and the packager (hkp_pack, see descriptor-packaging/python/hkp_pack/companions.py)
 * writes each further kernel, by role, into two string metadata fields the engine's KMD
 * declares:
 *
 * - `<role>_symbol`: the companion's symbol in the same code object;
 * - `<role>_signature`: its argument list as compact JSON, in exactly the shape of
 *   `kernel_source.signature`, read out of the object at pack time.
 *
 * A native launches the companion through the ordinary kpack path by handing
 * buildIngestorKernelCode the definition this returns: the primary's, with only the
 * symbol and signature swapped. The archive, toc key and digest are the primary's, so the
 * companion resolves out of the very module the primary loaded (KpackModuleCache keys on
 * archive, toc key and arch), and requireSignatureMatch compares the native's marshalled
 * list against the packaged one exactly as it does for the primary.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The metadata field naming the symbol of the companion with @p role.
inline std::string companionSymbolField(const std::string& role)
{
    return role + "_symbol";
}

/// The metadata field holding the packaged argument list of the companion with @p role.
inline std::string companionSignatureField(const std::string& role)
{
    return role + "_signature";
}

namespace detail
{

/// A non-empty string metadata value of @p kernel, or an INVALID_VALUE naming the field.
inline const std::string&
    requireCompanionString(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                           const std::string& field,
                           const std::string& label)
{
    const auto it = kernel.metadata.find(field);
    const std::string* value
        = it == kernel.metadata.end() ? nullptr : std::get_if<std::string>(&it->second);
    if(value == nullptr || value->empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            label + " carries no companion metadata '" + field
                + "': its code object was not packaged with that companion kernel");
    }
    return *value;
}

} // namespace detail

/// The definition of the companion kernel @p primary's own code object exports as @p role.
///
/// Throws INVALID_VALUE when @p primary is not a kpack kernel, when either companion field
/// is absent, empty or not a string, or when the signature is not the JSON argument list
/// the packager writes -- parsed by the same reader the descriptor loader applies to
/// `kernel_source.signature`, so the two cannot accept different shapes.
inline hipdnn_plugin_sdk::ingestor::KernelDefinition
    companionKernelDefinition(const hipdnn_plugin_sdk::ingestor::KernelDefinition& primary,
                              const std::string& role)
{
    using hipdnn_plugin_sdk::ingestor::KernelSourceKind;

    const std::string label
        = hipdnn_plugin_sdk::ingestor::describeDescriptor("kernel", primary.name, primary.kernelId);
    if(primary.source.kind != KernelSourceKind::KPACK)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            label + " is not a kpack kernel, so it has no code object to hold companion '" + role
                + "'");
    }

    const std::string symbolField = companionSymbolField(role);
    const std::string signatureField = companionSignatureField(role);
    const std::string& symbol = detail::requireCompanionString(primary, symbolField, label);
    const std::string& signatureText
        = detail::requireCompanionString(primary, signatureField, label);

    const std::string where = label + " metadata '" + signatureField + "'";
    nlohmann::json parsed = nlohmann::json::parse(signatureText, nullptr, false);
    if(parsed.is_discarded())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                       where + " is not valid JSON");
    }
    const nlohmann::json holder = {{"signature", std::move(parsed)}};

    hipdnn_plugin_sdk::ingestor::KernelDefinition companion = primary;
    companion.source.symbol = symbol;
    // hipDNN's plugin SDK exposes the descriptor loader's signature reader only as
    // ingestor::detail::requireKernelSignature; there is no public equivalent. It is used
    // anyway so the companion and kernel_source signatures cannot parse differently, and
    // a change to that reader breaks this build rather than drifting silently.
    companion.source.signature
        = hipdnn_plugin_sdk::ingestor::detail::requireKernelSignature(holder, where);
    return companion;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
