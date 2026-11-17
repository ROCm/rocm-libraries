// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/// @file IngestorLaunchRestore.hpp
/// The restore steps every ingestor pack shares, and the readers its launch values use.
///
/// A pack restores a saved dispatch in three checks, in this order: the recorded kernel
/// signature, the launch values, then the module load.
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The launch inputs a pack read from a saved plan, and the kernel code loaded for them.
template <typename TLaunchInputs>
struct RestoredLaunch
{
    TLaunchInputs launch;
    IngestorKernelCode code;
};

namespace detail
{

// How restore messages name a saved kernel.
inline std::string describeSavedKernel(const hipdnn_plugin_sdk::ingestor::SavedKernelCode& code,
                                       const std::string& dispatchSymbol)
{
    return "saved plan for kernel " + hipdnn_plugin_sdk::ingestor::toString(code.kernelId)
           + " (symbol '" + code.symbol + "', dispatch '" + dispatchSymbol + "')";
}

// The value named `name`. Refuses the plan when the value is absent.
inline const hipdnn_plugin_sdk::ingestor::MetadataValue&
    requireLaunchValue(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                       std::string_view name,
                       const std::string& expected)
{
    const auto found = inputs.values.find(std::string(name));
    if(found == inputs.values.end())
    {
        serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                          "dispatch '" + inputs.dispatchSymbol
                                              + "' needs launch value '" + std::string(name) + "' ("
                                              + expected + "), and the plan has none");
    }
    return found->second;
}

[[noreturn]] inline void
    refuseLaunchValue(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                      std::string_view name,
                      const std::string& expected,
                      const std::string& found)
{
    serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                      "launch value '" + std::string(name) + "' of dispatch '"
                                          + inputs.dispatchSymbol + "' must be " + expected
                                          + ", and the plan holds " + found);
}

} // namespace detail

/// Refuses the plan unless its dispatch name is @p alias, the contract the caller reads.
inline void requireLaunchContract(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                                  std::string_view alias)
{
    if(inputs.dispatchSymbol != alias)
    {
        serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                          "dispatch '" + inputs.dispatchSymbol
                                              + "' names no launch-value contract this "
                                                "provider reads; the handler reads the contract '"
                                              + std::string(alias) + "'");
    }
}

/// The integer value @p name. Refuses the plan when it is absent, has another type, or
/// is outside [@p lowest, @p highest].
inline int64_t requireLaunchInt(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                                std::string_view name,
                                int64_t lowest = std::numeric_limits<int64_t>::min(),
                                int64_t highest = std::numeric_limits<int64_t>::max())
{
    const std::string expected
        = hipdnn_plugin_sdk::ingestor::toString(hipdnn_plugin_sdk::ingestor::MetadataType::INT)
          + " in [" + std::to_string(lowest) + ", " + std::to_string(highest) + "]";
    const auto& value = detail::requireLaunchValue(inputs, name, expected);
    const auto* integer = std::get_if<int64_t>(&value);
    if(integer == nullptr)
    {
        detail::refuseLaunchValue(inputs,
                                  name,
                                  expected,
                                  hipdnn_plugin_sdk::ingestor::toString(
                                      hipdnn_plugin_sdk::ingestor::metadataTypeOf(value)));
    }
    if(*integer < lowest || *integer > highest)
    {
        detail::refuseLaunchValue(inputs, name, expected, std::to_string(*integer));
    }
    return *integer;
}

/// The double value @p name. Refuses the plan when it is absent or has another type.
inline double requireLaunchDouble(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                                  std::string_view name)
{
    const std::string expected
        = hipdnn_plugin_sdk::ingestor::toString(hipdnn_plugin_sdk::ingestor::MetadataType::FLOAT);
    const auto& value = detail::requireLaunchValue(inputs, name, expected);
    const auto* number = std::get_if<double>(&value);
    if(number == nullptr)
    {
        detail::refuseLaunchValue(inputs,
                                  name,
                                  expected,
                                  hipdnn_plugin_sdk::ingestor::toString(
                                      hipdnn_plugin_sdk::ingestor::metadataTypeOf(value)));
    }
    return *number;
}

/// The string value @p name. Refuses the plan when it is absent or has another type.
inline std::string requireLaunchString(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                                       std::string_view name)
{
    const std::string expected
        = hipdnn_plugin_sdk::ingestor::toString(hipdnn_plugin_sdk::ingestor::MetadataType::STRING);
    const auto& value = detail::requireLaunchValue(inputs, name, expected);
    const auto* text = std::get_if<std::string>(&value);
    if(text == nullptr)
    {
        detail::refuseLaunchValue(inputs,
                                  name,
                                  expected,
                                  hipdnn_plugin_sdk::ingestor::toString(
                                      hipdnn_plugin_sdk::ingestor::metadataTypeOf(value)));
    }
    return *text;
}

/// Refuses the plan when it holds a launch value that is not in @p names.
inline void requireOnlyLaunchValues(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                                    std::initializer_list<std::string_view> names)
{
    for(const auto& entry : inputs.values)
    {
        if(std::find(names.begin(), names.end(), std::string_view(entry.first)) == names.end())
        {
            serialization::refuseIngestorPlan(serialization::IngestorPlanRefusal::INCOMPATIBLE,
                                              "launch value '" + entry.first
                                                  + "' is not part of the contract of dispatch '"
                                                  + inputs.dispatchSymbol + "'");
        }
    }
}

/// Runs the restore checks a pack shares, in order, and loads the saved code on
/// @p deviceOrdinal:
///   1. The signature recorded for the saved kernel must match @p declaredSignature, the
///      arguments the pack launches.
///   2. @p readLaunchInputs reads and checks the launch values.
///   3. The code object loads, and the driver finds the symbol.
///
/// Every refusal names the kernel and the dispatch.
template <typename TLaunchInputs, typename TReader>
RestoredLaunch<TLaunchInputs> restoreIngestorLaunch(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
    hipdnn_plugin_sdk::ingestor::SavedKernelCode code,
    const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& declaredSignature,
    TReader readLaunchInputs,
    int deviceOrdinal)
{
    const std::string label = detail::describeSavedKernel(code, inputs.dispatchSymbol);

    // Compare the kernel's argument lists.
    try
    {
        requireSignatureMatch(code.recordedSignature, declaredSignature, code.symbol, label);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        if(error.getStatus() != HIPDNN_PLUGIN_STATUS_INVALID_VALUE)
        {
            throw;
        }
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "the saved kernel's arguments do not match what this provider launches: "
                + error.getMessage());
    }

    TLaunchInputs launch = readLaunchInputs(inputs);

    IngestorKernelCode loaded
        = IngestorKernelCode::fromSavedKernelCode(std::move(code), label, deviceOrdinal);

    return RestoredLaunch<TLaunchInputs>{std::move(launch), std::move(loaded)};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
