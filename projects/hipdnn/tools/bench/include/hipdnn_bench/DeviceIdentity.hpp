// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/heuristics/DeviceFeatures.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceKey.hpp>

#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

/// @file DeviceIdentity.hpp
/// @brief The `device` a measurement is of: the kind of GPU, never the GPU itself.
namespace hipdnn_bench
{

/// The properties the runtime keys a device by, read from @p properties as the
/// hip-kernel-provider's device resolver reads them.
inline hipdnn_plugin_sdk::ingestor::DeviceProperties
    devicePropertiesFrom(const hipDeviceProp_t& properties)
{
    hipdnn_plugin_sdk::ingestor::DeviceProperties resolved;
    resolved.gcnArchName = properties.gcnArchName;
    resolved.warpSize = properties.warpSize;
    resolved.multiProcessorCount = properties.multiProcessorCount;
    resolved.totalGlobalMem = properties.totalGlobalMem;
    resolved.memoryBusWidth = properties.memoryBusWidth;
    resolved.memoryClockRate = properties.memoryClockRate;
    resolved.sharedMemPerBlock = properties.sharedMemPerBlock;
    return resolved;
}

/// The `device_id` a row reports: the hash of the device's DeviceKey, in the form the
/// catalog page reports it, so every GPU of one kind is one device in the training data
/// (RFC 0019.13 §4.1).
inline std::string deviceKindId(const hipDeviceProp_t& properties)
{
    return std::to_string(
        hipdnn_plugin_sdk::ingestor::DeviceKey{devicePropertiesFrom(properties)}.hash());
}

/// The device columns of a CSV row, in order: `device`, then the `device.*` features the
/// runtime extracts for the same device, under the names the catalog page publishes them.
inline std::vector<std::pair<std::string, std::string>>
    deviceColumns(const hipDeviceProp_t& properties)
{
    std::vector<std::pair<std::string, std::string>> columns{{"device", deviceKindId(properties)}};
    for(const auto& [name, value] : hipdnn_plugin_sdk::heuristics::deviceFeatureValues(properties))
    {
        std::ostringstream text;
        text.precision(std::numeric_limits<double>::max_digits10);
        std::visit([&text](auto number) { text << number; }, value);
        columns.emplace_back("device." + name, text.str());
    }
    return columns;
}

} // namespace hipdnn_bench
