// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace hipdnn_plugin_sdk::heuristics
{

/// @brief The common device vocabulary for HIP and descriptor device properties.
template <typename TProperties>
inline std::vector<std::pair<std::string, std::variant<int64_t, double>>>
    deviceFeatureValues(const TProperties& properties)
{
    const auto integral
        = [](auto value) { return std::variant<int64_t, double>{static_cast<int64_t>(value)}; };
    const double bandwidth = properties.memoryClockRate > 0 && properties.memoryBusWidth > 0
                                 ? 2.0 * static_cast<double>(properties.memoryClockRate) * 1000.0
                                       * (static_cast<double>(properties.memoryBusWidth) / 8.0)
                                 : 0.0;
    return {{"cu_count", integral(properties.multiProcessorCount)},
            {"multi_processor_count", integral(properties.multiProcessorCount)},
            {"warp_size", integral(properties.warpSize)},
            {"total_global_mem", integral(properties.totalGlobalMem)},
            {"memory_bus_width", integral(properties.memoryBusWidth)},
            {"memory_clock_rate", integral(properties.memoryClockRate)},
            {"lds_size", integral(properties.sharedMemPerBlock)},
            {"peak_memory_bandwidth", bandwidth}};
}

} // namespace hipdnn_plugin_sdk::heuristics
