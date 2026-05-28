// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "GraphTest.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"

#include <flatbuffers/flatbuffer_builder.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace asm_sdpa_engine
{

/**
 * @brief Converts an SDPA forward config to a valid graph builder.
 *
 * Creates a graph with dimensions matching the config's hdim_q and hdim_v,
 * using arbitrary values for batch, num_heads, seq_q, and seq_kv.
 * Supports all MaskType values and BATCH/GROUP modes.
 *
 * @param config The fmha_v3_fwdConfig containing kernel configuration
 * @return flatbuffers::FlatBufferBuilder A builder containing the graph
 */
flatbuffers::FlatBufferBuilder configToCompatibleGraph(const fmha_v3_fwdConfig& config);

/**
 * @brief Retrieves all configurations matching a specific GPU architecture.
 *
 * Filters the provided configuration map and returns only those configurations
 * whose `arch` field matches the specified architecture identifier.
 *
 * @tparam ConfigType The configuration type (must have an `arch` member)
 * @param archId The GPU architecture identifier to filter by (e.g., "gfx942")
 * @param configMap A map of configuration names to configuration objects
 * @return std::vector<ConfigType> A vector of configurations matching the specified architecture
 */
template <class ConfigType>
inline std::vector<ConfigType>
    getConfigsForArch(const std::string& archId,
                      const std::unordered_map<std::string, ConfigType>& configMap)
{
    std::vector<ConfigType> configs;
    for(const auto& [key, config] : configMap)
    {
        if(config.arch == archId)
        {
            configs.push_back(config);
        }
    }

    return configs;
}

/**
 * @brief Creates compatible test graphs for all configurations matching a GPU architecture.
 *
 * Filters configurations by architecture using getConfigsForArch, then converts each
 * matching configuration to a compatible graph using configToCompatibleGraph.
 * Each resulting GraphTest contains the graph and the configuration's kernel name.
 *
 * @tparam ConfigType The configuration type (must have `arch` and `co_name` members)
 * @param archId The GPU architecture identifier to filter by (e.g., "gfx942")
 * @param configMap A map of configuration names to configuration objects
 * @return std::vector<GraphTest> A vector of GraphTest objects for the specified architecture
 */
template <class ConfigType>
auto getCompatibleGraphsForArch(const std::string& archId,
                                const std::unordered_map<std::string, ConfigType>& configMap)
{
    std::vector<GraphTest> graphs;
    for(const auto& config : getConfigsForArch(archId, configMap))
    {
        graphs.emplace_back(configToCompatibleGraph(config), config.co_name);
    }

    return graphs;
}

} // namespace asm_sdpa_engine
