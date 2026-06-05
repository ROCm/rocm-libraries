// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "asm_fmha_v3_fwd_configs.hpp"
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>

namespace asm_sdpa_engine
{

/**
 * @brief A test case containing a graph and name.
 *
 * Uses shared_ptr for Graph since Graph is not copyable.
 */
struct GraphTestCase
{
    std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
    std::string name;
    std::string arch;

    GraphTestCase(std::shared_ptr<hipdnn_frontend::graph::Graph> g,
                  std::string desc,
                  std::string archId)
        : graph(std::move(g))
        , name(std::move(desc))
        , arch(std::move(archId))
    {
    }

    /**
     * @brief GTest-compatible test name generator.
     *
     * Can be used directly in INSTANTIATE_TEST_SUITE_P as the test name generator:
     * @code
     * INSTANTIATE_TEST_SUITE_P(Suite, Fixture,
     *                          testing::ValuesIn(testCases),
     *                          GraphTestCase::getName);
     * @endcode
     *
     * @param info The test parameter info from GTest
     * @return The test name
     */
    static std::string getName(const testing::TestParamInfo<GraphTestCase>& info)
    {
        return info.param.name;
    }
};

/**
 * @brief Generates a descriptive string for a kernel config.
 *
 * @param config The kernel configuration
 * @return A human-readable description of the config
 */
std::string getConfigDescription(const fmha_v3_fwdConfig& config);

/**
 * @brief Converts a kernel config to a compatible hipdnn_frontend::Graph.
 *
 * Creates a graph with dimensions matching the config's hdim_q and hdim_v,
 * using arbitrary values for batch, num_heads, seq_q, and seq_kv.
 * Supports all MaskType values and BATCH/GROUP modes.
 *
 * @param config The fmha_v3_fwdConfig containing kernel configuration
 * @return GraphTestCase containing the graph and description
 */
GraphTestCase configToCompatibleGraphTestCase(const fmha_v3_fwdConfig& config);

/**
 * @brief Generates compatible graph test cases for all configs.
 * @note ConfigType requires a corresponding configToCompatibleGraphTestCase and getConfigDescription function
 * @todo If we upgrade to C++20, add a concept that guarantees these functions are declared
 *
 * @tparam ConfigType The config type
 * @param configMap The map of all configs
 * @return Vector of GraphTestCase objects for each config
 */
template <typename ConfigType>
std::vector<GraphTestCase>
    getCompatibleGraphTestCases(const std::unordered_map<std::string, ConfigType>& configMap)
{
    std::vector<GraphTestCase> testCases;
    testCases.reserve(configMap.size());
    for(const auto& [key, config] : configMap)
    {
        testCases.push_back(configToCompatibleGraphTestCase(config));
    }
    return testCases;
}

} // namespace asm_sdpa_engine
