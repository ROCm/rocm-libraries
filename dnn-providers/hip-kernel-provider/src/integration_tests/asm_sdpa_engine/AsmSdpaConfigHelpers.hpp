// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "asm_fmha_v3_fwd_configs.hpp"
#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>

namespace asm_sdpa_engine
{

/**
 * @brief Lightweight, GPU-free parameters for building an SDPA forward graph.
 *
 * The graph is built on demand in the test fixture (buildSdpaFwdGraph) so it and
 * its GPU-backed backend descriptors are destroyed while the HIP runtime is still
 * alive, rather than during GTest's atexit teardown of the static parameter list.
 */
struct GraphTestCase
{
    fmha_v3_fwdConfig config;

    int64_t batch = 2;
    int64_t numHeads = 4;
    int64_t seqQ = 256;
    int64_t seqKv = 128;

    std::optional<float> attnScale;

    std::string name;
    std::string arch;

    GraphTestCase(fmha_v3_fwdConfig cfg, std::string desc, std::string archId)
        : config(std::move(cfg))
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
 * @brief Wraps a kernel config in a GraphTestCase descriptor with default dimensions.
 */
GraphTestCase configToTestCase(const fmha_v3_fwdConfig& config);

/**
 * @brief Builds the SDPA forward graph topology described by a GraphTestCase.
 */
std::shared_ptr<hipdnn_frontend::graph::Graph> buildSdpaFwdGraph(const GraphTestCase& testCase);

/**
 * @brief Generates compatible graph test case descriptors for all configs.
 * @note ConfigType requires a corresponding configToTestCase and getConfigDescription function
 * @todo If we upgrade to C++20, add a concept that guarantees these functions are declared
 *
 * @tparam ConfigType The config type
 * @param configMap The map of all configs
 * @return Vector of GraphTestCase descriptors for each config
 */
template <typename ConfigType>
std::vector<GraphTestCase>
    getCompatibleGraphTestCases(const std::unordered_map<std::string, ConfigType>& configMap)
{
    std::vector<GraphTestCase> testCases;
    testCases.reserve(configMap.size());
    for(const auto& [key, config] : configMap)
    {
        testCases.push_back(configToTestCase(config));
    }
    return testCases;
}

} // namespace asm_sdpa_engine
