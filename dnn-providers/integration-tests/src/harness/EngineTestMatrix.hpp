// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <string>
#include <vector>

#include "harness/SharedHandle.hpp"

namespace hipdnn_integration_tests {

using namespace hipdnn_data_sdk;

// An (engine, testCase) pair where an engine has claimed support for graph built based on the
// testCase.
template <typename TestCase>
struct EngineTestCase {
    int64_t engineId;
    TestCase testCase;
};

// Name generator for EngineTestCase-parameterized tests.
// Produces names like "Fusilli_0", "MIOpen_1", etc.
// Falls back to numeric ID if engine name isn't registered.
template <typename TestCase>
std::string EngineTestNameGenerator(const testing::TestParamInfo<EngineTestCase<TestCase>>& info) {
    std::string engineName;
    try {
        engineName = std::string(utilities::getEngineNameFromId(info.param.engineId));
    } catch (const std::out_of_range&) {
        engineName = "Engine" + std::to_string(info.param.engineId);
    }
    return engineName + "_" + std::to_string(info.index);
}

// Builds a test matrix of (engine, testCase) pairs based on engine capability.
//
// For each test case, builds the graph and queries which engines support it
// using hipDNN frontend `get_ranked_engine_ids`.
//
// Usage:
//   INSTANTIATE_TEST_SUITE_P(Smoke, MyFixture,
//       testing::ValuesIn(BuildEngineTestMatrix<MyFixture, TestCaseType>(
//           testing::Combine(
//               testing::Values(TensorLayout::NCHW),
//               testing::ValuesIn(getTestCases())))),
//       EngineTestNameGenerator<TestCaseType>);
//
// Requirements:
//   FixtureClass must provide:
//     static std::pair<graph::Graph, GraphOutputs> buildGraph(
//         hipdnnHandle_t handle, const TestCase& tc);
template <typename FixtureClass, typename TestCase>
std::vector<EngineTestCase<TestCase>> BuildEngineTestMatrix(
    testing::internal::ParamGenerator<TestCase> testCaseGen) {
    std::vector<EngineTestCase<TestCase>> result;
    hipdnnHandle_t handle = getSharedHandle();

    for (const auto& testCase : testCaseGen) {
        auto [graph, outputs] = FixtureClass::buildGraph(handle, testCase);

        // Query which engines support this graph
        std::vector<int64_t> engineIds;
        auto status = graph.get_ranked_engine_ids(engineIds);
        if (status.is_bad()) {
            // If no currently loaded engine supports the graph an error is
            // returned.
            //
            // NOTE: this could be masking a more serious error. An error here
            // could mean that something is broken, or that there aren't loaded
            // engines supporting this graph (something that's common place for
            // the test runner) - we assume the latter.
            continue;
        }

        for (int64_t engineId : engineIds) {
            result.push_back(EngineTestCase<TestCase>{engineId, testCase});
        }
    }

    return result;
}

}  // namespace hipdnn_integration_tests
