// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include <hipdnn_sdk/utilities/LoadGraphAndTensors.hpp>

namespace hipdnn_sdk::test_utilities
{

class TestGoldenReferenceCpu : public ::testing::TestWithParam<std::filesystem::path>
{
protected:
    GraphAndTensorMap _graphAndTensors;
    std::unordered_map<int64_t, std::unique_ptr<ITensor>> _referenceOutputTensors;

    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
        const auto& path = GetParam();

        // TODO: Temporary fix until reference data can be properly installed
        if(path.empty())
        {
            HIPDNN_LOG_WARN("Reference not found for Cpu golden reference test");
            GTEST_SKIP();
        }

        _graphAndTensors = loadGraphAndTensors(path);
        _referenceOutputTensors = _graphAndTensors.extractAndClearOutputTensorData();
    }

    void goldenReferenceTestSuite(float absoluteTolerance, float relativeTolerance)
    {
        auto tensorMap = _graphAndTensors.hostBufferMap();
        EXPECT_EQ(tensorMap.size(), 6);

        CpuReferenceGraphExecutor().execute(
            _graphAndTensors.graphBuffer.data(), _graphAndTensors.graphBuffer.size(), tensorMap);

        EXPECT_TRUE(_graphAndTensors.validateTensors(
            _referenceOutputTensors, absoluteTolerance, relativeTolerance));
    }
};

auto getGoldenReferenceParams(const std::filesystem::path& subDirectory)
{
    return testing::ValuesIn(filesInDirectoryWithExtReturnEmptyPathOnThrow(
        hipdnn_sdk::utilities::getCurrentExecutableDirectory() / "../lib/hipdnn_reference_data"
            / subDirectory,
        ".json"));
}
}
