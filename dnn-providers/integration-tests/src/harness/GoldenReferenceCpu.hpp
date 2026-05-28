// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

namespace hipdnn_integration_tests
{

class TestGoldenReferenceCpu : public ::testing::TestWithParam<std::filesystem::path>
{
protected:
    hipdnn_test_sdk::utilities::GraphAndTensorMap _graphAndTensors;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>
        _referenceOutputTensors;

    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
        const auto& path = GetParam();

        // TODO: Temporary fix until reference data can be properly installed
        if(path.empty())
        {
            HIPDNN_SDK_LOG_WARN("Reference not found for Cpu golden reference test");
            GTEST_SKIP();
        }

        _graphAndTensors = hipdnn_test_sdk::utilities::loadGraphAndTensors(path);
        _referenceOutputTensors = _graphAndTensors.extractAndClearOutputTensorData();
    }

    void goldenReferenceTestSuite(float absoluteTolerance, float relativeTolerance)
    {
        SKIP_IF_WINDOWS();

        auto tensorMap = _graphAndTensors.hostBufferMap();

        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
            _graphAndTensors.graphBuffer.data(), _graphAndTensors.graphBuffer.size(), tensorMap);

        EXPECT_TRUE(_graphAndTensors.validateTensors(
            _referenceOutputTensors, absoluteTolerance, relativeTolerance));
    }
};

inline auto getGoldenReferenceParams(const std::filesystem::path& subDirectory)
{
    return testing::ValuesIn(
        hipdnn_test_sdk::utilities::filesInDirectoryWithExtReturnEmptyPathOnThrow(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
                / "../lib/hipdnn_reference_data" / subDirectory,
            ".json"));
}
}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
