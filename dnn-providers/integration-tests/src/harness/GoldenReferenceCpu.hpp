// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <gtest/gtest.h>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "harness/BundleMetadata.hpp"
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

namespace
{

inline bool isTensorLoadFailure(const std::runtime_error& error)
{
    constexpr std::string_view PREFIX = "Error: could not load tensor ";
    const std::string_view message(error.what());
    return message.size() >= PREFIX.size() && message.substr(0, PREFIX.size()) == PREFIX;
}

} // namespace

namespace hipdnn_integration_tests
{

class TestGoldenReferenceCpu : public ::testing::TestWithParam<std::filesystem::path>
{
protected:
    hipdnn_test_sdk::utilities::GraphAndTensorMap _graphAndTensors;
    std::optional<hipdnn_integration_tests::BundleMetadata> _bundleMetadata;
    std::unordered_map<int64_t, std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>>
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

        // Load bundle metadata if a .meta.json companion file exists.
        // CPU harness has no device-specific guards (no VRAM, no arch).
        // Future: add CPU-relevant checks here (e.g., minimum RAM).
        _bundleMetadata = hipdnn_integration_tests::loadBundleMetadata(path);

        try
        {
            _graphAndTensors = hipdnn_test_sdk::utilities::loadGraphAndTensors(path);
        }
        catch(const std::runtime_error& e)
        {
            if(!isTensorLoadFailure(e))
            {
                throw;
            }

            HIPDNN_SDK_LOG_WARN(
                "Tensor data not available for CPU golden reference test: " << e.what());
            GTEST_SKIP() << "Tensor data not available (DVC not pulled?): " << e.what();
        }

        _referenceOutputTensors = _graphAndTensors.extractAndClearOutputTensorData();
    }

    void goldenReferenceTestSuite(float absoluteTolerance, float relativeTolerance)
    {
        auto tensorMap = _graphAndTensors.hostBufferMap();

        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
            _graphAndTensors.graphBuffer.data(), _graphAndTensors.graphBuffer.size(), tensorMap);

        EXPECT_TRUE(_graphAndTensors.validateTensors(
            _referenceOutputTensors, absoluteTolerance, relativeTolerance));
    }
};

inline auto getGoldenReferenceParams(const std::filesystem::path& subDirectory)
{
    auto dir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
               / "../lib/integration-test-bundles" / subDirectory;

    auto paths = hipdnn_test_sdk::utilities::scanBundleJsonFiles(dir);
    if(paths.empty())
    {
        return testing::ValuesIn(std::vector<std::filesystem::path>{""});
    }
    return testing::ValuesIn(paths);
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
