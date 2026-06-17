// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <unordered_map>
#include <vector>

#include <MiopenPlugin.hpp>
#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DeviceQuery.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include <hipdnn_data_sdk/utilities/LoadGraphAndTensors.hpp>

#include "HipdnnMiopenContext.hpp"
#include "HipdnnMiopenHandle.hpp"

namespace test_helpers
{

class TestGoldenReferenceGpu : public testing::TestWithParam<std::filesystem::path>
{
protected:
    hipdnn_data_sdk::utilities::GraphAndTensorMap _graphAndTensors;
    std::optional<hipdnn_test_sdk::utilities::BundleMetadata> _bundleMetadata;
    hipdnnEnginePluginHandle_t _handle;
    flatbuffers::DetachedBuffer _engineConfigBuffer;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>
        _referenceOutputTensors;

    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        const auto& path = GetParam();

        // TODO: Temporary fix until reference data can be properly installed
        if(path.empty())
        {
            HIPDNN_PLUGIN_LOG_WARN("Reference not found for Gpu golden reference test");
            GTEST_SKIP();
        }

        // Load bundle metadata and apply device-specific guards.
        _bundleMetadata = hipdnn_test_sdk::utilities::loadBundleMetadata(path);
        if(_bundleMetadata)
        {
            if(auto reason = hipdnn_test_sdk::utilities::checkVramRequirement(
                   *_bundleMetadata, hipdnn_test_sdk::utilities::currentDeviceTotalVramMb()))
            {
                GTEST_SKIP() << *reason;
            }

            if(auto reason = hipdnn_test_sdk::utilities::checkArchCompatibility(
                   *_bundleMetadata, hipdnn_test_sdk::utilities::currentDeviceArch()))
            {
                GTEST_SKIP() << *reason;
            }
        }

        hipdnnPluginStatus_t status = hipdnnEnginePluginCreateImpl(&_handle);
        ASSERT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);

        _engineConfigBuffer = hipdnn_test_sdk::utilities::createValidEngineConfig(1).Release();

        _graphAndTensors = hipdnn_test_sdk::utilities::loadGraphAndTensors(path);
        _referenceOutputTensors = _graphAndTensors.extractAndClearOutputTensorData();
    }

    void goldenReferenceTestSuite(float absoluteTolerance, float relativeTolerance)
    {
        SKIP_IF_WINDOWS();

        hipdnnPluginConstData_t opGraph
            = {_graphAndTensors.graphBuffer.data(), _graphAndTensors.graphBuffer.size()};

        hipdnnPluginConstData_t engineConfig
            = {_engineConfigBuffer.data(), _engineConfigBuffer.size()};

        hipdnnPluginStatus_t status;
        hipdnnEnginePluginExecutionContext_t executionContext;
        status = hipdnnEnginePluginCreateExecutionContextImpl(
            _handle, &engineConfig, &opGraph, &executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
        auto deviceBuffers = _graphAndTensors.deviceBuffers();

        status = hipdnnEnginePluginExecuteOpGraphImpl(_handle,
                                                      executionContext,
                                                      nullptr,
                                                      deviceBuffers.data(),
                                                      static_cast<uint32_t>(deviceBuffers.size()));
        EXPECT_EQ(status, hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_SUCCESS);
        for(auto uid : _graphAndTensors.outputTensorUids)
        {
            _graphAndTensors.tensorMap.at(uid)->markDeviceModified();
        }

        EXPECT_TRUE(_graphAndTensors.validateTensors(
            _referenceOutputTensors, absoluteTolerance, relativeTolerance));
    }
};

using hipdnn_test_sdk::utilities::getGoldenReferenceParams;

}
