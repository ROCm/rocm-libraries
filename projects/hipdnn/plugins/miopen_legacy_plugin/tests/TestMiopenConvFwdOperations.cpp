// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/Workspace.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/ConvolutionCommon.hpp"
#include "common/Helpers.hpp"

using namespace hipdnn_sdk::test_utilities;
using namespace test_conv_common;
using namespace test_helpers;

namespace
{

template <typename DataType>
class ConvFwdExecuteGraphBase : public ::testing::TestWithParam<ConvTestCase>
{
protected:
    TensorLayout _layout;

    ConvFwdExecuteGraphBase(TensorLayout layout)
        : _layout(std::move(layout))
    {
    }

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        hipdnnPluginStatus_t status = hipdnnEnginePluginCreate(&_handle);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnPluginStatus_t status = hipdnnEnginePluginDestroy(_handle);
            ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
        }
    }

    void runConvFwdGraph(const ConvTestCase& testCase,
                         hipdnn_sdk::data_objects::DataType dataType,
                         DataType tolerance)
    {
        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        PinnedTensor<DataType> xTensor(testCase.xDims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(
            xTensor, 1, static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed));

        PinnedTensor<DataType> wTensor(testCase.wDims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(
            wTensor, 2, static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed));

        PinnedTensor<DataType> yTensor(testCase.yDims, _layout);
        deviceBuffers.push_back(generateEmptyDeviceBuffer(yTensor, 3));

        auto convBuilder = createValidConvFwdGraph(xTensor.dims(),
                                                   xTensor.strides(),
                                                   wTensor.dims(),
                                                   wTensor.strides(),
                                                   yTensor.dims(),
                                                   yTensor.strides(),
                                                   testCase.convPrePadding,
                                                   testCase.convPostPadding,
                                                   testCase.convStride,
                                                   testCase.convDilation,
                                                   dataType);

        hipdnnPluginConstData_t opGraph;
        opGraph.ptr = convBuilder.GetBufferPointer();
        opGraph.size = convBuilder.GetSize();

        auto engineConfigBuilder = createValidEngineConfig(1);
        hipdnnPluginConstData_t engineConfig;
        engineConfig.ptr = engineConfigBuilder.GetBufferPointer();
        engineConfig.size = engineConfigBuilder.GetSize();

        hipdnnPluginStatus_t status;
        hipdnnEnginePluginExecutionContext_t executionContext;
        status = hipdnnEnginePluginCreateExecutionContext(
            _handle, &engineConfig, &opGraph, &executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        size_t workspaceSize;
        status = hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
            _handle, executionContext, &workspaceSize);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
        hipdnn_sdk::utilities::Workspace workspace(workspaceSize);

        status = hipdnnEnginePluginExecuteOpGraph(_handle,
                                                  executionContext,
                                                  workspace.get(),
                                                  deviceBuffers.data(),
                                                  static_cast<uint32_t>(deviceBuffers.size()));
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        yTensor.memory().markDeviceModified();

        status = hipdnnEnginePluginDestroyExecutionContext(_handle, executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        Tensor<DataType> xTensorCpu(xTensor.dims(), _layout);
        xTensorCpu.fillWithRandomValues(
            static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed);
        Tensor<DataType> wTensorCpu(wTensor.dims(), _layout);
        wTensorCpu.fillWithRandomValues(
            static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed);
        Tensor<DataType> yTensorCpu(yTensor.dims(), _layout);

        CpuFpReferenceConvolutionImpl<DataType, float>::convFwdInference(xTensorCpu,
                                                                         wTensorCpu,
                                                                         yTensorCpu,
                                                                         testCase.convStride,
                                                                         testCase.convDilation,
                                                                         testCase.convPrePadding);

        CpuFpReferenceValidation<DataType> cpuRefValidationInput(tolerance, tolerance);

        EXPECT_TRUE(cpuRefValidationInput.allClose(yTensorCpu, yTensor));
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

std::vector<ConvTestCase> getTestCases()
{
    unsigned seed = getGlobalTestSeed();
    ;

    return {
        {{1, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
    };
}

class TestGpuMiopenConvFwdExecuteGraphNchwFp32 : public ConvFwdExecuteGraphBase<float>
{
public:
    TestGpuMiopenConvFwdExecuteGraphNchwFp32()
        : ConvFwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

} // namespace

TEST_P(TestGpuMiopenConvFwdExecuteGraphNchwFp32, Correctness)
{
    const ConvTestCase& testCase = GetParam();
    runConvFwdGraph(
        testCase, hipdnn_sdk::data_objects::DataType::FLOAT, conv::getToleranceFwd<float>());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdExecuteGraphNchwFp32,
                         testing::ValuesIn(getTestCases()));
