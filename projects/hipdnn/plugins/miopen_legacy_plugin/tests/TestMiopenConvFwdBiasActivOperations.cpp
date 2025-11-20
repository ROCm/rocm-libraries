// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <MiopenLegacyPlugin.hpp>
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/Workspace.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/ConvolutionFusionCommon.hpp"
#include "common/Helpers.hpp"

using namespace hipdnn_sdk::test_utilities;
using namespace test_fusion_common;
using namespace test_helpers;

namespace
{

template <typename DataType>
class ConvFwdBiasActivExecuteGraphBase : public ::testing::TestWithParam<ConvBiasActivTestCase>
{
protected:
    TensorLayout _layout;

    ConvFwdBiasActivExecuteGraphBase(TensorLayout layout)
        : _layout(std::move(layout))
    {
    }

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        hipdnnPluginStatus_t status = hipdnnEnginePluginCreateImpl(&_handle);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnPluginStatus_t status = hipdnnEnginePluginDestroyImpl(_handle);
            ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
        }
    }

    void runConvFwdBiasActivGraph(const ConvBiasActivTestCase& testCase,
                                  hipdnn_sdk::data_objects::DataType dataType,
                                  DataType tolerance)
    {
        int64_t tensorUid = 1;
        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        PinnedTensor<DataType> xTensor(testCase.conv.xDims, _layout);
        const auto xTensorUid = tensorUid++;
        deviceBuffers.push_back(generateRandomDeviceBuffer(xTensor,
                                                           static_cast<int>(xTensorUid),
                                                           static_cast<DataType>(-1.0f),
                                                           static_cast<DataType>(1.0f),
                                                           testCase.conv.seed));

        PinnedTensor<DataType> wTensor(testCase.conv.wDims, _layout);
        const auto wTensorUid = tensorUid++;
        deviceBuffers.push_back(generateRandomDeviceBuffer(wTensor,
                                                           static_cast<int>(wTensorUid),
                                                           static_cast<DataType>(-1.0f),
                                                           static_cast<DataType>(1.0f),
                                                           testCase.conv.seed));

        // Virtual y_conv tensor
        tensorUid++;

        PinnedTensor<DataType> biasTensor(getDerivedShape(testCase.conv.wDims), _layout);
        int64_t biasTensorUid;
        if(testCase.doBias)
        {
            biasTensorUid = tensorUid++;
            deviceBuffers.push_back(generateRandomDeviceBuffer(biasTensor,
                                                               static_cast<int>(biasTensorUid),
                                                               static_cast<DataType>(-1.0f),
                                                               static_cast<DataType>(1.0f),
                                                               testCase.conv.seed));
            // Virtual y_bias tensor
            tensorUid++;
        }

        PinnedTensor<DataType> yTensor(testCase.conv.yDims, _layout);
        const auto yTensorUid = tensorUid;
        deviceBuffers.push_back(generateEmptyDeviceBuffer(yTensor, static_cast<int>(yTensorUid)));

        auto fusionBuilder = createValidConvFwdBiasActivGraph(xTensor.dims(),
                                                              xTensor.strides(),
                                                              wTensor.dims(),
                                                              wTensor.strides(),
                                                              yTensor.dims(),
                                                              yTensor.strides(),
                                                              testCase.conv.convPrePadding,
                                                              testCase.conv.convPostPadding,
                                                              testCase.conv.convStride,
                                                              testCase.conv.convDilation,
                                                              testCase.doBias,
                                                              testCase.activ.mode,
                                                              testCase.activ.reluLowerClip,
                                                              testCase.activ.reluUpperClip,
                                                              testCase.activ.reluLowerClipSlope,
                                                              testCase.activ.swishBeta,
                                                              testCase.activ.eluAlpha,
                                                              testCase.activ.softplusBeta,
                                                              dataType);

        hipdnnPluginConstData_t opGraph;
        opGraph.ptr = fusionBuilder.GetBufferPointer();
        opGraph.size = fusionBuilder.GetSize();

        auto engineConfigBuilder = createValidEngineConfig(1);
        hipdnnPluginConstData_t engineConfig;
        engineConfig.ptr = engineConfigBuilder.GetBufferPointer();
        engineConfig.size = engineConfigBuilder.GetSize();

        hipdnnPluginStatus_t status;
        hipdnnEnginePluginExecutionContext_t executionContext;
        status = hipdnnEnginePluginCreateExecutionContextImpl(
            _handle, &engineConfig, &opGraph, &executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        size_t workspaceSize;
        status = hipdnnEnginePluginGetWorkspaceSizeFromExecutionContextImpl(
            _handle, executionContext, &workspaceSize);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
        hipdnn_sdk::utilities::Workspace workspace(workspaceSize);

        status = hipdnnEnginePluginExecuteOpGraphImpl(_handle,
                                                      executionContext,
                                                      workspace.get(),
                                                      deviceBuffers.data(),
                                                      static_cast<uint32_t>(deviceBuffers.size()));
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        yTensor.memory().markDeviceModified();

        status = hipdnnEnginePluginDestroyExecutionContextImpl(_handle, executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        Tensor<DataType> yTensorCpu(yTensor.dims(), _layout);

        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorUid] = xTensor.memory().hostData();
        variantPack[wTensorUid] = wTensor.memory().hostData();
        if(testCase.doBias)
        {
            variantPack[biasTensorUid] = biasTensor.memory().hostData();
        }
        variantPack[yTensorUid] = yTensorCpu.memory().hostData();

        CpuReferenceGraphExecutor().execute(
            fusionBuilder.GetBufferPointer(), fusionBuilder.GetSize(), variantPack);

        CpuFpReferenceValidation<DataType> cpuRefValidationInput(tolerance, tolerance);

        EXPECT_TRUE(cpuRefValidationInput.allClose(yTensorCpu, yTensor));
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

std::vector<ConvBiasActivTestCase> getTestCases()
{
    unsigned seed = getGlobalTestSeed();
    ;

    return {{{{1, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
             true,
             {hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD, {}, {}, {}, {}, {}, {}}}};
}

class TestGpuMiopenConvFwdBiasActivExecuteGraphNchwFp32
    : public ConvFwdBiasActivExecuteGraphBase<float>
{
public:
    TestGpuMiopenConvFwdBiasActivExecuteGraphNchwFp32()
        : ConvFwdBiasActivExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

} // namespace

// TODO: re-enable after integration tests are added
TEST_P(TestGpuMiopenConvFwdBiasActivExecuteGraphNchwFp32, DISABLED_Correctness)
{
    const ConvBiasActivTestCase& testCase = GetParam();
    runConvFwdBiasActivGraph(
        testCase, hipdnn_sdk::data_objects::DataType::FLOAT, conv::getToleranceFwd<float>());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenConvFwdBiasActivExecuteGraphNchwFp32,
                         testing::ValuesIn(getTestCases()));
