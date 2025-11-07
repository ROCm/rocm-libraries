// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include "MiopenLegacyPlugin.hpp"
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/BatchnormCommon.hpp"
#include "common/Helpers.hpp"

using namespace hipdnn_sdk::test_utilities;
using namespace test_bn_common;
using namespace test_helpers;

template <typename InputType, typename IntermediateType>
class BatchnormBwdActFusionExecuteGraphBase : public ::testing::TestWithParam<BatchnormTestCase>
{
protected:
    TensorLayout _layout;

    BatchnormBwdActFusionExecuteGraphBase(TensorLayout layout)
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

    void runBwdActFusionBatchnormGraph(const BatchnormTestCase& testCase,
                                       hipdnn_sdk::data_objects::DataType inputDataType,
                                       InputType tolerance)
    {
        std::vector<int64_t> dims = testCase.dims;
        std::vector<int64_t> derivedDims = getDerivedShape(dims);

        int64_t tensorUid = 1;
        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        // Input tensor (UID 1)
        PinnedTensor<InputType> xTensor(dims, _layout);
        const auto xTensorUid = tensorUid++;
        deviceBuffers.push_back(generateRandomDeviceBuffer(xTensor,
                                                           static_cast<int>(xTensorUid),
                                                           static_cast<InputType>(-1.8f),
                                                           static_cast<InputType>(1.8f),
                                                           testCase.seed));

        // Scale tensor (UID 2)
        PinnedTensor<IntermediateType> scaleTensor(derivedDims);
        const auto scaleTensorUid = tensorUid++;
        deviceBuffers.push_back(generateRandomDeviceBuffer(scaleTensor,
                                                           static_cast<int>(scaleTensorUid),
                                                           static_cast<IntermediateType>(0.5f),
                                                           static_cast<IntermediateType>(1.5f),
                                                           testCase.seed));

        // Bias tensor (UID 3)
        PinnedTensor<IntermediateType> biasTensor(derivedDims);
        const auto biasTensorUid = tensorUid++;
        deviceBuffers.push_back(generateRandomDeviceBuffer(biasTensor,
                                                           static_cast<int>(biasTensorUid),
                                                           static_cast<IntermediateType>(-0.1f),
                                                           static_cast<IntermediateType>(0.1f),
                                                           testCase.seed));

        // Mean tensor (UID 4)
        PinnedTensor<IntermediateType> meanTensor(derivedDims);
        const auto meanTensorUid = tensorUid++;
        deviceBuffers.push_back(generateRandomDeviceBuffer(meanTensor,
                                                           static_cast<int>(meanTensorUid),
                                                           static_cast<IntermediateType>(-0.1f),
                                                           static_cast<IntermediateType>(0.1f),
                                                           testCase.seed));

        // Inv variance tensor (UID 5)
        PinnedTensor<IntermediateType> invVarianceTensor(derivedDims);
        const auto invVarianceTensorUid = tensorUid++;
        deviceBuffers.push_back(generateRandomDeviceBuffer(invVarianceTensor,
                                                           static_cast<int>(invVarianceTensorUid),
                                                           static_cast<IntermediateType>(0.5f),
                                                           static_cast<IntermediateType>(2.0f),
                                                           testCase.seed));

        // dy tensor (UID 6)
        PinnedTensor<InputType> dyTensor(dims, _layout);
        const auto dyTensorUid = tensorUid++;
        deviceBuffers.push_back(generateRandomDeviceBuffer(dyTensor,
                                                           static_cast<int>(dyTensorUid),
                                                           static_cast<InputType>(-1.8f),
                                                           static_cast<InputType>(1.8f),
                                                           testCase.seed));

        // dx output tensor (UID 7)
        PinnedTensor<InputType> dxTensor(dims, _layout);
        const auto dxTensorUid = tensorUid++;
        deviceBuffers.push_back(generateEmptyDeviceBuffer(dxTensor, static_cast<int>(dxTensorUid)));

        // dscale output tensor (UID 8)
        PinnedTensor<IntermediateType> dscaleTensor(derivedDims);
        const auto dscaleTensorUid = tensorUid++;
        deviceBuffers.push_back(
            generateEmptyDeviceBuffer(dscaleTensor, static_cast<int>(dscaleTensorUid)));

        // dbias output tensor (UID 9)
        PinnedTensor<IntermediateType> dbiasTensor(derivedDims);
        const auto dbiasTensorUid = tensorUid;
        deviceBuffers.push_back(
            generateEmptyDeviceBuffer(dbiasTensor, static_cast<int>(dbiasTensorUid)));

        auto batchnormBuilder = hipdnn_sdk::test_utilities::createValidBatchnormInferActBwdGraph(
            xTensor.strides(), xTensor.dims(), true, inputDataType);

        hipdnnPluginConstData_t opGraph;
        opGraph.ptr = batchnormBuilder.GetBufferPointer();
        opGraph.size = batchnormBuilder.GetSize();

        auto engineConfigBuilder = hipdnn_sdk::test_utilities::createValidEngineConfig(1);
        hipdnnPluginConstData_t engineConfig;
        engineConfig.ptr = engineConfigBuilder.GetBufferPointer();
        engineConfig.size = engineConfigBuilder.GetSize();

        hipdnnPluginStatus_t status;
        hipdnnEnginePluginExecutionContext_t executionContext;
        status = hipdnnEnginePluginCreateExecutionContextImpl(
            _handle, &engineConfig, &opGraph, &executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        status = hipdnnEnginePluginExecuteOpGraphImpl(_handle,
                                                      executionContext,
                                                      nullptr,
                                                      deviceBuffers.data(),
                                                      static_cast<uint32_t>(deviceBuffers.size()));
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        dxTensor.memory().markDeviceModified();
        dscaleTensor.memory().markDeviceModified();
        dbiasTensor.memory().markDeviceModified();

        status = hipdnnEnginePluginDestroyExecutionContextImpl(_handle, executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        Tensor<InputType> dxTensorCpu(dims, _layout);
        Tensor<IntermediateType> dscaleTensorCpu(derivedDims);
        Tensor<IntermediateType> dbiasTensorCpu(derivedDims);

        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorUid] = xTensor.memory().hostData();
        variantPack[scaleTensorUid] = scaleTensor.memory().hostData();
        variantPack[biasTensorUid] = biasTensor.memory().hostData();
        variantPack[meanTensorUid] = meanTensor.memory().hostData();
        variantPack[invVarianceTensorUid] = invVarianceTensor.memory().hostData();
        variantPack[dyTensorUid] = dyTensor.memory().hostData();
        variantPack[dxTensorUid] = dxTensorCpu.memory().hostData();
        variantPack[dscaleTensorUid] = dscaleTensorCpu.memory().hostData();
        variantPack[dbiasTensorUid] = dbiasTensorCpu.memory().hostData();

        CpuReferenceGraphExecutor().execute(
            batchnormBuilder.GetBufferPointer(), batchnormBuilder.GetSize(), variantPack);

        CpuFpReferenceMiopenRmsValidation<InputType> rmsValidationInput(tolerance);
        CpuFpReferenceMiopenRmsValidation<IntermediateType> rmsValidationIntermediate(tolerance);

        EXPECT_TRUE(rmsValidationInput.allClose(dxTensorCpu, dxTensor));
        EXPECT_TRUE(rmsValidationIntermediate.allClose(dscaleTensorCpu, dscaleTensor));
        EXPECT_TRUE(rmsValidationIntermediate.allClose(dbiasTensorCpu, dbiasTensor));
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

class TestGpuMiopenBatchnormBwdActFusionExecuteGraphNchwFp32
    : public BatchnormBwdActFusionExecuteGraphBase<float, float>
{
public:
    TestGpuMiopenBatchnormBwdActFusionExecuteGraphNchwFp32()
        : BatchnormBwdActFusionExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

TEST_P(TestGpuMiopenBatchnormBwdActFusionExecuteGraphNchwFp32, Correctness)
{
    const auto& testCase = GetParam();
    runBwdActFusionBatchnormGraph(testCase,
                                  hipdnn_sdk::data_objects::DataType::FLOAT,
                                  batchnorm::getToleranceBackward<float>() * 2.0f);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdActFusionExecuteGraphNchwFp32,
                         testing::ValuesIn(getBnBwdTestCases()));
