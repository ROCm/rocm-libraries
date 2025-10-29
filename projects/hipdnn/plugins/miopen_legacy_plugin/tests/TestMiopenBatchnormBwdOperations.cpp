// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/BatchnormCommon.hpp"
#include "common/Helpers.hpp"

using namespace hipdnn_sdk::test_utilities;
using namespace test_bn_common;
using namespace test_helpers;

template <typename InputType, typename IntermediateType>
class BatchnormBwdExecuteGraphBase : public ::testing::TestWithParam<Batchnorm2dTestCase>
{
protected:
    TensorLayout _layout;

    BatchnormBwdExecuteGraphBase(TensorLayout layout)
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

    void runBwdBatchnormGraph(Batchnorm2dTestCase testCase,
                              hipdnn_sdk::data_objects::DataType inputDataType,
                              InputType tolerance)
    {
        std::vector<int64_t> dims = {testCase.n, testCase.c, testCase.h, testCase.w};

        std::vector<int64_t> derivedDims = getDerivedShape(dims);

        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        PinnedTensor<InputType> xTensor(dims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(xTensor,
                                                           1,
                                                           static_cast<InputType>(-1.0f),
                                                           static_cast<InputType>(1.0f),
                                                           testCase.seed));

        PinnedTensor<InputType> dyTensor(dims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(dyTensor,
                                                           2,
                                                           static_cast<InputType>(-0.1f),
                                                           static_cast<InputType>(0.1f),
                                                           testCase.seed));

        PinnedTensor<InputType> dxTensor(dims, _layout);
        deviceBuffers.push_back(generateEmptyDeviceBuffer(dxTensor, 3));

        PinnedTensor<IntermediateType> scaleTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(scaleTensor,
                                                           4,
                                                           static_cast<IntermediateType>(-0.1f),
                                                           static_cast<IntermediateType>(0.1f),
                                                           testCase.seed));

        PinnedTensor<IntermediateType> dscaleTensor(derivedDims);
        deviceBuffers.push_back(generateEmptyDeviceBuffer(dscaleTensor, 5));

        PinnedTensor<IntermediateType> dbiasTensor(derivedDims);
        deviceBuffers.push_back(generateEmptyDeviceBuffer(dbiasTensor, 6));

        PinnedTensor<IntermediateType> meanTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(meanTensor,
                                                           7,
                                                           static_cast<IntermediateType>(-0.1f),
                                                           static_cast<IntermediateType>(0.1f),
                                                           testCase.seed));

        PinnedTensor<IntermediateType> invVarianceTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(invVarianceTensor,
                                                           8,
                                                           static_cast<IntermediateType>(1.9f),
                                                           static_cast<IntermediateType>(2.0f),
                                                           testCase.seed));

        auto batchnormBuilder = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph(
            dyTensor.strides(), dyTensor.dims(), true, inputDataType);

        hipdnnPluginConstData_t opGraph;
        opGraph.ptr = batchnormBuilder.GetBufferPointer();
        opGraph.size = batchnormBuilder.GetSize();

        auto engineConfigBuilder = hipdnn_sdk::test_utilities::createValidEngineConfig(1);
        hipdnnPluginConstData_t engineConfig;
        engineConfig.ptr = engineConfigBuilder.GetBufferPointer();
        engineConfig.size = engineConfigBuilder.GetSize();

        hipdnnPluginStatus_t status;
        hipdnnEnginePluginExecutionContext_t executionContext;
        status = hipdnnEnginePluginCreateExecutionContext(
            _handle, &engineConfig, &opGraph, &executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        status = hipdnnEnginePluginExecuteOpGraph(_handle,
                                                  executionContext,
                                                  nullptr,
                                                  deviceBuffers.data(),
                                                  static_cast<uint32_t>(deviceBuffers.size()));
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        dxTensor.memory().markDeviceModified();
        dscaleTensor.memory().markDeviceModified();
        dbiasTensor.memory().markDeviceModified();

        status = hipdnnEnginePluginDestroyExecutionContext(_handle, executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        Tensor<InputType> xTensorCpu(dims, _layout);
        xTensorCpu.fillWithRandomValues(
            static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), testCase.seed);
        Tensor<InputType> dyTensorCpu(dims, _layout);
        dyTensorCpu.fillWithRandomValues(
            static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), testCase.seed);
        Tensor<InputType> dxTensorCpu(dims, _layout);

        Tensor<IntermediateType> scaleTensorCpu(derivedDims);
        scaleTensorCpu.fillWithRandomValues(static_cast<IntermediateType>(-0.1f),
                                            static_cast<IntermediateType>(0.1f),
                                            testCase.seed);
        Tensor<IntermediateType> dscaleTensorCpu(derivedDims);
        Tensor<IntermediateType> dbiasTensorCpu(derivedDims);
        Tensor<IntermediateType> meanTensorCpu(derivedDims);
        meanTensorCpu.fillWithRandomValues(static_cast<IntermediateType>(-0.1f),
                                           static_cast<IntermediateType>(0.1f),
                                           testCase.seed);

        Tensor<IntermediateType> invVarianceTensorCpu(derivedDims);
        invVarianceTensorCpu.fillWithRandomValues(static_cast<IntermediateType>(1.9f),
                                                  static_cast<IntermediateType>(2.0f),
                                                  testCase.seed);

        CpuFpReferenceBatchnormImpl<InputType, IntermediateType>::batchnormBwd(dyTensorCpu,
                                                                               xTensorCpu,
                                                                               meanTensorCpu,
                                                                               invVarianceTensorCpu,
                                                                               scaleTensorCpu,
                                                                               dxTensorCpu,
                                                                               dscaleTensorCpu,
                                                                               dbiasTensorCpu);

        CpuFpReferenceValidation<InputType> cpuRefValidationInput(tolerance, tolerance);
        CpuFpReferenceValidation<IntermediateType> cpuRefValidationIntermediate(
            static_cast<IntermediateType>(tolerance), static_cast<IntermediateType>(tolerance));

        EXPECT_TRUE(cpuRefValidationInput.allClose(dxTensorCpu, dxTensor));
        EXPECT_TRUE(cpuRefValidationIntermediate.allClose(dscaleTensorCpu, dscaleTensor));
        EXPECT_TRUE(cpuRefValidationIntermediate.allClose(dbiasTensorCpu, dbiasTensor));
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

class TestGpuMiopenBatchnormBwdExecuteGraphNchwFp32
    : public BatchnormBwdExecuteGraphBase<float, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphNchwFp32()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphNchwFp16
    : public BatchnormBwdExecuteGraphBase<half, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphNchwFp16()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphNchwBfp16
    : public BatchnormBwdExecuteGraphBase<hip_bfloat16, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphNchwBfp16()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphNchwFp64
    : public BatchnormBwdExecuteGraphBase<double, double>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphNchwFp64()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp32
    : public BatchnormBwdExecuteGraphBase<float, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp32()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp16
    : public BatchnormBwdExecuteGraphBase<half, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp16()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphNhwcBfp16
    : public BatchnormBwdExecuteGraphBase<hip_bfloat16, float>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphNhwcBfp16()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp64
    : public BatchnormBwdExecuteGraphBase<double, double>
{
public:
    TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp64()
        : BatchnormBwdExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphNchwFp32, Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::FLOAT,
                         batchnorm::getToleranceBackward<float>());
}

TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphNchwBfp16, Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::BFLOAT16,
                         batchnorm::getToleranceBackward<hip_bfloat16>());
}

TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphNchwFp16, Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::HALF,
                         batchnorm::getToleranceBackward<half>());
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphNchwFp64, DISABLED_Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::DOUBLE,
                         batchnorm::getToleranceBackward<double>());
}

TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp32, Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::FLOAT,
                         batchnorm::getToleranceBackward<float>());
}

// TODO: add unique test suite and conform to naming rules

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197
TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphNhwcBfp16, DISABLED_Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::BFLOAT16,
                         batchnorm::getToleranceBackward<hip_bfloat16>());
}

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197
TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp16, DISABLED_Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::HALF,
                         batchnorm::getToleranceBackward<half>());
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp64, DISABLED_Correctness)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::DOUBLE,
                         batchnorm::getToleranceBackward<double>());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphNchwFp32,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphNchwFp16,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphNchwBfp16,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphNchwFp64,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp32,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp16,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphNhwcBfp16,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormBwdExecuteGraphNhwcFp64,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
