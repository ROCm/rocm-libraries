// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FileUtilities.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Constants.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/BatchnormCommon.hpp"
#include "common/GoldenReferenceGpu.hpp"
#include "common/Helpers.hpp"

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace test_bn_common;
using namespace test_helpers;

template <class T>
class TestBatchnormFwdInferenceGoldenReference : public TestGoldenReferenceGpu
{
public:
    void testSuite()
    {
        return goldenReferenceTestSuite(batchnorm::getToleranceInference<T>(),
                                        batchnorm::getToleranceInference<T>());
    }
};

class TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNchwFp32
    : public TestBatchnormFwdInferenceGoldenReference<float>
{
};

class TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNchwFp16
    : public TestBatchnormFwdInferenceGoldenReference<half>
{
};

class TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNchwBfp16
    : public TestBatchnormFwdInferenceGoldenReference<hip_bfloat16>
{
};

class TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNcdhwFp32
    : public TestBatchnormFwdInferenceGoldenReference<float>
{
};

// Nchw Fp32------------
TEST_P(TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNchwFp32, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNchwFp32,
                         getGoldenReferenceParams("BatchnormFwdInference/nchw/fp32"));

// Nchw Fp16------------
TEST_P(TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNchwFp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNchwFp16,
                         getGoldenReferenceParams("BatchnormFwdInference/nchw/fp16"));

// Nchw Bfp16------------
TEST_P(TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNchwBfp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNchwBfp16,
                         getGoldenReferenceParams("BatchnormFwdInference/nchw/bfp16"));

// Ncdhw Fp32------------
TEST_P(TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNcdhwFp32, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMIOpenBatchnormFwdInferenceGoldenReferenceNcdhwFp32,
                         getGoldenReferenceParams("BatchnormFwdInference/ncdhw/fp32"));

//--------------------------

template <typename InputType, typename IntermediateType>
class BatchnormFwdInferenceExecuteGraphBase : public ::testing::TestWithParam<Batchnorm2dTestCase>
{
protected:
    TensorLayout _layout;

    BatchnormFwdInferenceExecuteGraphBase(TensorLayout layout)
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

    void runFwdBatchnormGraph(Batchnorm2dTestCase testCase,
                              hipdnn_sdk::data_objects::DataType inputDataType,
                              InputType tolerance)
    {
        auto dims = std::vector<int64_t>{testCase.n, testCase.c, testCase.h, testCase.w};

        auto derivedDims = getDerivedShape(dims);

        auto deviceBuffers = std::vector<hipdnnPluginDeviceBuffer_t>{};

        PinnedTensor<InputType> xTensor(dims, _layout);
        deviceBuffers.push_back(generateRandomDeviceBuffer(
            xTensor, 1, static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), testCase.seed));

        PinnedTensor<InputType> yTensor(dims, _layout);
        deviceBuffers.push_back(generateEmptyDeviceBuffer(yTensor, 2));

        PinnedTensor<IntermediateType> scaleTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(scaleTensor,
                                                           3,
                                                           static_cast<IntermediateType>(0.0f),
                                                           static_cast<IntermediateType>(1.0f),
                                                           testCase.seed));

        PinnedTensor<IntermediateType> biasTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(biasTensor,
                                                           4,
                                                           static_cast<IntermediateType>(0.0f),
                                                           static_cast<IntermediateType>(1.0f),
                                                           testCase.seed));

        PinnedTensor<IntermediateType> meanTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(meanTensor,
                                                           5,
                                                           static_cast<IntermediateType>(0.0f),
                                                           static_cast<IntermediateType>(1.0f),
                                                           testCase.seed));

        PinnedTensor<IntermediateType> varianceTensor(derivedDims);
        deviceBuffers.push_back(generateRandomDeviceBuffer(varianceTensor,
                                                           6,
                                                           static_cast<IntermediateType>(0.1f),
                                                           static_cast<IntermediateType>(1.0f),
                                                           testCase.seed));

        auto batchnormBuilder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph(
            xTensor.strides(), xTensor.dims(), inputDataType);

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

        yTensor.memory().markDeviceModified();

        status = hipdnnEnginePluginDestroyExecutionContext(_handle, executionContext);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

        Tensor<InputType> xTensorCpu(dims, _layout);
        xTensorCpu.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), testCase.seed);
        Tensor<InputType> yTensorCpu(dims, _layout);
        Tensor<IntermediateType> scaleTensorCpu(derivedDims);
        scaleTensorCpu.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                            static_cast<IntermediateType>(1.0f),
                                            testCase.seed);
        Tensor<IntermediateType> biasTensorCpu(derivedDims);
        biasTensorCpu.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                           static_cast<IntermediateType>(1.0f),
                                           testCase.seed);
        Tensor<IntermediateType> meanTensorCpu(derivedDims);
        meanTensorCpu.fillWithRandomValues(static_cast<IntermediateType>(0.0f),
                                           static_cast<IntermediateType>(1.0f),
                                           testCase.seed);
        Tensor<IntermediateType> varianceTensorCpu(derivedDims);
        varianceTensorCpu.fillWithRandomValues(static_cast<IntermediateType>(0.1f),
                                               static_cast<IntermediateType>(1.0f),
                                               testCase.seed);

        CpuFpReferenceBatchnormImpl<InputType, IntermediateType>::batchnormFwdInference(
            xTensorCpu,
            scaleTensorCpu,
            biasTensorCpu,
            meanTensorCpu,
            varianceTensorCpu,
            yTensorCpu,
            BATCHNORM_DEFAULT_EPSILON);

        CpuFpReferenceValidation<InputType> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(cpuRefValidation.allClose(yTensorCpu, yTensor));
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp32
    : public BatchnormFwdInferenceExecuteGraphBase<float, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp32()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp16
    : public BatchnormFwdInferenceExecuteGraphBase<half, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp16()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwBfp16
    : public BatchnormFwdInferenceExecuteGraphBase<hip_bfloat16, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwBfp16()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp64
    : public BatchnormFwdInferenceExecuteGraphBase<double, double>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp64()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NCHW)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp32
    : public BatchnormFwdInferenceExecuteGraphBase<float, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp32()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp16
    : public BatchnormFwdInferenceExecuteGraphBase<half, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp16()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcBfp16
    : public BatchnormFwdInferenceExecuteGraphBase<hip_bfloat16, float>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcBfp16()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

class TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp64
    : public BatchnormFwdInferenceExecuteGraphBase<double, double>
{
public:
    TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp64()
        : BatchnormFwdInferenceExecuteGraphBase(TensorLayout::NHWC)
    {
    }
};

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp32, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::FLOAT,
                         batchnorm::getToleranceInference<float>());
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwBfp16, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::BFLOAT16,
                         batchnorm::getToleranceInference<hip_bfloat16>());
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp16, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::HALF,
                         batchnorm::getToleranceInference<half>());
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp64, DISABLED_Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::DOUBLE,
                         batchnorm::getToleranceInference<double>());
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp32, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::FLOAT,
                         batchnorm::getToleranceInference<float>());
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcBfp16, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::BFLOAT16,
                         batchnorm::getToleranceInference<hip_bfloat16>());
}

TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp16, Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::HALF,
                         batchnorm::getToleranceInference<half>());
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp64, DISABLED_Correctness)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph(testCase,
                         hipdnn_sdk::data_objects::DataType::DOUBLE,
                         batchnorm::getToleranceInference<double>());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp32,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp16,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwBfp16,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphNchwFp64,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp32,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp16,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcBfp16,
                         testing::ValuesIn(getBatchnorm2dTestCases()));

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuMiopenBatchnormFwdInferenceExecuteGraphNhwcFp64,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
