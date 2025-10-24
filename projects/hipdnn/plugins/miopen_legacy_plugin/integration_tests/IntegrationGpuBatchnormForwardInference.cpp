// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Constants.hpp>
#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

namespace
{

struct Batchnorm2dTestCase
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm2dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w
                  << " seed:" << tc.seed << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, h, w};
    }
};

struct Batchnorm3dTestCase
{
    int64_t n;
    int64_t c;
    int64_t d;
    int64_t h;
    int64_t w;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& ss, const Batchnorm3dTestCase& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " d:" << tc.d << " h:" << tc.h
                  << " w:" << tc.w << " seed:" << tc.seed << ")";
    }

    std::vector<int64_t> getDims() const
    {
        return {n, c, d, h, w};
    }
};

template <typename InputType, typename IntermediateType, typename TestCaseType>
class BatchnormForwardInference : public ::testing::TestWithParam<TestCaseType>
{
    struct TensorBundle
    {
        TensorBundle(const std::vector<int64_t>& dims,
                     unsigned int seed = getGlobalTestSeed(),
                     const TensorLayout& layout = TensorLayout::NCHW)
            : derivedDims(getDerivedShape(dims))
            , xTensor(dims, layout)
            , yTensor(dims, layout)
            , scaleTensor(derivedDims)
            , biasTensor(derivedDims)
            , meanTensor(derivedDims)
            , varianceTensor(derivedDims)
        {
            xTensor.fillWithRandomValues(
                static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
            yTensor.fillWithRandomValues(
                static_cast<InputType>(-100.0f), static_cast<InputType>(100.0f), seed);

            scaleTensor.fillWithRandomValues(
                static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

            biasTensor.fillWithRandomValues(
                static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

            meanTensor.fillWithRandomValues(
                static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);

            varianceTensor.fillWithRandomValues(
                static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f), seed);
        }

        std::vector<int64_t> derivedDims;
        PinnedTensor<InputType> xTensor;
        PinnedTensor<InputType> yTensor;
        PinnedTensor<IntermediateType> scaleTensor;
        PinnedTensor<IntermediateType> biasTensor;
        PinnedTensor<IntermediateType> meanTensor;
        PinnedTensor<IntermediateType> varianceTensor;
    };

protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Uncomment if you want debug logging info.
        // setenv("HIPDNN_LOG_LEVEL", "info", 1);

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        // Note: The plugin paths has to be set before we create the hipdnn handle.
        auto pluginPath
            = std::filesystem::weakly_canonical(getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        // Create handle and stream
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const graph::TensorAttributes& xTensorAttr,
                          const graph::TensorAttributes& yTensorAttr,
                          const graph::TensorAttributes& meanTensorAttr,
                          const graph::TensorAttributes& invVarianceTensorAttr,
                          const graph::TensorAttributes& scaleTensorAttr,
                          const graph::TensorAttributes& biasTensorAttr,
                          TensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[meanTensorAttr.get_uid()] = tensorBundle.meanTensor.memory().deviceData();
        variantPack[invVarianceTensorAttr.get_uid()]
            = tensorBundle.varianceTensor.memory().deviceData();
        variantPack[scaleTensorAttr.get_uid()] = tensorBundle.scaleTensor.memory().deviceData();
        variantPack[biasTensorAttr.get_uid()] = tensorBundle.biasTensor.memory().deviceData();
        variantPack[yTensorAttr.get_uid()] = tensorBundle.yTensor.memory().deviceData();

        return variantPack;
    }

    void runMiopenBatchnormFwd(TensorBundle& graphTensorBundle,
                               hipdnn_frontend::DataType inputDataType,
                               hipdnn_frontend::DataType intermediateDataType)
    {
        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();

        graph->set_name("BatchnormInferenceTest");

        int64_t uid = 1;
        auto xAttr = graph::makeTensorAttributes("X", inputDataType, graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, graphTensorBundle.meanTensor);
        meanAttr.set_uid(uid++);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = graph::makeTensorAttributes(
            "inv_variance", intermediateDataType, graphTensorBundle.varianceTensor);
        invVarianceAttr.set_uid(uid++);
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, graphTensorBundle.scaleTensor);
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, graphTensorBundle.biasTensor);
        biasAttr.set_uid(uid++);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::BatchnormInferenceAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_inference");

        auto yTensorAttr = graph->batchnorm_inference(xTensorAttr,
                                                      meanTensorAttr,
                                                      invVarianceTensorAttr,
                                                      scaleTensorAttr,
                                                      biasTensorAttr,
                                                      bnAttrs);

        if(!yTensorAttr->has_uid())
        {
            yTensorAttr->set_uid(uid++);
        }

        yTensorAttr->set_data_type(inputDataType);
        yTensorAttr->set_dim(graphTensorBundle.yTensor.dims());
        yTensorAttr->set_stride(graphTensorBundle.yTensor.strides());

        // Validate and build graph
        auto result = graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto variantPack = createVariantPack(*xTensorAttr,
                                             *yTensorAttr,
                                             *meanTensorAttr,
                                             *invVarianceTensorAttr,
                                             *scaleTensorAttr,
                                             *biasTensorAttr,
                                             graphTensorBundle);

        result = graph->execute(_handle, variantPack, nullptr);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    void runCpuBatchnormFwd(TensorBundle& cpuTensorBundle)
    {
        CpuFpReferenceBatchnormImpl<InputType, IntermediateType>::batchnormFwdInference(
            cpuTensorBundle.xTensor,
            cpuTensorBundle.scaleTensor,
            cpuTensorBundle.biasTensor,
            cpuTensorBundle.meanTensor,
            cpuTensorBundle.varianceTensor,
            cpuTensorBundle.yTensor,
            BATCHNORM_DEFAULT_EPSILON);
    }

    void runBatchnormTest(InputType tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        TestCaseType testCase = this->GetParam();

        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", testCase.seed);

        TensorBundle graphTensorBundle(testCase.getDims(), testCase.seed, layout);

        TensorBundle cpuTensorBundle(testCase.getDims(), testCase.seed, layout);

        runMiopenBatchnormFwd(graphTensorBundle, inputDataType, intermediateDataType);
        graphTensorBundle.yTensor.memory().markDeviceModified();

        runCpuBatchnormFwd(cpuTensorBundle);

        CpuFpReferenceValidation<InputType> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(cpuRefValidation.allClose(cpuTensorBundle.yTensor, graphTensorBundle.yTensor));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

class IntegrationGpuBatchnormForwardInferenceNchwFp32
    : public BatchnormForwardInference<float, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNchwBfp16
    : public BatchnormForwardInference<hip_bfloat16, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNchwFp16
    : public BatchnormForwardInference<half, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNhwcFp32
    : public BatchnormForwardInference<float, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNhwcBfp16
    : public BatchnormForwardInference<hip_bfloat16, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNhwcFp16
    : public BatchnormForwardInference<half, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNcdhwFp32
    : public BatchnormForwardInference<float, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNcdhwBfp16
    : public BatchnormForwardInference<hip_bfloat16, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNcdhwFp16
    : public BatchnormForwardInference<half, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNdhwcFp32
    : public BatchnormForwardInference<float, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNdhwcBfp16
    : public BatchnormForwardInference<hip_bfloat16, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormForwardInferenceNdhwcFp16
    : public BatchnormForwardInference<half, float, Batchnorm3dTestCase>
{
};

std::vector<Batchnorm2dTestCase> getBnFwdInferenceTestCases()
{
    unsigned seed = getGlobalTestSeed();

    return {
        {1, 3, 14, 14, seed},
        {1, 256, 1, 1, seed},
        {2, 3, 1, 1, seed},
        {32, 1, 14, 14, seed},
        {32, 3, 1, 14, seed},
        {32, 3, 14, 1, seed},
        {64, 64, 112, 112, seed},
        {64, 512, 14, 14, seed},
    };
}

std::vector<Batchnorm3dTestCase> getBnFwdInference3dTestCases()
{
    unsigned seed = getGlobalTestSeed();

    return {
        {2, 3, 3, 1, 1, seed},
        {16, 3, 8, 14, 14, seed},
    };
}

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp32, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNchwFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp32, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNhwcBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp32, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNcdhwBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp32, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNdhwcBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceInference<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));
