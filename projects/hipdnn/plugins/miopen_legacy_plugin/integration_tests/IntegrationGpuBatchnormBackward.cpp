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
class BatchnormBackward : public ::testing::TestWithParam<TestCaseType>
{

    struct TensorBundle
    {
        TensorBundle(const std::vector<int64_t>& dims,
                     unsigned int seed = hipdnn_sdk::test_utilities::getGlobalTestSeed(),
                     const TensorLayout& layout = TensorLayout::NCHW)
            : derivedDims(getDerivedShape(dims))
            , xTensor(dims, layout)
            , dyTensor(dims, layout)
            , dxTensor(dims, layout)
            , scaleTensor(derivedDims)
            , dscaleTensor(derivedDims)
            , dbiasTensor(derivedDims)
            , meanTensor(derivedDims)
            , invVarianceTensor(derivedDims)
        {
            xTensor.fillWithRandomValues(
                static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed);

            dyTensor.fillWithRandomValues(
                static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed);
            scaleTensor.fillWithRandomValues(
                static_cast<IntermediateType>(-0.1f), static_cast<IntermediateType>(0.1f), seed);

            meanTensor.fillWithRandomValues(
                static_cast<IntermediateType>(-0.1f), static_cast<IntermediateType>(0.1f), seed);

            invVarianceTensor.fillWithRandomValues(
                static_cast<IntermediateType>(1.9f), static_cast<IntermediateType>(2.0f), seed);
        }

        std::vector<int64_t> derivedDims;
        PinnedTensor<InputType> xTensor;
        PinnedTensor<InputType> dyTensor;
        PinnedTensor<InputType> dxTensor;
        PinnedTensor<IntermediateType> scaleTensor;
        PinnedTensor<IntermediateType> dscaleTensor;
        PinnedTensor<IntermediateType> dbiasTensor;
        PinnedTensor<IntermediateType> meanTensor;
        PinnedTensor<IntermediateType> invVarianceTensor;
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
                          const graph::TensorAttributes& dyTensorAttr,
                          const graph::TensorAttributes& dxTensorAttr,
                          const graph::TensorAttributes& scaleTensorAttr,
                          const graph::TensorAttributes& dscaleTensorAttr,
                          const graph::TensorAttributes& dbiasTensorAttr,
                          const graph::TensorAttributes& meanTensorAttr,
                          const graph::TensorAttributes& invVarianceTensorAttr,
                          TensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[dyTensorAttr.get_uid()] = tensorBundle.dyTensor.memory().deviceData();
        variantPack[dxTensorAttr.get_uid()] = tensorBundle.dxTensor.memory().deviceData();
        variantPack[scaleTensorAttr.get_uid()] = tensorBundle.scaleTensor.memory().deviceData();
        variantPack[dscaleTensorAttr.get_uid()] = tensorBundle.dscaleTensor.memory().deviceData();
        variantPack[dbiasTensorAttr.get_uid()] = tensorBundle.dbiasTensor.memory().deviceData();
        variantPack[meanTensorAttr.get_uid()] = tensorBundle.meanTensor.memory().deviceData();
        variantPack[invVarianceTensorAttr.get_uid()]
            = tensorBundle.invVarianceTensor.memory().deviceData();

        return variantPack;
    }

    void runMiopenBatchnormBwd(TensorBundle& graphTensorBundle,
                               hipdnn_frontend::DataType inputDataType,
                               hipdnn_frontend::DataType intermediateDataType)
    {
        auto graphObj = std::make_shared<hipdnn_frontend::graph::Graph>();

        graphObj->set_name("BatchnormBackwardTest");

        int64_t uid = 1;

        auto xAttr = graph::makeTensorAttributes("x", inputDataType, graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr = graph::makeTensorAttributes("dy", inputDataType, graphTensorBundle.dyTensor);
        dyAttr.set_uid(uid++);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, graphTensorBundle.scaleTensor);
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, graphTensorBundle.meanTensor);
        meanAttr.set_uid(uid++);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = graph::makeTensorAttributes(
            "inv_variance", intermediateDataType, graphTensorBundle.invVarianceTensor);
        invVarianceAttr.set_uid(uid++);
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        graph::BatchnormBackwardAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_backward");
        bnAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);

        auto outputTensorsAttr
            = graphObj->batchnorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, bnAttrs);

        auto& dxTensorAttr = outputTensorsAttr[0];
        if(!dxTensorAttr->has_uid())
        {
            dxTensorAttr->set_uid(uid++);
        }
        dxTensorAttr->set_data_type(inputDataType);
        dxTensorAttr->set_dim(graphTensorBundle.dxTensor.dims());
        dxTensorAttr->set_stride(graphTensorBundle.dxTensor.strides());

        auto& dscaleTensorAttr = outputTensorsAttr[1];
        if(!dscaleTensorAttr->has_uid())
        {
            dscaleTensorAttr->set_uid(uid++);
        }
        dscaleTensorAttr->set_data_type(intermediateDataType);

        auto& dbiasTensorAttr = outputTensorsAttr[2];
        if(!dbiasTensorAttr->has_uid())
        {
            dbiasTensorAttr->set_uid(uid++);
        }
        dbiasTensorAttr->set_data_type(intermediateDataType);

        auto result = graphObj->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graphObj->build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graphObj->create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graphObj->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graphObj->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto variantPack = createVariantPack(*xTensorAttr,
                                             *dyTensorAttr,
                                             *dxTensorAttr,
                                             *scaleTensorAttr,
                                             *dscaleTensorAttr,
                                             *dbiasTensorAttr,
                                             *meanTensorAttr,
                                             *invVarianceTensorAttr,
                                             graphTensorBundle);

        result = graphObj->execute(_handle, variantPack, nullptr);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    void runCpuBatchnormBwd(TensorBundle& cpuTensorBundle)
    {
        CpuFpReferenceBatchnormImpl<InputType, IntermediateType>::batchnormBwd(
            cpuTensorBundle.dyTensor,
            cpuTensorBundle.xTensor,
            cpuTensorBundle.meanTensor,
            cpuTensorBundle.invVarianceTensor,
            cpuTensorBundle.scaleTensor,
            cpuTensorBundle.dxTensor,
            cpuTensorBundle.dscaleTensor,
            cpuTensorBundle.dbiasTensor);
    }

    void runBatchnormTest(InputType tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        TestCaseType testCase = this->GetParam();

        auto inputDataType = getDataTypeEnumFromType<InputType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        unsigned int seed = getGlobalTestSeed();
        HIPDNN_LOG_INFO("Test is using {} for its random seed", seed);

        TensorBundle graphTensorBundle(testCase.getDims(), seed, layout);

        TensorBundle cpuTensorBundle(testCase.getDims(), seed, layout);

        runMiopenBatchnormBwd(graphTensorBundle, inputDataType, intermediateDataType);
        graphTensorBundle.dxTensor.memory().markDeviceModified();
        graphTensorBundle.dscaleTensor.memory().markDeviceModified();
        graphTensorBundle.dbiasTensor.memory().markDeviceModified();

        runCpuBatchnormBwd(cpuTensorBundle);

        CpuFpReferenceValidation<InputType> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(
            cpuRefValidation.allClose(cpuTensorBundle.dxTensor, graphTensorBundle.dxTensor));

        CpuFpReferenceValidation<IntermediateType> cpuRefIntermediateValidation(
            static_cast<IntermediateType>(tolerance), static_cast<IntermediateType>(tolerance));
        EXPECT_TRUE(cpuRefIntermediateValidation.allClose(cpuTensorBundle.dscaleTensor,
                                                          graphTensorBundle.dscaleTensor));
        EXPECT_TRUE(cpuRefIntermediateValidation.allClose(cpuTensorBundle.dbiasTensor,
                                                          graphTensorBundle.dbiasTensor));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

class IntegrationGpuBatchnormBackwardNchwFp32
    : public BatchnormBackward<float, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNchwBfp16
    : public BatchnormBackward<hip_bfloat16, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNchwFp16
    : public BatchnormBackward<half, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNhwcFp32
    : public BatchnormBackward<float, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNhwcBfp16
    : public BatchnormBackward<hip_bfloat16, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNhwcFp16
    : public BatchnormBackward<half, float, Batchnorm2dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNcdhwFp32
    : public BatchnormBackward<float, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNcdhwBfp16
    : public BatchnormBackward<hip_bfloat16, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNcdhwFp16
    : public BatchnormBackward<half, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNdhwcFp32
    : public BatchnormBackward<float, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNdhwcBfp16
    : public BatchnormBackward<hip_bfloat16, float, Batchnorm3dTestCase>
{
};

class IntegrationGpuBatchnormBackwardNdhwcFp16
    : public BatchnormBackward<half, float, Batchnorm3dTestCase>
{
};

std::vector<Batchnorm2dTestCase> getBnBwdTestCases()
{
    unsigned seed = getGlobalTestSeed();

    return std::vector<Batchnorm2dTestCase>{
        {1, 3, 14, 14, seed},
        // MIOpen segfaults for this case, re-enable when fix is released:
        // https://github.com/ROCm/rocm-libraries/pull/1197
        // {1, 256, 1, 1, seed}, // Would produce near-zero variance in theory
        {2, 3, 1, 1, seed},
        {32, 1, 14, 14, seed},
        {32, 3, 1, 14, seed},
        {32, 3, 14, 1, seed},
        {64, 64, 112, 112, seed},
        {64, 512, 14, 14, seed},
    };
}

std::vector<Batchnorm3dTestCase> getBnBwd3dTestCases()
{
    unsigned seed = getGlobalTestSeed();

    return std::vector<Batchnorm3dTestCase>{
        {2, 3, 3, 1, 1, seed},
        {16, 3, 8, 14, 14, seed},
    };
}

} // namespace

TEST_P(IntegrationGpuBatchnormBackwardNchwFp32, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNchwFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNchwBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwFp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNchwFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcFp32, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNhwcFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197}
TEST_P(IntegrationGpuBatchnormBackwardNhwcBfp16, DISABLED_Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNhwcBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197
TEST_P(IntegrationGpuBatchnormBackwardNhwcFp16, DISABLED_Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNhwcFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp32, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNcdhwFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwBfp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNcdhwBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp16, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNcdhwFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp32, Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNdhwcFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

// MIOpen may have issues with NDHWC layout for certain data types
TEST_P(IntegrationGpuBatchnormBackwardNdhwcBfp16, DISABLED_Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNdhwcBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

// MIOpen may have issues with NDHWC layout for certain data types
TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp16, DISABLED_Correctness)
{
    runBatchnormTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNdhwcFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));
