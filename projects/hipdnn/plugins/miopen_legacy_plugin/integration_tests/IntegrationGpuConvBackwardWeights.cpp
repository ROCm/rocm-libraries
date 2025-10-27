// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <random>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/StringUtil.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/Workspace.hpp>

#include "../tests/common/ConvolutionCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace test_conv_common;

namespace
{

template <typename DataType>
class ConvBackwardWeights : public ::testing::TestWithParam<ConvTestCase>
{
    struct ConvTensorBundle
    {
        ConvTensorBundle(const ConvTestCase& testCase,
                         const TensorLayout& layout = TensorLayout::NCHW)
            : xTensor(testCase.xDims, layout)
            , dwTensor(testCase.wDims, layout)
            , dyTensor(testCase.yDims, layout)
        {
            dyTensor.fillWithRandomValues(
                static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed);
            xTensor.fillWithRandomValues(
                static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed);
            dwTensor.fillWithValue(static_cast<DataType>(0.0));
        }

        std::unordered_map<int64_t, void*>
            createVariantPack(int64_t xTensorUid, int64_t dwTensorUid, int64_t dyTensorUid)
        {
            std::unordered_map<int64_t, void*> variantPack;
            variantPack[xTensorUid] = xTensor.memory().deviceData();
            variantPack[dwTensorUid] = dwTensor.memory().deviceData();
            variantPack[dyTensorUid] = dyTensor.memory().deviceData();

            return variantPack;
        }

        PinnedTensor<DataType> xTensor;
        PinnedTensor<DataType> dwTensor;
        PinnedTensor<DataType> dyTensor;
    };

protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        // Note: The plugin paths has to be set before we create the hipdnn handle.
        const std::array<const char*, 1> paths = {PLUGIN_PATH};
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

    void runMiopenConvWrwData(const ConvTestCase& testCase,
                              ConvTensorBundle& graphTensorBundle,
                              hipdnn_frontend::DataType dataType)
    {
        auto graphObj = std::make_shared<hipdnn_frontend::graph::Graph>();

        graphObj->set_name("ConvolutionBackwardWeightTest");

        int64_t uid = 1;

        auto xAttr = graph::makeTensorAttributes("x", dataType, graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr = graph::makeTensorAttributes("dy", dataType, graphTensorBundle.dyTensor);
        dyAttr.set_uid(uid++);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        graph::ConvWgradAttributes convAttrs;
        convAttrs.set_name("convolution_backward_weights");
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dwTensorAttr = graphObj->conv_wgrad(dyTensorAttr, xTensorAttr, convAttrs);

        if(!dwTensorAttr->has_uid())
        {
            dwTensorAttr->set_uid(uid++);
        }
        dwTensorAttr->set_dim(graphTensorBundle.dwTensor.dims());
        dwTensorAttr->set_stride(graphTensorBundle.dwTensor.strides());
        dwTensorAttr->set_output(true);
        dwTensorAttr->set_data_type(dataType);

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

        int64_t workspaceSize;
        result = graphObj->get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0) << result.err_msg;
        Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack = graphTensorBundle.createVariantPack(
            xTensorAttr->get_uid(), dwTensorAttr->get_uid(), dyTensorAttr->get_uid());

        result = graphObj->execute(_handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    void runCpuConvWrwData(const ConvTestCase& testCase, ConvTensorBundle& cpuTensorBundle)
    {
        CpuFpReferenceConvolutionImpl<DataType, float>::convBwdWeight(cpuTensorBundle.xTensor,
                                                                      cpuTensorBundle.dwTensor,
                                                                      cpuTensorBundle.dyTensor,
                                                                      testCase.convStride,
                                                                      testCase.convDilation,
                                                                      testCase.convPrePadding,
                                                                      testCase.convPostPadding);
    }

    void runConvTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        const ConvTestCase& testCase = GetParam();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", testCase.seed);

        ConvTensorBundle graphTensorBundle(testCase, layout);
        ConvTensorBundle cpuTensorBundle(testCase, layout);

        auto dataType = getDataTypeEnumFromType<DataType>();
        runMiopenConvWrwData(testCase, graphTensorBundle, dataType);
        graphTensorBundle.dwTensor.memory().markDeviceModified();

        runCpuConvWrwData(testCase, cpuTensorBundle);

        CpuFpReferenceValidation<DataType> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(
            cpuRefValidation.allClose(cpuTensorBundle.dwTensor, graphTensorBundle.dwTensor));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

using IntegrationGpuConvWrwDataNchwFp32 = ConvBackwardWeights<float>;
using IntegrationGpuConvWrwDataNcdhwFp32 = ConvBackwardWeights<float>;

using IntegrationGpuConvWrwDataNchwBfp16 = ConvBackwardWeights<hip_bfloat16>;
using IntegrationGpuConvWrwDataNcdhwBfp16 = ConvBackwardWeights<hip_bfloat16>;

using IntegrationGpuConvWrwDataNchwFp16 = ConvBackwardWeights<half>;
using IntegrationGpuConvWrwDataNcdhwFp16 = ConvBackwardWeights<half>;

using IntegrationGpuConvWrwDataNhwcFp32 = ConvBackwardWeights<float>;
using IntegrationGpuConvWrwDataNdhwcFp32 = ConvBackwardWeights<float>;

using IntegrationGpuConvWrwDataNhwcBfp16 = ConvBackwardWeights<hip_bfloat16>;
using IntegrationGpuConvWrwDataNdhwcBfp16 = ConvBackwardWeights<hip_bfloat16>;

using IntegrationGpuConvWrwDataNhwcFp16 = ConvBackwardWeights<half>;
using IntegrationGpuConvWrwDataNdhwcFp16 = ConvBackwardWeights<half>;

} // namespace

TEST_P(IntegrationGpuConvWrwDataNchwFp32, Correctness)
{
    runConvTest(conv::getToleranceWrw<float>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvWrwDataNcdhwFp32, Correctness)
{
    runConvTest(conv::getToleranceWrw<float>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvWrwDataNchwBfp16, Correctness)
{
    runConvTest(conv::getToleranceWrw<hip_bfloat16>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvWrwDataNcdhwBfp16, Correctness)
{
    runConvTest(conv::getToleranceWrw<hip_bfloat16>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvWrwDataNchwFp16, Correctness)
{
    runConvTest(conv::getToleranceWrw<half>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvWrwDataNcdhwFp16, Correctness)
{
    runConvTest(conv::getToleranceWrw<half>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvWrwDataNhwcFp32, Correctness)
{
    runConvTest(conv::getToleranceWrw<float>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvWrwDataNdhwcFp32, Correctness)
{
    runConvTest(conv::getToleranceWrw<float>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvWrwDataNhwcBfp16, Correctness)
{
    runConvTest(conv::getToleranceWrw<hip_bfloat16>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvWrwDataNdhwcBfp16, Correctness)
{
    runConvTest(conv::getToleranceWrw<hip_bfloat16>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvWrwDataNhwcFp16, Correctness)
{
    runConvTest(conv::getToleranceWrw<half>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvWrwDataNdhwcFp16, Correctness)
{
    runConvTest(conv::getToleranceWrw<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNchwFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNchwBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNchwFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNhwcFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNhwcBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNhwcFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNcdhwFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNcdhwBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNcdhwFp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNdhwcFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNdhwcBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvWrwDataNdhwcFp16,
                         testing::ValuesIn(getConvTestCases5D()));
