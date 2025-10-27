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
class ConvBackwardData : public ::testing::TestWithParam<ConvTestCase>
{
    struct ConvTensorBundle
    {
        ConvTensorBundle(const ConvTestCase& testCase,
                         const TensorLayout& layout = TensorLayout::NCHW)
            : dxTensor(testCase.xDims, layout)
            , wTensor(testCase.wDims, layout)
            , dyTensor(testCase.yDims, layout)
        {
            dyTensor.fillWithRandomValues(
                static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed);
            wTensor.fillWithRandomValues(
                static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed);
            dxTensor.fillWithValue(static_cast<DataType>(0.0));
        }

        PinnedTensor<DataType> dxTensor;
        PinnedTensor<DataType> wTensor;
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

    std::unordered_map<int64_t, void*>
        createVariantPack(const graph::TensorAttributes& dxTensorAttr,
                          const graph::TensorAttributes& wTensorAttr,
                          const graph::TensorAttributes& dyTensorAttr,
                          ConvTensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[dxTensorAttr.get_uid()] = tensorBundle.dxTensor.memory().deviceData();
        variantPack[wTensorAttr.get_uid()] = tensorBundle.wTensor.memory().deviceData();
        variantPack[dyTensorAttr.get_uid()] = tensorBundle.dyTensor.memory().deviceData();

        return variantPack;
    }

    void runMiopenConvBwdData(const ConvTestCase& testCase,
                              ConvTensorBundle& graphTensorBundle,
                              hipdnn_frontend::DataType dataType)
    {
        auto graphObj = std::make_shared<hipdnn_frontend::graph::Graph>();

        graphObj->set_name("ConvolutionBackwardDataTest");

        int64_t uid = 1;

        auto dyAttr = graph::makeTensorAttributes("dy", dataType, graphTensorBundle.dyTensor);
        dyAttr.set_uid(uid++);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto wAttr = graph::makeTensorAttributes("w", dataType, graphTensorBundle.wTensor);
        wAttr.set_uid(uid++);
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvDgradAttributes convAttrs;
        convAttrs.set_name("convolution_backward_data");
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dxTensorAttr = graphObj->conv_dgrad(dyTensorAttr, wTensorAttr, convAttrs);

        if(!dxTensorAttr->has_uid())
        {
            dxTensorAttr->set_uid(uid++);
        }
        dxTensorAttr->set_dim(graphTensorBundle.dxTensor.dims());
        dxTensorAttr->set_stride(graphTensorBundle.dxTensor.strides());
        dxTensorAttr->set_output(true);
        dxTensorAttr->set_data_type(dataType);

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

        auto variantPack
            = createVariantPack(*dxTensorAttr, *wTensorAttr, *dyTensorAttr, graphTensorBundle);

        result = graphObj->execute(_handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    void runCpuConvBwdData(const ConvTestCase& testCase, ConvTensorBundle& cpuTensorBundle)
    {
        CpuFpReferenceConvolutionImpl<DataType, float>::convBwdData(cpuTensorBundle.dxTensor,
                                                                    cpuTensorBundle.wTensor,
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
        runMiopenConvBwdData(testCase, graphTensorBundle, dataType);
        graphTensorBundle.dxTensor.memory().markDeviceModified();

        runCpuConvBwdData(testCase, cpuTensorBundle);

        CpuFpReferenceValidation<DataType> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(
            cpuRefValidation.allClose(cpuTensorBundle.dxTensor, graphTensorBundle.dxTensor));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

using IntegrationGpuConvBwdDataNchwFp32 = ConvBackwardData<float>;
using IntegrationGpuConvBwdDataNcdhwFp32 = ConvBackwardData<float>;

using IntegrationGpuConvBwdDataNchwBfp16 = ConvBackwardData<hip_bfloat16>;
using IntegrationGpuConvBwdDataNcdhwBfp16 = ConvBackwardData<hip_bfloat16>;

using IntegrationGpuConvBwdDataNchwFp16 = ConvBackwardData<half>;
using IntegrationGpuConvBwdDataNcdhwFp16 = ConvBackwardData<half>;

using IntegrationGpuConvBwdDataNhwcFp32 = ConvBackwardData<float>;
using IntegrationGpuConvBwdDataNdhwcFp32 = ConvBackwardData<float>;

using IntegrationGpuConvBwdDataNhwcBfp16 = ConvBackwardData<hip_bfloat16>;
using IntegrationGpuConvBwdDataNdhwcBfp16 = ConvBackwardData<hip_bfloat16>;

using IntegrationGpuConvBwdDataNhwcFp16 = ConvBackwardData<half>;
using IntegrationGpuConvBwdDataNdhwcFp16 = ConvBackwardData<half>;

} // namespace

TEST_P(IntegrationGpuConvBwdDataNchwFp32, Correctness)
{
    runConvTest(4e-6f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvBwdDataNcdhwFp32, Correctness)
{
    runConvTest(conv::getToleranceBwd<float>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvBwdDataNchwBfp16, Correctness)
{
    runConvTest(conv::getToleranceBwd<hip_bfloat16>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvBwdDataNcdhwBfp16, Correctness)
{
    runConvTest(conv::getToleranceBwd<hip_bfloat16>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvBwdDataNchwFp16, Correctness)
{
    runConvTest(conv::getToleranceBwd<half>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvBwdDataNcdhwFp16, Correctness)
{
    runConvTest(conv::getToleranceBwd<half>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvBwdDataNhwcFp32, Correctness)
{
    runConvTest(conv::getToleranceBwd<float>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvBwdDataNdhwcFp32, Correctness)
{
    runConvTest(conv::getToleranceBwd<float>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvBwdDataNhwcBfp16, Correctness)
{
    runConvTest(conv::getToleranceBwd<hip_bfloat16>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvBwdDataNdhwcBfp16, Correctness)
{
    runConvTest(conv::getToleranceBwd<hip_bfloat16>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvBwdDataNhwcFp16, Correctness)
{
    runConvTest(conv::getToleranceBwd<half>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvBwdDataNdhwcFp16, Correctness)
{
    runConvTest(conv::getToleranceBwd<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNchwFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNchwBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNchwFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNhwcFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNhwcBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNhwcFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNcdhwFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNcdhwBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNcdhwFp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNdhwcFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNdhwcBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvBwdDataNdhwcFp16,
                         testing::ValuesIn(getConvTestCases5D()));
