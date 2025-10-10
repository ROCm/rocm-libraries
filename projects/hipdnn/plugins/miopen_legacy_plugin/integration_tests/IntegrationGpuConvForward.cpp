// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
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
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
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
class ConvForward : public ::testing::TestWithParam<ConvTestCase>
{
    struct ConvTensorBundle
    {
        ConvTensorBundle(const ConvTestCase& testCase,
                         const TensorLayout& layout = TensorLayout::NCHW)
            : xTensor(testCase.xDims, layout)
            , wTensor(testCase.wDims, layout)
            , yTensor(testCase.yDims, layout)
        {
            xTensor.fillWithRandomValues(
                static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed);
            wTensor.fillWithRandomValues(
                static_cast<DataType>(-1.0f), static_cast<DataType>(1.0f), testCase.seed);
            yTensor.fillWithValue(static_cast<DataType>(0.0));
        }

        PinnedTensor<DataType> xTensor;
        PinnedTensor<DataType> wTensor;
        PinnedTensor<DataType> yTensor;
    };

protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

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

    std::unordered_map<int64_t, void*> createVariantPack(const graph::TensorAttributes& xTensorAttr,
                                                         const graph::TensorAttributes& wTensorAttr,
                                                         const graph::TensorAttributes& yTensorAttr,
                                                         ConvTensorBundle& tensorBundle)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = tensorBundle.xTensor.memory().deviceData();
        variantPack[wTensorAttr.get_uid()] = tensorBundle.wTensor.memory().deviceData();
        variantPack[yTensorAttr.get_uid()] = tensorBundle.yTensor.memory().deviceData();

        return variantPack;
    }

    void runMiopenConvFwd(const ConvTestCase& testCase,
                          ConvTensorBundle& graphTensorBundle,
                          hipdnn_frontend::DataType dataType)
    {
        auto graphObj = std::make_shared<hipdnn_frontend::graph::Graph>();

        graphObj->set_name("ConvolutionForwardTest");

        int64_t uid = 1;

        auto xAttr = graph::makeTensorAttributes("x", dataType, graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr = graph::makeTensorAttributes("w", dataType, graphTensorBundle.wTensor);
        wAttr.set_uid(uid++);
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_name("convolution_forward");
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto yTensorAttr = graphObj->conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

        if(!yTensorAttr->has_uid())
        {
            yTensorAttr->set_uid(uid++);
        }
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(dataType);

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
            = createVariantPack(*xTensorAttr, *wTensorAttr, *yTensorAttr, graphTensorBundle);

        result = graphObj->execute(_handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }

    void runCpuConvFwd(const ConvTestCase& testCase, ConvTensorBundle& cpuTensorBundle)
    {
        CpuFpReferenceConvolutionImpl<DataType, float>::convFwdInference(cpuTensorBundle.xTensor,
                                                                         cpuTensorBundle.wTensor,
                                                                         cpuTensorBundle.yTensor,
                                                                         testCase.convStride,
                                                                         testCase.convDilation,
                                                                         testCase.convPrePadding);
    }

    void runConvTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        const ConvTestCase& testCase = GetParam();

        HIPDNN_LOG_INFO("Test is using {} for its random seed", testCase.seed);

        ConvTensorBundle graphTensorBundle(testCase, layout);
        ConvTensorBundle cpuTensorBundle(testCase, layout);

        auto dataType = getDataTypeEnumFromType<DataType>();
        runMiopenConvFwd(testCase, graphTensorBundle, dataType);
        graphTensorBundle.yTensor.memory().markDeviceModified();

        runCpuConvFwd(testCase, cpuTensorBundle);

        CpuFpReferenceValidation<DataType> cpuRefValidation(tolerance, tolerance);
        EXPECT_TRUE(cpuRefValidation.allClose(cpuTensorBundle.yTensor.memory(),
                                              graphTensorBundle.yTensor.memory()));
    }

private:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

using IntegrationGpuConvFwdNchwFp32 = ConvForward<float>;
using IntegrationGpuConvFwdNcdhwFp32 = ConvForward<float>;

using IntegrationGpuConvFwdNchwBfp16 = ConvForward<hip_bfloat16>;
using IntegrationGpuConvFwdNcdhwBfp16 = ConvForward<hip_bfloat16>;

using IntegrationGpuConvFwdNchwFp16 = ConvForward<half>;
using IntegrationGpuConvFwdNcdhwFp16 = ConvForward<half>;

using IntegrationGpuConvFwdNhwcFp32 = ConvForward<float>;
using IntegrationGpuConvFwdNdhwcFp32 = ConvForward<float>;

using IntegrationGpuConvFwdNhwcBfp16 = ConvForward<hip_bfloat16>;
using IntegrationGpuConvFwdNdhwcBfp16 = ConvForward<hip_bfloat16>;

using IntegrationGpuConvFwdNhwcFp16 = ConvForward<half>;
using IntegrationGpuConvFwdNdhwcFp16 = ConvForward<half>;

} // namespace

TEST_P(IntegrationGpuConvFwdNchwFp32, Correctness)
{
    runConvTest(4e-6f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdNcdhwFp32, Correctness)
{
    runConvTest(conv::getToleranceFwd<float>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdNchwBfp16, Correctness)
{
    runConvTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdNcdhwBfp16, Correctness)
{
    runConvTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdNchwFp16, Correctness)
{
    runConvTest(conv::getToleranceFwd<half>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdNcdhwFp16, Correctness)
{
    runConvTest(conv::getToleranceFwd<half>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdNhwcFp32, Correctness)
{
    runConvTest(conv::getToleranceFwd<float>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdNdhwcFp32, Correctness)
{
    runConvTest(conv::getToleranceFwd<float>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvFwdNhwcBfp16, Correctness)
{
    runConvTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdNdhwcBfp16, Correctness)
{
    runConvTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvFwdNhwcFp16, Correctness)
{
    runConvTest(conv::getToleranceFwd<half>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdNdhwcFp16, Correctness)
{
    runConvTest(conv::getToleranceFwd<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNchwFp32, testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNchwBfp16, testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNchwFp16, testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNhwcFp32, testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNhwcBfp16, testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNhwcFp16, testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNcdhwFp32, testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvFwdNcdhwBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNcdhwFp16, testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNdhwcFp32, testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuConvFwdNdhwcBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNdhwcFp16, testing::ValuesIn(getConvTestCases5D()));
