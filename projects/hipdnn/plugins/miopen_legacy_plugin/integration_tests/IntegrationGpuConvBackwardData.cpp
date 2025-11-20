// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

#include "../tests/common/ConvolutionCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace test_conv_common;

namespace
{

template <typename DataType>
class ConvBackwardData : public IntegrationGraphVerificationHarness<DataType, ConvTestCase>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        // Skipping until CK is working on Windows
        SKIP_IF_WINDOWS();

        const ConvTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("ConvolutionBackwardDataTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        int64_t uid = 1;

        auto dataType = getDataTypeEnumFromType<DataType>();

        auto dyAttr = graph::makeTensorAttributes(
            "dy", dataType, testCase.yDims, generateStrides(testCase.yDims, layout.strideOrder));
        dyAttr.set_uid(uid++);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto wAttr = graph::makeTensorAttributes(
            "w", dataType, testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder));
        wAttr.set_uid(uid++);
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvDgradAttributes convAttrs;
        convAttrs.set_name("convolution_backward_data");
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dxTensorAttr = graphObj.conv_dgrad(dyTensorAttr, wTensorAttr, convAttrs);

        if(!dxTensorAttr->has_uid())
        {
            dxTensorAttr->set_uid(uid++);
        }
        dxTensorAttr->set_dim(testCase.xDims);
        dxTensorAttr->set_stride(generateStrides(testCase.xDims, layout.strideOrder));
        dxTensorAttr->set_output(true);
        dxTensorAttr->set_data_type(dataType);

        this->registerValidator(dxTensorAttr, tolerance);
        this->verifyGraph(graphObj, testCase.seed);
    }
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
    runGraphTest(4e-6f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvBwdDataNcdhwFp32, Correctness)
{
    runGraphTest(conv::getToleranceBwd<float>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvBwdDataNchwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceBwd<hip_bfloat16>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvBwdDataNcdhwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceBwd<hip_bfloat16>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvBwdDataNchwFp16, Correctness)
{
    runGraphTest(conv::getToleranceBwd<half>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvBwdDataNcdhwFp16, Correctness)
{
    runGraphTest(conv::getToleranceBwd<half>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvBwdDataNhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceBwd<float>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvBwdDataNdhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceBwd<float>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvBwdDataNhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceBwd<hip_bfloat16>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvBwdDataNdhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceBwd<hip_bfloat16>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvBwdDataNhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceBwd<half>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvBwdDataNdhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceBwd<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNchwFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNchwBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNchwFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNhwcFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNhwcBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNhwcFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNcdhwFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNcdhwBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNcdhwFp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNdhwcFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNdhwcBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvBwdDataNdhwcFp16,
                         testing::ValuesIn(getConvTestCases5D()));
