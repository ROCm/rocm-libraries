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
class ConvBackwardWeights : public IntegrationGraphVerificationHarness<DataType, ConvTestCase>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const ConvTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("ConvolutionBackwardWeightTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        int64_t uid = 1;

        auto dataType = getDataTypeEnumFromType<DataType>();

        auto xAttr = graph::makeTensorAttributes(
            "x", dataType, testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr = graph::makeTensorAttributes(
            "dy", dataType, testCase.yDims, generateStrides(testCase.yDims, layout.strideOrder));
        dyAttr.set_uid(uid++);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        graph::ConvWgradAttributes convAttrs;
        convAttrs.set_name("convolution_backward_weights");
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dwTensorAttr = graphObj.conv_wgrad(dyTensorAttr, xTensorAttr, convAttrs);

        if(!dwTensorAttr->has_uid())
        {
            dwTensorAttr->set_uid(uid++);
        }
        dwTensorAttr->set_dim(testCase.wDims);
        dwTensorAttr->set_stride(generateStrides(testCase.wDims, layout.strideOrder));
        dwTensorAttr->set_output(true);
        dwTensorAttr->set_data_type(dataType);

        this->registerValidator(dwTensorAttr, tolerance);
        this->verifyGraph(graphObj, testCase.seed);
    }
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
    runGraphTest(conv::getToleranceWrw<float>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvWrwDataNcdhwFp32, Correctness)
{
    runGraphTest(conv::getToleranceWrw<float>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvWrwDataNchwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceWrw<hip_bfloat16>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvWrwDataNcdhwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceWrw<hip_bfloat16>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvWrwDataNchwFp16, Correctness)
{
    runGraphTest(conv::getToleranceWrw<half>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvWrwDataNcdhwFp16, Correctness)
{
    runGraphTest(conv::getToleranceWrw<half>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvWrwDataNhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceWrw<float>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvWrwDataNdhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceWrw<float>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvWrwDataNhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceWrw<hip_bfloat16>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvWrwDataNdhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceWrw<hip_bfloat16>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvWrwDataNhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceWrw<half>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvWrwDataNdhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceWrw<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNchwFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNchwBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNchwFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNhwcFp32,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNhwcBfp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNhwcFp16,
                         testing::ValuesIn(getConvTestCases4D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNcdhwFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNcdhwBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNcdhwFp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNdhwcFp32,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNdhwcBfp16,
                         testing::ValuesIn(getConvTestCases5D()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuConvWrwDataNdhwcFp16,
                         testing::ValuesIn(getConvTestCases5D()));
