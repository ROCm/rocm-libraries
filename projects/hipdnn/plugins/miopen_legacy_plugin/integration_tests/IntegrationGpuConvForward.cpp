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
#include "IntegrationTestUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace test_conv_common;

namespace
{

template <typename DataType>
class ConvForward : public GraphVerifierTest<DataType, ConvTestCase>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const ConvTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("ConvolutionForwardTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        int64_t uid = 1;

        auto dataType = getDataTypeEnumFromType<DataType>();

        auto xAttr = graph::makeTensorAttributes(
            "x", dataType, testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr = graph::makeTensorAttributes(
            "w", dataType, testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder));
        wAttr.set_uid(uid++);
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_name("convolution_forward");
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto yAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

        if(!yAttr->has_uid())
        {
            yAttr->set_uid(uid++);
        }
        yAttr->set_output(true);
        yAttr->set_data_type(dataType);
        yAttr->set_dim(testCase.yDims);
        yAttr->set_stride(generateStrides(testCase.yDims, layout.strideOrder));

        CpuFpReferenceValidation<DataType> validator(tolerance, tolerance);
        this->verifyGraph(graphObj, testCase.seed, validator);
    }
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
    runGraphTest(4e-6f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdNcdhwFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdNchwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdNcdhwBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdNchwFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NCHW);
}

TEST_P(IntegrationGpuConvFwdNcdhwFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NCDHW);
}

TEST_P(IntegrationGpuConvFwdNhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdNdhwcFp32, Correctness)
{
    runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvFwdNhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdNdhwcBfp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NDHWC);
}

TEST_P(IntegrationGpuConvFwdNhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NHWC);
}

TEST_P(IntegrationGpuConvFwdNdhwcFp16, Correctness)
{
    runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NDHWC);
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
