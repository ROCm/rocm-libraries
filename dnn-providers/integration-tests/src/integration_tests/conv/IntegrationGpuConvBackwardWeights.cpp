// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/ConvolutionCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_conv_common;

namespace
{

using ConvWgradTestCase = std::tuple<TensorLayout, test_conv_common::ConvTestCase>;

template <typename DataType>
class ConvBackwardWeights : public IntegrationGraphVerificationHarness<DataType, ConvWgradTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dw;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const ConvWgradTestCase& tc)
    {
        const auto& [layout, testCase] = tc;

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ConvolutionBackwardWeightTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr = graph::makeTensorAttributes(
            "dy", testCase.yDims, generateStrides(testCase.yDims, layout.strideOrder));
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        graph::ConvWgradAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto dwAttr = graphObj.conv_wgrad(dyTensorAttr, xTensorAttr, convAttrs);
        dwAttr->set_output(true);

        // Set these explicitly since grouped convs cannot infer tensor shape.
        // Infer behavior will assume groups == 1, but some cases have groups > 1.
        dwAttr->set_dim(testCase.wDims);
        dwAttr->set_stride(generateStrides(testCase.wDims, layout.strideOrder));

        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            throw std::runtime_error("Failed to validate graph: " + validateResult.get_message());
        }

        auto buildResult = graphObj.build_operation_graph(handle);
        if(buildResult.is_bad())
        {
            throw std::runtime_error("Failed to build operation graph: "
                                     + buildResult.get_message());
        }

        return std::make_pair(std::move(graphObj), GraphOutputs{dwAttr});
    }

protected:
    void runGraphTest() override
    {
        // rocBLAS/Tensile heap-buffer-overflow on gfx90a; CK ASAN stall on gfx942
        SKIP_IF_ASAN();

        const auto& testCase = this->GetParam();
        const auto& [layout, convTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.dw, this->getTolerance(graphObj, outputs.dw));

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(convTestCase.note);
        this->verifyGraph(graphObj, convTestCase.seed);
    }
};

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuConvWrw2dFp32 = ConvBackwardWeights<float>;
using IntegrationGpuConvWrw2dBfp16 = ConvBackwardWeights<bfloat16>;
using IntegrationGpuConvWrw2dFp16 = ConvBackwardWeights<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuConvWrw3dFp32 = ConvBackwardWeights<float>;
using IntegrationGpuConvWrw3dBfp16 = ConvBackwardWeights<bfloat16>;
using IntegrationGpuConvWrw3dFp16 = ConvBackwardWeights<half>;

// Large values variant — tests with wider input range [-10, 10]
template <typename DataType>
class ConvBackwardWeightsLargeValues : public ConvBackwardWeights<DataType>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, -10.0f, 10.0f, seed);
        }
    }
};
// Large input value range [-10, 10] stress-tests numerical precision in wgrad
// accumulation. Limited to fp32 because half/bfloat16 would overflow during
// reduction over batch and spatial dimensions.
using IntegrationGpuConvWrwLargeValues2dFp32 = ConvBackwardWeightsLargeValues<float>;

} // namespace

// 2D tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvWrw2dFp32);
TEST_P(IntegrationGpuConvWrw2dFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvWrw2dBfp16);
TEST_P(IntegrationGpuConvWrw2dBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvWrw2dFp16);
TEST_P(IntegrationGpuConvWrw2dFp16, Correctness)
{
    runGraphTest();
}

// 3D tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvWrw3dFp32);
TEST_P(IntegrationGpuConvWrw3dFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvWrw3dBfp16);
TEST_P(IntegrationGpuConvWrw3dBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvWrw3dFp16);
TEST_P(IntegrationGpuConvWrw3dFp16, Correctness)
{
    runGraphTest();
}

// Large values test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvWrwLargeValues2dFp32);
TEST_P(IntegrationGpuConvWrwLargeValues2dFp32, Correctness)
{
    runGraphTest();
}

// 2D instantiations
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvWrw2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvWrw2dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvWrw2dFp16,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));

// 3D instantiations
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvWrw3dFp32,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvWrw3dBfp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D())));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvWrw3dFp16,
    testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases5D())));

// Large values instantiation — only first 4D case, NCHW layout
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvWrwLargeValues2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW),
                     testing::Values(test_conv_common::getConvTestCases4D()[0])));
