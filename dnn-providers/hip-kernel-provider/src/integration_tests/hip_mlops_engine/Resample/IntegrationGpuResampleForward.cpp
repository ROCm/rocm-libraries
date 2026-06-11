// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;

namespace hip_kernel_provider::resample::test
{

namespace
{

struct ResampleFwdTestCase
{
    std::string name;
    std::vector<int64_t> xDims;
    TensorLayout layout;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    ResampleMode mode;
    PaddingMode paddingMode;
};

std::ostream& operator<<(std::ostream& os, const ResampleFwdTestCase& testCase)
{
    return os << testCase.name;
}

std::vector<ResampleFwdTestCase> getResampleFwdTestCases()
{
    struct TensorCase
    {
        std::string name;
        std::vector<int64_t> xDims;
        TensorLayout layout;
    };

    struct ParameterCase
    {
        std::string name;
        std::vector<int64_t> prePadding;
        std::vector<int64_t> postPadding;
        std::vector<int64_t> stride;
        std::vector<int64_t> window;
        ResampleMode mode;
        PaddingMode paddingMode;
    };

    const std::vector<TensorCase> tensorCases{{"2d_nchw", {2, 3, 7, 5}, TensorLayout::NCHW},
                                              {"2d_nhwc", {2, 3, 7, 5}, TensorLayout::NHWC},
                                              {"2d_wide", {1, 2, 8, 6}, TensorLayout::NCHW},
                                              {"3d_ncdhw", {1, 2, 4, 5, 3}, TensorLayout::NCDHW},
                                              {"3d_ndhwc", {1, 2, 4, 5, 3}, TensorLayout::NDHWC}};

    const std::vector<ParameterCase> twoDimParameterCases{{"max_neg_inf",
                                                           {0, 0},
                                                           {0, 0},
                                                           {2, 2},
                                                           {2, 2},
                                                           ResampleMode::MAXPOOL,
                                                           PaddingMode::NEG_INF_PAD},
                                                          {"max_zero_pad",
                                                           {1, 1},
                                                           {1, 1},
                                                           {2, 2},
                                                           {3, 3},
                                                           ResampleMode::MAXPOOL,
                                                           PaddingMode::ZERO_PAD},
                                                          {"avg_exclude",
                                                           {1, 0},
                                                           {0, 1},
                                                           {2, 1},
                                                           {3, 2},
                                                           ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                                           PaddingMode::ZERO_PAD},
                                                          {"avg_include",
                                                           {0, 1},
                                                           {1, 0},
                                                           {1, 2},
                                                           {2, 3},
                                                           ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                           PaddingMode::ZERO_PAD}};

    const std::vector<ParameterCase> threeDimParameterCases{{"max_neg_inf",
                                                             {0, 0, 0},
                                                             {0, 0, 0},
                                                             {1, 2, 1},
                                                             {2, 2, 2},
                                                             ResampleMode::MAXPOOL,
                                                             PaddingMode::NEG_INF_PAD},
                                                            {"avg_include",
                                                             {1, 0, 1},
                                                             {0, 1, 0},
                                                             {1, 2, 1},
                                                             {2, 2, 2},
                                                             ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                                             PaddingMode::ZERO_PAD}};

    std::vector<ResampleFwdTestCase> testCases;
    for(const auto& tensorCase : tensorCases)
    {
        const auto& parameterCases
            = tensorCase.xDims.size() == 4 ? twoDimParameterCases : threeDimParameterCases;
        for(const auto& parameterCase : parameterCases)
        {
            testCases.push_back({tensorCase.name + "_" + parameterCase.name,
                                 tensorCase.xDims,
                                 tensorCase.layout,
                                 parameterCase.prePadding,
                                 parameterCase.postPadding,
                                 parameterCase.stride,
                                 parameterCase.window,
                                 parameterCase.mode,
                                 parameterCase.paddingMode});
        }
    }

    return testCases;
}

template <typename XDataType, typename YDataType, typename ComputeDataType>
class ResampleForward : public IntegrationGraphVerificationHarness<XDataType, ResampleFwdTestCase>
{
protected:
    void runGraphTest()
    {
        const auto& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ResampleFwdTest");

        auto inputDataType = getDataTypeEnumFromType<XDataType>();
        auto yDataType = getDataTypeEnumFromType<YDataType>();
        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        graphObj.set_compute_data_type(computeDataType)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(inputDataType);

        auto xAttr
            = makeTensorAttributes("X",
                                   inputDataType,
                                   testCase.xDims,
                                   generateStrides(testCase.xDims, testCase.layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        graph::ResampleFwdAttributes resampleAttrs;
        resampleAttrs.set_pre_padding(testCase.prePadding)
            .set_post_padding(testCase.postPadding)
            .set_stride(testCase.stride)
            .set_window(testCase.window)
            .set_resample_mode(testCase.mode)
            .set_padding_mode(testCase.paddingMode);

        auto [yTensorAttr, indexTensorAttr] = graphObj.resample(xTensorAttr, resampleAttrs);
        (void)indexTensorAttr;
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(yDataType);
        this->registerValidator(yTensorAttr, 1e-5f);

        this->verifyGraph(graphObj, hipdnn_test_sdk::utilities::getGlobalTestSeed());
    }
};

using IntegrationGpuResampleForwardFp32 = ResampleForward<float, float, float>;

TEST_P(IntegrationGpuResampleForwardFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuResampleForwardFp32,
                         testing::ValuesIn(getResampleFwdTestCases()));

} // namespace

} // namespace hip_kernel_provider::resample::test
