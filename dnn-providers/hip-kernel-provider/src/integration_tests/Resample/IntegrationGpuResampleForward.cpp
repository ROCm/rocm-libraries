// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;

namespace hip_kernel_provider::resample::test
{

struct ResampleFwdTestCase
{
    std::string name;
    std::vector<int64_t> xDims;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    ResampleMode mode;
    PaddingMode paddingMode;
    unsigned int seed;
};

static std::ostream& operator<<(std::ostream& os, const ResampleFwdTestCase& testCase)
{
    return os << testCase.name;
}

static std::vector<ResampleFwdTestCase> getResampleFwdTestCases()
{
    return {ResampleFwdTestCase{"max_2d_neg_inf",
                                {2, 3, 7, 5},
                                {0, 0},
                                {0, 0},
                                {2, 2},
                                {2, 2},
                                ResampleMode::MAXPOOL,
                                PaddingMode::NEG_INF_PAD,
                                1001},
            ResampleFwdTestCase{"max_2d_zero_pad",
                                {1, 2, 5, 4},
                                {1, 1},
                                {1, 1},
                                {2, 2},
                                {3, 3},
                                ResampleMode::MAXPOOL,
                                PaddingMode::ZERO_PAD,
                                1002},
            ResampleFwdTestCase{"avg_exclude_2d",
                                {2, 2, 6, 5},
                                {1, 0},
                                {0, 1},
                                {2, 1},
                                {3, 2},
                                ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                                PaddingMode::ZERO_PAD,
                                1003},
            ResampleFwdTestCase{"avg_include_3d",
                                {1, 2, 4, 5, 3},
                                {1, 0, 1},
                                {0, 1, 0},
                                {1, 2, 1},
                                {2, 2, 2},
                                ResampleMode::AVGPOOL_INCLUDE_PADDING,
                                PaddingMode::ZERO_PAD,
                                1004}};
}

template <typename InputDataType, typename OutputDataType, typename ComputeDataType>
class ResampleForward
    : public IntegrationGraphVerificationHarness<InputDataType, ResampleFwdTestCase>
{
protected:
    void runGraphTest()
    {
        const auto& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ResampleFwdTest");

        auto inputDataType = getDataTypeEnumFromType<InputDataType>();
        auto outputDataType = getDataTypeEnumFromType<OutputDataType>();
        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        graphObj.set_compute_data_type(computeDataType)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(inputDataType);

        auto xAttr = makeTensorAttributes(
            "X", inputDataType, testCase.xDims, generateStrides(testCase.xDims));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        graph::ResampleFwdAttributes resampleAttrs;
        resampleAttrs.set_pre_padding(testCase.prePadding)
            .set_post_padding(testCase.postPadding)
            .set_stride(testCase.stride)
            .set_window(testCase.window)
            .set_resample_mode(testCase.mode)
            .set_padding_mode(testCase.paddingMode);

        auto yTensorAttr = graphObj.resample_fwd(xTensorAttr, resampleAttrs);
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(outputDataType);
        this->registerValidator(yTensorAttr, 1e-5f);

        this->verifyGraph(graphObj, testCase.seed);
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

} // namespace hip_kernel_provider::resample::test
