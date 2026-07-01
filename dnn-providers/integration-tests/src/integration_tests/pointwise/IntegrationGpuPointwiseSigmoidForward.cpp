// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;

namespace
{

struct PointwiseTestCase
{
    std::vector<int64_t> dims;
    unsigned int seed = 0;
};

std::vector<PointwiseTestCase> getPointwiseTestCases()
{
    return {
        {{2, 4, 8, 8}, 0},
        {{1, 16, 4, 4}, 1},
        {{4, 8, 16, 16}, 2},
    };
}

template <typename DataType>
class PointwiseSigmoidForward
    : public IntegrationGraphVerificationHarness<DataType, PointwiseTestCase>
{
protected:
    void runGraphTest() override
    {
        runGraphTest(1e-5f, TensorLayout::NCHW);
    }

    void runGraphTest(float tolerance, const TensorLayout& layout)
    {
        const PointwiseTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("PointwiseSigmoidForwardTest");

        auto dataType = getDataTypeEnumFromType<DataType>();

        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = makeTensorAttributes(
            "x", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));

        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        graph::PointwiseAttributes pwAttrs;

        pwAttrs.set_mode(hipdnn_frontend::PointwiseMode::SIGMOID_FWD);

        auto yTensorAttr = graphObj.pointwise(xTensorAttr, pwAttrs);

        yTensorAttr->set_output(true);

        this->registerValidator(yTensorAttr, tolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuPointwiseSigmoidNchwFp32 = PointwiseSigmoidForward<float>;

using IntegrationGpuPointwiseSigmoidNchwFp16 = PointwiseSigmoidForward<half>;

} // namespace

TEST_P(IntegrationGpuPointwiseSigmoidNchwFp32, Correctness)
{
    runGraphTest(1e-5f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuPointwiseSigmoidNchwFp16, Correctness)
{
    runGraphTest(1e-3f, TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseSigmoidNchwFp32,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseSigmoidNchwFp16,
                         testing::ValuesIn(getPointwiseTestCases()));
