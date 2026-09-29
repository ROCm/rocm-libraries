// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <tuple>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/BinaryPointwiseCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_binary_pointwise_common;

namespace
{

using SdkPointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
using PointwiseBinaryTestCase = std::tuple<BinaryPointwiseShapeCase, SdkPointwiseMode>;

template <typename DataType>
class PointwiseBinary
    : public IntegrationGraphVerificationHarness<DataType, PointwiseBinaryTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> out;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const PointwiseBinaryTestCase& tc)
    {
        const auto& [shape, mode] = tc;

        graph::Graph graphObj;
        graphObj.set_name("PointwiseBinaryTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes("x", shape.xDims, shape.xStrides);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto yAttr = graph::makeTensorAttributes("y", shape.yDims, shape.yStrides);
        auto yTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(yAttr));

        graph::PointwiseAttributes pwAttrs;
        pwAttrs.set_mode(sdkToFrontendPointwiseMode(mode));

        auto outTensorAttr = graphObj.pointwise(xTensorAttr, yTensorAttr, pwAttrs);
        outTensorAttr->set_output(true);

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

        return std::make_pair(std::move(graphObj), GraphOutputs{outTensorAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& shape = std::get<BinaryPointwiseShapeCase>(testCase);

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.out, this->getTolerance(graphObj, outputs.out));

        this->setTestCaseNote(shape.note);
        this->inputFillRecipes().setGlobalSeed(hipdnn_test_sdk::utilities::getGlobalTestSeed());
        this->verifyGraph(graphObj);
    }
};

using IntegrationGpuPointwiseBinaryFp32 = PointwiseBinary<float>;
using IntegrationGpuPointwiseBinaryFp16 = PointwiseBinary<half>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuPointwiseBinaryFp32);
TEST_P(IntegrationGpuPointwiseBinaryFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuPointwiseBinaryFp16);
TEST_P(IntegrationGpuPointwiseBinaryFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseBinaryFp32,
                         testing::Combine(testing::ValuesIn(createBinaryPointwiseShapeCases()),
                                          testing::ValuesIn(createBinaryPointwiseModes())));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseBinaryFp16,
                         testing::Combine(testing::ValuesIn(createBinaryPointwiseShapeCases()),
                                          testing::ValuesIn(createBinaryPointwiseModes())));
