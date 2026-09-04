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
using namespace miopen_plugin::test_utilities;

namespace
{

struct PointwiseTestCase
{
    std::vector<int64_t> dims;
    unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();
};

std::vector<PointwiseTestCase> getPointwiseTestCases()
{
    return {
        {{2, 4, 8, 8}},
        {{1, 16, 4, 4}},
        {{4, 8, 16, 16}},
    };
}

template <hipdnn_frontend::PointwiseMode Mode>
constexpr const char* getModeName()
{
    if constexpr(Mode == hipdnn_frontend::PointwiseMode::ADD)
    {
        return "Add";
    }
    if constexpr(Mode == hipdnn_frontend::PointwiseMode::SUB)
    {
        return "Sub";
    }
    if constexpr(Mode == hipdnn_frontend::PointwiseMode::MUL)
    {
        return "Mul";
    }

    static_assert(Mode == hipdnn_frontend::PointwiseMode::ADD
                    || Mode == hipdnn_frontend::PointwiseMode::SUB
                    || Mode == hipdnn_frontend::PointwiseMode::MUL,
                    "getModeName: unsupported PointwiseMode");
}

template <typename DataType, hipdnn_frontend::PointwiseMode Mode>
class BinaryPointwise : public IntegrationGraphVerificationHarness<DataType, PointwiseTestCase>
{
protected:
    void runGraphTest(float tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        const PointwiseTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        const std::string graphName = std::string("Pointwise") + getModeName<Mode>() + "Test";
        graphObj.set_name(graphName);

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        // Generate attributes for Input Tensor 0 ("x0")
        auto x0Attr = makeTensorAttributes(
            "x0", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        auto x0TensorAttr = std::make_shared<graph::TensorAttributes>(std::move(x0Attr));

        // Generate attributes for Input Tensor 1 ("x1")
        auto x1Attr = makeTensorAttributes(
            "x1", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        auto x1TensorAttr = std::make_shared<graph::TensorAttributes>(std::move(x1Attr));

        // Configure the pointwise operation attributes dynamically
        graph::PointwiseAttributes pwAttrs;
        pwAttrs.set_mode(Mode);

        // Dispatch the binary variant of the frontend graph's pointwise node generator
        auto yTensorAttr = graphObj.pointwise(x0TensorAttr, x1TensorAttr, pwAttrs);
        yTensorAttr->set_output(true);

        this->registerValidator(yTensorAttr, this->getTolerance(graphObj, yTensorAttr));
        this->verifyGraph(graphObj, testCase.seed);
    }
};

// Operation Type Aliases for Test Framework Mapping
using IntegrationGpuPointwiseAddNchwFp32
    = BinaryPointwise<float, hipdnn_frontend::PointwiseMode::ADD>;
using IntegrationGpuPointwiseAddNchwFp16
    = BinaryPointwise<half, hipdnn_frontend::PointwiseMode::ADD>;
using IntegrationGpuPointwiseSubNchwFp32
    = BinaryPointwise<float, hipdnn_frontend::PointwiseMode::SUB>;
using IntegrationGpuPointwiseSubNchwFp16
    = BinaryPointwise<half, hipdnn_frontend::PointwiseMode::SUB>;
using IntegrationGpuPointwiseMulNchwFp32
    = BinaryPointwise<float, hipdnn_frontend::PointwiseMode::MUL>;
using IntegrationGpuPointwiseMulNchwFp16
    = BinaryPointwise<half, hipdnn_frontend::PointwiseMode::MUL>;

} // namespace

// ==================== ADD Mode Tests ====================
TEST_P(IntegrationGpuPointwiseAddNchwFp32, Correctness)
{
    runGraphTest(1e-5f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuPointwiseAddNchwFp16, Correctness)
{
    runGraphTest(1e-3f, TensorLayout::NCHW);
}

// ==================== SUB Mode Tests ====================
TEST_P(IntegrationGpuPointwiseSubNchwFp32, Correctness)
{
    runGraphTest(1e-5f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuPointwiseSubNchwFp16, Correctness)
{
    runGraphTest(1e-3f, TensorLayout::NCHW);
}

// ==================== MUL Mode Tests ====================
TEST_P(IntegrationGpuPointwiseMulNchwFp32, Correctness)
{
    runGraphTest(1e-5f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuPointwiseMulNchwFp16, Correctness)
{
    runGraphTest(1e-3f, TensorLayout::NCHW);
}

// ==================== Parameterized Invocations ====================
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseAddNchwFp32,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseAddNchwFp16,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseSubNchwFp32,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseSubNchwFp16,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseMulNchwFp32,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseMulNchwFp16,
                         testing::ValuesIn(getPointwiseTestCases()));
