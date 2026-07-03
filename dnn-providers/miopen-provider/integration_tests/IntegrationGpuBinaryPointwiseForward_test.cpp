// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "IntegrationGraphVerificationHarness.hpp"

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
    if constexpr(Mode == hipdnn_frontend::PointwiseMode::MIN)
    {
        return "Min";
    }
    if constexpr(Mode == hipdnn_frontend::PointwiseMode::MAX)
    {
        return "Max";
    }

    return "Unknown";
}

template <typename DataType, hipdnn_frontend::PointwiseMode Mode>
class BinaryPointwiseForward
    : public IntegrationGraphVerificationHarness<DataType, PointwiseTestCase>
{
protected:
    void runGraphTest(float tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        const PointwiseTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        std::string graphName = std::string("Pointwise") + getModeName<Mode>() + "ForwardTest";
        graphObj.set_name(graphName);

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(
                hipdnn_frontend::DataType::FLOAT) // Matches our builder validation!
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

        this->registerValidator(yTensorAttr, tolerance);
        this->verifyGraph(graphObj, testCase.seed);
    }
};

// Operation Type Aliases for Test Framework Mapping
using IntegrationGpuPointwiseAddForwardNchwFp32
    = BinaryPointwiseForward<float, hipdnn_frontend::PointwiseMode::ADD>;
using IntegrationGpuPointwiseAddForwardNchwFp16
    = BinaryPointwiseForward<half, hipdnn_frontend::PointwiseMode::ADD>;

using IntegrationGpuPointwiseSubForwardNchwFp32
    = BinaryPointwiseForward<float, hipdnn_frontend::PointwiseMode::SUB>;
using IntegrationGpuPointwiseSubForwardNchwFp16
    = BinaryPointwiseForward<half, hipdnn_frontend::PointwiseMode::SUB>;

using IntegrationGpuPointwiseMulForwardNchwFp32
    = BinaryPointwiseForward<float, hipdnn_frontend::PointwiseMode::MUL>;
using IntegrationGpuPointwiseMulForwardNchwFp16
    = BinaryPointwiseForward<half, hipdnn_frontend::PointwiseMode::MUL>;

using IntegrationGpuPointwiseMinForwardNchwFp32
    = BinaryPointwiseForward<float, hipdnn_frontend::PointwiseMode::MIN>;
using IntegrationGpuPointwiseMinForwardNchwFp16
    = BinaryPointwiseForward<half, hipdnn_frontend::PointwiseMode::MIN>;

using IntegrationGpuPointwiseMaxForwardNchwFp32
    = BinaryPointwiseForward<float, hipdnn_frontend::PointwiseMode::MAX>;
using IntegrationGpuPointwiseMaxForwardNchwFp16
    = BinaryPointwiseForward<half, hipdnn_frontend::PointwiseMode::MAX>;

} // namespace

// ==================== ADD Mode Tests ====================
TEST_P(IntegrationGpuPointwiseAddForwardNchwFp32, Correctness)
{
    runGraphTest(1e-5f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuPointwiseAddForwardNchwFp16, Correctness)
{
    runGraphTest(1e-3f, TensorLayout::NCHW);
}

// ==================== SUB Mode Tests ====================
TEST_P(IntegrationGpuPointwiseSubForwardNchwFp32, Correctness)
{
    runGraphTest(1e-5f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuPointwiseSubForwardNchwFp16, Correctness)
{
    runGraphTest(1e-3f, TensorLayout::NCHW);
}

// ==================== MUL Mode Tests ====================
TEST_P(IntegrationGpuPointwiseMulForwardNchwFp32, Correctness)
{
    runGraphTest(1e-5f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuPointwiseMulForwardNchwFp16, Correctness)
{
    runGraphTest(1e-3f, TensorLayout::NCHW);
}

// ==================== MIN Mode Tests ====================
TEST_P(IntegrationGpuPointwiseMinForwardNchwFp32, Correctness)
{
    runGraphTest(1e-5f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuPointwiseMinForwardNchwFp16, Correctness)
{
    runGraphTest(1e-3f, TensorLayout::NCHW);
}

// ==================== MAX Mode Tests ====================
TEST_P(IntegrationGpuPointwiseMaxForwardNchwFp32, Correctness)
{
    runGraphTest(1e-5f, TensorLayout::NCHW);
}

TEST_P(IntegrationGpuPointwiseMaxForwardNchwFp16, Correctness)
{
    runGraphTest(1e-3f, TensorLayout::NCHW);
}

// ==================== Parameterized Invocations ====================
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseAddForwardNchwFp32,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseAddForwardNchwFp16,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseSubForwardNchwFp32,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseSubForwardNchwFp16,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseMulForwardNchwFp32,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseMulForwardNchwFp16,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseMinForwardNchwFp32,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseMinForwardNchwFp16,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseMaxForwardNchwFp32,
                         testing::ValuesIn(getPointwiseTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuPointwiseMaxForwardNchwFp16,
                         testing::ValuesIn(getPointwiseTestCases()));
