// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/BatchnormCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_bn_common;

namespace
{

struct BatchnormBwdTensorIds
{
    static constexpr int64_t X_UID = 1;
    static constexpr int64_t DY_UID = 2;
    static constexpr int64_t SCALE_UID = 3;
    static constexpr int64_t MEAN_UID = 4;
    static constexpr int64_t INV_VARIANCE_UID = 5;
};

using BnBwdTestCase = std::tuple<TensorLayout, test_bn_common::BatchnormTestCase>;

template <typename DataType, bool CalcStats = false>
class BatchnormBackward : public IntegrationGraphVerificationHarness<DataType, BnBwdTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dx;
        std::shared_ptr<graph::TensorAttributes> dscale;
        std::shared_ptr<graph::TensorAttributes> dbias;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const BnBwdTestCase& tc)
    {
        const auto& [layout, bnTestCase] = tc;
        auto dims = bnTestCase.dims;
        auto derivedDims = getDerivedShape(dims);

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;

        graph::Graph graphObj;
        graphObj.set_name("BatchnormBackwardTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
        graphObj.set_intermediate_data_type(intermediateDataType);
        graphObj.set_io_data_type(dataType);

        auto xAttr
            = graph::makeTensorAttributes("x", dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(BatchnormBwdTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr
            = graph::makeTensorAttributes("dy", dims, generateStrides(dims, layout.strideOrder));
        dyAttr.set_uid(BatchnormBwdTensorIds::DY_UID);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(BatchnormBwdTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        graph::BatchnormBackwardAttributes bnAttrs;

        std::shared_ptr<graph::TensorAttributes> meanTensorAttr;
        std::shared_ptr<graph::TensorAttributes> invVarianceTensorAttr;

        if(!CalcStats)
        {
            auto meanAttr = graph::makeTensorAttributes(
                "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
            meanAttr.set_uid(BatchnormBwdTensorIds::MEAN_UID);
            meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

            auto invVarianceAttr = graph::makeTensorAttributes(
                "inv_variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
            invVarianceAttr.set_uid(BatchnormBwdTensorIds::INV_VARIANCE_UID);
            invVarianceTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

            bnAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);
        }

        auto outputTensorsAttr
            = graphObj.batchnorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, bnAttrs);

        auto& dxTensorAttr = outputTensorsAttr[0];
        dxTensorAttr->set_output(true);

        auto& dscaleTensorAttr = outputTensorsAttr[1];
        dscaleTensorAttr->set_data_type(intermediateDataType);
        dscaleTensorAttr->set_output(true);

        auto& dbiasTensorAttr = outputTensorsAttr[2];
        dbiasTensorAttr->set_data_type(intermediateDataType);
        dbiasTensorAttr->set_output(true);

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

        return {std::move(graphObj), GraphOutputs{dxTensorAttr, dscaleTensorAttr, dbiasTensorAttr}};
    }

protected:
    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.sentinelFillOutputTensors();

        bundle.tensors.at(BatchnormBwdTensorIds::X_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
        bundle.tensors.at(BatchnormBwdTensorIds::DY_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);
        bundle.tensors.at(BatchnormBwdTensorIds::SCALE_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);

        if(!CalcStats)
        {
            bundle.tensors.at(BatchnormBwdTensorIds::MEAN_UID)
                ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);

            bundle.tensors.at(BatchnormBwdTensorIds::INV_VARIANCE_UID)
                ->fillTensorWithRandomValues(1.9f, 2.0f, seed);
        }
    }

    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();
        const auto& [layout, bnTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        // Register validators
        this->registerValidator(outputs.dx, this->getTolerance(graphObj, outputs.dx));
        this->registerValidator(outputs.dscale, this->getTolerance(graphObj, outputs.dscale));
        this->registerValidator(outputs.dbias, this->getTolerance(graphObj, outputs.dbias));

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(bnTestCase.note);
        this->verifyGraph(graphObj, bnTestCase.seed);
    }
};

// 1D layout tests (NCL, NLC)
using IntegrationGpuBatchnormBackward1dFp32 = BatchnormBackward<float>;
using IntegrationGpuBatchnormBackward1dBfp16 = BatchnormBackward<bfloat16>;
using IntegrationGpuBatchnormBackward1dFp16 = BatchnormBackward<half>;

// 2D layout tests (NCHW, NHWC)
using IntegrationGpuBatchnormBackward2dFp32 = BatchnormBackward<float>;
using IntegrationGpuBatchnormBackward2dBfp16 = BatchnormBackward<bfloat16>;
using IntegrationGpuBatchnormBackward2dFp16 = BatchnormBackward<half>;

// 3D layout tests (NCDHW, NDHWC)
using IntegrationGpuBatchnormBackward3dFp32 = BatchnormBackward<float>;
using IntegrationGpuBatchnormBackward3dBfp16 = BatchnormBackward<bfloat16>;
using IntegrationGpuBatchnormBackward3dFp16 = BatchnormBackward<half>;

// 1D CalcStats layout tests
using IntegrationGpuBatchnormBackwardCalcStats1dFp32 = BatchnormBackward<float, true>;
using IntegrationGpuBatchnormBackwardCalcStats1dBfp16 = BatchnormBackward<bfloat16, true>;
using IntegrationGpuBatchnormBackwardCalcStats1dFp16 = BatchnormBackward<half, true>;

// 2D CalcStats layout tests
using IntegrationGpuBatchnormBackwardCalcStats2dFp32 = BatchnormBackward<float, true>;
using IntegrationGpuBatchnormBackwardCalcStats2dBfp16 = BatchnormBackward<bfloat16, true>;
using IntegrationGpuBatchnormBackwardCalcStats2dFp16 = BatchnormBackward<half, true>;

// 3D CalcStats layout tests
using IntegrationGpuBatchnormBackwardCalcStats3dFp32 = BatchnormBackward<float, true>;
using IntegrationGpuBatchnormBackwardCalcStats3dBfp16 = BatchnormBackward<bfloat16, true>;
using IntegrationGpuBatchnormBackwardCalcStats3dFp16 = BatchnormBackward<half, true>;

} // namespace

// ============================================================================
// 1D Tests - Standard (saved mean/inv_variance)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward1dFp32);
TEST_P(IntegrationGpuBatchnormBackward1dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward1dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnBwd1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackward1dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnBwd1dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward1dBfp16);
TEST_P(IntegrationGpuBatchnormBackward1dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward1dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnBwd1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackward1dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnBwd1dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward1dFp16);
TEST_P(IntegrationGpuBatchnormBackward1dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward1dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnBwd1dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackward1dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnBwd1dFullTestCases())));

// ============================================================================
// 2D Tests - Standard (saved mean/inv_variance)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward2dFp32);
TEST_P(IntegrationGpuBatchnormBackward2dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward2dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnBwdTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackward2dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnBwdFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward2dBfp16);
TEST_P(IntegrationGpuBatchnormBackward2dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward2dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnBwdTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackward2dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnBwdFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward2dFp16);
TEST_P(IntegrationGpuBatchnormBackward2dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward2dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnBwdTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackward2dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnBwdFullTestCases())));

// ============================================================================
// 3D Tests - Standard (saved mean/inv_variance)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward3dFp32);
TEST_P(IntegrationGpuBatchnormBackward3dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward3dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackward3dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward3dBfp16);
TEST_P(IntegrationGpuBatchnormBackward3dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward3dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackward3dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward3dFp16);
TEST_P(IntegrationGpuBatchnormBackward3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward3dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackward3dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dFullTestCases())));

// ============================================================================
// 1D Tests - CalcStats (no saved mean/inv_variance)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardCalcStats1dFp32);
TEST_P(IntegrationGpuBatchnormBackwardCalcStats1dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStats1dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnBwd1dTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardCalcStats1dBfp16);
TEST_P(IntegrationGpuBatchnormBackwardCalcStats1dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStats1dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnBwd1dTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardCalcStats1dFp16);
TEST_P(IntegrationGpuBatchnormBackwardCalcStats1dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStats1dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCL, TensorLayout::NLC),
                                          testing::ValuesIn(getBnBwd1dTestCases())));

// ============================================================================
// 2D Tests - CalcStats (no saved mean/inv_variance)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardCalcStats2dFp32);
TEST_P(IntegrationGpuBatchnormBackwardCalcStats2dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStats2dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnBwdTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardCalcStats2dBfp16);
TEST_P(IntegrationGpuBatchnormBackwardCalcStats2dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStats2dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnBwdTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardCalcStats2dFp16);
TEST_P(IntegrationGpuBatchnormBackwardCalcStats2dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStats2dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                                          testing::ValuesIn(getBnBwdTestCases())));

// ============================================================================
// 3D Tests - CalcStats (no saved mean/inv_variance)
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardCalcStats3dFp32);
TEST_P(IntegrationGpuBatchnormBackwardCalcStats3dFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStats3dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardCalcStats3dFp32,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardCalcStats3dBfp16);
TEST_P(IntegrationGpuBatchnormBackwardCalcStats3dBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStats3dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardCalcStats3dBfp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dFullTestCases())));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardCalcStats3dFp16);
TEST_P(IntegrationGpuBatchnormBackwardCalcStats3dFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardCalcStats3dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dTestCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardCalcStats3dFp16,
                         testing::Combine(testing::Values(TensorLayout::NCDHW, TensorLayout::NDHWC),
                                          testing::ValuesIn(getBnBwd3dFullTestCases())));
