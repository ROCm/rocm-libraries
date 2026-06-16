// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../IntegrationGraphVerificationHarness.hpp"
#include "BatchnormCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;
using namespace hip_kernel_provider::batchnorm::test::common;
using namespace hipdnn_test_sdk::utilities::batchnorm;

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

template <typename InputType, bool CalcStats = true>
class BatchnormBackward : public IntegrationGraphVerificationHarness<InputType, BatchnormTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dx;
        std::shared_ptr<graph::TensorAttributes> dscale;
        std::shared_ptr<graph::TensorAttributes> dbias;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(const BatchnormTestCase& tc,
                                                            const TensorLayout& layout)
    {
        auto dims = tc.dims;
        auto derivedDims = getDerivedShape(dims);

        graph::Graph graphObj;
        graphObj.set_name("BatchnormBackwardTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);
        graphObj.set_io_data_type(getDataTypeEnumFromType<InputType>());

        auto xAttr
            = graph::makeTensorAttributes("x", dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(BatchnormBwdTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr
            = graph::makeTensorAttributes("dy", dims, generateStrides(dims, layout.strideOrder));
        dyAttr.set_uid(BatchnormBwdTensorIds::DY_UID);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale",
                                          hipdnn_frontend::DataType::FLOAT,
                                          derivedDims,
                                          generateStrides(derivedDims, layout.strideOrder));
        scaleAttr.set_uid(BatchnormBwdTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        graph::BatchnormBackwardAttributes bnAttrs;
        std::shared_ptr<graph::TensorAttributes> meanTensorAttr;
        std::shared_ptr<graph::TensorAttributes> invVarianceTensorAttr;

        if(!CalcStats)
        {
            auto meanAttr = graph::makeTensorAttributes("mean",
                                                        hipdnn_frontend::DataType::FLOAT,
                                                        derivedDims,
                                                        generateStrides(derivedDims));
            meanAttr.set_uid(BatchnormBwdTensorIds::MEAN_UID);
            meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

            auto invVarianceAttr = graph::makeTensorAttributes("inv_variance",
                                                               hipdnn_frontend::DataType::FLOAT,
                                                               derivedDims,
                                                               generateStrides(derivedDims));
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
        dscaleTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);
        dscaleTensorAttr->set_output(true);

        auto& dbiasTensorAttr = outputTensorsAttr[2];
        dbiasTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);
        dbiasTensorAttr->set_output(true);

        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            throw std::runtime_error("Failed to validate graph: " + validateResult.get_message());
        }

        return {std::move(graphObj), GraphOutputs{dxTensorAttr, dscaleTensorAttr, dbiasTensorAttr}};
    }

protected:
    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.tensors.at(BatchnormBwdTensorIds::X_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
        bundle.tensors.at(BatchnormBwdTensorIds::DY_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed + 1);
        bundle.tensors.at(BatchnormBwdTensorIds::SCALE_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed + 2);

        if(!CalcStats)
        {
            bundle.tensors.at(BatchnormBwdTensorIds::MEAN_UID)
                ->fillTensorWithRandomValues(-0.1f, 0.1f, seed + 3);
            bundle.tensors.at(BatchnormBwdTensorIds::INV_VARIANCE_UID)
                ->fillTensorWithRandomValues(1.9f, 2.0f, seed + 4);
        }
    }

    void runGraphTest(const TensorLayout& layout)
    {
        const auto& testCase = this->GetParam();
        auto [graphObj, outputs] = buildGraph(testCase, layout);

        auto tolerance = getToleranceTraining<InputType>();
        this->registerValidator(outputs.dx, tolerance);
        this->registerValidator(outputs.dscale, tolerance);
        this->registerValidator(outputs.dbias, tolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

// NCHW/NDHWC layouts
using IntegrationGpuBatchnormBackward2dNchwFp32 = BatchnormBackward<float>;
using IntegrationGpuBatchnormBackward2dNchwFp16 = BatchnormBackward<half>;
using IntegrationGpuBatchnormBackward2dNchwBfp16 = BatchnormBackward<bfloat16>;
using IntegrationGpuBatchnormBackward3dNcdhwFp32 = BatchnormBackward<float>;
using IntegrationGpuBatchnormBackward3dNcdhwFp16 = BatchnormBackward<half>;
using IntegrationGpuBatchnormBackward3dNcdhwBfp16 = BatchnormBackward<bfloat16>;

// NHWC/NDHWC layouts
using IntegrationGpuBatchnormBackward2dNhwcFp32 = BatchnormBackward<float>;
using IntegrationGpuBatchnormBackward2dNhwcFp16 = BatchnormBackward<half>;
using IntegrationGpuBatchnormBackward2dNhwcBfp16 = BatchnormBackward<bfloat16>;
using IntegrationGpuBatchnormBackward3dNdhwcFp32 = BatchnormBackward<float>;
using IntegrationGpuBatchnormBackward3dNdhwcFp16 = BatchnormBackward<half>;
using IntegrationGpuBatchnormBackward3dNdhwcBfp16 = BatchnormBackward<bfloat16>;

} // namespace

// ============================================================================
// NCHW 2D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward2dNchwFp32);
TEST_P(IntegrationGpuBatchnormBackward2dNchwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward2dNchwFp32,
                         testing::ValuesIn(getBnBwdSmoke2dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward2dNchwFp16);
TEST_P(IntegrationGpuBatchnormBackward2dNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward2dNchwFp16,
                         testing::ValuesIn(getBnBwdSmoke2dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward2dNchwBfp16);
TEST_P(IntegrationGpuBatchnormBackward2dNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward2dNchwBfp16,
                         testing::ValuesIn(getBnBwdSmoke2dTestCases()));

// ============================================================================
// NCDHW 3D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward3dNcdhwFp32);
TEST_P(IntegrationGpuBatchnormBackward3dNcdhwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward3dNcdhwFp32,
                         testing::ValuesIn(getBnBwdSmoke3dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward3dNcdhwFp16);
TEST_P(IntegrationGpuBatchnormBackward3dNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward3dNcdhwFp16,
                         testing::ValuesIn(getBnBwdSmoke3dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward3dNcdhwBfp16);
TEST_P(IntegrationGpuBatchnormBackward3dNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward3dNcdhwBfp16,
                         testing::ValuesIn(getBnBwdSmoke3dTestCases()));

// ============================================================================
// NHWC 2D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward2dNhwcFp32);
TEST_P(IntegrationGpuBatchnormBackward2dNhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward2dNhwcFp32,
                         testing::ValuesIn(getBnBwdSmoke2dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward2dNhwcFp16);
TEST_P(IntegrationGpuBatchnormBackward2dNhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward2dNhwcFp16,
                         testing::ValuesIn(getBnBwdSmoke2dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward2dNhwcBfp16);
TEST_P(IntegrationGpuBatchnormBackward2dNhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward2dNhwcBfp16,
                         testing::ValuesIn(getBnBwdSmoke2dTestCases()));

// ============================================================================
// NDHWC 3D Tests
// ============================================================================

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward3dNdhwcFp32);
TEST_P(IntegrationGpuBatchnormBackward3dNdhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward3dNdhwcFp32,
                         testing::ValuesIn(getBnBwdSmoke3dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward3dNdhwcFp16);
TEST_P(IntegrationGpuBatchnormBackward3dNdhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward3dNdhwcFp16,
                         testing::ValuesIn(getBnBwdSmoke3dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackward3dNdhwcBfp16);
TEST_P(IntegrationGpuBatchnormBackward3dNdhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackward3dNdhwcBfp16,
                         testing::ValuesIn(getBnBwdSmoke3dTestCases()));
