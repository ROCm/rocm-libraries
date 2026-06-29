// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../IntegrationGraphVerificationHarness.hpp"
#include "../Common/ActivationCommon.hpp"
#include "BatchnormCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;
using namespace hip_kernel_provider::batchnorm::test::common;
using namespace hipdnn_test_sdk::utilities::batchnorm;

namespace
{

struct BatchnormActivationTensorIds
{
    static constexpr int64_t X_UID = 1;
    static constexpr int64_t SCALE_UID = 2;
    static constexpr int64_t BIAS_UID = 3;
    static constexpr int64_t MEAN_UID = 4;
    static constexpr int64_t INV_VARIANCE_UID = 5;
    static constexpr int64_t DY_UID = 6;
};

template <typename InputType>
class BatchnormBackwardActivation
    : public IntegrationGraphVerificationHarness<InputType, BatchnormTestCase>
{
protected:
    void runGraphTest(const TensorLayout& layout)
    {
        const auto& testCase = this->GetParam();
        auto dims = testCase.dims;
        const std::vector<int64_t> channelDims = getDerivedShape(dims);

        graph::Graph graphObj;
        graphObj.set_name("BatchnormBackwardActivationTest");
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(getDataTypeEnumFromType<InputType>());

        auto xAttr
            = graph::makeTensorAttributes("x", dims, generateStrides(dims, layout.strideOrder));
        xAttr.set_uid(BatchnormActivationTensorIds::X_UID);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", hipdnn_frontend::DataType::FLOAT, channelDims, generateStrides(channelDims));
        scaleAttr.set_uid(BatchnormActivationTensorIds::SCALE_UID);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", hipdnn_frontend::DataType::FLOAT, channelDims, generateStrides(channelDims));
        biasAttr.set_uid(BatchnormActivationTensorIds::BIAS_UID);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto meanAttr = graph::makeTensorAttributes(
            "mean", hipdnn_frontend::DataType::FLOAT, channelDims, generateStrides(channelDims));
        meanAttr.set_uid(BatchnormActivationTensorIds::MEAN_UID);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarAttr = graph::makeTensorAttributes("inv_variance",
                                                      hipdnn_frontend::DataType::FLOAT,
                                                      channelDims,
                                                      generateStrides(channelDims));
        invVarAttr.set_uid(BatchnormActivationTensorIds::INV_VARIANCE_UID);
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarAttr));

        const graph::BatchnormInferenceAttributes bnInfAttrs;
        auto bnY = graphObj.batchnorm_inference(xTensorAttr,
                                                meanTensorAttr,
                                                invVarianceTensorAttr,
                                                scaleTensorAttr,
                                                biasTensorAttr,
                                                bnInfAttrs);

        auto dyAttr
            = graph::makeTensorAttributes("dy", dims, generateStrides(dims, layout.strideOrder));
        dyAttr.set_uid(BatchnormActivationTensorIds::DY_UID);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        graph::PointwiseAttributes activBwdAttrs;
        activBwdAttrs.set_mode(PointwiseMode::RELU_BWD);
        auto dxDrelu = graphObj.pointwise(dyTensorAttr, bnY, activBwdAttrs);

        graph::BatchnormBackwardAttributes bnBwdAttrs;
        bnBwdAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);
        auto bnBwdOuts
            = graphObj.batchnorm_backward(dxDrelu, xTensorAttr, scaleTensorAttr, bnBwdAttrs);

        auto& dxOut = bnBwdOuts[0];
        dxOut->set_output(true);
        auto& dscaleOut = bnBwdOuts[1];
        dscaleOut->set_data_type(hipdnn_frontend::DataType::FLOAT);
        dscaleOut->set_output(true);
        auto& dbiasOut = bnBwdOuts[2];
        dbiasOut->set_data_type(hipdnn_frontend::DataType::FLOAT);
        dbiasOut->set_output(true);

        auto tolerance = getToleranceTraining<InputType>();
        this->registerValidator(dxOut, tolerance);
        this->registerValidator(dscaleOut, tolerance);
        this->registerValidator(dbiasOut, tolerance);
        this->verifyGraph(graphObj, testCase.seed);
    }

    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.tensors.at(BatchnormActivationTensorIds::X_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);
        bundle.tensors.at(BatchnormActivationTensorIds::DY_UID)
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed + 1);
        bundle.tensors.at(BatchnormActivationTensorIds::SCALE_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed + 2);
        bundle.tensors.at(BatchnormActivationTensorIds::BIAS_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed + 3);
        bundle.tensors.at(BatchnormActivationTensorIds::MEAN_UID)
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed + 4);
        bundle.tensors.at(BatchnormActivationTensorIds::INV_VARIANCE_UID)
            ->fillTensorWithRandomValues(1.9f, 2.0f, seed + 5);
    }
};

// NCHW 2D
using IntegrationGpuBatchnormBackwardActivationNchwFp32 = BatchnormBackwardActivation<float>;
using IntegrationGpuBatchnormBackwardActivationNchwFp16 = BatchnormBackwardActivation<half>;
using IntegrationGpuBatchnormBackwardActivationNchwBfp16 = BatchnormBackwardActivation<bfloat16>;

// NHWC 2D
using IntegrationGpuBatchnormBackwardActivationNhwcFp32 = BatchnormBackwardActivation<float>;
using IntegrationGpuBatchnormBackwardActivationNhwcFp16 = BatchnormBackwardActivation<half>;
using IntegrationGpuBatchnormBackwardActivationNhwcBfp16 = BatchnormBackwardActivation<bfloat16>;

// NCDHW 3D
using IntegrationGpuBatchnormBackwardActivationNcdhwFp32 = BatchnormBackwardActivation<float>;
using IntegrationGpuBatchnormBackwardActivationNcdhwFp16 = BatchnormBackwardActivation<half>;
using IntegrationGpuBatchnormBackwardActivationNcdhwBfp16 = BatchnormBackwardActivation<bfloat16>;

// NDHWC 3D
using IntegrationGpuBatchnormBackwardActivationNdhwcFp32 = BatchnormBackwardActivation<float>;
using IntegrationGpuBatchnormBackwardActivationNdhwcFp16 = BatchnormBackwardActivation<half>;
using IntegrationGpuBatchnormBackwardActivationNdhwcBfp16 = BatchnormBackwardActivation<bfloat16>;

} // namespace

// ============================================================================
// NCHW 2D Tests
// ============================================================================
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNchwFp32);
TEST_P(IntegrationGpuBatchnormBackwardActivationNchwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNchwFp32,
                         testing::ValuesIn(getBnFwdTrainingSmoke2dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNchwFp16);
TEST_P(IntegrationGpuBatchnormBackwardActivationNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNchwFp16,
                         testing::ValuesIn(getBnFwdTrainingSmoke2dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNchwBfp16);
TEST_P(IntegrationGpuBatchnormBackwardActivationNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNchwBfp16,
                         testing::ValuesIn(getBnFwdTrainingSmoke2dTestCases()));

// ============================================================================
// NHWC 2D Tests
// ============================================================================
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNhwcFp32);
TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNhwcFp32,
                         testing::ValuesIn(getBnFwdTrainingSmoke2dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNhwcFp16);
TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNhwcFp16,
                         testing::ValuesIn(getBnFwdTrainingSmoke2dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNhwcBfp16);
TEST_P(IntegrationGpuBatchnormBackwardActivationNhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNhwcBfp16,
                         testing::ValuesIn(getBnFwdTrainingSmoke2dTestCases()));

// ============================================================================
// NCDHW 3D Tests
// ============================================================================
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNcdhwFp32);
TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNcdhwFp32,
                         testing::ValuesIn(getBnFwdTrainingSmoke3dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNcdhwFp16);
TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNcdhwFp16,
                         testing::ValuesIn(getBnFwdTrainingSmoke3dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNcdhwBfp16);
TEST_P(IntegrationGpuBatchnormBackwardActivationNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNcdhwBfp16,
                         testing::ValuesIn(getBnFwdTrainingSmoke3dTestCases()));

// ============================================================================
// NDHWC 3D Tests
// ============================================================================
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNdhwcFp32);
TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNdhwcFp32,
                         testing::ValuesIn(getBnFwdTrainingSmoke3dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNdhwcFp16);
TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNdhwcFp16,
                         testing::ValuesIn(getBnFwdTrainingSmoke3dTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuBatchnormBackwardActivationNdhwcBfp16);
TEST_P(IntegrationGpuBatchnormBackwardActivationNdhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardActivationNdhwcBfp16,
                         testing::ValuesIn(getBnFwdTrainingSmoke3dTestCases()));
