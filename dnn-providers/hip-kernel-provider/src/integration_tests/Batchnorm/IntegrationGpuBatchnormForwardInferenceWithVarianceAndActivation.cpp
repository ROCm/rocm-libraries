// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../Common/ActivationCommon.hpp"
#include "../IntegrationGraphVerificationHarness.hpp"
#include "BatchnormCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::batchnorm;
using namespace hip_kernel_provider::test_utilities;
using namespace hip_kernel_provider::test_activation_common;

namespace hip_kernel_provider::batchnorm::test
{

using namespace common;

namespace
{

template <typename DataType, typename IntermediateType, typename TestCaseType>
class BatchnormForwardInferenceWithVarianceAndActivation
    : public IntegrationGraphVerificationHarness<DataType, TestCaseType>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.sentinelFillOutputTensors();

        for(auto& tensorPair : bundle.tensors)
        {
            if(bundle.isOutput(tensorPair.first))
            {
                continue;
            }

            if(_varianceTensorAttr && tensorPair.first == _varianceTensorAttr->get_uid())
            {
                // Variance must be non-negative; use positive range
                bundle.randomizeTensor(tensorPair.first, 0.1f, 1.0f, seed);
            }
            else
            {
                bundle.randomizeTensor(tensorPair.first, -1.0f, 1.0f, seed);
            }
        }
    }

    void runGraphTest(float tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        const auto& [testCase, activeCase] = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceWithVarianceAndActivationTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = makeTensorAttributes(
            "X", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanAttr = makeTensorAttributes(
            "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto varianceAttr = makeTensorAttributes(
            "variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        _varianceTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(varianceAttr));

        auto scaleAttr = makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        // Epsilon (pass-by-value)
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_name("epsilon").set_value(1e-5);

        const graph::BatchnormInferenceAttributesVarianceExt bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference_variance_ext(xTensorAttr,
                                                                     meanTensorAttr,
                                                                     _varianceTensorAttr,
                                                                     scaleTensorAttr,
                                                                     biasTensorAttr,
                                                                     epsilonTensorAttr,
                                                                     bnAttrs);

        yTensorAttr->set_data_type(intermediateDataType);

        graph::PointwiseAttributes pointwiseAttrs;
        pointwiseAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activeCase.mode));
        if(activeCase.reluLowerClip.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip(activeCase.reluLowerClip.value());
        }
        if(activeCase.reluUpperClip.has_value())
        {
            pointwiseAttrs.set_relu_upper_clip(activeCase.reluUpperClip.value());
        }
        if(activeCase.reluLowerClipSlope.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip_slope(activeCase.reluLowerClipSlope.value());
        }
        if(activeCase.swishBeta.has_value())
        {
            pointwiseAttrs.set_swish_beta(activeCase.swishBeta.value());
        }
        if(activeCase.eluAlpha.has_value())
        {
            pointwiseAttrs.set_elu_alpha(activeCase.eluAlpha.value());
        }
        if(activeCase.softplusBeta.has_value())
        {
            pointwiseAttrs.set_softplus_beta(activeCase.softplusBeta.value());
        }

        auto outTensorAttr = graphObj.pointwise(yTensorAttr, pointwiseAttrs);
        outTensorAttr->set_output(true);

        this->registerValidator(outTensorAttr, tolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }

    std::shared_ptr<graph::TensorAttributes> _varianceTensorAttr;
};

// NCHW layouts
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

// NHWC layouts
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

// 5D layouts
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, ActivTestCase>>;

} // namespace

// ============================================================================
// NCHW FP32
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NCHW BFP16
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NCHW FP16
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NHWC FP32
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NHWC BFP16
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NHWC FP16
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

// ============================================================================
// NCDHW FP32 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NCDHW BFP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NCDHW FP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NDHWC FP32 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NDHWC BFP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

// ============================================================================
// NDHWC FP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16, Correctness)
{
    runGraphTest(getToleranceInferenceWithVariance<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

} // hip_kernel_provider::batchnorm::test
