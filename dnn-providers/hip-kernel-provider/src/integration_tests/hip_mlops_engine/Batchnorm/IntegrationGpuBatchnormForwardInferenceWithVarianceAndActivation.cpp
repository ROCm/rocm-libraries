// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../IntegrationGraphVerificationHarness.hpp"
#include "../Common/ActivationCommon.hpp"
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

template <typename InputDataType,
          typename OutputDataType,
          typename ScaleDataType = float,
          typename MeanVarDataType = float,
          typename ComputeDataType = float>
class BatchnormForwardInferenceWithVarianceAndActivation
    : public IntegrationGraphVerificationHarness<InputDataType,
                                                 std::tuple<BatchnormTestCase, ActivTestCase>>
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

    void runGraphTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        const auto& [testCase, activeCase] = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceWithVarianceAndActivationTest");

        auto inputDataType = getDataTypeEnumFromType<InputDataType>();
        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(computeDataType)
            .set_io_data_type(inputDataType);

        auto xAttr = makeTensorAttributes(
            "X", inputDataType, testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanVarDataType = getDataTypeEnumFromType<MeanVarDataType>();
        auto meanAttr = makeTensorAttributes(
            "mean", meanVarDataType, derivedDims, generateStrides(derivedDims));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto varianceAttr = makeTensorAttributes(
            "variance", meanVarDataType, derivedDims, generateStrides(derivedDims));
        _varianceTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(varianceAttr));

        auto scaleDataType = getDataTypeEnumFromType<ScaleDataType>();
        auto scaleAttr = makeTensorAttributes(
            "scale", scaleDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = makeTensorAttributes(
            "bias", scaleDataType, derivedDims, generateStrides(derivedDims));
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
        auto outputDataType = getDataTypeEnumFromType<OutputDataType>();
        outTensorAttr->set_output(true);
        outTensorAttr->set_data_type(outputDataType);

        this->registerValidator(outTensorAttr, getToleranceInferenceWithVariance<OutputDataType>());

        this->verifyGraph(graphObj, testCase.seed);
    }

    std::shared_ptr<graph::TensorAttributes> _varianceTensorAttr;
};

// ============================================================================
// NCHW layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwUpcastBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwUpcastFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<half, float>;

// ============================================================================
// NHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcUpcastBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcUpcastFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<half, float>;

// ============================================================================
// NCDHW layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwUpcastBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwUpcastFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<half, float>;

// ============================================================================
// NDHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcUpcastBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcUpcastFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<half, float>;

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwUpcastBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwUpcastBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwUpcastFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwUpcastFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcUpcastBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcUpcastBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcUpcastFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcUpcastFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwUpcastBfp16,
       Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwUpcastBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwUpcastFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcUpcastBfp16,
       Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcUpcastBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcUpcastFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

} // hip_kernel_provider::batchnorm::test
