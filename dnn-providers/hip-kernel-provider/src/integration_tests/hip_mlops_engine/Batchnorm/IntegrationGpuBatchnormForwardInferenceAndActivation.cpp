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
class BatchnormForwardInferenceAndActivation
    : public IntegrationGraphVerificationHarness<InputDataType,
                                                 std::tuple<BatchnormTestCase, ActivTestCase>>
{
protected:
    void runGraphTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        const auto& [testCase, activeCase] = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceAndActivationTest");

        auto inputDataType = getDataTypeEnumFromType<InputDataType>();
        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        auto intermediateDataType = hipdnn_frontend::DataType::FLOAT;
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(computeDataType)
            .set_io_data_type(inputDataType);

        auto xAttr = graph::makeTensorAttributes(
            "X", inputDataType, testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanVarDataType = getDataTypeEnumFromType<MeanVarDataType>();
        auto meanAttr = graph::makeTensorAttributes(
            "mean", meanVarDataType, derivedDims, generateStrides(derivedDims));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = graph::makeTensorAttributes(
            "inv_variance", meanVarDataType, derivedDims, generateStrides(derivedDims));
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleDataType = getDataTypeEnumFromType<ScaleDataType>();
        auto scaleAttr = graph::makeTensorAttributes(
            "scale", scaleDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", scaleDataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        const graph::BatchnormInferenceAttributes bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
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

        this->registerValidator(outTensorAttr, getToleranceInference<OutputDataType>());

        this->verifyGraph(graphObj, testCase.seed);
    }
};

// ============================================================================
// NCHW layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32
    = BatchnormForwardInferenceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwUpcastBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16
    = BatchnormForwardInferenceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwUpcastFp16
    = BatchnormForwardInferenceAndActivation<half, float>;

// ============================================================================
// NHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32
    = BatchnormForwardInferenceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcUpcastBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16
    = BatchnormForwardInferenceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcUpcastFp16
    = BatchnormForwardInferenceAndActivation<half, float>;

// ============================================================================
// NCDHW layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp32
    = BatchnormForwardInferenceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwUpcastBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16
    = BatchnormForwardInferenceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwUpcastFp16
    = BatchnormForwardInferenceAndActivation<half, float>;

// ============================================================================
// NDHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp32
    = BatchnormForwardInferenceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcUpcastBfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16
    = BatchnormForwardInferenceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcUpcastFp16
    = BatchnormForwardInferenceAndActivation<half, float>;

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwUpcastBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwUpcastBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwUpcastFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwUpcastFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcUpcastBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcUpcastBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcUpcastFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcUpcastFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwUpcastBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwUpcastFp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcUpcastBfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcUpcastFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

} // namespace hip_kernel_provider::batchnorm::test
