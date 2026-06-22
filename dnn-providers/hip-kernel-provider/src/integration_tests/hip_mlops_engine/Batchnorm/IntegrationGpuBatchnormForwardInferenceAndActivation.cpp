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
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32Fp32
    = BatchnormForwardInferenceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16Bfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16Fp32
    = BatchnormForwardInferenceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16Fp16
    = BatchnormForwardInferenceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16Fp32
    = BatchnormForwardInferenceAndActivation<half, float>;

// ============================================================================
// NHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32Fp32
    = BatchnormForwardInferenceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16Bfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16Fp32
    = BatchnormForwardInferenceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16Fp16
    = BatchnormForwardInferenceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16Fp32
    = BatchnormForwardInferenceAndActivation<half, float>;

// ============================================================================
// NCDHW layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp32Fp32
    = BatchnormForwardInferenceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16Bfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16Fp32
    = BatchnormForwardInferenceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16Fp16
    = BatchnormForwardInferenceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16Fp32
    = BatchnormForwardInferenceAndActivation<half, float>;

// ============================================================================
// NDHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp32Fp32
    = BatchnormForwardInferenceAndActivation<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16Bfp16
    = BatchnormForwardInferenceAndActivation<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16Fp32
    = BatchnormForwardInferenceAndActivation<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16Fp16
    = BatchnormForwardInferenceAndActivation<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16Fp32
    = BatchnormForwardInferenceAndActivation<half, float>;

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp32Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16Bfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16Bfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwBfp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16Fp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16Fp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNchwFp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp32Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16Bfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16Bfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcBfp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16Fp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16Fp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNhwcFp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                                          testing::ValuesIn(createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp32Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16Bfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwBfp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16Fp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNcdhwFp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp32Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16Bfp16,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcBfp16Fp32,
                         testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                                          testing::ValuesIn(createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16Fp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceAndActivationNdhwcFp16Fp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

} // namespace hip_kernel_provider::batchnorm::test
