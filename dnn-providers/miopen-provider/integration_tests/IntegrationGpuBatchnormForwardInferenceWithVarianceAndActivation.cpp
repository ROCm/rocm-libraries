// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_plugin::test_utilities;
using namespace test_bn_common;

namespace
{

template <typename DataType, typename IntermediateType, typename TestCaseType>
class BatchnormForwardInferenceWithVarianceAndActivation
    : public IntegrationGraphVerificationHarness<DataType, TestCaseType>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        // Fill output tensors with sentinel values
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

        graph::BatchnormInferenceAttributesVarianceExt bnAttrs;

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

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32
    = BatchnormForwardInferenceWithVarianceAndActivation<
        float,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        bfloat16,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16
    = BatchnormForwardInferenceWithVarianceAndActivation<
        half,
        float,
        std::tuple<BatchnormTestCase, test_activation_common::ActivTestCase>>;

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<float>(), TensorLayout::NCL);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NCL);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<half>(), TensorLayout::NCL);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNclFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<float>(), TensorLayout::NLC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NLC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<half>(), TensorLayout::NLC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNlcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference1dFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNchwFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

INSTANTIATE_TEST_SUITE_P(
    Full,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceFullTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationFullCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNcdhwFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcBfp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInferenceWithVariance<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormForwardInferenceWithVarianceAndActivationNdhwcFp16,
    testing::Combine(testing::ValuesIn(getBnFwdInference3dTestCases()),
                     testing::ValuesIn(test_activation_common::createFwdActivationSmokeCases())));
