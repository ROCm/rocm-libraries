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

#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_legacy_plugin::test_utilities;
using namespace test_bn_common;

namespace
{

template <typename DataType, typename IntermediateType>
class BatchnormForwardInferenceWithVariance
    : public IntegrationGraphVerificationHarness<DataType, BatchnormTestCase>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const BatchnormTestCase& testCase = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceWithVarianceTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "X", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto varianceAttr = graph::makeTensorAttributes(
            "variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto varianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(varianceAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        // Epsilon (pass-by-value)
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>();
        epsilonTensorAttr->set_name("epsilon").set_value(1e-5f);

        graph::BatchnormInferenceAttributesVarianceExt bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference_variance_ext(xTensorAttr,
                                                                     meanTensorAttr,
                                                                     varianceTensorAttr,
                                                                     scaleTensorAttr,
                                                                     biasTensorAttr,
                                                                     epsilonTensorAttr,
                                                                     bnAttrs);

        yTensorAttr->set_output(true);

        this->registerValidator(yTensorAttr, tolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp32
    = BatchnormForwardInferenceWithVariance<float, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16
    = BatchnormForwardInferenceWithVariance<hip_bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16
    = BatchnormForwardInferenceWithVariance<half, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp32
    = BatchnormForwardInferenceWithVariance<float, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16
    = BatchnormForwardInferenceWithVariance<hip_bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16
    = BatchnormForwardInferenceWithVariance<half, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp32
    = BatchnormForwardInferenceWithVariance<float, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwBfp16
    = BatchnormForwardInferenceWithVariance<hip_bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp16
    = BatchnormForwardInferenceWithVariance<half, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp32
    = BatchnormForwardInferenceWithVariance<float, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcBfp16
    = BatchnormForwardInferenceWithVariance<hip_bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp16
    = BatchnormForwardInferenceWithVariance<half, float>;

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));
