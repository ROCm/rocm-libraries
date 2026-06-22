// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../IntegrationGraphVerificationHarness.hpp"
#include "BatchnormCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::batchnorm;
using namespace hip_kernel_provider::test_utilities;

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
class BatchnormForwardInference
    : public IntegrationGraphVerificationHarness<InputDataType, BatchnormTestCase>
{
protected:
    void runGraphTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        const BatchnormTestCase& testCase = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceTest");

        auto inputDataType = getDataTypeEnumFromType<InputDataType>();
        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
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

        auto invVarianceAttr = makeTensorAttributes(
            "inv_variance", meanVarDataType, derivedDims, generateStrides(derivedDims));
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleDataType = getDataTypeEnumFromType<ScaleDataType>();
        auto scaleAttr = makeTensorAttributes(
            "scale", scaleDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = makeTensorAttributes(
            "bias", scaleDataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        const graph::BatchnormInferenceAttributes bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
                                                        bnAttrs);

        auto outputDataType = getDataTypeEnumFromType<OutputDataType>();
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(outputDataType);

        this->registerValidator(yTensorAttr, getToleranceInference<OutputDataType>());

        this->verifyGraph(graphObj, testCase.seed);
    }
};

// ============================================================================
// NCHW layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNchwFp32Fp32 = BatchnormForwardInference<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNchwBfp16Bfp16
    = BatchnormForwardInference<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNchwBfp16Fp32
    = BatchnormForwardInference<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNchwFp16Fp16 = BatchnormForwardInference<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNchwFp16Fp32 = BatchnormForwardInference<half, float>;

// ============================================================================
// NHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNhwcFp32Fp32 = BatchnormForwardInference<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNhwcBfp16Bfp16
    = BatchnormForwardInference<bfloat16, bfloat16>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNhwcFp16Fp16 = BatchnormForwardInference<half, half>;

// ============================================================================
// NCDHW layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNcdhwFp32Fp32
    = BatchnormForwardInference<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNcdhwBfp16Bfp16
    = BatchnormForwardInference<bfloat16, bfloat16>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNcdhwFp16Fp16 = BatchnormForwardInference<half, half>;

// ============================================================================
// NDHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNdhwcFp32Fp32
    = BatchnormForwardInference<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNdhwcBfp16Bfp16
    = BatchnormForwardInference<bfloat16, bfloat16>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNdhwcFp16Fp16 = BatchnormForwardInference<half, half>;

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwFp32Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwFp32Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwBfp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16Fp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16Fp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp32Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp32Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp16Fp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp16Fp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp32Fp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp16Fp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp32Fp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp16Fp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));
} // namespace hip_kernel_provider::batchnorm::test
