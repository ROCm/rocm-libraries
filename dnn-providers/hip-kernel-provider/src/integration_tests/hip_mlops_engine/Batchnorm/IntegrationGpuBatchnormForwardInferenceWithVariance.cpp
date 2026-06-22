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
class BatchnormForwardInferenceWithVariance
    : public IntegrationGraphVerificationHarness<InputDataType, BatchnormTestCase>
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
        const BatchnormTestCase& testCase = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceWithVarianceTest");

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

        auto outputDataType = getDataTypeEnumFromType<OutputDataType>();
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(outputDataType);

        this->registerValidator(yTensorAttr, getToleranceInferenceWithVariance<OutputDataType>());

        this->verifyGraph(graphObj, testCase.seed);
    }

    std::shared_ptr<graph::TensorAttributes> _varianceTensorAttr;
};

// ============================================================================
// NCHW layouts
// ============================================================================

using IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp32Fp32
    = BatchnormForwardInferenceWithVariance<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16Bfp16
    = BatchnormForwardInferenceWithVariance<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16Fp32
    = BatchnormForwardInferenceWithVariance<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16Fp16
    = BatchnormForwardInferenceWithVariance<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16Fp32
    = BatchnormForwardInferenceWithVariance<half, float>;

// ============================================================================
// NHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp32Fp32
    = BatchnormForwardInferenceWithVariance<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16Bfp16
    = BatchnormForwardInferenceWithVariance<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16Fp32
    = BatchnormForwardInferenceWithVariance<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16Fp16
    = BatchnormForwardInferenceWithVariance<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16Fp32
    = BatchnormForwardInferenceWithVariance<half, float>;

// ============================================================================
// NCDHW layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp32Fp32
    = BatchnormForwardInferenceWithVariance<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwBfp16Bfp16
    = BatchnormForwardInferenceWithVariance<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwBfp16Fp32
    = BatchnormForwardInferenceWithVariance<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp16Fp16
    = BatchnormForwardInferenceWithVariance<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp16Fp32
    = BatchnormForwardInferenceWithVariance<half, float>;

// ============================================================================
// NDHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp32Fp32
    = BatchnormForwardInferenceWithVariance<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcBfp16Bfp16
    = BatchnormForwardInferenceWithVariance<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcBfp16Fp32
    = BatchnormForwardInferenceWithVariance<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp16Fp16
    = BatchnormForwardInferenceWithVariance<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp16Fp32
    = BatchnormForwardInferenceWithVariance<half, float>;
} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp32Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp32Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwBfp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16Fp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16Fp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNchwFp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp32Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp32Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcBfp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16Fp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16Fp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNhwcFp16Fp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp32Fp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwBfp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwBfp16Fp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp16Fp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNcdhwFp16Fp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp32Fp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp32Fp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcBfp16Bfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcBfp16Bfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcBfp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcBfp16Fp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp16Fp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp16Fp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp16Fp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceWithVarianceNdhwcFp16Fp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

} // namespace hip_kernel_provider::batchnorm::test
