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
using IntegrationGpuBatchnormForwardInferenceNchwFp32 = BatchnormForwardInference<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNchwBfp16
    = BatchnormForwardInference<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNchwUpcastBfp16
    = BatchnormForwardInference<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNchwFp16 = BatchnormForwardInference<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNchwUpcastFp16
    = BatchnormForwardInference<half, float>;

// ============================================================================
// NHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNhwcFp32 = BatchnormForwardInference<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNhwcBfp16
    = BatchnormForwardInference<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNhwcUpcastBfp16
    = BatchnormForwardInference<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNhwcFp16 = BatchnormForwardInference<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNhwcUpcastFp16
    = BatchnormForwardInference<half, float>;

// ============================================================================
// NCDHW layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNcdhwFp32 = BatchnormForwardInference<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNcdhwBfp16
    = BatchnormForwardInference<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNcdhwUpcastBfp16
    = BatchnormForwardInference<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNcdhwFp16 = BatchnormForwardInference<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNcdhwUpcastFp16
    = BatchnormForwardInference<half, float>;

// ============================================================================
// NDHWC layouts
// ============================================================================

// Input: float, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNdhwcFp32 = BatchnormForwardInference<float, float>;
// Input: bfloat16, Output: bfloat16, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNdhwcBfp16
    = BatchnormForwardInference<bfloat16, bfloat16>;
// Input: bfloat16, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNdhwcUpcastBfp16
    = BatchnormForwardInference<bfloat16, float>;
// Input: half, Output: half, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNdhwcFp16 = BatchnormForwardInference<half, half>;
// Input: half, Output: float, Scale: float, Mean: float, Compute: float
using IntegrationGpuBatchnormForwardInferenceNdhwcUpcastFp16
    = BatchnormForwardInference<half, float>;
} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwFp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwUpcastBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwUpcastBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwUpcastFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwUpcastFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcUpcastBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcUpcastBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcUpcastFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcUpcastFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwUpcastBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwUpcastFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcUpcastBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcUpcastFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));
} // namespace hip_kernel_provider::batchnorm::test
