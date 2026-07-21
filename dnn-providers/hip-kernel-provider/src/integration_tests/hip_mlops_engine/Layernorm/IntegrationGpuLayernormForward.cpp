// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../IntegrationGraphVerificationHarness.hpp"
#include "LayernormCommon.hpp"
#include "hipdnn_data_sdk/utilities/Tensor.hpp"
#include "hipdnn_frontend/Types.hpp"
#include "hipdnn_frontend/attributes/LayernormAttributes.hpp"
#include "hipdnn_frontend/attributes/TensorAttributes.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::layernorm;
using namespace hip_kernel_provider::test_utilities;

namespace hip_kernel_provider::layernorm::test
{

using namespace common;

namespace
{

template <typename InputDataType,
          typename OutputDataType,
          typename ScaleBiasDataType,
          typename MeanInvVarianceDataType>
class LayernormForward
    : public IntegrationGraphVerificationHarness<InputDataType, LayernormTestCase>
{
protected:
    void runGraphTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        const LayernormTestCase& testCase = this->GetParam();

        std::vector<int64_t> affineDims(testCase.dims.size(), 1);
        for(size_t i = testCase.normalizedDim; i < testCase.dims.size(); ++i)
        {
            affineDims[i] = testCase.dims[i];
        }

        graph::Graph graphObj;

        graphObj.set_name("LayernormFwdTest");

        auto inputDataType = getDataTypeEnumFromType<InputDataType>();
        graphObj.set_intermediate_data_type(DataType::FLOAT).set_compute_data_type(DataType::FLOAT);

        auto ioStrides = generateStrides(testCase.dims, layout.strideOrder);
        auto affineStrides = generateStrides(affineDims, layout.strideOrder);

        auto xAttr = graph::makeTensorAttributes("X", inputDataType, testCase.dims, ioStrides);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleBiasDataType = getDataTypeEnumFromType<ScaleBiasDataType>();
        auto scaleAttr
            = graph::makeTensorAttributes("scale", scaleBiasDataType, affineDims, affineStrides);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr
            = graph::makeTensorAttributes("bias", scaleBiasDataType, affineDims, affineStrides);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto epsilonAttr = graph::makeTensorAttributes("epsilon", 1e-5f);
        auto epsilonTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(epsilonAttr));

        graph::LayernormAttributes lnAttrs;
        lnAttrs.set_epsilon(std::move(epsilonTensorAttr));
        lnAttrs.set_forward_phase(testCase.weightBias ? NormFwdPhase::TRAINING
                                                      : NormFwdPhase::INFERENCE);

        auto results = graphObj.layernorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, lnAttrs);
        const auto& yTensorAttr = results[0];
        const auto& meanTensorAttr = results[1];
        const auto& invVarianceTensorAttr = results[2];

        if(!testCase.weightBias)
        {
            if(!std::is_same_v<InputDataType, MeanInvVarianceDataType>)
            {
                GTEST_SKIP() << "\nSkipping since the CPU reference implementation does not work "
                                "properly for this test case.";
            }
            EXPECT_EQ(meanTensorAttr, nullptr) << "Mean tensor should be null for inference";
            EXPECT_EQ(invVarianceTensorAttr, nullptr)
                << "Inverse variance tensor should be null for inference";
        }

        auto outputDataType = getDataTypeEnumFromType<OutputDataType>();
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(outputDataType);
        if(testCase.weightBias)
        {
            auto meanInvVarianceDataType = getDataTypeEnumFromType<MeanInvVarianceDataType>();
            meanTensorAttr->set_output(true);
            meanTensorAttr->set_data_type(meanInvVarianceDataType);
            invVarianceTensorAttr->set_output(true);
            invVarianceTensorAttr->set_data_type(meanInvVarianceDataType);
        }

        this->registerValidator(yTensorAttr, getTolerance<OutputDataType>());
        if(testCase.weightBias)
        {
            this->registerValidator(meanTensorAttr, getTolerance<MeanInvVarianceDataType>());
            this->registerValidator(invVarianceTensorAttr, getTolerance<MeanInvVarianceDataType>());
        }

        this->verifyGraph(graphObj, testCase.seed);
    }
};

// ============================================================================
// Test cases
// ============================================================================

// "Pure" = input, output, scale/bias, and mean/invvariance all the same type.
// "Mixed" = input/output same type (FP16/BFP16), but scale/bias and mean/invvariance are FP32.
// "Upcast" = input is FP16/BFP16 but output widens to FP32 (scale/bias and
// mean/invvariance are FP32).

// ============================================================================
// NCHW
// ============================================================================

// 1. Input: FLOAT, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNchwPureFp32 = LayernormForward<float, float, float, float>;

// 2. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: HALF
using IntegrationGpuLayernormForwardNchwMixedFp16 = LayernormForward<half, half, float, float>;

// 3. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: BFLOAT16
using IntegrationGpuLayernormForwardNchwMixedBfp16
    = LayernormForward<bfloat16, bfloat16, float, float>;

// 4. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNchwUpcastFp16 = LayernormForward<half, float, float, float>;

// 5. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNchwUpcastBfp16
    = LayernormForward<bfloat16, float, float, float>;

// 6. Input: HALF, ScaleBias: HALF, MeanInvVariance: HALF, Output: HALF
using IntegrationGpuLayernormForwardNchwPureFp16 = LayernormForward<half, half, half, half>;

// 7. Input: BFLOAT16, ScaleBias: BFLOAT16, MeanInvVariance: BFLOAT16, Output: BFLOAT16
using IntegrationGpuLayernormForwardNchwPureBfp16
    = LayernormForward<bfloat16, bfloat16, bfloat16, bfloat16>;

// ============================================================================
// NHWC
// ============================================================================

// 1. Input: FLOAT, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNhwcPureFp32 = LayernormForward<float, float, float, float>;

// 2. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: HALF
using IntegrationGpuLayernormForwardNhwcMixedFp16 = LayernormForward<half, half, float, float>;

// 3. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: BFLOAT16
using IntegrationGpuLayernormForwardNhwcMixedBfp16
    = LayernormForward<bfloat16, bfloat16, float, float>;

// 4. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNhwcUpcastFp16 = LayernormForward<half, float, float, float>;

// 5. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNhwcUpcastBfp16
    = LayernormForward<bfloat16, float, float, float>;

// 6. Input: HALF, ScaleBias: HALF, MeanInvVariance: HALF, Output: HALF
using IntegrationGpuLayernormForwardNhwcPureFp16 = LayernormForward<half, half, half, half>;

// 7. Input: BFLOAT16, ScaleBias: BFLOAT16, MeanInvVariance: BFLOAT16, Output: BFLOAT16
using IntegrationGpuLayernormForwardNhwcPureBfp16
    = LayernormForward<bfloat16, bfloat16, bfloat16, bfloat16>;

// ============================================================================
// NCDHW
// ============================================================================

// 1. Input: FLOAT, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNcdhwPureFp32 = LayernormForward<float, float, float, float>;

// 2. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: HALF
using IntegrationGpuLayernormForwardNcdhwMixedFp16 = LayernormForward<half, half, float, float>;

// 3. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: BFLOAT16
using IntegrationGpuLayernormForwardNcdhwMixedBfp16
    = LayernormForward<bfloat16, bfloat16, float, float>;

// 4. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNcdhwUpcastFp16 = LayernormForward<half, float, float, float>;

// 5. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNcdhwUpcastBfp16
    = LayernormForward<bfloat16, float, float, float>;

// 6. Input: HALF, ScaleBias: HALF, MeanInvVariance: HALF, Output: HALF
using IntegrationGpuLayernormForwardNcdhwPureFp16 = LayernormForward<half, half, half, half>;

// 7. Input: BFLOAT16, ScaleBias: BFLOAT16, MeanInvVariance: BFLOAT16, Output: BFLOAT16
using IntegrationGpuLayernormForwardNcdhwPureBfp16
    = LayernormForward<bfloat16, bfloat16, bfloat16, bfloat16>;

// ============================================================================
// NDHWC
// ============================================================================

// 1. Input: FLOAT, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNdhwcPureFp32 = LayernormForward<float, float, float, float>;

// 2. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: HALF
using IntegrationGpuLayernormForwardNdhwcMixedFp16 = LayernormForward<half, half, float, float>;

// 3. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: BFLOAT16
using IntegrationGpuLayernormForwardNdhwcMixedBfp16
    = LayernormForward<bfloat16, bfloat16, float, float>;

// 4. Input: HALF, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNdhwcUpcastFp16 = LayernormForward<half, float, float, float>;

// 5. Input: BFLOAT16, ScaleBias: FLOAT, MeanInvVariance: FLOAT, Output: FLOAT
using IntegrationGpuLayernormForwardNdhwcUpcastBfp16
    = LayernormForward<bfloat16, float, float, float>;

// 6. Input: HALF, ScaleBias: HALF, MeanInvVariance: HALF, Output: HALF
using IntegrationGpuLayernormForwardNdhwcPureFp16 = LayernormForward<half, half, half, half>;

// 7. Input: BFLOAT16, ScaleBias: BFLOAT16, MeanInvVariance: BFLOAT16, Output: BFLOAT16
using IntegrationGpuLayernormForwardNdhwcPureBfp16
    = LayernormForward<bfloat16, bfloat16, bfloat16, bfloat16>;

} // namespace

// ============================================================================
// Test Registrations
// ============================================================================

TEST_P(IntegrationGpuLayernormForwardNchwPureFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwPureFp32,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwPureFp32,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNchwMixedFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwMixedFp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwMixedFp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNchwMixedBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwMixedBfp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwMixedBfp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNchwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwUpcastFp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwUpcastFp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNchwUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwUpcastBfp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwUpcastBfp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNchwPureFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwPureFp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwPureFp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNchwPureBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwPureBfp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwPureBfp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcPureFp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcPureFp32,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcPureFp32,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcMixedFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcMixedFp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcMixedFp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcMixedBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcMixedBfp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcMixedBfp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcUpcastFp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcUpcastFp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcUpcastBfp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcUpcastBfp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcPureFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcPureFp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcPureFp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcPureBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcPureBfp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcPureBfp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwPureFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwPureFp32,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwPureFp32,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwMixedFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwMixedFp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwMixedFp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwMixedBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwMixedBfp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwMixedBfp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwUpcastFp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwUpcastFp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwUpcastBfp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwUpcastBfp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwPureFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwPureFp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwPureFp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwPureBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwPureBfp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwPureBfp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcPureFp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcPureFp32,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcPureFp32,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcMixedFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcMixedFp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcMixedFp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcMixedBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcMixedBfp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcMixedBfp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcUpcastFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcUpcastFp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcUpcastFp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcUpcastBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcUpcastBfp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcUpcastBfp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcPureFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcPureFp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcPureFp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcPureBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcPureBfp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcPureBfp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

} // namespace hip_kernel_provider::layernorm::test
