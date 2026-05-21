// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "RMSnormCommon.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::rmsnorm;
using namespace hip_kernel_provider::test_utilities;

namespace hip_kernel_provider::rmsnorm::test
{
using namespace common;

namespace
{

template <typename InputDataType,
          typename ScaleDataType,
          typename OutputDataType,
          typename ComputeDataType>
class RMSNormForwardTraining
    : public IntegrationGraphVerificationHarness<InputDataType, RMSnormTestCase>
{
protected:
    void runGraphTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        const RMSnormTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("RMSnormTest");

        auto inputDataType = getDataTypeEnumFromType<InputDataType>();
        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        graphObj.set_compute_data_type(computeDataType)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(inputDataType);

        auto xAttr = makeTensorAttributes("X",
                                          inputDataType,
                                          testCase.ioDims,
                                          generateStrides(testCase.ioDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleDataType = getDataTypeEnumFromType<ScaleDataType>();
        auto scaleAttr = makeTensorAttributes(
            "scale", scaleDataType, testCase.scaleDims, generateStrides(testCase.scaleDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        // type must match scale
        auto biasAttr = makeTensorAttributes(
            "bias", scaleDataType, testCase.scaleDims, generateStrides(testCase.scaleDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        auto epsilon = std::make_shared<TensorAttributes>(1e-5f);

        graph::RMSNormAttributes rmsnormAttrs;
        rmsnormAttrs.set_epsilon(epsilon);
        rmsnormAttrs.set_bias(biasTensorAttr);
        rmsnormAttrs.set_forward_phase(testCase.isTraining ? NormFwdPhase::TRAINING
                                                           : NormFwdPhase::INFERENCE);

        auto [yTensorAttr, invRMSAttr]
            = graphObj.rmsnorm(xTensorAttr, scaleTensorAttr, rmsnormAttrs);

        yTensorAttr->set_output(true);
        auto outputDataType = getDataTypeEnumFromType<OutputDataType>();
        yTensorAttr->set_data_type(outputDataType);
        this->registerValidator(yTensorAttr, getTolerance<OutputDataType>());

        if(testCase.isTraining)
        {
            invRMSAttr->set_output(true);
            invRMSAttr->set_data_type(computeDataType);
            this->registerValidator(invRMSAttr, getTolerance<ComputeDataType>());
        }
        else
        {
            EXPECT_EQ(invRMSAttr, nullptr)
                << "Inverse RMS output tensor should be null for inference";
        }

        this->verifyGraph(graphObj, testCase.seed);
    }
};

// ============================================================================
// Test cases
// ============================================================================

// 1. Input: FP32, Scale: FP32, Output: FP32, Compute: FP32
using IntegrationGpuRMSnormForwardFp32Fp32Fp32Fp32
    = RMSNormForwardTraining<float, float, float, float>;

// 2. Input: FP16, Scale: FP32, Output: FP16, Compute: FP32
using IntegrationGpuRMSnormForwardFp16Fp32Fp16Fp32
    = RMSNormForwardTraining<half, float, half, float>;

// 3. Input: BFP16, Scale: FP32, Output: BFP16, Compute: FP32
using IntegrationGpuRMSnormForwardBfp16Fp32Bfp16Fp32
    = RMSNormForwardTraining<bfloat16, float, bfloat16, float>;

// 4. Input: FP16, Scale: FP32, Output: FP32, Compute: FP32
using IntegrationGpuRMSnormForwardFp16Fp32Fp32Fp32
    = RMSNormForwardTraining<half, float, float, float>;

// 5. Input: BFP16, Scale: FP32, Output: FP32, Compute: FP32
using IntegrationGpuRMSnormForwardBfp16Fp32Fp32Fp32
    = RMSNormForwardTraining<bfloat16, float, float, float>;
}

// ============================================================================
// Test Registrations
// ============================================================================

// Register tests with different data types and layouts
#define REGISTER_RMS_TEST(TestCase)                                                               \
    /* --- NCHW --- */                                                                            \
    using TestCase##NCHW = TestCase;                                                              \
    TEST_P(TestCase##NCHW, Correctness)                                                           \
    {                                                                                             \
        runGraphTest(TensorLayout::NCHW);                                                         \
    }                                                                                             \
    INSTANTIATE_TEST_SUITE_P(Smoke, TestCase##NCHW, testing::ValuesIn(getRMSnormTestCases()));    \
    INSTANTIATE_TEST_SUITE_P(Full, TestCase##NCHW, testing::ValuesIn(getRMSnormFullTestCases())); \
    /* --- NCDHW --- */                                                                           \
    using TestCase##NCDHW = TestCase;                                                             \
    TEST_P(TestCase##NCDHW, Correctness)                                                          \
    {                                                                                             \
        runGraphTest(TensorLayout::NCDHW);                                                        \
    }                                                                                             \
    INSTANTIATE_TEST_SUITE_P(Smoke, TestCase##NCDHW, testing::ValuesIn(getRMSnorm3dTestCases()));

REGISTER_RMS_TEST(IntegrationGpuRMSnormForwardFp32Fp32Fp32Fp32);
REGISTER_RMS_TEST(IntegrationGpuRMSnormForwardFp16Fp32Fp16Fp32);
REGISTER_RMS_TEST(IntegrationGpuRMSnormForwardBfp16Fp32Bfp16Fp32);
REGISTER_RMS_TEST(IntegrationGpuRMSnormForwardFp16Fp32Fp32Fp32);
REGISTER_RMS_TEST(IntegrationGpuRMSnormForwardBfp16Fp32Fp32Fp32);

} // namespace hip_kernel_provider::rmsnorm::test
