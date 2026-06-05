// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../IntegrationGraphVerificationHarness.hpp"
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
        auto scaleAttr
            = makeTensorAttributes("scale",
                                   scaleDataType,
                                   testCase.scaleDims,
                                   generateStrides(testCase.scaleDims, layout.strideOrder));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        // type must match scale
        auto biasAttr
            = makeTensorAttributes("bias",
                                   scaleDataType,
                                   testCase.scaleDims,
                                   generateStrides(testCase.scaleDims, layout.strideOrder));
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

// "Pure" = input, output, scale, compute all matching (scale/compute always FP32);
// "Upcast" = input is FP16/BFP16 but output widens to FP32.

// 1. Input: FP32, Scale: FP32, Output: FP32, Compute: FP32
using IntegrationGpuRMSnormForwardPureFp32 = RMSNormForwardTraining<float, float, float, float>;

// 2. Input: FP16, Scale: FP32, Output: FP16, Compute: FP32
using IntegrationGpuRMSnormForwardPureFp16 = RMSNormForwardTraining<half, float, half, float>;

// 3. Input: BFP16, Scale: FP32, Output: BFP16, Compute: FP32
using IntegrationGpuRMSnormForwardPureBfp16
    = RMSNormForwardTraining<bfloat16, float, bfloat16, float>;

// 4. Input: FP16, Scale: FP32, Output: FP32, Compute: FP32
using IntegrationGpuRMSnormForwardUpcastFp16 = RMSNormForwardTraining<half, float, float, float>;

// 5. Input: BFP16, Scale: FP32, Output: FP32, Compute: FP32
using IntegrationGpuRMSnormForwardUpcastBfp16
    = RMSNormForwardTraining<bfloat16, float, float, float>;
}

// ============================================================================
// Test Registrations
// ============================================================================

// Pure Fp32 -----------------------------------------------------------------

using IntegrationGpuRMSnormForwardPureNchwFp32 = IntegrationGpuRMSnormForwardPureFp32;
TEST_P(IntegrationGpuRMSnormForwardPureNchwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNchwFp32,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardPureNchwFp32,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardPureNhwcFp32 = IntegrationGpuRMSnormForwardPureFp32;
TEST_P(IntegrationGpuRMSnormForwardPureNhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNhwcFp32,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardPureNhwcFp32,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardPureNcdhwFp32 = IntegrationGpuRMSnormForwardPureFp32;
TEST_P(IntegrationGpuRMSnormForwardPureNcdhwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNcdhwFp32,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormForwardPureNdhwcFp32 = IntegrationGpuRMSnormForwardPureFp32;
TEST_P(IntegrationGpuRMSnormForwardPureNdhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNdhwcFp32,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

// Pure Fp16 -----------------------------------------------------------------

using IntegrationGpuRMSnormForwardPureNchwFp16 = IntegrationGpuRMSnormForwardPureFp16;
TEST_P(IntegrationGpuRMSnormForwardPureNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNchwFp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardPureNchwFp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardPureNhwcFp16 = IntegrationGpuRMSnormForwardPureFp16;
TEST_P(IntegrationGpuRMSnormForwardPureNhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNhwcFp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardPureNhwcFp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardPureNcdhwFp16 = IntegrationGpuRMSnormForwardPureFp16;
TEST_P(IntegrationGpuRMSnormForwardPureNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNcdhwFp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormForwardPureNdhwcFp16 = IntegrationGpuRMSnormForwardPureFp16;
TEST_P(IntegrationGpuRMSnormForwardPureNdhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNdhwcFp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

// Pure Bfp16 ----------------------------------------------------------------

using IntegrationGpuRMSnormForwardPureNchwBfp16 = IntegrationGpuRMSnormForwardPureBfp16;
TEST_P(IntegrationGpuRMSnormForwardPureNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNchwBfp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardPureNchwBfp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardPureNhwcBfp16 = IntegrationGpuRMSnormForwardPureBfp16;
TEST_P(IntegrationGpuRMSnormForwardPureNhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNhwcBfp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardPureNhwcBfp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardPureNcdhwBfp16 = IntegrationGpuRMSnormForwardPureBfp16;
TEST_P(IntegrationGpuRMSnormForwardPureNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNcdhwBfp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormForwardPureNdhwcBfp16 = IntegrationGpuRMSnormForwardPureBfp16;
TEST_P(IntegrationGpuRMSnormForwardPureNdhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardPureNdhwcBfp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

// Upcast Fp16 ---------------------------------------------------------------

using IntegrationGpuRMSnormForwardUpcastNchwFp16 = IntegrationGpuRMSnormForwardUpcastFp16;
TEST_P(IntegrationGpuRMSnormForwardUpcastNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardUpcastNchwFp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardUpcastNchwFp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardUpcastNhwcFp16 = IntegrationGpuRMSnormForwardUpcastFp16;
TEST_P(IntegrationGpuRMSnormForwardUpcastNhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardUpcastNhwcFp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardUpcastNhwcFp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardUpcastNcdhwFp16 = IntegrationGpuRMSnormForwardUpcastFp16;
TEST_P(IntegrationGpuRMSnormForwardUpcastNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardUpcastNcdhwFp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormForwardUpcastNdhwcFp16 = IntegrationGpuRMSnormForwardUpcastFp16;
TEST_P(IntegrationGpuRMSnormForwardUpcastNdhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardUpcastNdhwcFp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

// Upcast Bfp16 --------------------------------------------------------------

using IntegrationGpuRMSnormForwardUpcastNchwBfp16 = IntegrationGpuRMSnormForwardUpcastBfp16;
TEST_P(IntegrationGpuRMSnormForwardUpcastNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardUpcastNchwBfp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardUpcastNchwBfp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardUpcastNhwcBfp16 = IntegrationGpuRMSnormForwardUpcastBfp16;
TEST_P(IntegrationGpuRMSnormForwardUpcastNhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardUpcastNhwcBfp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardUpcastNhwcBfp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormForwardUpcastNcdhwBfp16 = IntegrationGpuRMSnormForwardUpcastBfp16;
TEST_P(IntegrationGpuRMSnormForwardUpcastNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardUpcastNcdhwBfp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormForwardUpcastNdhwcBfp16 = IntegrationGpuRMSnormForwardUpcastBfp16;
TEST_P(IntegrationGpuRMSnormForwardUpcastNdhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardUpcastNdhwcBfp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

} // namespace hip_kernel_provider::rmsnorm::test
