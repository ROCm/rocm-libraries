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

template <typename IODataType, typename ComputeDataType>
class RMSNormForwardTraining
    : public IntegrationGraphVerificationHarness<IODataType, RMSnormTestCase>
{
protected:
    void runGraphTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        const RMSnormTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("RMSnormTest");

        auto dataType = getDataTypeEnumFromType<IODataType>();
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = makeTensorAttributes(
            "X", dataType, testCase.ioDims, generateStrides(testCase.ioDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        auto scaleAttr = makeTensorAttributes(
            "scale", computeDataType, testCase.scaleDims, generateStrides(testCase.scaleDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        // type must match scale
        auto biasAttr = makeTensorAttributes(
            "bias", computeDataType, testCase.scaleDims, generateStrides(testCase.scaleDims));
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
        this->registerValidator(yTensorAttr, getTolerance<IODataType>());

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

// NCHW
using IntegrationGpuRMSnormForwardNchwFp32 = RMSNormForwardTraining<float, float>;
using IntegrationGpuRMSnormForwardNchwFp16 = RMSNormForwardTraining<half, float>;
using IntegrationGpuRMSnormForwardNchwBfp16 = RMSNormForwardTraining<bfloat16, float>;

// NCDHW layouts
using IntegrationGpuRMSnormForwardNcdhwFp32 = RMSNormForwardTraining<float, float>;
using IntegrationGpuRMSnormForwardNcdhwFp16 = RMSNormForwardTraining<half, float>;
using IntegrationGpuRMSnormForwardNcdhwBfp16 = RMSNormForwardTraining<bfloat16, float>;

}

// ============================================================================
// NCHW FP32
// ============================================================================

TEST_P(IntegrationGpuRMSnormForwardNchwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardNchwFp32,
                         testing::ValuesIn(getRMSnormTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardNchwFp32,
                         testing::ValuesIn(getRMSnormFullTestCases()));

// ============================================================================
// NCHW FP16
// ============================================================================

TEST_P(IntegrationGpuRMSnormForwardNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardNchwFp16,
                         testing::ValuesIn(getRMSnormTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardNchwFp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

// ============================================================================
// NCHW BFP16
// ============================================================================

TEST_P(IntegrationGpuRMSnormForwardNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardNchwBfp16,
                         testing::ValuesIn(getRMSnormTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormForwardNchwBfp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

// ============================================================================
// NCDHW FP32 (5D)
// ============================================================================

TEST_P(IntegrationGpuRMSnormForwardNcdhwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardNcdhwFp32,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

// ============================================================================
// NCDHW FP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuRMSnormForwardNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardNcdhwFp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));
// ============================================================================
// NCDHW BFP16 (5D)
// ============================================================================

TEST_P(IntegrationGpuRMSnormForwardNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormForwardNcdhwBfp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

} // namespace hip_kernel_provider::rmsnorm::test
