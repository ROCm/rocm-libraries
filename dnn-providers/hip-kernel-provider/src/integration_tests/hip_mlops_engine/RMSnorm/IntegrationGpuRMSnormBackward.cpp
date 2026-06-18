// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
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

template <typename DyDataType,
          typename XDataType,
          typename ScaleDataType,
          typename DxDataType,
          typename ComputeDataType>
class RMSNormBackward : public IntegrationGraphVerificationHarness<XDataType, RMSnormTestCase>
{
protected:
    void runGraphTest(const TensorLayout& layout = TensorLayout::NCHW)
    {
        const RMSnormTestCase& testCase = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("RMSnormTest");

        auto dyDataType = getDataTypeEnumFromType<DyDataType>();
        auto xDataType = getDataTypeEnumFromType<XDataType>();
        auto computeDataType = getDataTypeEnumFromType<ComputeDataType>();
        graphObj.set_compute_data_type(computeDataType)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

        auto dyAttr = makeTensorAttributes("dy",
                                           dyDataType,
                                           testCase.ioDims,
                                           generateStrides(testCase.ioDims, layout.strideOrder));
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));
        auto xAttr = makeTensorAttributes(
            "x", xDataType, testCase.ioDims, generateStrides(testCase.ioDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        auto scaleDataType = getDataTypeEnumFromType<ScaleDataType>();
        auto scaleAttr
            = makeTensorAttributes("scale",
                                   scaleDataType,
                                   testCase.scaleDims,
                                   generateStrides(testCase.scaleDims, layout.strideOrder));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));
        auto invRmsDims = testCase.ioDims;
        for(size_t i = 1; i < invRmsDims.size(); ++i)
        {
            if(testCase.scaleDims[i] != 1)
            {
                invRmsDims[i] = 1;
            }
        }
        auto invRmsAttr = makeTensorAttributes("inv_rms",
                                               computeDataType,
                                               invRmsDims,
                                               generateStrides(invRmsDims, layout.strideOrder));
        auto invRmsTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(invRmsAttr));

        graph::RMSNormBackwardAttributes rmsnormBwdAttrs;
        rmsnormBwdAttrs.set_name("rmsnorm_bwd");
        rmsnormBwdAttrs.set_compute_data_type(computeDataType);
        rmsnormBwdAttrs.set_compute_dbias(true);

        auto [dxTensorAttr, dscaleTensorAttr, dbiasTensorAttr] = graphObj.rmsnorm_backward(
            dyTensorAttr, xTensorAttr, scaleTensorAttr, invRmsTensorAttr, rmsnormBwdAttrs);

        dxTensorAttr->set_output(true);
        auto dxDataType = getDataTypeEnumFromType<DxDataType>();
        dxTensorAttr->set_data_type(dxDataType);
        this->registerValidator(dxTensorAttr, getTolerance<DxDataType>());

        dscaleTensorAttr->set_output(true);
        dscaleTensorAttr->set_data_type(scaleDataType);
        this->registerValidator(dscaleTensorAttr, getTolerance<ScaleDataType>());

        dbiasTensorAttr->set_output(true);
        dbiasTensorAttr->set_data_type(scaleDataType);
        this->registerValidator(dbiasTensorAttr, getTolerance<ScaleDataType>());

        this->verifyGraph(graphObj, testCase.seed);
    }
};

// ============================================================================
// Test cases
// ============================================================================

// "Pure" = Both forward inputs (X) and backward gradients (Dy, Dx) with matching precision (scale/compute always FP32).
// "Mixed" = Gradients (Dy) are lower precision (FP16/BF16) while forward inputs (X) are kept at higher precision (FP32).

// 1. Input (Dy): FP32, X: FP32, Scale: FP32, Output (Dx): FP32, Compute: FP32
using IntegrationGpuRMSnormBackwardPureFp32 = RMSNormBackward<float, float, float, float, float>;

// 2. Input (Dy): FP16, X: FP16, Scale: FP32, Output (Dx): FP16, Compute: FP32
using IntegrationGpuRMSnormBackwardPureFp16 = RMSNormBackward<half, half, float, half, float>;

// 3. Input (Dy): BFP16, X: BFP16, Scale: FP32, Output (Dx): BFP16, Compute: FP32
using IntegrationGpuRMSnormBackwardPureBfp16
    = RMSNormBackward<bfloat16, bfloat16, float, bfloat16, float>;

// 4. Input (Dy): FP16, X: FP32, Scale: FP32, Output (Dx): FP32, Compute: FP32
using IntegrationGpuRMSnormBackwardMixedFp16 = RMSNormBackward<half, float, float, float, float>;

// 5. Input (Dy): BFP16, X: FP32, Scale: FP32, Output (Dx): FP32, Compute: FP32
using IntegrationGpuRMSnormBackwardMixedBfp16
    = RMSNormBackward<bfloat16, float, float, float, float>;
}

// ============================================================================
// Test Registrations
// ============================================================================

// Pure Fp32 -----------------------------------------------------------------

using IntegrationGpuRMSnormBackwardPureNchwFp32 = IntegrationGpuRMSnormBackwardPureFp32;
TEST_P(IntegrationGpuRMSnormBackwardPureNchwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNchwFp32,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardPureNchwFp32,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardPureNhwcFp32 = IntegrationGpuRMSnormBackwardPureFp32;
TEST_P(IntegrationGpuRMSnormBackwardPureNhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNhwcFp32,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardPureNhwcFp32,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardPureNcdhwFp32 = IntegrationGpuRMSnormBackwardPureFp32;
TEST_P(IntegrationGpuRMSnormBackwardPureNcdhwFp32, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNcdhwFp32,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormBackwardPureNdhwcFp32 = IntegrationGpuRMSnormBackwardPureFp32;
TEST_P(IntegrationGpuRMSnormBackwardPureNdhwcFp32, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNdhwcFp32,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

// Pure Fp16 -----------------------------------------------------------------

using IntegrationGpuRMSnormBackwardPureNchwFp16 = IntegrationGpuRMSnormBackwardPureFp16;
TEST_P(IntegrationGpuRMSnormBackwardPureNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNchwFp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardPureNchwFp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardPureNhwcFp16 = IntegrationGpuRMSnormBackwardPureFp16;
TEST_P(IntegrationGpuRMSnormBackwardPureNhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNhwcFp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardPureNhwcFp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardPureNcdhwFp16 = IntegrationGpuRMSnormBackwardPureFp16;
TEST_P(IntegrationGpuRMSnormBackwardPureNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNcdhwFp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormBackwardPureNdhwcFp16 = IntegrationGpuRMSnormBackwardPureFp16;
TEST_P(IntegrationGpuRMSnormBackwardPureNdhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNdhwcFp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

// Pure Bfp16 ----------------------------------------------------------------

using IntegrationGpuRMSnormBackwardPureNchwBfp16 = IntegrationGpuRMSnormBackwardPureBfp16;
TEST_P(IntegrationGpuRMSnormBackwardPureNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNchwBfp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardPureNchwBfp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardPureNhwcBfp16 = IntegrationGpuRMSnormBackwardPureBfp16;
TEST_P(IntegrationGpuRMSnormBackwardPureNhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNhwcBfp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardPureNhwcBfp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardPureNcdhwBfp16 = IntegrationGpuRMSnormBackwardPureBfp16;
TEST_P(IntegrationGpuRMSnormBackwardPureNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNcdhwBfp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormBackwardPureNdhwcBfp16 = IntegrationGpuRMSnormBackwardPureBfp16;
TEST_P(IntegrationGpuRMSnormBackwardPureNdhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardPureNdhwcBfp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

// Mixed Fp16 ---------------------------------------------------------------

using IntegrationGpuRMSnormBackwardMixedNchwFp16 = IntegrationGpuRMSnormBackwardMixedFp16;
TEST_P(IntegrationGpuRMSnormBackwardMixedNchwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardMixedNchwFp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardMixedNchwFp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardMixedNhwcFp16 = IntegrationGpuRMSnormBackwardMixedFp16;
TEST_P(IntegrationGpuRMSnormBackwardMixedNhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardMixedNhwcFp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardMixedNhwcFp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardMixedNcdhwFp16 = IntegrationGpuRMSnormBackwardMixedFp16;
TEST_P(IntegrationGpuRMSnormBackwardMixedNcdhwFp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardMixedNcdhwFp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormBackwardMixedNdhwcFp16 = IntegrationGpuRMSnormBackwardMixedFp16;
TEST_P(IntegrationGpuRMSnormBackwardMixedNdhwcFp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardMixedNdhwcFp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

// Mixed Bfp16 --------------------------------------------------------------

using IntegrationGpuRMSnormBackwardMixedNchwBfp16 = IntegrationGpuRMSnormBackwardMixedBfp16;
TEST_P(IntegrationGpuRMSnormBackwardMixedNchwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardMixedNchwBfp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardMixedNchwBfp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardMixedNhwcBfp16 = IntegrationGpuRMSnormBackwardMixedBfp16;
TEST_P(IntegrationGpuRMSnormBackwardMixedNhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardMixedNhwcBfp16,
                         testing::ValuesIn(getRMSnormTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuRMSnormBackwardMixedNhwcBfp16,
                         testing::ValuesIn(getRMSnormFullTestCases()));

using IntegrationGpuRMSnormBackwardMixedNcdhwBfp16 = IntegrationGpuRMSnormBackwardMixedBfp16;
TEST_P(IntegrationGpuRMSnormBackwardMixedNcdhwBfp16, Correctness)
{
    runGraphTest(TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardMixedNcdhwBfp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

using IntegrationGpuRMSnormBackwardMixedNdhwcBfp16 = IntegrationGpuRMSnormBackwardMixedBfp16;
TEST_P(IntegrationGpuRMSnormBackwardMixedNdhwcBfp16, Correctness)
{
    runGraphTest(TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuRMSnormBackwardMixedNdhwcBfp16,
                         testing::ValuesIn(getRMSnorm3dTestCases()));

} // namespace hip_kernel_provider::rmsnorm::test
