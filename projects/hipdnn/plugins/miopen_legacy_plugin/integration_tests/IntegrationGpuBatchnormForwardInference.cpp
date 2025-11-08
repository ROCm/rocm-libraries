// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

#include "../tests/common/BatchnormCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace test_bn_common;

// Note: Tests temporarily disabled due to https://github.com/ROCm/rocm-libraries/issues/2459

namespace
{

template <typename DataType, typename IntermediateType>
class BatchnormForwardInference
    : public IntegrationGraphVerificationHarness<DataType, BatchnormTestCase>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const BatchnormTestCase& testCase = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        int64_t uid = 1;

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        auto xAttr = graph::makeTensorAttributes(
            "X", dataType, testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto meanAttr
            = graph::makeTensorAttributes("mean",
                                          intermediateDataType,
                                          derivedDims,
                                          generateStrides(derivedDims, layout.strideOrder));
        meanAttr.set_uid(uid++);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr
            = graph::makeTensorAttributes("inv_variance",
                                          intermediateDataType,
                                          derivedDims,
                                          generateStrides(derivedDims, layout.strideOrder));
        invVarianceAttr.set_uid(uid++);
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleAttr
            = graph::makeTensorAttributes("scale",
                                          intermediateDataType,
                                          derivedDims,
                                          generateStrides(derivedDims, layout.strideOrder));
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr
            = graph::makeTensorAttributes("bias",
                                          intermediateDataType,
                                          derivedDims,
                                          generateStrides(derivedDims, layout.strideOrder));
        biasAttr.set_uid(uid++);
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::BatchnormInferenceAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_inference");

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
                                                        bnAttrs);

        if(!yTensorAttr->has_uid())
        {
            yTensorAttr->set_uid(uid++);
        }

        yTensorAttr->set_data_type(dataType);
        yTensorAttr->set_dim(testCase.dims);
        yTensorAttr->set_stride(generateStrides(testCase.dims, layout.strideOrder));
        yTensorAttr->set_output(true);

        this->registerValidator(yTensorAttr, tolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuBatchnormForwardInferenceNchwFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNchwBfp16
    = BatchnormForwardInference<hip_bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNchwFp16 = BatchnormForwardInference<half, float>;

using IntegrationGpuBatchnormForwardInferenceNhwcFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNhwcBfp16
    = BatchnormForwardInference<hip_bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNhwcFp16 = BatchnormForwardInference<half, float>;

using IntegrationGpuBatchnormForwardInferenceNcdhwFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNcdhwBfp16
    = BatchnormForwardInference<hip_bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNcdhwFp16 = BatchnormForwardInference<half, float>;

using IntegrationGpuBatchnormForwardInferenceNdhwcFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNdhwcBfp16
    = BatchnormForwardInference<hip_bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNdhwcFp16 = BatchnormForwardInference<half, float>;

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp32, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwFp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp32, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp32, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp32, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));
