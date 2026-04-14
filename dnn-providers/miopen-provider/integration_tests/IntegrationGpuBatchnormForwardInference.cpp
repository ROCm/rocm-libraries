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
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace miopen_plugin::test_utilities;
using namespace test_bn_common;

namespace
{

template <typename DataType, typename IntermediateType>
class BatchnormForwardInference
    : public IntegrationGraphVerificationHarness<DataType, BatchnormTestCase>
{
protected:
    void runGraphTest(float tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        const BatchnormTestCase& testCase = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormInferenceTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();
        graphObj.set_intermediate_data_type(intermediateDataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = makeTensorAttributes(
            "X", testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        // Channel-only tensors are layout-agnostic, specifying stride order is unnecessary
        auto meanAttr = makeTensorAttributes(
            "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = makeTensorAttributes(
            "inv_variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        auto scaleAttr = makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = makeTensorAttributes(
            "bias", intermediateDataType, derivedDims, generateStrides(derivedDims));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::BatchnormInferenceAttributes bnAttrs;

        auto yTensorAttr = graphObj.batchnorm_inference(xTensorAttr,
                                                        meanTensorAttr,
                                                        invVarianceTensorAttr,
                                                        scaleTensorAttr,
                                                        biasTensorAttr,
                                                        bnAttrs);

        yTensorAttr->set_output(true);

        this->registerValidator(yTensorAttr, tolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuBatchnormForwardInferenceNclFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNclBfp16 = BatchnormForwardInference<bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNclFp16 = BatchnormForwardInference<half, float>;

using IntegrationGpuBatchnormForwardInferenceNlcFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNlcBfp16 = BatchnormForwardInference<bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNlcFp16 = BatchnormForwardInference<half, float>;

using IntegrationGpuBatchnormForwardInferenceNchwFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNchwBfp16 = BatchnormForwardInference<bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNchwFp16 = BatchnormForwardInference<half, float>;

using IntegrationGpuBatchnormForwardInferenceNhwcFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNhwcBfp16 = BatchnormForwardInference<bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNhwcFp16 = BatchnormForwardInference<half, float>;

using IntegrationGpuBatchnormForwardInferenceNcdhwFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNcdhwBfp16
    = BatchnormForwardInference<bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNcdhwFp16 = BatchnormForwardInference<half, float>;

using IntegrationGpuBatchnormForwardInferenceNdhwcFp32 = BatchnormForwardInference<float, float>;

using IntegrationGpuBatchnormForwardInferenceNdhwcBfp16
    = BatchnormForwardInference<bfloat16, float>;

using IntegrationGpuBatchnormForwardInferenceNdhwcFp16 = BatchnormForwardInference<half, float>;

} // namespace

TEST_P(IntegrationGpuBatchnormForwardInferenceNclFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCL);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNclFp32,
                         testing::ValuesIn(getBnFwdInference1dTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNclFp32,
                         testing::ValuesIn(getBnFwdInference1dFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNclBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<bfloat16>(), TensorLayout::NCL);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNclBfp16,
                         testing::ValuesIn(getBnFwdInference1dTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNclBfp16,
                         testing::ValuesIn(getBnFwdInference1dFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNclFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCL);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNclFp16,
                         testing::ValuesIn(getBnFwdInference1dTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNclFp16,
                         testing::ValuesIn(getBnFwdInference1dFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNlcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NLC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNlcFp32,
                         testing::ValuesIn(getBnFwdInference1dTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNlcFp32,
                         testing::ValuesIn(getBnFwdInference1dFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNlcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<bfloat16>(), TensorLayout::NLC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNlcBfp16,
                         testing::ValuesIn(getBnFwdInference1dTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNlcBfp16,
                         testing::ValuesIn(getBnFwdInference1dFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNlcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NLC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNlcFp16,
                         testing::ValuesIn(getBnFwdInference1dTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNlcFp16,
                         testing::ValuesIn(getBnFwdInference1dFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwFp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNchwFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp32,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp32,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcBfp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcBfp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp16,
                         testing::ValuesIn(getBnFwdInferenceTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormForwardInferenceNhwcFp16,
                         testing::ValuesIn(getBnFwdInferenceFullTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNcdhwFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp32,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcBfp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));

TEST_P(IntegrationGpuBatchnormForwardInferenceNdhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceInference<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormForwardInferenceNdhwcFp16,
                         testing::ValuesIn(getBnFwdInference3dTestCases()));
