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
#include "IntegrationTestUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace test_bn_common;

namespace
{

template <typename DataType, typename IntermediateType, typename TestCaseType>
class BatchnormBackward : public GraphVerifierTest<DataType, TestCaseType>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const TestCaseType& testCase = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.getDims());

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormBackwardTest");

        int64_t uid = 1;

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        auto xAttr
            = graph::makeTensorAttributes("x",
                                          dataType,
                                          testCase.getDims(),
                                          generateStrides(testCase.getDims(), layout.strideOrder));
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto dyAttr
            = graph::makeTensorAttributes("dy",
                                          dataType,
                                          testCase.getDims(),
                                          generateStrides(testCase.getDims(), layout.strideOrder));
        dyAttr.set_uid(uid++);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto meanAttr = graph::makeTensorAttributes(
            "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
        meanAttr.set_uid(uid++);
        auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));

        auto invVarianceAttr = graph::makeTensorAttributes(
            "inv_variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
        invVarianceAttr.set_uid(uid++);
        auto invVarianceTensorAttr
            = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));

        graph::BatchnormBackwardAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_backward");
        bnAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);

        auto outputTensorsAttr
            = graphObj.batchnorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, bnAttrs);

        auto& dxTensorAttr = outputTensorsAttr[0];
        if(!dxTensorAttr->has_uid())
        {
            dxTensorAttr->set_uid(uid++);
        }
        dxTensorAttr->set_data_type(dataType);
        dxTensorAttr->set_dim(testCase.getDims());
        dxTensorAttr->set_stride(generateStrides(testCase.getDims(), layout.strideOrder));
        dxTensorAttr->set_output(true);

        auto& dscaleTensorAttr = outputTensorsAttr[1];
        if(!dscaleTensorAttr->has_uid())
        {
            dscaleTensorAttr->set_uid(uid++);
        }
        dscaleTensorAttr->set_data_type(intermediateDataType);
        dscaleTensorAttr->set_output(true);

        auto& dbiasTensorAttr = outputTensorsAttr[2];
        if(!dbiasTensorAttr->has_uid())
        {
            dbiasTensorAttr->set_uid(uid++);
        }
        dbiasTensorAttr->set_data_type(intermediateDataType);
        dbiasTensorAttr->set_output(true);

        auto intermediateTolerance = batchnorm::getToleranceBackward<IntermediateType>();

        this->registerValidator(dxTensorAttr->get_uid(),
                                createAllCloseValidator(toSdkType(dxTensorAttr->get_data_type()),
                                                        tolerance,
                                                        tolerance));

        this->registerValidator(
            dscaleTensorAttr->get_uid(),
            createAllCloseValidator(toSdkType(dscaleTensorAttr->get_data_type()),
                                    intermediateTolerance,
                                    intermediateTolerance));

        this->registerValidator(dbiasTensorAttr->get_uid(),
                                createAllCloseValidator(toSdkType(dbiasTensorAttr->get_data_type()),
                                                        intermediateTolerance,
                                                        intermediateTolerance));

        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuBatchnormBackwardNchwFp32
    = BatchnormBackward<float, float, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardNchwBfp16
    = BatchnormBackward<hip_bfloat16, float, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardNchwFp16 = BatchnormBackward<half, float, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardNhwcFp32
    = BatchnormBackward<float, float, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardNhwcBfp16
    = BatchnormBackward<hip_bfloat16, float, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardNhwcFp16 = BatchnormBackward<half, float, Batchnorm2dTestCase>;

using IntegrationGpuBatchnormBackwardNcdhwFp32
    = BatchnormBackward<float, float, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardNcdhwBfp16
    = BatchnormBackward<hip_bfloat16, float, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardNcdhwFp16
    = BatchnormBackward<half, float, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardNdhwcFp32
    = BatchnormBackward<float, float, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardNdhwcBfp16
    = BatchnormBackward<hip_bfloat16, float, Batchnorm3dTestCase>;

using IntegrationGpuBatchnormBackwardNdhwcFp16
    = BatchnormBackward<half, float, Batchnorm3dTestCase>;

} // namespace

TEST_P(IntegrationGpuBatchnormBackwardNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNchwFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNchwBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNchwFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNhwcFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197}
TEST_P(IntegrationGpuBatchnormBackwardNhwcBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNhwcBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

// MIOpen segfaults for this case, re-enable when fix is released:
// https://github.com/ROCm/rocm-libraries/pull/1197
TEST_P(IntegrationGpuBatchnormBackwardNhwcFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNhwcFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNcdhwFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNcdhwBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNcdhwFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNdhwcFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

// MIOpen may have issues with NDHWC layout for certain data types
TEST_P(IntegrationGpuBatchnormBackwardNdhwcBfp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNdhwcBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

// MIOpen may have issues with NDHWC layout for certain data types
TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp16, DISABLED_Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(,
                         IntegrationGpuBatchnormBackwardNdhwcFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));
