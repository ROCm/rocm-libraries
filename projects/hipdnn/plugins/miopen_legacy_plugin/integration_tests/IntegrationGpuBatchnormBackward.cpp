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
using namespace miopen_legacy_plugin::test_utilities;
using namespace test_bn_common;

namespace
{

template <typename DataType, typename IntermediateType, bool ProvideMeanInvVar = true>
class BatchnormBackward : public IntegrationGraphVerificationHarness<DataType, BatchnormTestCase>
{
protected:
    std::unordered_map<graph::BatchnormBackwardAttributes::InputNames, int64_t> _inputTensorIds;

    void initializeBundle([[maybe_unused]] const graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        bundle.tensors.at(_inputTensorIds.at(graph::BatchnormBackwardAttributes::InputNames::X))
            ->fillTensorWithRandomValues(-1.0f, 1.0f, seed);

        bundle.tensors.at(_inputTensorIds.at(graph::BatchnormBackwardAttributes::InputNames::DY))
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);

        bundle.tensors
            .at(_inputTensorIds.at(graph::BatchnormBackwardAttributes::InputNames::SCALE))
            ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);

        if(ProvideMeanInvVar)
        {
            bundle.tensors
                .at(_inputTensorIds.at(graph::BatchnormBackwardAttributes::InputNames::MEAN))
                ->fillTensorWithRandomValues(-0.1f, 0.1f, seed);

            bundle.tensors
                .at(_inputTensorIds.at(
                    graph::BatchnormBackwardAttributes::InputNames::INV_VARIANCE))
                ->fillTensorWithRandomValues(1.9f, 2.0f, seed);
        }
    }

    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const BatchnormTestCase& testCase = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.dims);

        hipdnn_frontend::graph::Graph graphObj;

        graphObj.set_name("BatchnormBackwardTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        int64_t uid = 1;

        auto dataType = getDataTypeEnumFromType<DataType>();
        auto intermediateDataType = getDataTypeEnumFromType<IntermediateType>();

        auto xAttr = graph::makeTensorAttributes(
            "x", dataType, testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        xAttr.set_uid(uid++);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));
        _inputTensorIds.insert(
            {graph::BatchnormBackwardAttributes::InputNames::X, xTensorAttr->get_uid()});

        auto dyAttr = graph::makeTensorAttributes(
            "dy", dataType, testCase.dims, generateStrides(testCase.dims, layout.strideOrder));
        dyAttr.set_uid(uid++);
        auto dyTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(dyAttr));
        _inputTensorIds.insert(
            {graph::BatchnormBackwardAttributes::InputNames::DY, dyTensorAttr->get_uid()});

        auto scaleAttr = graph::makeTensorAttributes(
            "scale", intermediateDataType, derivedDims, generateStrides(derivedDims));
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));
        _inputTensorIds.insert(
            {graph::BatchnormBackwardAttributes::InputNames::SCALE, scaleTensorAttr->get_uid()});

        graph::BatchnormBackwardAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_backward");

        if(ProvideMeanInvVar)
        {
            auto meanAttr = graph::makeTensorAttributes(
                "mean", intermediateDataType, derivedDims, generateStrides(derivedDims));
            meanAttr.set_uid(uid++);
            auto meanTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(meanAttr));
            _inputTensorIds.insert(
                {graph::BatchnormBackwardAttributes::InputNames::MEAN, meanTensorAttr->get_uid()});

            auto invVarianceAttr = graph::makeTensorAttributes(
                "inv_variance", intermediateDataType, derivedDims, generateStrides(derivedDims));
            invVarianceAttr.set_uid(uid++);
            auto invVarianceTensorAttr
                = std::make_shared<graph::TensorAttributes>(std::move(invVarianceAttr));
            _inputTensorIds.insert({graph::BatchnormBackwardAttributes::InputNames::INV_VARIANCE,
                                    invVarianceTensorAttr->get_uid()});

            bnAttrs.set_saved_mean_and_inv_variance(meanTensorAttr, invVarianceTensorAttr);
        }

        auto outputTensorsAttr
            = graphObj.batchnorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, bnAttrs);

        auto& dxTensorAttr = outputTensorsAttr[0];
        if(!dxTensorAttr->has_uid())
        {
            dxTensorAttr->set_uid(uid++);
        }
        dxTensorAttr->set_data_type(dataType);
        dxTensorAttr->set_dim(testCase.dims);
        dxTensorAttr->set_stride(generateStrides(testCase.dims, layout.strideOrder));
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

        this->registerValidator(dxTensorAttr, tolerance);
        this->registerValidator(dscaleTensorAttr, intermediateTolerance);
        this->registerValidator(dbiasTensorAttr, intermediateTolerance);

        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuBatchnormBackwardNchwFp32 = BatchnormBackward<float, float>;

using IntegrationGpuBatchnormBackwardNchwBfp16 = BatchnormBackward<hip_bfloat16, float>;

using IntegrationGpuBatchnormBackwardNchwFp16 = BatchnormBackward<half, float>;

using IntegrationGpuBatchnormBackwardNhwcFp32 = BatchnormBackward<float, float>;

using IntegrationGpuBatchnormBackwardNhwcBfp16 = BatchnormBackward<hip_bfloat16, float>;

using IntegrationGpuBatchnormBackwardNhwcFp16 = BatchnormBackward<half, float>;

using IntegrationGpuBatchnormBackwardNcdhwFp32 = BatchnormBackward<float, float>;

using IntegrationGpuBatchnormBackwardNcdhwBfp16 = BatchnormBackward<hip_bfloat16, float>;

using IntegrationGpuBatchnormBackwardNcdhwFp16 = BatchnormBackward<half, float>;

using IntegrationGpuBatchnormBackwardNdhwcFp32 = BatchnormBackward<float, float>;

using IntegrationGpuBatchnormBackwardNdhwcBfp16 = BatchnormBackward<hip_bfloat16, float>;

using IntegrationGpuBatchnormBackwardNdhwcFp16 = BatchnormBackward<half, float>;

using IntegrationGpuBatchnormBackwardNchwFp32Optional = BatchnormBackward<float, float, false>;

using IntegrationGpuBatchnormBackwardNchwBfp16Optional
    = BatchnormBackward<hip_bfloat16, float, false>;

using IntegrationGpuBatchnormBackwardNchwFp16Optional = BatchnormBackward<half, float, false>;

using IntegrationGpuBatchnormBackwardNhwcFp32Optional = BatchnormBackward<float, float, false>;

using IntegrationGpuBatchnormBackwardNhwcBfp16Optional
    = BatchnormBackward<hip_bfloat16, float, false>;

using IntegrationGpuBatchnormBackwardNhwcFp16Optional = BatchnormBackward<half, float, false>;

using IntegrationGpuBatchnormBackwardNcdhwFp32Optional = BatchnormBackward<float, float, false>;

using IntegrationGpuBatchnormBackwardNcdhwBfp16Optional
    = BatchnormBackward<hip_bfloat16, float, false>;

using IntegrationGpuBatchnormBackwardNcdhwFp16Optional = BatchnormBackward<half, float, false>;

using IntegrationGpuBatchnormBackwardNdhwcFp32Optional = BatchnormBackward<float, float, false>;

using IntegrationGpuBatchnormBackwardNdhwcBfp16Optional
    = BatchnormBackward<hip_bfloat16, float, false>;

using IntegrationGpuBatchnormBackwardNdhwcFp16Optional = BatchnormBackward<half, float, false>;

} // namespace

TEST_P(IntegrationGpuBatchnormBackwardNchwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNchwFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNchwFp32,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNchwBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNchwBfp16,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNchwFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNchwFp16,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNhwcFp32,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNhwcFp32,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNhwcBfp16,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNhwcBfp16,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNhwcFp16,
                         testing::ValuesIn(getBnBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuBatchnormBackwardNhwcFp16,
                         testing::ValuesIn(getBnBwdFullTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNcdhwFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNcdhwBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNcdhwFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp32, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNdhwcFp32,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcBfp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNdhwcBfp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp16, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNdhwcFp16,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwFp32Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNchwFp32Optional,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwBfp16Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNchwBfp16Optional,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNchwFp16Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNchwFp16Optional,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcFp32Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNhwcFp32Optional,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcBfp16Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNhwcBfp16Optional,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNhwcFp16Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNhwcFp16Optional,
                         testing::ValuesIn(getBnBwdTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp32Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNcdhwFp32Optional,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwBfp16Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNcdhwBfp16Optional,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNcdhwFp16Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NCDHW);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNcdhwFp16Optional,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp32Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<float>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNdhwcFp32Optional,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcBfp16Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<hip_bfloat16>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNdhwcBfp16Optional,
                         testing::ValuesIn(getBnBwd3dTestCases()));

TEST_P(IntegrationGpuBatchnormBackwardNdhwcFp16Optional, Correctness)
{
    runGraphTest(batchnorm::getToleranceBackward<half>(), TensorLayout::NDHWC);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuBatchnormBackwardNdhwcFp16Optional,
                         testing::ValuesIn(getBnBwd3dTestCases()));
