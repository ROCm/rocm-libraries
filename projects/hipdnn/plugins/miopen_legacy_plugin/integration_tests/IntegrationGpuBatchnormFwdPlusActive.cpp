// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestTolerances.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

#include "../tests/common/ActivationCommon.hpp"
#include "../tests/common/BatchnormCommon.hpp"

#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace test_bn_common;

namespace
{

template <typename DataType, typename IntermediateType, typename TestCaseType>
class BatchnormFwdPlusActiv : public IntegrationGraphVerificationHarness<DataType, TestCaseType>
{
protected:
    void runGraphTest(DataType tolerance, const TensorLayout& layout = TensorLayout::NCHW) override
    {
        const auto& [testCase, activeCase] = this->GetParam();

        auto derivedDims = getDerivedShape(testCase.getDims());

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("BatchnormFwd+ActivTest");
        graphObj.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

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
        setTensorAttributeDetails(yTensorAttr, uid, dataType, testCase.getDims(), layout, false);

        graph::PointwiseAttributes pointwiseAttrs;
        pointwiseAttrs.set_name("activation");
        pointwiseAttrs.set_mode(static_cast<hipdnn_frontend::PointwiseMode>(activeCase.mode));
        if(activeCase.reluLowerClip.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip(activeCase.reluLowerClip.value());
        }
        if(activeCase.reluUpperClip.has_value())
        {
            pointwiseAttrs.set_relu_upper_clip(activeCase.reluUpperClip.value());
        }
        if(activeCase.reluLowerClipSlope.has_value())
        {
            pointwiseAttrs.set_relu_lower_clip_slope(activeCase.reluLowerClipSlope.value());
        }
        if(activeCase.swishBeta.has_value())
        {
            pointwiseAttrs.set_swish_beta(activeCase.swishBeta.value());
        }
        if(activeCase.eluAlpha.has_value())
        {
            pointwiseAttrs.set_elu_alpha(activeCase.eluAlpha.value());
        }
        if(activeCase.softplusBeta.has_value())
        {
            pointwiseAttrs.set_softplus_beta(activeCase.softplusBeta.value());
        }

        auto outTensorAttr = graphObj.pointwise(yTensorAttr, pointwiseAttrs);
        setTensorAttributeDetails(outTensorAttr, uid, dataType, testCase.getDims(), layout, true);

        this->registerValidator(outTensorAttr, tolerance);
        this->verifyGraph(graphObj, testCase.seed);
    }

    void setTensorAttributeDetails(std::shared_ptr<graph::TensorAttributes>& tensorAttr,
                                   int64_t& uid,
                                   hipdnn_frontend::DataType dataType,
                                   const std::vector<int64_t>& dims,
                                   const TensorLayout& layout,
                                   bool isOutput)
    {
        tensorAttr->set_data_type(dataType);
        tensorAttr->set_dim(dims);
        tensorAttr->set_stride(generateStrides(dims, layout.strideOrder));
        tensorAttr->set_output(isOutput);
        if(!tensorAttr->has_uid())
        {
            tensorAttr->set_uid(uid++);
        }
    }
};

using IntegrationGpuBatchnormFwdPlusActivNchwFp32
    = BatchnormFwdPlusActiv<float,
                            float,
                            std::tuple<Batchnorm2dTestCase, test_activation_common::ActivTestCase>>;
using IntegrationGpuBatchnormFwdPlusActivNcdhwFp32
    = BatchnormFwdPlusActiv<float,
                            float,
                            std::tuple<Batchnorm2dTestCase, test_activation_common::ActivTestCase>>;

// using IntegrationGpuBatchnormFwdPlusActivNchwBfp16 = BatchnormFwdPlusActiv<hip_bfloat16>;
// using IntegrationGpuBatchnormFwdPlusActivNcdhwBfp16 = BatchnormFwdPlusActiv<hip_bfloat16>;

// using IntegrationGpuBatchnormFwdPlusActivNchwFp16 = BatchnormFwdPlusActiv<half>;
// using IntegrationGpuBatchnormFwdPlusActivNcdhwFp16 = BatchnormFwdPlusActiv<half>;

// using IntegrationGpuBatchnormFwdPlusActivNhwcFp32 = BatchnormFwdPlusActiv<float>;
// using IntegrationGpuBatchnormFwdPlusActivNdhwcFp32 = BatchnormFwdPlusActiv<float>;

// using IntegrationGpuBatchnormFwdPlusActivNhwcBfp16 = BatchnormFwdPlusActiv<hip_bfloat16>;
// using IntegrationGpuBatchnormFwdPlusActivNdhwcBfp16 = BatchnormFwdPlusActiv<hip_bfloat16>;

// using IntegrationGpuBatchnormFwdPlusActivNhwcFp16 = BatchnormFwdPlusActiv<half>;
// using IntegrationGpuBatchnormFwdPlusActivNdhwcFp16 = BatchnormFwdPlusActiv<half>;

} // namespace

TEST_P(IntegrationGpuBatchnormFwdPlusActivNchwFp32, Correctness)
{
    runGraphTest(4e-6f, TensorLayout::NCHW);
}

// TEST_P(IntegrationGpuConvFwdNcdhwFp32, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NCDHW);
// }

// TEST_P(IntegrationGpuConvFwdNchwBfp16, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCHW);
// }

// TEST_P(IntegrationGpuConvFwdNcdhwBfp16, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NCDHW);
// }

// TEST_P(IntegrationGpuConvFwdNchwFp16, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NCHW);
// }

// TEST_P(IntegrationGpuConvFwdNcdhwFp16, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NCDHW);
// }

// TEST_P(IntegrationGpuConvFwdNhwcFp32, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NHWC);
// }

// TEST_P(IntegrationGpuConvFwdNdhwcFp32, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<float>(), TensorLayout::NDHWC);
// }

// TEST_P(IntegrationGpuConvFwdNhwcBfp16, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NHWC);
// }

// TEST_P(IntegrationGpuConvFwdNdhwcBfp16, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<hip_bfloat16>(), TensorLayout::NDHWC);
// }

// TEST_P(IntegrationGpuConvFwdNhwcFp16, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NHWC);
// }

// TEST_P(IntegrationGpuConvFwdNdhwcFp16, Correctness)
// {
//     runGraphTest(conv::getToleranceFwd<half>(), TensorLayout::NDHWC);
// }

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormFwdPlusActivNchwFp32,
    testing::Combine(testing::ValuesIn(getBnFwdInferenceTestCases()),
                     testing::ValuesIn(test_activation_common::getActivationTestCases())));

// INSTANTIATE_TEST_SUITE_P(Full,
//                          IntegrationGpuBatchnormFwdPlusActivNchwFp32,
//                          testing::ValuesIn(getBnFwdInferenceFullTestCases()));

// INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNchwBfp16, testing::ValuesIn(getConvTestCases4D()));

// INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNchwFp16, testing::ValuesIn(getConvTestCases4D()));

// INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNhwcFp32, testing::ValuesIn(getConvTestCases4D()));

// INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNhwcBfp16, testing::ValuesIn(getConvTestCases4D()));

// INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNhwcFp16, testing::ValuesIn(getConvTestCases4D()));

// INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNcdhwFp32, testing::ValuesIn(getConvTestCases5D()));

// INSTANTIATE_TEST_SUITE_P(,
//                          IntegrationGpuConvFwdNcdhwBfp16,
//                          testing::ValuesIn(getConvTestCases5D()));

// INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNcdhwFp16, testing::ValuesIn(getConvTestCases5D()));

// INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNdhwcFp32, testing::ValuesIn(getConvTestCases5D()));

// INSTANTIATE_TEST_SUITE_P(,
//                          IntegrationGpuConvFwdNdhwcBfp16,
//                          testing::ValuesIn(getConvTestCases5D()));

// INSTANTIATE_TEST_SUITE_P(, IntegrationGpuConvFwdNdhwcFp16, testing::ValuesIn(getConvTestCases5D()));
