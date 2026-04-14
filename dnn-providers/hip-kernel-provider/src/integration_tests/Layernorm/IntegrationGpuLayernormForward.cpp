// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
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

template <typename IoType>
class LayernormForward : public IntegrationGraphVerificationHarness<IoType, LayernormTestCase>
{
protected:
    void runGraphTest(float tolerance, const TensorLayout& layout = TensorLayout::NCHW)
    {
        const LayernormTestCase& testCase = this->GetParam();

        std::vector<int64_t> affineDims(testCase.dims.size(), 1);
        for(size_t i = testCase.normalizedDim; i < testCase.dims.size(); ++i)
        {
            affineDims[i] = testCase.dims[i];
        }

        graph::Graph graphObj;

        graphObj.set_name("LayernormFwdTest");

        auto dataType = getDataTypeEnumFromType<IoType>();
        graphObj.set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT)
            .set_io_data_type(dataType);

        auto ioStrides = generateStrides(testCase.dims, layout.strideOrder);
        auto affineStrides = generateStrides(affineDims, layout.strideOrder);

        auto xAttr = graph::makeTensorAttributes("X", testCase.dims, ioStrides);
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr = graph::makeTensorAttributes("scale", affineDims, affineStrides);
        auto scaleTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = graph::makeTensorAttributes("bias", affineDims, affineStrides);
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
            EXPECT_EQ(meanTensorAttr, nullptr) << "Mean tensor should be null for inference";
            EXPECT_EQ(invVarianceTensorAttr, nullptr)
                << "Inverse variance tensor should be null for inference";
        }

        yTensorAttr->set_output(true);
        if(testCase.weightBias)
        {
            meanTensorAttr->set_output(true);
            invVarianceTensorAttr->set_output(true);
        }

        this->registerValidator(yTensorAttr, tolerance);
        if(testCase.weightBias)
        {
            this->registerValidator(meanTensorAttr, tolerance);
            this->registerValidator(invVarianceTensorAttr, tolerance);
        }

        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuLayernormForwardNchwFp32 = LayernormForward<float>;
using IntegrationGpuLayernormForwardNchwFp16 = LayernormForward<half>;
using IntegrationGpuLayernormForwardNchwBfp16 = LayernormForward<bfloat16>;

using IntegrationGpuLayernormForwardNhwcFp32 = LayernormForward<float>;
using IntegrationGpuLayernormForwardNhwcFp16 = LayernormForward<half>;
using IntegrationGpuLayernormForwardNhwcBfp16 = LayernormForward<bfloat16>;

using IntegrationGpuLayernormForwardNcdhwFp32 = LayernormForward<float>;
using IntegrationGpuLayernormForwardNcdhwFp16 = LayernormForward<half>;
using IntegrationGpuLayernormForwardNcdhwBfp16 = LayernormForward<bfloat16>;

using IntegrationGpuLayernormForwardNdhwcFp32 = LayernormForward<float>;
using IntegrationGpuLayernormForwardNdhwcFp16 = LayernormForward<half>;
using IntegrationGpuLayernormForwardNdhwcBfp16 = LayernormForward<bfloat16>;

} // namespace

TEST_P(IntegrationGpuLayernormForwardNchwFp32, Correctness)
{
    runGraphTest(getTolerance<float>(), TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwFp32,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwFp32,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNchwFp16, Correctness)
{
    runGraphTest(getTolerance<half>(), TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwFp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwFp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNchwBfp16, Correctness)
{
    runGraphTest(getTolerance<bfloat16>(), TensorLayout::NCHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNchwBfp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNchwBfp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcFp32, Correctness)
{
    runGraphTest(getTolerance<float>(), TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcFp32,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcFp32,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcFp16, Correctness)
{
    runGraphTest(getTolerance<half>(), TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcFp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcFp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNhwcBfp16, Correctness)
{
    runGraphTest(getTolerance<bfloat16>(), TensorLayout::NHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNhwcBfp16,
                         testing::ValuesIn(getLayernormFwd4DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNhwcBfp16,
                         testing::ValuesIn(getLayernormFwd4DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwFp32, Correctness)
{
    runGraphTest(getTolerance<float>(), TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwFp32,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwFp32,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwFp16, Correctness)
{
    runGraphTest(getTolerance<half>(), TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwFp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwFp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNcdhwBfp16, Correctness)
{
    runGraphTest(getTolerance<bfloat16>(), TensorLayout::NCDHW);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNcdhwBfp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNcdhwBfp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcFp32, Correctness)
{
    runGraphTest(getTolerance<float>(), TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcFp32,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcFp32,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcFp16, Correctness)
{
    runGraphTest(getTolerance<half>(), TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcFp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcFp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

TEST_P(IntegrationGpuLayernormForwardNdhwcBfp16, Correctness)
{
    runGraphTest(getTolerance<bfloat16>(), TensorLayout::NDHWC);
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuLayernormForwardNdhwcBfp16,
                         testing::ValuesIn(getLayernormFwd5DSmokeTestCases()));
INSTANTIATE_TEST_SUITE_P(Full,
                         IntegrationGpuLayernormForwardNdhwcBfp16,
                         testing::ValuesIn(getLayernormFwd5DFullTestCases()));

} // namespace hip_kernel_provider::layernorm::test
