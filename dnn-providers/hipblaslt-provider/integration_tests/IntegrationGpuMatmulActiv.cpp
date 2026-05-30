// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "IntegrationGpuMatmulBase.hpp"
#include "utils/ActivationUtils.hpp"
#include "utils/MatmulUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipblaslt_plugin::test_utilities;
using namespace test_matmul_common;
using namespace test_activation_common;

namespace
{

using TestParamsType = std::tuple<MatmulTestCase, ActivTestCase>;

template <typename DataType>
class IntegrationGpuMatmulActiv : public IntegrationGpuMatmulBase<DataType, TestParamsType>
{
protected:
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
        initGraph(const TestParamsType& testParams,
                  hipdnn_frontend::graph::Graph& graphObj) const override
    {
        const auto& [matmulParams, activParams] = testParams;

        auto aAttr = graph::makeTensorAttributes(
            "a",
            matmulParams.aDims,
            this->generateInputStrideOrder(matmulParams.aDims, matmulParams.transA));
        auto aTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(aAttr));

        auto bAttr = graph::makeTensorAttributes(
            "b",
            matmulParams.bDims,
            this->generateInputStrideOrder(matmulParams.bDims, matmulParams.transB));
        auto bTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(bAttr));

        graph::MatmulAttributes const matmulAttrs;
        auto cAttr = graphObj.matmul(aTensorAttr, bTensorAttr, matmulAttrs);

        graph::PointwiseAttributes activAttrs;
        activAttrs.set_mode(activParams.mode);
        if(activParams.reluLowerClip.has_value())
        {
            activAttrs.set_relu_lower_clip(activParams.reluLowerClip.value());
        }
        if(activParams.reluUpperClip.has_value())
        {
            activAttrs.set_relu_upper_clip(activParams.reluUpperClip.value());
        }
        if(activParams.swishBeta.has_value())
        {
            activAttrs.set_swish_beta(activParams.swishBeta.value());
        }

        return graphObj.pointwise(cAttr, activAttrs);
    }

    std::string getGraphName() const override
    {
        return "MatmulActivTest";
    }

    unsigned int getSeed(const TestParamsType& testParams) const override
    {
        return std::get<0>(testParams).seed;
    }
};

using IntegrationGpuMatmulActivFp32 = IntegrationGpuMatmulActiv<float>;
using IntegrationGpuMatmulActivFp16 = IntegrationGpuMatmulActiv<hipdnn_data_sdk::types::half>;
using IntegrationGpuMatmulActivBf16 = IntegrationGpuMatmulActiv<hipdnn_data_sdk::types::bfloat16>;

} // namespace

TEST_P(IntegrationGpuMatmulActivFp32, Correctness)
{
    runGraphTest(matmul::getTolerance<float>());
}

TEST_P(IntegrationGpuMatmulActivFp16, Correctness)
{
    runGraphTest(matmul::getTolerance<hipdnn_data_sdk::types::half>());
}

TEST_P(IntegrationGpuMatmulActivBf16, Correctness)
{
    runGraphTest(matmul::getTolerance<hipdnn_data_sdk::types::bfloat16>());
}

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulActiv,
                         IntegrationGpuMatmulActivFp32,
                         testing::Combine(testing::ValuesIn(getMatmulBiasActivTestCases()),
                                          testing::ValuesIn(createFwdActivationCases())));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulActiv,
                         IntegrationGpuMatmulActivFp16,
                         testing::Combine(testing::ValuesIn(getMatmulBiasActivTestCases()),
                                          testing::ValuesIn(createFwdActivationCases())));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulActiv,
                         IntegrationGpuMatmulActivBf16,
                         testing::Combine(testing::ValuesIn(getMatmulBiasActivTestCases()),
                                          testing::ValuesIn(createFwdActivationCases())));
