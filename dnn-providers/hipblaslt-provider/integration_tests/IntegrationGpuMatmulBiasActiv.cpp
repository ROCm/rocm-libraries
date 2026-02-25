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
class IntegrationGpuMatmulBiasActiv : public IntegrationGpuMatmulBase<DataType, TestParamsType>
{
protected:
    virtual std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
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

        graph::MatmulAttributes matmulAttrs;
        auto cAttr = graphObj.matmul(aTensorAttr, bTensorAttr, matmulAttrs);

        std::vector<int64_t> biasDims(matmulParams.cDims.size(), 1);
        biasDims[biasDims.size() - 1] = matmulParams.cDims.back();

        auto biasAttr = graph::makeTensorAttributes(
            "bias", biasDims, this->generateInputStrideOrder(biasDims, false));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::PointwiseAttributes biasAttrs;
        biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);

        auto cBiasAttr = graphObj.pointwise(cAttr, biasTensorAttr, biasAttrs);

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

        return graphObj.pointwise(cBiasAttr, activAttrs);
    }

    virtual std::string getGraphName() const override
    {
        return "MatmulBiasActivTest";
    }

    virtual unsigned int getSeed(const TestParamsType& testParams) const override
    {
        return std::get<0>(testParams).seed;
    }
};

using IntegrationGpuMatmulBiasActivFp32 = IntegrationGpuMatmulBiasActiv<float>;
using IntegrationGpuMatmulBiasActivFp16
    = IntegrationGpuMatmulBiasActiv<hipdnn_data_sdk::types::half>;
using IntegrationGpuMatmulBiasActivBf16
    = IntegrationGpuMatmulBiasActiv<hipdnn_data_sdk::types::bfloat16>;

} // namespace

TEST_P(IntegrationGpuMatmulBiasActivFp32, Correctness)
{
    runGraphTest(matmul::getTolerance<float>());
}

TEST_P(IntegrationGpuMatmulBiasActivFp16, Correctness)
{
    runGraphTest(matmul::getTolerance<hipdnn_data_sdk::types::half>());
}

TEST_P(IntegrationGpuMatmulBiasActivBf16, Correctness)
{
    runGraphTest(matmul::getTolerance<hipdnn_data_sdk::types::bfloat16>());
}

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulBiasActiv,
                         IntegrationGpuMatmulBiasActivFp32,
                         testing::Combine(testing::ValuesIn(getMatmulBiasActivTestCases()),
                                          testing::ValuesIn(createFwdActivationCases())));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulBiasActiv,
                         IntegrationGpuMatmulBiasActivFp16,
                         testing::Combine(testing::ValuesIn(getMatmulBiasActivTestCases()),
                                          testing::ValuesIn(createFwdActivationCases())));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulBiasActiv,
                         IntegrationGpuMatmulBiasActivBf16,
                         testing::Combine(testing::ValuesIn(getMatmulBiasActivTestCases()),
                                          testing::ValuesIn(createFwdActivationCases())));
