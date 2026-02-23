// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "IntegrationGpuMatmulBase.hpp"
#include "utils/MatmulUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipblaslt_plugin::test_utilities;
using namespace test_matmul_common;

namespace
{

template <typename DataType>
class IntegrationGpuMatmulBias : public IntegrationGpuMatmulBase<DataType, MatmulTestCase>
{
protected:
    virtual std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
        initGraph(const MatmulTestCase& testParams,
                  hipdnn_frontend::graph::Graph& graphObj) const override
    {
        auto aAttr = graph::makeTensorAttributes(
            "a",
            testParams.aDims,
            this->generateInputStrideOrder(testParams.aDims, testParams.transA));
        auto aTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(aAttr));

        auto bAttr = graph::makeTensorAttributes(
            "b",
            testParams.bDims,
            this->generateInputStrideOrder(testParams.bDims, testParams.transB));
        auto bTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(bAttr));

        graph::MatmulAttributes matmulAttrs;

        auto cAttr = graphObj.matmul(aTensorAttr, bTensorAttr, matmulAttrs);

        std::vector<int64_t> biasDims(testParams.cDims.size(), 1);
        biasDims[biasDims.size() - 1] = testParams.cDims.back();

        auto biasAttr = graph::makeTensorAttributes(
            "bias", biasDims, this->generateInputStrideOrder(biasDims, false));
        auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

        graph::PointwiseAttributes biasAttrs;
        biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);

        return graphObj.pointwise(cAttr, biasTensorAttr, biasAttrs);
    }

    virtual std::string getGraphName() const override
    {
        return "MatmulBiasTest";
    }

    virtual unsigned int getSeed(const MatmulTestCase& testParams) const override
    {
        return testParams.seed;
    }
};

using IntegrationGpuMatmulBiasFp32 = IntegrationGpuMatmulBias<float>;
using IntegrationGpuMatmulBiasFp16 = IntegrationGpuMatmulBias<hipdnn_data_sdk::types::half>;
using IntegrationGpuMatmulBiasBf16 = IntegrationGpuMatmulBias<hipdnn_data_sdk::types::bfloat16>;

} // namespace

TEST_P(IntegrationGpuMatmulBiasFp32, Correctness)
{
    runGraphTest(matmul::getTolerance<float>());
}

TEST_P(IntegrationGpuMatmulBiasFp16, Correctness)
{
    runGraphTest(matmul::getTolerance<hipdnn_data_sdk::types::half>());
}

TEST_P(IntegrationGpuMatmulBiasBf16, Correctness)
{
    runGraphTest(matmul::getTolerance<hipdnn_data_sdk::types::bfloat16>());
}

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulBias,
                         IntegrationGpuMatmulBiasFp32,
                         testing::ValuesIn(getMatmulBiasActivTestCases()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulBias,
                         IntegrationGpuMatmulBiasFp16,
                         testing::ValuesIn(getMatmulBiasActivTestCases()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulBias,
                         IntegrationGpuMatmulBiasBf16,
                         testing::ValuesIn(getMatmulBiasActivTestCases()));
