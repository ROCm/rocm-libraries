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
class IntegrationGpuMatmul : public IntegrationGpuMatmulBase<DataType, MatmulTestCase>
{
protected:
    std::shared_ptr<graph::TensorAttributes>
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

        graph::MatmulAttributes const matmulAttrs;

        return graphObj.matmul(aTensorAttr, bTensorAttr, matmulAttrs);
    };

    std::string getGraphName() const override
    {
        return "MatmulTest";
    }

    unsigned int getSeed(const MatmulTestCase& testParams) const override
    {
        return testParams.seed;
    }
};

using IntegrationGpuMatmulFp32 = IntegrationGpuMatmul<float>;
using IntegrationGpuMatmulFp16 = IntegrationGpuMatmul<hipdnn_data_sdk::types::half>;
using IntegrationGpuMatmulBf16 = IntegrationGpuMatmul<hipdnn_data_sdk::types::bfloat16>;

} // namespace

TEST_P(IntegrationGpuMatmulFp32, Correctness)
{
    runGraphTest(matmul::getTolerance<float>());
}

TEST_P(IntegrationGpuMatmulFp16, Correctness)
{
    runGraphTest(matmul::getTolerance<hipdnn_data_sdk::types::half>());
}

TEST_P(IntegrationGpuMatmulBf16, Correctness)
{
    runGraphTest(matmul::getTolerance<hipdnn_data_sdk::types::bfloat16>());
}

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmul,
                         IntegrationGpuMatmulFp32,
                         testing::ValuesIn(getMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmul,
                         IntegrationGpuMatmulFp16,
                         testing::ValuesIn(getMatmulTestCases()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmul,
                         IntegrationGpuMatmulBf16,
                         testing::ValuesIn(getMatmulTestCases()));
