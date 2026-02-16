// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <random>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../tests/common/MatmulCommon.hpp"
#include "IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipblaslt_plugin::test_utilities;
using namespace test_matmul_common;

namespace
{

template <typename DataType>
class IntegrationGpuMatmul
    : public IntegrationGraphVerificationHarness<DataType, std::tuple<MatmulTestCase, bool>>
{
protected:
    void runGraphTest(float tolerance) override
    {
        const auto& [testCase, doBias] = this->GetParam();

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("MatmulTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto aAttr = graph::makeTensorAttributes(
            "a", testCase.aDims, generateInputStrideOrder(testCase.aDims, testCase.transA));
        auto aTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(aAttr));

        auto bAttr = graph::makeTensorAttributes(
            "b", testCase.bDims, generateInputStrideOrder(testCase.bDims, testCase.transB));
        auto bTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(bAttr));

        graph::MatmulAttributes matmulAttrs;

        auto cAttr = graphObj.matmul(aTensorAttr, bTensorAttr, matmulAttrs);

        std::shared_ptr<graph::TensorAttributes> yBiasTensorAttr;
        if(doBias)
        {
            std::vector<int64_t> biasDims(testCase.cDims.size(), 1);
            biasDims[biasDims.size() - 1] = testCase.cDims.back();

            auto biasAttr
                = graph::makeTensorAttributes("bias", biasDims, generateStrides(biasDims));
            auto biasTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(biasAttr));

            graph::PointwiseAttributes biasAttrs;
            biasAttrs.set_mode(hipdnn_frontend::PointwiseMode::ADD);
            biasAttrs.set_compute_data_type(dataType);

            yBiasTensorAttr = graphObj.pointwise(cAttr, biasTensorAttr, biasAttrs);
        }

        auto yTensorAttr = doBias ? yBiasTensorAttr : cAttr;
        yTensorAttr->set_output(true);

        this->registerValidator(yTensorAttr, tolerance);

        ASSERT_NO_FATAL_FAILURE(this->verifyGraph(graphObj, testCase.seed));
    }

protected:
    static std::vector<int64_t> generateInputStrideOrder(const std::vector<int64_t>& dims,
                                                         bool transpose)
    {
        std::vector<int64_t> strides = generateStrides(dims);
        if(transpose)
        {
            const size_t rank = dims.size();
            strides[rank - 1] = dims[rank - 2];
            strides[rank - 2] = 1;
        }
        return strides;
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
                         testing::Combine(testing::ValuesIn(getMatmulTestCases()),
                                          testing::Bool()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmul,
                         IntegrationGpuMatmulFp16,
                         testing::Combine(testing::ValuesIn(getMatmulTestCases()),
                                          testing::Bool()));

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmul,
                         IntegrationGpuMatmulBf16,
                         testing::Combine(testing::ValuesIn(getMatmulTestCases()),
                                          testing::Bool()));
