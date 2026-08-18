// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_SDPA

#include <utility>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "common/SdpaCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using test_sdpa_common::SdpaBwdTestCase;

namespace
{

template <typename DataType>
class SdpaBackward : public IntegrationGraphVerificationHarness<DataType, SdpaBwdTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dq;
        std::shared_ptr<graph::TensorAttributes> dk;
        std::shared_ptr<graph::TensorAttributes> dv;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const SdpaBwdTestCase& tc)
    {
        graph::Graph graphObj;
        graphObj.set_name("SdpaBackwardTest");

        const auto ioType = getDataTypeEnumFromType<DataType>();
        graphObj.set_io_data_type(ioType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

        // O has the same dims/strides as Q (with V's head dim for the last dimension).
        const auto& oDims = tc.qDims;
        const auto oStrides = generateStrides(oDims);

        // dO has the same dims/strides as O.
        const auto& doDims = oDims;
        const auto& doStrides = oStrides;

        // Stats (LSE) dims: [B, H_q, S_q] with contiguous strides.
        const std::vector<int64_t> statsDims{tc.qDims[0], tc.qDims[1], tc.qDims[2]};
        const auto statsStrides = generateStrides(statsDims);

        auto q = std::make_shared<graph::TensorAttributes>(
            graph::makeTensorAttributes("Q", ioType, tc.qDims, tc.qStrides));
        auto k = std::make_shared<graph::TensorAttributes>(
            graph::makeTensorAttributes("K", ioType, tc.kDims, tc.kStrides));
        auto v = std::make_shared<graph::TensorAttributes>(
            graph::makeTensorAttributes("V", ioType, tc.vDims, tc.vStrides));
        auto o = std::make_shared<graph::TensorAttributes>(
            graph::makeTensorAttributes("O", ioType, oDims, oStrides));
        auto dO = std::make_shared<graph::TensorAttributes>(
            graph::makeTensorAttributes("dO", ioType, doDims, doStrides));
        auto stats = std::make_shared<graph::TensorAttributes>(graph::makeTensorAttributes(
            "stats", hipdnn_frontend::DataType::FLOAT, statsDims, statsStrides));

        graph::SdpaBackwardAttributes bwdAttrs;
        if(tc.attnScaleValue.has_value())
        {
            bwdAttrs.set_attn_scale(*tc.attnScaleValue);
        }
        if(tc.leftBound >= 0)
        {
            bwdAttrs.set_diagonal_band_left_bound(tc.leftBound);
        }
        if(tc.rightBound >= 0)
        {
            bwdAttrs.set_diagonal_band_right_bound(tc.rightBound);
        }
        if(tc.leftBound >= 0 || tc.rightBound >= 0)
        {
            bwdAttrs.set_diagonal_alignment(tc.topLeftAlignment ? DiagonalAlignment::TOP_LEFT
                                                                : DiagonalAlignment::BOTTOM_RIGHT);
        }

        auto [dq, dk, dv] = graphObj.sdpa_backward(q, k, v, o, dO, stats, std::move(bwdAttrs));

        dq->set_output(true).set_data_type(ioType);
        dk->set_output(true).set_data_type(ioType);
        dv->set_output(true).set_data_type(ioType);

        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            throw std::runtime_error("Failed to validate graph: " + validateResult.get_message());
        }

        auto buildResult = graphObj.build_operation_graph(handle);
        if(buildResult.is_bad())
        {
            throw std::runtime_error("Failed to build operation graph: "
                                     + buildResult.get_message());
        }

        return std::make_pair(std::move(graphObj), GraphOutputs{dq, dk, dv});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.dq, this->getTolerance(graphObj, outputs.dq));
        this->registerValidator(outputs.dk, this->getTolerance(graphObj, outputs.dk));
        this->registerValidator(outputs.dv, this->getTolerance(graphObj, outputs.dv));

        this->inputFillRecipes().setGlobalSeed(SEED);
        this->verifyGraph(graphObj);
    }

private:
    static constexpr unsigned int SEED = 0;
};

using IntegrationGpuSdpaBwdBfp16 = SdpaBackward<bfloat16>;
using IntegrationGpuSdpaBwdFp16 = SdpaBackward<hipdnn_data_sdk::types::half>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuSdpaBwdBfp16);
TEST_P(IntegrationGpuSdpaBwdBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaBwdBfp16,
                         testing::ValuesIn(test_sdpa_common::getSdpaBwdTestCases()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuSdpaBwdFp16);
TEST_P(IntegrationGpuSdpaBwdFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaBwdFp16,
                         testing::ValuesIn(test_sdpa_common::getSdpaBwdTestCases()));

#endif // HIPDNN_ENABLE_SDPA
