// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_SDPA

#include <memory>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
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
    // Inputs are kept alongside the gradients because initializeBundle() has to
    // find Q, K, V, O and stats by UID after the graph has been built.
    struct GraphTensors
    {
        std::shared_ptr<graph::TensorAttributes> q;
        std::shared_ptr<graph::TensorAttributes> k;
        std::shared_ptr<graph::TensorAttributes> v;
        std::shared_ptr<graph::TensorAttributes> o;
        std::shared_ptr<graph::TensorAttributes> stats;
        std::shared_ptr<graph::TensorAttributes> dq;
        std::shared_ptr<graph::TensorAttributes> dk;
        std::shared_ptr<graph::TensorAttributes> dv;
    };

    static std::pair<graph::Graph, GraphTensors> buildGraph(hipdnnHandle_t handle,
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

        // Stats (LSE) dims: [B, H_q, S_q, 1]. That is the rank the CPU backward
        // reference requires and the one the frontend infers for the forward
        // stats output. The ASM backward engine reads only the batch and head
        // strides, so either rank serves it.
        const std::vector<int64_t> statsDims{tc.qDims[0], tc.qDims[1], tc.qDims[2], 1};
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

        return std::make_pair(std::move(graphObj), GraphTensors{q, k, v, o, stats, dq, dk, dv});
    }

protected:
    using Harness = IntegrationGraphVerificationHarness<DataType, SdpaBwdTestCase>;

    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();

        auto [graphObj, tensors] = buildGraph(getSharedHandle(), testCase);
        _tensors = tensors;

        this->registerValidator(tensors.dq, this->getTolerance(graphObj, tensors.dq));
        this->registerValidator(tensors.dk, this->getTolerance(graphObj, tensors.dk));
        this->registerValidator(tensors.dv, this->getTolerance(graphObj, tensors.dv));

        this->inputFillRecipes().setGlobalSeed(K_SEED);
        this->verifyGraph(graphObj);
    }

    // Runs once per bundle, after the base class has filled every graph input.
    // The reference bundle and the GPU bundle are filled from the same seeds, so
    // both still start from identical Q, K, V and dO.
    //
    // The random fill alone leaves O and the LSE unrelated to Q, K and V. Both
    // the CPU reference and the backward kernels then recompute
    // P = exp(Q K^T * scale - lse) from an LSE that is not the forward's, so P is
    // not row-stochastic and dS = P * (dP - D) loses its cancellation: bf16
    // produces large absolute outliers and fp16 saturates. Deriving O and the LSE
    // from the forward is what the provider-local ASM backward test has always
    // done, and it is the only input regime in which these gradients are defined.
    FillResult initializeBundle(const hipdnn_frontend::graph::Graph& graph,
                                hipdnn_test_sdk::utilities::GraphTensorBundle& bundle) override
    {
        auto result = Harness::initializeBundle(graph, bundle);
        if(!result.filled)
        {
            return result;
        }

        // UIDs are assigned by graph.build(), which verifyGraph() runs before this
        // call, so they are only resolvable here.
        auto& qTensor = bundle.getTensor(_tensors.q->get_uid());
        auto& kTensor = bundle.getTensor(_tensors.k->get_uid());
        auto& vTensor = bundle.getTensor(_tensors.v->get_uid());
        auto& oTensor = bundle.getTensor(_tensors.o->get_uid());
        auto& statsTensor = bundle.getTensor(_tensors.stats->get_uid());

        const ShallowTensor<DataType> shallowQ(
            qTensor.rawHostData(), qTensor.dims(), qTensor.strides());
        const ShallowTensor<DataType> shallowK(
            kTensor.rawHostData(), kTensor.dims(), kTensor.strides());
        const ShallowTensor<DataType> shallowV(
            vTensor.rawHostData(), vTensor.dims(), vTensor.strides());
        ShallowTensor<DataType> shallowO(oTensor.rawHostData(), oTensor.dims(), oTensor.strides());
        ShallowTensor<float> shallowStats(
            statsTensor.rawHostData(), statsTensor.dims(), statsTensor.strides());

        test_sdpa_common::fillForwardOutputsForBackward<DataType>(
            shallowQ, shallowK, shallowV, shallowO, shallowStats, this->GetParam());

        oTensor.markHostModified();
        statsTensor.markHostModified();

        return result;
    }

private:
    static constexpr unsigned int K_SEED = 0;

    GraphTensors _tensors;
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
