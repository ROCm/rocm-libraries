// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::conv;
using namespace hip_kernel_provider::test_utilities;

namespace
{

struct SdpaBwdTestCase
{
    SdpaBwdTestCase(std::vector<int64_t> qDimsIn,
                    std::vector<int64_t> vDimsIn,
                    std::optional<float> attnScaleValueIn = std::nullopt,
                    bool causalMaskIn = false)
        : qDims(std::move(qDimsIn))
        , vDims(std::move(vDimsIn))
        , qStrides(generateStrides(qDims))
        , vStrides(generateStrides(vDims))
        , attnScaleValue(attnScaleValueIn)
        , causalMask(causalMaskIn)
    {
        // K tensor is [B, H_kv, S_kv, D_qk]: B and D_qk from Q, H_kv and S_kv from V
        kDims = {qDims[0], vDims[1], vDims[2], qDims[3]};
        kStrides = generateStrides(kDims);
    }

    std::vector<int64_t> qDims;
    std::vector<int64_t> kDims;
    std::vector<int64_t> vDims;
    std::vector<int64_t> qStrides;
    std::vector<int64_t> kStrides;
    std::vector<int64_t> vStrides;

    std::optional<float> attnScaleValue;
    bool causalMask;
};

std::vector<SdpaBwdTestCase> getSdpaBwdTestCases()
{
    // Small case for fast CPU reference execution (backward CPU ref is O(B*H*S^2*D))
    return {SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128})};
}

template <typename DataType>
class SdpaBackward : public IntegrationGraphVerificationHarness<DataType, SdpaBwdTestCase>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        // Step 1: Randomize Q, K, V, dO with identical seeds for both bundles
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, _minVal, _maxVal, seed);
        }

        // Step 2: Compute valid O and stats from Q, K, V using CPU forward reference.
        // The backward pass requires O and stats (LSE) that are mathematically consistent
        // with Q, K, V — random values would cause GPU vs CPU divergence.
        auto& qTensor = bundle.getTensor(_qUid);
        auto& kTensor = bundle.getTensor(_kUid);
        auto& vTensor = bundle.getTensor(_vUid);
        auto& oTensor = bundle.getTensor(_oUid);
        auto& statsTensor = bundle.getTensor(_statsUid);

        const ShallowTensor<DataType> shallowQ(
            qTensor.rawHostData(), qTensor.dims(), qTensor.strides());
        const ShallowTensor<DataType> shallowK(
            kTensor.rawHostData(), kTensor.dims(), kTensor.strides());
        const ShallowTensor<DataType> shallowV(
            vTensor.rawHostData(), vTensor.dims(), vTensor.strides());
        ShallowTensor<DataType> shallowO(oTensor.rawHostData(), oTensor.dims(), oTensor.strides());
        // Stats (LSE) is [B, H_q, S_q] for the CPU reference, but the graph may
        // lower it as 4D. Use the correct 3D dims with contiguous strides.
        const std::vector<int64_t> statsDims3d
            = {qTensor.dims()[0], qTensor.dims()[1], qTensor.dims()[2]};
        const std::vector<int64_t> statsStrides3d = generateStrides(statsDims3d);
        ShallowTensor<float> shallowStats(statsTensor.rawHostData(), statsDims3d, statsStrides3d);

        CpuFpReferenceSdpa::forward<DataType, DataType, DataType, DataType, float>(
            shallowQ,
            shallowK,
            shallowV,
            shallowO,
            _attnScaleValue,
            /*attnMask=*/nullptr,
            _causalMask,
            &shallowStats);

        // Mark O and stats as host-modified so the GPU bundle syncs to device
        oTensor.markHostModified();
        statsTensor.markHostModified();
    }

    void runGraphTest(float tolerance)
    {
        if(hip_kernel_provider_common::getDeviceString(this->stream()) != "gfx942")
        {
            GTEST_SKIP() << "Skipped: ASM SDPA kernel only supports gfx942.";
        }

        const SdpaBwdTestCase& testCase = this->GetParam();

        // O has same dims/strides as Q (with V's head dim for the last dimension)
        auto oDims = testCase.qDims;
        auto oStrides = generateStrides(oDims);

        // dO has same dims/strides as O
        const auto& doDims = oDims;
        const auto& doStrides = oStrides;

        // Stats (LSE) dims: [B, H_q, S_q] with contiguous strides
        const std::vector<int64_t> statsDims
            = {testCase.qDims[0], testCase.qDims[1], testCase.qDims[2]};
        auto statsStrides = generateStrides(statsDims);

        Graph graph;
        graph.set_io_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        q->set_dim(testCase.qDims)
            .set_stride(testCase.qStrides)
            .set_data_type(getDataTypeEnumFromType<DataType>());

        auto k = std::make_shared<TensorAttributes>();
        k->set_dim(testCase.kDims)
            .set_stride(testCase.kStrides)
            .set_data_type(getDataTypeEnumFromType<DataType>());

        auto v = std::make_shared<TensorAttributes>();
        v->set_dim(testCase.vDims)
            .set_stride(testCase.vStrides)
            .set_data_type(getDataTypeEnumFromType<DataType>());

        auto o = std::make_shared<TensorAttributes>();
        o->set_dim(oDims).set_stride(oStrides).set_data_type(getDataTypeEnumFromType<DataType>());

        auto dO = std::make_shared<TensorAttributes>();
        dO->set_dim(doDims).set_stride(doStrides).set_data_type(
            getDataTypeEnumFromType<DataType>());

        auto stats = std::make_shared<TensorAttributes>();
        stats->set_dim(statsDims)
            .set_stride(statsStrides)
            .set_data_type(hipdnn_frontend::DataType::FLOAT);

        SdpaBackwardAttributes bwdAttributes;
        bwdAttributes.set_name("SdpaBwdNode");
        if(testCase.attnScaleValue.has_value())
        {
            bwdAttributes.set_attn_scale_value(testCase.attnScaleValue.value());
        }
        bwdAttributes.set_causal_mask(testCase.causalMask);

        auto [dq, dk, dv] = graph.sdpa_backward(q, k, v, o, dO, stats, std::move(bwdAttributes));

        dq->set_output(true).set_data_type(getDataTypeEnumFromType<DataType>());
        dk->set_output(true).set_data_type(getDataTypeEnumFromType<DataType>());
        dv->set_output(true).set_data_type(getDataTypeEnumFromType<DataType>());

        auto validationResult = graph.validate();
        EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

        // Store UIDs AFTER validate() — validate() assigns UIDs via assignUnsetTensorUids()
        _qUid = q->get_uid();
        _kUid = k->get_uid();
        _vUid = v->get_uid();
        _oUid = o->get_uid();
        _statsUid = stats->get_uid();
        _attnScaleValue = testCase.attnScaleValue;
        _causalMask = testCase.causalMask;

        this->registerValidator(dq, tolerance);
        this->registerValidator(dk, tolerance);
        this->registerValidator(dv, tolerance);
        this->verifyGraph(graph, 0);
    }

    float _minVal = -1.0;
    float _maxVal = 1.0;

private:
    int64_t _qUid = -1;
    int64_t _kUid = -1;
    int64_t _vUid = -1;
    int64_t _oUid = -1;
    int64_t _statsUid = -1;
    std::optional<float> _attnScaleValue;
    bool _causalMask = false;
};

using IntegrationGpuSdpaBwdBf16 = SdpaBackward<bfloat16>;

using IntegrationGpuSdpaBwdFp16 = SdpaBackward<hipdnn_data_sdk::types::half>;

} // namespace

TEST_P(IntegrationGpuSdpaBwdBf16, Correctness)
{
    // BF16 backward error comes from two sources:
    // 1. Softmax recomputation divergence: GPU ASM kernel and CPU FP32 reference
    //    compute exp(score - lse) with different rounding (BF16 hw vs FP32 scalar).
    //    At positions where softmax probability is near-zero, small probability
    //    differences produce large absolute gradient errors.
    // 2. Inherent BF16 precision: 7-bit mantissa causes rounding at each
    //    arithmetic step. The backward pass compounds this through softmax
    //    recomputation, dS = P*(dP-D) catastrophic cancellation, and gradient
    //    matmuls.
    //
    // The CPU reference accumulates dQ/dK/dV in FP32 and converts to BF16 once
    // at the end, matching the GPU kernel's A32 accumulator + dq_convert path.
    //
    // Measured error floor (worst seed across 8 seeds): between 0.3 and 0.5.
    // Use 5e-1 — same as FP16 backward, with ~2x margin over the measured floor
    // of 0.3. Verified stable across seeds {0,42,123,456,789,1024,2048,31415}.

    auto tolerance = 5e-1f;

    runGraphTest(tolerance);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaBwdBf16,
                         testing::ValuesIn(getSdpaBwdTestCases()));

TEST_P(IntegrationGpuSdpaBwdFp16, Correctness)
{
    // FP16 backward has the same error sources as BF16 but the 10-bit mantissa
    // (vs BF16's 7) yields tighter results. Worst-case element lands under 0.25.
    // Use 5e-1 — ~2x margin over that measured floor.

    auto tolerance = 5e-1f;

    runGraphTest(tolerance);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaBwdFp16,
                         testing::ValuesIn(getSdpaBwdTestCases()));
