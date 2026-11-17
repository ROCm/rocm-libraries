// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
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
                    int64_t leftBoundIn = -1,
                    int64_t rightBoundIn = -1,
                    bool topLeftAlignmentIn = true)
        : qDims(std::move(qDimsIn))
        , vDims(std::move(vDimsIn))
        , qStrides(generateStrides(qDims))
        , vStrides(generateStrides(vDims))
        , attnScaleValue(attnScaleValueIn)
        , leftBound(leftBoundIn)
        , rightBound(rightBoundIn)
        , topLeftAlignment(topLeftAlignmentIn)
    {
        // K tensor is [B, H_kv, S_kv, D_qk]: B and D_qk from Q, H_kv and S_kv from V
        kDims = {qDims[0], vDims[1], vDims[2], qDims[3]};
        kStrides = generateStrides(kDims);
    }

    // Non-packed layouts the kernels address: Q, K, V and dO stored [B, S, H, D],
    // and O and dQ rows padded to D+1 elements. O, dO and dQ have Q's dims.
    SdpaBwdTestCase& withNonPackedLayouts()
    {
        // Stride order (lower = tighter) for [B, H, S, D] dims stored as [B, S, H, D].
        const std::vector<int64_t> bshdOrder = {3, 1, 2, 0};
        qStrides = generateStrides(qDims, bshdOrder);
        kStrides = generateStrides(kDims, bshdOrder);
        vStrides = generateStrides(vDims, bshdOrder);
        doStrides = generateStrides(qDims, bshdOrder);
        const int64_t paddedRow = qDims[3] + 1;
        oStrides = {qDims[1] * qDims[2] * paddedRow, qDims[2] * paddedRow, paddedRow, 1};
        dqStrides = oStrides;
        return *this;
    }

    std::vector<int64_t> qDims;
    std::vector<int64_t> kDims;
    std::vector<int64_t> vDims;
    std::vector<int64_t> qStrides;
    std::vector<int64_t> kStrides;
    std::vector<int64_t> vStrides;
    // Empty: packed strides for Q's dims (dO takes O's strides).
    std::vector<int64_t> oStrides;
    std::vector<int64_t> doStrides;
    std::vector<int64_t> dqStrides;

    std::optional<float> attnScaleValue;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
};

std::vector<SdpaBwdTestCase> getSdpaBwdTestCases()
{
    // Small dims for fast CPU reference execution (backward CPU ref is O(B*H*S^2*D))
    return {
        // NO_MASK (mask=0)
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}),
        // TOP_LEFT_CAUSAL (mask=1): rightBound=0, topLeftAlignment=true
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}, std::nullopt, -1, 0, true),
        // BOTTOM_RIGHT_CAUSAL (mask=2): rightBound=0, topLeftAlignment=false
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}, std::nullopt, -1, 0, false),
        // SLIDING_WINDOW / SWA (mask=3): top-left alignment
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}, std::nullopt, 64, 64, true),
        // SLIDING_WINDOW / SWA (mask=3): bottom-right alignment
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}, std::nullopt, 64, 64, false),
        // Asymmetric Sq != Skv — no mask
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 512, 128}),
        // Asymmetric Sq != Skv — top-left causal
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 512, 128}, std::nullopt, -1, 0, true),
        // Asymmetric Sq != Skv — bottom-right causal
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 512, 128}, std::nullopt, -1, 0, false),
        // B=2 and H=2, so every batch and head stride, and the packed LSE/D
        // layout that buildPlan derives from Q's dims, reach the kernels.
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}),
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}, std::nullopt, -1, 0, true),
        // The same with non-packed layouts that isApplicable accepts.
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}).withNonPackedLayouts(),
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}, std::nullopt, -1, 0, true)
            .withNonPackedLayouts(),
        // ALMIOPEN-2079: Re-enable when GQA support is implemented
        // // GQA: 4 Q heads, 1 KV head — no mask
        // SdpaBwdTestCase({1, 4, 256, 128}, {1, 1, 256, 128}),
        // // GQA: 4 Q heads, 1 KV head — top-left causal
        // SdpaBwdTestCase({1, 4, 256, 128}, {1, 1, 256, 128}, std::nullopt, -1, 0, true),
    };
}

// Cases for the sink-without-dSink test: unmasked, batched and multi-head, and causal.
std::vector<SdpaBwdTestCase> getSdpaBwdSinkTestCases()
{
    return {
        SdpaBwdTestCase({1, 1, 256, 128}, {1, 1, 256, 128}),
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}),
        SdpaBwdTestCase({2, 2, 256, 128}, {2, 2, 256, 128}, std::nullopt, -1, 0, true),
    };
}

// A bf16 gradient may miss the reference by this fraction of the reference's largest element.
constexpr double K_SINK_REL_TOLERANCE = 4e-2;

// Compares a bf16 gradient with the reference at K_SINK_REL_TOLERANCE of the reference's largest
// element. A NaN, such as an element no kernel wrote, counts as a miss.
class RelativeToLargestValidator : public IReferenceValidation
{
public:
    explicit RelativeToLargestValidator(double relativeTolerance)
        : _relativeTolerance(relativeTolerance)
    {
    }

    bool allClose(ITensor& reference, ITensor& implementation) const override
    {
        const size_t count = reference.elementSpace();
        if(count == 0 || implementation.elementSpace() != count
           || reference.elementSize() != sizeof(bfloat16)
           || implementation.elementSize() != sizeof(bfloat16))
        {
            ADD_FAILURE() << "the gradient must be a non-empty bf16 tensor of the reference's size";
            return false;
        }
        const auto* expected = static_cast<const bfloat16*>(reference.rawHostData());
        const auto* actual = static_cast<const bfloat16*>(implementation.rawHostData());

        double largest = 0.0;
        for(size_t i = 0; i < count; ++i)
        {
            largest
                = std::max(largest, std::abs(static_cast<double>(static_cast<float>(expected[i]))));
        }
        const double bound = _relativeTolerance * largest;
        size_t misses = 0;
        size_t firstMiss = 0;
        for(size_t i = 0; i < count; ++i)
        {
            const double diff = std::abs(static_cast<double>(static_cast<float>(actual[i]))
                                         - static_cast<double>(static_cast<float>(expected[i])));
            if(!(diff <= bound))
            {
                firstMiss = misses == 0 ? i : firstMiss;
                ++misses;
            }
        }
        EXPECT_EQ(misses, 0U) << "first miss at element " << firstMiss << ": got "
                              << static_cast<float>(actual[firstMiss]) << ", expected "
                              << static_cast<float>(expected[firstMiss]) << ", bound " << bound;
        return misses == 0;
    }

private:
    double _relativeTolerance;
};

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
        //
        // UIDs are resolved here (not in runGraphTest) because graph.build() —
        // called by verifyGraph() before initializeBundle() — is what assigns
        // UIDs via assignUnsetTensorUids().  Querying get_uid() before build()
        // returns the default (0) for all tensors.
        auto& qTensor = bundle.getTensor(_qAttr->get_uid());
        auto& kTensor = bundle.getTensor(_kAttr->get_uid());
        auto& vTensor = bundle.getTensor(_vAttr->get_uid());
        auto& oTensor = bundle.getTensor(_oAttr->get_uid());
        auto& statsTensor = bundle.getTensor(_statsAttr->get_uid());

        const ShallowTensor<DataType> shallowQ(
            qTensor.rawHostData(), qTensor.dims(), qTensor.strides());
        const ShallowTensor<DataType> shallowK(
            kTensor.rawHostData(), kTensor.dims(), kTensor.strides());
        const ShallowTensor<DataType> shallowV(
            vTensor.rawHostData(), vTensor.dims(), vTensor.strides());
        ShallowTensor<DataType> shallowO(oTensor.rawHostData(), oTensor.dims(), oTensor.strides());
        // Stats (LSE) is [B, H_q, S_q, 1] for the CPU reference
        const std::vector<int64_t> statsDims
            = {qTensor.dims()[0], qTensor.dims()[1], qTensor.dims()[2], 1};
        const std::vector<int64_t> statsStrides = generateStrides(statsDims);
        ShallowTensor<float> shallowStats(statsTensor.rawHostData(), statsDims, statsStrides);

        SdpaFwdOptions<float> fwdOptions;
        fwdOptions.setScale(_attnScaleValue)
            .setBand(_leftBound, _rightBound, _topLeftAlignment)
            .setLse(&shallowStats);

        // A sink large enough to move the Stats well past the tolerance: the Stats must carry
        // it, as the backward kernels read the sink only through them.
        std::optional<ShallowTensor<float>> shallowSink;
        if(_sinkAttr)
        {
            auto& sinkTensor = bundle.getTensor(_sinkAttr->get_uid());
            auto* sinkValues = static_cast<float*>(sinkTensor.rawHostData());
            for(size_t head = 0; head < sinkTensor.elementSpace(); ++head)
            {
                sinkValues[head] = 6.0F + 0.5F * static_cast<float>(head);
            }
            sinkTensor.markHostModified();
            shallowSink.emplace(sinkTensor.rawHostData(), sinkTensor.dims(), sinkTensor.strides());
            fwdOptions.setSink(&*shallowSink);
        }

        CpuFpReferenceSdpa::forward<DataType, DataType, DataType, DataType, float>(
            shallowQ, shallowK, shallowV, shallowO, fwdOptions);

        // Mark O and stats as host-modified so the GPU bundle syncs to device
        oTensor.markHostModified();
        statsTensor.markHostModified();
    }

    void runGraphTest(float tolerance, bool withSink = false)
    {
        if(hip_kernel_provider_common::getDeviceString(this->stream()) != "gfx942")
        {
            GTEST_SKIP() << "Skipped: ASM SDPA kernel only supports gfx942.";
        }

        const SdpaBwdTestCase& testCase = this->GetParam();

        // O has same dims as Q (with V's head dim for the last dimension)
        auto oDims = testCase.qDims;
        auto oStrides = testCase.oStrides.empty() ? generateStrides(oDims) : testCase.oStrides;

        // dO has same dims as O, and O's strides unless the test case sets its own
        const auto& doDims = oDims;
        const auto doStrides = testCase.doStrides.empty() ? oStrides : testCase.doStrides;

        // Stats (LSE) dims: [B, H_q, S_q] with contiguous strides
        const std::vector<int64_t> statsDims
            = {testCase.qDims[0], testCase.qDims[1], testCase.qDims[2], 1};
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
            bwdAttributes.set_attn_scale(testCase.attnScaleValue.value());
        }
        if(testCase.leftBound >= 0)
        {
            bwdAttributes.set_diagonal_band_left_bound(testCase.leftBound);
        }
        if(testCase.rightBound >= 0)
        {
            bwdAttributes.set_diagonal_band_right_bound(testCase.rightBound);
        }
        if(testCase.leftBound >= 0 || testCase.rightBound >= 0)
        {
            bwdAttributes.set_diagonal_alignment(testCase.topLeftAlignment
                                                     ? DiagonalAlignment::TOP_LEFT
                                                     : DiagonalAlignment::BOTTOM_RIGHT);
        }

        // sink_token without dsink_token: [1, H_q, 1, 1], FLOAT.
        std::shared_ptr<TensorAttributes> sink;
        if(withSink)
        {
            sink = std::make_shared<TensorAttributes>();
            sink->set_dim({1, testCase.qDims[1], 1, 1})
                .set_stride({testCase.qDims[1], 1, 1, 1})
                .set_data_type(hipdnn_frontend::DataType::FLOAT);
            bwdAttributes.set_sink_token(sink);
        }

        auto [dq, dk, dv] = graph.sdpa_backward(q, k, v, o, dO, stats, std::move(bwdAttributes));

        dq->set_output(true).set_data_type(getDataTypeEnumFromType<DataType>());
        if(!testCase.dqStrides.empty())
        {
            dq->set_dim(testCase.qDims).set_stride(testCase.dqStrides);
        }
        dk->set_output(true).set_data_type(getDataTypeEnumFromType<DataType>());
        dv->set_output(true).set_data_type(getDataTypeEnumFromType<DataType>());

        auto validationResult = graph.validate();
        EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

        // Store tensor attribute pointers — NOT UIDs.  UIDs are assigned by
        // graph.build() (via assignUnsetTensorUids()), which runs inside
        // verifyGraph().  Storing get_uid() here would return 0 for all tensors
        // because validate() does not assign UIDs.  initializeBundle() resolves
        // UIDs lazily via _qAttr->get_uid() after build() has run.
        _qAttr = q;
        _kAttr = k;
        _vAttr = v;
        _oAttr = o;
        _statsAttr = stats;
        _sinkAttr = sink;
        _attnScaleValue = testCase.attnScaleValue;
        _leftBound = testCase.leftBound;
        _rightBound = testCase.rightBound;
        _topLeftAlignment = testCase.topLeftAlignment;

        if(withSink)
        {
            // Pin the ASM engine: a graph another engine serves would pass the comparison below
            // and leave the ASM kernels unexercised.
            auto result = graph.build_operation_graph(this->_handle);
            ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
            std::vector<int64_t> rankedEngineIds;
            result = graph.get_ranked_engine_ids(rankedEngineIds);
            ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
            const auto asmEngineId = hipdnn_data_sdk::utilities::engineNameToId(
                hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_NAME);
            ASSERT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), asmEngineId),
                      rankedEngineIds.end())
                << "the ASM SDPA engine did not offer to serve a backward graph with a sink and "
                   "no dSink on gfx942";
            result = graph.create_execution_plan_ext(asmEngineId, {});
            ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
            result = graph.check_support();
            ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
            result = graph.build_plans();
            ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
            int64_t servedEngineId = 0;
            result = graph.get_execution_plan_engine_id(servedEngineId);
            ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
            ASSERT_EQ(servedEngineId, asmEngineId);

            GraphVerificationContext sinkContext(graph);
            for(const auto& gradient : {dq, dk, dv})
            {
                this->registerValidator(
                    sinkContext,
                    gradient,
                    std::make_unique<RelativeToLargestValidator>(K_SINK_REL_TOLERANCE));
            }
            VerificationRun run;
            ASSERT_NO_FATAL_FAILURE(this->prepareVerification(sinkContext, 0, run));
            // Poison the gradients so an element the kernels do not write fails the comparison.
            run.gpuBundle.sentinelFillOutputTensors();
            ASSERT_NO_FATAL_FAILURE(this->executeAndValidate(run, graph, this->_handle));

            // The reference above carries the sink in its Stats. Without the sink the same
            // inputs must give gradients that differ by far more than the tolerance, or the
            // comparison could not tell a sink-inclusive Stats from a sink-free one.
            const auto cpuShallow = [&run](int64_t uid) {
                auto& tensor = run.cpuBundle.getTensor(uid);
                return std::make_unique<ShallowTensor<DataType>>(
                    tensor.rawHostData(), tensor.dims(), tensor.strides());
            };
            const auto qRef = cpuShallow(q->get_uid());
            const auto kRef = cpuShallow(k->get_uid());
            const auto vRef = cpuShallow(v->get_uid());
            const auto dORef = cpuShallow(dO->get_uid());
            hipdnn_data_sdk::utilities::Tensor<DataType> oFree(oDims);
            hipdnn_data_sdk::utilities::Tensor<float> lseFree(statsDims);
            hipdnn_data_sdk::utilities::Tensor<DataType> dqFree(testCase.qDims);
            hipdnn_data_sdk::utilities::Tensor<DataType> dkFree(testCase.kDims);
            hipdnn_data_sdk::utilities::Tensor<DataType> dvFree(testCase.vDims);
            SdpaFwdOptions<float> freeFwdOptions;
            freeFwdOptions.setScale(testCase.attnScaleValue)
                .setBand(testCase.leftBound, testCase.rightBound, testCase.topLeftAlignment)
                .setLse(&lseFree);
            CpuFpReferenceSdpa::forward<DataType, DataType, DataType, DataType, float>(
                *qRef, *kRef, *vRef, oFree, freeFwdOptions);
            SdpaBwdOptions<float> freeBwdOptions;
            freeBwdOptions.setScale(testCase.attnScaleValue)
                .setBand(testCase.leftBound, testCase.rightBound, testCase.topLeftAlignment)
                .setLse(&lseFree);
            CpuFpReferenceSdpa::backward<DataType,
                                         DataType,
                                         DataType,
                                         DataType,
                                         DataType,
                                         DataType,
                                         DataType,
                                         DataType,
                                         float,
                                         float>(
                *qRef, *kRef, *vRef, oFree, *dORef, dqFree, dkFree, dvFree, freeBwdOptions);

            const auto separates
                = [&run](int64_t uid,
                         const hipdnn_data_sdk::utilities::Tensor<DataType>& withoutSink) {
                      auto& reference = run.cpuBundle.getTensor(uid);
                      const auto* withSinkValues
                          = static_cast<const DataType*>(reference.rawHostData());
                      const auto* freeValues = withoutSink.memory().hostData();
                      float largest = 0.0F;
                      float largestDiff = 0.0F;
                      for(size_t i = 0; i < reference.elementSpace(); ++i)
                      {
                          const auto a = static_cast<float>(withSinkValues[i]);
                          const auto b = static_cast<float>(freeValues[i]);
                          largest = std::max(largest, std::abs(a));
                          largestDiff = std::max(largestDiff, std::abs(a - b));
                      }
                      return static_cast<double>(largestDiff)
                             > 4.0 * K_SINK_REL_TOLERANCE * static_cast<double>(largest);
                  };
            ASSERT_TRUE(separates(dq->get_uid(), dqFree)) << "dQ does not depend on the sink";
            ASSERT_TRUE(separates(dk->get_uid(), dkFree)) << "dK does not depend on the sink";
            ASSERT_TRUE(separates(dv->get_uid(), dvFree)) << "dV does not depend on the sink";
            return;
        }

        GraphVerificationContext context(graph);
        this->registerValidator(context, dq, tolerance);
        this->registerValidator(context, dk, tolerance);
        this->registerValidator(context, dv, tolerance);
        this->verifyGraph(context, 0);
    }

    float _minVal = -1.0;
    float _maxVal = 1.0;

private:
    std::shared_ptr<TensorAttributes> _qAttr;
    std::shared_ptr<TensorAttributes> _kAttr;
    std::shared_ptr<TensorAttributes> _vAttr;
    std::shared_ptr<TensorAttributes> _oAttr;
    std::shared_ptr<TensorAttributes> _statsAttr;
    std::shared_ptr<TensorAttributes> _sinkAttr;
    std::optional<float> _attnScaleValue;
    int64_t _leftBound = -1;
    int64_t _rightBound = -1;
    bool _topLeftAlignment = true;
};

using IntegrationGpuSdpaBwdBf16 = SdpaBackward<bfloat16>;

using IntegrationGpuSdpaBwdFp16 = SdpaBackward<hipdnn_data_sdk::types::half>;

// Backward with a sink and no dSink, pinned to the ASM SDPA engine. The inputs are small enough
// that the sink (6.0 and up) is comparable to the logits and moves the Stats and every gradient.
class IntegrationGpuSdpaBwdSinkBf16 : public SdpaBackward<bfloat16>
{
public:
    IntegrationGpuSdpaBwdSinkBf16()
    {
        _minVal = -0.5F;
        _maxVal = 0.5F;
    }
};

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

// dQ/dK/dV depend on the sink only through the forward Stats, which the ASM kernels read from the
// caller's buffer, so the engine serves a sink without a dSink. The Stats and O fed to the
// backward come from the reference forward with the sink.
TEST_P(IntegrationGpuSdpaBwdSinkBf16, SinkWithoutDsinkMatchesSinkInclusiveReference)
{
    runGraphTest(0.0F, /*withSink=*/true);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaBwdSinkBf16,
                         testing::ValuesIn(getSdpaBwdSinkTestCases()));
