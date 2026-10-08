// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <optional>
#include <ostream>
#include <vector>

namespace test_sdpa_common
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
        , qStrides(hipdnn_data_sdk::utilities::generateStrides(qDims))
        , vStrides(hipdnn_data_sdk::utilities::generateStrides(vDims))
        , attnScaleValue(attnScaleValueIn)
        , leftBound(leftBoundIn)
        , rightBound(rightBoundIn)
        , topLeftAlignment(topLeftAlignmentIn)
    {
        // K tensor is [B, H_kv, S_kv, D_qk]: B and D_qk from Q, H_kv and S_kv from V
        kDims = {qDims[0], vDims[1], vDims[2], qDims[3]};
        kStrides = hipdnn_data_sdk::utilities::generateStrides(kDims);
    }

    std::vector<int64_t> qDims;
    std::vector<int64_t> kDims;
    std::vector<int64_t> vDims;
    std::vector<int64_t> qStrides;
    std::vector<int64_t> kStrides;
    std::vector<int64_t> vStrides;

    std::optional<float> attnScaleValue;
    // fillForwardOutputsForBackward() passes these three straight to the CPU
    // forward, and the backward is graded against a reference that re-derives
    // its own band from the graph through extractDiagonalBandParams(). The two
    // agree only while no case sets the deprecated causal_mask or
    // causal_mask_bottom_right: either one makes that function discard the
    // bounds and substitute (-1, 0, top-left), while these fields keep what was
    // asked for, so O and the LSE would describe a different band than the one
    // under test. Adding such a field here means teaching the helper to check
    // the derived triple first.
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;

    friend std::ostream& operator<<(std::ostream& ss, const SdpaBwdTestCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(q:";
        vecToStream(ss, tc.qDims);
        ss << " k:";
        vecToStream(ss, tc.kDims);
        ss << " v:";
        vecToStream(ss, tc.vDims);
        if(tc.attnScaleValue.has_value())
        {
            ss << " scale:" << *tc.attnScaleValue;
        }
        ss << " leftBound:" << tc.leftBound;
        ss << " rightBound:" << tc.rightBound;
        ss << " alignment:" << (tc.topLeftAlignment ? "TOP_LEFT" : "BOTTOM_RIGHT");
        ss << ")";

        return ss;
    }
};

/// Overwrites `o` and `stats` with the forward outputs of (q, k, v) under this
/// test case's scale, band bounds and diagonal alignment.
///
/// An SDPA backward node takes O and the LSE as inputs and both the CPU reference
/// and the backward kernels recompute P = exp(Q K^T * scale - lse) from them. Only
/// the forward's own LSE makes that recomputation row-stochastic, so only then are
/// dQ, dK and dV the gradients the engine is meant to produce. Filling O and the
/// LSE from a random generator instead leaves P unnormalized by a factor that
/// spans several orders of magnitude, which is numerically meaningless rather than
/// merely a harder test case.
///
/// `stats` must be rank-4 [B, H_q, S_q, 1]; the CPU reference rejects any other
/// rank, and that is the shape the frontend infers for the forward stats output.
template <class DataType>
void fillForwardOutputsForBackward(const hipdnn_data_sdk::utilities::TensorBase<DataType>& q,
                                   const hipdnn_data_sdk::utilities::TensorBase<DataType>& k,
                                   const hipdnn_data_sdk::utilities::TensorBase<DataType>& v,
                                   hipdnn_data_sdk::utilities::TensorBase<DataType>& o,
                                   hipdnn_data_sdk::utilities::TensorBase<float>& stats,
                                   const SdpaBwdTestCase& tc)
{
    using hipdnn_test_sdk::utilities::CpuFpReferenceSdpa;

    CpuFpReferenceSdpa::forward<DataType, DataType, DataType, DataType, float>(
        q,
        k,
        v,
        o,
        tc.attnScaleValue,
        /*attnMask=*/nullptr,
        tc.leftBound,
        tc.rightBound,
        tc.topLeftAlignment,
        &stats);
}

// Engine-agnostic backward coverage, ported verbatim from the ASM SDPA
// provider-local suite. Small dims for fast CPU reference execution (backward
// CPU ref is O(B*H*S^2*D)).
inline std::vector<SdpaBwdTestCase> getSdpaBwdTestCases()
{
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
        // ALMIOPEN-2079: Re-enable when GQA support is implemented
        // // GQA: 4 Q heads, 1 KV head — no mask
        // SdpaBwdTestCase({1, 4, 256, 128}, {1, 1, 256, 128}),
        // // GQA: 4 Q heads, 1 KV head — top-left causal
        // SdpaBwdTestCase({1, 4, 256, 128}, {1, 1, 256, 128}, std::nullopt, -1, 0, true),
    };
}

} // namespace test_sdpa_common
