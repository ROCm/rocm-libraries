// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
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
