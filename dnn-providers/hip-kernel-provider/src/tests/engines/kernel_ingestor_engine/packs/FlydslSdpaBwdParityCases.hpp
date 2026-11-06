// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdVerdictTable.hpp"

/**
 * @file FlydslSdpaBwdParityCases.hpp
 * @brief The cases the FlyDSL SDPA forward's RDNA dispatch suite runs, for the split backward
 *        engine's feature set: each is one graph the split matcher (FlydslSdpaBwdCommon.hpp; the
 *        gfx11 verdict table flydsl/tests/data/sdpa_bwd_gfx11_verdicts.csv) either serves or
 *        declines for a named rule. TestFlydslSdpaBwdParityGraphs.cpp checks them on the host
 *        against the installed gfx11 shard; TestFlydslSdpaBwdParity.cpp launches them on a gfx11
 *        device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_parity
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;

inline constexpr const char* SPLIT_ENGINE_NAME = "hipkernel:flydsl_sdpa_bwd_split";

/// One graph and what the split engine does with it.
struct ParityCase
{
    std::string name;
    /// What a served graph is launched as; unset for a declined one.
    std::optional<BwdCase> run;
    SdpaBwdSpec spec;
    /// The matcher rule that declines the graph; empty for a served one.
    std::string rule;
    /// The decline's cause, as the matcher names it.
    std::string cause;
};

inline ParityCase served(std::string name, BwdCase c)
{
    ParityCase parity;
    parity.name = std::move(name);
    parity.spec = specOf(c);
    parity.run = std::move(c);
    return parity;
}

/// A graph declined for @p rule. A rule the verdict table names has the cause the table's
/// mapping gives it; @p cause names the cause of one it does not.
inline ParityCase
    declined(std::string name, SdpaBwdSpec spec, std::string rule, const std::string& cause = {})
{
    ParityCase parity;
    parity.name = std::move(name);
    parity.spec = std::move(spec);
    parity.cause = cause.empty() ? ruleCauses().at(rule) : cause;
    parity.rule = std::move(rule);
    return parity;
}

inline BwdCase caseOf(DataType dtype,
                      int64_t headDim,
                      int64_t seqQ,
                      int64_t seqKv,
                      int64_t heads = 4,
                      int64_t kvHeads = 4)
{
    BwdCase c;
    c.dtype = dtype;
    c.headDim = headDim;
    c.seqQ = seqQ;
    c.seqKv = seqKv;
    c.heads = heads;
    c.kvHeads = kvHeads;
    return c;
}

inline BwdCase withMask(BwdCase c, BwdMask mask, int64_t left = -1, int64_t right = -1)
{
    c.mask = mask;
    c.windowLeft = left;
    c.windowRight = right;
    return c;
}

inline BwdCase withBias(BwdCase c, BwdBiasFill fill = BwdBiasFill::VALUES)
{
    c.bias = true;
    c.biasFill = fill;
    return c;
}

inline BwdCase withLayout(BwdCase c, BwdLayout layout)
{
    c.layout = layout;
    return c;
}

/// A graph whose bias has dtype @p biasType beside Q's.
inline SdpaBwdSpec biasOfTypeSpec(DataType queryType, DataType biasType)
{
    SdpaBwdSpec spec;
    spec.dtype = queryType;
    return spec.withEdit([=](SdpaBwdGraph& g) {
        addBias(g);
        g.tensor("bias").dtype = biasType;
    });
}

/// A graph whose V and dV have @p valueHeads heads beside K's.
inline SdpaBwdSpec valueHeadsSpec(int64_t valueHeads)
{
    const SdpaBwdSpec spec;
    return spec.withEdit([=](SdpaBwdGraph& g) {
        g.tensor("v").dims[1] = valueHeads;
        g.tensor("dv").dims[1] = valueHeads;
    });
}

/// A graph whose bias, in Q's dtype @p queryType, has rows @p rowPitch elements apart, with a
/// dBias of the same layout when @p withDbias.
inline SdpaBwdSpec biasRowPitchSpec(DataType queryType, int64_t rowPitch, bool withDbias)
{
    SdpaBwdSpec spec;
    spec.dtype = queryType;
    return spec.withEdit([=](SdpaBwdGraph& g) {
        addBroadcastBias(g, false, false, withDbias, std::nullopt, rowPitch);
    });
}

inline const std::vector<ParityCase>& parityCases()
{
    constexpr auto HALF = DataType::HALF;
    constexpr auto BF16 = DataType::BFLOAT16;
    constexpr auto FP32 = DataType::FLOAT;
    static const std::vector<ParityCase> s_cases = [] {
        std::vector<ParityCase> list;

        // 1. Bottom-right causal cross-attention, in both length orders.
        list.push_back(served("ComputesBottomRightCausalCrossAttentionWithFewerQueriesThanKeys",
                              withMask(caseOf(BF16, 128, 64, 300, 4, 2), BwdMask::BOTTOM_RIGHT)));
        list.push_back(served("ComputesBottomRightCausalCrossAttentionWithMoreQueriesThanKeys",
                              withMask(caseOf(HALF, 64, 150, 100, 4, 2), BwdMask::BOTTOM_RIGHT)));

        // 2. Windows.
        list.push_back(served("AppliesASlidingWindowNarrowerThanATile",
                              withMask(caseOf(BF16, 64, 300, 300, 4, 2), BwdMask::TOP_LEFT, 37)));
        list.push_back(served("AppliesALeftOnlyWindow",
                              withMask(caseOf(HALF, 64, 129, 129), BwdMask::NONE, 64)));
        list.push_back(served("AppliesABandToTheRightOfTheDiagonal",
                              withMask(caseOf(HALF, 64, 200, 200), BwdMask::TOP_LEFT, -1, 5)));
        list.push_back(served("AppliesAWideBandWithMoreQueriesThanKeys",
                              withMask(caseOf(BF16, 64, 300, 64), BwdMask::BOTTOM_RIGHT, -1, 250)));

        // 3. V narrower and wider than Q/K.
        auto narrower = withMask(caseOf(BF16, 192, 300, 300), BwdMask::TOP_LEFT);
        narrower.headDimV = 128;
        list.push_back(served("ServesAValueHeadDimNarrowerThanTheQueryOne", narrower));
        auto narrowerGrouped = caseOf(HALF, 192, 150, 333, 4, 2);
        narrowerGrouped.headDimV = 128;
        list.push_back(
            served("ServesANarrowerValueHeadDimOverGroupedCrossAttention", narrowerGrouped));
        auto wider = withMask(caseOf(BF16, 64, 100, 257, 4, 2), BwdMask::BOTTOM_RIGHT);
        wider.headDimV = 128;
        list.push_back(served("ServesAValueHeadDimWiderThanTheQueryOne", wider));
        auto widerBias = withBias(caseOf(HALF, 80, 96, 96));
        widerBias.headDimV = 96;
        list.push_back(served("ServesAWiderValueHeadDimWithABias", widerBias));

        // 4. Rows no key reaches.
        list.push_back(served("ZeroesRowsNoKeyReaches",
                              withMask(caseOf(BF16, 64, 300, 64), BwdMask::BOTTOM_RIGHT)));
        list.push_back(served("ZeroesRowsATopLeftCausalWindowStartsPastTheKeys",
                              withMask(caseOf(BF16, 64, 300, 128), BwdMask::TOP_LEFT, 40)));
        list.push_back(served("ZeroesRowsALeftOnlyWindowStartsPastTheKeys",
                              withMask(caseOf(BF16, 64, 300, 128), BwdMask::NONE, 40)));

        // 5. Biases that mask: whole rows at the lowest finite value, whole rows at -inf and
        // the leading keys at -inf. A bias beside a mask is outside the contract, so none has one.
        list.push_back(served("AddsABiasThatMasksWholeRowsAtTheLowestFiniteValueInBf16",
                              withBias(caseOf(BF16, 64, 200, 300), BwdBiasFill::LOWEST_ROWS)));
        list.push_back(served("AddsABiasThatMasksWholeRowsAtTheLowestFiniteValueInF16",
                              withBias(caseOf(HALF, 64, 200, 300), BwdBiasFill::LOWEST_ROWS)));
        list.push_back(served("AddsABiasThatMasksWholeRowsAtTheLowestFiniteValueInFp32",
                              withBias(caseOf(FP32, 64, 200, 300), BwdBiasFill::LOWEST_ROWS)));
        list.push_back(served("AddsABiasThatMasksWholeRowsWithMinusInfinityInBf16",
                              withBias(caseOf(BF16, 64, 200, 300), BwdBiasFill::NEG_INF_ROWS)));
        list.push_back(served("AddsABiasThatMasksWholeRowsWithMinusInfinityInF16",
                              withBias(caseOf(HALF, 64, 200, 300), BwdBiasFill::NEG_INF_ROWS)));
        list.push_back(served("AddsABiasThatMasksWholeRowsWithMinusInfinityInFp32",
                              withBias(caseOf(FP32, 64, 200, 300), BwdBiasFill::NEG_INF_ROWS)));
        list.push_back(
            served("AddsABiasThatMasksLeadingKeysWithMinusInfinityInBf16",
                   withBias(caseOf(BF16, 64, 200, 300), BwdBiasFill::NEG_INF_LEADING_KEYS)));
        list.push_back(
            served("AddsABiasThatMasksLeadingKeysWithMinusInfinityInFp32",
                   withBias(caseOf(FP32, 64, 200, 300), BwdBiasFill::NEG_INF_LEADING_KEYS)));

        // 6. Head dim 256 and the generic head dims, across the rung boundaries of the gfx11
        // ladder (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256).
        list.push_back(served("ComputesHeadDim256CausalGroupedQueryAttention",
                              withMask(caseOf(BF16, 256, 150, 150, 8, 2), BwdMask::TOP_LEFT)));
        list.push_back(served("ComputesHeadDim256NonCausal", caseOf(HALF, 256, 70, 200)));
        list.push_back(served("ComputesAGenericHeadDim40", caseOf(HALF, 40, 70, 190)));
        list.push_back(served("ComputesAGenericHeadDim80",
                              withMask(caseOf(BF16, 80, 133, 133, 8, 2), BwdMask::TOP_LEFT)));
        list.push_back(served("ComputesAGenericHeadDim152", caseOf(HALF, 152, 150, 301)));
        list.push_back(served("ComputesAGenericHeadDim192",
                              withMask(caseOf(BF16, 192, 90, 150, 4, 2), BwdMask::BOTTOM_RIGHT)));
        list.push_back(served("ComputesAGenericHeadDim240",
                              withMask(caseOf(BF16, 240, 140, 140, 4, 2), BwdMask::TOP_LEFT)));

        // 7. A per-head bias at head dim 128, and biases in the query dtype, or an f32 bias beside
        // a 16-bit query (the split engine opts into it; the merged engine declines it).
        auto perHead = withBias(caseOf(HALF, 128, 70, 150));
        perHead.biasOverBatch = true;
        list.push_back(served("AddsAPerHeadBiasAtHeadDim128", perHead));
        list.push_back(served("AddsABiasInTheQueryDtype", withBias(caseOf(BF16, 64, 77, 99))));
        list.push_back(served("AddsAnFp32BiasToAnFp32Query", withBias(caseOf(FP32, 64, 77, 99))));
        auto fp32BiasHalf = withBias(caseOf(HALF, 64, 77, 99));
        fp32BiasHalf.biasType = DataType::FLOAT;
        fp32BiasHalf.dbias = false;
        list.push_back(served("ServesAnFp32BiasBesideAHalfQuery", fp32BiasHalf));
        auto fp32BiasBf16 = withBias(caseOf(BF16, 64, 77, 99));
        fp32BiasBf16.biasType = DataType::FLOAT;
        fp32BiasBf16.dbias = false;
        list.push_back(served("ServesAnFp32BiasBesideABf16Query", fp32BiasBf16));
        auto fp32BiasWithDbias = withBias(caseOf(HALF, 64, 77, 99));
        fp32BiasWithDbias.biasType = DataType::FLOAT;
        list.push_back(served("ServesAnFp32BiasAndDbiasBesideAHalfQuery", fp32BiasWithDbias));
        // The dBias has Q's strides; one whose rows are not a whole vector apart is declined
        // (BI-9) whatever the bias's own pitch.
        list.push_back(declined("DeclinesADbiasWhoseRowsAreNotAWholeVectorApart",
                                biasRowPitchSpec(HALF, 100, true),
                                "BI-9",
                                "layout"));
        auto unalignedPitch = withBias(caseOf(BF16, 64, 77, 99));
        unalignedPitch.biasPitch = 100;
        list.push_back(declined("DeclinesABiasPitchThatLeavesTheDbiasUnaligned",
                                specOf(unalignedPitch),
                                "BI-9",
                                "layout"));

        // 8. Grouped and multi-query attention, and a V with heads of its own.
        list.push_back(served("ReadsKAndVThroughTheirHeadGroups",
                              withMask(caseOf(BF16, 64, 100, 100, 8, 2), BwdMask::TOP_LEFT)));
        list.push_back(served("ReadsKAndVThroughOneHeadForAMultiQueryGroup",
                              caseOf(HALF, 64, 100, 100, 8, 1)));
        // V heads that differ from K's: served when Hq is a multiple of both and the graph has
        // no bias, is not fp32 and is not ragged (the split engine; the merged engine declines).
        auto valueHeadsOfItsOwn = caseOf(HALF, 64, 100, 100, 8, 8);
        valueHeadsOfItsOwn.valueHeads = 2;
        list.push_back(served("ServesAVWithHeadsOfItsOwn", valueHeadsOfItsOwn));
        auto valueHeads = caseOf(BF16, 64, 100, 100, 8, 2);
        valueHeads.valueHeads = 4;
        list.push_back(served("ServesAVWithMoreHeadsThanKUnderGroupedQueries", valueHeads));
        auto valueHeadsCausal = withMask(caseOf(BF16, 128, 96, 96, 32, 8), BwdMask::TOP_LEFT);
        valueHeadsCausal.valueHeads = 4;
        list.push_back(
            served("ServesAVWithFewerHeadsThanKUnderCausalGroupedQueries", valueHeadsCausal));
        auto valueHeadsWithBias = withBias(caseOf(HALF, 64, 100, 100, 8, 4));
        valueHeadsWithBias.valueHeads = 2;
        list.push_back(
            declined("DeclinesVHeadsOfItsOwnWithABias", specOf(valueHeadsWithBias), "SH-1"));
        auto valueHeadsFp32 = caseOf(FP32, 64, 100, 100, 8, 4);
        valueHeadsFp32.valueHeads = 2;
        list.push_back(declined("DeclinesVHeadsOfItsOwnUnderFp32", specOf(valueHeadsFp32), "SH-1"));
        list.push_back(declined("DeclinesVHeadsOfItsOwnWithRaggedKeys",
                                valueHeadsSpec(2).withEdit([](SdpaBwdGraph& g) {
                                    addSeqLens(g);
                                    for(const char* name : {"k", "v", "dk", "dv"})
                                    {
                                        addRagged(g, name, 1);
                                    }
                                }),
                                "SH-1"));
        auto valueHeadsNotDividing = caseOf(HALF, 64, 100, 100, 8, 8);
        valueHeadsNotDividing.valueHeads = 3;
        list.push_back(declined("DeclinesQueryHeadsThatAreNotAMultipleOfTheValueHeads",
                                specOf(valueHeadsNotDividing),
                                "DT-6",
                                "gqa"));

        // 9. Layouts: head-major against sequence-major strides, packed QKV views, and V, dO and
        // the log-sum-exp through layouts of their own.
        list.push_back(served("ComputesHeadMajorStrides",
                              withLayout(caseOf(HALF, 128, 70, 150), BwdLayout::HEAD_MAJOR)));
        list.push_back(served("ComputesSequenceMajorStrides",
                              withLayout(caseOf(BF16, 64, 100, 100, 8, 2), BwdLayout::SEQ_MAJOR)));
        list.push_back(served("ComputesPackedQkvViews",
                              withLayout(withMask(caseOf(HALF, 128, 150, 150), BwdMask::TOP_LEFT),
                                         BwdLayout::PACKED_QKV)));
        list.push_back(
            served("ReadsVAndDoAndTheLseThroughTheirOwnLayouts",
                   withLayout(caseOf(BF16, 64, 90, 130, 4, 2), BwdLayout::SEQ_MAJOR_VALUE)));

        // 10. The scale: a runtime tensor read at execute, and the default 1 when the graph gives
        // none.
        auto runtimeScale = caseOf(HALF, 64, 96, 96);
        runtimeScale.scaleForm = BwdScale::RUNTIME_TENSOR;
        list.push_back(served("ReadsARuntimeScaleAtExecute", runtimeScale));
        auto defaultScale = caseOf(BF16, 64, 96, 96);
        defaultScale.scaleForm = BwdScale::NONE;
        list.push_back(served("AppliesTheDefaultScaleWhenTheGraphGivesNone", defaultScale));
        return list;
    }();
    return s_cases;
}

inline std::string parityName(const ::testing::TestParamInfo<size_t>& info)
{
    return parityCases()[info.param].name;
}

inline auto parityIndices()
{
    return ::testing::Range(size_t{0}, parityCases().size());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_parity

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
