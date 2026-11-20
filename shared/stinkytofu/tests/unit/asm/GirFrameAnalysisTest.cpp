/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include <gtest/gtest.h>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/transforms/asm/GirFrameContractImportPass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "transforms/asm/dag/RegionDAG.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {
constexpr const char* kRingTwoContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=1 src=1 gen=0 value=1 relative=1\n";

constexpr const char* kFenceContract = "gir-frame-contract\n";

constexpr const char* kRingWrapEntranceContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=2 src=0 gen=0 value=1\n";

constexpr const char* kForwardingExitContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=0 src=0 gen=0 value=1 relative=1\n"
    "incoming  dst=1 src=0 gen=0 value=0 relative=1\n";

constexpr const char* kCanonicalSourceAnchorContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=2 src=0 gen=0 value=1\n";

void setFunctionNumWaves(Function& function, uint32_t numWaves) {
    GemmTileConfig config = function.getGemmTileConfig();
    config.NumWaves = numWaves;
    function.setGemmTileConfig(config);
}

}  // namespace

TEST(GirFrameAnalysisTest, ReconstructsRingOnStCfgAndFindsSameTripRaw) {
    Function function("frame");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(loop).size(), 2u);
    EXPECT_TRUE(frames.isBackEdge(loop, loop));
    ASSERT_EQ(frames.occurrences.size(), 4u);

    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    auto raw = std::find_if(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::RAW && hazard.producer == copy &&
                   hazard.consumer == read && hazard.gap == 0;
        });
    EXPECT_NE(raw, hazards.hazards.end());
}

TEST(GirFrameAnalysisTest, ForwardingPhiDistinguishesLatchBackEdgeFromDrainExit) {
    Function function("forwarding_exit");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(loop, loop);
    function.addEdge(loop, drain);

    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
    StinkyInstruction* read = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kForwardingExitContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);

    bool sawBackEdgeAdvance = false;
    bool sawUnadvancedExit = false;
    for (const auto& [node, successors] : frames.edges) {
        if (node.block != loop || node.frame.phaseOf(0) != 0) continue;
        for (const GirFrameNode& successor : successors) {
            if (successor.block == loop && successor.frame.phaseOf(0) == 1)
                sawBackEdgeAdvance = true;
            if (successor.block == drain && successor.frame.phaseOf(0) == 0)
                sawUnadvancedExit = true;
        }
    }
    EXPECT_TRUE(sawBackEdgeAdvance);
    EXPECT_TRUE(sawUnadvancedExit);
}

TEST(GirFrameAnalysisTest, LogicalAnchorsSurviveBlockSplittingAndFoldedFirstAction) {
    Function function("split_source_anchor");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* first = function.createBasicBlock("first");
    BasicBlock* second = function.createBasicBlock("second");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(first, second);
    function.addEdge(second, drain);

    AsmIRBuilder firstBuilder(*first, GfxArchID::Gfx1250);
    StinkyInstruction* firstAction =
        firstBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    firstAction->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Other, {}});
    AsmIRBuilder secondBuilder(*second, GfxArchID::Gfx1250);
    StinkyInstruction* secondAction =
        secondBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    secondAction->addModifier<GirActionData>(GirActionData{1, 0, GirActionKind::Other, {}});
    StinkyInstruction* read = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{3, 2, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kCanonicalSourceAnchorContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(drain).size(), 1u);
    EXPECT_EQ(frames.frames(drain).front().phaseOf(0), 1);
}

TEST(GirFrameAnalysisTest, LoopWrapStillUsesUnreducedGenerationSpan) {
    Function function("loop_wrap_generation_span");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 2, -1, false}}});
    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Tensor, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    auto raw = std::find_if(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::RAW && hazard.producer == copy &&
                   hazard.consumer == read && hazard.gap == 2;
        });
    EXPECT_NE(raw, hazards.hazards.end());
}

/// The test above at EVERY ring depth: the write that next refills the slab the read took is
/// `-span mod ring` nodes on -- a whole lap at ring 2, one node at ring 3, two at ring 4. The
/// depth is a number the contract states, so imposing the unreduced span assumes one.
TEST(GirFrameAnalysisTest, LoopWrapKeepsTheNodeDistanceAtEveryRingDepth) {
    for (int ring = 2; ring <= 6; ++ring) {
        const int span = 2;
        // The next refill of that slab, in nodes: a whole lap when the generations land together.
        const int expected = (ring - span % ring) % ring == 0 ? ring : (ring - span % ring) % ring;
        Function function("loop_wrap_ring_" + std::to_string(ring));
        setFunctionArch(function, GfxArchID::Gfx1250);
        BasicBlock* loop = function.createBasicBlock("loop");
        function.addEdge(loop, loop);

        StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
        read->addModifier<GirActionData>(GirActionData{
            0, 0, GirActionKind::Ds, {GirAccessData{false, 0, ring, 0, span, -1, false}}});
        StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
        copy->addModifier<GirActionData>(GirActionData{
            1, 1, GirActionKind::Tensor, {GirAccessData{true, 0, ring, 0, 0, -1, false}}});
        function.setStringMetaData(
            kGirFrameContractKey, "gir-frame-contract\ngen 0  ring=" + std::to_string(ring) +
                                      " entry=0\nincoming  dst=1 src=1 gen=0 value=1 relative=1\n");

        AnalysisManager analyses;
        registerAllAnalyses(analyses);
        const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
        const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
        auto raw = std::find_if(hazards.hazards.begin(), hazards.hazards.end(),
                                [&](const GirFrameHazard& hazard) {
                                    return hazard.kind == GirHazardKind::RAW &&
                                           hazard.producer == copy && hazard.consumer == read;
                                });
        ASSERT_NE(raw, hazards.hazards.end()) << "ring " << ring;
        EXPECT_EQ(raw->gap, expected) << "ring " << ring;
        const GirPlanFenceCoverage plan = buildGirPlanFenceCoverage(frames, hazards);
        EXPECT_TRUE(plan.byHazard[raw - hazards.hazards.begin()].hasHazardPath) << "ring " << ring;
    }
}

TEST(GirFrameAnalysisTest, EntranceIdentitySurvivesSamePhaseJoin) {
    Function function("same_phase_join");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* left = function.createBasicBlock("left");
    BasicBlock* right = function.createBasicBlock("right");
    BasicBlock* join = function.createBasicBlock("join");
    BasicBlock* destination = function.createBasicBlock("destination");
    function.addEdge(entry, left);
    function.addEdge(entry, right);
    function.addEdge(left, join);
    function.addEdge(right, join);
    function.addEdge(join, destination);

    AsmIRBuilder leftBuilder(*left, GfxArchID::Gfx1250);
    StinkyInstruction* leftAnchor =
        leftBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    leftAnchor->addModifier<GirActionData>(GirActionData{0, 0, GirActionKind::Other, {}});
    AsmIRBuilder rightBuilder(*right, GfxArchID::Gfx1250);
    StinkyInstruction* rightAnchor =
        rightBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    rightAnchor->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Other, {}});
    AsmIRBuilder joinBuilder(*join, GfxArchID::Gfx1250);
    joinBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));
    StinkyInstruction* read = createDsReadB128InBlock(destination, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kRingWrapEntranceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(destination).size(), 2u);
    EXPECT_EQ(frames.frames(destination)[0].phaseOf(0), 0);
    EXPECT_EQ(frames.frames(destination)[1].phaseOf(0), 1);
}

namespace {

/// The hazards between two instructions, as kinds.  A WAW survives only when no read orders the
/// write pair, so these four tests read the kind set directly rather than the fences it produces.
std::multiset<GirHazardKind> hazardKindsBetween(AnalysisManager& analyses, Function& function,
                                                const StinkyInstruction* producer,
                                                const StinkyInstruction* consumer) {
    std::multiset<GirHazardKind> kinds;
    const auto& result = analyses.getResult<GirFrameHazardAnalysis>(function);
    for (const GirFrameHazard& hazard : result.hazards)
        if (hazard.producer == producer && hazard.consumer == consumer) kinds.insert(hazard.kind);
    return kinds;
}

}  // namespace

// A read only discharges a WAW on the paths that run it.  The `skip` arm reaches the second write
// without the read, so the write pair keeps its own order there.
TEST(GirFrameAnalysisTest, WawSurvivesTheArmThatSkipsTheRead) {
    Function function("waw_skipped_read");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* consume = function.createBasicBlock("consume");
    BasicBlock* skip = function.createBasicBlock("skip");
    BasicBlock* join = function.createBasicBlock("join");
    function.addEdge(entry, consume);
    function.addEdge(entry, skip);
    function.addEdge(consume, join);
    function.addEdge(skip, join);

    StinkyInstruction* first = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(consume, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(join, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, second),
              (std::multiset{GirHazardKind::WAW}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, read),
              (std::multiset{GirHazardKind::RAW}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, read, second),
              (std::multiset{GirHazardKind::WAR}));
}

// With the read on every path, RAW then WAR already order the two writes.  Every read still keeps
// its own RAW: the first one does not stand in for the second.
TEST(GirFrameAnalysisTest, InterveningReadDropsTheWawAndKeepsEveryRaw) {
    Function function("waw_ordered_by_read");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* first = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* early = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    early->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* late = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 4, 24);
    late->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Ds, {GirAccessData{false, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{3, 3, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    EXPECT_TRUE(hazardKindsBetween(analyses, function, first, second).empty());
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, early),
              (std::multiset{GirHazardKind::RAW}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, late),
              (std::multiset{GirHazardKind::RAW}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, early, second),
              (std::multiset{GirHazardKind::WAR}));
    EXPECT_EQ(hazardKindsBetween(analyses, function, late, second),
              (std::multiset{GirHazardKind::WAR}));
}

// The read names generation 1, the writes name generation 0.  Sharing an operand is not sharing a
// slot, so this read orders nothing.
TEST(GirFrameAnalysisTest, AReadOfAnotherGenerationDoesNotOrderTheWaw) {
    Function function("waw_other_generation");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* first = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 2, -1, 0, 1, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Tensor, {GirAccessData{true, 0, 2, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, second),
              (std::multiset{GirHazardKind::WAW}));
    EXPECT_TRUE(hazardKindsBetween(analyses, function, first, read).empty());
}

// The claim is about rotating storage.  A ring of 1 has one slot and keeps its WAW.
TEST(GirFrameAnalysisTest, RingOneKeepsItsWawThroughTheRead) {
    Function function("waw_ring_one");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* first = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Tensor, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    EXPECT_EQ(hazardKindsBetween(analyses, function, first, second),
              (std::multiset{GirHazardKind::WAW}));
}

// A Ds action is realized by whichever DS op performs it. Recognising only ds_read dropped a
// ds_write's occurrences, and with them every hazard the later PRs place fences from.
TEST(GirFrameAnalysisTest, DsActionRealizedByADsWriteIsNotDropped) {
    Function function("ds_write_realizes");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* write = createDSWriteInBlock(block, GfxArchID::Gfx1250, 0, 4);
    write->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Ds, {GirAccessData{true, 0, 1, -1, 0, 0, false}}});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 8, 0);
    read->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Ds, {GirAccessData{false, 0, 1, -1, 0, 0, false}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    EXPECT_EQ(std::count_if(frames.occurrences.begin(), frames.occurrences.end(),
                            [&](const GirAccessOccurrence& o) { return o.inst == write; }),
              1);
    EXPECT_EQ(hazardKindsBetween(analyses, function, write, read),
              (std::multiset{GirHazardKind::RAW}));
}

// A kind that matches none of its action's instructions used to drop the access without a word,
// which the fence placement in the later PRs cannot detect. It must be fatal instead.
TEST(GirFrameAnalysisDeathTest, AnActionNoInstructionRealizesIsFatal) {
    Function function("mislabelled_kind");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");

    // A Tensor action tagged onto a ds_read: nothing here performs a tensor access.
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Tensor, {GirAccessData{false, 0, 1, -1, 0, 0, false}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    EXPECT_DEATH((void)analyses.getResult<GirFrameAnalysis>(function),
                 "none of its instructions performs that access");
}

// The plan coverage answers, per hazard, which fences stand on its paths and whether any path is
// left open. A materialized fence between the two ends owns the hazard and closes every path;
// `alsoFences` is the hook placement uses to ask the same question about a proposed marker.
TEST(GirFrameAnalysisTest, PlanCoverageNamesTheFenceOwningACrossAgentHazard) {
    constexpr const char* kPlanCoverageContract =
        "gir-frame-contract\n"
        "gen 0  ring=2 entry=0\n"
        "incoming  dst=2 src=0 gen=0 value=1\n";

    Function function("plan_coverage");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(entry, drain);

    StinkyInstruction* producer = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    producer->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 1, 2, -1, 0, 0, true}}});
    AsmIRBuilder builder(*drain, GfxArchID::Gfx1250);
    StinkyInstruction* fence = builder.createFence();
    fence->addModifier<GirActionData>(GirActionData{2, 2, GirActionKind::Other, {}});
    StinkyInstruction* consumer = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    consumer->addModifier<GirActionData>(
        GirActionData{3, 2, GirActionKind::Ds, {GirAccessData{false, 1, 2, 0, -1, -1, true}}});
    function.setStringMetaData(kGirFrameContractKey, kPlanCoverageContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    ASSERT_EQ(hazards.hazards.size(), 1u);

    const GirPlanFenceCoverage plan = buildGirPlanFenceCoverage(frames, hazards);
    ASSERT_EQ(plan.byHazard.size(), 1u);
    const GirHazardFenceCover& cover = plan.byHazard.front();
    EXPECT_TRUE(cover.hasHazardPath);
    EXPECT_FALSE(cover.hasUnfencedPath);
    ASSERT_EQ(cover.owners.size(), 1u);
    EXPECT_EQ(cover.owners.front().anchor, fence);

    // Nominating the consumer is what placement does while searching: it already sits after the
    // fence, so the answer is unchanged rather than double-counted.
    const GirPlanFenceCoverage marked = buildGirPlanFenceCoverage(
        frames, hazards, [&](const StinkyInstruction& inst) { return &inst == consumer; });
    ASSERT_EQ(marked.byHazard.size(), 1u);
    EXPECT_FALSE(marked.byHazard.front().hasUnfencedPath);
}

// The import pass is what hands the module's contract to each function; without it the analysis
// above has nothing to read.
TEST(GirFrameAnalysisTest, ContractImportPassPutsTheModuleContractOnTheFunction) {
    StinkyAsmModule::ModuleOptions options{};
    options.OptLevel = 0;
    StinkyAsmModule module("import", {12, 5, 0}, options);
    auto contract = std::make_shared<GirFrameContract>();
    contract->loaded = true;
    contract->generations.emplace(0, GirGenerationSpec{0, 2, 0});
    module.setGirFrameContract(contract);
    module.setPluginDataI64("gir.action_tag_count", 1);

    Function function("imported");
    setFunctionArch(function, GfxArchID::Gfx1250);
    function.createBasicBlock("entry");
    ASSERT_EQ(function.getStructMetaData<GirFrameContract>(kGirFrameContractKey), nullptr);

    PassContext context;
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirFrameContractImportPass(module)->run(function, context, analyses);

    const auto* imported = function.getStructMetaData<GirFrameContract>(kGirFrameContractKey);
    ASSERT_NE(imported, nullptr);
    EXPECT_TRUE(imported->loaded);
    EXPECT_EQ(imported->generations.at(0).ring, 2);
}

// The contract is machine-written, so anything it does not say exactly is an authoring mistake.
// Each of these used to parse into a contract that quietly said something else.
TEST(GirFrameAnalysisTest, ContractParserRejectsWhatItCannotMeanExactly) {
    const auto rejected = [](const char* body) {
        std::string error;
        return parseGirFrameContract(body, &error) == nullptr && !error.empty();
    };
    const auto accepted = [](const char* body) {
        return parseGirFrameContract(body, nullptr) != nullptr;
    };

    EXPECT_TRUE(accepted("gir-frame-contract\ngen 0  ring=2 entry=0\n"));
    // A trailing-garbage integer used to read as its numeric prefix.
    EXPECT_TRUE(rejected("gir-frame-contract\ngen 0  ring=2abc entry=0\n"));
    EXPECT_TRUE(rejected("gir-frame-contract\ngen 3x  ring=2 entry=0\n"));
    // A duplicate generation used to overwrite the first silently.
    EXPECT_TRUE(rejected("gir-frame-contract\ngen 0  ring=2 entry=0\ngen 0  ring=4 entry=1\n"));
    // A repeated key used to keep the first value.
    EXPECT_TRUE(rejected("gir-frame-contract\ngen 0  ring=2 ring=4 entry=0\n"));
    // A misspelled key used to be dropped, changing the frame phases with no diagnostic.
    EXPECT_TRUE(
        rejected("gir-frame-contract\ngen 0  ring=2 entry=0\n"
                 "incoming  dst=1 src=1 gen=0 value=1 relativ=1\n"));
    EXPECT_TRUE(
        rejected("gir-frame-contract\ngen 0  ring=2 entry=0\n"
                 "incoming  dst=1 src=1 gen=0 value=1 relative=abc\n"));
    // dst=12junk used to read as 12.
    EXPECT_TRUE(
        rejected("gir-frame-contract\ngen 0  ring=2 entry=0\n"
                 "incoming  dst=12junk src=1 gen=0 value=1\n"));
    // An empty values= used to parse as "constrains nothing".
    EXPECT_TRUE(
        rejected("gir-frame-contract\ngen 0  ring=2 entry=0\n"
                 "requires  dst=1 src=1 gen=0 values=\n"));
    // A key that lost its `=` used to be filed as a positional word and ignored.
    EXPECT_TRUE(
        rejected("gir-frame-contract\ngen 0  ring=2 entry=0\n"
                 "incoming  dst=1 src=1 gen=0 value=1 relative\n"));
    EXPECT_TRUE(rejected("gir-frame-contract\ngen 0 7  ring=2 entry=0\n"));
}

// Lowering may split ONE GIR block across several asm blocks -- a conditionally-issued movement
// branches over itself, as the sparse metadata load does under wave separation.  The frame
// contract never names that seam, so the distance between two touches must be counted in GIR
// blocks: crossing it is not a trip.  Counting asm blocks aged a single trip twice, and at ring 2
// that reads as a whole period, which retires a producer that is still in flight.
TEST(GirFrameAnalysisTest, ASeamInsideOneGirBlockIsNotDistance) {
    Function function("seam");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* pre = function.createBasicBlock("pre");
    BasicBlock* head = function.createBasicBlock("head");
    BasicBlock* guarded = function.createBasicBlock("guarded");
    BasicBlock* join = function.createBasicBlock("join");
    function.addEdge(pre, head);
    function.addEdge(head, guarded);
    function.addEdge(head, join);
    function.addEdge(guarded, join);
    function.addEdge(join, head);

    // Every action anchors on 0, so all three blocks lower from one GIR block.
    StinkyInstruction* write = createTensorLoadInBlock(head, GfxArchID::Gfx1250, 0, 8);
    write->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Tensor, {GirAccessData{true, 0, 2, 0, 0, -1, true}}});
    StinkyInstruction* guardedWrite = createTensorLoadInBlock(guarded, GfxArchID::Gfx1250, 16, 24);
    guardedWrite->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Tensor, {GirAccessData{true, 1, 2, 0, 0, -1, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{2, 0, GirActionKind::Ds, {GirAccessData{false, 0, 2, 0, 0, -1, true}}});
    function.setStringMetaData(kGirFrameContractKey,
                               "gir-frame-contract\n"
                               "gen 0  ring=2 entry=0\n"
                               "incoming  dst=0 src=0 gen=0 value=1 relative=1\n");
    setFunctionNumWaves(function, 2);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    EXPECT_TRUE(frames.insideOneGirBlock(head, guarded));
    EXPECT_TRUE(frames.insideOneGirBlock(head, join));
    EXPECT_TRUE(frames.insideOneGirBlock(guarded, join));
    // The back edge IS the rotation; freeing it stops the trip from counting at all.
    EXPECT_TRUE(frames.isBackEdge(join, head));
    EXPECT_FALSE(frames.insideOneGirBlock(join, head));
    // `pre` lowers from no GIR block, so nothing about it is inside one.
    EXPECT_FALSE(frames.insideOneGirBlock(pre, head));

    std::set<int> gaps;
    for (const GirFrameHazard& hazard :
         analyses.getResult<GirFrameHazardAnalysis>(function).hazards)
        if (hazard.kind == GirHazardKind::RAW && hazard.producer == write &&
            hazard.consumer == read)
            gaps.insert(hazard.gap);
    // The write and the read run on the same trip with two seams between them, so 0 and only 0 --
    // counting asm blocks reported 1 on the short arm and 2 on the guarded one.
    EXPECT_EQ(gaps, (std::set<int>{0}));
}
