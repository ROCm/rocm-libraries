/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include <gtest/gtest.h>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/transforms/asm/GirFencePlacementPass.hpp"
#include "stinkytofu/transforms/asm/GirWaitCntInsertionPass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "transforms/asm/dag/RegionDAG.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {
constexpr const char* kRingTwoContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=1 src=1 gen=0 value=1 relative=1\n";

constexpr const char* kDiamondContract = "gir-frame-contract\n";

constexpr const char* kWarContract = "gir-frame-contract\n";

constexpr const char* kFenceContract = "gir-frame-contract\n";

constexpr const char* kPromotionContract = "gir-frame-contract\n";

constexpr const char* kEntrancePhiContract =
    "gir-frame-contract\n"
    "gen 0  ring=2 entry=0\n"
    "incoming  dst=2 src=0 gen=0 value=1\n";

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

constexpr const char* kAgentRelativeRegionsContract =
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
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
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
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
    StinkyInstruction* read = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
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
        GirActionData{3, 2, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kCanonicalSourceAnchorContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(drain).size(), 1u);
    EXPECT_EQ(frames.frames(drain).front().phaseOf(0), 1);
}

TEST(GirFrameAnalysisTest, AgentRelativeReadWaitsForEveryPhysicalRegionProducer) {
    Function function("agent_relative_regions");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(entry, drain);

    StinkyInstruction* first = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 1, 2, -1, 0, 0, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Copy, {GirAccessData{true, 2, 2, -1, 0, 0, true}}});
    AsmIRBuilder builder(*drain, GfxArchID::Gfx1250);
    StinkyInstruction* fence = builder.createFence();
    fence->addModifier<GirActionData>(GirActionData{2, 2, GirActionKind::Fence, {}});
    StinkyInstruction* read = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{3,
                                                   2,
                                                   GirActionKind::Read,
                                                   {GirAccessData{false, 1, 2, 0, -1, -1, true},
                                                    GirAccessData{false, 2, 2, 0, -1, -1, true}}});
    function.setStringMetaData(kGirFrameContractKey, kAgentRelativeRegionsContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeFence = nullptr;
    for (IRBase& node : *drain) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == fence) break;
        beforeFence = inst;
    }
    ASSERT_NE(beforeFence, nullptr);
    const auto* wait = beforeFence->getModifier<SWaitTensorCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->tlcnt, 0);
}

// A graded wait for the first RAW leaves the newer producer in flight. Its later-block RAW needs
// another late barrier and wait; treating the first barrier as publishing every future RAW drops
// the second one.
TEST(GirFrameAnalysisTest, LateRawFenceRanksSuccessorRegionAtEndpoint) {
    Function function("late_fence_region");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(entry, drain);

    StinkyInstruction* first = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Copy, {GirAccessData{true, 1, 1, -1, 0, 0, true}}});

    StinkyInstruction* firstRead = createDsReadB128InBlock(entry, GfxArchID::Gfx1250, 0, 20);
    firstRead->addModifier<GirActionData>(
        GirActionData{2, 0, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* secondRead = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 4, 24);
    secondRead->addModifier<GirActionData>(
        GirActionData{3, 3, GirActionKind::Read, {GirAccessData{false, 1, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kDiamondContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);

    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    auto successorRaw = std::find_if(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::RAW && hazard.producer == second &&
                   hazard.consumer == secondRead && hazard.gap == 1 && hazard.crossAgent;
        });
    ASSERT_NE(successorRaw, hazards.hazards.end());

    createGirFencePlacementPass()->run(function, context, analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    const auto waitsIn = [](BasicBlock* block) {
        std::vector<int> waits;
        for (IRBase& node : *block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            if (const auto* wait = inst->getModifier<SWaitTensorCntData>())
                waits.push_back(wait->tlcnt);
        }
        return waits;
    };
    EXPECT_EQ(waitsIn(entry), (std::vector<int>{1}));
    EXPECT_EQ(waitsIn(drain), (std::vector<int>{0}));
}

TEST(GirFrameAnalysisTest, UnwaitedTensorRawFenceIsRemoved) {
    Function function("late_raw_fences");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* first = function.createBasicBlock("first");
    BasicBlock* merge = function.createBasicBlock("merge");
    function.addEdge(entry, first);
    function.addEdge(first, merge);

    StinkyInstruction* producerA = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    producerA->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* producerB = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    producerB->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Copy, {GirAccessData{true, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* filler = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 32, 40);
    filler->addModifier<GirActionData>(
        GirActionData{2, 0, GirActionKind::Copy, {GirAccessData{true, 2, 1, -1, 0, 0, true}}});
    StinkyInstruction* firstRead = createDsReadB128InBlock(first, GfxArchID::Gfx1250, 4, 24);
    firstRead->addModifier<GirActionData>(
        GirActionData{3, 3, GirActionKind::Read, {GirAccessData{false, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* mergeRead = createDsReadB128InBlock(merge, GfxArchID::Gfx1250, 0, 20);
    mergeRead->addModifier<GirActionData>(
        GirActionData{4, 4, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    EXPECT_TRUE(std::any_of(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::RAW && hazard.producer == producerB &&
                   hazard.consumer == firstRead && hazard.crossAgent;
        }));
    EXPECT_TRUE(std::any_of(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::RAW && hazard.producer == producerA &&
                   hazard.consumer == mergeRead && hazard.crossAgent;
        }));

    const auto counts = [](BasicBlock* block) {
        std::pair<int, int> result;
        for (IRBase& node : *block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst) continue;
            if (isBarrierSignal(*inst)) ++result.first;
            if (inst->getModifier<SWaitTensorCntData>()) ++result.second;
        }
        return result;
    };

    createGirFencePlacementPass()->run(function, context, analyses);
    EXPECT_EQ(counts(entry), (std::pair{0, 0}));
    EXPECT_EQ(counts(first), (std::pair{1, 0}));
    EXPECT_EQ(counts(merge), (std::pair{0, 0}));

    createGirWaitCntInsertionPass()->run(function, context, analyses);
    EXPECT_EQ(counts(entry), (std::pair{0, 0}));
    EXPECT_EQ(counts(first), (std::pair{1, 1}));
    EXPECT_EQ(counts(merge), (std::pair{0, 0}));

    std::vector<int> firstWaits;
    for (IRBase& node : *first)
        if (auto* inst = dyn_cast<StinkyInstruction>(&node))
            if (const auto* wait = inst->getModifier<SWaitTensorCntData>())
                firstWaits.push_back(wait->tlcnt);
    EXPECT_EQ(firstWaits, (std::vector<int>{1}));
}

TEST(GirFrameAnalysisTest, UnwaitedRawOwnershipDropsFromWarFence) {
    Function function("raw_to_war_fence");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* copyA = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copyA->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* firstReadA = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    firstReadA->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* readB = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 4, 24);
    readB->addModifier<GirActionData>(
        GirActionData{2, 0, GirActionKind::Read, {GirAccessData{false, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* secondReadA = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 8, 28);
    secondReadA->addModifier<GirActionData>(
        GirActionData{3, 0, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* copyB = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    copyB->addModifier<GirActionData>(
        GirActionData{4, 0, GirActionKind::Copy, {GirAccessData{true, 1, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirFencePlacementPass()->run(function, context, analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    std::vector<std::string> stamps;
    int tensorWaits = 0;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        if (isBarrierWait(*inst)) {
            const auto* comment = inst->getModifier<CommentData>();
            ASSERT_NE(comment, nullptr);
            stamps.push_back(comment->comment);
        }
        tensorWaits += inst->getModifier<SWaitTensorCntData>() != nullptr;
    }
    EXPECT_EQ(stamps, (std::vector<std::string>{"GIR fence (RAW)", "GIR fence (WAR)"}));
    EXPECT_EQ(tensorWaits, 1);
}

TEST(GirFrameAnalysisTest, UnwaitedRawRemovalReclosesWarCut) {
    Function function("raw_marker_cuts_branch_war");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* first = function.createBasicBlock("first");
    BasicBlock* merge = function.createBasicBlock("merge");
    BasicBlock* left = function.createBasicBlock("left");
    BasicBlock* right = function.createBasicBlock("right");
    BasicBlock* join = function.createBasicBlock("join");
    function.addEdge(entry, first);
    function.addEdge(first, merge);
    function.addEdge(merge, left);
    function.addEdge(merge, right);
    function.addEdge(left, join);
    function.addEdge(right, join);

    StinkyInstruction* copyA = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    copyA->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* copyB = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    copyB->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Copy, {GirAccessData{true, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* filler = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 32, 40);
    filler->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Copy, {GirAccessData{true, 2, 1, -1, 0, 0, true}}});
    StinkyInstruction* copyY = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 48, 56);
    copyY->addModifier<GirActionData>(
        GirActionData{3, 3, GirActionKind::Copy, {GirAccessData{true, 3, 1, -1, 0, 0, true}}});

    StinkyInstruction* readB = createDsReadB128InBlock(first, GfxArchID::Gfx1250, 4, 24);
    readB->addModifier<GirActionData>(
        GirActionData{4, 4, GirActionKind::Read, {GirAccessData{false, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* readX = createDsReadB128InBlock(merge, GfxArchID::Gfx1250, 8, 28);
    readX->addModifier<GirActionData>(
        GirActionData{5, 5, GirActionKind::Read, {GirAccessData{false, 4, 1, -1, 0, 0, true}}});
    StinkyInstruction* readA = createDsReadB128InBlock(merge, GfxArchID::Gfx1250, 0, 20);
    readA->addModifier<GirActionData>(
        GirActionData{6, 6, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* readY = createDsReadB128InBlock(right, GfxArchID::Gfx1250, 12, 32);
    readY->addModifier<GirActionData>(
        GirActionData{7, 7, GirActionKind::Read, {GirAccessData{false, 3, 1, -1, 0, 0, true}}});
    StinkyInstruction* copyX = createTensorLoadInBlock(join, GfxArchID::Gfx1250, 64, 72);
    copyX->addModifier<GirActionData>(
        GirActionData{8, 8, GirActionKind::Copy, {GirAccessData{true, 4, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirFencePlacementPass()->run(function, context, analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    const auto stampsIn = [](BasicBlock* block) {
        std::vector<std::string> result;
        for (IRBase& node : *block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (!inst || !isBarrierWait(*inst)) continue;
            const auto* comment = inst->getModifier<CommentData>();
            if (comment) result.push_back(comment->comment);
        }
        return result;
    };
    EXPECT_EQ(stampsIn(first), (std::vector<std::string>{"GIR fence (RAW)"}));
    EXPECT_TRUE(stampsIn(merge).empty());
    EXPECT_TRUE(stampsIn(left).empty());
    EXPECT_EQ(stampsIn(right), (std::vector<std::string>{"GIR fence (RAW)"}));
    EXPECT_EQ(stampsIn(join), (std::vector<std::string>{"GIR fence (WAR)"}));

    StinkyInstruction* beforeCopyX = nullptr;
    for (IRBase& node : *join) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == copyX) break;
        beforeCopyX = inst;
    }
    ASSERT_NE(beforeCopyX, nullptr);
    EXPECT_TRUE(isBarrierWait(*beforeCopyX));
}

TEST(GirFrameAnalysisTest, CoLocatedRawsShareOneFence) {
    Function function("colocated_raws");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* first = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    first->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* second = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    second->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Copy, {GirAccessData{true, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(GirActionData{
        2,
        0,
        GirActionKind::Read,
        {GirAccessData{false, 0, 1, -1, 0, 0, true}, GirAccessData{false, 1, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirFencePlacementPass()->run(function, context, analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    int signals = 0;
    int tensorWaits = 0;
    const CommentData* stamp = nullptr;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        signals += isBarrierSignal(*inst);
        tensorWaits += inst->getModifier<SWaitTensorCntData>() != nullptr;
        if (isBarrierWait(*inst)) stamp = inst->getModifier<CommentData>();
    }
    EXPECT_EQ(signals, 1);
    EXPECT_EQ(tensorWaits, 1);
    ASSERT_NE(stamp, nullptr);
    EXPECT_EQ(stamp->comment, "GIR fence (RAW)");
}

TEST(GirFrameAnalysisTest, SharedFenceStampsEveryOwnedHazardKind) {
    Function function("stamped_fence");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* copyA = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copyA->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* firstCopyC = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    firstCopyC->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Copy, {GirAccessData{true, 2, 1, -1, 0, 0, true}}});
    StinkyInstruction* readB = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 4, 24);
    readB->addModifier<GirActionData>(
        GirActionData{2, 0, GirActionKind::Read, {GirAccessData{false, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* readA = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    readA->addModifier<GirActionData>(
        GirActionData{3, 0, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* copyB = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 32, 40);
    copyB->addModifier<GirActionData>(
        GirActionData{4, 0, GirActionKind::Copy, {GirAccessData{true, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* secondCopyC = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 48, 56);
    secondCopyC->addModifier<GirActionData>(
        GirActionData{5, 0, GirActionKind::Copy, {GirAccessData{true, 2, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    std::set<GirHazardKind> kinds;
    for (const GirFrameHazard& hazard : hazards.hazards)
        if (hazard.crossAgent) kinds.insert(hazard.kind);
    EXPECT_EQ(kinds, (std::set{GirHazardKind::RAW, GirHazardKind::WAR, GirHazardKind::WAW}));

    createGirFencePlacementPass()->run(function, context, analyses);

    int signals = 0;
    const CommentData* stamp = nullptr;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        signals += isBarrierSignal(*inst);
        if (isBarrierWait(*inst)) stamp = inst->getModifier<CommentData>();
    }
    EXPECT_EQ(signals, 1);
    ASSERT_NE(stamp, nullptr);
    EXPECT_EQ(stamp->comment, "GIR fence (RAW,WAR,WAW)");
}

TEST(GirFrameAnalysisTest, WarBarriersMergeAtLaterSlotWithoutTensorIssue) {
    Function function("merge_war_fences");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* readA = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    readA->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* readB = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 4, 24);
    readB->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Read, {GirAccessData{false, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* copyA = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copyA->addModifier<GirActionData>(
        GirActionData{2, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* copyB = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    copyB->addModifier<GirActionData>(
        GirActionData{3, 0, GirActionKind::Copy, {GirAccessData{true, 1, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirFencePlacementPass()->run(function, context, analyses);

    int signals = 0;
    const CommentData* stamp = nullptr;
    StinkyInstruction* instructionAfterWait = nullptr;
    bool sawWait = false;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (!inst) continue;
        signals += isBarrierSignal(*inst);
        if (isBarrierWait(*inst)) {
            stamp = inst->getModifier<CommentData>();
            sawWait = true;
        } else if (sawWait && !instructionAfterWait) {
            instructionAfterWait = inst;
        }
    }
    EXPECT_EQ(signals, 1);
    EXPECT_EQ(instructionAfterWait, copyA);
    ASSERT_NE(stamp, nullptr);
    EXPECT_EQ(stamp->comment, "GIR fence (WAR)");
}

TEST(GirFrameAnalysisTest, TensorIssueKeepsWarBarriersSeparate) {
    Function function("split_war_fences");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* readA = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    readA->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* copyA = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copyA->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* readB = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 4, 24);
    readB->addModifier<GirActionData>(
        GirActionData{2, 0, GirActionKind::Read, {GirAccessData{false, 1, 1, -1, 0, 0, true}}});
    StinkyInstruction* copyB = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 16, 24);
    copyB->addModifier<GirActionData>(
        GirActionData{3, 0, GirActionKind::Copy, {GirAccessData{true, 1, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirFencePlacementPass()->run(function, context, analyses);

    int signals = 0;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst && isBarrierSignal(*inst)) ++signals;
    }
    EXPECT_EQ(signals, 2);
}

TEST(GirFrameAnalysisTest, HazardBecomesAnOrdinaryDagEdge) {
    Function function("dag");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);

    dag::RegionDAG graph = dag::buildRegisterDependencyDAG({copy, read});
    GirFrameHazardAnalysis::Result hazards;
    hazards.hazards.push_back({GirHazardKind::RAW, copy, read, block, block, 0, 1, GirFrame{},
                               GirFrame{}, 0, false, 0, 1});
    GirFrameAnalysis::Result frames;
    dag::addGirFrameHazardEdges(graph, hazards, frames);

    ASSERT_TRUE(graph.graph[0].contains(1));
    EXPECT_EQ(graph.nodes[0].inDegree, 0u);
    EXPECT_EQ(graph.nodes[1].inDegree, 1u);
}

TEST(GirFrameAnalysisTest, FrameSchedulingDoesNotMaterializeLegacyMemoryTokenRegisters) {
    Function function("frame_without_tokens");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    copy->addModifier<MemTokenData>(MemTokenData{std::vector<int>{7}});
    read->addModifier<MemTokenData>(MemTokenData{std::vector<int>{7}});

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createStinkyBuildImplicitDependencyPass(false)->run(function, context, analyses);

    auto hasLdsRegister = [](const auto& regs) {
        return std::any_of(regs.begin(), regs.end(), [](const StinkyRegister& reg) {
            return reg.isRegister() && reg.reg.type == RegType::LDS;
        });
    };
    EXPECT_FALSE(hasLdsRegister(copy->getSrcRegs()));
    EXPECT_FALSE(hasLdsRegister(copy->getDestRegs()));
    EXPECT_FALSE(hasLdsRegister(read->getSrcRegs()));
    EXPECT_FALSE(hasLdsRegister(read->getDestRegs()));
}

TEST(GirFrameAnalysisTest, FrameWaitPassInsertsTensorWaitAtTaggedConsumer) {
    Function function("wait");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);
    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeRead = nullptr;
    for (IRBase& node : *loop) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == read) break;
        beforeRead = inst;
    }
    ASSERT_NE(beforeRead, nullptr);
    const auto* wait = beforeRead->getModifier<SWaitTensorCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_GE(wait->tlcnt, 0);
}

TEST(GirFrameAnalysisTest, JoinIgnoresAbsentPathWithoutInventingPromotion) {
    Function function("diamond");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* producing = function.createBasicBlock("producing");
    BasicBlock* empty = function.createBasicBlock("empty");
    BasicBlock* join = function.createBasicBlock("join");
    function.addEdge(entry, producing);
    function.addEdge(entry, empty);
    function.addEdge(producing, join);
    function.addEdge(empty, join);

    StinkyInstruction* copy = createTensorLoadInBlock(producing, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, false}}});
    StinkyInstruction* read = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, false}}});
    function.setStringMetaData(kGirFrameContractKey, kDiamondContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    bool producingWait = false;
    for (IRBase& node : *producing) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        producingWait |= inst && inst->getModifier<SWaitTensorCntData>();
    }
    bool joinWait = false;
    for (IRBase& node : *join) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        joinWait |= inst && inst->getModifier<SWaitTensorCntData>();
    }
    EXPECT_FALSE(producingWait);
    EXPECT_TRUE(joinWait);
}

TEST(GirFrameAnalysisTest, FrameWarRetiresVacatingDsRead) {
    Function function("war");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, false}}});
    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, false}}});
    function.setStringMetaData(kGirFrameContractKey, kWarContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeCopy = nullptr;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == copy) break;
        beforeCopy = inst;
    }
    ASSERT_NE(beforeCopy, nullptr);
    const auto* wait = beforeCopy->getModifier<SWaitCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_GE(wait->dlcnt, 0);
}

TEST(GirFrameAnalysisTest, SameTripRingAliasWarDrainsBeforeOverwrite) {
    Function function("same_trip_ring_alias_war");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 1);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Copy, {GirAccessData{true, 0, 2, 0, 2, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);

    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    auto war = std::find_if(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::WAR && hazard.producer == read &&
                   hazard.consumer == copy && hazard.gap == 0 && !hazard.crossAgent;
        });
    ASSERT_NE(war, hazards.hazards.end());

    createGirFencePlacementPass()->run(function, context, analyses);
    EXPECT_TRUE(std::none_of(loop->begin(), loop->end(), [](IRBase& node) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        return inst && isBarrier(*inst);
    }));

    createGirWaitCntInsertionPass()->run(function, context, analyses);
    StinkyInstruction* beforeCopy = nullptr;
    for (IRBase& node : *loop) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == copy) break;
        beforeCopy = inst;
    }
    ASSERT_NE(beforeCopy, nullptr);
    const auto* wait = beforeCopy->getModifier<SWaitCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->dlcnt, 0);
}

TEST(GirFrameAnalysisTest, CrossAgentRingAliasWarDrainsBeforeBarrier) {
    Function function("multi_wave_ring_alias_war");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 0, -1, true}}});
    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Copy, {GirAccessData{true, 0, 2, 0, 2, -1, true}}});
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    auto war = std::find_if(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::WAR && hazard.producer == read &&
                   hazard.consumer == copy && hazard.gap == 0 && hazard.crossAgent;
        });
    ASSERT_NE(war, hazards.hazards.end());

    createGirFencePlacementPass()->run(function, context, analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    std::vector<StinkyInstruction*> instructions;
    for (IRBase& node : *loop)
        if (auto* inst = dyn_cast<StinkyInstruction>(&node)) instructions.push_back(inst);

    bool allWarSignalsDrained = true;
    int warSignals = 0;
    for (size_t i = 0; i + 1 < instructions.size(); ++i) {
        if (!isBarrierSignal(*instructions[i]) || !isBarrierWait(*instructions[i + 1])) continue;
        const auto* stamp = instructions[i + 1]->getModifier<CommentData>();
        if (!stamp || stamp->comment.find("WAR") == std::string::npos) continue;
        const auto* wait = i > 0 ? instructions[i - 1]->getModifier<SWaitCntData>() : nullptr;
        allWarSignalsDrained &= wait && wait->dlcnt == 0;
        ++warSignals;
    }
    EXPECT_GT(warSignals, 0);
    EXPECT_TRUE(allWarSignalsDrained);
}

// The read is late in one trip and the overwriting copy is at the next trip's head. The enclosing
// fence is at the predecessor tail, one frame edge before the consumer; reusing the consumer gap
// for its counter walk skips this occurrence and silently emits no DS drain.
TEST(GirFrameAnalysisTest, LoopCarriedWarDrainsAtPredecessorTailFence) {
    Function function("loop_carried_war_tail_fence");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 2, 0, 2, -1, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 1, -1, true}}});
    AsmIRBuilder builder(*loop, GfxArchID::Gfx1250);
    StinkyInstruction* fence = builder.createFence();
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    EXPECT_EQ(frames.occurrences.size(), 4u);
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    auto war = std::find_if(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::WAR && hazard.producer == read &&
                   hazard.consumer == copy && hazard.crossAgent;
        });
    ASSERT_NE(war, hazards.hazards.end());
    EXPECT_EQ(war->gap, 1);

    createGirWaitCntInsertionPass()->run(function, context, analyses);

    const SWaitCntData* dsWait = nullptr;
    for (IRBase& node : *loop) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == fence) break;
        if (inst) {
            if (const auto* wait = inst->getModifier<SWaitCntData>()) dsWait = wait;
        }
    }
    ASSERT_NE(dsWait, nullptr);
    EXPECT_EQ(dsWait->dlcnt, 0);
}

TEST(GirFrameAnalysisTest, EntryAndBackedgeWarUseDifferentFences) {
    Function function("entry_and_backedge_war");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(entry, loop);
    function.addEdge(loop, loop);

    StinkyInstruction* entryRead = createDsReadB128InBlock(entry, GfxArchID::Gfx1250, 0, 20);
    entryRead->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 0, -1, true}}});
    AsmIRBuilder entryBuilder(*entry, GfxArchID::Gfx1250);
    StinkyInstruction* entryFence = entryBuilder.createFence();

    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Copy, {GirAccessData{true, 0, 2, 0, 0, -1, true}}});
    StinkyInstruction* loopRead = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 4, 24);
    loopRead->addModifier<GirActionData>(
        GirActionData{2, 1, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 1, -1, true}}});
    AsmIRBuilder loopBuilder(*loop, GfxArchID::Gfx1250);
    StinkyInstruction* tailFence = loopBuilder.createFence();
    function.setStringMetaData(kGirFrameContractKey, kRingTwoContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);
    const auto drainBefore = [](BasicBlock* block, const StinkyInstruction* anchor) {
        const SWaitCntData* result = nullptr;
        for (IRBase& node : *block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst == anchor) break;
            if (inst && inst->getModifier<SWaitCntData>())
                result = inst->getModifier<SWaitCntData>();
        }
        return result;
    };
    const SWaitCntData* entryDrain = drainBefore(entry, entryFence);
    const SWaitCntData* tailDrain = drainBefore(loop, tailFence);
    ASSERT_NE(entryDrain, nullptr);
    ASSERT_NE(tailDrain, nullptr);
    EXPECT_EQ(entryDrain->dlcnt, 0);
    EXPECT_EQ(tailDrain->dlcnt, 0);
}

TEST(GirFrameAnalysisTest, LoopWrapStillUsesUnreducedGenerationSpan) {
    Function function("loop_wrap_generation_span");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* loop = function.createBasicBlock("loop");
    function.addEdge(loop, loop);

    StinkyInstruction* read = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 2, -1, false}}});
    StinkyInstruction* copy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Copy, {GirAccessData{true, 0, 2, 0, 0, -1, false}}});
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

TEST(GirFrameAnalysisTest, CrossAgentWaitAnchorsAtVirtualFence) {
    Function function("fence");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* block = function.createBasicBlock("entry");
    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    AsmIRBuilder builder(*block, GfxArchID::Gfx1250);
    StinkyInstruction* fence = builder.createFence();
    fence->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Fence, {}});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeFence = nullptr;
    StinkyInstruction* beforeRead = nullptr;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == fence) beforeFence = beforeRead;
        if (inst == read) break;
        beforeRead = inst;
    }
    ASSERT_NE(beforeFence, nullptr);
    EXPECT_NE(beforeFence->getModifier<SWaitTensorCntData>(), nullptr);
    EXPECT_EQ(beforeRead, fence);
}

TEST(GirFrameAnalysisTest, SingleWaveUsesWaitWithoutWorkgroupBarrier) {
    Function function("single_wave");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 1);
    BasicBlock* block = function.createBasicBlock("entry");

    StinkyInstruction* copy = createTensorLoadInBlock(block, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});
    StinkyInstruction* read = createDsReadB128InBlock(block, GfxArchID::Gfx1250, 0, 20);
    read->addModifier<GirActionData>(
        GirActionData{1, 0, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    ASSERT_FALSE(hazards.hazards.empty());
    EXPECT_TRUE(std::none_of(hazards.hazards.begin(), hazards.hazards.end(),
                             [](const GirFrameHazard& hazard) { return hazard.crossAgent; }));

    createGirFencePlacementPass()->run(function, context, analyses);
    EXPECT_TRUE(std::none_of(block->begin(), block->end(), [](IRBase& node) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        return inst && isBarrier(*inst);
    }));

    createGirWaitCntInsertionPass()->run(function, context, analyses);
    StinkyInstruction* beforeRead = nullptr;
    for (IRBase& node : *block) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == read) break;
        beforeRead = inst;
    }
    ASSERT_NE(beforeRead, nullptr);
    const auto* wait = beforeRead->getModifier<SWaitTensorCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->tlcnt, 0);
}

TEST(GirFrameAnalysisTest, ClonedFenceActionAnchorsWaitOnEachCfgPath) {
    Function function("cloned_fence");
    setFunctionArch(function, GfxArchID::Gfx1250);
    setFunctionNumWaves(function, 2);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* left = function.createBasicBlock("left");
    BasicBlock* right = function.createBasicBlock("right");
    function.addEdge(entry, left);
    function.addEdge(entry, right);

    StinkyInstruction* copy = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    copy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, true}}});

    AsmIRBuilder leftBuilder(*left, GfxArchID::Gfx1250);
    StinkyInstruction* leftFence = leftBuilder.createFence();
    leftFence->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Fence, {}});
    StinkyInstruction* leftRead = createDsReadB128InBlock(left, GfxArchID::Gfx1250, 0, 20);
    leftRead->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});

    AsmIRBuilder rightBuilder(*right, GfxArchID::Gfx1250);
    StinkyInstruction* rightFence = rightBuilder.createFence();
    rightFence->addModifier<GirActionData>(GirActionData{1, 1, GirActionKind::Fence, {}});
    StinkyInstruction* rightRead = createDsReadB128InBlock(right, GfxArchID::Gfx1250, 0, 20);
    rightRead->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, true}}});
    function.setStringMetaData(kGirFrameContractKey, kFenceContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    auto waitImmediatelyBefore = [](BasicBlock* block, StinkyInstruction* anchor) {
        StinkyInstruction* prior = nullptr;
        for (IRBase& node : *block) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst == anchor) break;
            prior = inst;
        }
        return prior ? prior->getModifier<SWaitTensorCntData>() : nullptr;
    };
    EXPECT_NE(waitImmediatelyBefore(left, leftFence), nullptr);
    EXPECT_NE(waitImmediatelyBefore(right, rightFence), nullptr);
}

TEST(GirFrameAnalysisTest, EntrancePhiSurvivesUntaggedScaffoldBlocks) {
    Function function("entrance_phi");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* bypass = function.createBasicBlock("bypass");
    BasicBlock* loop = function.createBasicBlock("loop");
    BasicBlock* drain = function.createBasicBlock("drain");
    function.addEdge(entry, bypass);
    function.addEdge(entry, loop);
    function.addEdge(bypass, drain);
    function.addEdge(loop, loop);
    function.addEdge(loop, drain);

    StinkyInstruction* prologueCopy = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    prologueCopy->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 2, -1, 0, 0, false}}});
    AsmIRBuilder bypassBuilder(*bypass, GfxArchID::Gfx1250);
    bypassBuilder.create(getMCIDByUOp(GFX::s_nop, GfxArchID::Gfx1250));

    StinkyInstruction* loopCopy = createTensorLoadInBlock(loop, GfxArchID::Gfx1250, 16, 24);
    loopCopy->addModifier<GirActionData>(
        GirActionData{3, 3, GirActionKind::Copy, {GirAccessData{true, 0, 2, 0, 1, -1, false}}});
    StinkyInstruction* drainRead = createDsReadB128InBlock(drain, GfxArchID::Gfx1250, 0, 20);
    drainRead->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, -1, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kEntrancePhiContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(drain).size(), 2u);
    EXPECT_TRUE(std::any_of(frames.frames(drain).begin(), frames.frames(drain).end(),
                            [](const GirFrame& frame) { return frame.phaseOf(0) == 1; }));
    const auto& hazards = analyses.getResult<GirFrameHazardAnalysis>(function);
    EXPECT_TRUE(std::any_of(
        hazards.hazards.begin(), hazards.hazards.end(), [&](const GirFrameHazard& hazard) {
            return hazard.kind == GirHazardKind::RAW && hazard.producer == prologueCopy &&
                   hazard.consumer == drainRead;
        }));

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    createGirWaitCntInsertionPass()->run(function, context, analyses);
    StinkyInstruction* beforeRead = nullptr;
    for (IRBase& node : *drain) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == drainRead) break;
        beforeRead = inst;
    }
    ASSERT_NE(beforeRead, nullptr);
    EXPECT_NE(beforeRead->getModifier<SWaitTensorCntData>(), nullptr);
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
        GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, 0, 2, 0, 0, -1, false}}});
    function.setStringMetaData(kGirFrameContractKey, kRingWrapEntranceContract);

    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    const auto& frames = analyses.getResult<GirFrameAnalysis>(function);
    ASSERT_EQ(frames.frames(destination).size(), 2u);
    EXPECT_EQ(frames.frames(destination)[0].phaseOf(0), 0);
    EXPECT_EQ(frames.frames(destination)[1].phaseOf(0), 1);
}

TEST(GirFrameAnalysisTest, TripDomainExcludesShortPrefetchPathFromSteadyWaitRank) {
    Function function("trip_domain");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* shortPath = function.createBasicBlock("label_skipPGR2_1");
    BasicBlock* fullPath = function.createBasicBlock("entry_1");
    BasicBlock* guard = function.createBasicBlock("label_openLoopL_1");
    BasicBlock* loop = function.createBasicBlock("label_LoopBeginL");
    BasicBlock* exit = function.createBasicBlock("label_LoopEndL");
    function.addEdge(entry, shortPath);
    function.addEdge(entry, fullPath);
    function.addEdge(shortPath, guard);
    function.addEdge(fullPath, guard);
    function.addEdge(guard, exit);
    function.addEdge(guard, loop);

    StinkyInstruction* producer = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    producer->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, false}}});
    createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 32, 40);
    {
        AsmIRBuilder builder(*entry, GfxArchID::Gfx1250);
        StinkyInstruction* compare =
            builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, GfxArchID::Gfx1250));
        compare->addSrcReg(StinkyRegister("s", 14, 1));
        compare->addSrcReg(StinkyRegister("0x1"));
        StinkyInstruction* branch =
            builder.create(getMCIDByUOp(GFX::s_cbranch_scc1, GfxArchID::Gfx1250));
        branch->addModifier<LabelData>(LabelData{"label_skipPGR2_1"});
    }

    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 48, 56);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 64, 72);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 80, 88);
    {
        AsmIRBuilder builder(*guard, GfxArchID::Gfx1250);
        StinkyInstruction* compare =
            builder.create(getMCIDByUOp(GFX::s_cmp_le_u32, GfxArchID::Gfx1250));
        compare->addSrcReg(StinkyRegister("s", 14, 1));
        compare->addSrcReg(StinkyRegister("0x2"));
        StinkyInstruction* branch =
            builder.create(getMCIDByUOp(GFX::s_cbranch_scc1, GfxArchID::Gfx1250));
        branch->addModifier<LabelData>(LabelData{"label_LoopEndL"});
    }

    StinkyInstruction* consumer = createDsReadB128InBlock(loop, GfxArchID::Gfx1250, 0, 20);
    consumer->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, false}}});
    function.setStringMetaData(kGirFrameContractKey, kDiamondContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    StinkyInstruction* beforeConsumer = nullptr;
    for (IRBase& node : *loop) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst == consumer) break;
        beforeConsumer = inst;
    }
    ASSERT_NE(beforeConsumer, nullptr);
    const auto* wait = beforeConsumer->getModifier<SWaitTensorCntData>();
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->tlcnt, 5);
}

TEST(GirFrameAnalysisTest, PredecessorPromotionRerunsDownstreamCounterflow) {
    Function function("promotion");
    setFunctionArch(function, GfxArchID::Gfx1250);
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* shortPath = function.createBasicBlock("short");
    BasicBlock* fullPath = function.createBasicBlock("full");
    BasicBlock* join = function.createBasicBlock("join");
    function.addEdge(entry, shortPath);
    function.addEdge(entry, fullPath);
    function.addEdge(shortPath, join);
    function.addEdge(fullPath, join);

    StinkyInstruction* mx = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 0, 8);
    mx->addModifier<GirActionData>(
        GirActionData{0, 0, GirActionKind::Copy, {GirAccessData{true, 3, 1, -1, 0, 0, false}}});
    StinkyInstruction* a = createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 16, 24);
    a->addModifier<GirActionData>(
        GirActionData{1, 1, GirActionKind::Copy, {GirAccessData{true, 0, 1, -1, 0, 0, false}}});
    createTensorLoadInBlock(entry, GfxArchID::Gfx1250, 32, 40);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 48, 56);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 64, 72);
    createTensorLoadInBlock(fullPath, GfxArchID::Gfx1250, 80, 88);

    StinkyInstruction* mxRead = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 0, 20);
    mxRead->addModifier<GirActionData>(
        GirActionData{2, 2, GirActionKind::Read, {GirAccessData{false, 3, 1, -1, 0, 0, false}}});
    StinkyInstruction* aRead = createDsReadB128InBlock(join, GfxArchID::Gfx1250, 4, 24);
    aRead->addModifier<GirActionData>(
        GirActionData{3, 3, GirActionKind::Read, {GirAccessData{false, 0, 1, -1, 0, 0, false}}});
    function.setStringMetaData(kGirFrameContractKey, kPromotionContract);

    PassContext context;
    context.setGemmTileConfig(function.getGemmTileConfig());
    AnalysisManager analyses;
    registerAllAnalyses(analyses);
    createGirWaitCntInsertionPass()->run(function, context, analyses);

    const SWaitTensorCntData* shortWait = nullptr;
    for (IRBase& node : *shortPath) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst && inst->getModifier<SWaitTensorCntData>())
            shortWait = inst->getModifier<SWaitTensorCntData>();
    }
    ASSERT_NE(shortWait, nullptr);
    EXPECT_EQ(shortWait->tlcnt, 1);

    std::vector<int> joinWaits;
    for (IRBase& node : *join) {
        auto* inst = dyn_cast<StinkyInstruction>(&node);
        if (inst && inst->getModifier<SWaitTensorCntData>())
            joinWaits.push_back(inst->getModifier<SWaitTensorCntData>()->tlcnt);
    }
    EXPECT_EQ(joinWaits, (std::vector<int>{5, 4}));
}
