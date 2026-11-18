// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The allocation report's s_set_vgpr_msb prediction, against the rules
// InsertVgprMsbPass emits by: a switch whenever the next VGPR instruction needs
// another bank word, with every block, label and call starting from none.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "stinkytofu/transforms/asm/ra/VgprMsbPrediction.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

/// What an operand can encode on the test architecture.
constexpr uint32_t kEncodableVgprs = 1024;

/// v[dst:dst+7] = v_wmma_f32_16x16x32_bf16(v[a:a+7], v[b:b+7], v[c:c+7]): one
/// operand in each bank slot, src0, src1, src2 and dst.
StinkyInstruction* wmma(BasicBlock* bb, int dst, int a, int b, int c) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* inst =
        builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, kRaTestArch));
    inst->addDestReg(StinkyRegister("v", dst, 8));
    inst->addSrcReg(StinkyRegister("v", a, 8));
    inst->addSrcReg(StinkyRegister("v", b, 8));
    inst->addSrcReg(StinkyRegister("v", c, 8));
    return inst;
}

/// v[dst:dst+3] = ds_load_b128(v[addr]): src0 and dst only.
StinkyInstruction* dsLoad(BasicBlock* bb, int dst, int addr) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::ds_load_b128, kRaTestArch));
    inst->addDestReg(StinkyRegister("v", dst, 4));
    inst->addSrcReg(StinkyRegister("v", addr, 1));
    return inst;
}

/// A WMMA, which fills every slot; a ds_load whose two slots the WMMA's word
/// already holds; and the WMMA again. Three switches under the exact rule, one
/// when the slots the ds_load does not use are don't-care.
void wmmaLoadWmma(BasicBlock* bb) {
    wmma(bb, /*dst=*/256, /*a=*/0, /*b=*/512, /*c=*/256);
    dsLoad(bb, /*dst=*/264, /*addr=*/40);
    wmma(bb, /*dst=*/256, /*a=*/0, /*b=*/512, /*c=*/256);
}

/// The epilogue's shape: an instruction that leaves slots unread, here a
/// ds_load reading only src0 and dst, then one needing those slots in other
/// banks, a WMMA with B in bank 2 and C in bank 1. Two switches under the exact
/// rule, and two if the load's setter only kept the slots it did not set; one
/// when that setter takes the WMMA's banks.
void loadThenWmma(BasicBlock* bb) {
    dsLoad(bb, /*dst=*/264, /*addr=*/40);
    wmma(bb, /*dst=*/256, /*a=*/0, /*b=*/512, /*c=*/256);
}

/// The s_set_vgpr_msb InsertVgprMsbPass emits into one block built by \p build,
/// under \p policy.
size_t switchesThePassEmits(void (*build)(BasicBlock*), VgprMsbPolicy policy) {
    Function function("kernel");
    setFunctionArch(function, kRaTestArch);
    BasicBlock* only = function.createBasicBlock("only");
    build(only);
    PassContext passCtx;
    GemmTileConfig tile;
    tile.arch = {12, 5, 0};
    passCtx.setGemmTileConfig(tile);
    AsmCapsConfig caps;
    caps.vgprMsbMode = VgprMsbMode::Msb16;
    caps.vgprMsbUnusedSlotsDontCare = policy == VgprMsbPolicy::UnusedSlotsDontCare;
    passCtx.setAsmCapsConfig(caps);
    AnalysisManager am;
    registerAllAnalyses(am);
    createInsertVgprMsbPass()->run(function, passCtx, am);
    size_t switches = 0;
    for (const IRBase& ir : *only) {
        const auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst != nullptr && inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) ++switches;
    }
    return switches;
}

/// \p from, with every unit of source operand \p operand of \p inst moved to
/// consecutive registers from \p base.
AllocationResult withSourceAt(const AllocationResult& from, const StinkyInstruction& inst,
                              size_t operand, uint32_t base) {
    AllocationResult moved = from;
    const std::vector<StinkySSAValue*> units = ssaSourceUnits(inst, operand);
    for (size_t unit = 0; unit < units.size(); ++unit) {
        moved.assign(units[unit]->valueId(),
                     RegKey{RegType::V, base + static_cast<uint32_t>(unit), RegHalf::NONE});
    }
    return moved;
}

class VgprMsbPredictionTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
        entry = func->createBasicBlock("entry");
        loop = func->createBasicBlock("loop");
    }

    /// entry -> loop -> loop, and out to exit.
    void selfLoop() {
        exitBlock = func->createBasicBlock("exit");
        func->addEdge(entry, loop);
        func->addEdge(loop, loop);
        func->addEdge(loop, exitBlock);
    }

    /// entry -> loop -> body -> loop, and out of body to exit.
    void twoBlockLoop() {
        body = func->createBasicBlock("body");
        exitBlock = func->createBasicBlock("exit");
        func->addEdge(entry, loop);
        func->addEdge(loop, body);
        func->addEdge(body, loop);
        func->addEdge(body, exitBlock);
    }

    /// The prediction for the function's one loop, headed by "loop".
    std::optional<uint32_t> predicted(const AllocationResult& colouring,
                                      VgprMsbPolicy policy = VgprMsbPolicy::ExactWord) {
        const std::vector<Loop> loops = detectLoops(*func);
        const std::vector<LoopVgprMsbSwitches> found =
            predictLoopVgprMsbSwitches(*func, colouring, loops, kEncodableVgprs, policy);
        if (found.size() != 1) {
            ADD_FAILURE() << "expected one loop, found " << found.size();
            return std::nullopt;
        }
        EXPECT_EQ(found.front().header, "loop");
        return found.front().switches;
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
    BasicBlock* loop = nullptr;
    BasicBlock* body = nullptr;
    BasicBlock* exitBlock = nullptr;
};

}  // namespace

TEST_F(VgprMsbPredictionTest, CountsEachChangeOfTheBankWord) {
    // The first instruction sets a word, since the pass starts each block with
    // none. The second needs every slot from the same bank as the first; the
    // third moves src1 from bank 2 to bank 3.
    selfLoop();
    wmma(loop, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
    wmma(loop, /*dst=*/8, /*a=*/264, /*b=*/520, /*c=*/8);
    wmma(loop, /*dst=*/16, /*a=*/272, /*b=*/768, /*c=*/16);
    ASSERT_TRUE(liftForAllocation(*func));

    EXPECT_EQ(predicted(createLegacyColoring(*func)), 2u);
}

TEST_F(VgprMsbPredictionTest, EachBlockStartsWithoutAWord) {
    // Two instructions needing one word, but in two blocks of the loop: the pass
    // emits at the top of each.
    twoBlockLoop();
    wmma(loop, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
    wmma(body, /*dst=*/8, /*a=*/264, /*b=*/520, /*c=*/8);
    ASSERT_TRUE(liftForAllocation(*func));

    EXPECT_EQ(predicted(createLegacyColoring(*func)), 2u);
}

TEST_F(VgprMsbPredictionTest, ALabelForgetsTheWord) {
    // A label inside a block is a branch target, so the pass cannot know what
    // word arrives there and sets it again.
    selfLoop();
    wmma(loop, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
    AsmIRBuilder(*loop, kRaTestArch).createLabel("mid");
    wmma(loop, /*dst=*/8, /*a=*/264, /*b=*/520, /*c=*/8);
    ASSERT_TRUE(liftForAllocation(*func));

    EXPECT_EQ(predicted(createLegacyColoring(*func)), 2u);
}

TEST_F(VgprMsbPredictionTest, ACallForgetsTheWord) {
    // A callee may leave any word behind. Lifting refuses call sites, so no
    // colouring ever covers one; without attached SSA every operand is read at
    // its own register, which is the case this exercises.
    selfLoop();
    wmma(loop, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
    StinkyInstruction* call =
        AsmIRBuilder(*loop, kRaTestArch).create(getMCIDByUOp(GFX::s_swappc_b64, kRaTestArch));
    call->addDestReg(StinkyRegister("s", 30, 2));
    call->addSrcReg(StinkyRegister("s", 32, 2));
    wmma(loop, /*dst=*/8, /*a=*/264, /*b=*/520, /*c=*/8);

    EXPECT_EQ(predicted(AllocationResult()), 2u);
}

TEST_F(VgprMsbPredictionTest, TheColouringDecidesTheBank) {
    // Moving the third instruction's src1 back into bank 2 gives all three one
    // word, without touching the operands themselves.
    selfLoop();
    wmma(loop, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
    wmma(loop, /*dst=*/8, /*a=*/264, /*b=*/520, /*c=*/8);
    StinkyInstruction* third = wmma(loop, /*dst=*/16, /*a=*/272, /*b=*/768, /*c=*/16);
    ASSERT_TRUE(liftForAllocation(*func));

    const AllocationResult producer = createLegacyColoring(*func);
    EXPECT_EQ(predicted(producer), 2u);
    EXPECT_EQ(predicted(withSourceAt(producer, *third, /*operand=*/1, /*base=*/600)), 1u);
}

TEST_F(VgprMsbPredictionTest, AnOperandPastTheFileLeavesTheLoopUnpredicted) {
    // v1024 has no bank word an s_set_vgpr_msb could select.
    selfLoop();
    StinkyInstruction* only = wmma(loop, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
    ASSERT_TRUE(liftForAllocation(*func));

    const AllocationResult producer = createLegacyColoring(*func);
    EXPECT_EQ(predicted(producer), 1u);
    EXPECT_EQ(predicted(withSourceAt(producer, *only, /*operand=*/1, /*base=*/1024)), std::nullopt);
}

TEST_F(VgprMsbPredictionTest, UnusedSlotsDontCareThePredictionStillFollowsThePass) {
    // The report counts by the pass's own policy, so the two agree under either.
    selfLoop();
    wmmaLoadWmma(loop);
    ASSERT_TRUE(liftForAllocation(*func));

    const AllocationResult producer = createLegacyColoring(*func);
    EXPECT_EQ(predicted(producer), 3u);
    EXPECT_EQ(switchesThePassEmits(wmmaLoadWmma, VgprMsbPolicy::ExactWord), 3u);
    EXPECT_EQ(predicted(producer, VgprMsbPolicy::UnusedSlotsDontCare), 1u);
    EXPECT_EQ(switchesThePassEmits(wmmaLoadWmma, VgprMsbPolicy::UnusedSlotsDontCare), 1u);
}

TEST_F(VgprMsbPredictionTest, UnusedSlotsDontCareARewrittenSetterCostsNothing) {
    // The pass rewrites the load's setter for the WMMA rather than emitting a
    // second, and the report counts only the setters emitted.
    selfLoop();
    loadThenWmma(loop);
    ASSERT_TRUE(liftForAllocation(*func));

    const AllocationResult producer = createLegacyColoring(*func);
    EXPECT_EQ(predicted(producer), 2u);
    EXPECT_EQ(switchesThePassEmits(loadThenWmma, VgprMsbPolicy::ExactWord), 2u);
    EXPECT_EQ(predicted(producer, VgprMsbPolicy::UnusedSlotsDontCare), 1u);
    EXPECT_EQ(switchesThePassEmits(loadThenWmma, VgprMsbPolicy::UnusedSlotsDontCare), 1u);
}

TEST_F(VgprMsbPredictionTest, BlocksOutsideTheLoopAreNotCounted) {
    // The entry changes word twice and runs once; only the loop is reported.
    selfLoop();
    wmma(entry, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
    wmma(entry, /*dst=*/8, /*a=*/264, /*b=*/768, /*c=*/8);
    wmma(loop, /*dst=*/16, /*a=*/272, /*b=*/520, /*c=*/16);
    wmma(exitBlock, /*dst=*/24, /*a=*/280, /*b=*/784, /*c=*/24);
    ASSERT_TRUE(liftForAllocation(*func));

    EXPECT_EQ(predicted(createLegacyColoring(*func)), 1u);
}
