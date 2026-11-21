// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The s_set_vgpr_msb stream model, against what InsertVgprMsbPass emits.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "stinkytofu/transforms/asm/ra/VgprMsbStreamModel.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

/// v[dst:dst+7] = v_wmma_f32_16x16x32_bf16(v[a:a+7], v[b:b+7], v[c:c+7]): the
/// word is (bank of a, bank of b, bank of c, bank of dst).
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

/// v[dst] = v_add_f32(v[src0], v[src1]): no src2, which the word holds as bank 0.
StinkyInstruction* vAdd(BasicBlock* bb, int dst, int src0, int src1) {
    return createVAddInBlock(bb, kRaTestArch, dst, src0, src1);
}

/// An instruction with no bank word.
void sMov(BasicBlock* bb) {
    StinkyInstruction* inst =
        AsmIRBuilder(*bb, kRaTestArch).create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
    inst->addDestReg(StinkyRegister("s", 0, 1));
    inst->addSrcReg(StinkyRegister("s", 1, 1));
}

void call(BasicBlock* bb) {
    StinkyInstruction* inst =
        AsmIRBuilder(*bb, kRaTestArch).create(getMCIDByUOp(GFX::s_swappc_b64, kRaTestArch));
    inst->addDestReg(StinkyRegister("s", 30, 2));
    inst->addSrcReg(StinkyRegister("s", 32, 2));
}

/// Each value's bank under \p colouring.
std::vector<uint8_t> banksOf(const Function& function, const AllocationResult& colouring) {
    std::vector<uint8_t> banks(function.ssaArena().valueCount() + 1, kUnknownBank);
    for (const StinkySSAValue* value : function.ssaArena().values()) {
        if (value == nullptr || !colouring.isAssigned(value->valueId())) continue;
        const RegKey reg = colouring.assignmentOf(value->valueId());
        if (reg.type == RegType::V) banks[value->valueId()] = reg.idx / kVgprBankSize;
    }
    return banks;
}

/// The s_set_vgpr_msb InsertVgprMsbPass inserts into \p function.
double emittedSwitches(Function& function) {
    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(function.getGemmTileConfig());
    AsmCapsConfig caps;
    caps.vgprMsbMode = VgprMsbMode::Msb16;
    pm.setAsmCapsConfig(caps);
    pm.addPass(createInsertVgprMsbPass());
    pm.run(function);

    double count = 0;
    for (const BasicBlock& bb : function) {
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst != nullptr && inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb) ++count;
        }
    }
    return count;
}

std::unique_ptr<Function> newFunction() {
    auto function = std::make_unique<Function>("kernel");
    setFunctionArch(*function, kRaTestArch);
    return function;
}

/// Five switches: the first instruction of each block, a VALU word that differs
/// from the WMMA's, the same VALU word again after a label, and a WMMA whose
/// src1 moves to bank 3. The scalar move and the second of each equal pair add
/// nothing.
void mixedStream(Function& function) {
    BasicBlock* entry = function.createBasicBlock("entry");
    BasicBlock* next = function.createBasicBlock("next");
    function.addEdge(entry, next);
    wmma(entry, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
    wmma(entry, /*dst=*/8, /*a=*/264, /*b=*/520, /*c=*/8);
    vAdd(entry, /*dst=*/300, /*src0=*/16, /*src1=*/600);
    sMov(entry);
    vAdd(entry, /*dst=*/301, /*src0=*/17, /*src1=*/601);
    AsmIRBuilder(*entry, kRaTestArch).createLabel("mid");
    vAdd(entry, /*dst=*/302, /*src0=*/18, /*src1=*/602);
    wmma(entry, /*dst=*/24, /*a=*/272, /*b=*/768, /*c=*/24);
    wmma(next, /*dst=*/32, /*a=*/280, /*b=*/776, /*c=*/32);
}

}  // namespace

TEST(BankCostModelTest, TheCostIsWhatTheInsertionPassEmits) {
    std::unique_ptr<Function> priced = newFunction();
    mixedStream(*priced);
    ASSERT_TRUE(liftForAllocation(*priced));
    const VgprMsbStreamModel model(*priced, detectLoops(*priced));
    const double cost = model.cost(banksOf(*priced, createLegacyColoring(*priced)));

    std::unique_ptr<Function> emitted = newFunction();
    mixedStream(*emitted);
    EXPECT_EQ(cost, emittedSwitches(*emitted));
    EXPECT_EQ(cost, 5.0);
}

TEST(BankCostModelTest, AnUnliftedFunctionIsPricedAtItsOwnRegisters) {
    // Lifting refuses call sites, so a call is priced here, with every operand
    // fixed at its register. The call forgets the word, as a label does.
    const auto build = [](Function& function) {
        BasicBlock* entry = function.createBasicBlock("entry");
        wmma(entry, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
        call(entry);
        wmma(entry, /*dst=*/8, /*a=*/264, /*b=*/520, /*c=*/8);
        wmma(entry, /*dst=*/16, /*a=*/272, /*b=*/776, /*c=*/16);
    };
    std::unique_ptr<Function> priced = newFunction();
    build(*priced);
    const VgprMsbStreamModel model(*priced, detectLoops(*priced));
    EXPECT_TRUE(model.bankSensitiveValues().empty());

    std::unique_ptr<Function> emitted = newFunction();
    build(*emitted);
    EXPECT_EQ(model.cost({}), emittedSwitches(*emitted));
    EXPECT_EQ(model.cost({}), 3.0);
}

TEST(BankCostModelTest, ASwitchInALoopWeighsTenTimesOneOutside) {
    // entry -> loop -> loop, and out to exit. The entry, the top of the loop,
    // the loop's change of src1, and the exit each switch once.
    std::unique_ptr<Function> function = newFunction();
    BasicBlock* entry = function->createBasicBlock("entry");
    BasicBlock* loop = function->createBasicBlock("loop");
    BasicBlock* exitBlock = function->createBasicBlock("exit");
    function->addEdge(entry, loop);
    function->addEdge(loop, loop);
    function->addEdge(loop, exitBlock);
    const StinkyInstruction* outside = wmma(entry, /*dst=*/0, /*a=*/256, /*b=*/512, /*c=*/0);
    const StinkyInstruction* inside = wmma(loop, /*dst=*/8, /*a=*/264, /*b=*/520, /*c=*/8);
    wmma(loop, /*dst=*/16, /*a=*/272, /*b=*/776, /*c=*/16);
    wmma(exitBlock, /*dst=*/24, /*a=*/280, /*b=*/528, /*c=*/24);
    ASSERT_TRUE(liftForAllocation(*function));
    const std::vector<uint8_t> banks = banksOf(*function, createLegacyColoring(*function));

    EXPECT_EQ(VgprMsbStreamModel(*function, {}).cost(banks), 4.0);
    const VgprMsbStreamModel model(*function, detectLoops(*function));
    EXPECT_EQ(model.cost(banks), 1.0 + 10.0 + 10.0 + 1.0);

    EXPECT_EQ(model.useWeight(ssaSourceUnits(*inside, 0).front()->valueId()), 10.0);
    EXPECT_EQ(model.useWeight(ssaSourceUnits(*outside, 0).front()->valueId()), 1.0);
    // Only unit 0 names an operand's bank.
    EXPECT_EQ(model.useWeight(ssaSourceUnits(*outside, 0).back()->valueId()), 0.0);
}

TEST(BankCostModelTest, AnUndecidedValueAgreesOnlyWithAnotherUndecidedOne) {
    std::unique_ptr<Function> function = newFunction();
    BasicBlock* entry = function->createBasicBlock("entry");
    const StinkyInstruction* first = vAdd(entry, /*dst=*/300, /*src0=*/16, /*src1=*/600);
    const StinkyInstruction* second = vAdd(entry, /*dst=*/301, /*src0=*/17, /*src1=*/601);
    ASSERT_TRUE(liftForAllocation(*function));
    const VgprMsbStreamModel model(*function, detectLoops(*function));
    std::vector<uint8_t> banks = banksOf(*function, createLegacyColoring(*function));
    EXPECT_EQ(model.cost(banks), 1.0);

    const SSAValueID firstSrc0 = ssaSourceUnits(*first, 0).front()->valueId();
    const SSAValueID secondSrc0 = ssaSourceUnits(*second, 0).front()->valueId();
    banks[firstSrc0] = kUnknownBank;
    banks[secondSrc0] = kUnknownBank;
    EXPECT_EQ(model.cost(banks), 1.0) << "two undecided src0s agree";
    banks[firstSrc0] = 0;
    EXPECT_EQ(model.cost(banks), 2.0) << "an undecided src0 differs from bank 0";
}

TEST(BankCostModelTest, DeltaIsTheChangeInCost) {
    // Values with several uses, across a label and two blocks: every run of up
    // to three of them moved to every bank, the undecided one included, from a
    // starting point that keeps changing.
    std::unique_ptr<Function> function = newFunction();
    BasicBlock* entry = function->createBasicBlock("entry");
    BasicBlock* next = function->createBasicBlock("next");
    function->addEdge(entry, next);
    for (int round = 0; round < 3; ++round) {
        wmma(entry, /*dst=*/0, /*a=*/256 + 8 * round, /*b=*/512, /*c=*/0);
        vAdd(entry, /*dst=*/300 + round, /*src0=*/16, /*src1=*/600);
        vAdd(entry, /*dst=*/17, /*src0=*/300 + round, /*src1=*/16);
    }
    AsmIRBuilder(*entry, kRaTestArch).createLabel("mid");
    wmma(entry, /*dst=*/0, /*a=*/256, /*b=*/768, /*c=*/0);
    vAdd(next, /*dst=*/18, /*src0=*/17, /*src1=*/600);
    wmma(next, /*dst=*/8, /*a=*/264, /*b=*/512, /*c=*/0);
    ASSERT_TRUE(liftForAllocation(*function));
    const VgprMsbStreamModel model(*function, detectLoops(*function));
    std::vector<uint8_t> banks = banksOf(*function, createLegacyColoring(*function));

    const std::vector<SSAValueID> values(model.bankSensitiveValues().begin(),
                                         model.bankSensitiveValues().end());
    ASSERT_TRUE(std::is_sorted(values.begin(), values.end()));
    ASSERT_EQ(std::adjacent_find(values.begin(), values.end()), values.end());
    ASSERT_GE(values.size(), 6u);

    size_t trial = 0;
    for (size_t first = 0; first < values.size(); ++first) {
        for (uint8_t bank = 0; bank <= kUnknownBank; ++bank) {
            for (size_t count = 1; count <= 3 && first + count <= values.size(); ++count) {
                const auto begin = values.begin() + static_cast<std::ptrdiff_t>(first);
                const std::vector<SSAValueID> moved(begin,
                                                    begin + static_cast<std::ptrdiff_t>(count));
                std::vector<uint8_t> after = banks;
                for (const SSAValueID value : moved) after[value] = bank;
                EXPECT_EQ(model.delta(banks, moved, bank), model.cost(after) - model.cost(banks))
                    << "trial " << trial;
                if (++trial % 2 == 0) banks = after;
            }
        }
    }
}
