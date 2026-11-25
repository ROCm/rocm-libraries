// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// InsertVgprMsbPass end to end: the setters it emits, rewrites and leaves, the
// replay check on what it produced, and random functions round-tripped through
// RaiseVgprMsbPass under both policies and both setter widths.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/RaiseVgprMsbPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

/// A setter's low byte from per-slot banks.
int msbWord(int src0, int src1, int src2, int dst) {
    return encodeVgprMsbForSlot(0, src0) + encodeVgprMsbForSlot(1, src1) +
           encodeVgprMsbForSlot(2, src2) + encodeVgprMsbForSlot(3, dst);
}

StinkyInstruction* createInst(BasicBlock* bb, GFX opcode, const std::vector<StinkyRegister>& dests,
                              const std::vector<StinkyRegister>& srcs) {
    AsmIRBuilder builder(*bb, kArch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(opcode, kArch));
    for (const StinkyRegister& dest : dests) inst->addDestReg(dest);
    for (const StinkyRegister& src : srcs) inst->addSrcReg(src);
    return inst;
}

void runPass(Function& func, std::unique_ptr<Pass> pass, VgprMsbMode mode, VgprMsbPolicy policy) {
    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(func.getGemmTileConfig());
    AsmCapsConfig caps;
    caps.vgprMsbMode = mode;
    caps.vgprMsbUnusedSlotsDontCare = policy == VgprMsbPolicy::UnusedSlotsDontCare;
    pm.setAsmCapsConfig(caps);
    pm.addPass(std::move(pass));
    pm.run(func);
}

std::vector<StinkyInstruction*> setters(Function& func) {
    std::vector<StinkyInstruction*> found;
    for (BasicBlock& bb : func) {
        for (IRBase& ir : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst != nullptr && inst->getUnifiedOpcode() == GFX::s_set_vgpr_msb)
                found.push_back(inst);
        }
    }
    return found;
}

class InsertVgprMsbPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("insert_vgpr_msb_test");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
};

}  // namespace

// The setter emitted for v_mov takes v_fma's banks too, and its comment follows.
TEST_F(InsertVgprMsbPassTest, ARewrittenSetterCommentMatchesItsWord) {
    createInst(entry, GFX::v_mov_b32, {vgpr(257)}, {vgpr(258)});
    createInst(entry, GFX::v_fma_f32, {vgpr(259)}, {vgpr(260), vgpr(261), vgpr(262)});
    runPass(*func, createInsertVgprMsbPass(), VgprMsbMode::Msb8,
            VgprMsbPolicy::UnusedSlotsDontCare);

    const std::vector<StinkyInstruction*> found = setters(*func);
    ASSERT_EQ(found.size(), 1U);
    EXPECT_EQ(found[0]->getSrcRegs()[0].getLiteralInt(), msbWord(1, 1, 1, 1));
    const CommentData* comment = found[0]->getModifier<CommentData>();
    ASSERT_NE(comment, nullptr);
    EXPECT_EQ(comment->comment, "src0: 1, src1: 1, src2: 1, dst: 1");
    EXPECT_FALSE(firstVgprMsbMismatch(*func).has_value());
}

// A setter already in the stream sets the word, but is not the pass's to rewrite.
TEST_F(InsertVgprMsbPassTest, ASetterAlreadyInTheStreamIsLeftAsItIs) {
    StinkyInstruction* existing = createInst(entry, GFX::s_set_vgpr_msb, {}, {});
    existing->addSrcReg(StinkyRegister(msbWord(1, 0, 0, 1)));
    createInst(entry, GFX::v_mov_b32, {vgpr(257)}, {vgpr(258)});
    createInst(entry, GFX::v_fma_f32, {vgpr(259)}, {vgpr(260), vgpr(261), vgpr(262)});
    runPass(*func, createInsertVgprMsbPass(), VgprMsbMode::Msb8,
            VgprMsbPolicy::UnusedSlotsDontCare);

    // v_mov runs under the existing word; v_fma, whose src1 and src2 nothing
    // has read, still gets a setter of its own.
    const std::vector<StinkyInstruction*> found = setters(*func);
    ASSERT_EQ(found.size(), 2U);
    EXPECT_EQ(found[0], existing);
    EXPECT_EQ(found[0]->getSrcRegs()[0].getLiteralInt(), msbWord(1, 0, 0, 1));
    EXPECT_EQ(found[1]->getSrcRegs()[0].getLiteralInt(), msbWord(1, 1, 1, 1));
    EXPECT_FALSE(firstVgprMsbMismatch(*func).has_value());
}

// At the default too, the word a setter in the stream sets is in force: the
// v_mov it serves gets no second setter.
TEST_F(InsertVgprMsbPassTest, AtTheDefaultASetterInTheStreamServesWhatItsWordServes) {
    StinkyInstruction* existing = createInst(entry, GFX::s_set_vgpr_msb, {}, {});
    existing->addSrcReg(StinkyRegister(msbWord(1, 0, 0, 1)));
    createInst(entry, GFX::v_mov_b32, {vgpr(257)}, {vgpr(258)});
    runPass(*func, createInsertVgprMsbPass(), VgprMsbMode::Msb8, VgprMsbPolicy::ExactWord);

    const std::vector<StinkyInstruction*> found = setters(*func);
    ASSERT_EQ(found.size(), 1U);
    EXPECT_EQ(found[0], existing);
    EXPECT_FALSE(firstVgprMsbMismatch(*func).has_value());
}

// After v_mov the setter v_fma needs would wait in front of the next
// instruction, which is the setter in the stream, and that setter would then
// override it. It goes after it instead.
TEST_F(InsertVgprMsbPassTest, AtTheDefaultNoSetterGoesInFrontOfOneInTheStream) {
    createInst(entry, GFX::v_mov_b32, {vgpr(257)}, {vgpr(258)});
    StinkyInstruction* existing = createInst(entry, GFX::s_set_vgpr_msb, {}, {});
    existing->addSrcReg(StinkyRegister(msbWord(0, 0, 0, 0)));
    createInst(entry, GFX::v_fma_f32, {vgpr(259)}, {vgpr(260), vgpr(261), vgpr(262)});
    runPass(*func, createInsertVgprMsbPass(), VgprMsbMode::Msb8, VgprMsbPolicy::ExactWord);

    const std::vector<StinkyInstruction*> found = setters(*func);
    ASSERT_EQ(found.size(), 3U);
    EXPECT_EQ(found[0]->getSrcRegs()[0].getLiteralInt(), msbWord(1, 0, 0, 1));
    EXPECT_EQ(found[1], existing);
    EXPECT_EQ(found[2]->getSrcRegs()[0].getLiteralInt(), msbWord(1, 1, 1, 1));
    EXPECT_FALSE(firstVgprMsbMismatch(*func).has_value());
}

TEST_F(InsertVgprMsbPassTest, TheReplayCheckNamesAnInstructionInTheWrongBank) {
    createInst(entry, GFX::v_mov_b32, {vgpr(257)}, {vgpr(258)});
    const std::optional<std::string> unset = firstVgprMsbMismatch(*func);
    ASSERT_TRUE(unset.has_value()) << "no setter at all";
    EXPECT_NE(unset->find("v_mov_b32"), std::string::npos) << *unset;

    runPass(*func, createInsertVgprMsbPass(), VgprMsbMode::Msb8, VgprMsbPolicy::ExactWord);
    EXPECT_FALSE(firstVgprMsbMismatch(*func).has_value());

    // src0 back in bank 0 under the setter: v258 would read v2.
    setters(*func)[0]->setSrcReg(0, StinkyRegister(msbWord(0, 0, 0, 1)));
    const std::optional<std::string> wrong = firstVgprMsbMismatch(*func);
    ASSERT_TRUE(wrong.has_value());
    EXPECT_NE(wrong->find("v_mov_b32"), std::string::npos) << *wrong;

    // The slots v_mov does not use are not its business.
    setters(*func)[0]->setSrcReg(0, StinkyRegister(msbWord(1, 3, 2, 1)));
    EXPECT_FALSE(firstVgprMsbMismatch(*func).has_value());
}

namespace {

/// Random blocks of VALU, compare, LDS and SALU instructions with random banks,
/// some behind labels or calls, the same for the same \p seed.
std::unique_ptr<Function> randomFunction(uint32_t seed) {
    std::mt19937 rng(seed);
    std::discrete_distribution<int> bank({4, 3, 2, 1});
    std::uniform_int_distribution<int> low(0, 251);
    std::uniform_int_distribution<int> blockCount(1, 4);
    std::uniform_int_distribution<int> blockSize(1, 24);
    std::uniform_int_distribution<int> kind(0, 99);
    std::bernoulli_distribution coin(0.5);
    std::bernoulli_distribution useSgpr(0.2);

    auto func = std::make_unique<Function>("msb_round_trip_" + std::to_string(seed));
    setFunctionArch(*func, kArch);
    auto pick = [&](int width) { return vgpr(bank(rng) * 256 + low(rng), width); };
    auto pickSrc = [&]() { return useSgpr(rng) ? sgpr(low(rng)) : pick(1); };

    const int blocks = blockCount(rng);
    for (int b = 0; b < blocks; ++b) {
        BasicBlock* bb = func->createBasicBlock("bb" + std::to_string(b));
        if (b > 0 && coin(rng)) {
            AsmIRBuilder builder(*bb, kArch);
            builder.createLabel("label_" + std::to_string(seed) + "_" + std::to_string(b));
        }
        const int size = blockSize(rng);
        for (int i = 0; i < size; ++i) {
            const int k = kind(rng);
            if (k < 25)
                createInst(bb, GFX::v_add_f32, {pick(1)}, {pickSrc(), pick(1)});
            else if (k < 45)
                createInst(bb, GFX::v_fma_f32, {pick(1)}, {pickSrc(), pick(1), pick(1)});
            else if (k < 55)
                createInst(bb, GFX::v_cmp_eq_u32, {sgpr(28)}, {pickSrc(), pick(1)});
            else if (k < 65)
                createInst(bb, GFX::v_mov_b32, {pick(1)}, {pickSrc()});
            else if (k < 77)
                createInst(bb, GFX::ds_load_b128, {pick(4)}, {pick(1)});
            else if (k < 87)
                createInst(bb, GFX::ds_store_b64, {}, {pick(1), pick(2)});
            else if (k < 97)
                createInst(bb, GFX::s_mov_b32, {sgpr(12)}, {sgpr(13)});
            else
                createInst(bb, GFX::s_swappc_b64, {sgpr(26, 2)}, {sgpr(30, 2)});
        }
    }
    return func;
}

constexpr int kRounds = 300;

using RoundTripParam = std::tuple<VgprMsbMode, VgprMsbPolicy>;

/// After InsertVgprMsbPass, each VGPR operand is put in the byte form the
/// assembler sees (v[idx + offset]) and RaiseVgprMsbPass reads its bank back
/// from the setters: every index must come back unchanged.
class InsertVgprMsbRoundTripTest : public ::testing::TestWithParam<RoundTripParam> {};

}  // namespace

TEST_P(InsertVgprMsbRoundTripTest, RandomFunctionsRoundTripThroughRaise) {
    const auto [mode, policy] = GetParam();
    for (int round = 0; round < kRounds; ++round) {
        std::unique_ptr<Function> func = randomFunction(static_cast<uint32_t>(round));

        std::map<StinkyInstruction*, std::vector<uint32_t>> before;
        auto vgprIndices = [](const StinkyInstruction* inst) {
            std::vector<uint32_t> indices;
            for (const StinkyRegister& reg : inst->getDestRegs())
                if (getMsbFromVgpr(reg) >= 0) indices.push_back(reg.reg.idx);
            for (const StinkyRegister& reg : inst->getSrcRegs())
                if (getMsbFromVgpr(reg) >= 0) indices.push_back(reg.reg.idx);
            return indices;
        };
        for (BasicBlock& bb : *func)
            for (IRBase& ir : bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) before[inst] = vgprIndices(inst);

        runPass(*func, createInsertVgprMsbPass(), mode, policy);
        const std::optional<std::string> mismatch = firstVgprMsbMismatch(*func);
        ASSERT_FALSE(mismatch.has_value()) << "round " << round << ": " << *mismatch;

        // What the assembler reads: the byte index, with the bank left to the setter.
        for (BasicBlock& bb : *func) {
            for (IRBase& ir : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (inst == nullptr) continue;
                auto toByteForm = [](std::vector<StinkyRegister> regs) {
                    for (StinkyRegister& reg : regs) {
                        if (getMsbFromVgpr(reg) < 0) continue;
                        reg.reg.idx =
                            static_cast<uint32_t>(static_cast<int>(reg.reg.idx) + reg.reg.offset);
                        reg.reg.offset = 0;
                    }
                    return regs;
                };
                inst->setDestRegs(toByteForm(inst->getDestRegs()));
                inst->setSrcRegs(toByteForm(inst->getSrcRegs()));
            }
        }

        runPass(*func, createRaiseVgprMsbPass(), mode, policy);

        for (const auto& [inst, indices] : before)
            ASSERT_EQ(vgprIndices(inst), indices) << "round " << round;
        EXPECT_TRUE(setters(*func).empty()) << "round " << round;
    }
}

INSTANTIATE_TEST_SUITE_P(
    ModesAndPolicies, InsertVgprMsbRoundTripTest,
    ::testing::Combine(::testing::Values(VgprMsbMode::Msb8, VgprMsbMode::Msb16),
                       ::testing::Values(VgprMsbPolicy::ExactWord,
                                         VgprMsbPolicy::UnusedSlotsDontCare)),
    [](const ::testing::TestParamInfo<RoundTripParam>& info) {
        const std::string mode = std::get<0>(info.param) == VgprMsbMode::Msb8 ? "Msb8" : "Msb16";
        const std::string policy = std::get<1>(info.param) == VgprMsbPolicy::ExactWord
                                       ? "ExactWord"
                                       : "UnusedSlotsDontCare";
        return mode + "_" + policy;
    });

// For the same instructions in the same order, never more setters than the
// default, and fewer over the lot.
TEST(InsertVgprMsbPolicyTest, UnusedSlotsDontCareNeverEmitsMoreSettersThanExactWord) {
    size_t exactTotal = 0;
    size_t dontCareTotal = 0;
    for (int round = 0; round < kRounds; ++round) {
        std::unique_ptr<Function> exact = randomFunction(static_cast<uint32_t>(round));
        std::unique_ptr<Function> dontCare = randomFunction(static_cast<uint32_t>(round));
        runPass(*exact, createInsertVgprMsbPass(), VgprMsbMode::Msb16, VgprMsbPolicy::ExactWord);
        runPass(*dontCare, createInsertVgprMsbPass(), VgprMsbMode::Msb16,
                VgprMsbPolicy::UnusedSlotsDontCare);
        const size_t exactSetters = setters(*exact).size();
        const size_t dontCareSetters = setters(*dontCare).size();
        EXPECT_LE(dontCareSetters, exactSetters) << "round " << round;
        exactTotal += exactSetters;
        dontCareTotal += dontCareSetters;
    }
    EXPECT_LT(dontCareTotal, exactTotal);
}
