// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// A VALU write under a narrow exec mask covers only the active lanes, so its
// destination keeps its previous contents everywhere else. These tests hold the
// two halves of modelling that: TieExecMaskedWritesPass giving the write a read
// of what it preserves, and AllocationConstraints turning that read into an
// affinity the colourer has to honour.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/AsmTargetRegisters.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/asm/TieExecMaskedWritesPass.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationConstraints.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "transforms/asm/ra/allocators/GreedyAllocator.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

/// True when some affinity set holds both values, which is what forces the
/// colourer to give them one register.
bool tiedTogether(const AllocationConstraints& constraints, SSAValueID first, SSAValueID second) {
    for (const AffinitySet& set : constraints.affinitySets()) {
        const auto has = [&set](SSAValueID id) {
            return std::find(set.members.begin(), set.members.end(), id) != set.members.end();
        };
        if (has(first) && has(second)) return true;
    }
    return false;
}

class ExecMaskedWriteTieTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
        // The pass reads the wavefront size from the context to know which EXEC
        // register to look for.
        passCtx.setGemmTileConfig(func->getGemmTileConfig());
        bb = func->createBasicBlock("entry");
    }

    void runTiePass() {
        auto pass = createTieExecMaskedWritesPass();
        pass->run(*func, passCtx, am);
    }

    /// `s_mov_b32 exec_lo, s<src>` -- narrows the mask, opening a span.
    StinkyInstruction* execNarrow(int srcSgpr = 10) {
        AsmIRBuilder builder(*bb, kRaTestArch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
        inst->addDestReg(StinkyRegister::getEXECRegister(32));
        inst->addSrcReg(StinkyRegister("s", srcSgpr, 1));
        return inst;
    }

    /// `s_mov_b32 exec_lo, -1` -- the full-mask reset that closes a span.
    StinkyInstruction* execReset() {
        AsmIRBuilder builder(*bb, kRaTestArch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
        inst->addDestReg(StinkyRegister::getEXECRegister(32));
        inst->addSrcReg(StinkyRegister(-1));
        return inst;
    }

    /// `v_mov_b32 v<dest>, 0` -- a write with no source naming its destination,
    /// so under a mask it is the case the same-register lookup cannot reach
    /// until the pass supplies the operand.
    StinkyInstruction* vMovZero(int destVgpr) {
        AsmIRBuilder builder(*bb, kRaTestArch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
        inst->addDestReg(StinkyRegister("v", destVgpr, 1));
        inst->addSrcReg(StinkyRegister(0));
        return inst;
    }

    /// `s_mov_b32 s<dest>, 0` -- a scalar write, which no lane mask gates.
    StinkyInstruction* sMovZero(int destSgpr) {
        AsmIRBuilder builder(*bb, kRaTestArch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
        inst->addDestReg(StinkyRegister("s", destSgpr, 1));
        inst->addSrcReg(StinkyRegister(0));
        return inst;
    }

    size_t srcCount(const StinkyInstruction& inst) const {
        return inst.getSrcRegs().size();
    }

    bool readsRegister(const StinkyInstruction& inst, uint32_t vgpr) const {
        for (const StinkyRegister& src : inst.getSrcRegs()) {
            if (!src.isRegister()) continue;
            if (src.reg.type == RegType::V && src.reg.idx == vgpr) return true;
        }
        return false;
    }

    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    PassContext passCtx;
    AnalysisManager am;
};

}  // namespace

// ---------------------------------------------------------------------------
// TieExecMaskedWritesPass
// ---------------------------------------------------------------------------

TEST_F(ExecMaskedWriteTieTest, AMaskedWriteGainsAReadOfItsDestination) {
    execNarrow();
    StinkyInstruction* mov = vMovZero(1);
    // The producer's own `v_add_nc_u32 v0, v0, 1` in shape: the operand the tie
    // needs is already there.
    StinkyInstruction* add = createVAddInBlock(bb, kRaTestArch, /*dest=*/0, /*src0=*/0, /*src1=*/1);
    execReset();

    runTiePass();

    // The literal it already had, plus v1: the value its inactive lanes keep.
    EXPECT_EQ(srcCount(*mov), 2u);
    EXPECT_TRUE(readsRegister(*mov, 1));
    EXPECT_EQ(srcCount(*add), 2u);

    // The same "unless it is already there" guard makes the pass idempotent.
    runTiePass();
    EXPECT_EQ(srcCount(*mov), 2u);
    EXPECT_EQ(srcCount(*add), 2u);
}

TEST_F(ExecMaskedWriteTieTest, ThePassLeavesUnmaskedAndScalarWritesAlone) {
    // Both halves of the trigger. Nothing masks the first, so it really does
    // define its destination completely. The second is masked but scalar, and a
    // scalar register holds one value per wave, so no lane mask applies to it.
    StinkyInstruction* unmasked = vMovZero(1);
    execNarrow();
    StinkyInstruction* scalar = sMovZero(20);
    execReset();

    runTiePass();

    EXPECT_EQ(srcCount(*unmasked), 1u);
    EXPECT_FALSE(readsRegister(*unmasked, 1));
    EXPECT_EQ(srcCount(*scalar), 1u);
}

// ---------------------------------------------------------------------------
// The affinity that comes out of it
// ---------------------------------------------------------------------------

TEST_F(ExecMaskedWriteTieTest, OnlyTheMaskedWriteIsTied) {
    // v1 is defined, partly overwritten under the mask, then read: the read sees
    // a merge of the two, so the write has to land on the register holding the
    // definition it preserves.
    //
    // The add after the span reads its own destination too, and must not be
    // tied. Outside a span it defines v4 completely, and tying it would cost a
    // register for nothing -- which is what goes wrong if the trigger is read as
    // "some source names the destination" rather than "the mask covers this".
    StinkyInstruction* def = createVAddInBlock(bb, kRaTestArch, /*dest=*/1, /*src0=*/2,
                                               /*src1=*/3);
    execNarrow();
    StinkyInstruction* mov = vMovZero(1);
    execReset();
    StinkyInstruction* unmasked = createVAddInBlock(bb, kRaTestArch, /*dest=*/4, /*src0=*/4,
                                                    /*src1=*/1);

    runTiePass();
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* preserved = ssaDefinedValue(*def, 0);
    const StinkySSAValue* written = ssaDefinedValue(*mov, 0);
    const StinkySSAValue* unconstrained = ssaDefinedValue(*unmasked, 0);
    const StinkySSAValue* itsOwnSource = ssaSourceValue(*unmasked, 0);
    ASSERT_NE(preserved, nullptr);
    ASSERT_NE(written, nullptr);
    ASSERT_NE(unconstrained, nullptr);
    ASSERT_NE(itsOwnSource, nullptr);
    // The appended operand binds the reaching definition, which is what makes
    // the tie point at the right value rather than merely at some value.
    EXPECT_EQ(ssaSourceValue(*mov, 1), preserved);

    const AllocationRules rules;
    const AsmTargetRegisters target = AsmTargetRegisters::forFunction(*func);
    const AllocationConstraints constraints = AllocationConstraints::build(*func, target, rules);

    EXPECT_TRUE(tiedTogether(constraints, written->valueId(), preserved->valueId()));
    EXPECT_FALSE(tiedTogether(constraints, unconstrained->valueId(), itsOwnSource->valueId()));
}

TEST_F(ExecMaskedWriteTieTest, TheVerifierRejectsAColouringThatSeparatesThem) {
    // The enforcement point, and the one place the tie can be shown to change an
    // outcome. A colouring that puts the two apart is refused here rather than
    // assembled and run.
    //
    // Worth stating why this is the test and not "the allocator keeps them
    // together": greedy-compact reuses a register whose value dies exactly where
    // the next one is born, so it lands on the tie unprompted in a small
    // function. That is the coincidence the real kernel relied on until register
    // pressure moved and it stopped holding.
    StinkyInstruction* def = createVAddInBlock(bb, kRaTestArch, /*dest=*/1, /*src0=*/2,
                                               /*src1=*/3);
    execNarrow();
    StinkyInstruction* mov = vMovZero(1);
    execReset();
    createVAddInBlock(bb, kRaTestArch, /*dest=*/4, /*src0=*/1, /*src1=*/1);

    runTiePass();
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* preserved = ssaDefinedValue(*def, 0);
    const StinkySSAValue* written = ssaDefinedValue(*mov, 0);
    ASSERT_NE(preserved, nullptr);
    ASSERT_NE(written, nullptr);

    AllocationSetup setup(*func, RegClassSet::only(RegType::V));
    AllocationResult separated = createLegacyColoring(*func);
    const RegKey preservedKey = separated.assignmentOf(preserved->valueId());
    separated.assign(written->valueId(), RegKey{RegType::V, preservedKey.idx + 8, RegHalf::NONE});

    const AllocationVerificationResult checked =
        verifyAllocation(*func, separated, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_NE(checked.toString().find("affinity"), std::string::npos) << checked.toString();
}
