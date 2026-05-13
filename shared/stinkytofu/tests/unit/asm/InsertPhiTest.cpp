/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include <gtest/gtest.h>

#include "PhiTestFixtures.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/transforms/asm/PhiPlacement.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class InsertPhiTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
};

// =============================================================================
// CFG: Pass-through block with multiple predecessors
//
// PHI for v0 should appear in C (the join point) with operands from A, F, B.
// D and E have a single predecessor (C) so they need no PHIs.
//
//       Entry
//      /  |  \
//     A   F   B
//     |   |   |
//   v0=x  |  v0=y  (F has no v0)
//       \ | /
//         C (pass-through, no v0 use/def)
//        / \
//       D   E  (both use v0)
// =============================================================================

TEST_F(InsertPhiTest, PassThroughJoinWithMultiplePredecessors) {
    Function func("pass_through_join");
    setFunctionArch(func, arch);
    BasicBlock* entry = func.createBasicBlock("entry");
    BasicBlock* A = func.createBasicBlock("A");
    BasicBlock* F = func.createBasicBlock("F");
    BasicBlock* B = func.createBasicBlock("B");
    BasicBlock* C = func.createBasicBlock("C");
    BasicBlock* D = func.createBasicBlock("D");
    BasicBlock* E = func.createBasicBlock("E");

    func.addEdge(entry, A);
    func.addEdge(entry, F);
    func.addEdge(entry, B);
    func.addEdge(A, C);
    func.addEdge(F, C);
    func.addEdge(B, C);
    func.addEdge(C, D);
    func.addEdge(C, E);

    // A and B each define v0
    StinkyInstruction* aAdd = createVAddInBlock(A, arch, 0, 1, 2);
    StinkyInstruction* bAdd = createVAddInBlock(B, arch, 0, 3, 4);
    // F: no v0 definition (pass-through from Entry, which has no v0 either)
    createVAddInBlock(F, arch, 5, 6, 7);  // v5 = v6 + v7
    // C: pass-through (no v0 use or def)
    createVAddInBlock(C, arch, 10, 11, 12);  // v10 = v11 + v12
    // D and E both use v0
    createVAddInBlock(D, arch, 20, 0, 21);
    createVAddInBlock(E, arch, 30, 0, 31);

    // --- Run the pass ---
    insertPhiInstructions(func, false);

    // --- Verify PHI placement ---

    // C is the only join point where v0 has multiple reaching definitions (A, B).
    // D and E have a single predecessor (C), so no PHIs there.
    EXPECT_EQ(countPhis(*C), 1u) << "C should have exactly one PHI (for v0)";
    EXPECT_EQ(countPhis(*D), 0u) << "D (single predecessor) needs no PHI";
    EXPECT_EQ(countPhis(*E), 0u) << "E (single predecessor) needs no PHI";
    EXPECT_EQ(countPhisInFunction(func), 1u) << "Only one PHI in the entire function";

    // --- Verify the PHI's operands ---

    StinkyInstruction* phi = &getStinkyInst(C->begin());
    ASSERT_EQ(phi->getUnifiedOpcode(), GFX::PHI);

    // PHI defines v0
    ASSERT_EQ(phi->getNumDestRegs(), 1u);
    const StinkyRegister& phiDest = phi->getDestReg(0);
    EXPECT_EQ(phiDest.reg.type, RegType::V);
    EXPECT_EQ(phiDest.reg.idx, 0u);

    // One src operand per predecessor (A, F, B)
    ASSERT_EQ(phi->getNumSrcRegs(), 3u);

    const std::vector<BasicBlock*>& cPreds = C->getPredecessors();
    ASSERT_EQ(cPreds.size(), 3u);

    for (size_t i = 0; i < cPreds.size(); ++i) {
        if (cPreds[i] == A) {
            EXPECT_EQ(phi->getSrcReg(i), aAdd->getDestReg(0))
                << "PHI src for A should match A's v_add dest (v0)";
        } else if (cPreds[i] == F) {
            EXPECT_FALSE(phi->getSrcReg(i).isRegister())
                << "PHI src for F should be non-register (v0 never defined on this path)";
        } else if (cPreds[i] == B) {
            EXPECT_EQ(phi->getSrcReg(i), bAdd->getDestReg(0))
                << "PHI src for B should match B's v_add dest (v0)";
        } else {
            FAIL() << "Unexpected predecessor of C";
        }
    }
}

// =============================================================================
// 1. Iterated Dominance Frontier
//
// PHI at G comes from DF(C). Placing PHI at G makes G a def site, and
// DF(G) = {H}, so a second PHI is placed at H by the iterated DF worklist.
// =============================================================================

TEST_F(InsertPhiTest, IteratedDominanceFrontier) {
    Function func("iterated_df");
    IteratedDFCfg cfg = buildIteratedDFCfg(func, arch);

    insertPhiInstructions(func, false);

    EXPECT_EQ(countPhis(*cfg.G), 1u) << "G needs PHI (DF of C)";
    EXPECT_EQ(countPhis(*cfg.H), 1u) << "H needs PHI (iterated DF of G)";
    EXPECT_EQ(countPhis(*cfg.entry), 0u);
    EXPECT_EQ(countPhis(*cfg.A), 0u);
    EXPECT_EQ(countPhis(*cfg.B), 0u);
    EXPECT_EQ(countPhis(*cfg.C), 0u);
    EXPECT_EQ(countPhis(*cfg.D), 0u);
    EXPECT_EQ(countPhis(*cfg.E), 0u);
    EXPECT_EQ(countPhis(*cfg.F), 0u);
    EXPECT_EQ(countPhisInFunction(func), 2u);

    // PHI_G:  preds [E, F] → src regs: entryDef dest (via B→E), cDef dest (via C→F)
    StinkyInstruction* phiG = &getStinkyInst(cfg.G->begin());
    ASSERT_EQ(phiG->getUnifiedOpcode(), GFX::PHI);
    ASSERT_EQ(phiG->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiG->getSrcReg(0), cfg.entryDef->getDestReg(0));
    EXPECT_EQ(phiG->getSrcReg(1), cfg.cDef->getDestReg(0));

    // PHI_H:  preds [D, G] → src regs: entryDef dest (via B→D), PHI_G dest
    StinkyInstruction* phiH = &getStinkyInst(cfg.H->begin());
    ASSERT_EQ(phiH->getUnifiedOpcode(), GFX::PHI);
    ASSERT_EQ(phiH->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiH->getSrcReg(0), cfg.entryDef->getDestReg(0));
    EXPECT_EQ(phiH->getSrcReg(1), phiG->getDestReg(0));
}

// =============================================================================
// 2. Nested Loops — PHIs at both loop headers
// =============================================================================

TEST_F(InsertPhiTest, NestedLoops_DualPhiHeaders) {
    Function func("nested_loops");
    NestedLoopCfg cfg = buildNestedLoopCfg(func, arch);

    insertPhiInstructions(func, false);

    EXPECT_EQ(countPhis(*cfg.A), 1u) << "outer header A needs PHI";
    EXPECT_EQ(countPhis(*cfg.B), 1u) << "inner header B needs PHI";
    EXPECT_EQ(countPhis(*cfg.C), 0u);
    EXPECT_EQ(countPhis(*cfg.D), 0u);
    EXPECT_EQ(countPhisInFunction(func), 2u);

    // PHI_A:  preds [Entry, D] → entryDef dest, cDef dest (reaches A via C→D→A)
    StinkyInstruction* phiA = &getStinkyInst(cfg.A->begin());
    ASSERT_EQ(phiA->getUnifiedOpcode(), GFX::PHI);
    ASSERT_EQ(phiA->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiA->getSrcReg(0), cfg.entryDef->getDestReg(0));
    EXPECT_EQ(phiA->getSrcReg(1), cfg.cDef->getDestReg(0));

    // PHI_B:  preds [A, C] → PHI_A dest, cDef dest
    StinkyInstruction* phiB = &getStinkyInst(cfg.B->begin());
    ASSERT_EQ(phiB->getUnifiedOpcode(), GFX::PHI);
    ASSERT_EQ(phiB->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiB->getSrcReg(0), phiA->getDestReg(0));
    EXPECT_EQ(phiB->getSrcReg(1), cfg.cDef->getDestReg(0));
}

// =============================================================================
// 3. Self-loop at a join point — self-referential PHI operand
// =============================================================================

TEST_F(InsertPhiTest, SelfLoopAtJoinPoint) {
    Function func("self_loop_join");
    SelfLoopJoinCfg cfg = buildSelfLoopJoinCfg(func, arch);

    insertPhiInstructions(func, false);

    EXPECT_EQ(countPhis(*cfg.C), 1u) << "C (join + self-loop) needs PHI";
    EXPECT_EQ(countPhis(*cfg.D), 0u) << "D has single pred";
    EXPECT_EQ(countPhisInFunction(func), 1u);

    // PHI_C:  preds [A, B, C] → aDef dest, bDef dest, PHI_C dest (self-referential)
    StinkyInstruction* phiC = &getStinkyInst(cfg.C->begin());
    ASSERT_EQ(phiC->getUnifiedOpcode(), GFX::PHI);
    ASSERT_EQ(phiC->getNumSrcRegs(), 3u);
    EXPECT_EQ(phiC->getSrcReg(0), cfg.aDef->getDestReg(0));
    EXPECT_EQ(phiC->getSrcReg(1), cfg.bDef->getDestReg(0));
    EXPECT_EQ(phiC->getSrcReg(2), phiC->getDestReg(0)) << "self-loop edge → self-referential";
}

// =============================================================================
// 4. Irreducible CFG — mutually recursive PHIs
// =============================================================================

TEST_F(InsertPhiTest, IrreducibleCFG_MutuallyRecursivePhis) {
    Function func("irreducible");
    IrreducibleCfg cfg = buildIrreducibleCfg(func, arch);

    insertPhiInstructions(func, false);

    EXPECT_EQ(countPhis(*cfg.C), 1u) << "C needs PHI (preds A, D)";
    EXPECT_EQ(countPhis(*cfg.D), 1u) << "D needs PHI (preds B, C)";
    EXPECT_EQ(countPhis(*cfg.E), 1u) << "E needs PHI (preds C, D)";
    EXPECT_EQ(countPhisInFunction(func), 3u);

    StinkyInstruction* phiC = &getStinkyInst(cfg.C->begin());
    StinkyInstruction* phiD = &getStinkyInst(cfg.D->begin());
    StinkyInstruction* phiE = &getStinkyInst(cfg.E->begin());

    // PHI_C:  preds [A, D] → aDef dest, PHI_D dest
    ASSERT_EQ(phiC->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiC->getSrcReg(0), cfg.aDef->getDestReg(0));
    EXPECT_EQ(phiC->getSrcReg(1), phiD->getDestReg(0)) << "mutually recursive with D";

    // PHI_D:  preds [B, C] → bDef dest, PHI_C dest
    ASSERT_EQ(phiD->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiD->getSrcReg(0), cfg.bDef->getDestReg(0));
    EXPECT_EQ(phiD->getSrcReg(1), phiC->getDestReg(0)) << "mutually recursive with C";

    // PHI_E:  preds [C, D] → PHI_C dest, PHI_D dest
    ASSERT_EQ(phiE->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiE->getSrcReg(0), phiC->getDestReg(0));
    EXPECT_EQ(phiE->getSrcReg(1), phiD->getDestReg(0));
}

// =============================================================================
// 5. Multiple registers at the same join point
// =============================================================================

TEST_F(InsertPhiTest, MultipleRegistersAtSameJoin) {
    Function func("multi_reg_join");
    MultiRegJoinCfg cfg = buildMultiRegJoinCfg(func, arch);

    insertPhiInstructions(func, false);

    EXPECT_EQ(countPhis(*cfg.C), 2u) << "C needs PHIs for v0 and v1";
    EXPECT_EQ(countPhisInFunction(func), 2u);

    StinkyInstruction* phiV0 = findPhi(*cfg.C, RegType::V, 0);
    StinkyInstruction* phiV1 = findPhi(*cfg.C, RegType::V, 1);
    ASSERT_NE(phiV0, nullptr) << "PHI for v0 must exist in C";
    ASSERT_NE(phiV1, nullptr) << "PHI for v1 must exist in C";

    // PHI(v0):  preds [A, B] → aDefV0 dest, bDefV0 dest
    ASSERT_EQ(phiV0->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiV0->getSrcReg(0), cfg.aDefV0->getDestReg(0));
    EXPECT_EQ(phiV0->getSrcReg(1), cfg.bDefV0->getDestReg(0));

    // PHI(v1):  preds [A, B] → aDefV1 dest, bDefV1 dest
    ASSERT_EQ(phiV1->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiV1->getSrcReg(0), cfg.aDefV1->getDestReg(0));
    EXPECT_EQ(phiV1->getSrcReg(1), cfg.bDefV1->getDestReg(0));
}

// =============================================================================
// 6. Chain of diamonds — 3 sequential merge points
// =============================================================================

TEST_F(InsertPhiTest, ChainOfDiamonds) {
    Function func("chain_diamonds");
    ChainOfDiamondsCfg cfg = buildChainOfDiamondsCfg(func, arch);

    insertPhiInstructions(func, false);

    EXPECT_EQ(countPhis(*cfg.C), 1u);
    EXPECT_EQ(countPhis(*cfg.F), 1u);
    EXPECT_EQ(countPhis(*cfg.I), 1u);
    EXPECT_EQ(countPhisInFunction(func), 3u);

    StinkyInstruction* phiC = &getStinkyInst(cfg.C->begin());
    StinkyInstruction* phiF = &getStinkyInst(cfg.F->begin());
    StinkyInstruction* phiI = &getStinkyInst(cfg.I->begin());

    // PHI_C:  preds [A, B] → entryDef dest (inherited via A), bDef dest
    ASSERT_EQ(phiC->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiC->getSrcReg(0), cfg.entryDef->getDestReg(0));
    EXPECT_EQ(phiC->getSrcReg(1), cfg.bDef->getDestReg(0));

    // PHI_F:  preds [D, E] → PHI_C dest (inherited via D), eDef dest
    ASSERT_EQ(phiF->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiF->getSrcReg(0), phiC->getDestReg(0));
    EXPECT_EQ(phiF->getSrcReg(1), cfg.eDef->getDestReg(0));

    // PHI_I:  preds [G, H] → PHI_F dest (inherited via G), hDef dest
    ASSERT_EQ(phiI->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiI->getSrcReg(0), phiF->getDestReg(0));
    EXPECT_EQ(phiI->getSrcReg(1), cfg.hDef->getDestReg(0));
}

// =============================================================================
// 7. Dead register — semi-pruned SSA should skip v0 (never read)
// =============================================================================

TEST_F(InsertPhiTest, DeadRegister_SemiPrunedSSA) {
    Function func("dead_reg");
    (void)buildDeadRegCfg(func, arch);

    insertPhiInstructions(func, false);

    EXPECT_EQ(countPhisInFunction(func), 0u)
        << "v0 is never used as a source — no PHI should be placed";
}

// =============================================================================
// 8. Re-definition within the same block — lastDef shadows first
// =============================================================================

TEST_F(InsertPhiTest, RedefinitionInSameBlock) {
    Function func("redef_same_block");
    RedefSameBlockCfg cfg = buildRedefSameBlockCfg(func, arch);

    insertPhiInstructions(func, false);

    EXPECT_EQ(countPhis(*cfg.C), 1u);
    EXPECT_EQ(countPhisInFunction(func), 1u);

    StinkyInstruction* phiC = &getStinkyInst(cfg.C->begin());
    ASSERT_EQ(phiC->getUnifiedOpcode(), GFX::PHI);
    ASSERT_EQ(phiC->getNumSrcRegs(), 2u);

    // Src from A must match aDef2's dest (the SECOND def), not aDef1
    EXPECT_EQ(phiC->getSrcReg(0), cfg.aDef2->getDestReg(0))
        << "PHI should use the last def in A, not the first";
    EXPECT_EQ(phiC->getSrcReg(1), cfg.bDef->getDestReg(0));
}

// =============================================================================
// 9. Wide register with partial sub-register redefine
//
// Entry defines v[0:3] (4 DWORDs, like buffer_load).  C redefines only v0.
// PHIs should appear only for v0 (at G and H); v1, v2, v3 are unchanged.
// =============================================================================

TEST_F(InsertPhiTest, WideRegPartialSubregRedefine) {
    Function func("wide_partial_redef");
    WideRegPartialRedefCfg cfg = buildWideRegPartialRedefCfg(func, arch);

    insertPhiInstructions(func, false);

    // Only v0 has conflicting defs (Entry wide load vs. C's partial redef).
    EXPECT_EQ(countPhis(*cfg.G), 1u) << "G needs PHI for v0 (DF of C)";
    EXPECT_EQ(countPhis(*cfg.H), 1u) << "H needs PHI for v0 (iterated DF of G)";
    EXPECT_EQ(countPhis(*cfg.entry), 0u);
    EXPECT_EQ(countPhis(*cfg.A), 0u);
    EXPECT_EQ(countPhis(*cfg.B), 0u);
    EXPECT_EQ(countPhis(*cfg.C), 0u);
    EXPECT_EQ(countPhis(*cfg.D), 0u);
    EXPECT_EQ(countPhis(*cfg.E), 0u);
    EXPECT_EQ(countPhis(*cfg.F), 0u);
    EXPECT_EQ(countPhisInFunction(func), 2u);

    StinkyInstruction* phiG = findPhi(*cfg.G, RegType::V, 0);
    StinkyInstruction* phiH = findPhi(*cfg.H, RegType::V, 0);
    ASSERT_NE(phiG, nullptr);
    ASSERT_NE(phiH, nullptr);

    // PHI_G: preds [E, F] → entryWideDef dest (via B→E), cPartialDef dest (via C→F)
    ASSERT_EQ(phiG->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiG->getSrcReg(0), cfg.entryWideDef->getDestReg(0));
    EXPECT_EQ(phiG->getSrcReg(1), cfg.cPartialDef->getDestReg(0));

    // PHI_H: preds [D, G] → entryWideDef dest (via B→D), PHI_G dest
    ASSERT_EQ(phiH->getNumSrcRegs(), 2u);
    EXPECT_EQ(phiH->getSrcReg(0), cfg.entryWideDef->getDestReg(0));
    EXPECT_EQ(phiH->getSrcReg(1), phiG->getDestReg(0));

    // v1, v2, v3 have a single reaching def (Entry) — no PHIs anywhere
    EXPECT_EQ(findPhi(*cfg.G, RegType::V, 1), nullptr);
    EXPECT_EQ(findPhi(*cfg.G, RegType::V, 2), nullptr);
    EXPECT_EQ(findPhi(*cfg.G, RegType::V, 3), nullptr);
    EXPECT_EQ(findPhi(*cfg.H, RegType::V, 1), nullptr);
    EXPECT_EQ(findPhi(*cfg.H, RegType::V, 2), nullptr);
    EXPECT_EQ(findPhi(*cfg.H, RegType::V, 3), nullptr);
}
