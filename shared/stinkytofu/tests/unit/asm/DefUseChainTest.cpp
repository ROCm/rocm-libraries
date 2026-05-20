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
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

// Helper to create an instruction with BARRIER (pseudo register) as dest.
// Simulates s_waitcnt-like instructions used for dependency tracking.
static StinkyInstruction* createBarrierDestInBlock(BasicBlock* bb, GfxArchID arch) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_waitcnt, arch));
    inst->addDestReg(StinkyRegister(RegType::LDS, 0, 1));
    inst->addSrcReg(StinkyRegister(0));  // literal 0
    return inst;
}

// Helper to create s_cmp_eq_u32 (writes SCC implicit register).
static StinkyInstruction* createSCmpEqU32InBlock(BasicBlock* bb, GfxArchID arch, int src0Reg,
                                                 int src1Reg) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, arch));
    inst->addDestReg(StinkyRegister::getSCCRegister());
    inst->addSrcReg(StinkyRegister("s", src0Reg, 1));
    inst->addSrcReg(StinkyRegister("s", src1Reg, 1));
    return inst;
}

// Helper to create s_cbranch_scc0 (reads SCC implicit register).
static StinkyInstruction* createSCbranchScc0InBlock(BasicBlock* bb, GfxArchID arch) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cbranch_scc0, arch));
    inst->addSrcReg(StinkyRegister::getSCCRegister());
    return inst;
}

class DefUseChainTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
};

// =============================================================================
// CFG:
//    A <-+       v0 = v1 + v2
//    |   |
//    v   |
//    B --+       v0 = v0 + v3
// =============================================================================
TEST_F(DefUseChainTest, LoopBack_TwoBlocks) {
    Function func("loop_back_two");
    setFunctionArch(func, arch);
    BasicBlock* A = func.createBasicBlock("A");
    BasicBlock* B = func.createBasicBlock("B");

    func.addEdge(A, B);
    func.addEdge(B, A);

    // A: v0 = v1 + v2 (defines v0)
    StinkyInstruction* aAdd = createVAddInBlock(A, arch, 0, 1, 2);
    // B: v0 = v0 + v3 (uses v0 from A or prev iteration, redefines v0)
    StinkyInstruction* bAdd = createVAddInBlock(B, arch, 0, 0, 3);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    // A is the entry and sole path to B.  B has 1 predecessor (A), so no PHI.
    // bAdd reads v0 from A via dominator-inherited reaching defs.
    EXPECT_EQ(countPhisInFunction(func), 0u) << "No merge point — no PHIs needed";
    EXPECT_EQ(aAdd->getSources().size(), 0u) << "A uses v1,v2 (undefined)";
    EXPECT_EQ(aAdd->getUsers().size(), 1u) << "A's v0 is used by B";
    EXPECT_EQ(bAdd->getSources().size(), 1u) << "B uses v0 from A";
    EXPECT_EQ(bAdd->getSources()[0], aAdd) << "B's v0 def is A's add";
    EXPECT_EQ(bAdd->getUsers().size(), 0u) << "B's v0 flows to A but A redefines first";
}

// =============================================================================
// CFG:
//    A    v0 = v0 + v1
//    | \.
//    +--+
// =============================================================================
TEST_F(DefUseChainTest, LoopBack_SelfLoop) {
    Function func("loop_self");
    setFunctionArch(func, arch);
    BasicBlock* A = func.createBasicBlock("A");

    func.addEdge(A, A);

    // A: v0 = v0 + v1 (uses v0 from prev iteration, redefines)
    StinkyInstruction* addInst = createVAddInBlock(A, arch, 0, 0, 1);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    // A has 1 predecessor (itself) — no merge point, so no PHI.
    // v0 and v1 are undefined at entry.
    EXPECT_EQ(countPhisInFunction(func), 0u) << "Single-predecessor self-loop — no PHI";
    EXPECT_EQ(addInst->getSources().size(), 0u) << "v0, v1 both undefined at A's entry";
}

// =============================================================================
// CFG:
//    A <---+-+-+   v10 = v0 + v1, v0 = v1 + v2
//    |     | | |
//    v     | | |
//    B ----+ | |   v0 = v0 + v3
//    |       | |
//    v       | |
//    C ------+ |   v0 = v0 + v4
//    |         |
//    v         |
//    D --------+   v0 = v0 + v5
// =============================================================================

TEST_F(DefUseChainTest, ThreePredecessors_ABCBDA) {
    Function func("three_preds");
    setFunctionArch(func, arch);
    BasicBlock* A = func.createBasicBlock("A");  // entry
    BasicBlock* B = func.createBasicBlock("B");
    BasicBlock* C = func.createBasicBlock("C");
    BasicBlock* D = func.createBasicBlock("D");

    // Linear: A -> B -> C -> D
    func.addEdge(A, B);
    func.addEdge(B, C);
    func.addEdge(C, D);

    // Loop backs: B -> A, C -> A, D -> A
    func.addEdge(B, A);
    func.addEdge(C, A);
    func.addEdge(D, A);

    // A: First instruction USES v0 (from B/C/D), then defines v0. So we need PHI(v0_from_B,
    // v0_from_C, v0_from_D). v10 = v0 + v1  (uses v0 - requires PHI to merge from B,C,D) v0 = v1 +
    // v2   (redefines v0)
    StinkyInstruction* aFirstAdd = createVAddInBlock(A, arch, 10, 0, 1);
    StinkyInstruction* aSecondAdd = createVAddInBlock(A, arch, 0, 1, 2);

    // B, C, D: v0 = v0 + v3/v4/v5 (uses v0 from predecessor, redefines)
    StinkyInstruction* bAdd = createVAddInBlock(B, arch, 0, 0, 3);
    StinkyInstruction* cAdd = createVAddInBlock(C, arch, 0, 0, 4);
    StinkyInstruction* dAdd = createVAddInBlock(D, arch, 0, 0, 5);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    // A has 3 predecessors (B, C, D) — the only join point, so PHI only in A.
    // B, C, D each have 1 predecessor and use the dominator-inherited reaching def.
    EXPECT_EQ(A->getPredecessors().size(), 3u) << "A must have 3 predecessors (B, C, D)";
    EXPECT_EQ(countPhis(*A), 1u) << "PHI for v0 in A";
    EXPECT_EQ(countPhis(*B), 0u) << "B has 1 pred — no PHI";
    EXPECT_EQ(countPhis(*C), 0u) << "C has 1 pred — no PHI";
    EXPECT_EQ(countPhis(*D), 0u) << "D has 1 pred — no PHI";

    StinkyInstruction* phi = &getStinkyInst(A->begin());
    ASSERT_EQ(phi->getUnifiedOpcode(), GFX::PHI) << "A's first inst is PHI";
    EXPECT_GE(aFirstAdd->getSources().size(), 1u);
    EXPECT_EQ(aFirstAdd->getSources()[0], phi) << "A's first add uses v0 from PHI";
    ASSERT_EQ(phi->getSources().size(), 3u) << "PHI in A should have 3 operand defs (from B, C, D)";

    // PHI operands are ordered by A's predecessors: sources[i] = def from preds[i].
    const std::vector<BasicBlock*>& preds = A->getPredecessors();
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == B) {
            EXPECT_EQ(phi->getSources()[i], bAdd) << "PHI operand for B should be B's add";
        } else if (preds[i] == C) {
            EXPECT_EQ(phi->getSources()[i], cAdd) << "PHI operand for C should be C's add";
        } else if (preds[i] == D) {
            EXPECT_EQ(phi->getSources()[i], dAdd) << "PHI operand for D should be D's add";
        }
    }

    // B, C, D each have 1 predecessor — they use the reaching def directly (no PHI).
    // B inherits v0 from A (aSecondAdd), C inherits from B (bAdd), D inherits from C (cAdd).
    EXPECT_EQ(bAdd->getSources()[0], aSecondAdd) << "B uses v0 from A (dominator)";
    EXPECT_EQ(cAdd->getSources()[0], bAdd) << "C uses v0 from B (dominator)";
    EXPECT_EQ(dAdd->getSources()[0], cAdd) << "D uses v0 from C (dominator)";
}

// =============================================================================
// CFG:
//      Entry      v0 = v1 + v2
//     /  |  \.
//    B   C   D    v0 = v0 + v3/v4/v5
//     \  |  /
//      Exit       v10 = v0 + v11
// =============================================================================

TEST_F(DefUseChainTest, ThreePredecessors_DiamondMerge) {
    Function func("diamond_three");
    setFunctionArch(func, arch);
    BasicBlock* entry = func.createBasicBlock("entry");
    BasicBlock* B = func.createBasicBlock("B");
    BasicBlock* C = func.createBasicBlock("C");
    BasicBlock* D = func.createBasicBlock("D");
    BasicBlock* exit = func.createBasicBlock("exit");

    func.addEdge(entry, B);
    func.addEdge(entry, C);
    func.addEdge(entry, D);
    func.addEdge(B, exit);
    func.addEdge(C, exit);
    func.addEdge(D, exit);

    EXPECT_EQ(exit->getPredecessors().size(), 3u) << "Exit must have 3 predecessors";

    // Entry: v0 = v1 + v2
    createVAddInBlock(entry, arch, 0, 1, 2);

    // B, C, D: each use v0 and define v0
    StinkyInstruction* bAdd = createVAddInBlock(B, arch, 0, 0, 3);
    StinkyInstruction* cAdd = createVAddInBlock(C, arch, 0, 0, 4);
    StinkyInstruction* dAdd = createVAddInBlock(D, arch, 0, 0, 5);

    // Exit: use v0
    StinkyInstruction* exitAdd = createVAddInBlock(exit, arch, 10, 0, 11);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    // Exit uses v0, which is defined in B, C, D. So Exit needs PHI(v0_from_B, v0_from_C,
    // v0_from_D).
    StinkyInstruction* phi = &getStinkyInst(exit->begin());
    ASSERT_EQ(phi->getUnifiedOpcode(), GFX::PHI) << "Exit's first inst is PHI";
    EXPECT_EQ(exitAdd->getSources().size(), 1u) << "Exit uses v0 from PHI";
    EXPECT_EQ(exitAdd->getSources()[0], phi) << "Exit add's v0 defined by PHI";
    EXPECT_EQ(phi->getSources().size(), 3u) << "PHI in Exit should have 3 operand defs";
    EXPECT_TRUE(std::find(phi->getSources().begin(), phi->getSources().end(), bAdd) !=
                phi->getSources().end());
    EXPECT_TRUE(std::find(phi->getSources().begin(), phi->getSources().end(), cAdd) !=
                phi->getSources().end());
    EXPECT_TRUE(std::find(phi->getSources().begin(), phi->getSources().end(), dAdd) !=
                phi->getSources().end());
}

// =============================================================================
// CFG:
//    Entry      v0 = v1 + v2
//    |
//    v
//    A <---+    v0 = v1 + v2
//    |     |
//    v     |
//    B     |    v10 = v11 + v12  (no v0 use/def)
//    |     |
//    v     |
//    C ----+    v20 = v0 + v21
// =============================================================================

TEST_F(DefUseChainTest, PassThrough_ValueUsedBySuccessorsSuccessor) {
    Function func("pass_through");
    setFunctionArch(func, arch);
    BasicBlock* entry = func.createBasicBlock("entry");
    BasicBlock* A = func.createBasicBlock("A");
    BasicBlock* B = func.createBasicBlock("B");
    BasicBlock* C = func.createBasicBlock("C");

    func.addEdge(entry, A);
    func.addEdge(A, B);
    func.addEdge(B, C);
    func.addEdge(C, A);

    // Entry: v0 = v1 + v2
    createVAddInBlock(entry, arch, 0, 1, 2);
    // A: v0 = v1 + v2 (redefines v0)
    StinkyInstruction* aAdd = createVAddInBlock(A, arch, 0, 1, 2);
    // B: no v0 use/def (pass-through)
    createVAddInBlock(B, arch, 10, 11, 12);  // v10 = v11 + v12
    // C: uses v0 (from B, which passes through from A)
    StinkyInstruction* cAdd = createVAddInBlock(C, arch, 20, 0, 21);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    // A has 2 predecessors (Entry, C) — a join point — but its PHI for v0 is dead
    // because aAdd redefines v0 immediately without reading the PHI.
    // Dead PHI removal erases it, leaving zero PHIs.
    EXPECT_EQ(countPhisInFunction(func), 0u) << "PHI in A is dead — removed";

    // C uses v0. v0 is defined by aAdd in A and passes through B unchanged.
    // C inherits the reaching def from its dominator chain (A → B → C).
    EXPECT_GE(cAdd->getSources().size(), 1u) << "C uses v0";
    EXPECT_EQ(cAdd->getSources()[0], aAdd) << "C's v0 from A (pass-through via B)";
}

// =============================================================================
// CFG: Exercises collapsePredDefs when a predecessor has defByPreds with size > 1.
//
// C is a pass-through (no v0 use/def) with three predecessors A, F, B.
// A and B define v0; F has no v0 (pass-through from Entry, which has no v0).
//
// When merging into D and E, we call collapsePredDefs(C, v0, [aAdd, nullptr, bAdd])
// which creates a PHI in C. The second merge hits the update path.
//
// Edge order: Entry -> A, Entry -> F, Entry -> B (so C's preds = [A, F, B])
//
//       Entry
//      /  |  \.
//     A   F   B
//     |   |   |
//   v0=x  |  v0=y  (F has no v0)
//       \ | /
//         C (pass-through, no v0)
//        / \.
//       D   E  (both use v0)
// =============================================================================

TEST_F(DefUseChainTest, CollapsePredDefs_PassThroughWithMultiplePredecessors) {
    Function func("collapse_pred_defs");
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
    // F: no v0 (pass-through from Entry, which has no v0)
    createVAddInBlock(F, arch, 5, 6, 7);  // v5 = v6 + v7
    // C: pass-through - will have defByPreds[v0] = [aAdd, nullptr, bAdd]
    createVAddInBlock(C, arch, 10, 11, 12);  // v10 = v11 + v12
    // D and E both use v0 - triggers collapsePredDefs(C, v0, [aAdd, nullptr, bAdd])
    StinkyInstruction* dAdd = createVAddInBlock(D, arch, 20, 0, 21);
    StinkyInstruction* eAdd = createVAddInBlock(E, arch, 30, 0, 31);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    // C has 3 predecessors (A, F, B) — the only join point with a needed PHI.
    // D and E each have 1 predecessor (C) — no PHIs there.
    EXPECT_EQ(countPhis(*C), 1u) << "PHI for v0 in C";
    EXPECT_EQ(countPhis(*D), 0u) << "D has 1 pred — no PHI";
    EXPECT_EQ(countPhis(*E), 0u) << "E has 1 pred — no PHI";

    StinkyInstruction* cPhi = &getStinkyInst(C->begin());
    ASSERT_EQ(cPhi->getUnifiedOpcode(), GFX::PHI) << "C's first inst is PHI";
    ASSERT_EQ(cPhi->getSources().size(), 3u) << "PHI in C should have 3 operands (from A, F, B)";
    EXPECT_EQ(cPhi->getSources()[0], aAdd) << "C's PHI operand for A should be aAdd";
    EXPECT_EQ(cPhi->getSources()[1], nullptr)
        << "C's PHI operand for F should be nullptr (F has no v0)";
    EXPECT_EQ(cPhi->getSources()[2], bAdd) << "C's PHI operand for B should be bAdd";

    // D and E use v0 directly from C's PHI (single predecessor — no extra PHI needed).
    EXPECT_EQ(dAdd->getSources()[0], cPhi) << "D's add uses C's PHI directly";
    EXPECT_EQ(eAdd->getSources()[0], cPhi) << "E's add uses C's PHI directly";
}

// =============================================================================
// CFG:
//    Entry      v0 = v1 + v2; s_waitcnt (BARRIER); v2 = v0 + v3
// =============================================================================

TEST_F(DefUseChainTest, PseudoRegistersIgnored) {
    Function func("pseudo_regs");
    setFunctionArch(func, arch);
    BasicBlock* bb = func.createBasicBlock("entry");

    StinkyInstruction* defV0 = createVAddInBlock(bb, arch, 0, 1, 2);      // v0 = v1 + v2
    StinkyInstruction* barrierInst = createBarrierDestInBlock(bb, arch);  // BARRIER = s_waitcnt 0
    StinkyInstruction* useV0 = createVAddInBlock(bb, arch, 2, 0, 3);      // v2 = v0 + v3

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    // barrierInst writes BARRIER (pseudo). Nothing should use BARRIER in def-use chain.
    EXPECT_TRUE(barrierInst->getUsers().empty())
        << "BARRIER dest instruction should have no users (pseudo regs ignored)";

    // useV0's sources: v0 from defV0 (v3 may be undefined). No BARRIER in chain.
    EXPECT_GE(useV0->getSources().size(), 1u) << "v_add uses v0";
    EXPECT_EQ(useV0->getSources()[0], defV0) << "First operand (v0) defined by defV0";
}

// =============================================================================
// CFG:
//    Entry      s_cmp_eq_u32 s0,s1 -> SCC; s_cbranch_scc0 (uses SCC)
// =============================================================================

TEST_F(DefUseChainTest, ImplicitRegistersDefUseChain) {
    Function func("implicit_scc");
    setFunctionArch(func, arch);
    BasicBlock* bb = func.createBasicBlock("entry");

    StinkyInstruction* cmpInst =
        createSCmpEqU32InBlock(bb, arch, 0, 1);  // s_cmp_eq_u32 s0, s1 -> SCC
    StinkyInstruction* branchInst =
        createSCbranchScc0InBlock(bb, arch);  // s_cbranch_scc0 (uses SCC)

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    // branch's sources should contain cmp (SCC operand).
    EXPECT_GE(branchInst->getSources().size(), 1u) << "branch uses SCC";
    EXPECT_EQ(branchInst->getSources()[0], cmpInst) << "branch's SCC operand defined by cmp";
}

// =============================================================================
// 1. Iterated Dominance Frontier — verify chains through cascading PHIs
// =============================================================================

TEST_F(DefUseChainTest, IteratedDominanceFrontier) {
    Function func("iterated_df");
    IteratedDFCfg cfg = buildIteratedDFCfg(func, arch);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    EXPECT_EQ(countPhis(*cfg.G), 1u);
    EXPECT_EQ(countPhis(*cfg.H), 1u);
    EXPECT_EQ(countPhisInFunction(func), 2u);

    StinkyInstruction* phiG = findPhi(*cfg.G, RegType::V, 0);
    StinkyInstruction* phiH = findPhi(*cfg.H, RegType::V, 0);
    ASSERT_NE(phiG, nullptr);
    ASSERT_NE(phiH, nullptr);

    // PHI_G: preds [E, F] → entryDef (via B→E), cDef (via C→F)
    ASSERT_EQ(phiG->getSources().size(), 2u);
    EXPECT_EQ(phiG->getSources()[0], cfg.entryDef);
    EXPECT_EQ(phiG->getSources()[1], cfg.cDef);

    // PHI_H: preds [D, G] → entryDef (via B→D), PHI_G
    ASSERT_EQ(phiH->getSources().size(), 2u);
    EXPECT_EQ(phiH->getSources()[0], cfg.entryDef);
    EXPECT_EQ(phiH->getSources()[1], phiG);

    // G's use → PHI_G, H's use → PHI_H
    EXPECT_GE(cfg.gUse->getSources().size(), 1u);
    EXPECT_EQ(cfg.gUse->getSources()[0], phiG);
    EXPECT_GE(cfg.hUse->getSources().size(), 1u);
    EXPECT_EQ(cfg.hUse->getSources()[0], phiH);

    // entryDef feeds both PHIs
    const auto& entryUsers = cfg.entryDef->getUsers();
    EXPECT_TRUE(std::find(entryUsers.begin(), entryUsers.end(), phiG) != entryUsers.end());
    EXPECT_TRUE(std::find(entryUsers.begin(), entryUsers.end(), phiH) != entryUsers.end());

    // PHI_G feeds PHI_H and G's use
    const auto& phiGUsers = phiG->getUsers();
    EXPECT_TRUE(std::find(phiGUsers.begin(), phiGUsers.end(), phiH) != phiGUsers.end());
    EXPECT_TRUE(std::find(phiGUsers.begin(), phiGUsers.end(), cfg.gUse) != phiGUsers.end());
}

// =============================================================================
// 2. Nested Loops — chains through both loop-header PHIs
// =============================================================================

TEST_F(DefUseChainTest, NestedLoops_DualPhiHeaderChains) {
    Function func("nested_loops");
    NestedLoopCfg cfg = buildNestedLoopCfg(func, arch);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    EXPECT_EQ(countPhis(*cfg.A), 1u);
    EXPECT_EQ(countPhis(*cfg.B), 1u);
    EXPECT_EQ(countPhisInFunction(func), 2u);

    StinkyInstruction* phiA = findPhi(*cfg.A, RegType::V, 0);
    StinkyInstruction* phiB = findPhi(*cfg.B, RegType::V, 0);
    ASSERT_NE(phiA, nullptr);
    ASSERT_NE(phiB, nullptr);

    // PHI_A:  preds [Entry, D] → entryDef, cDef
    ASSERT_EQ(phiA->getSources().size(), 2u);
    EXPECT_EQ(phiA->getSources()[0], cfg.entryDef);
    EXPECT_EQ(phiA->getSources()[1], cfg.cDef);

    // PHI_B:  preds [A, C] → PHI_A, cDef
    ASSERT_EQ(phiB->getSources().size(), 2u);
    EXPECT_EQ(phiB->getSources()[0], phiA);
    EXPECT_EQ(phiB->getSources()[1], cfg.cDef);

    // B's use → PHI_B
    EXPECT_GE(cfg.bUse->getSources().size(), 1u);
    EXPECT_EQ(cfg.bUse->getSources()[0], phiB);

    // D's use → cDef (inherited from C through dominator chain)
    EXPECT_GE(cfg.dUse->getSources().size(), 1u);
    EXPECT_EQ(cfg.dUse->getSources()[0], cfg.cDef);

    // cDef feeds PHI_A, PHI_B, and D's use
    const auto& cDefUsers = cfg.cDef->getUsers();
    EXPECT_TRUE(std::find(cDefUsers.begin(), cDefUsers.end(), phiA) != cDefUsers.end());
    EXPECT_TRUE(std::find(cDefUsers.begin(), cDefUsers.end(), phiB) != cDefUsers.end());
    EXPECT_TRUE(std::find(cDefUsers.begin(), cDefUsers.end(), cfg.dUse) != cDefUsers.end());
}

// =============================================================================
// 3. Self-loop at join — self-referential PHI in chain
// =============================================================================

TEST_F(DefUseChainTest, SelfLoopAtJoinPoint) {
    Function func("self_loop_join");
    SelfLoopJoinCfg cfg = buildSelfLoopJoinCfg(func, arch);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    EXPECT_EQ(countPhis(*cfg.C), 1u);
    EXPECT_EQ(countPhisInFunction(func), 1u);

    StinkyInstruction* phiC = findPhi(*cfg.C, RegType::V, 0);
    ASSERT_NE(phiC, nullptr);

    // PHI_C: preds [A, B, C] → aDef, bDef, phiC (self-referential)
    ASSERT_EQ(phiC->getSources().size(), 3u);
    EXPECT_EQ(phiC->getSources()[0], cfg.aDef);
    EXPECT_EQ(phiC->getSources()[1], cfg.bDef);
    EXPECT_EQ(phiC->getSources()[2], phiC);

    // C's use and D's use both chain to PHI_C
    EXPECT_GE(cfg.cUse->getSources().size(), 1u);
    EXPECT_EQ(cfg.cUse->getSources()[0], phiC);
    EXPECT_GE(cfg.dUse->getSources().size(), 1u);
    EXPECT_EQ(cfg.dUse->getSources()[0], phiC);

    // PHI_C is its own user (self-loop operand)
    const auto& phiCUsers = phiC->getUsers();
    EXPECT_TRUE(std::find(phiCUsers.begin(), phiCUsers.end(), phiC) != phiCUsers.end());
}

// =============================================================================
// 4. Irreducible CFG — mutually recursive PHI chains
// =============================================================================

TEST_F(DefUseChainTest, IrreducibleCFG_MutualRecursiveChains) {
    Function func("irreducible");
    IrreducibleCfg cfg = buildIrreducibleCfg(func, arch);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    EXPECT_EQ(countPhis(*cfg.C), 1u);
    EXPECT_EQ(countPhis(*cfg.D), 1u);
    EXPECT_EQ(countPhis(*cfg.E), 1u);
    EXPECT_EQ(countPhisInFunction(func), 3u);

    StinkyInstruction* phiC = findPhi(*cfg.C, RegType::V, 0);
    StinkyInstruction* phiD = findPhi(*cfg.D, RegType::V, 0);
    StinkyInstruction* phiE = findPhi(*cfg.E, RegType::V, 0);
    ASSERT_NE(phiC, nullptr);
    ASSERT_NE(phiD, nullptr);
    ASSERT_NE(phiE, nullptr);

    // PHI_C and PHI_D are mutual operands
    ASSERT_EQ(phiC->getSources().size(), 2u);
    EXPECT_EQ(phiC->getSources()[0], cfg.aDef);
    EXPECT_EQ(phiC->getSources()[1], phiD);

    ASSERT_EQ(phiD->getSources().size(), 2u);
    EXPECT_EQ(phiD->getSources()[0], cfg.bDef);
    EXPECT_EQ(phiD->getSources()[1], phiC);

    // PHI_E merges PHI_C and PHI_D
    ASSERT_EQ(phiE->getSources().size(), 2u);
    EXPECT_EQ(phiE->getSources()[0], phiC);
    EXPECT_EQ(phiE->getSources()[1], phiD);

    // E's use → PHI_E
    EXPECT_GE(cfg.eUse->getSources().size(), 1u);
    EXPECT_EQ(cfg.eUse->getSources()[0], phiE);
}

// =============================================================================
// 5. Multiple registers at same join — per-register chains
// =============================================================================

TEST_F(DefUseChainTest, MultipleRegistersAtSameJoin) {
    Function func("multi_reg_join");
    MultiRegJoinCfg cfg = buildMultiRegJoinCfg(func, arch);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    EXPECT_EQ(countPhis(*cfg.C), 2u);

    StinkyInstruction* phiV0 = findPhi(*cfg.C, RegType::V, 0);
    StinkyInstruction* phiV1 = findPhi(*cfg.C, RegType::V, 1);
    ASSERT_NE(phiV0, nullptr);
    ASSERT_NE(phiV1, nullptr);

    // cUseV0's v0 operand → PHI(v0)
    EXPECT_GE(cfg.cUseV0->getSources().size(), 1u);
    EXPECT_EQ(cfg.cUseV0->getSources()[0], phiV0);

    // cUseV1's v1 operand → PHI(v1)
    EXPECT_GE(cfg.cUseV1->getSources().size(), 1u);
    EXPECT_EQ(cfg.cUseV1->getSources()[0], phiV1);
}

// =============================================================================
// 6. Chain of diamonds — chains through 3 levels of PHIs
// =============================================================================

TEST_F(DefUseChainTest, ChainOfDiamonds) {
    Function func("chain_diamonds");
    ChainOfDiamondsCfg cfg = buildChainOfDiamondsCfg(func, arch);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    EXPECT_EQ(countPhis(*cfg.C), 1u);
    EXPECT_EQ(countPhis(*cfg.F), 1u);
    EXPECT_EQ(countPhis(*cfg.I), 1u);
    EXPECT_EQ(countPhisInFunction(func), 3u);

    StinkyInstruction* phiC = findPhi(*cfg.C, RegType::V, 0);
    StinkyInstruction* phiF = findPhi(*cfg.F, RegType::V, 0);
    StinkyInstruction* phiI = findPhi(*cfg.I, RegType::V, 0);
    ASSERT_NE(phiC, nullptr);
    ASSERT_NE(phiF, nullptr);
    ASSERT_NE(phiI, nullptr);

    // Cascading chain: entryDef/bDef → PHI_C → PHI_F → PHI_I → iUse
    EXPECT_EQ(phiC->getSources()[0], cfg.entryDef);
    EXPECT_EQ(phiC->getSources()[1], cfg.bDef);
    EXPECT_EQ(phiF->getSources()[0], phiC);
    EXPECT_EQ(phiF->getSources()[1], cfg.eDef);
    EXPECT_EQ(phiI->getSources()[0], phiF);
    EXPECT_EQ(phiI->getSources()[1], cfg.hDef);

    EXPECT_GE(cfg.iUse->getSources().size(), 1u);
    EXPECT_EQ(cfg.iUse->getSources()[0], phiI);

    // Verify user linkage through the chain
    const auto& phiCUsers = phiC->getUsers();
    EXPECT_TRUE(std::find(phiCUsers.begin(), phiCUsers.end(), phiF) != phiCUsers.end());
    const auto& phiFUsers = phiF->getUsers();
    EXPECT_TRUE(std::find(phiFUsers.begin(), phiFUsers.end(), phiI) != phiFUsers.end());
}

// =============================================================================
// 7. Dead register — no PHIs, no v0 chains
// =============================================================================

TEST_F(DefUseChainTest, DeadRegister_NoChains) {
    Function func("dead_reg");
    DeadRegCfg cfg = buildDeadRegCfg(func, arch);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    EXPECT_EQ(countPhisInFunction(func), 0u) << "v0 is never used as source — no PHIs";

    // Dead v0 defs should have no users
    EXPECT_TRUE(cfg.aDef->getUsers().empty()) << "aDef (v0) is dead — no users";
    EXPECT_TRUE(cfg.bDef->getUsers().empty()) << "bDef (v0) is dead — no users";
}

// =============================================================================
// 8. Re-definition within same block — last def wins
// =============================================================================

TEST_F(DefUseChainTest, RedefinitionInSameBlock) {
    Function func("redef_same_block");
    RedefSameBlockCfg cfg = buildRedefSameBlockCfg(func, arch);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    EXPECT_EQ(countPhis(*cfg.C), 1u);

    StinkyInstruction* phiC = findPhi(*cfg.C, RegType::V, 0);
    ASSERT_NE(phiC, nullptr);

    // PHI operand from A must be aDef2 (second def shadows first)
    ASSERT_EQ(phiC->getSources().size(), 2u);
    EXPECT_EQ(phiC->getSources()[0], cfg.aDef2);
    EXPECT_EQ(phiC->getSources()[1], cfg.bDef);

    // C's use → PHI_C
    EXPECT_GE(cfg.cUse->getSources().size(), 1u);
    EXPECT_EQ(cfg.cUse->getSources()[0], phiC);

    // aDef1 is dead (shadowed by aDef2 without being read)
    EXPECT_TRUE(cfg.aDef1->getUsers().empty()) << "First def in A is dead — shadowed by second def";
    EXPECT_FALSE(cfg.aDef2->getUsers().empty()) << "Second def in A is used by PHI";
}

// =============================================================================
// 9. Wide register with partial sub-register redefine — per-DWORD chains
//
// Entry defines v[0:3] (4 DWORDs).  C redefines only v0.
// PHIs appear only for v0 (at G and H).
// v1, v2, v3 chain directly back to Entry's wide load with no PHIs.
// =============================================================================

TEST_F(DefUseChainTest, WideRegPartialSubregRedefine) {
    Function func("wide_partial_redef");
    WideRegPartialRedefCfg cfg = buildWideRegPartialRedefCfg(func, arch);

    buildUseDefChain(func, false);

    verifyDefUseChainConsistency(func);
    verifyUsersSourcesConsistency(func);

    EXPECT_EQ(countPhis(*cfg.G), 1u);
    EXPECT_EQ(countPhis(*cfg.H), 1u);
    EXPECT_EQ(countPhisInFunction(func), 2u);

    StinkyInstruction* phiG = findPhi(*cfg.G, RegType::V, 0);
    StinkyInstruction* phiH = findPhi(*cfg.H, RegType::V, 0);
    ASSERT_NE(phiG, nullptr);
    ASSERT_NE(phiH, nullptr);

    // PHI_G: preds [E, F] → entryWideDef (via B→E), cPartialDef (via C→F)
    ASSERT_EQ(phiG->getSources().size(), 2u);
    EXPECT_EQ(phiG->getSources()[0], cfg.entryWideDef);
    EXPECT_EQ(phiG->getSources()[1], cfg.cPartialDef);

    // PHI_H: preds [D, G] → entryWideDef (via B→D), PHI_G
    ASSERT_EQ(phiH->getSources().size(), 2u);
    EXPECT_EQ(phiH->getSources()[0], cfg.entryWideDef);
    EXPECT_EQ(phiH->getSources()[1], phiG);

    // G's use: v10 = v0 + v2
    //   v0 → PHI_G,  v2 → entryWideDef (from the 4-DWORD load)
    ASSERT_EQ(cfg.gUse->getSources().size(), 2u);
    EXPECT_EQ(cfg.gUse->getSources()[0], phiG);
    EXPECT_EQ(cfg.gUse->getSources()[1], cfg.entryWideDef);

    // H's use: v11 = v0 + v1
    //   v0 → PHI_H,  v1 → entryWideDef (from the 4-DWORD load)
    ASSERT_EQ(cfg.hUse->getSources().size(), 2u);
    EXPECT_EQ(cfg.hUse->getSources()[0], phiH);
    EXPECT_EQ(cfg.hUse->getSources()[1], cfg.entryWideDef);

    // entryWideDef users: PHI_G (v0 via E), PHI_H (v0 via D), gUse (v2), hUse (v1)
    const auto& entryUsers = cfg.entryWideDef->getUsers();
    EXPECT_TRUE(std::find(entryUsers.begin(), entryUsers.end(), phiG) != entryUsers.end());
    EXPECT_TRUE(std::find(entryUsers.begin(), entryUsers.end(), phiH) != entryUsers.end());
    EXPECT_TRUE(std::find(entryUsers.begin(), entryUsers.end(), cfg.gUse) != entryUsers.end());
    EXPECT_TRUE(std::find(entryUsers.begin(), entryUsers.end(), cfg.hUse) != entryUsers.end());

    // PHI_G feeds PHI_H and G's use
    const auto& phiGUsers = phiG->getUsers();
    EXPECT_TRUE(std::find(phiGUsers.begin(), phiGUsers.end(), phiH) != phiGUsers.end());
    EXPECT_TRUE(std::find(phiGUsers.begin(), phiGUsers.end(), cfg.gUse) != phiGUsers.end());
}
