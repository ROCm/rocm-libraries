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

#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

using namespace stinkytofu;

class WaitCntInsertionTest : public ::testing::Test {
   protected:
    void SetUp() override {
        gemmConfig.arch[0] = 12;
        gemmConfig.arch[1] = 5;
        gemmConfig.arch[2] = 0;
    }

    const std::array<int, 3>& getArch() const {
        return gemmConfig.arch;
    }

    Function* parseIR(const std::string& irString, StinkyIRConverter& converter) {
        auto* func = converter.convertToFunction(irString);
        if (func) func->setGemmTileConfig(gemmConfig);
        return func;
    }

    void runInsertionPass(Function& func, WaitCntInsertionOptions options = {}) {
        PassContext passCtx;
        passCtx.setGemmTileConfig(gemmConfig);
        AnalysisManager am;
        registerAllAnalyses(am);
        auto pass = stinkytofu::createStinkyWaitCntInsertionPass(options);
        pass->run(func, passCtx, am);
    }

    struct WaitCntInfo {
        StinkyInstruction* inst;
        SWaitCntData* waitData;
        int position;

        WaitCntInfo(StinkyInstruction* i, SWaitCntData* w, int p)
            : inst(i), waitData(w), position(p) {}
    };

    struct TensorWaitCntInfo {
        StinkyInstruction* inst;
        SWaitTensorCntData* tensorWaitData;
        int position;

        TensorWaitCntInfo(StinkyInstruction* i, SWaitTensorCntData* t, int p)
            : inst(i), tensorWaitData(t), position(p) {}
    };

    int countWaitCnt(BasicBlock& bb) {
        int count = 0;
        for (auto& irBase : bb) {
            if (static_cast<StinkyInstruction&>(irBase).getModifier<SWaitCntData>()) count++;
        }
        return count;
    }

    int countTensorWaitCnt(BasicBlock& bb) {
        int count = 0;
        for (auto& irBase : bb) {
            if (static_cast<StinkyInstruction&>(irBase).getModifier<SWaitTensorCntData>()) count++;
        }
        return count;
    }

    std::vector<WaitCntInfo> getAllWaitCnts(BasicBlock& bb) {
        std::vector<WaitCntInfo> waitcnts;
        int position = 0;
        for (auto& irBase : bb) {
            StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
            if (SWaitCntData* wait = inst.getModifier<SWaitCntData>())
                waitcnts.emplace_back(&inst, wait, position);
            position++;
        }
        return waitcnts;
    }

    std::vector<TensorWaitCntInfo> getAllTensorWaitCnts(BasicBlock& bb) {
        std::vector<TensorWaitCntInfo> tensorWaitcnts;
        int position = 0;
        for (auto& irBase : bb) {
            StinkyInstruction& inst = static_cast<StinkyInstruction&>(irBase);
            if (SWaitTensorCntData* tw = inst.getModifier<SWaitTensorCntData>())
                tensorWaitcnts.emplace_back(&inst, tw, position);
            position++;
        }
        return tensorWaitcnts;
    }

    int getInstructionPosition(BasicBlock& bb, StinkyInstruction* target) {
        int position = 0;
        for (auto& irBase : bb) {
            if (&static_cast<StinkyInstruction&>(irBase) == target) return position;
            position++;
        }
        return -1;
    }

    template <typename OpcodeT>
    StinkyInstruction* findNthInst(BasicBlock& bb, OpcodeT opcode, int n = 0) {
        int count = 0;
        for (auto& irBase : bb) {
            auto& inst = static_cast<StinkyInstruction&>(irBase);
            if (inst.getUnifiedOpcode() == opcode) {
                if (count == n) return &inst;
                count++;
            }
        }
        return nullptr;
    }

    SWaitCntData* findWaitCntBefore(BasicBlock& bb, StinkyInstruction* target) {
        BasicBlock::iterator targetIt = bb.end();
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            if (&static_cast<StinkyInstruction&>(*it) == target) {
                targetIt = it;
                break;
            }
        }

        if (targetIt == bb.end() || targetIt == bb.begin()) return nullptr;

        auto prevIt = targetIt;
        --prevIt;

        while (true) {
            StinkyInstruction& prevInst = static_cast<StinkyInstruction&>(*prevIt);
            if (SWaitCntData* wait = prevInst.getModifier<SWaitCntData>()) return wait;
            if (prevInst.getModifier<SWaitTensorCntData>()) {
                if (prevIt == bb.begin()) return nullptr;
                --prevIt;
                continue;
            }
            return nullptr;
        }
    }

    SWaitTensorCntData* findTensorWaitCntBefore(BasicBlock& bb, StinkyInstruction* target) {
        BasicBlock::iterator targetIt = bb.end();
        for (auto it = bb.begin(); it != bb.end(); ++it) {
            if (&static_cast<StinkyInstruction&>(*it) == target) {
                targetIt = it;
                break;
            }
        }

        if (targetIt == bb.end() || targetIt == bb.begin()) return nullptr;

        auto prevIt = targetIt;
        --prevIt;

        while (true) {
            StinkyInstruction& prevInst = static_cast<StinkyInstruction&>(*prevIt);
            if (SWaitTensorCntData* tw = prevInst.getModifier<SWaitTensorCntData>()) return tw;
            if (prevInst.getModifier<SWaitCntData>()) {
                if (prevIt == bb.begin()) return nullptr;
                --prevIt;
                continue;
            }
            return nullptr;
        }
    }

    static bool hasLoopBackEdge(const BasicBlock* bb) {
        const auto& successors = bb->getSuccessors();
        return std::find(successors.begin(), successors.end(), bb) != successors.end();
    }

    GemmTileConfig gemmConfig;
};

// ============================================================================
// Test Suite 1: Barrier Wait Insertion
//
// Tests that barriers do NOT trigger waitcnts when there are no def-use
// dependencies through the barrier (StinkyWaitCntInsertionPass uses
// def-use chain analysis, not config-based barrier policies).
//
// MemTokenData is intentionally attached with disjoint token IDs so that
// the conservative-fallback drain (which fires when either the barrier or
// a pending DS op lacks MemTokenData) does not trigger; that fallback is
// covered separately by the ConservativeFallback_* tests in Test Suite 6.
// ============================================================================

/**
 * @brief ds_load with no consumer followed by barrier -> no waitcnt.
 *
 * IR:
 *   ds_load_b64 v[0:1], v10     tokens=[100]
 *   s_barrier                    tokens=[200]
 *
 * The ds_load result v[0:1] is never consumed, so the pass inserts no waitcnt.
 * Tokens are disjoint so the barrier-vs-DS conflict path stays silent.
 */
TEST_F(WaitCntInsertionTest, BarrierWithDSRead) {
    std::string irString = R"(
st.func @test_barrier_ds_read() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.memtoken = { tokens = [100] } }
  "st.s_barrier"() { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [200] } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    EXPECT_EQ(countWaitCnt(entryBB), 0) << "No consumer of ds_load result, no waitcnt needed";
    EXPECT_EQ(countTensorWaitCnt(entryBB), 0);
}

/**
 * @brief tensor_load + ds_load with no consumer followed by barrier -> no waitcnt.
 *
 * IR:
 *   tensor_load_to_lds s[0:3], s[10:17]   tokens=[100]
 *   ds_load_b64 v[0:1], v10                tokens=[200]
 *   s_barrier                               tokens=[300]
 *
 * Neither the tensor_load nor the ds_load results are consumed by any
 * subsequent instruction. Tokens are disjoint so neither the DS-barrier
 * conflict path nor the tensor-barrier matching path fires.
 */
TEST_F(WaitCntInsertionTest, BarrierWithDSReadTensorLoad) {
    std::string irString = R"(
st.func @test_barrier_tensor_ds() {
^entry:
  "st.tensor_load_to_lds"(s[0:3], s[10:17]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [100] } }
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.memtoken = { tokens = [200] } }
  "st.s_barrier"() { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [300] } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    EXPECT_EQ(countWaitCnt(entryBB), 0) << "No consumer of ds_load/tensor_load, no waitcnt needed";
    EXPECT_EQ(countTensorWaitCnt(entryBB), 0)
        << "Disjoint tokens on tensor_load/barrier, no tensor waitcnt";
}

// ============================================================================
// Test Suite 2: DS Read Insertion before WMMA
//
// Tests that waitcnts are inserted before instructions that consume
// ds_load results through def-use dependencies.
// ============================================================================

/**
 * @brief ds_loads consumed by two WMMA instructions -> waitcnt before each WMMA.
 *
 * IR:
 *   ds_load_b64 v[20:21], v0   (load 0)
 *   ds_load_b64 v[30:31], v0   (load 1)
 *   ds_load_b64 v[40:41], v0   (load 2)
 *   ds_load_b64 v[50:51], v0   (load 3)
 *   v_add_f32 v60, v61, v62    (independent)
 *   wmma a[10:17], v[20:27], v[30:37], a[10:17]  (uses loads 0,1)
 *   v_add_f32 v60, v61, v62    (independent)
 *   wmma a[10:17], v[40:47], v[50:57], a[10:17]  (uses loads 2,3)
 *
 * Expected:
 *   s_wait_dscnt 2 before wmma1 (wait for loads 0,1; leave loads 2,3)
 *   s_wait_dscnt 0 before wmma2 (wait for loads 2,3)
 */
TEST_F(WaitCntInsertionTest, DSReadBeforeWMMA) {
    std::string irString = R"(
st.func @test_ds_read_wmma() {
^entry:
  v[20:21] = "st.ds_load_b64"(v0) { issueCycles = 1, latencyCycles = 52 }
  v[30:31] = "st.ds_load_b64"(v0) { issueCycles = 1, latencyCycles = 52 }
  v[40:41] = "st.ds_load_b64"(v0) { issueCycles = 1, latencyCycles = 52 }
  v[50:51] = "st.ds_load_b64"(v0) { issueCycles = 1, latencyCycles = 52 }
  v60 = "st.v_add_f32"(v61, v62) { issueCycles = 1, latencyCycles = 1 }
  a[10:17] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[30:37], a[10:17]) { issueCycles = 4, latencyCycles = 8 }
  v60 = "st.v_add_f32"(v61, v62) { issueCycles = 1, latencyCycles = 1 }
  a[10:17] = "st.v_wmma_f32_16x16x32_bf16"(v[40:47], v[50:57], a[10:17]) { issueCycles = 4, latencyCycles = 8 }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    auto waitcnts = getAllWaitCnts(entryBB);

    ASSERT_EQ(waitcnts.size(), 2) << "Should have exactly 2 waitcnts (before each WMMA)";

    StinkyInstruction* wmma1 = findNthInst(entryBB, GFX::v_wmma_f32_16x16x32_bf16, 0);
    StinkyInstruction* wmma2 = findNthInst(entryBB, GFX::v_wmma_f32_16x16x32_bf16, 1);
    ASSERT_NE(wmma1, nullptr);
    ASSERT_NE(wmma2, nullptr);

    int wmma1Pos = getInstructionPosition(entryBB, wmma1);
    int wmma2Pos = getInstructionPosition(entryBB, wmma2);

    EXPECT_EQ(waitcnts[0].position, wmma1Pos - 1);
    EXPECT_EQ(waitcnts[0].waitData->dlcnt, 2) << "Wait for loads 0,1 (leave loads 2,3) -> dlcnt=2";
    EXPECT_EQ(waitcnts[0].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->vscnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->kmcnt, -1);

    EXPECT_EQ(waitcnts[1].position, wmma2Pos - 1);
    EXPECT_EQ(waitcnts[1].waitData->dlcnt, 0) << "Wait for all remaining loads (2,3) -> dlcnt=0";
    EXPECT_EQ(waitcnts[1].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->vscnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->kmcnt, -1);
}

/**
 * @brief SMRD scalar loads (s_load_*) consumed downstream -> s_wait_kmcnt.
 *
 * SMRD scalar loads retire on the kmcnt counter, not loadcnt/dscnt. The pass
 * must classify them as CK_KM and emit s_wait_kmcnt (SWaitCntData::kmcnt)
 * before each consumer, with the FIFO counter math identical to the other
 * counters.
 *
 * IR:
 *   s[8:9]   = s_load_b64 (scalar load 0)
 *   s[10:11] = s_load_b64 (scalar load 1)
 *   s20 = s_add_u32 s8, s8     (uses load 0)
 *   s21 = s_add_u32 s10, s10   (uses load 1)
 *
 * Expected:
 *   s_wait_kmcnt 1 before first  s_add_u32 (wait for load 0; leave load 1)
 *   s_wait_kmcnt 0 before second s_add_u32 (wait for the remaining load 1)
 * and no other counter fields are set.
 */
TEST_F(WaitCntInsertionTest, SMemLoadBeforeConsumerKmcnt) {
    std::string irString = R"(
st.func @test_smem_load_kmcnt() {
^entry:
  s[8:9] = "st.s_load_b64"(s[0:1]) { issueCycles = 1, latencyCycles = 20 }
  s[10:11] = "st.s_load_b64"(s[0:1]) { issueCycles = 1, latencyCycles = 20 }
  s20 = "st.s_add_u32"(s8, s8) { issueCycles = 1, latencyCycles = 1 }
  s21 = "st.s_add_u32"(s10, s10) { issueCycles = 1, latencyCycles = 1 }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    auto waitcnts = getAllWaitCnts(entryBB);

    ASSERT_EQ(waitcnts.size(), 2) << "Should have exactly 2 waitcnts (before each consumer)";

    StinkyInstruction* add1 = findNthInst(entryBB, GFX::s_add_u32, 0);
    StinkyInstruction* add2 = findNthInst(entryBB, GFX::s_add_u32, 1);
    ASSERT_NE(add1, nullptr);
    ASSERT_NE(add2, nullptr);

    int add1Pos = getInstructionPosition(entryBB, add1);
    int add2Pos = getInstructionPosition(entryBB, add2);

    EXPECT_EQ(waitcnts[0].position, add1Pos - 1);
    EXPECT_EQ(waitcnts[0].waitData->kmcnt, 1) << "Wait for load 0 (leave load 1) -> kmcnt=1";
    EXPECT_EQ(waitcnts[0].waitData->dlcnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->vscnt, -1);

    EXPECT_EQ(waitcnts[1].position, add2Pos - 1);
    EXPECT_EQ(waitcnts[1].waitData->kmcnt, 0) << "Wait for remaining load 1 -> kmcnt=0";
    EXPECT_EQ(waitcnts[1].waitData->dlcnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->vscnt, -1);
}

// ============================================================================
// Test Suite 3: Complete Preloop + WMMA + Barrier Pattern
//
// Tests the full pattern of preloop ds_loads, tensor_load, in-loop ds_loads,
// WMMA consumers, and barriers in a single basic block.
// ============================================================================

/**
 * @brief Full preloop pattern with two WMMA consumers and barriers.
 *
 * IR:
 *   ds_load_b128 v[0:3], v40     (preloop load 0)
 *   ds_load_b128 v[4:7], v40     (preloop load 1)
 *   ds_load_b128 v[8:11], v40    (preloop load 2)
 *   ds_load_b128 v[12:15], v40   (preloop load 3)
 *   tensor_load_to_lds            (tensor prefetch)
 *   ds_load_b128 v[16:19], v40   (load 4)
 *   ds_load_b128 v[20:23], v40   (load 5)
 *   ds_load_b128 v[24:27], v40   (load 6)
 *   ds_load_b128 v[28:31], v40   (load 7)
 *   wmma a[50:57], v[0:7], v[8:15], a[50:57]    (uses loads 0-3)
 *   s_barrier
 *   ds_load_b128 v[0:3], v40     (load 8)
 *   ds_load_b128 v[4:7], v40     (load 9)
 *   ds_load_b128 v[8:11], v40    (load 10)
 *   ds_load_b128 v[12:15], v40   (load 11)
 *   wmma a[50:57], v[16:23], v[24:31], a[50:57]  (uses loads 4-7)
 *   s_barrier
 *
 * Expected:
 *   s_wait_dscnt 4 before wmma1 (wait for preloop loads 0-3, leave loads 4-7)
 *   s_wait_dscnt 4 before wmma2 (wait for loads 4-7, leave loads 8-11)
 *   No waitcnt before barriers (no MemTokenData, no def-use dependency)
 */
TEST_F(WaitCntInsertionTest, CompleteTest) {
    // Disjoint tokens per phase so neither the barrier-vs-DS conflict path nor
    // the WAR-on-LDS path fires; this isolates the test to the def-use-driven
    // DS wait insertion before each WMMA consumer.
    std::string irString = R"(
st.func @test_complete() {
^entry:
  v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [100] } }
  v[4:7] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [100] } }
  v[8:11] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [100] } }
  v[12:15] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [100] } }
  "st.tensor_load_to_lds"(s[0:3], s[10:17]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [200] } }
  v[16:19] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [300] } }
  v[20:23] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [300] } }
  v[24:27] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [300] } }
  v[28:31] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [300] } }
  a[50:57] = "st.v_wmma_f32_16x16x32_bf16"(v[0:7], v[8:15], a[50:57]) { issueCycles = 4, latencyCycles = 8 }
  "st.s_barrier"() { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [400] } }
  v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [500] } }
  v[4:7] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [500] } }
  v[8:11] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [500] } }
  v[12:15] = "st.ds_load_b128"(v40) { issueCycles = 1, latencyCycles = 56, mod.memtoken = { tokens = [500] } }
  a[50:57] = "st.v_wmma_f32_16x16x32_bf16"(v[16:23], v[24:31], a[50:57]) { issueCycles = 4, latencyCycles = 8 }
  "st.s_barrier"() { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [600] } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    auto waitcnts = getAllWaitCnts(entryBB);
    auto tensorWaitcnts = getAllTensorWaitCnts(entryBB);

    StinkyInstruction* wmma1 = findNthInst(entryBB, GFX::v_wmma_f32_16x16x32_bf16, 0);
    StinkyInstruction* wmma2 = findNthInst(entryBB, GFX::v_wmma_f32_16x16x32_bf16, 1);
    StinkyInstruction* barrier1 = findNthInst(entryBB, GFX::s_barrier, 0);
    StinkyInstruction* barrier2 = findNthInst(entryBB, GFX::s_barrier, 1);
    ASSERT_NE(wmma1, nullptr);
    ASSERT_NE(wmma2, nullptr);
    ASSERT_NE(barrier1, nullptr);
    ASSERT_NE(barrier2, nullptr);

    int wmma1Pos = getInstructionPosition(entryBB, wmma1);
    int wmma2Pos = getInstructionPosition(entryBB, wmma2);
    int barrier1Pos = getInstructionPosition(entryBB, barrier1);
    int barrier2Pos = getInstructionPosition(entryBB, barrier2);

    ASSERT_EQ(waitcnts.size(), 2) << "Should have 2 waitcnts (before each WMMA)";

    EXPECT_EQ(waitcnts[0].position, wmma1Pos - 1) << "First waitcnt should be right before wmma1";
    EXPECT_EQ(waitcnts[0].waitData->dlcnt, 4)
        << "Wait for preloop ds_loads (v0-15) before wmma1 -> dlcnt=4";
    EXPECT_EQ(waitcnts[0].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->vscnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[0].waitData->kmcnt, -1);

    EXPECT_EQ(waitcnts[1].position, wmma2Pos - 1) << "Second waitcnt should be right before wmma2";
    EXPECT_EQ(waitcnts[1].waitData->dlcnt, 4)
        << "Wait for ds_loads (v16-31) before wmma2 -> dlcnt=4";
    EXPECT_EQ(waitcnts[1].waitData->vlcnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->vscnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->dscnt, -1);
    EXPECT_EQ(waitcnts[1].waitData->kmcnt, -1);

    for (const auto& wait : waitcnts) {
        EXPECT_NE(wait.position, barrier1Pos - 1)
            << "Should not have waitcnt right before barrier1";
        EXPECT_NE(wait.position, barrier2Pos - 1)
            << "Should not have waitcnt right before barrier2";
    }
}

// ============================================================================
// Test Suite 4: Basic Block State Tracking
//
// Tests cross-block def-use dependency tracking: non-loop, self-loop,
// preloop+loop, and diamond CFG patterns.
// ============================================================================

/**
 * @brief Single block, uses appear before loads (no loop) -> no waitcnt.
 *
 * IR:
 *   v_add_f32 v4, v0, v2   (uses v0, v2 - not loaded yet)
 *   v_add_f32 v4, v1, v3   (uses v1, v3 - not loaded yet)
 *   ds_load_b32 v0, v10    (load v0 AFTER use)
 *   ds_load_b32 v2, v10    (load v2 AFTER use)
 *   ds_load_b32 v1, v10    (load v1 AFTER use)
 *   ds_load_b32 v3, v10    (load v3 AFTER use)
 *
 * Since this is NOT a loop, there is no dependency from a previous iteration.
 * The loads come after the uses, so no waitcnt is needed.
 */
TEST_F(WaitCntInsertionTest, BasicBlockStateTracking_NoLoop) {
    std::string irString = R"(
st.func @test_no_loop() {
^entry:
  v4 = "st.v_add_f32"(v0, v2) { issueCycles = 1, latencyCycles = 1 }
  v4 = "st.v_add_f32"(v1, v3) { issueCycles = 1, latencyCycles = 1 }
  v0 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v2 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v1 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v3 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    BasicBlock& entryBB = *func->begin();
    EXPECT_FALSE(hasLoopBackEdge(&entryBB)) << "Should NOT be a loop block";

    runInsertionPass(*func);

    StinkyInstruction* fmac1 = findNthInst(entryBB, GFX::v_add_f32, 0);
    ASSERT_NE(fmac1, nullptr);

    SWaitCntData* waitBeforeFmac1 = findWaitCntBefore(entryBB, fmac1);
    EXPECT_EQ(waitBeforeFmac1, nullptr)
        << "Should NOT insert waitcnt - loads come AFTER uses (no dependency)";
}

/**
 * @brief Loop block with back-edge to itself -> waitcnt for cross-iteration deps.
 *
 * CFG:
 *   entry -> loop_start -> loop_start (self-loop)
 *
 * loop_start:
 *   v_add_f32 v4, v0, v2   (uses v0, v2 from previous iteration)
 *   v_add_f32 v4, v1, v3   (uses v1, v3 from previous iteration)
 *   ds_load_b32 v0, v10    (load for next iteration)
 *   ds_load_b32 v2, v10
 *   ds_load_b32 v1, v10
 *   ds_load_b32 v3, v10
 *
 * Expected:
 *   s_wait_dscnt 2 before first v_add (wait for v0,v2; leave v1,v3)
 *   s_wait_dscnt 0 before second v_add (wait for v1,v3)
 */
TEST_F(WaitCntInsertionTest, BasicBlockStateTracking_LoopOnly) {
    std::string irString = R"(
st.func @test_loop_only() {
^entry:
  Successors: ^loop_start
^loop_start:
  v4 = "st.v_add_f32"(v0, v2) { issueCycles = 1, latencyCycles = 1 }
  v4 = "st.v_add_f32"(v1, v3) { issueCycles = 1, latencyCycles = 1 }
  v0 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 1 }
  v2 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 1 }
  v1 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 1 }
  v3 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^loop_start
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    BasicBlock& loopBlock = *std::next(func->begin());
    EXPECT_TRUE(hasLoopBackEdge(&loopBlock)) << "Loop block should have back-edge to itself";

    runInsertionPass(*func);

    auto waitcnts = getAllWaitCnts(loopBlock);

    StinkyInstruction* fmac1 = findNthInst(loopBlock, GFX::v_add_f32, 0);
    StinkyInstruction* fmac2 = findNthInst(loopBlock, GFX::v_add_f32, 1);
    ASSERT_NE(fmac1, nullptr) << "Should find fmac1 in loop block";
    ASSERT_NE(fmac2, nullptr) << "Should find fmac2 in loop block";

    int fmac1Pos = getInstructionPosition(loopBlock, fmac1);
    int fmac2Pos = getInstructionPosition(loopBlock, fmac2);

    SWaitCntData* waitBeforeFmac1 = nullptr;
    for (const auto& wait : waitcnts) {
        if (wait.position < fmac1Pos && wait.position >= fmac1Pos - 2) {
            waitBeforeFmac1 = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac1, nullptr)
        << "Should insert waitcnt before first v_add (loop dependency)";
    EXPECT_EQ(waitBeforeFmac1->dlcnt, 2)
        << "Should wait for v0,v2 from previous iteration (leave v1,v3) -> dlcnt=2";

    SWaitCntData* waitBeforeFmac2 = nullptr;
    int waitBeforeFmac1Pos = -1;
    for (const auto& wait : waitcnts) {
        if (wait.waitData == waitBeforeFmac1) {
            waitBeforeFmac1Pos = wait.position;
            break;
        }
    }

    for (const auto& wait : waitcnts) {
        if (wait.position < fmac2Pos && wait.position >= fmac2Pos - 2 &&
            wait.position != waitBeforeFmac1Pos) {
            waitBeforeFmac2 = wait.waitData;
            break;
        }
    }

    if (waitBeforeFmac2) {
        EXPECT_EQ(waitBeforeFmac2->dlcnt, 0)
            << "Should wait for v1,v3 (all remaining loads) -> dlcnt=0";
    }
}

/**
 * @brief Two-block chain: entry loads, loop block uses (load order v0,v1,v2,v3).
 *
 * CFG:
 *   entry -> loop_start -> loop_start (self-loop)
 *
 * entry:
 *   ds_load_b32 v0, v10
 *   ds_load_b32 v1, v10
 *   ds_load_b32 v2, v10
 *   ds_load_b32 v3, v10
 *
 * loop_start:
 *   v_add_f32 v4, v0, v2   (uses v0 at pos 0, v2 at pos 2 -> wait up to pos 2)
 *   v_add_f32 v4, v1, v3   (uses v1, v3 -> wait for remaining)
 *   ds_load_b32 v0, v10
 *   ds_load_b32 v2, v10
 *   ds_load_b32 v1, v10
 *   ds_load_b32 v3, v10
 *
 * Expected:
 *   s_wait_dscnt 1 before first v_add (leave v3 outstanding) -> dlcnt=1
 *   s_wait_dscnt 0 before second v_add (wait for v3) -> dlcnt=0
 */
TEST_F(WaitCntInsertionTest, BasicBlockStateTracking_TwoBlockChain) {
    std::string irString = R"(
st.func @test_two_block_chain() {
^entry:
  v0 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v1 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v2 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v3 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  Successors: ^loop_start
^loop_start:
  v4 = "st.v_add_f32"(v0, v2) { issueCycles = 1, latencyCycles = 1 }
  v4 = "st.v_add_f32"(v1, v3) { issueCycles = 1, latencyCycles = 1 }
  v0 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v2 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v1 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v3 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  Successors: ^loop_start
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    BasicBlock& entryBB = *func->begin();
    BasicBlock& loopBlock = *std::next(func->begin());

    EXPECT_FALSE(hasLoopBackEdge(&entryBB)) << "Entry should not have loop back-edge";
    EXPECT_TRUE(hasLoopBackEdge(&loopBlock)) << "Loop block should have back-edge to itself";

    runInsertionPass(*func);

    auto waitcnts = getAllWaitCnts(loopBlock);

    StinkyInstruction* fmac1 = findNthInst(loopBlock, GFX::v_add_f32, 0);
    StinkyInstruction* fmac2 = findNthInst(loopBlock, GFX::v_add_f32, 1);
    ASSERT_NE(fmac1, nullptr) << "Should find fmac1 in loop block";
    ASSERT_NE(fmac2, nullptr) << "Should find fmac2 in loop block";

    int fmac1Pos = getInstructionPosition(loopBlock, fmac1);
    int fmac2Pos = getInstructionPosition(loopBlock, fmac2);

    SWaitCntData* waitBeforeFmac1 = nullptr;
    for (const auto& wait : waitcnts) {
        if (wait.position < fmac1Pos && wait.position >= fmac1Pos - 2) {
            waitBeforeFmac1 = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac1, nullptr)
        << "Should insert waitcnt before first v_add (cross-block dependency)";
    EXPECT_EQ(waitBeforeFmac1->dlcnt, 1)
        << "Should wait for v0,v1,v2 from entry (leave v3) -> dlcnt=1";

    SWaitCntData* waitBeforeFmac2 = nullptr;
    int waitBeforeFmac1Pos = -1;
    for (const auto& wait : waitcnts) {
        if (wait.waitData == waitBeforeFmac1) {
            waitBeforeFmac1Pos = wait.position;
            break;
        }
    }

    for (const auto& wait : waitcnts) {
        if (wait.position < fmac2Pos && wait.position >= fmac2Pos - 2 &&
            wait.position != waitBeforeFmac1Pos) {
            waitBeforeFmac2 = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac2, nullptr)
        << "MUST insert waitcnt before second v_add (v3 still outstanding)";
    EXPECT_EQ(waitBeforeFmac2->dlcnt, 0) << "Should wait for remaining loads (v3) -> dlcnt=0";
}

/**
 * @brief Two-block chain with different load order (v0,v2,v1,v3 in entry).
 *
 * CFG:
 *   entry -> loop_start -> loop_start (self-loop)
 *
 * entry:
 *   ds_load_b32 v0, v10   (pos 0)
 *   ds_load_b32 v2, v10   (pos 1)
 *   ds_load_b32 v1, v10   (pos 2)
 *   ds_load_b32 v3, v10   (pos 3)
 *
 * loop_start:
 *   v_add_f32 v4, v0, v2   (uses v0 at 0, v2 at 1 -> wait up to 1)
 *   v_add_f32 v4, v1, v3   (uses v1, v3 -> wait for remaining)
 *   ds_load_b32 v0, v10
 *   ds_load_b32 v1, v10
 *   ds_load_b32 v2, v10
 *   ds_load_b32 v3, v10
 *
 * Expected:
 *   s_wait_dscnt 1 before first v_add -> dlcnt=1
 *   s_wait_dscnt 0 before second v_add -> dlcnt=0
 */
TEST_F(WaitCntInsertionTest, BasicBlockStateTracking_TwoBlockChain2) {
    std::string irString = R"(
st.func @test_two_block_chain2() {
^entry:
  v0 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v2 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v1 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v3 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  Successors: ^loop_start
^loop_start:
  v4 = "st.v_add_f32"(v0, v2) { issueCycles = 1, latencyCycles = 1 }
  v4 = "st.v_add_f32"(v1, v3) { issueCycles = 1, latencyCycles = 1 }
  v0 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v1 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v2 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v3 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  Successors: ^loop_start
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    BasicBlock& entryBB = *func->begin();
    BasicBlock& loopBlock = *std::next(func->begin());

    EXPECT_FALSE(hasLoopBackEdge(&entryBB)) << "Entry should not have loop back-edge";
    EXPECT_TRUE(hasLoopBackEdge(&loopBlock)) << "Loop block should have back-edge to itself";

    runInsertionPass(*func);

    auto waitcnts = getAllWaitCnts(loopBlock);

    StinkyInstruction* fmac1 = findNthInst(loopBlock, GFX::v_add_f32, 0);
    StinkyInstruction* fmac2 = findNthInst(loopBlock, GFX::v_add_f32, 1);
    ASSERT_NE(fmac1, nullptr) << "Should find fmac1 in loop block";
    ASSERT_NE(fmac2, nullptr) << "Should find fmac2 in loop block";

    int fmac1Pos = getInstructionPosition(loopBlock, fmac1);
    int fmac2Pos = getInstructionPosition(loopBlock, fmac2);

    SWaitCntData* waitBeforeFmac1 = nullptr;
    for (const auto& wait : waitcnts) {
        if (wait.position < fmac1Pos && wait.position >= fmac1Pos - 2) {
            waitBeforeFmac1 = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac1, nullptr)
        << "Should insert waitcnt before first v_add (cross-block dependency)";
    EXPECT_EQ(waitBeforeFmac1->dlcnt, 1)
        << "Should wait for v0,v1,v2 from itself (leave v3) -> dlcnt=1";

    SWaitCntData* waitBeforeFmac2 = nullptr;
    int waitBeforeFmac1Pos = -1;
    for (const auto& wait : waitcnts) {
        if (wait.waitData == waitBeforeFmac1) {
            waitBeforeFmac1Pos = wait.position;
            break;
        }
    }

    for (const auto& wait : waitcnts) {
        if (wait.position < fmac2Pos && wait.position >= fmac2Pos - 2 &&
            wait.position != waitBeforeFmac1Pos) {
            waitBeforeFmac2 = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac2, nullptr)
        << "MUST insert waitcnt before second v_add (v3 still outstanding)";
    EXPECT_EQ(waitBeforeFmac2->dlcnt, 0) << "Should wait for remaining loads (v3) -> dlcnt=0";
}

/**
 * @brief Loop-carried LDS token deps do not force a header tensor wait.
 *
 * CFG:
 *   entry -> loop_header -> loop_body -> loop_tail -> loop_header
 *
 * The tensor load in loop_header defines LDS token 0. When loop_tail flows
 * back to loop_header, that token dependency is loop-carried and should not
 * make the header barrier wait on its own earlier-iteration tensor load.
 */
TEST_F(WaitCntInsertionTest, LoopCarriedMemTokenPredDoesNotForceHeaderTensorWait) {
    std::string irString = R"(
st.func @test_loop_carried_memtoken_header() {
^entry:
  Successors: ^loop_header
^loop_header:
  LDS0 = "st.s_barrier_signal"(-1, LDS0) { issueCycles = 1, latencyCycles = 2, mod.memtoken = { tokens = [0] } }
  LDS0 = "st.s_barrier_wait"(-1, LDS0) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
  LDS0 = "st.tensor_load_to_lds"(s[0:3], s[4:11]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
  Successors: ^loop_body
^loop_body:
  Successors: ^loop_tail
^loop_tail:
  Successors: ^loop_header
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    BasicBlock& loopHeader = *std::next(func->begin());

    runInsertionPass(*func);

    StinkyInstruction* barrierSignal = findNthInst(loopHeader, GFX::s_barrier_signal, 0);
    ASSERT_NE(barrierSignal, nullptr);

    EXPECT_EQ(findTensorWaitCntBefore(loopHeader, barrierSignal), nullptr)
        << "Frozen CK_Tensor state should not force "
           "s_wait_tensorcnt before the loop header barrier";
    EXPECT_EQ(countTensorWaitCnt(loopHeader), 0);
}

TEST_F(WaitCntInsertionTest, LoopCarriedMemTokenPredCanForceHeaderTensorWaitWhenEnabled) {
    std::string irString = R"(
st.func @test_loop_carried_memtoken_header_enabled() {
^entry:
  Successors: ^loop_header
^loop_header:
  LDS0 = "st.s_barrier_signal"(-1, LDS0) { issueCycles = 1, latencyCycles = 2, mod.memtoken = { tokens = [0] } }
  LDS0 = "st.s_barrier_wait"(-1, LDS0) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
  LDS0 = "st.tensor_load_to_lds"(s[0:3], s[4:11]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
  Successors: ^loop_body
^loop_body:
  Successors: ^loop_tail
^loop_tail:
  Successors: ^loop_header
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    BasicBlock& loopHeader = *std::next(func->begin());

    WaitCntInsertionOptions options;
    options.enableLoopCarriedTokenDeps = true;
    runInsertionPass(*func, options);

    StinkyInstruction* barrierSignal = findNthInst(loopHeader, GFX::s_barrier_signal, 0);
    ASSERT_NE(barrierSignal, nullptr);

    SWaitTensorCntData* tensorWait = findTensorWaitCntBefore(loopHeader, barrierSignal);
    ASSERT_NE(tensorWait, nullptr)
        << "Conservative mode should iterate CK_Tensor through loop-carried token state";
    EXPECT_EQ(tensorWait->tlcnt, 0);
    EXPECT_EQ(countTensorWaitCnt(loopHeader), 1);
}

/**
 * @brief Diamond CFG with multi-predecessor merge.
 *
 * CFG:
 *   entry -> b1 (loads v0, v1) -> b3 (uses v0, v1)
 *   entry -> b2 (loads v2, v3, v4) -> b3
 *
 * b3: v_add_f32 v5, v0, v1
 *
 * Multi-path analysis:
 * - Path 1 (b1): [v0, v1] -> fmac uses v0, v1 -> needs dlcnt=0
 * - Path 2 (b2): [v2, v3, v4] -> fmac uses v0, v1 (not present) -> no wait needed
 * - Result: min(0, IGNORE) = 0 -> Optimal for path 1, safe for path 2
 */
TEST_F(WaitCntInsertionTest, BasicBlockStateTracking_MultiPredecessorMerge) {
    std::string irString = R"(
st.func @test_multi_pred() {
^entry:
  Successors: ^b1, ^b2
^b1:
  v0 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v1 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  Successors: ^b3
^b2:
  v2 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v3 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v4 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  Successors: ^b3
^b3:
  v5 = "st.v_add_f32"(v0, v1) { issueCycles = 1, latencyCycles = 1 }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    auto bbIt = func->begin();
    std::advance(bbIt, 3);  // skip entry, b1, b2 -> b3
    BasicBlock& block3 = *bbIt;

    auto waitcnts = getAllWaitCnts(block3);

    StinkyInstruction* fmac = findNthInst(block3, GFX::v_add_f32, 0);
    ASSERT_NE(fmac, nullptr) << "Should find fmac in block3";

    int fmacPos = getInstructionPosition(block3, fmac);

    SWaitCntData* waitBeforeFmac = nullptr;
    for (const auto& wait : waitcnts) {
        if (wait.position < fmacPos && wait.position >= fmacPos - 2) {
            waitBeforeFmac = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac, nullptr)
        << "Should insert waitcnt before fmac (multi-predecessor merge)";
    EXPECT_EQ(waitBeforeFmac->dlcnt, 0)
        << "Should wait for v0,v1 from path 1 (multi-path analysis) -> dlcnt=0";
}

/**
 * @brief Chained diamond CFG demonstrating cross-block dependency propagation.
 *
 * CFG:
 *   entry -> b1 (loads v0, v1, v2) -> b3 (uses v3) -> b4 (uses v0, v1)
 *   entry -> b2 (loads v3, v4, v5) -> b3
 *
 * b3: v_add_f32 v6, v3, v3 (uses v3 from b2 path)
 * b4: v_add_f32 v7, v0, v1 (uses v0, v1 from b1 path, propagated through b3)
 *
 * Block3 analysis:
 * - Path b2: [v3, v4, v5] -> uses v3 -> dlcnt=2 (leaves v4, v5)
 * - Path b1: [v0, v1, v2] -> uses v3 (not present) -> no wait
 * - Result: dlcnt=2
 *
 * Block4 analysis:
 * - b4 receives outstanding loads from b3's predecessor paths
 * - Path via b1: v0, v1 still outstanding -> needs wait
 * - Expected: dlcnt=1
 */
TEST_F(WaitCntInsertionTest, BasicBlockStateTracking_MultiPredecessorMerge2) {
    std::string irString = R"(
st.func @test_multi_pred2() {
^entry:
  Successors: ^b1, ^b2
^b1:
  v0 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v1 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v2 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  Successors: ^b3
^b2:
  v3 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v4 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  v5 = "st.ds_load_b32"(v10) { issueCycles = 1, latencyCycles = 52 }
  Successors: ^b3
^b3:
  v6 = "st.v_add_f32"(v3, v3) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^b4
^b4:
  v7 = "st.v_add_f32"(v0, v1) { issueCycles = 1, latencyCycles = 1 }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    // Navigate to block3 (4th block: entry, b1, b2, b3)
    auto bbIt = func->begin();
    std::advance(bbIt, 3);
    BasicBlock& block3 = *bbIt;

    // Navigate to block4 (5th block)
    std::advance(bbIt, 1);
    BasicBlock& block4 = *bbIt;

    // Verify block3
    auto waitcnts3 = getAllWaitCnts(block3);

    StinkyInstruction* fmac = findNthInst(block3, GFX::v_add_f32, 0);
    ASSERT_NE(fmac, nullptr) << "Should find fmac in block3";

    int fmacPos = getInstructionPosition(block3, fmac);

    SWaitCntData* waitBeforeFmac = nullptr;
    for (const auto& wait : waitcnts3) {
        if (wait.position < fmacPos && wait.position >= fmacPos - 2) {
            waitBeforeFmac = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac, nullptr)
        << "Should insert waitcnt before fmac in block3 (multi-predecessor case)";
    EXPECT_EQ(waitBeforeFmac->dlcnt, 2)
        << "Should wait for v3, v3 from path 2 (multi-path analysis) -> dlcnt=2";

    // Verify block4
    auto waitcnts4 = getAllWaitCnts(block4);

    StinkyInstruction* fmac2 = findNthInst(block4, GFX::v_add_f32, 0);
    ASSERT_NE(fmac2, nullptr) << "Should find fmac2 in block4";

    int fmac2Pos = getInstructionPosition(block4, fmac2);

    SWaitCntData* waitBeforeFmac2 = nullptr;
    for (const auto& wait : waitcnts4) {
        if (wait.position < fmac2Pos && wait.position >= fmac2Pos - 2) {
            waitBeforeFmac2 = wait.waitData;
            break;
        }
    }

    ASSERT_NE(waitBeforeFmac2, nullptr)
        << "Should insert waitcnt before fmac2 in block4 (uses v0,v1)";
    EXPECT_EQ(waitBeforeFmac2->dlcnt, 1)
        << "Per-path analysis: Path1 needs dlcnt=1, Path2 needs none -> min=1";
}

// ============================================================================
// Test Suite 5: LDS Write-After-Read Hazard
//
// Tests that an LDS writer (tensor_load_to_lds / ds_store) carrying
// MemTokenData triggers a synthesized WAR dependency against any prior
// DS read whose tokens overlap. The SSA def-use chain does not encode
// these hazards; the pass detects them via the in-flight DS-read queue
// and the writer's MemTokenData modifier.
//
// collectLdsWarDependencies applies a per-pair same-pipeline filter
// (isOnSameDSPipeline): if the writer and a candidate reader share the same
// hardware memory pipeline, the pair is dropped because FIFO retirement on
// the shared counter already orders them. In practice this means a ds_store
// writer skips prior ds_load / ds_atomic readers (both on dlcnt), while
// tensor_load_to_lds (on tlcnt) still synthesizes WAR against DS readers.
// ============================================================================

/**
 * @brief Token-overlapping ds_loads followed by tensor_load_to_lds -> drain all.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10  tokens=[0]    (read 0)
 *   v[2:3] = ds_load_b64 v10  tokens=[0]    (read 1)
 *   v[4:5] = ds_load_b64 v10  tokens=[0]    (read 2)
 *   v[6:7] = ds_load_b64 v10  tokens=[0]    (read 3)
 *   tensor_load_to_lds        tokens=[0]    (LDS writer)
 *
 * All four reads overlap with the writer's token. tensor_load_to_lds is not
 * a DS op (it lives on tlcnt), so the queue at the writer is exactly the
 * four reads. The youngest conflicting read sits at position 1 from end, so
 * the algorithm picks min(count - 1) = 0 and emits s_wait_dscnt 0.
 */
TEST_F(WaitCntInsertionTest, LdsWarBeforeTensorLoad) {
    std::string irString = R"(
st.func @test_lds_war_tensor_load() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [0] } }
  v[2:3] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 16, gds = false }, mod.memtoken = { tokens = [0] } }
  v[4:5] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 32, gds = false }, mod.memtoken = { tokens = [0] } }
  v[6:7] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 48, gds = false }, mod.memtoken = { tokens = [0] } }
  "st.tensor_load_to_lds"(s[0:3], s[10:17]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    StinkyInstruction* tensorLoad = findNthInst(entryBB, GFX::tensor_load_to_lds, 0);
    ASSERT_NE(tensorLoad, nullptr);

    SWaitCntData* waitBeforeWriter = findWaitCntBefore(entryBB, tensorLoad);
    ASSERT_NE(waitBeforeWriter, nullptr)
        << "Pass must insert s_wait_dscnt before tensor_load_to_lds (WAR-on-LDS)";
    EXPECT_EQ(waitBeforeWriter->dlcnt, 0)
        << "All four token-0 ds_loads must drain before the LDS writer -> dlcnt=0";
    EXPECT_EQ(waitBeforeWriter->vlcnt, -1);
    EXPECT_EQ(waitBeforeWriter->vscnt, -1);
    EXPECT_EQ(waitBeforeWriter->dscnt, -1);
    EXPECT_EQ(waitBeforeWriter->kmcnt, -1);

    EXPECT_EQ(countWaitCnt(entryBB), 1)
        << "Exactly one waitcnt should be inserted (before tensor_load_to_lds)";
    EXPECT_EQ(countTensorWaitCnt(entryBB), 0);
}

/**
 * @brief Mixed-token ds_loads -> only token-overlapping reads form the WAR set.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10  tokens=[0]    (read 0, conflict)
 *   v[2:3] = ds_load_b64 v10  tokens=[0]    (read 1, conflict)
 *   v[4:5] = ds_load_b64 v10  tokens=[1]    (read 2, NOT a conflict)
 *   v[6:7] = ds_load_b64 v10  tokens=[1]    (read 3, NOT a conflict)
 *   tensor_load_to_lds        tokens=[0]    (LDS writer)
 *
 * Queue at writer: [r0, r1, r2, r3]. WAR set picks {r0, r1} only.
 * pendingDSCountFrom(r1) = 3, so wait = 3 - 1 = 2: drain r0, r1; leave r2, r3.
 */
TEST_F(WaitCntInsertionTest, LdsWarMixedTokens) {
    std::string irString = R"(
st.func @test_lds_war_mixed_tokens() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [0] } }
  v[2:3] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 16, gds = false }, mod.memtoken = { tokens = [0] } }
  v[4:5] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 32, gds = false }, mod.memtoken = { tokens = [1] } }
  v[6:7] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 48, gds = false }, mod.memtoken = { tokens = [1] } }
  "st.tensor_load_to_lds"(s[0:3], s[10:17]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    StinkyInstruction* tensorLoad = findNthInst(entryBB, GFX::tensor_load_to_lds, 0);
    ASSERT_NE(tensorLoad, nullptr);

    SWaitCntData* waitBeforeWriter = findWaitCntBefore(entryBB, tensorLoad);
    ASSERT_NE(waitBeforeWriter, nullptr)
        << "Pass must insert s_wait_dscnt before tensor_load_to_lds";
    EXPECT_EQ(waitBeforeWriter->dlcnt, 2)
        << "Drain 2 token-0 reads, leave 2 token-1 reads outstanding -> dlcnt=2";
    EXPECT_EQ(waitBeforeWriter->vlcnt, -1);
    EXPECT_EQ(waitBeforeWriter->vscnt, -1);
    EXPECT_EQ(waitBeforeWriter->dscnt, -1);
    EXPECT_EQ(waitBeforeWriter->kmcnt, -1);

    EXPECT_EQ(countWaitCnt(entryBB), 1);
    EXPECT_EQ(countTensorWaitCnt(entryBB), 0);
}

/**
 * @brief Disjoint tokens -> no WAR wait inserted.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10  tokens=[1]
 *   v[2:3] = ds_load_b64 v10  tokens=[1]
 *   tensor_load_to_lds        tokens=[0]
 *
 * No reads share a token with the writer, so the WAR set is empty and
 * memOpDependencies stays empty. The pass should not insert any waitcnt.
 */
TEST_F(WaitCntInsertionTest, LdsWarDisjointTokens) {
    std::string irString = R"(
st.func @test_lds_war_disjoint_tokens() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [1] } }
  v[2:3] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 16, gds = false }, mod.memtoken = { tokens = [1] } }
  "st.tensor_load_to_lds"(s[0:3], s[10:17]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    EXPECT_EQ(countWaitCnt(entryBB), 0)
        << "No token overlap with the writer -> no WAR wait should be emitted";
    EXPECT_EQ(countTensorWaitCnt(entryBB), 0);
}

/**
 * @brief Same-pipeline filter: ds_store with token-overlapping prior reads
 *        emits NO synthetic WAR wait.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10           tokens=[0]
 *   v[2:3] = ds_load_b64 v10           tokens=[0]
 *   v[4:5] = ds_load_b64 v10           tokens=[0]
 *   ds_store_b64 v100, v[20:21]        tokens=[0]    (LDS writer, same DS pipe)
 *
 * collectLdsWarDependencies runs the per-pair isOnSameDSPipeline filter:
 * writer (ds_store) and each candidate reader (ds_load) are both DS memory
 * ops, so every (writer, reader) pair is dropped from the WAR set. The
 * hardware retires them FIFO on dlcnt, so no synthetic s_wait_dscnt is
 * required. The data operand v[20:21] is independent of the prior reads,
 * so there is also no RAW dependency to emit.
 */
TEST_F(WaitCntInsertionTest, DsStoreWithPriorDsLoadsNoExplicitWar) {
    std::string irString = R"(
st.func @test_ds_store_no_war() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [0] } }
  v[2:3] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 16, gds = false }, mod.memtoken = { tokens = [0] } }
  v[4:5] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 32, gds = false }, mod.memtoken = { tokens = [0] } }
  "st.ds_store_b64"(v100, v[20:21]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [0] } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    StinkyInstruction* dsStore = findNthInst(entryBB, GFX::ds_store_b64, 0);
    ASSERT_NE(dsStore, nullptr);

    SWaitCntData* waitBeforeWriter = findWaitCntBefore(entryBB, dsStore);
    EXPECT_EQ(waitBeforeWriter, nullptr)
        << "isOnSameDSPipeline filters every (ds_store, ds_load) pair out of the "
           "WAR set; hardware FIFO on dlcnt already enforces ordering, so no "
           "synthetic s_wait_dscnt should be emitted";

    EXPECT_EQ(countWaitCnt(entryBB), 0);
    EXPECT_EQ(countTensorWaitCnt(entryBB), 0);
}

/**
 * @brief ds_store with a regular RAW DS dep -> wait drains the producer.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10
 *   ds_store_b64 v100, v[0:1]   (data sourced from the prior ds_load via def-use)
 *
 * Locks in the side-benefit of deferring DS recording: ds_store is itself a DS
 * op, but its wait should still drain the prior ds_load that produced v[0:1].
 * With the recording deferral, pendingDSCountFrom(ds_load) = 1 at wait time, so
 * the algorithm emits s_wait_dscnt 0 (was 1 before the fix).
 */
TEST_F(WaitCntInsertionTest, DsStoreWithRawDsLoadDep) {
    std::string irString = R"(
st.func @test_ds_store_raw_ds_load() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false } }
  "st.ds_store_b64"(v100, v[0:1]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 0, gds = false } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    StinkyInstruction* dsStore = findNthInst(entryBB, GFX::ds_store_b64, 0);
    ASSERT_NE(dsStore, nullptr);

    SWaitCntData* waitBeforeWriter = findWaitCntBefore(entryBB, dsStore);
    ASSERT_NE(waitBeforeWriter, nullptr)
        << "Pass must insert s_wait_dscnt before ds_store that consumes ds_load data";
    EXPECT_EQ(waitBeforeWriter->dlcnt, 0)
        << "Recording deferral: queue at wait time = [ds_load]; count-1 = 0";
    EXPECT_EQ(waitBeforeWriter->vlcnt, -1);
    EXPECT_EQ(waitBeforeWriter->vscnt, -1);
    EXPECT_EQ(waitBeforeWriter->dscnt, -1);
    EXPECT_EQ(waitBeforeWriter->kmcnt, -1);

    EXPECT_EQ(countWaitCnt(entryBB), 1);
}

// ============================================================================
// Test Suite 6: Conservative Fallback for Missing MemTokenData
//
// The pass uses MemTokenData for three LDS-related decisions: barrier-vs-DS
// conflict, WAR-on-LDS synthesis, and tensor-load/barrier matching. When any
// specific op is missing MemTokenData, a hybrid conservative policy fires:
//
//   * Anchor missing tokens (writer / barrier) -> full drain on the relevant
//     counter (s_wait_dscnt 0 or s_wait_tensorcnt 0).
//   * Candidate missing tokens (reader / pending DS op / pending tensor load)
//     -> widen the dep set; normal min(count-1) algorithm picks the value.
//
// These tests exercise each branch in isolation.
// ============================================================================

/**
 * @brief LDS writer (tensor_load_to_lds) without MemTokenData triggers
 *        s_wait_dscnt 0 before issue.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10   tokens=[0]    (read 0)
 *   v[2:3] = ds_load_b64 v10   tokens=[0]    (read 1)
 *   tensor_load_to_lds                        (LDS writer, NO MemTokenData)
 *
 * Writer-anchor-missing-tokens branch of collectLdsWarDependencies:
 * forceDsDrain becomes true because there are non-same-pipeline DS read
 * candidates pending; computeRequiredWaits emits s_wait_dscnt 0 and clears
 * the DS state before the tensor_load_to_lds issues.
 */
TEST_F(WaitCntInsertionTest, ConservativeFallback_WarOnLds_WriterMissingTokens_ForcesDscntZero) {
    std::string irString = R"(
st.func @test_war_writer_missing_tokens() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [0] } }
  v[2:3] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 16, gds = false }, mod.memtoken = { tokens = [0] } }
  "st.tensor_load_to_lds"(s[0:3], s[10:17]) { issueCycles = 1, latencyCycles = 1 }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    StinkyInstruction* tensorLoad = findNthInst(entryBB, GFX::tensor_load_to_lds, 0);
    ASSERT_NE(tensorLoad, nullptr);

    SWaitCntData* waitBeforeWriter = findWaitCntBefore(entryBB, tensorLoad);
    ASSERT_NE(waitBeforeWriter, nullptr)
        << "Writer without MemTokenData must conservatively drain DS counter";
    EXPECT_EQ(waitBeforeWriter->dlcnt, 0)
        << "Conservative drain emits s_wait_dscnt 0 because token disjointness "
           "cannot be proven";
    EXPECT_EQ(waitBeforeWriter->vlcnt, -1);
    EXPECT_EQ(waitBeforeWriter->vscnt, -1);
    EXPECT_EQ(waitBeforeWriter->dscnt, -1);
    EXPECT_EQ(waitBeforeWriter->kmcnt, -1);

    EXPECT_EQ(countWaitCnt(entryBB), 1);
    EXPECT_EQ(countTensorWaitCnt(entryBB), 0);
}

/**
 * @brief Same-pipeline filter still applies when the writer is a ds_store
 *        without MemTokenData: no conservative wait is emitted because the
 *        hardware FIFO already orders the pair.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10   tokens=[0]
 *   v[2:3] = ds_load_b64 v10   tokens=[0]
 *   ds_store_b64 v100, v[20:21]                 (LDS writer, NO MemTokenData)
 *
 * collectLdsWarDependencies: writerLacksTokens is true, but the candidate
 * isOnSameDSPipeline check filters every ds_load out. anyCandidate stays
 * false, forceDsDrain stays false, no conservative wait is emitted. The
 * data operand v[20:21] is independent of the prior reads, so the SSA path
 * also yields no wait.
 */
TEST_F(WaitCntInsertionTest,
       ConservativeFallback_WarOnLds_DsStoreMissingTokens_SamePipelineSkipsDrain) {
    std::string irString = R"(
st.func @test_war_ds_store_missing_tokens_same_pipeline() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [0] } }
  v[2:3] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 16, gds = false }, mod.memtoken = { tokens = [0] } }
  "st.ds_store_b64"(v100, v[20:21]) { issueCycles = 1, latencyCycles = 1, mod.ds = { na = 1, offset = 0, gds = false } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    StinkyInstruction* dsStore = findNthInst(entryBB, GFX::ds_store_b64, 0);
    ASSERT_NE(dsStore, nullptr);

    SWaitCntData* waitBeforeWriter = findWaitCntBefore(entryBB, dsStore);
    EXPECT_EQ(waitBeforeWriter, nullptr)
        << "Same-pipeline (ds_store + ds_load) pairs are FIFO-ordered by hardware; "
           "no conservative drain should fire even when the writer lacks tokens";

    EXPECT_EQ(countWaitCnt(entryBB), 0);
}

/**
 * @brief Candidate-missing-tokens widening: writer carries MemTokenData but
 *        a prior ds_load does not. The reader is treated as conflicting and
 *        the normal min(count-1) algorithm computes the wait value.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10   tokens=[0]      (read 0)
 *   v[2:3] = ds_load_b64 v10   (NO MemTokenData) (read 1)
 *   v[4:5] = ds_load_b64 v10   tokens=[0]      (read 2)
 *   tensor_load_to_lds         tokens=[0]      (LDS writer)
 *
 * Queue at writer: [r0, r1, r2]. r0 overlaps; r1 lacks tokens so is widened
 * in; r2 overlaps. All three become conflicting. The youngest is r2 at
 * position 1 from end -> wait = 1 - 1 = 0. Emit s_wait_dscnt 0.
 */
TEST_F(WaitCntInsertionTest, ConservativeFallback_WarOnLds_CandidateMissingTokens_WidensDepSet) {
    std::string irString = R"(
st.func @test_war_candidate_missing_tokens() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [0] } }
  v[2:3] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 16, gds = false } }
  v[4:5] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 32, gds = false }, mod.memtoken = { tokens = [0] } }
  "st.tensor_load_to_lds"(s[0:3], s[10:17]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    StinkyInstruction* tensorLoad = findNthInst(entryBB, GFX::tensor_load_to_lds, 0);
    ASSERT_NE(tensorLoad, nullptr);

    SWaitCntData* waitBeforeWriter = findWaitCntBefore(entryBB, tensorLoad);
    ASSERT_NE(waitBeforeWriter, nullptr)
        << "Untagged reader must be conservatively widened into the WAR set";
    EXPECT_EQ(waitBeforeWriter->dlcnt, 0)
        << "Youngest conflicting read is r2 at position 1; wait = 1 - 1 = 0";

    EXPECT_EQ(countWaitCnt(entryBB), 1);
}

/**
 * @brief Barrier path conservative fallback: an untagged pending DS op forces
 *        a drain even when the barrier itself carries MemTokenData and no
 *        token overlap exists with activeDSTokens.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10   (NO MemTokenData)   (untagged candidate)
 *   s_barrier                   tokens=[42]
 *
 * PendingMemOpTracker::hasUntaggedDSOp() returns true at the barrier;
 * needsDrain becomes true; emit s_wait_dscnt 0.
 */
TEST_F(WaitCntInsertionTest, ConservativeFallback_Barrier_UntaggedPendingDSOp_ForcesDscntZero) {
    std::string irString = R"(
st.func @test_barrier_candidate_missing_tokens() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false } }
  "st.s_barrier"() { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [42] } }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    StinkyInstruction* barrier = findNthInst(entryBB, GFX::s_barrier, 0);
    ASSERT_NE(barrier, nullptr);

    SWaitCntData* waitBeforeBarrier = findWaitCntBefore(entryBB, barrier);
    ASSERT_NE(waitBeforeBarrier, nullptr)
        << "Untagged pending DS op forces a conservative drain on a tagged barrier";
    EXPECT_EQ(waitBeforeBarrier->dlcnt, 0);

    EXPECT_EQ(countWaitCnt(entryBB), 1);
}

/**
 * @brief Barrier path conservative fallback: untagged barrier with a tagged
 *        pending DS op also forces a drain.
 *
 * IR (single block):
 *   v[0:1] = ds_load_b64 v10   tokens=[7]
 *   s_barrier                   (NO MemTokenData)
 *
 * barrierLacksTokens is true; needsDrain becomes true because pendingDSOps
 * is non-empty; emit s_wait_dscnt 0.
 */
TEST_F(WaitCntInsertionTest, ConservativeFallback_Barrier_AnchorMissingTokens_ForcesDscntZero) {
    std::string irString = R"(
st.func @test_barrier_anchor_missing_tokens() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [7] } }
  "st.s_barrier"() { issueCycles = 1, latencyCycles = 1 }
}
)";

    StinkyIRConverter converter(getArch());
    auto* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);

    runInsertionPass(*func);

    BasicBlock& entryBB = *func->begin();
    StinkyInstruction* barrier = findNthInst(entryBB, GFX::s_barrier, 0);
    ASSERT_NE(barrier, nullptr);

    SWaitCntData* waitBeforeBarrier = findWaitCntBefore(entryBB, barrier);
    ASSERT_NE(waitBeforeBarrier, nullptr)
        << "Untagged barrier with pending DS op must conservatively drain";
    EXPECT_EQ(waitBeforeBarrier->dlcnt, 0);

    EXPECT_EQ(countWaitCnt(entryBB), 1);
}
