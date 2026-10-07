// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Which single-instruction moves the co-issue repair may make: only SALU/VALU fillers move,
// never across a barrier, a branch, a label, a block or IR-node boundary, a register
// dependence (SCC included), or a memory wait guarding a register they touch.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "transforms/asm/coissue/IssueTimeline.hpp"
#include "transforms/asm/coissue/MoveChecker.hpp"

using namespace stinkytofu;
using namespace stinkytofu::coissue;

namespace {

constexpr std::array<int, 3> kArch{12, 5, 0};

// Positions of the instructions in kBlock.
enum : size_t {
    kWmma,
    kMov1,
    kAdd,
    kAddc,
    kLoad,
    kWait,
    kUseLoad,
    kValu,
    kBarrier,
    kMov4,
    kBranch,
    kCount
};

const char* kBlock = R"(
    v[0:7] = "st.v_wmma_scale_f32_16x16x128_f8f6f4"(v[100:115], v[116:131], v[0:7], v20, v21) { mod.matrix_fmt = { fmtA = "MATRIX_FMT_FP8", fmtB = "MATRIX_FMT_FP8" } }
    s1 = "st.s_mov_b32"(1)
    s2, SCC0 = "st.s_add_u32"(s2, s8)
    s3, SCC0 = "st.s_addc_u32"(s3, 0, SCC0)
    v[60:63] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 0, gds = false } }
    "st.s_wait_dscnt"(0)
    v30 = "st.v_add_nc_u32"(v60, v31)
    v32 = "st.v_add_nc_u32"(v33, v34)
    "st.s_barrier_wait"(-1)
    s4 = "st.s_mov_b32"(4)
    "st.s_cbranch_scc0"(label_x, SCC0)
)";

class MoveCheckerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        Function* func = converter.convertToFunction(kBlock);
        ASSERT_NE(func, nullptr);
        for (BasicBlock& bb : *func)
            for (IRBase& node : bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node)) order.push_back(inst);
        ASSERT_EQ(order.size(), kCount);
        blockOf.assign(order.size(), 0);
        slotOf.assign(order.size(), 0);
    }

    StinkyIRConverter converter{kArch};
    TimedInstCache cache{hwModelForArch(kArch)};
    std::vector<StinkyInstruction*> order;
    std::vector<int> blockOf;
    std::vector<int> slotOf;
};

TEST_F(MoveCheckerTest, IndependentFillersMoveFreely) {
    MoveChecker checker(order, blockOf, slotOf, cache);
    // s_mov s1 crosses the SCC pair, the load, the wait and both VALUs.
    EXPECT_TRUE(checker.legal(order, kMov1, kBarrier));
    // A VALU that touches no loaded register crosses the wait and the load.
    EXPECT_TRUE(checker.legal(order, kValu, kLoad));
    EXPECT_TRUE(checker.legal(order, kValu, kMov1));
    // Moving in front of the next position, or onto itself, is no move.
    EXPECT_FALSE(checker.legal(order, kMov1, kMov1));
    EXPECT_FALSE(checker.legal(order, kMov1, kMov1 + 1));
}

TEST_F(MoveCheckerTest, OnlyFillersMove) {
    MoveChecker checker(order, blockOf, slotOf, cache);
    EXPECT_FALSE(checker.legal(order, kWmma, kAdd));
    EXPECT_FALSE(checker.legal(order, kLoad, kMov1));
    EXPECT_FALSE(checker.legal(order, kWait, kMov1));
    EXPECT_FALSE(checker.legal(order, kBarrier, kMov1));
    EXPECT_FALSE(checker.legal(order, kBranch, kMov4));
}

TEST_F(MoveCheckerTest, NoCrossingBarrierOrBranch) {
    MoveChecker checker(order, blockOf, slotOf, cache);
    EXPECT_FALSE(checker.legal(order, kMov1, kMov4));
    EXPECT_FALSE(checker.legal(order, kMov4, kValu));
    EXPECT_FALSE(checker.legal(order, kMov4, order.size()));
}

TEST_F(MoveCheckerTest, SccDependenceHolds) {
    MoveChecker checker(order, blockOf, slotOf, cache);
    // s_addc reads the SCC s_add writes: neither goes past the other.
    EXPECT_FALSE(checker.legal(order, kAdd, kLoad));
    EXPECT_FALSE(checker.legal(order, kAddc, kAdd));
    // Away from each other's SCC they move.
    EXPECT_TRUE(checker.legal(order, kAdd, kMov1));
    EXPECT_TRUE(checker.legal(order, kAddc, kWait));
}

TEST_F(MoveCheckerTest, LoadedRegisterStaysBehindItsWait) {
    MoveChecker checker(order, blockOf, slotOf, cache);
    // v_add reads v60, which the load writes: it stays behind the wait (and the load).
    EXPECT_FALSE(checker.legal(order, kUseLoad, kWait));
    EXPECT_FALSE(checker.legal(order, kUseLoad, kLoad));
    EXPECT_TRUE(checker.legal(order, kUseLoad, kBarrier));
}

TEST_F(MoveCheckerTest, PinnedInstructionStays) {
    MoveChecker checker(order, blockOf, slotOf, cache);
    checker.pin(order[kValu]);
    EXPECT_FALSE(checker.legal(order, kValu, kLoad));
    EXPECT_TRUE(checker.legal(order, kMov1, kBarrier));
}

TEST_F(MoveCheckerTest, NoCrossingBlockOrIrNode) {
    // A second block from the load on: s_mov s1 can no longer reach the VALUs.
    std::vector<int> twoBlocks = blockOf;
    for (size_t k = kLoad; k < order.size(); ++k) twoBlocks[k] = 1;
    MoveChecker blocks(order, twoBlocks, slotOf, cache);
    EXPECT_FALSE(blocks.legal(order, kMov1, kBarrier));
    EXPECT_TRUE(blocks.legal(order, kMov1, kLoad));

    // A non-instruction IR node in front of the load: the same.
    std::vector<int> slots = slotOf;
    for (size_t k = kLoad; k < order.size(); ++k) slots[k] = 1;
    MoveChecker nodes(order, blockOf, slots, cache);
    EXPECT_FALSE(nodes.legal(order, kMov1, kBarrier));
    EXPECT_TRUE(nodes.legal(order, kMov1, kLoad));
}

}  // namespace
