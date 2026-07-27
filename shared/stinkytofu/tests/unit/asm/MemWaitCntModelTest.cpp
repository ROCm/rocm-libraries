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
//
// Unit tests for MemWaitCntModel — the model that reserves an issue cycle for the
// s_wait_<c>cnt instructions the wait-count pass inserts after scheduling. The
// key-level core is exercised directly (no IR needed); the instruction-level API is
// exercised with real ds_load / wmma / tensor_load ops built via AsmIRBuilder.

#include <gtest/gtest.h>

#include "TestHelpers.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "transforms/asm/dag/MemWaitCntModel.hpp"

using namespace stinkytofu;
using stinkytofu::test::setFunctionArch;
using waitcnt::CK_DS;
using waitcnt::CK_Tensor;

namespace {

// Key for a v-register index, matching MemWaitCntModel::regKey.
int vkey(uint32_t idx) {
    return MemWaitCntModel::regKey(RegType::V, idx);
}

}  // namespace

// ---------------------------------------------------------------------------
// Key-level core: FIFO in-order drain semantics.
// ---------------------------------------------------------------------------

// ds v0; ds v1; consume v0; consume v1 -> each consumer needs its own wait (2 total).
TEST(MemWaitCntModel, InOrderConsumeNeedsTwoWaits) {
    MemWaitCntModel m;  // reserved = {CK_DS}
    m.recordProducer(CK_DS, {vkey(0)});
    m.recordProducer(CK_DS, {vkey(1)});

    EXPECT_EQ(m.applyWaitForSrcKeys({vkey(0)}), 1);  // wait on v0 (v1 still in flight)
    EXPECT_EQ(m.applyWaitForSrcKeys({vkey(1)}), 1);  // wait on v1
}

// ds v0; ds v1; consume v1; consume v0 -> waiting on the newer v1 drains the older v0
// too (single in-order counter), so the second consumer needs no wait (1 total).
TEST(MemWaitCntModel, ReverseConsumeCoalescesToOneWait) {
    MemWaitCntModel m;
    m.recordProducer(CK_DS, {vkey(0)});
    m.recordProducer(CK_DS, {vkey(1)});

    EXPECT_EQ(m.applyWaitForSrcKeys({vkey(1)}), 1);  // drains v1 AND older v0
    EXPECT_EQ(m.applyWaitForSrcKeys({vkey(0)}), 0);  // already drained
}

// A consumer that reads no outstanding memory reg reserves nothing.
TEST(MemWaitCntModel, NoOutstandingReadNoWait) {
    MemWaitCntModel m;
    m.recordProducer(CK_DS, {vkey(0)});
    EXPECT_EQ(m.applyWaitForSrcKeys({vkey(9)}), 0);
    EXPECT_FALSE(m.isWaitNeededForSrcKeys({vkey(9)}));
    EXPECT_TRUE(m.isWaitNeededForSrcKeys({vkey(0)}));
}

// One wait per consumer regardless of how many outstanding ds regs it reads.
TEST(MemWaitCntModel, MultipleSrcsChargeOneCycle) {
    MemWaitCntModel m;
    m.recordProducer(CK_DS, {vkey(0)});
    m.recordProducer(CK_DS, {vkey(1)});
    EXPECT_EQ(m.applyWaitForSrcKeys({vkey(0), vkey(1)}), 1);
}

// reset() clears all brackets.
TEST(MemWaitCntModel, ResetClearsState) {
    MemWaitCntModel m;
    m.recordProducer(CK_DS, {vkey(0)});
    m.reset();
    EXPECT_EQ(m.applyWaitForSrcKeys({vkey(0)}), 0);
    EXPECT_EQ(m.outstanding(CK_DS), 0);
}

// ---------------------------------------------------------------------------
// Extensibility contract: a producer on a non-reserved counter is recorded (so a
// future opt-in can use it) but charges no cycle until that counter is reserved.
// ---------------------------------------------------------------------------
TEST(MemWaitCntModel, NonReservedCounterRecordsButDoesNotCharge) {
    MemWaitCntModel dsOnly;  // reserved = {CK_DS}
    dsOnly.recordProducer(CK_Tensor, {vkey(0)});
    EXPECT_EQ(dsOnly.outstanding(CK_Tensor), 1);          // recorded
    EXPECT_EQ(dsOnly.applyWaitForSrcKeys({vkey(0)}), 0);  // but no reserved cycle

    MemWaitCntModel withTensor({CK_DS, CK_Tensor});
    withTensor.recordProducer(CK_Tensor, {vkey(0)});
    EXPECT_EQ(withTensor.applyWaitForSrcKeys({vkey(0)}), 1);  // now charges
}

// Distinct reserved counters each contribute a cycle for the same consumer.
TEST(MemWaitCntModel, DistinctReservedCountersEachCharge) {
    MemWaitCntModel m({CK_DS, CK_Tensor});
    m.recordProducer(CK_DS, {vkey(0)});
    m.recordProducer(CK_Tensor, {vkey(1)});
    EXPECT_EQ(m.applyWaitForSrcKeys({vkey(0), vkey(1)}), 2);
}

// ---------------------------------------------------------------------------
// Instruction-level API: classifyMemOp routing via real ds_load / wmma / tensor ops.
// ---------------------------------------------------------------------------
namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

StinkyInstruction* makeDsLoad(BasicBlock* bb, int destV) {
    AsmIRBuilder b(*bb, kArch);
    StinkyInstruction* i = b.create(getMCIDByUOp(GFX::ds_load_b128, kArch));
    i->addDestReg(StinkyRegister("v", destV, 4));
    i->addSrcReg(StinkyRegister("v", 100, 1));  // address
    return i;
}

StinkyInstruction* makeWmma(BasicBlock* bb, int srcV) {
    AsmIRBuilder b(*bb, kArch);
    StinkyInstruction* i = b.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, kArch));
    i->addDestReg(StinkyRegister("v", 200, 8));
    i->addSrcReg(StinkyRegister("v", srcV, 4));
    return i;
}

}  // namespace

// End-to-end through the instruction API: a WMMA consuming a ds_load's dest reserves
// exactly one dscnt cycle; a WMMA reading unrelated regs reserves none.
TEST(MemWaitCntModel, InstructionApiDsReadToWmma) {
    Function fn("test");
    setFunctionArch(fn, kArch);
    BasicBlock* bb = fn.createBasicBlock("entry");

    StinkyInstruction* ds0 = makeDsLoad(bb, 0);
    StinkyInstruction* wmmaConsume = makeWmma(bb, 0);     // reads v0..v3 (from ds0)
    StinkyInstruction* wmmaUnrelated = makeWmma(bb, 50);  // reads v50.. (nothing pending)

    MemWaitCntModel m;
    m.addProducer(*ds0);
    EXPECT_TRUE(m.isWaitNeeded(*wmmaConsume));
    EXPECT_EQ(m.applyWait(*wmmaConsume), 1);
    EXPECT_FALSE(m.isWaitNeeded(*wmmaUnrelated));
    EXPECT_EQ(m.applyWait(*wmmaUnrelated), 0);
}

// tensor_load routes to CK_Tensor (its own counter), so with the default {CK_DS}
// reserved set it records but charges nothing — confirming counter separation.
TEST(MemWaitCntModel, InstructionApiTensorLoadNotDsByDefault) {
    Function fn("test");
    setFunctionArch(fn, kArch);
    BasicBlock* bb = fn.createBasicBlock("entry");

    AsmIRBuilder b(*bb, kArch);
    StinkyInstruction* tl = b.create(getMCIDByUOp(GFX::tensor_load_to_lds, kArch));
    tl->addDestReg(StinkyRegister("v", 0, 4));

    StinkyInstruction* consumer = makeWmma(bb, 0);

    MemWaitCntModel m;  // {CK_DS}
    m.addProducer(*tl);
    EXPECT_EQ(m.outstanding(CK_Tensor), 1);
    EXPECT_EQ(m.applyWait(*consumer), 0);
}
