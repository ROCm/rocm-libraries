// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

using namespace stinkytofu;

namespace {

constexpr const char* kWmma =
    R"("st.v_wmma_f32_16x16x32_bf16"(v[200:207], v[208:215], v[0:7]) { mod.mfma = { reuseA = false, reuseB = false } })";
constexpr const char* kDsAttrs =
    R"({ issueCycles = 1, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false } })";

class WaitAluTrackerTest : public ::testing::Test {
   protected:
    std::array<int, 3> arch{12, 5, 0};
    StinkyIRConverter converter{arch};
    PassContext passCtx;

    void SetUp() override {
        GemmTileConfig config;
        config.arch = arch;
        passCtx.setGemmTileConfig(config);
    }

    std::vector<StinkyInstruction*> parse(const std::string& body) {
        Function* func = converter.convertToFunction("st.func @t() {\n^entry:\n" + body + "}\n");
        std::vector<StinkyInstruction*> insts;
        if (!func) return insts;
        for (BasicBlock& bb : *func)
            for (IRBase& ir : bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) insts.push_back(inst);
        return insts;
    }

    WaitAluTracker makeTracker() {
        return WaitAluTracker(passCtx, gfx1250InsertWaitAluOptions(true));
    }

    static std::string wmmas(int n) {
        std::string s;
        for (int i = 0; i < n; ++i)
            s += "  v[" + std::to_string(16 + 8 * i) + ":" + std::to_string(23 + 8 * i) +
                 "] = " + kWmma + "\n";
        return s;
    }

    // Commit `followers` one by one: untilFree drops by one per op and query() clears the
    // counter exactly when it reaches 0.
    static void expectCountsDown(WaitAluTracker& t, const StinkyInstruction& consumer,
                                 WaitAluHideOp op, const std::vector<StinkyInstruction*>& followers,
                                 bool vm) {
        auto waits = [&] {
            const WaitAluNeed n = t.query(consumer);
            return (vm ? n.vmVsrc : n.vaVdst) >= 0;
        };
        int left = t.untilFree(consumer, op);
        ASSERT_GT(left, 0);
        ASSERT_LE(static_cast<size_t>(left), followers.size());
        for (size_t i = 0; left > 0; ++i, --left) {
            EXPECT_TRUE(waits());
            t.commit(*followers[i]);
            EXPECT_EQ(t.untilFree(consumer, op), left - 1);
        }
        EXPECT_FALSE(waits());
    }
};

// untilFree counts down by one per WMMA and reaches 0 exactly when query() drops the va_vdst.
TEST_F(WaitAluTrackerTest, ValuToDsCountsDownPerWmma) {
    const std::vector<StinkyInstruction*> insts =
        parse("  v[0:7] = " + std::string(kWmma) + "\n" +
              "  v250 = \"st.v_mov_b32\"(v251) { issueCycles = 1, latencyCycles = 1 }\n" +
              wmmas(16) + "  v252 = \"st.ds_load_b32\"(v250) " + kDsAttrs + "\n");
    ASSERT_EQ(insts.size(), 19u);
    const StinkyInstruction& ds = *insts.back();
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    t.commit(*insts[1]);
    int left = t.untilFree(ds, WaitAluHideOp::Wmma);
    ASSERT_GT(left, 0);
    EXPECT_GE(t.query(ds).vaVdst, 0);
    EXPECT_EQ(t.requiredGap(*insts[1], ds, WaitAluHideOp::Wmma), left);
    for (size_t i = 2; left > 0; ++i, --left) {
        ASSERT_LT(i, insts.size() - 1);
        EXPECT_GE(t.query(ds).vaVdst, 0);
        t.commit(*insts[i]);
        EXPECT_EQ(t.untilFree(ds, WaitAluHideOp::Wmma), left - 1);
    }
    EXPECT_LT(t.query(ds).vaVdst, 0);
}

// A VALU overwriting a ds_load's address is freed by later LDS ops, not by WMMAs.
TEST_F(WaitAluTrackerTest, DsSourceOverwriteCountsLdsOps) {
    std::string body = "  v100 = \"st.ds_load_b32\"(v40) " + std::string(kDsAttrs) + "\n";
    for (int i = 0; i < 16; ++i)
        body += "  v" + std::to_string(101 + i) + " = \"st.ds_load_b32\"(v50) " + kDsAttrs + "\n";
    body += "  v40 = \"st.v_mov_b32\"(v41) { issueCycles = 1, latencyCycles = 1 }\n";
    const std::vector<StinkyInstruction*> insts = parse(body);
    ASSERT_EQ(insts.size(), 18u);
    const StinkyInstruction& mov = *insts.back();
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    int left = t.untilFree(mov, WaitAluHideOp::Lds);
    ASSERT_GT(left, 0);
    EXPECT_EQ(t.untilFree(mov, WaitAluHideOp::Wmma), 0);
    EXPECT_EQ(t.requiredGap(*insts[0], mov, WaitAluHideOp::Lds), left);
    for (size_t i = 1; left > 0; ++i, --left) {
        ASSERT_LT(i, insts.size() - 1);
        EXPECT_GE(t.query(mov).vmVsrc, 0);
        t.commit(*insts[i]);
        EXPECT_EQ(t.untilFree(mov, WaitAluHideOp::Lds), left - 1);
    }
    EXPECT_LT(t.query(mov).vmVsrc, 0);
}

// A transcendental after the producer breaks the matrix-op hide: only a wait drains it.
TEST_F(WaitAluTrackerTest, TransBreaksTheHide) {
    const std::vector<StinkyInstruction*> insts =
        parse("  v[0:7] = " + std::string(kWmma) + "\n" +
              "  v40 = \"st.v_mov_b32\"(v41) { issueCycles = 1, latencyCycles = 1 }\n" +
              "  v42 = \"st.v_rcp_f32\"(v43) { issueCycles = 1, latencyCycles = 8 }\n" +
              "  v100 = \"st.ds_load_b32\"(v40) " + kDsAttrs + "\n");
    ASSERT_EQ(insts.size(), 4u);
    WaitAluTracker t = makeTracker();
    for (size_t i = 0; i < 3; ++i) t.commit(*insts[i]);
    EXPECT_EQ(t.untilFree(*insts[3], WaitAluHideOp::Wmma), -1);
}

// A VALU consumer never gets a va_vdst, and the form comes from formWmma before any WMMA issued.
TEST_F(WaitAluTrackerTest, RequiredGapUsesFormWmma) {
    const std::vector<StinkyInstruction*> insts =
        parse("  v[0:7] = " + std::string(kWmma) + "\n" +
              "  v40 = \"st.v_mov_b32\"(v41) { issueCycles = 1, latencyCycles = 1 }\n" +
              "  v44 = \"st.v_mov_b32\"(v40) { issueCycles = 1, latencyCycles = 1 }\n" +
              "  v100 = \"st.ds_load_b32\"(v40) " + kDsAttrs + "\n");
    ASSERT_EQ(insts.size(), 4u);
    WaitAluTracker t = makeTracker();
    EXPECT_EQ(t.requiredGap(*insts[1], *insts[2], WaitAluHideOp::Wmma), 0);
    EXPECT_EQ(t.requiredGap(*insts[1], *insts[3], WaitAluHideOp::Wmma), -1);
    EXPECT_GT(t.requiredGap(*insts[1], *insts[3], WaitAluHideOp::Wmma, insts[0]), 0);
}

// A VALU overwriting a flat_prefetch address is freed by flat ops (both FIFOs).
TEST_F(WaitAluTrackerTest, FlatPrefetchAddressOverwrite) {
    std::string body = "  \"st.flat_prefetch_b8\"(v[0:1]) { issueCycles = 1, latencyCycles = 1 }\n";
    for (int i = 0; i < 16; ++i)
        body += "  \"st.flat_prefetch_b8\"(v[" + std::to_string(2 + 2 * i) + ":" +
                std::to_string(3 + 2 * i) + "]) { issueCycles = 1, latencyCycles = 1 }\n";
    body += "  v0 = \"st.v_mov_b32\"(v100) { issueCycles = 1, latencyCycles = 1 }\n";
    std::vector<StinkyInstruction*> insts = parse(body);
    ASSERT_EQ(insts.size(), 18u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    const std::vector<StinkyInstruction*> followers(insts.begin() + 1, insts.end() - 1);
    expectCountsDown(t, *insts.back(), WaitAluHideOp::Flat, followers, /*vm=*/true);
}

// A VALU overwriting a global_prefetch address is freed by TEX-FIFO ops, not LDS ones.
TEST_F(WaitAluTrackerTest, GlobalPrefetchAddressOverwrite) {
    std::string body =
        "  \"st.global_prefetch_b8\"(v0, s[0:1]) { issueCycles = 1, latencyCycles = 1 }\n";
    for (int i = 0; i < 16; ++i)
        body += "  \"st.global_prefetch_b8\"(v" + std::to_string(10 + i) +
                ", s[0:1]) { issueCycles = 1, latencyCycles = 1 }\n";
    body += "  v0 = \"st.v_mov_b32\"(v100) { issueCycles = 1, latencyCycles = 1 }\n";
    std::vector<StinkyInstruction*> insts = parse(body);
    ASSERT_EQ(insts.size(), 18u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    EXPECT_EQ(t.untilFree(*insts.back(), WaitAluHideOp::Lds), -1);
    const std::vector<StinkyInstruction*> followers(insts.begin() + 1, insts.end() - 1);
    expectCountsDown(t, *insts.back(), WaitAluHideOp::Tex, followers, /*vm=*/true);
}

// An EXEC writer drains va_vdst whenever any VALU is in flight; no WMMA count hides it.
TEST_F(WaitAluTrackerTest, ExecWriterNeverHidden) {
    const std::vector<StinkyInstruction*> insts =
        parse("  v[0:7] = " + std::string(kWmma) + "\n" +
              "  v40 = \"st.v_mov_b32\"(v41) { issueCycles = 1, latencyCycles = 1 }\n" +
              "  s0 = \"st.s_and_saveexec_b32\"(s1) { issueCycles = 1, latencyCycles = 2 }\n");
    ASSERT_EQ(insts.size(), 3u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    t.commit(*insts[1]);
    EXPECT_EQ(t.untilFree(*insts[2], WaitAluHideOp::Wmma), -1);
}

// Before any WMMA issued, untilFree takes the form from formWmma; a non-WMMA gives none.
TEST_F(WaitAluTrackerTest, UntilFreeUsesFormWmma) {
    const std::vector<StinkyInstruction*> insts =
        parse("  v[0:7] = " + std::string(kWmma) + "\n" +
              "  v40 = \"st.v_mov_b32\"(v41) { issueCycles = 1, latencyCycles = 1 }\n" +
              "  v100 = \"st.ds_load_b32\"(v40) " + kDsAttrs + "\n");
    ASSERT_EQ(insts.size(), 3u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[1]);
    EXPECT_EQ(t.untilFree(*insts[2], WaitAluHideOp::Wmma), -1);
    EXPECT_EQ(t.untilFree(*insts[2], WaitAluHideOp::Wmma, insts[1]), -1);
    EXPECT_GT(t.untilFree(*insts[2], WaitAluHideOp::Wmma, insts[0]), 0);
    EXPECT_EQ(t.requiredGap(*insts[1], *insts[2], WaitAluHideOp::Wmma, insts[1]), -1);
}

// The loop's first ds_load refills v[200:203], which the loop's WMMAs read. Only a tracker seeded
// with the loop block's entry sees the last read of the previous iteration, and it predicts the
// s_wait_alu the pass puts in front of the ds_load.
TEST_F(WaitAluTrackerTest, SeededTrackerSeesTheBackEdge) {
    const std::string text =
        "st.func @t() {\n"
        "^entry:\n"
        "  v300 = \"st.v_mov_b32\"(v301) { issueCycles = 1, latencyCycles = 1 }\n"
        "  Successors: ^loop\n"
        "^loop:\n"
        "  v[200:203] = \"st.ds_load_b128\"(v300) " +
        std::string(kDsAttrs) + "\n" + wmmas(3) +
        "  SCC0 = \"st.s_cmp_eq_u32\"(s0, s1) { issueCycles = 1, latencyCycles = 1 }\n"
        "  \"st.s_cbranch_scc1\"(\"loop\", SCC0) { issueCycles = 1, latencyCycles = 1 }\n"
        "  Successors: ^loop, ^exit\n"
        "^exit:\n"
        "  \"st.s_endpgm\"() { issueCycles = 1, latencyCycles = 1 }\n"
        "^dead:\n"
        "  v400 = \"st.v_mov_b32\"(v401) { issueCycles = 1, latencyCycles = 1 }\n"
        "}\n";
    Function* func = converter.convertToFunction(text);
    ASSERT_NE(func, nullptr);
    auto blockNamed = [&](const std::string& label) -> BasicBlock* {
        for (BasicBlock& bb : *func)
            if (bb.getLabel() == label) return &bb;
        return nullptr;
    };
    BasicBlock* loop = blockNamed("loop");
    BasicBlock* dead = blockNamed("dead");
    ASSERT_NE(loop, nullptr);
    ASSERT_NE(dead, nullptr);
    auto* ds = dyn_cast<StinkyInstruction>(&*loop->begin());
    ASSERT_NE(ds, nullptr);

    const InsertWaitAluOptions opts = gfx1250InsertWaitAluOptions(true);
    WaitAluEntryStates states(*func, passCtx, opts);
    EXPECT_TRUE(states.has(*loop));
    EXPECT_FALSE(states.has(*dead));

    WaitAluTracker empty = makeTracker();
    EXPECT_EQ(empty.untilFree(*ds, WaitAluHideOp::Wmma), 0);
    EXPECT_LT(empty.query(*ds).vaVdst, 0);

    WaitAluTracker seeded(states, *loop);
    const int predicted = seeded.query(*ds).vaVdst;
    EXPECT_GE(predicted, 0);
    EXPECT_GT(seeded.untilFree(*ds, WaitAluHideOp::Wmma), 0);

    AnalysisManager am;
    registerAllAnalyses(am);
    createInsertWaitAluPass(opts)->run(*func, passCtx, am);
    auto prevIt = IRList::iterator(ds);
    ASSERT_NE(prevIt, loop->begin());
    --prevIt;
    auto* wait = dyn_cast<StinkyInstruction>(prevIt.getNodePtr());
    ASSERT_NE(wait, nullptr);
    ASSERT_EQ(wait->getUnifiedOpcode(), GFX::s_wait_alu);
    const auto* data = wait->getModifier<SWaitAluData>();
    ASSERT_NE(data, nullptr);
    ASSERT_TRUE(data->hasField(SWaitAluData::VA_VDST));
    EXPECT_EQ(static_cast<int>(data->getField(SWaitAluData::VA_VDST)), predicted);
}

}  // namespace
