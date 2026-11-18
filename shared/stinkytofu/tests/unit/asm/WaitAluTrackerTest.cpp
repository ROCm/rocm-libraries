// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
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

// A transcendental or an f64 op after the producer neither counts as a matrix op nor blocks
// the hide: 12 matrix ops still free the store.
TEST_F(WaitAluTrackerTest, TransAndDpmaccDoNotBreakTheHide) {
    const std::vector<StinkyInstruction*> insts =
        parse("  v[0:7] = " + std::string(kWmma) + "\n" +
              "  v40 = \"st.v_mov_b32\"(v41) { issueCycles = 1, latencyCycles = 1 }\n" +
              "  v42 = \"st.v_rcp_f32\"(v43) { issueCycles = 1, latencyCycles = 8 }\n" +
              "  v[44:45] = \"st.v_fma_f64\"(v[46:47], v[48:49], v[50:51]) { issueCycles = 1, latencyCycles = 32 }\n" +
              "  v100 = \"st.ds_load_b32\"(v40) " + kDsAttrs + "\n");
    ASSERT_EQ(insts.size(), 5u);
    WaitAluTracker t = makeTracker();
    for (size_t i = 0; i < 4; ++i) t.commit(*insts[i]);
    EXPECT_EQ(t.untilFree(*insts[4], WaitAluHideOp::Wmma), 12);
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

// Mixed hide of the latency-8 / 8-VGPR form (lead 5, tail 7): VALUs count only once 5 matrix
// ops issued, then each one fills a tail slot.
TEST_F(WaitAluTrackerTest, MixedHideCountsValusAfterLead) {
    // Each one accumulates into its own 8 VGPRs, so none re-stamps the producer's v[0:7].
    auto wmma = [](int i) {
        const std::string r = "v[" + std::to_string(8 * i) + ":" + std::to_string(8 * i + 7) + "]";
        return "  " + r + R"( = "st.v_wmma_f32_16x16x128_fp8_fp8"(v[300:315], v[316:331], )" + r +
               R"() { issueCycles = 1, latencyCycles = 8, mod.mfma = { reuseA = false, reuseB = false } })" +
               "\n";
    };
    std::string body;
    for (int i = 0; i <= 5; ++i) body += wmma(i);
    for (int i = 0; i < 7; ++i)
        body += "  v" + std::to_string(200 + i) + " = \"st.v_add_f32\"(v250, v251) " +
                "{ issueCycles = 1, latencyCycles = 5 }\n";
    body += "  v260 = \"st.ds_load_b32\"(v0) " + std::string(kDsAttrs) + "\n";
    const std::vector<StinkyInstruction*> insts = parse(body);
    ASSERT_EQ(insts.size(), 14u);
    const StinkyInstruction& ds = *insts.back();
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    // Matrix ops alone: the lead plus a tail of matrix ops, the form's matrix count.
    EXPECT_EQ(t.untilFree(ds, WaitAluHideOp::Wmma), 12);
    EXPECT_EQ(t.requiredGap(*insts[0], ds, WaitAluHideOp::Wmma), 12);
    // One VALU per WMMA window: the tail after windows 5..8 is 1, 3, 5, 7. Four per window: the
    // no-order rule counts the VALUs alone (matrix ops fill none of its 21 slots), so it is met
    // after 6 windows (24), before the mixed rule's 7 (windows 6 and 7 give 2 + 8).
    EXPECT_EQ(t.requiredGap(*insts[0], ds, WaitAluHideOp::Wmma, nullptr, 1), 8);
    EXPECT_EQ(t.requiredGap(*insts[0], ds, WaitAluHideOp::Wmma, nullptr, 4), 6);
    // Before the lead a VALU fills no mixed-rule slot, only the no-order rule's: 21, and still
    // 21 after 4 matrix ops, which fill no slot of that rule.
    EXPECT_EQ(t.untilFree(ds, WaitAluHideOp::Valu), 21);
    for (int i = 1; i <= 4; ++i) t.commit(*insts[i]);
    EXPECT_EQ(t.untilFree(ds, WaitAluHideOp::Valu), 21);
    t.commit(*insts[5]);
    // Lead met: 7 VALUs or 7 matrix ops, one slot each.
    EXPECT_EQ(t.untilFree(ds, WaitAluHideOp::Wmma), 7);
    std::vector<StinkyInstruction*> valus(insts.begin() + 6, insts.begin() + 13);
    expectCountsDown(t, ds, WaitAluHideOp::Valu, valus, /*vm=*/false);
}

// A read never waits for a read: the store waits for the v_mov that wrote v1 (one follower in
// the shared order), not for the v_add that read it.
TEST_F(WaitAluTrackerTest, ReadWaitsForWriterNotReader) {
    const std::vector<StinkyInstruction*> insts =
        parse("  v1 = \"st.v_mov_b32\"(v5) { issueCycles = 1, latencyCycles = 1 }\n"
              "  v2 = \"st.v_add_f32\"(v3, v1) { issueCycles = 1, latencyCycles = 5 }\n"
              "  \"st.ds_store_b32\"(v7, v1) " + std::string(kDsAttrs) + "\n" +
              "  v1 = \"st.ds_load_b32\"(v6) " + kDsAttrs + "\n");
    ASSERT_EQ(insts.size(), 4u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    t.commit(*insts[1]);
    EXPECT_EQ(t.query(*insts[2]).vaVdst, 1);
    // The store's wait retires the writer, not the reader: the load overwriting v1 still waits
    // for the v_add's read.
    t.commit(*insts[2]);
    EXPECT_EQ(t.query(*insts[3]).vaVdst, 0);
}

// A read does not restart the writer's hide progress: WMMAs issued before the read still count.
TEST_F(WaitAluTrackerTest, ReadKeepsWriterProgress) {
    std::string body = "  v1 = \"st.v_mov_b32\"(v5) { issueCycles = 1, latencyCycles = 1 }\n" + wmmas(6) +
                       "  v2 = \"st.v_add_f32\"(v3, v1) { issueCycles = 1, latencyCycles = 5 }\n" +
                       "  \"st.ds_store_b32\"(v7, v1) " + std::string(kDsAttrs) + "\n";
    const std::vector<StinkyInstruction*> insts = parse(body);
    ASSERT_EQ(insts.size(), 9u);
    WaitAluTracker t = makeTracker();
    for (size_t i = 0; i + 1 < insts.size(); ++i) t.commit(*insts[i]);
    // The form's mixed rule (lead 5, tail 7): 6 matrix ops met the lead and filled one tail slot,
    // the v_add after the lead filled another, so 5 more matrix ops free it. A stamp reset by
    // the read would have needed 12.
    EXPECT_EQ(t.untilFree(*insts.back(), WaitAluHideOp::Wmma), 5);
}

// The no-order rule: 21 VALUs after a CSMACC producer hide its writeback with no matrix op
// after it, once the form is latched.
TEST_F(WaitAluTrackerTest, ValuOnlyHidesWithoutLead) {
    std::string body = "  v[8:15] = " + std::string(kWmma).replace(std::string(kWmma).find("v[0:7]"), 6, "v[8:15]") + "\n" +
                       "  v40 = \"st.v_mov_b32\"(v41) { issueCycles = 1, latencyCycles = 1 }\n";
    for (int i = 0; i < 21; ++i)
        body += "  v" + std::to_string(100 + i) + " = \"st.v_add_f32\"(v250, v251) { issueCycles = 1, latencyCycles = 5 }\n";
    body += "  \"st.ds_store_b32\"(v7, v40) " + std::string(kDsAttrs) + "\n";
    const std::vector<StinkyInstruction*> insts = parse(body);
    ASSERT_EQ(insts.size(), 24u);
    const StinkyInstruction& store = *insts.back();
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    t.commit(*insts[1]);
    EXPECT_EQ(t.untilFree(store, WaitAluHideOp::Valu), 21);
    EXPECT_EQ(t.untilFree(store, WaitAluHideOp::Wmma), 12);
    std::vector<StinkyInstruction*> valus(insts.begin() + 2, insts.begin() + 23);
    expectCountsDown(t, store, WaitAluHideOp::Valu, valus, /*vm=*/false);
}

// A WAR consumer (overwrites a source the WMMA read) is freed by the src rule, 9 ops for this
// form, while the WMMA's C (read and written) still needs the dst rule's 12.
TEST_F(WaitAluTrackerTest, SourceOverwriteUsesSrcRule) {
    std::string body =
        "  v[0:7] = \"st.v_wmma_f32_16x16x128_fp8_fp8\"(v[300:315], v[316:331], v[0:7]) { issueCycles = 1, latencyCycles = 8, mod.mfma = { reuseA = false, reuseB = false } }\n";
    for (int i = 0; i < 12; ++i) {
        const std::string r = "v[" + std::to_string(8 + 8 * i) + ":" + std::to_string(15 + 8 * i) + "]";
        body += "  " + r + " = \"st.v_wmma_f32_16x16x128_fp8_fp8\"(v[400:415], v[416:431], " + r +
                ") { issueCycles = 1, latencyCycles = 8, mod.mfma = { reuseA = false, reuseB = false } }\n";
    }
    body += "  v[300:303] = \"st.ds_load_b128\"(v250) " + std::string(kDsAttrs) + "\n";
    body += "  v[0:3] = \"st.ds_load_b128\"(v250) " + std::string(kDsAttrs) + "\n";
    const std::vector<StinkyInstruction*> insts = parse(body);
    ASSERT_EQ(insts.size(), 15u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    EXPECT_EQ(t.untilFree(*insts[13], WaitAluHideOp::Wmma), 9);
    EXPECT_EQ(t.untilFree(*insts[13], WaitAluHideOp::Valu), 9);
    EXPECT_EQ(t.untilFree(*insts[14], WaitAluHideOp::Wmma), 12);
    EXPECT_EQ(t.untilFree(*insts[14], WaitAluHideOp::Valu), 21);
    for (int i = 1; i <= 9; ++i) t.commit(*insts[i]);
    EXPECT_LT(t.query(*insts[13]).vaVdst, 0);
    EXPECT_EQ(t.query(*insts[14]).vaVdst, 9);
}

// An f64 producer: no count of matrix ops frees a consumer (-1), 12 VALU-class ops free one that
// overwrites a source, and nothing frees one that reads or overwrites its result.
TEST_F(WaitAluTrackerTest, DpmaccProducerCapacityRule) {
    std::string body =
        "  v[0:1] = \"st.v_fma_f64\"(v[2:3], v[4:5], v[0:1]) { issueCycles = 1, latencyCycles = 32 }\n";
    for (int i = 0; i < 12; ++i)
        body += "  v" + std::to_string(100 + i) + " = \"st.v_add_f32\"(v250, v251) { issueCycles = 1, latencyCycles = 5 }\n";
    body += "  v[2:3] = \"st.ds_load_b64\"(v252) " + std::string(kDsAttrs) + "\n";
    body += "  v[0:1] = \"st.ds_load_b64\"(v252) " + std::string(kDsAttrs) + "\n";
    body += "  \"st.ds_store_b64\"(v252, v[0:1]) " + std::string(kDsAttrs) + "\n";
    const std::vector<StinkyInstruction*> insts = parse(body);
    ASSERT_EQ(insts.size(), 16u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    EXPECT_EQ(t.untilFree(*insts[13], WaitAluHideOp::Wmma), -1);
    EXPECT_EQ(t.untilFree(*insts[13], WaitAluHideOp::Valu), 12);
    EXPECT_EQ(t.untilFree(*insts[14], WaitAluHideOp::Valu), -1);
    EXPECT_EQ(t.untilFree(*insts[15], WaitAluHideOp::Valu), -1);
    std::vector<StinkyInstruction*> valus(insts.begin() + 1, insts.begin() + 13);
    expectCountsDown(t, *insts[13], WaitAluHideOp::Valu, valus, /*vm=*/false);
    EXPECT_EQ(t.query(*insts[14]).vaVdst, 0);
    EXPECT_EQ(t.query(*insts[15]).vaVdst, 0);
}

// A CSMACC producer before any matrix op: the no-latch set frees a source overwrite after 12
// VALUs; matrix ops are not what it counts.
TEST_F(WaitAluTrackerTest, NoLatchCsmaccCapacityRule) {
    std::string body = "  v9 = \"st.v_add_f32\"(v300, v301) { issueCycles = 1, latencyCycles = 5 }\n";
    for (int i = 0; i < 12; ++i)
        body += "  v" + std::to_string(100 + i) + " = \"st.v_add_f32\"(v250, v251) { issueCycles = 1, latencyCycles = 5 }\n";
    body += "  v300 = \"st.ds_load_b32\"(v252) " + std::string(kDsAttrs) + "\n";
    const std::vector<StinkyInstruction*> insts = parse(body);
    ASSERT_EQ(insts.size(), 14u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    EXPECT_EQ(t.untilFree(*insts[13], WaitAluHideOp::Wmma), -1);
    EXPECT_EQ(t.untilFree(*insts[13], WaitAluHideOp::Valu), 12);
    std::vector<StinkyInstruction*> valus(insts.begin() + 1, insts.begin() + 13);
    expectCountsDown(t, *insts[13], WaitAluHideOp::Valu, valus, /*vm=*/false);
}

// A rule only VALUs satisfy (the f64 capacity set) behind each WMMA: one VALU per WMMA needs 12
// WMMAs, two need 6, none never.
TEST_F(WaitAluTrackerTest, RequiredGapCountsValusForAValuOnlyRule) {
    const std::vector<StinkyInstruction*> insts = parse(
        "  v[32:39] = \"st.v_wmma_scale_f32_16x16x128_f8f6f4\"(v[400:415], v[416:431], v[32:39], v290, v291) { issueCycles = 1, latencyCycles = 8, mod.matrix_fmt = { fmtA = \"MATRIX_FMT_FP8\", fmtB = \"MATRIX_FMT_FP8\" } }\n"
        "  v[0:1] = \"st.v_fma_f64\"(v[2:3], v[4:5], v[0:1]) { issueCycles = 1, latencyCycles = 32 }\n"
        "  v[2:3] = \"st.ds_load_b64\"(v252) " + std::string(kDsAttrs) + "\n");
    ASSERT_EQ(insts.size(), 3u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    EXPECT_EQ(t.requiredGap(*insts[1], *insts[2], WaitAluHideOp::Wmma, nullptr, 0), -1);
    EXPECT_EQ(t.requiredGap(*insts[1], *insts[2], WaitAluHideOp::Wmma, nullptr, 1), 12);
    EXPECT_EQ(t.requiredGap(*insts[1], *insts[2], WaitAluHideOp::Wmma, nullptr, 2), 6);
}

}  // namespace
