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

// Mixed hide of the latency-8 / 8-VGPR form: VALUs count only once the lead count of matrix ops
// issued, then each one fills a tail slot.
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
    // The mixed rule as a remainder: matrix ops to the lead, then tail slots of one each.
    EXPECT_EQ(t.requiredGapMixed(*insts[0], ds), (WaitAluMixGap{5, 7, 1}));
    EXPECT_EQ(t.untilFreeMixed(ds), (WaitAluMixGap{5, 7, 1}));
    // Before the lead a VALU fills no mixed-rule slot, only the no-order rule's; matrix ops
    // fill no slot of that rule, so its count is unchanged one matrix op short of the lead.
    EXPECT_EQ(t.untilFree(ds, WaitAluHideOp::Valu), 21);
    for (int i = 1; i <= 4; ++i) t.commit(*insts[i]);
    EXPECT_EQ(t.untilFree(ds, WaitAluHideOp::Valu), 21);
    EXPECT_EQ(t.untilFreeMixed(ds), (WaitAluMixGap{1, 7, 1}));
    t.commit(*insts[5]);
    // Lead met: the tail remains, VALUs or matrix ops one slot each; a VALU takes one slot off it.
    EXPECT_EQ(t.untilFree(ds, WaitAluHideOp::Wmma), 7);
    EXPECT_EQ(t.untilFreeMixed(ds), (WaitAluMixGap{0, 7, 1}));
    t.commit(*insts[6]);
    EXPECT_EQ(t.untilFreeMixed(ds), (WaitAluMixGap{0, 6, 1}));
    std::vector<StinkyInstruction*> valus(insts.begin() + 7, insts.begin() + 13);
    expectCountsDown(t, ds, WaitAluHideOp::Valu, valus, /*vm=*/false);
}

// Overwriting a WMMA source uses the src rule; overwriting its C (read and written) the dst rule.
TEST_F(WaitAluTrackerTest, SourceOverwriteUsesSrcRule) {
    std::string body =
        "  v[0:7] = \"st.v_wmma_f32_16x16x128_fp8_fp8\"(v[300:315], v[316:331], v[0:7]) { "
        "issueCycles = 1, latencyCycles = 8, mod.mfma = { reuseA = false, reuseB = false } }\n";
    for (int i = 0; i < 12; ++i) {
        const std::string r =
            "v[" + std::to_string(8 + 8 * i) + ":" + std::to_string(15 + 8 * i) + "]";
        body += "  ";
        body += r;
        body += " = \"st.v_wmma_f32_16x16x128_fp8_fp8\"(v[400:415], v[416:431], ";
        body += r;
        body +=
            ") { issueCycles = 1, latencyCycles = 8, mod.mfma = { reuseA = false, reuseB = false } "
            "}\n";
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

// An f64 producer: only VALU-class ops free a source overwrite; nothing frees a use of its result.
TEST_F(WaitAluTrackerTest, DpmaccProducerCapacityRule) {
    std::string body =
        "  v[0:1] = \"st.v_fma_f64\"(v[2:3], v[4:5], v[0:1]) { issueCycles = 1, latencyCycles = 32 "
        "}\n";
    for (int i = 0; i < 12; ++i)
        body += "  v" + std::to_string(100 + i) +
                " = \"st.v_add_f32\"(v250, v251) { issueCycles = 1, latencyCycles = 5 }\n";
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

// A VALU-only rule (the f64 set): requiredGap gives -1 for matrix ops, a count for VALUs, no mix.
TEST_F(WaitAluTrackerTest, RequiredGapValuOnlyRule) {
    const std::vector<StinkyInstruction*> insts = parse(
        "  v[32:39] = \"st.v_wmma_scale_f32_16x16x128_f8f6f4\"(v[400:415], v[416:431], v[32:39], "
        "v290, v291) { issueCycles = 1, latencyCycles = 8, mod.matrix_fmt = { fmtA = "
        "\"MATRIX_FMT_FP8\", fmtB = \"MATRIX_FMT_FP8\" } }\n"
        "  v[0:1] = \"st.v_fma_f64\"(v[2:3], v[4:5], v[0:1]) { issueCycles = 1, latencyCycles = 32 "
        "}\n"
        "  v[2:3] = \"st.ds_load_b64\"(v252) " +
        std::string(kDsAttrs) + "\n");
    ASSERT_EQ(insts.size(), 3u);
    WaitAluTracker t = makeTracker();
    t.commit(*insts[0]);
    EXPECT_EQ(t.requiredGap(*insts[1], *insts[2], WaitAluHideOp::Wmma), -1);
    EXPECT_EQ(t.requiredGap(*insts[1], *insts[2], WaitAluHideOp::Valu), 12);
    EXPECT_EQ(t.requiredGapMixed(*insts[1], *insts[2]).leadWmmas, -1);
}

}  // namespace
