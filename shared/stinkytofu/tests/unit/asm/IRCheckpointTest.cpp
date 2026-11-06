/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#include <gtest/gtest.h>

#include <string>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/pipeline/IRCheckpoint.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"

using namespace stinkytofu;

TEST(IRCheckpointSpecTest, ParsesLabelPassAndOccurrence) {
    auto spec = parseIRCheckpointSpec("loopWithPrefetch+noLoadLoopBody:CFG Builder#2");
    ASSERT_TRUE(spec);
    EXPECT_EQ(spec->label, "loopWithPrefetch+noLoadLoopBody");
    EXPECT_EQ(spec->passName, "CFG Builder");
    EXPECT_EQ(spec->occurrence, 2u);
    EXPECT_EQ(spec->str(), "loopWithPrefetch+noLoadLoopBody:CFG Builder#2");

    auto defaulted = parseIRCheckpointSpec("module:StinkyBuildImplicitDependencyPass");
    ASSERT_TRUE(defaulted);
    EXPECT_EQ(defaulted->occurrence, 1u);
}

TEST(IRCheckpointSpecTest, RejectsMalformedSpecs) {
    std::string error;
    EXPECT_FALSE(parseIRCheckpointSpec("NoLabel", &error));
    EXPECT_NE(error.find("<pmLabel>"), std::string::npos);
    EXPECT_FALSE(parseIRCheckpointSpec(":Pass"));
    EXPECT_FALSE(parseIRCheckpointSpec("label:"));
    EXPECT_FALSE(parseIRCheckpointSpec("label:Pass#0"));
    EXPECT_FALSE(parseIRCheckpointSpec("label:Pass#x"));
}

TEST(IRCheckpointHashTest, MatchesKnownSha256) {
    EXPECT_EQ(sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // Two blocks: the padding crosses the 56-byte boundary.
    EXPECT_EQ(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(IRCheckpointIdTest, TakesIdsFromBothForms) {
    std::string inst =
        R"(v0 = "st.v_mov_b32"(v1) { issueCycles = 1, latencyCycles = 2, mod.ckpt = { id = 7 } })";
    EXPECT_EQ(takeCheckpointId(inst), 7);
    EXPECT_EQ(inst, R"(v0 = "st.v_mov_b32"(v1) { issueCycles = 1, latencyCycles = 2 })");

    std::string directive = R"("st.asm_directive"(".align", "8") { mod.ckpt = { id = 12 } })";
    EXPECT_EQ(takeCheckpointId(directive), 12);
    EXPECT_EQ(directive, R"("st.asm_directive"(".align", "8"))");

    std::string plain = R"(v0 = "st.v_mov_b32"(v1) { issueCycles = 1, latencyCycles = 2 })";
    EXPECT_EQ(takeCheckpointId(plain), -1);
}

TEST(IRCheckpointRenderTest, IdsStripBackToTheFingerprintedText) {
    const std::string source = R"(st.func @render() {
^entry:
  v0 = "st.v_mov_b32"(v1) { issueCycles = 1, latencyCycles = 2 }
  "st.s_branch"("tail")
  Successors: ^tail
^tail:
  v2 = "st.v_add_f32"(v0, v1)
  "st.s_endpgm"()
}
)";
    Function function("render");
    PassContext ctx;
    ASSERT_EQ(StinkyIRConverter::populateFunctionFromString(source, function, ctx,
                                                            getGfxArchID(12, 5, 0)),
              StinkyErrorCode::SUCCESS);

    const IRCheckpointRendering plain = renderCheckpointRegion(function, /*withIds=*/false);
    const IRCheckpointRendering tagged = renderCheckpointRegion(function, /*withIds=*/true);
    ASSERT_EQ(plain.nodes.size(), 4u);
    EXPECT_EQ(plain.blockLabels, (std::vector<std::string>{"entry", "tail"}));
    EXPECT_EQ(plain.blockOf, (std::vector<size_t>{0, 0, 1, 1}));
    EXPECT_NE(tagged.text.find("mod.ckpt = { id = 3 }"), std::string::npos);
    EXPECT_EQ(stripCheckpointIds(tagged.text), plain.text);
    EXPECT_NE(plain.text.find("Successors: ^tail"), std::string::npos);
}
