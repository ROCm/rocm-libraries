// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// FmhaDecodePaged5dKernel (gfx125) against the CPU reference of fmha_decode_paged5d_runner.

#include <gtest/gtest.h>

#include "fmha_decode_paged5d_runner.hpp"

namespace {

using fmha_decode_paged5d::Case;

struct Param
{
    std::string name;
    Case c;
    // layout the case must run with: "coop" or "wave", "+groups" for several clusters per
    // (batch, kv head); empty: not checked
    const char* layout = "";
};

std::vector<Param> params()
{
    std::vector<Param> p;
    const auto add = [&](const std::string& name, Case c, const char* layout = "") {
        p.push_back({name, c, layout});
    };
    // window 2048 at short, medium and long contexts
    add("swa_b1_s1024", {1, 16, 1, {1024}, 2048}, "coop");
    add("swa_b8_s8192", {8, 16, 1, {8192}, 2048}, "coop");
    add("swa_b2_s73728", {2, 16, 1, {73728}, 2048}, "coop");
    // partial last page, sequences shorter than one step, one page
    add("swa_b3_s1000", {3, 16, 1, {1000}, 2048});
    add("swa_b2_s77", {2, 16, 1, {77}, 2048});
    add("swa_b1_s5", {1, 16, 1, {5}, 2048});
    // different lengths in one batch: some splits of the shorter sequences are empty
    add("swa_mixed", {4, 16, 1, {9000, 100, 2048, 3333}, 2048});
    // no window: one cluster of either layout, several clusters
    add("full_b2_s3000", {2, 16, 1, {3000}, 0}, "coop");
    add("full_b8_s8192", {8, 16, 1, {8192}, 0}, "wave");
    add("full_b1_s20000", {1, 16, 1, {20000}, 0}, "coop+groups");
    add("groups_full_b1_s32768", {1, 16, 1, {32768}, 0}, "coop+groups");
    add("groups_mixed", {3, 16, 1, {40000, 5000, 70}, 0}, "coop+groups");
    add("groups_forced", {2, 16, 2, {6000}, 2048, 30}, "coop+groups");
    // forced layouts
    add("full_coop", {2, 16, 1, {5000}, 0, 0, 0, "coop"}, "coop");
    add("swa_wave", {2, 16, 1, {5000}, 2048, 0, 0, "wave"}, "wave");
    // one split (direct output), the most splits a cluster takes
    add("one_split", {2, 16, 1, {700}, 2048, 1});
    add("fifteen_splits", {2, 16, 1, {960}, 2048, 15});
    // other GQA shapes: fewer than 16 rows, several kv heads
    add("gqa_8_1", {2, 8, 1, {2500}, 2048});
    add("gqa_16_2", {2, 16, 2, {4000}, 2048});
    add("gqa_32_4_wave", {2, 32, 4, {999}, 2048, 0, 0, "wave"}, "wave");
    add("gqa_6_3", {2, 6, 3, {1500}, 512});
    // launches sized for a longer bound than the contexts, as in graph replay: the ranges spread
    // over the grid, clusters without pages exit
    add("bound_full_b8", {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072}, "wave+groups");
    add("bound_groups_mixed",
        {3, 16, 1, {40000, 5000, 70}, 0, 0, 0, "auto", 131072},
        "wave+groups");
    add("bound_coop_groups", {2, 16, 1, {2000}, 0, 0, 0, "auto", 30000}, "coop+groups");
    add("bound_one_cluster", {2, 16, 1, {300}, 0, 0, 0, "auto", 3840}, "coop");
    add("bound_swa_gqa", {4, 16, 2, {100, 7000}, 2048, 0, 0, "auto", 65536}, "coop");
    // cache pages of 64 and 128 tokens: windows starting inside a cache page, partial last
    // cache pages, both layouts, several clusters, a longer launch bound
    for(const ck_tile::index_t ps : {64, 128})
    {
        const auto name = [&](const char* n) { return "p" + std::to_string(ps) + "_" + n; };
        add(name("swa_b2_s8192"), {2, 16, 1, {8192}, 2048, 0, 0, "auto", 0, ps}, "coop");
        add(name("swa_b3_s1000_w300"), {3, 16, 1, {1000}, 300, 0, 0, "auto", 0, ps});
        add(name("swa_b1_s5"), {1, 16, 1, {5}, 2048, 0, 0, "auto", 0, ps});
        add(name("swa_mixed"), {4, 16, 1, {9000, 100, 2048, 3333}, 2048, 0, 0, "auto", 0, ps});
        add(name("swa_wave"), {2, 16, 1, {5000}, 2048, 0, 0, "wave", 0, ps}, "wave");
        add(name("full_b8_s8192"), {8, 16, 1, {8192}, 0, 0, 0, "auto", 0, ps}, "wave");
        add(name("full_b1_s20000"), {1, 16, 1, {20000}, 0, 0, 0, "auto", 0, ps}, "coop+groups");
        add(name("groups_forced"), {2, 16, 2, {6000}, 2048, 30, 0, "auto", 0, ps}, "coop+groups");
        add(name("gqa_16_2"), {2, 16, 2, {4000}, 2048, 0, 0, "auto", 0, ps});
        add(name("gqa_32_4_wave"), {2, 32, 4, {999}, 2048, 0, 0, "wave", 0, ps}, "wave");
        add(name("bound_full_b8"), {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072, ps}, "wave+groups");
    }
    // fp16 Q and output: direct output, both layouts within a cluster, several clusters
    add("fp16_one_split", {2, 16, 1, {700}, 2048, 1, 0, "auto", 0, 16, true});
    add("fp16_swa_b8_s8192", {8, 16, 1, {8192}, 2048, 0, 0, "auto", 0, 16, true}, "coop");
    add("fp16_full_b8_s8192", {8, 16, 1, {8192}, 0, 0, 0, "auto", 0, 16, true}, "wave");
    add("fp16_full_b1_s20000", {1, 16, 1, {20000}, 0, 0, 0, "auto", 0, 16, true}, "coop+groups");
    add("fp16_p64_bound_full_b8",
        {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072, 64, true},
        "wave+groups");
    return p;
}

class DecodePaged5d : public ::testing::TestWithParam<Param>
{
};

TEST_P(DecodePaged5d, MatchesReference)
{
    if(ck_tile::get_device_name().rfind("gfx125", 0) != 0)
        GTEST_SKIP() << "gfx125 only";
    // one timed launch: also replays a captured graph and compares its output
    const auto r = fmha_decode_paged5d::run(GetParam().c, true, 0, 1);
    ASSERT_TRUE(r.error.empty()) << r.error;
    EXPECT_TRUE(r.graph_same) << "graph replay output differs from a stream launch";
    const std::string layout = std::string(r.config.wave_split ? "wave" : "coop") +
                               (r.config.num_groups > 1 ? "+groups" : "");
    if(*GetParam().layout != '\0')
        EXPECT_EQ(layout, GetParam().layout);
    EXPECT_TRUE(r.valid) << "max abs err " << r.max_err << " (max |ref| " << r.max_ref << "), cos "
                         << r.cosine << ", splits " << r.config.num_splits << ", pages/split "
                         << r.config.pages_per_split << (r.config.wave_split ? ", wave" : ", coop");
}

INSTANTIATE_TEST_SUITE_P(Gfx125,
                         DecodePaged5d,
                         ::testing::ValuesIn(params()),
                         [](const auto& test_info) { return test_info.param.name; });

} // namespace
