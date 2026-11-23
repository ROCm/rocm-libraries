// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// fmha_fwd_decode_pagedkv (gfx125) against the CPU reference of its example runner.

#include <gtest/gtest.h>

#include "fmha_fwd_decode_pagedkv_runner.hpp"

namespace {

using fmha_fwd_decode_pagedkv_runner::Case;

struct Param
{
    std::string name;
    Case c;
    // what the case must run with: "coop" or "wave", "/" workgroups per cluster, "x" clusters
    // per (batch, kv head) when there are several
    std::string layout;
};

std::string layout_of(const ck_tile::FmhaFwdDecodePagedKVConfig& c)
{
    return std::string(c.wave_split ? "wave" : "coop") + "/" + std::to_string(c.num_splits) +
           (c.num_groups > 1 ? "x" + std::to_string(c.num_groups) : "");
}

std::string describe(const fmha_fwd_decode_pagedkv_runner::Result& r)
{
    return "row err/tol " + std::to_string(r.row_err) + ", max abs err " +
           std::to_string(r.max_err) + " (max |ref| " + std::to_string(r.max_ref) + "), cos " +
           std::to_string(r.cosine) + ", " + layout_of(r.config);
}

std::vector<Param> params()
{
    std::vector<Param> p;
    const auto add = [&](const std::string& name, Case c, const std::string& layout) {
        p.push_back({name, c, layout});
    };
    // window 2048 at short, medium and long contexts
    add("swa_b1_s1024", {1, 16, 1, {1024}, 2048}, "coop/16");
    add("swa_b8_s8192", {8, 16, 1, {8192}, 2048}, "coop/11");
    add("swa_b2_s73728", {2, 16, 1, {73728}, 2048}, "coop/11");
    // partial last page, sequences shorter than one step, one page
    add("swa_b3_s1000", {3, 16, 1, {1000}, 2048}, "coop/16");
    add("swa_b2_s77", {2, 16, 1, {77}, 2048}, "coop/1");
    add("swa_b1_s5", {1, 16, 1, {5}, 2048}, "coop/1");
    // different lengths in one batch: some splits of the shorter sequences are empty
    add("swa_mixed", {4, 16, 1, {9000, 100, 2048, 3333}, 2048}, "coop/11");
    // no key, fewer keys than one page; windows below and just above one page
    add("swa_s0_to_3", {4, 16, 1, {0, 1, 2, 3}, 2048}, "coop/1");
    add("full_s0_to_3", {4, 16, 1, {0, 1, 2, 3}, 0}, "coop/1");
    add("swa_mixed_empty", {3, 16, 1, {0, 5000, 16}, 2048}, "coop/11");
    for(const ck_tile::index_t w : {1, 5, 17})
        add("swa_w" + std::to_string(w), {3, 16, 1, {1000, 77, 3}, w}, "coop/1");
    // no window: one cluster of either layout, several clusters
    add("full_b2_s3000", {2, 16, 1, {3000}, 0}, "coop/16");
    add("full_b8_s8192", {8, 16, 1, {8192}, 0}, "wave/16");
    add("full_b1_s20000", {1, 16, 1, {20000}, 0}, "coop/16x10");
    add("groups_full_b1_s32768", {1, 16, 1, {32768}, 0}, "coop/16x11");
    add("groups_mixed", {3, 16, 1, {40000, 5000, 70}, 0}, "coop/16x10");
    add("groups_forced", {2, 16, 2, {6000}, 2048, 30}, "coop/13x2");
    // one wave-split split above 12 pages and for more kv heads than pages; with many kv heads
    // one per started 64 pages, as many as 480 workgroups allow, two only above 192 pages
    add("short_wave", {2, 16, 1, {250}, 0}, "wave/1");
    add("heads_wave_one", {5, 16, 8, {500}, 512}, "wave/1");
    add("heads_wave_four", {8, 16, 8, {4000}, 0}, "wave/4");
    add("heads_wave_three", {10, 16, 16, {4000}, 0}, "wave/3");
    add("heads_wave_two", {25, 8, 8, {4000}, 0}, "wave/2");
    add("heads_wave_not_two", {25, 8, 8, {3000}, 0}, "wave/1");
    // 513 to 1024 pages: one wave-split cluster from 3 kv heads, cooperative clusters below
    add("full_b3_s10000", {3, 16, 1, {10000}, 0}, "wave/14");
    add("full_b2_s10000", {2, 16, 1, {10000}, 0}, "coop/16x5");
    // one wave-split split where two would exceed 480 workgroups
    add("heads_wave_long", {16, 16, 16, {4200}, 0}, "wave/1");
    // a device whose clusters hold fewer than 16 workgroups: no cluster above the cap, more
    // clusters instead; without clusters one split
    const auto capped = [](Case c, ck_tile::index_t cap) {
        c.max_cluster = cap;
        return c;
    };
    add("cap8_swa_b8_s8192", capped({8, 16, 1, {8192}, 2048}, 8), "coop/7");
    add("cap8_full_b8_s8192", capped({8, 16, 1, {8192}, 0}, 8), "wave/8");
    add("cap4_full_b1_s20000", capped({1, 16, 1, {20000}, 0}, 4), "coop/4x40");
    add("cap2_full_b2_s3000", capped({2, 16, 1, {3000}, 0}, 2), "coop/2");
    add("cap1_swa_b2_s8192", capped({2, 16, 1, {8192}, 2048}, 1), "wave/1");
    // forced layouts
    add("full_coop", {2, 16, 1, {5000}, 0, 0, 0, "coop"}, "coop/10");
    add("swa_wave", {2, 16, 1, {5000}, 2048, 0, 0, "wave"}, "wave/11");
    // one split (direct output); 15 splits, the most with a compile-time cluster size; 16, the
    // most a cluster takes (cluster size given at launch)
    add("one_split", {2, 16, 1, {700}, 2048, 1}, "coop/1");
    add("fifteen_splits", {2, 16, 1, {960}, 2048, 15}, "coop/15");
    add("sixteen_splits", {2, 16, 1, {1024}, 2048, 16}, "coop/16");
    add("sixteen_splits_wave", {2, 16, 1, {8192}, 0, 16, 0, "wave"}, "wave/16");
    // other GQA shapes: fewer than 16 rows, several kv heads
    add("gqa_8_1", {2, 8, 1, {2500}, 2048}, "coop/11");
    add("gqa_16_2", {2, 16, 2, {4000}, 2048}, "coop/11");
    add("gqa_32_4_wave", {2, 32, 4, {999}, 2048, 0, 0, "wave"}, "wave/16");
    add("gqa_6_3", {2, 6, 3, {1500}, 512}, "coop/9");
    // launches sized for a longer bound than the contexts, as in graph replay: the ranges spread
    // over the grid, clusters without pages exit
    add("bound_full_b8", {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072}, "wave/16x3");
    add("bound_groups_mixed", {3, 16, 1, {40000, 5000, 70}, 0, 0, 0, "auto", 131072}, "wave/16x8");
    add("bound_coop_groups", {2, 16, 1, {2000}, 0, 0, 0, "auto", 30000}, "coop/16x10");
    add("bound_one_cluster", {2, 16, 1, {300}, 0, 0, 0, "auto", 3840}, "coop/15");
    add("bound_swa_gqa", {4, 16, 2, {100, 7000}, 2048, 0, 0, "auto", 65536}, "coop/11");
    // a window over a context just below the bound covers one page (64-token pages: four) more
    // than the bound's own range; splits sized for the bound alone would drop the newest keys
    add("bound_swa_b64_s8191", {64, 16, 1, {8191}, 2048, 0, 0, "auto", 8192}, "coop/9");
    add("bound_groups_s29999", {1, 16, 1, {29999}, 23040, 0, 0, "auto", 30000}, "coop/16x12");
    for(const ck_tile::index_t ps : {16, 64, 128})
        for(const ck_tile::index_t w : {1920, 2048})
            for(const ck_tile::index_t ctx : {8191, 8180})
                add("bound_p" + std::to_string(ps) + "_w" + std::to_string(w) + "_s" +
                        std::to_string(ctx),
                    {1, 16, 1, {ctx}, w, 0, 0, "auto", 8192, ps},
                    w == 1920 ? "coop/16" : "coop/11");
    // cache pages of 64 and 128 tokens: windows starting inside a cache page, partial last
    // cache pages, both layouts, several clusters, a longer launch bound
    for(const ck_tile::index_t ps : {64, 128})
    {
        const auto name = [&](const char* n) { return "p" + std::to_string(ps) + "_" + n; };
        add(name("swa_b2_s8192"), {2, 16, 1, {8192}, 2048, 0, 0, "auto", 0, ps}, "coop/11");
        add(name("swa_b3_s1000_w300"), {3, 16, 1, {1000}, 300, 0, 0, "auto", 0, ps}, "coop/6");
        add(name("swa_b1_s5"), {1, 16, 1, {5}, 2048, 0, 0, "auto", 0, ps}, "coop/1");
        add(name("swa_s0_to_3"), {4, 16, 1, {0, 1, 2, 3}, 2048, 0, 0, "auto", 0, ps}, "coop/1");
        for(const ck_tile::index_t w : {1, 17})
            add(name("swa_w") + std::to_string(w),
                {3, 16, 1, {1000, 77, 3}, w, 0, 0, "auto", 0, ps},
                "coop/1");
        add(name("swa_mixed"),
            {4, 16, 1, {9000, 100, 2048, 3333}, 2048, 0, 0, "auto", 0, ps},
            "coop/11");
        add(name("swa_wave"), {2, 16, 1, {5000}, 2048, 0, 0, "wave", 0, ps}, "wave/11");
        add(name("full_b8_s8192"), {8, 16, 1, {8192}, 0, 0, 0, "auto", 0, ps}, "wave/16");
        add(name("full_b1_s20000"), {1, 16, 1, {20000}, 0, 0, 0, "auto", 0, ps}, "coop/16x10");
        add(name("groups_forced"), {2, 16, 2, {6000}, 2048, 30, 0, "auto", 0, ps}, "coop/9x2");
        add(name("gqa_16_2"), {2, 16, 2, {4000}, 2048, 0, 0, "auto", 0, ps}, "coop/11");
        add(name("gqa_32_4_wave"), {2, 32, 4, {999}, 2048, 0, 0, "wave", 0, ps}, "wave/16");
        add(name("bound_full_b8"), {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072, ps}, "wave/16x3");
    }
    // fp16 Q and output: direct output, both layouts within a cluster, several clusters
    add("fp16_one_split", {2, 16, 1, {700}, 2048, 1, 0, "auto", 0, 16, true}, "coop/1");
    add("fp16_swa_b8_s8192", {8, 16, 1, {8192}, 2048, 0, 0, "auto", 0, 16, true}, "coop/11");
    add("fp16_full_b8_s8192", {8, 16, 1, {8192}, 0, 0, 0, "auto", 0, 16, true}, "wave/16");
    add("fp16_full_b1_s20000", {1, 16, 1, {20000}, 0, 0, 0, "auto", 0, 16, true}, "coop/16x10");
    add("fp16_p64_bound_full_b8",
        {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072, 64, true},
        "wave/16x3");
    return p;
}

class FwdDecodePagedKV : public ::testing::TestWithParam<Param>
{
};

TEST_P(FwdDecodePagedKV, MatchesReference)
{
    if(ck_tile::get_device_name().rfind("gfx125", 0) != 0)
        GTEST_SKIP() << "gfx125 only";
    // one timed launch: also replays a captured graph and compares its output
    const auto r = fmha_fwd_decode_pagedkv_runner::run(GetParam().c, true, 0, 1);
    ASSERT_TRUE(r.error.empty()) << r.error;
    EXPECT_EQ(layout_of(r.config), GetParam().layout);
    EXPECT_TRUE(r.graph_same) << "graph replay output differs from a stream launch";
    EXPECT_TRUE(r.valid) << describe(r);
}

// every key inside the window counted once, none outside it
TEST_P(FwdDecodePagedKV, CountsEveryKeyOnce)
{
    if(ck_tile::get_device_name().rfind("gfx125", 0) != 0)
        GTEST_SKIP() << "gfx125 only";
    Case c       = GetParam().c;
    c.needles    = true;
    const auto r = fmha_fwd_decode_pagedkv_runner::run(c, true);
    ASSERT_TRUE(r.error.empty()) << r.error;
    EXPECT_TRUE(r.valid) << describe(r);
}

// fmha_fwd_decode_pagedkv_max_workspace_bytes bounds every selected configuration and grows with
// the kv heads (host only: callers size one workspace buffer with it)
TEST(FwdDecodePagedKVHost, MaxWorkspaceBoundsSelector)
{
    std::vector<ck_tile::index_t> lengths;
    for(ck_tile::index_t s = 0; s <= 40000; s += 16)
        lengths.push_back(s);
    for(ck_tile::index_t s = 40000; s <= 1 << 20; s += s / 7)
        lengths.push_back(s);
    size_t checked = 0;
    // cap: the workgroups a cluster holds on the device
    for(const ck_tile::index_t cap : {16, 15, 8, 4, 2, 1})
        for(const ck_tile::index_t nhead_k : {1, 8})
            for(const ck_tile::index_t batch :
                {1,  2,  3,  4,   5,   8,   13,  16,  24,  31,  32,  33,  48,
                 64, 65, 96, 128, 129, 160, 192, 255, 256, 384, 480, 481, 512})
            {
                const size_t bound =
                    ck_tile::fmha_fwd_decode_pagedkv_max_workspace_bytes(batch, nhead_k, cap);
                for(const ck_tile::index_t ps : {16, 64, 128})
                    for(const ck_tile::index_t w : {0, 1, 300, 2048, 4000, 23040})
                        for(const ck_tile::index_t s : lengths)
                        {
                            const auto c = ck_tile::fmha_fwd_decode_pagedkv_config(
                                batch, nhead_k, s, w, ps, cap);
                            const size_t need = ck_tile::fmha_fwd_decode_pagedkv_workspace_bytes(
                                batch, nhead_k, c.num_splits, c.num_groups);
                            ASSERT_LE(need, bound)
                                << "cap " << cap << " b " << batch << " h_k " << nhead_k << " s_k "
                                << s << " window " << w << " page " << ps;
                            ASSERT_LE(c.num_splits, cap);
                            // clusters merge only what a cluster of several splits merged
                            ASSERT_TRUE(c.num_groups == 1 || c.num_splits >= 2);
                            ++checked;
                        }
            }
    EXPECT_GT(checked, 0u);
    for(ck_tile::index_t heads = 1; heads < 2048; ++heads)
        ASSERT_LE(ck_tile::fmha_fwd_decode_pagedkv_max_workspace_bytes(heads, 1),
                  ck_tile::fmha_fwd_decode_pagedkv_max_workspace_bytes(heads + 1, 1))
            << heads;
}

INSTANTIATE_TEST_SUITE_P(Gfx125,
                         FwdDecodePagedKV,
                         ::testing::ValuesIn(params()),
                         [](const auto& test_info) { return test_info.param.name; });

} // namespace
