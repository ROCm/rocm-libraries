// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// FmhaDecodePaged5dKernel (gfx1250) against the CPU reference of fmha_decode_paged5d_runner.

#include <gtest/gtest.h>

#include "fmha_decode_paged5d_runner.hpp"

namespace {

using fmha_decode_paged5d::Case;

struct Param
{
    const char* name;
    Case c;
};

std::vector<Param> params()
{
    std::vector<Param> p;
    const auto add = [&](const char* name, Case c) { p.push_back({name, c}); };
    // window 2048 (Muse decode shape) at short, medium and long contexts
    add("swa_b1_s1024", {1, 16, 1, {1024}, 2048});
    add("swa_b8_s8192", {8, 16, 1, {8192}, 2048});
    add("swa_b2_s73728", {2, 16, 1, {73728}, 2048});
    // partial last page, sequences shorter than one step, one page
    add("swa_b3_s1000", {3, 16, 1, {1000}, 2048});
    add("swa_b2_s77", {2, 16, 1, {77}, 2048});
    add("swa_b1_s5", {1, 16, 1, {5}, 2048});
    // different lengths in one batch: some splits of the shorter sequences are empty
    add("swa_mixed", {4, 16, 1, {9000, 100, 2048, 3333}, 2048});
    // no window, both layouts and the long-split wave layout
    add("full_b2_s3000", {2, 16, 1, {3000}, 0});
    add("full_b1_s20000", {1, 16, 1, {20000}, 0});
    add("full_coop", {2, 16, 1, {5000}, 0, 0, 0, "coop"});
    add("swa_wave", {2, 16, 1, {5000}, 2048, 0, 0, "wave"});
    // one split (direct output), many small splits
    add("one_split", {2, 16, 1, {700}, 2048, 1});
    add("fifteen_splits", {2, 16, 1, {960}, 2048, 15});
    // other GQA shapes: fewer than 16 rows, several kv heads
    add("gqa_8_1", {2, 8, 1, {2500}, 2048});
    add("gqa_16_2", {2, 16, 2, {4000}, 2048});
    add("gqa_32_4_wave", {2, 32, 4, {999}, 2048, 0, 0, "wave"});
    add("gqa_6_3", {2, 6, 3, {1500}, 512});
    return p;
}

class DecodePaged5d : public ::testing::TestWithParam<Param>
{
};

TEST_P(DecodePaged5d, MatchesReference)
{
    if(ck_tile::get_device_name() != "gfx1250")
        GTEST_SKIP() << "gfx1250 only";
    const auto r = fmha_decode_paged5d::run(GetParam().c, true);
    ASSERT_TRUE(r.error.empty()) << r.error;
    EXPECT_TRUE(r.valid) << "max abs err " << r.max_err << " (max |ref| " << r.max_ref << "), cos "
                         << r.cosine << ", splits " << r.config.num_splits << ", pages/split "
                         << r.config.pages_per_split << (r.config.wave_split ? ", wave" : ", coop");
}

INSTANTIATE_TEST_SUITE_P(Gfx1250,
                         DecodePaged5d,
                         ::testing::ValuesIn(params()),
                         [](const auto& test_info) { return std::string(test_info.param.name); });

} // namespace
