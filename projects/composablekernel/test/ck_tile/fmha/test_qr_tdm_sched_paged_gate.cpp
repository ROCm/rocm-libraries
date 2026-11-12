// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Pure host test: fmha_fwd_pagedkv_tdm_sched_applicable is a plain bool function, no GPU
// needed. Pins the gate's behavior directly, independent of whether anything in the
// generated dispatch chain currently reaches it or happens to coincide with the runner's
// own blanket page_block_size % 128 check (kN0 == 128 is only the one currently-shipped
// geometry; the gate itself takes kN0 as a parameter and must hold for any future one).

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "gtest/gtest.h"

namespace {

using ck_tile::index_t;

struct GateCase
{
    index_t page_block_size;
    bool is_gappy;
    index_t kN0;
    bool expected;
};

} // namespace

TEST(FmhaFwdPagedKVTdmSchedGate, MatchesExpectedForEachCase)
{
    constexpr GateCase cases[] = {
        // page_block_size, is_gappy, kN0, expected
        {192, false, 128, false}, // not a multiple of kN0
        {128, false, 128, true},  // exactly one page per tile
        {256, false, 128, true},  // two pages per tile, still evenly divides
        {128, false, 256, false}, // page smaller than the tile
        {0, false, 128, false},   // no paging at all
        {64, false, 128, false},  // page smaller than the tile, does not divide
        {192, false, 64, true},   // diverges from the hardcoded "% 128 == 0" check: true here,
                                  // whereas {192, 128} above is false for the same page_block_size
        {256, true, 128, false},  // gappy is declined even when the page size fits
        {192, true, 128, false},  // and when it does not
    };

    for(const auto& c : cases)
    {
        // fmha_fwd_pagedkv_tdm_sched_applicable lives in the global namespace (fmha_fwd.hpp
        // does not wrap its declarations in namespace ck_tile), same as every call site in
        // the generated dispatch chain.
        EXPECT_EQ(fmha_fwd_pagedkv_tdm_sched_applicable(c.page_block_size, c.is_gappy, c.kN0),
                  c.expected)
            << "page_block_size=" << c.page_block_size << " is_gappy=" << c.is_gappy
            << " kN0=" << c.kN0;
    }
}
