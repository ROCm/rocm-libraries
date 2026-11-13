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
    bool gappy_variant;
    index_t kN0;
    bool expected;
};

} // namespace

TEST(FmhaFwdPagedKVTdmSchedGate, MatchesExpectedForEachCase)
{
    constexpr GateCase cases[] = {
        // page_block_size, is_gappy, gappy_variant, kN0, expected
        {192, false, false, 128, false}, // not a multiple of kN0
        {128, false, false, 128, true},  // exactly one page per tile
        {256, false, false, 128, true},  // two pages per tile, still evenly divides
        {128, false, false, 256, false}, // page smaller than the tile
        {0, false, false, 128, false},   // no paging at all
        {64, false, false, 128, false},  // page smaller than the tile, does not divide
        {192, false, false, 64, true},   // diverges from the hardcoded "% 128 == 0" check: true
                                         // here, whereas {192, 128} above is false
        {256, true, true, 128, true},    // gappy runs the gappy variant
        {256, true, false, 128, false},  // and not the default one
        {256, false, true, 128, false},  // non-gappy does not run the gappy variant
        {192, true, true, 128, false},   // gappy still needs a page size that fits
    };

    for(const auto& c : cases)
    {
        // fmha_fwd_pagedkv_tdm_sched_applicable lives in the global namespace (fmha_fwd.hpp
        // does not wrap its declarations in namespace ck_tile), same as every call site in
        // the generated dispatch chain.
        EXPECT_EQ(fmha_fwd_pagedkv_tdm_sched_applicable(
                      c.page_block_size, c.is_gappy, c.gappy_variant, c.kN0),
                  c.expected)
            << "page_block_size=" << c.page_block_size << " is_gappy=" << c.is_gappy
            << " gappy_variant=" << c.gappy_variant << " kN0=" << c.kN0;
    }
}
