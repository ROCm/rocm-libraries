// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

using ::testing::TestWithParam;
using ::testing::Values;

namespace {

const ck_tile::stream_config stream_config{
    nullptr,
    true,
    1,
    0,
    1,
    true,
    false,
    1,
};

class FmhaFwdSplitKvGfx1100 : public TestWithParam<std::tuple<int, int, int>>
{
};

TEST_P(FmhaFwdSplitKvGfx1100, PagedSelectionAndValidation)
{
    if(ck_tile::get_device_name().compare(0, 7, "gfx1100") != 0)
        GTEST_SKIP() << "gfx1100-only split-KV regression";

    const auto [seqlen_q, num_splits, page_block_size] = GetParam();
    if(num_splits == 0)
        ASSERT_EQ(override_num_splits_if_necessary(1, 8, seqlen_q, 128, 0.0f, num_splits), 1);

    const auto result = fmha_fwd_run<FmhaFwdFp16>(mode_enum::batch,
                                                  1,
                                                  8,
                                                  8,
                                                  {seqlen_q},
                                                  {128},
                                                  128,
                                                  128,
                                                  0,
                                                  {-1},
                                                  {-1},
                                                  {},
                                                  {},
                                                  0,
                                                  true,
                                                  true,
                                                  0,
                                                  0,
                                                  true,
                                                  false,
                                                  page_block_size,
                                                  false,
                                                  "n",
                                                  0.0f,
                                                  0,
                                                  0,
                                                  false,
                                                  "0",
                                                  "n",
                                                  true,
                                                  num_splits,
                                                  "uf",
                                                  11939,
                                                  1,
                                                  0,
                                                  0,
                                                  stream_config);

    ASSERT_EQ(result, fwd_result::success);
}

INSTANTIATE_TEST_SUITE_P(PageAlignment,
                         FmhaFwdSplitKvGfx1100,
                         Values(std::tuple{128, 2, 80},
                                std::tuple{128, 2, 96},
                                std::tuple{128, 1, 16},
                                std::tuple{2048, 0, 16}));

} // namespace
