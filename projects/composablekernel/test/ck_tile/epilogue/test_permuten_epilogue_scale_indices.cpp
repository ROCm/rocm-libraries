// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/ops/epilogue/permuten_epilogue.hpp"

namespace {

using Problem = ck_tile::PermuteNEpilogueProblem<ck_tile::fp8_t,
                                                 ck_tile::fp8_t,
                                                 ck_tile::tuple<>,
                                                 float,
                                                 ck_tile::half_t,
                                                 ck_tile::tuple<>,
                                                 ck_tile::tensor_layout::gemm::RowMajor,
                                                 ck_tile::element_wise::PassThrough,
                                                 128,
                                                 256,
                                                 1,
                                                 4,
                                                 16,
                                                 16,
                                                 128,
                                                 false>;

using Epilogue = ck_tile::PermuteNEpilogue<Problem>;

TEST(PermuteNEpilogueScaleIndices, SelectsBroadcastCoordinates)
{
    static_assert(Epilogue::NRepeat > 1, "The regression requires multiple N repeats");

    for(ck_tile::index_t m_lane = 0; m_lane < 4; ++m_lane)
    {
        const auto first = Epilogue::GetScaleThreadBufferIndices(m_lane, 0);

        for(ck_tile::index_t n_idx = 0; n_idx < Epilogue::NRepeat; ++n_idx)
        {
            const auto indices = Epilogue::GetScaleThreadBufferIndices(m_lane, n_idx);

            // The M scale is broadcast across N, and the N scale is broadcast across M.
            EXPECT_EQ(indices.m, first.m);
            EXPECT_EQ(indices.n, n_idx);
        }
    }
}

TEST(PermuteNEpilogueScaleIndices, AdvancesPerTokenScaleForEveryMRepeat)
{
    static_assert(Epilogue::MRepeat > 1, "The regression requires multiple M repeats");
    EXPECT_EQ(Epilogue::GetScaleMWindowStep(), Epilogue::MPerXdl * Epilogue::MWave);
}

} // namespace
