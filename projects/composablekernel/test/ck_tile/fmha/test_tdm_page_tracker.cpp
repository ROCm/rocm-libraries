// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host test of TdmPageTracker's address math: page of a row, page-local origin, page pointer,
// view choice and saturation past the last page. The window plumbing is device-only and is
// covered by test_qr_tdm_sched_paged.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/tdm_page_tracker.hpp"

namespace {

using ck_tile::index_t;

using Elem = ck_tile::half_t;

constexpr index_t kPage   = 4;                // rows per page
constexpr index_t kDim    = 2;                // elements per row
constexpr index_t kFixed  = 1;                // fixed_offset, in elements
constexpr index_t kStride = kPage * kDim + 3; // block_stride != kPage * kDim on purpose

auto make_view(index_t rows)
{
    return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        static_cast<const Elem*>(nullptr),
        ck_tile::make_tuple(rows, kDim),
        ck_tile::make_tuple(kDim, 1),
        ck_tile::number<1>{},
        ck_tile::number<1>{});
}

struct Fixture
{
    std::vector<Elem> pool = std::vector<Elem>(8 * kStride);
    // Logical page p lives in physical page table[p].
    std::vector<std::int32_t> table{2, 0, 3};
    index_t last_len = 2; // the last page holds 2 of its 4 rows

    auto tracker() const
    {
        return ck_tile::make_tdm_page_tracker<const Elem>(pool.data(),
                                                          kStride,
                                                          kFixed,
                                                          table.data(),
                                                          static_cast<index_t>(table.size()),
                                                          kPage,
                                                          make_view(kPage),
                                                          make_view(last_len));
    }
    const Elem* page_ptr(index_t logical_page) const
    {
        return pool.data() + table[logical_page] * kStride + kFixed;
    }
};

template <typename View>
index_t rows(const View& v)
{
    return v.get_tensor_descriptor().get_lengths()[ck_tile::number<0>{}];
}

} // namespace

TEST(TdmPageTracker, RowMapsToTablePageAndLocalOrigin)
{
    const Fixture f;
    const auto t = f.tracker();
    for(index_t row = 0; row < 10; ++row)
    {
        const index_t block = t.get_block_index({row, 0});
        EXPECT_EQ(block, row / kPage) << row;
        EXPECT_EQ(t.get_block_ptr(block), f.page_ptr(row / kPage)) << row;
        const auto local = t.to_local_window_origin(block, {row, 0});
        EXPECT_EQ(local[ck_tile::number<0>{}], row % kPage) << row;
        EXPECT_EQ(t.to_global_window_origin(block, local)[ck_tile::number<0>{}], row) << row;
    }
}

TEST(TdmPageTracker, OnlyTheLastPageUsesLastView)
{
    const Fixture f;
    const auto t = f.tracker();
    EXPECT_EQ(rows(t.view_of(0)), kPage);
    EXPECT_EQ(rows(t.view_of(1)), kPage);
    EXPECT_EQ(rows(t.view_of(2)), f.last_len);
}

// A look-ahead row past the end stays on the last page (the table is never read past its
// end) with a local origin at or past last_view's rows, so a TDM load there reads nothing.
TEST(TdmPageTracker, SaturatesPastLastPage)
{
    const Fixture f;
    const auto t = f.tracker();
    for(index_t row : {10, 12, 16, 100})
    {
        const index_t block = t.get_block_index({row, 0});
        EXPECT_EQ(block, 2) << row;
        EXPECT_EQ(t.get_block_ptr(block), f.page_ptr(2)) << row;
        const auto local = t.to_local_window_origin(block, {row, 0});
        EXPECT_GE(local[ck_tile::number<0>{}], f.last_len) << row;
        EXPECT_EQ(t.to_global_window_origin(block, local)[ck_tile::number<0>{}], row) << row;
    }
}
