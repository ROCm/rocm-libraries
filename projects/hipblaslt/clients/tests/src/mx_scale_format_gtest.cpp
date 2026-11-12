// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host test for the MX scale layout rule every kernel-selection path in
// tensile_host.cpp and rocblaslt_grouped_matmul.cpp shares, and for the
// widening search that keeps the rule from shortening a top-N request.

#include <gtest/gtest.h>

#include "../../../library/src/amd_detail/rocblaslt/include/rocblaslt-types.h"
#include "../../../library/src/amd_detail/rocblaslt/src/include/mx_scale_format.hpp"

#include <algorithm>
#include <vector>

namespace
{
    using Scale  = RocblasltContractionProblem::ScalingFormat;
    using Need   = rocblaslt_mx_scale_need;
    using Layout = TensileLite::MXScaleFormat;

    constexpr int kNoSwizzle       = static_cast<int>(Layout::NoSwizzle);
    constexpr int kHostPreSwizzle  = static_cast<int>(Layout::HostPreSwizzle);
    constexpr int kInMemorySwizzle = static_cast<int>(Layout::InMemorySwizzle);

    TEST(MXScaleFormat_smoke, packed_scales_never_take_a_preswizzled_kernel)
    {
        for(auto packed : {Scale::Block_32_UE8M0,
                           Scale::Block_16_UE8M0,
                           Scale::Block_32_UE4M3,
                           Scale::Block_16_UE4M3,
                           Scale::Block_32_UE5M3,
                           Scale::Block_16_UE5M3})
        {
            SCOPED_TRACE(static_cast<int>(packed));
            const Need need = rocblaslt_mx_scale_need_for(packed, packed);
            EXPECT_EQ(need, Need::Natural);
            EXPECT_TRUE(rocblaslt_mx_scale_layout_ok(need, kNoSwizzle));
            EXPECT_TRUE(rocblaslt_mx_scale_layout_ok(need, kInMemorySwizzle));
            EXPECT_FALSE(rocblaslt_mx_scale_layout_ok(need, kHostPreSwizzle));
        }
    }

    TEST(MXScaleFormat_smoke, mx_scales_refuse_unknown_layouts)
    {
        for(auto need : {Need::Natural, Need::PreSwizzled})
            for(int layout : {-1, 3, 255})
                EXPECT_FALSE(rocblaslt_mx_scale_layout_ok(need, layout)) << layout;
        EXPECT_TRUE(rocblaslt_mx_scale_layout_ok(Need::Any, 3));
    }

    TEST(MXScaleFormat_smoke, preswizzled_scales_take_only_a_preswizzled_kernel)
    {
        const Need need = rocblaslt_mx_scale_need_for(Scale::Block_32_UE8M0_32_8_EXT,
                                                      Scale::Block_32_UE8M0_32_8_EXT);
        EXPECT_EQ(need, Need::PreSwizzled);
        EXPECT_TRUE(rocblaslt_mx_scale_layout_ok(need, kHostPreSwizzle));
        EXPECT_FALSE(rocblaslt_mx_scale_layout_ok(need, kNoSwizzle));
        EXPECT_FALSE(rocblaslt_mx_scale_layout_ok(need, kInMemorySwizzle));
    }

    TEST(MXScaleFormat_smoke, sides_and_groups_combine)
    {
        EXPECT_EQ(rocblaslt_mx_scale_need_for(Scale::Block_32_UE8M0, Scale::None), Need::Natural);
        EXPECT_EQ(rocblaslt_mx_scale_need_for(Scale::Scalar, Scale::Block_32_UE8M0_32_8_EXT),
                  Need::PreSwizzled);
        EXPECT_EQ(
            rocblaslt_mx_scale_need_for(Scale::Block_32_UE8M0, Scale::Block_32_UE8M0_32_8_EXT),
            Need::Conflict);
        EXPECT_EQ(rocblaslt_mx_scale_need_combine(Need::Natural, Need::PreSwizzled),
                  Need::Conflict);
        EXPECT_EQ(rocblaslt_mx_scale_need_combine(Need::Conflict, Need::Any), Need::Conflict);
        for(int layout : {kNoSwizzle, kHostPreSwizzle, kInMemorySwizzle})
            EXPECT_FALSE(rocblaslt_mx_scale_layout_ok(Need::Conflict, layout));
    }

    TEST(MXScaleFormat_smoke, non_mx_scales_take_any_kernel)
    {
        for(auto scale : {Scale::None, Scale::Scalar, Scale::Vector, Scale::Block_32_BF16})
        {
            SCOPED_TRACE(static_cast<int>(scale));
            const Need need = rocblaslt_mx_scale_need_for(scale, scale);
            EXPECT_EQ(need, Need::Any);
            for(int layout : {kNoSwizzle, kHostPreSwizzle, kInMemorySwizzle})
                EXPECT_TRUE(rocblaslt_mx_scale_layout_ok(need, layout));
        }
    }

    // A ranked library: each entry is a kernel's MX scale layout, best first.
    struct RankedSearch
    {
        std::vector<int> ranked;
        std::vector<int> requests;

        std::vector<int> operator()(int count)
        {
            requests.push_back(count);
            const size_t n = std::min(ranked.size(), static_cast<size_t>(count));
            return {ranked.begin(), ranked.begin() + n};
        }
    };

    auto dropFor(Need need)
    {
        return [need](int layout) { return !rocblaslt_mx_scale_layout_ok(need, layout); };
    }

    TEST(MXScaleFormat_smoke, top_one_reaches_past_a_dropped_first_rank)
    {
        RankedSearch search{{kHostPreSwizzle, kNoSwizzle, kNoSwizzle}};
        auto         kept = rocblaslt_find_top_kept(search, 1, dropFor(Need::Natural));
        EXPECT_EQ(kept, (std::vector<int>{kNoSwizzle}));
        EXPECT_EQ(search.requests, (std::vector<int>{1, 2}));
    }

    // The request doubles while every candidate is dropped, and the result is
    // trimmed to the request.
    TEST(MXScaleFormat_smoke, widening_doubles_and_trims_to_the_request)
    {
        RankedSearch search{{kHostPreSwizzle, kHostPreSwizzle, kNoSwizzle, kInMemorySwizzle}};
        auto         kept = rocblaslt_find_top_kept(search, 1, dropFor(Need::Natural));
        EXPECT_EQ(kept, (std::vector<int>{kNoSwizzle}));
        EXPECT_EQ(search.requests, (std::vector<int>{1, 2, 4}));
    }

    TEST(MXScaleFormat_smoke, widening_keeps_rank_order)
    {
        RankedSearch search{
            {kHostPreSwizzle, kNoSwizzle, kHostPreSwizzle, kInMemorySwizzle, kNoSwizzle}};
        auto kept = rocblaslt_find_top_kept(search, 3, dropFor(Need::Natural));
        EXPECT_EQ(kept, (std::vector<int>{kNoSwizzle, kInMemorySwizzle, kNoSwizzle}));
        EXPECT_EQ(search.requests, (std::vector<int>{3, 5}));
    }

    // With no compatible kernel, the search doubles until the library runs out:
    // 1,500 kernels take 12 searches.
    TEST(MXScaleFormat_smoke, no_compatible_kernel_ends_when_the_library_runs_out)
    {
        RankedSearch search{std::vector<int>(1500, kHostPreSwizzle)};
        auto         kept = rocblaslt_find_top_kept(search, 1, dropFor(Need::Natural));
        EXPECT_TRUE(kept.empty());
        std::vector<int> expected;
        for(int request = 1; request <= 2048; request *= 2)
            expected.push_back(request);
        EXPECT_EQ(search.requests, expected);
    }

    TEST(MXScaleFormat_smoke, nothing_dropped_searches_once)
    {
        RankedSearch search{{kHostPreSwizzle, kNoSwizzle}};
        auto         kept = rocblaslt_find_top_kept(search, 1, dropFor(Need::Any));
        EXPECT_EQ(kept, (std::vector<int>{kHostPreSwizzle}));
        EXPECT_EQ(search.requests, (std::vector<int>{1}));
    }
}
