// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Checks of the explicit LDS K padding specs used by the gfx125 GEMM LDS layouts. The gtests run
// on the host: resolved layout geometry, TDM pad codes and a simple LDS bank model. The gfx125
// device pass additionally checks the real LDS descriptors and TDM codes of the TDM, universal and
// async policies at compile time, so a layout copy that ignores the spec fails to build.

#include <algorithm>
#include <array>
#include <type_traits>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

using ck_tile::index_t;

namespace {

template <typename Resolved>
constexpr std::array<index_t, 3> tdm_codes()
{
    constexpr auto config = Resolved::get_padding_config();
    return {static_cast<index_t>(config.at(ck_tile::number<0>{})),
            static_cast<index_t>(config.at(ck_tile::number<1>{})),
            static_cast<index_t>(config.at(ck_tile::number<2>{}))};
}

// Worst-case bank conflict when 16 lanes each read 16 B from the same column of 16 consecutive
// rows, with 64 banks of 4 B (gfx125). Returns 1 when conflict-free.
index_t bank_conflict_ways(index_t row_stride_bytes)
{
    constexpr index_t n_banks = 64;
    constexpr index_t n_lanes = 16;
    std::array<index_t, n_banks> hits{};
    for(index_t lane = 0; lane < n_lanes; ++lane)
    {
        const index_t first_bank = (lane * row_stride_bytes / 4) % n_banks;
        for(index_t dw = 0; dw < 4; ++dw)
            ++hits[(first_bank + dw) % n_banks];
    }
    return *std::max_element(hits.begin(), hits.end());
}

struct TraitsWithoutSpec
{
};

struct ProblemWithoutSpec
{
    using Traits = TraitsWithoutSpec;
};

template <typename Traits_>
struct ProblemWith
{
    using Traits = Traits_;
};

} // namespace

#if defined(__gfx125__)
namespace device_checks {

using namespace ck_tile;
using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

template <index_t K>
using Shape      = TileGemmShape<sequence<256, 256, K>, sequence<2, 4, 1>, sequence<16, 16, 32>>;
using BaseTraits = TileGemmUniversalTraits<false, false, false, true, Row, Col, Row>;
template <index_t K, typename Traits>
using Prob = UniversalGemmPipelineProblem<bf16_t,
                                          bf16_t,
                                          float,
                                          Shape<K>,
                                          Traits,
                                          GemmPipelineScheduler::Intrawave>;

template <typename Policy, typename P>
__device__ constexpr index_t offset_a(index_t m, index_t k)
{
    constexpr auto desc = Policy::template MakeALdsBlockDescriptor<P>();
    return desc.calculate_offset(make_tuple(m, k));
}

template <typename Policy, typename P>
__device__ constexpr index_t offset_b(index_t n, index_t k)
{
    constexpr auto desc = Policy::template MakeBLdsBlockDescriptor<P>();
    return desc.calculate_offset(make_tuple(n, k));
}

using PAuto   = Prob<64, BaseTraits>;
using PRow8   = Prob<64, WithLdsPad<BaseTraits, LdsKPadPerRow<8>>>;
using PRow24  = Prob<32, WithLdsPad<BaseTraits, LdsKPadPerRow<24>>>;
using PInterv = Prob<64, WithLdsPad<BaseTraits, LdsKPadPerInterval<8, 128>>>;
using PNone   = Prob<64, WithLdsPad<BaseTraits, LdsKPadNone>>;
using PMixed  = Prob<64, WithLdsPad<BaseTraits, LdsKPadPerRow<8>, LdsKPadPerRow<16>>>;
// 192 B K rows: not a TDM-encodable interval, but a valid ds_write layout
using PRow8K96 = Prob<96, WithLdsPad<BaseTraits, LdsKPadPerRow<8>>>;

template <typename Policy>
__device__ void check_layouts()
{
    // Auto: 16 B after every two 128 B rows
    static_assert(offset_a<Policy, PAuto>(1, 0) == 64 && offset_a<Policy, PAuto>(2, 0) == 136);
    static_assert(offset_a<Policy, PInterv>(1, 0) == 64 && offset_a<Policy, PInterv>(2, 0) == 136);
    static_assert(offset_a<Policy, PRow8>(1, 0) == 72 && offset_a<Policy, PRow8>(2, 8) == 152);
    static_assert(offset_b<Policy, PRow8>(1, 0) == 72);
    static_assert(offset_a<Policy, PRow24>(1, 0) == 56 && offset_a<Policy, PRow24>(3, 16) == 184);
    static_assert(offset_a<Policy, PNone>(1, 0) == 64 && offset_a<Policy, PNone>(3, 8) == 200);
    static_assert(offset_a<Policy, PMixed>(1, 0) == 72 && offset_b<Policy, PMixed>(1, 0) == 80);
}

__global__ void check_policies()
{
    using TdmPolicy       = GemmPipelineAgBgCrCompTDMDefaultPolicy<false>;
    using UniversalPolicy = GemmPipelineAgBgCrDefaultPolicy;
    using AsyncPolicy     = GemmPipelineAgBgCrCompAsyncDefaultPolicy<>;
    check_layouts<TdmPolicy>();
    check_layouts<UniversalPolicy>();
    check_layouts<AsyncPolicy>();

    // Non-TDM pipelines do not need TDM-encodable pad codes
    static_assert(offset_a<UniversalPolicy, PRow8K96>(1, 0) == 104 &&
                  offset_b<UniversalPolicy, PRow8K96>(2, 8) == 216);

    // One stage of A and B; the trailing pad of each buffer is not allocated
    static_assert(TdmPolicy::GetSmemSize<PAuto>() == 69600);
    static_assert(TdmPolicy::GetSmemSize<PRow8>() == 2 * (256 * 72 - 8) * 2);
    static_assert(TdmPolicy::GetSmemSize<PRow24>() == 2 * (256 * 56 - 24) * 2);
    static_assert(TdmPolicy::GetSmemSize<PInterv>() == TdmPolicy::GetSmemSize<PAuto>());
    static_assert(UniversalPolicy::GetSmemSize<PRow8>() == TdmPolicy::GetSmemSize<PRow8>());

    // TDM codes consumed by the TDM pipelines
    constexpr auto row8 = TdmPolicy::GetLdsPaddingConfig<PRow8, true>();
    static_assert(row8[number<0>{}] && row8[number<1>{}] == 3 && row8[number<2>{}] == 4);
    constexpr auto row24 = TdmPolicy::GetLdsPaddingConfig<PRow24, false>();
    static_assert(row24[number<0>{}] && row24[number<1>{}] == 11 && row24[number<2>{}] == 3);
    constexpr auto automatic = TdmPolicy::GetLdsPaddingConfig<PAuto, true>();
    constexpr auto interval  = TdmPolicy::GetLdsPaddingConfig<PInterv, true>();
    static_assert(automatic[number<1>{}] == interval[number<1>{}] &&
                  automatic[number<2>{}] == interval[number<2>{}]);
    constexpr auto none = TdmPolicy::GetLdsPaddingConfig<PNone, true>();
    static_assert(!none[number<0>{}]);
}

} // namespace device_checks
#endif

TEST(GemmLdsPadding, SpecDefaultsToAuto)
{
    using ck_tile::lds_pad_spec_t;
    using ck_tile::LdsKPadAuto;
    EXPECT_TRUE((std::is_same_v<lds_pad_spec_t<ProblemWithoutSpec, true>, LdsKPadAuto>));
    EXPECT_TRUE((std::is_same_v<lds_pad_spec_t<ProblemWithoutSpec, false>, LdsKPadAuto>));

    using Traits =
        ck_tile::WithLdsPad<TraitsWithoutSpec, ck_tile::LdsKPadPerRow<8>, ck_tile::LdsKPadNone>;
    EXPECT_TRUE(
        (std::is_same_v<lds_pad_spec_t<ProblemWith<Traits>, true>, ck_tile::LdsKPadPerRow<8>>));
    EXPECT_TRUE((std::is_same_v<lds_pad_spec_t<ProblemWith<Traits>, false>, ck_tile::LdsKPadNone>));

    // B defaults to the A spec
    using TraitsSame = ck_tile::WithLdsPad<TraitsWithoutSpec, ck_tile::LdsKPadPerRow<16>>;
    EXPECT_TRUE((std::is_same_v<lds_pad_spec_t<ProblemWith<TraitsSame>, false>,
                                ck_tile::LdsKPadPerRow<16>>));
}

TEST(GemmLdsPadding, AutoLayoutForwardsPolicyValues)
{
    using Layout = ck_tile::LdsKPadLayout<ck_tile::LdsKPadAuto, ck_tile::bf16_t, 256, 64, 2, 8>;
    EXPECT_EQ(Layout::rows_per_group, 2);
    EXPECT_EQ(Layout::pad_elems, 8);
}

TEST(GemmLdsPadding, ExplicitLayoutUsesResolvedGroup)
{
    // 8 rows of 64 bf16 per group; the group must divide the tile rows (static_assert)
    using Layout =
        ck_tile::LdsKPadLayout<ck_tile::LdsKPadPerInterval<8, 512>, ck_tile::bf16_t, 256, 64, 2, 8>;
    EXPECT_EQ(Layout::rows_per_group, 8);
    EXPECT_EQ(Layout::pad_elems, 8);
}

TEST(GemmLdsPadding, PerRow8Bf16K64)
{
    // rocKE Opt-3 (lds_k_pad=8, tile_k=64): row stride 0x90, TDM group-1 word 0 0x07110000
    using R = ck_tile::ResolvedLdsPad<ck_tile::bf16_t, 64, ck_tile::LdsKPadPerRow<8>>;
    EXPECT_EQ(R::rows_per_group, 1);
    EXPECT_EQ(R::group_stride_elems, 72);
    EXPECT_EQ(R::group_stride_elems * 2, 0x90);
    EXPECT_EQ(R::pad_dw, 4);
    EXPECT_EQ(R::interval_dw, 32);
    EXPECT_TRUE(R::pad_enable);
    constexpr auto codes = tdm_codes<R>();
    EXPECT_EQ(codes[0], 1);
    EXPECT_EQ(codes[1], 3); // amount code: 4 dwords
    EXPECT_EQ(codes[2], 4); // interval code: 32 dwords
    EXPECT_EQ(bank_conflict_ways(R::group_stride_elems * 2), 1);
}

TEST(GemmLdsPadding, PerRow24Bf16K32)
{
    // rocKE Opt-2-1 (lds_k_pad=24, tile_k=32): row stride 0x70
    using R = ck_tile::ResolvedLdsPad<ck_tile::bf16_t, 32, ck_tile::LdsKPadPerRow<24>>;
    EXPECT_EQ(R::group_stride_elems, 56);
    EXPECT_EQ(R::group_stride_elems * 2, 0x70);
    EXPECT_EQ(R::pad_dw, 12);
    EXPECT_EQ(R::interval_dw, 16);
    constexpr auto codes = tdm_codes<R>();
    EXPECT_EQ(codes[0], 1);
    EXPECT_EQ(codes[1], 11);
    EXPECT_EQ(codes[2], 3);
    EXPECT_EQ(bank_conflict_ways(R::group_stride_elems * 2), 1);
}

TEST(GemmLdsPadding, PerRow16Bf16K64)
{
    using R = ck_tile::ResolvedLdsPad<ck_tile::bf16_t, 64, ck_tile::LdsKPadPerRow<16>>;
    EXPECT_EQ(R::group_stride_elems, 80);
    constexpr auto codes = tdm_codes<R>();
    EXPECT_EQ(codes[1], 7);
    EXPECT_EQ(codes[2], 4);
    EXPECT_EQ(bank_conflict_ways(R::group_stride_elems * 2), 2);
}

TEST(GemmLdsPadding, PerIntervalMatchesAutoBf16K64)
{
    // 16 B after every 256 B is the Auto layout for bf16 at K=64 (two rows per group)
    using R = ck_tile::ResolvedLdsPad<ck_tile::bf16_t, 64, ck_tile::LdsKPadPerInterval<8, 128>>;
    EXPECT_EQ(R::rows_per_group, 2);
    EXPECT_EQ(R::group_stride_elems, 136);
    EXPECT_EQ(R::group_stride_elems * 2, 0x110);
    constexpr auto codes = tdm_codes<R>();
    EXPECT_EQ(codes[0], 1);
    EXPECT_EQ(codes[1], 3);
    EXPECT_EQ(codes[2], 5); // 64 dwords: log2(n_lds_banks) - 1 on gfx125
}

TEST(GemmLdsPadding, NoneBf16K64)
{
    using R = ck_tile::ResolvedLdsPad<ck_tile::bf16_t, 64, ck_tile::LdsKPadNone>;
    EXPECT_EQ(R::rows_per_group, 1);
    EXPECT_EQ(R::pad_elems, 0);
    EXPECT_EQ(R::group_stride_elems, 64);
    EXPECT_FALSE(R::pad_enable);
    constexpr auto codes = tdm_codes<R>();
    EXPECT_EQ(codes[0], 0);
    EXPECT_EQ(codes[1], 0);
    EXPECT_EQ(codes[2], 0);
    EXPECT_EQ(bank_conflict_ways(R::group_stride_elems * 2), 8);
}

TEST(GemmLdsPadding, PerRow16Fp8K128)
{
    using R = ck_tile::ResolvedLdsPad<ck_tile::fp8_t, 128, ck_tile::LdsKPadPerRow<16>>;
    EXPECT_EQ(R::group_stride_elems, 144);
    EXPECT_EQ(R::pad_dw, 4);
    EXPECT_EQ(R::interval_dw, 32);
    EXPECT_EQ(bank_conflict_ways(R::group_stride_elems), 1);
}

TEST(GemmLdsPadding, BankModelTable)
{
    // bf16 16-lane x 16 B column reads, 64 banks: pad {0, 8, 16, 24} at tile_k {32, 64}
    const std::array<index_t, 4> pads{0, 8, 16, 24};
    const std::array<index_t, 4> expect_k32{4, 1, 2, 1};
    const std::array<index_t, 4> expect_k64{8, 1, 2, 1};
    for(std::size_t i = 0; i < pads.size(); ++i)
    {
        EXPECT_EQ(bank_conflict_ways((32 + pads[i]) * 2), expect_k32[i]) << "pad " << pads[i];
        EXPECT_EQ(bank_conflict_ways((64 + pads[i]) * 2), expect_k64[i]) << "pad " << pads[i];
    }
}
