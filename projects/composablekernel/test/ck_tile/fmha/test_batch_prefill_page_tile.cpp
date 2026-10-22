// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// gfx1250 batch-prefill kernel with a physical page size independent of the logical N128 tile.
// One binary per (page size, head dimension, mask[, page table]) tuple, each compiling exactly one
// kernel so the
// tested code object matches a single-kernel library build of the same tuple. Covered:
//   - the host launch contract (dynamic LDS residency request and CheckLaunch refusals; no GPU),
//   - forward accuracy against the independent FP32 reference over identity, permuted and aliased
//     page tables, partial pages, unreferenced pool rows filled with a large finite value, random
//     and marker-page V; identity and permuted pools of the same logical data must give
//     bitwise-equal outputs,
//   - bitwise run-to-run determinism at a device-filling shape, plus a sparse-marker V there.
// Known blind spot: at long KV (32768) a single wrong logical page under random V is below the
// accuracy contract; the marker patterns cover only their marked pages.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_batch_prefill_qr_tdm_page_tile_kernel.hpp"
#include "fmha_reference.hpp"
#include "fmha_test_common.hpp"

#if !defined(CK_TILE_FMHA_PAGE_TILE_TEST_PAGE) || \
    !defined(CK_TILE_FMHA_PAGE_TILE_TEST_HEAD_DIM) || !defined(CK_TILE_FMHA_PAGE_TILE_TEST_CAUSAL)
#error "the page size, head dimension and mask of the tested tuple must be defined"
#endif
// 0: vLLM 2D block table (default); 1: SGLang 1D page list (kv_indptr / kv_page_indices /
// kv_last_page_lens), built from the same per-sequence pages.
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_SGLANG
#define CK_TILE_FMHA_PAGE_TILE_TEST_SGLANG 0
#endif
// 1: FP16 Q/K/V/O instead of BF16.
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_FP16
#define CK_TILE_FMHA_PAGE_TILE_TEST_FP16 0
#endif
// 1: the kernel also writes the per-row LSE ([Hq, total_q] floats), checked against the reference.
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_LSE
#define CK_TILE_FMHA_PAGE_TILE_TEST_LSE 0
#endif

namespace {

namespace reference = ck_tile::test::reference;
using ck_tile::index_t;
constexpr bool kFp16 = CK_TILE_FMHA_PAGE_TILE_TEST_FP16 != 0;
using Data           = std::conditional_t<kFp16, ck_tile::half_t, ck_tile::bf16_t>;
using DataConfig     = std::conditional_t<kFp16, FmhaFwdFp16, FmhaFwdBf16>;
// Mantissa bits of the 16-bit type, for the accuracy contract (P is rounded to it).
constexpr int kMantissaBits = kFp16 ? 10 : 7;

constexpr index_t kPage = CK_TILE_FMHA_PAGE_TILE_TEST_PAGE;
static_assert(kPage == 16 || kPage == 128 || kPage == 256 || kPage == 1024);
constexpr bool kSglang = CK_TILE_FMHA_PAGE_TILE_TEST_SGLANG != 0;
constexpr bool kLse    = CK_TILE_FMHA_PAGE_TILE_TEST_LSE != 0;
constexpr auto kLookupTable =
    kSglang ? ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D
            : ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D;

// BF16 or FP16, vLLM 2D or SGLang 1D page table, linear [num_pages, P, Hkv, D] K/V pool, group
// mode, M128/N128.
template <index_t HeadDim, bool Causal>
struct Model
{
    using TypeConfig = FmhaFwdTypeConfig<DataConfig>;
    using Shape      = ck_tile::TileFmhaShape<ck_tile::sequence<128, 128, 32, HeadDim, 32, HeadDim>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              true,
                                              false>;
    using Traits     = ck_tile::TileFmhaBatchPrefillTraits<
            true,
            true,
            false,
            false,
            false,
            ck_tile::BlockAttentionBiasEnum::NO_BIAS,
            false,
            kLse,
            false,
            ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
            1,
            false,
            false,
            static_cast<int>(kPage),
            ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT,
            kLookupTable,
            ck_tile::BlockAttentionKVCacheLoadModeEnum::BUFFER_LOAD>;
    using Problem = ck_tile::BlockFmhaBatchPrefillPipelineProblem<
        typename TypeConfig::QDataType,
        typename TypeConfig::KDataType,
        typename TypeConfig::VDataType,
        typename TypeConfig::SaccDataType,
        typename TypeConfig::SMPLComputeDataType,
        typename TypeConfig::BiasDataType,
        typename TypeConfig::RandValOutputDataType,
        typename TypeConfig::LSEDataType,
        typename TypeConfig::PDataType,
        typename TypeConfig::OaccDataType,
        typename TypeConfig::ODataType,
        Shape,
        true,
        ck_tile::ComposedAttention<false, CK_TILE_FMHA_FWD_FAST_EXP2>,
        ck_tile::SimplifiedGenericAttentionMask<Causal>,
        false,
        static_cast<int>(kPage),
        Traits>;
    using Pipeline = ck_tile::BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedPageTile<Problem>;
    using Epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<typename TypeConfig::OaccDataType,
                                          typename TypeConfig::ODataType,
                                          true,
                                          false>>;
    using Kernel = ck_tile::FmhaBatchPrefillQrTdmPageTileKernel<Pipeline, Epilogue>;

    static constexpr index_t kHeadDim = HeadDim;
    static constexpr bool kCausal     = Causal;
};

static_assert(CK_TILE_FMHA_PAGE_TILE_TEST_HEAD_DIM == 64 ||
              CK_TILE_FMHA_PAGE_TILE_TEST_HEAD_DIM == 128);
using Models = ::testing::Types<
    Model<CK_TILE_FMHA_PAGE_TILE_TEST_HEAD_DIM, CK_TILE_FMHA_PAGE_TILE_TEST_CAUSAL != 0>>;

struct ModelNames
{
    template <typename M>
    static std::string GetName(int)
    {
        return "P" + std::to_string(kPage) + "D" + std::to_string(M::kHeadDim) +
               (M::kCausal ? "Causal" : "Dense") + (kSglang ? "Sglang" : "") + (kLse ? "Lse" : "") +
               (kFp16 ? "Fp16" : "");
    }
};

// Pointers are only stored in Kargs; the contract test passes addresses it never dereferences.
struct Pointers
{
    const void* q                 = nullptr;
    const void* k                 = nullptr;
    const void* v                 = nullptr;
    void* o                       = nullptr;
    const void* seqstart_q        = nullptr;
    const void* seqlen_k          = nullptr;
    const void* block_table       = nullptr; // vLLM block table, or the SGLang kv_page_indices
    const void* kv_indptr         = nullptr;
    const void* kv_last_page_lens = nullptr;
    void* lse                     = nullptr;
};

// Problem in the kernel's native layout: Q/O packed [sum(q_len), Hq, D]; K/V pools
// [num_pages, P, Hkv, D]; one int32 block-table row of table_stride entries per sequence. The
// SGLang lists hold the same pages: sequence b owns kv_page_indices[kv_indptr[b], kv_indptr[b+1])
// and its last page holds kv_last_page_lens[b] tokens.
struct Layout
{
    int q_heads         = 0;
    int kv_heads        = 0;
    int num_total_pages = 0;
    int table_stride    = 0;
    std::vector<int32_t> seqstart_q;
    std::vector<int32_t> seqlen_k;
    std::vector<int32_t> block_table;
    std::vector<int32_t> kv_indptr;
    std::vector<int32_t> kv_page_indices;
    std::vector<int32_t> kv_last_page_lens;

    int batch() const { return static_cast<int>(seqlen_k.size()); }
    int q_length(int b) const { return seqstart_q[b + 1] - seqstart_q[b]; }
};

// top_left selects the top-left aligned causal mask (causal models only); bottom-right otherwise.
template <typename M>
fmha_batch_prefill_args MakeArgs(const Layout& l, const Pointers& d, bool top_left = false)
{
    int max_q = 0, max_k = 0;
    for(int b = 0; b < l.batch(); ++b)
    {
        max_q = std::max(max_q, l.q_length(b));
        max_k = std::max(max_k, l.seqlen_k[b]);
    }
    constexpr index_t D = M::kHeadDim;
    fmha_batch_prefill_args a{};
    a.q_ptr                    = d.q;
    a.k_ptr                    = d.k;
    a.v_ptr                    = d.v;
    a.o_ptr                    = d.o;
    a.seqstart_q_ptr           = d.seqstart_q;
    a.kv_page_indices          = const_cast<void*>(d.block_table);
    a.seqlen_k_ptr             = kSglang ? nullptr : const_cast<void*>(d.seqlen_k);
    a.kv_indptr                = const_cast<void*>(d.kv_indptr);
    a.kv_last_page_lens        = const_cast<void*>(d.kv_last_page_lens);
    a.batch                    = l.batch();
    a.seqlen_q                 = max_q;
    a.seqlen_k                 = max_k;
    a.max_seqlen_q             = max_q;
    a.hdim_q                   = D;
    a.hdim_v                   = D;
    a.nhead_q                  = l.q_heads;
    a.nhead_k                  = l.kv_heads;
    a.num_total_pages          = l.num_total_pages;
    a.page_block_size          = kPage;
    a.kv_memory_layout         = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT;
    a.kv_lookup_table          = kLookupTable;
    a.batch_stride_block_table = l.table_stride;
    a.scale_s                  = 1.0f / std::sqrt(static_cast<float>(D));
    a.stride_q                 = l.q_heads * D;
    a.stride_k                 = l.kv_heads * D;
    a.stride_v                 = l.kv_heads * D;
    a.stride_o                 = l.q_heads * D;
    a.nhead_stride_q           = D;
    a.nhead_stride_k           = D;
    a.nhead_stride_v           = D;
    a.nhead_stride_o           = D;
    a.batch_stride_k           = kPage * l.kv_heads * D;
    a.batch_stride_v           = kPage * l.kv_heads * D;
    a.lse_ptr                  = d.lse;
    a.nhead_stride_lse         = l.seqstart_q.back(); // LSE is [Hq, total_q]
    a.batch_stride_lse         = 0;
    a.window_size_left         = -1;
    a.window_size_right        = M::kCausal ? 0 : -1;
    a.mask_type                = static_cast<int>(
        !M::kCausal ? mask_enum::no_mask
                    : (top_left ? mask_enum::mask_top_left : mask_enum::mask_bottom_right));
    return a;
}

// ---------------------------------------------------------------------------------------------
// Launch contract (host only)

template <typename M>
class PageTileContract : public ::testing::Test
{
};
TYPED_TEST_SUITE(PageTileContract, Models, ModelNames);

TYPED_TEST(PageTileContract, ResidencyRequest)
{
    using Kernel = typename TypeParam::Kernel;
    // Only a P16 tile issues two TDM fragments per wave and operand. P16/D64 would admit three
    // resident workgroups on its static LDS alone, so the contract asks for dynamic LDS until
    // the reservation is 54 granules (110592 B); P16/D128 is already limited to two.
    const index_t expected =
        (kPage == 16 && TypeParam::kHeadDim == 64) ? 110592 - Kernel::GetSmemSize() : 0;
    EXPECT_EQ(Kernel::kNeedsResidencyCap, kPage == 16);
    EXPECT_EQ(Kernel::GetLaunchDynamicSmemSize(), expected);
    if(kPage == 16 && TypeParam::kHeadDim == 64)
        EXPECT_EQ(Kernel::GetLaunchDynamicSmemSize(), 32736);
    const auto reserved = ck_tile::integer_divide_ceil(
                              Kernel::GetSmemSize() + Kernel::GetLaunchDynamicSmemSize(), 2048) *
                          2048;
    EXPECT_LE(2 * reserved, 320 * 1024) << "two workgroups must stay resident";
    if(Kernel::kNeedsResidencyCap)
        EXPECT_GT(3 * reserved, 320 * 1024) << "a third workgroup must not fit";
}

template <typename K>
void SetTopLeft(K& kargs)
{
    if constexpr(requires { kargs.mask_type; })
        kargs.mask_type = ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT;
}

TYPED_TEST(PageTileContract, CheckLaunch)
{
    using Kernel = typename TypeParam::Kernel;
    using Kargs  = typename Kernel::Kargs;
    auto fake    = [](std::uintptr_t x) { return reinterpret_cast<void*>(x); };
    Layout l;
    l.q_heads         = 8;
    l.kv_heads        = 8;
    l.table_stride    = (256 + kPage - 1) / kPage;
    l.num_total_pages = l.table_stride;
    l.seqstart_q      = {0, 256};
    l.seqlen_k        = {256};
    const Pointers d{fake(0x10000),
                     fake(0x20000),
                     fake(0x30000),
                     fake(0x40000),
                     fake(0x50000),
                     fake(0x70000),
                     fake(0x60000),
                     fake(0x80000),
                     fake(0x90000),
                     fake(0xa0000)};
    const auto [base, grids] =
        fmha_batch_prefill_create_kargs_and_grids<Kernel>(MakeArgs<TypeParam>(l, d));
    const index_t need = Kernel::GetLaunchDynamicSmemSize();

    // Empty when CheckLaunch accepts, otherwise the refusal message.
    auto refusal = [&](index_t dynamic, const std::function<void(Kargs&)>& edit) {
        Kargs k = base;
        edit(k);
        try
        {
            Kernel::CheckLaunch(k, dynamic);
            return std::string{};
        }
        catch(const std::invalid_argument& error)
        {
            return std::string{error.what()};
        }
    };
    // Each case breaks exactly one rule and must be refused by that rule.
    auto expect_refused =
        [&](const char* rule, index_t dynamic, const std::function<void(Kargs&)>& edit) {
            const auto message = refusal(dynamic, edit);
            EXPECT_NE(message.find(rule), std::string::npos)
                << "expected refusal \"" << rule << "\", got \"" << message << '"';
        };
    auto none = [](Kargs&) {};
    EXPECT_EQ(refusal(need, none), "");
    EXPECT_EQ(refusal(need + 4096, none), "");
    if(need > 0)
    {
        expect_refused("dynamic LDS request below", need - 1, none);
        expect_refused("dynamic LDS request below", 0, none);
    }
    expect_refused("page_block_size differs", need, [](Kargs& k) { k.page_block_size *= 2; });
    expect_refused("hdim_q differs", need, [](Kargs& k) { k.hdim_q += 64; });
    expect_refused("hdim_v differs", need, [](Kargs& k) { k.hdim_v += 64; });
    expect_refused("nhead_ratio_qk must be positive", need, [](Kargs& k) { k.nhead_ratio_qk = 0; });
    if constexpr(kSglang)
    {
        expect_refused(
            "kv_indptr is null", need, [](Kargs& k) { k.page_table.kv_indptr = nullptr; });
        expect_refused("kv_page_indices is null", need, [](Kargs& k) {
            k.page_table.kv_page_indices = nullptr;
        });
        expect_refused("kv_last_page_lens is null", need, [](Kargs& k) {
            k.page_table.kv_last_page_lens = nullptr;
        });
    }
    else
    {
        expect_refused(
            "seqlen_k_ptr is null", need, [](Kargs& k) { k.page_table.seqlen_k_ptr = nullptr; });
        expect_refused("block_table_ptr is null", need, [](Kargs& k) {
            k.page_table.block_table_ptr = nullptr;
        });
    }
    expect_refused("seqstart_q_ptr is null", need, [](Kargs& k) { k.seqstart_q_ptr = nullptr; });
    if constexpr(kLse)
        expect_refused("lse_ptr is null", need, [](Kargs& k) { k.lse_ptr = nullptr; });
    expect_refused("page stride smaller", need, [](Kargs& k) { k.batch_stride_k = k.stride_k; });
    expect_refused("page stride smaller", need, [](Kargs& k) { k.batch_stride_v = k.stride_v; });
    expect_refused("not a multiple of the vector alignment", need, [](Kargs& k) {
        k.stride_k += 1;
        k.batch_stride_k = kPage * k.stride_k;
    });
    expect_refused("not a multiple of the vector alignment", need, [](Kargs& k) {
        k.stride_v += 1;
        k.batch_stride_v = kPage * k.stride_v;
    });
    expect_refused("token stride smaller than the head dimension", need, [](Kargs& k) {
        k.stride_k       = k.hdim_q / 2;
        k.batch_stride_k = kPage * k.stride_k;
    });
    expect_refused("token stride smaller than the head dimension", need, [](Kargs& k) {
        k.stride_v       = k.hdim_v / 2;
        k.batch_stride_v = kPage * k.stride_v;
    });
    if constexpr(TypeParam::kCausal)
    {
        EXPECT_EQ(refusal(need, [](Kargs& k) { SetTopLeft(k); }), "");
        expect_refused("only top-left or bottom-right causal masks", need, [](Kargs& k) {
            k.mask_type = ck_tile::GenericAttentionMaskEnum::MASK_GENERIC;
        });
        expect_refused("only the causal window", need, [](Kargs& k) { k.window_size_left = 64; });
        expect_refused("only the causal window", need, [](Kargs& k) { k.window_size_right = 1; });
    }

    // The checked launch path builds the functor for valid arguments (nothing is launched)
    // and refuses invalid ones before a functor exists.
    EXPECT_NO_THROW((void)Kernel::MakeCheckedKernel(base, grids));
    Kargs bad = base;
    if constexpr(kSglang)
        bad.page_table.kv_indptr = nullptr;
    else
        bad.page_table.seqlen_k_ptr = nullptr;
    EXPECT_THROW((void)Kernel::MakeCheckedKernel(bad, grids), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
// Device problems

enum class Mapping
{
    Identity, // sequence pages are consecutive physical pages
    Permuted, // distinct random physical pages in a pool twice the needed size
    Aliased,  // logical pages cycle over three physical pages shared by every sequence
};

const char* MappingName(Mapping m)
{
    return m == Mapping::Identity ? "identity" : m == Mapping::Permuted ? "permuted" : "aliased";
}

struct Case
{
    const char* name;
    int q_heads;
    int kv_heads;
    std::vector<int> q_lengths;
    std::vector<int> k_lengths;
};

// Rows never read as a valid key get this value: a leak into a valid score, or into P*V with a
// nonzero P, is far outside the accuracy contract. Unlike NaN it stays harmless when multiplied
// by P = 0, so rows masked inside a tile are not checked this way.
constexpr float kUnusedRowValue = 1000.0f;

struct HostProblem
{
    Layout layout;
    int head_dim = 0;
    std::vector<Data> q, k, v;
    std::vector<float> q_float;
};

// Values of V. Marker: V is zero except for the keys of one logical page (the middle page of
// every sequence), where it is one, so each output component is the softmax mass of that page and
// a fault involving that page is large relative to the output. SparseMarker: the same with the
// first, quarter, middle, three-quarter and last logical pages.
enum class Values
{
    Random,
    Marker,
    SparseMarker,
};

bool IsMarkerPage(Values values, int page, int pages)
{
    if(values == Values::Marker)
        return page == (pages - 1) / 2;
    if(values == Values::SparseMarker)
        return page == 0 || page == pages / 4 || page == (pages - 1) / 2 || page == 3 * pages / 4 ||
               page == pages - 1;
    return false;
}

// Block-table entries past each sequence's last page. By default every row has two spare
// entries holding an in-range page that no sequence reads; Poisoned holds an invalid page id
// instead, so a read through a spare entry traps in the kernel's page-id range check.
struct TableOptions
{
    int spare_entries  = 2;
    bool poison_spares = false;
};

// Q and the logical K/V rows are drawn in logical order from the seed alone, so two problems
// that differ only in the mapping (or in the spare table entries) hold the same logical data and
// must give identical outputs.
HostProblem MakeProblem(const Case& s,
                        Mapping mapping,
                        int head_dim,
                        bool causal,
                        std::uint32_t seed,
                        Values values             = Values::Random,
                        const TableOptions& table = {})
{
    HostProblem h;
    auto& l         = h.layout;
    h.head_dim      = head_dim;
    l.q_heads       = s.q_heads;
    l.kv_heads      = s.kv_heads;
    const int batch = static_cast<int>(s.k_lengths.size());
    l.seqstart_q    = {0};
    int max_pages = 0, needed_pages = 0;
    for(int b = 0; b < batch; ++b)
    {
        // Bottom-right causal rows need at least one key each.
        const int q_len = causal ? std::min(s.q_lengths[b], s.k_lengths[b]) : s.q_lengths[b];
        l.seqstart_q.push_back(l.seqstart_q.back() + q_len);
        l.seqlen_k.push_back(s.k_lengths[b]);
        const int pages = (s.k_lengths[b] + kPage - 1) / kPage;
        max_pages       = std::max(max_pages, pages);
        needed_pages += pages;
    }
    // Spare entries per row: the kernel must not use table entries past the sequence.
    l.table_stride = max_pages + table.spare_entries;

    std::mt19937 table_rng(seed ^ 0x9e3779b9u);
    std::vector<int32_t> physical;
    if(mapping == Mapping::Identity)
    {
        l.num_total_pages = needed_pages + 1;
        physical.resize(needed_pages);
        std::iota(physical.begin(), physical.end(), 0);
    }
    else if(mapping == Mapping::Permuted)
    {
        l.num_total_pages = 2 * needed_pages + 1;
        std::vector<int32_t> pool(l.num_total_pages);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), table_rng);
        physical.assign(pool.begin(), pool.begin() + needed_pages);
    }
    else
    {
        l.num_total_pages = 4;
        for(int i = 0; i < needed_pages; ++i)
            physical.push_back(std::array<int32_t, 3>{3, 0, 2}[i % 3]);
    }
    // Spare entries point at an in-range page that no sequence reads (filled with
    // kUnusedRowValue); a read through them is detected only if it reaches a valid score.
    // Poisoned spare entries hold an out-of-range id instead.
    const int32_t spare = [&] {
        if(table.poison_spares)
            return std::numeric_limits<int32_t>::min();
        std::vector<bool> used(l.num_total_pages, false);
        for(auto p : physical)
            used[p] = true;
        return static_cast<int32_t>(std::find(used.begin(), used.end(), false) - used.begin());
    }();
    l.block_table.assign(static_cast<std::size_t>(batch) * l.table_stride, spare);
    for(int b = 0, next = 0; b < batch; ++b)
        for(int page = 0; page * kPage < l.seqlen_k[b]; ++page)
            l.block_table[b * l.table_stride + page] = physical[next++];
    // The SGLang lists name the same pages. The spare entries follow the last sequence: a read
    // past a sequence's pages lands on the next sequence's pages, or on the spares after the last.
    l.kv_indptr = {0};
    for(int b = 0; b < batch; ++b)
    {
        const int pages = (l.seqlen_k[b] + kPage - 1) / kPage;
        for(int page = 0; page < pages; ++page)
            l.kv_page_indices.push_back(l.block_table[b * l.table_stride + page]);
        l.kv_indptr.push_back(static_cast<int32_t>(l.kv_page_indices.size()));
        l.kv_last_page_lens.push_back(pages > 0 ? l.seqlen_k[b] - (pages - 1) * kPage : 0);
    }
    l.kv_page_indices.insert(l.kv_page_indices.end(), table.spare_entries, spare);

    const std::size_t row      = static_cast<std::size_t>(l.kv_heads) * head_dim;
    const std::size_t pool_len = static_cast<std::size_t>(l.num_total_pages) * kPage * row;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    auto draw = [&] { return ck_tile::type_convert<Data>(uniform(rng)); };
    h.q.resize(static_cast<std::size_t>(l.seqstart_q.back()) * l.q_heads * head_dim);
    std::generate(h.q.begin(), h.q.end(), draw);
    // Rows that no sequence reads as a valid key keep kUnusedRowValue.
    const Data unused = ck_tile::type_convert<Data>(kUnusedRowValue);
    h.k.assign(pool_len, unused);
    h.v.assign(pool_len, unused);
    for(int b = 0; b < batch; ++b)
    {
        const int pages = (l.seqlen_k[b] + kPage - 1) / kPage;
        for(int key = 0; key < l.seqlen_k[b]; ++key)
        {
            const std::size_t dst =
                (static_cast<std::size_t>(l.block_table[b * l.table_stride + key / kPage]) * kPage +
                 key % kPage) *
                row;
            for(std::size_t x = 0; x < row; ++x)
            {
                h.k[dst + x]   = draw();
                const Data v   = draw();
                const float on = IsMarkerPage(values, key / kPage, pages) ? 1.0f : 0.0f;
                h.v[dst + x]   = values == Values::Random ? v : ck_tile::type_convert<Data>(on);
            }
        }
    }
    auto decode = [](const std::vector<Data>& data) {
        std::vector<float> out(data.size());
        std::transform(data.begin(), data.end(), out.begin(), [](Data x) {
            return ck_tile::type_convert<float>(x);
        });
        return out;
    };
    h.q_float = decode(h.q);
    return h;
}

struct DeviceProblem
{
    ck_tile::DeviceMem q, k, v, o, seqstart_q, seqlen_k, block_table, kv_indptr, kv_last_page_lens,
        lse;

    // The block_table buffer holds the vLLM table, or the SGLang kv_page_indices.
    static const std::vector<int32_t>& Pages(const Layout& l)
    {
        return kSglang ? l.kv_page_indices : l.block_table;
    }

    explicit DeviceProblem(const HostProblem& h)
        : q(h.q.size() * sizeof(Data)),
          k(h.k.size() * sizeof(Data)),
          v(h.v.size() * sizeof(Data)),
          o(h.q.size() * sizeof(Data)),
          seqstart_q(h.layout.seqstart_q.size() * sizeof(int32_t)),
          seqlen_k(h.layout.seqlen_k.size() * sizeof(int32_t)),
          block_table(std::max<std::size_t>(Pages(h.layout).size(), 1) * sizeof(int32_t)),
          kv_indptr(h.layout.kv_indptr.size() * sizeof(int32_t)),
          kv_last_page_lens(std::max<std::size_t>(h.layout.kv_last_page_lens.size(), 1) *
                            sizeof(int32_t)),
          lse(std::max<std::size_t>(
                  static_cast<std::size_t>(h.layout.q_heads) * h.layout.seqstart_q.back(), 1) *
              sizeof(float))
    {
        q.ToDevice(h.q.data());
        k.ToDevice(h.k.data());
        v.ToDevice(h.v.data());
        seqstart_q.ToDevice(h.layout.seqstart_q.data());
        seqlen_k.ToDevice(h.layout.seqlen_k.data());
        if(!Pages(h.layout).empty())
            block_table.ToDevice(Pages(h.layout).data());
        kv_indptr.ToDevice(h.layout.kv_indptr.data());
        if(!h.layout.kv_last_page_lens.empty())
            kv_last_page_lens.ToDevice(h.layout.kv_last_page_lens.data());
    }

    Pointers pointers()
    {
        return {q.GetDeviceBuffer(),
                k.GetDeviceBuffer(),
                v.GetDeviceBuffer(),
                o.GetDeviceBuffer(),
                seqstart_q.GetDeviceBuffer(),
                seqlen_k.GetDeviceBuffer(),
                block_table.GetDeviceBuffer(),
                kv_indptr.GetDeviceBuffer(),
                kv_last_page_lens.GetDeviceBuffer(),
                lse.GetDeviceBuffer()};
    }
};

// Runs one checked launch on a NaN-filled output (and LSE) and returns the output; with an LSE
// kernel the [Hq, total_q] LSE is returned through lse_out when given.
template <typename M>
std::vector<Data> Run(const HostProblem& h,
                      DeviceProblem& d,
                      bool top_left               = false,
                      std::vector<float>* lse_out = nullptr)
{
    using Kernel = typename M::Kernel;
    std::vector<Data> o(h.q.size(),
                        ck_tile::type_convert<Data>(std::numeric_limits<float>::quiet_NaN()));
    d.o.ToDevice(o.data());
    std::vector<float> lse(static_cast<std::size_t>(h.layout.q_heads) * h.layout.seqstart_q.back(),
                           std::numeric_limits<float>::quiet_NaN());
    if(kLse && !lse.empty())
        d.lse.ToDevice(lse.data());
    const auto [kargs, grids] = fmha_batch_prefill_create_kargs_and_grids<Kernel>(
        MakeArgs<M>(h.layout, d.pointers(), top_left));
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           Kernel::MakeCheckedKernel(kargs, grids));
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    d.o.FromDevice(o.data());
    if(kLse && lse_out != nullptr)
    {
        if(!lse.empty())
            d.lse.FromDevice(lse.data());
        *lse_out = std::move(lse);
    }
    return o;
}

// Gathers each sequence's keys and values through the block table into contiguous
// [k_len, Hkv, D] arrays and compares every (or every sampled) output row.
reference::Metrics CompareWithReference(const HostProblem& h,
                                        const std::vector<Data>& o,
                                        bool causal,
                                        const std::vector<std::size_t>& sampled_rows = {},
                                        bool top_left                                = false,
                                        const std::vector<float>* lse                = nullptr)
{
    const auto& l          = h.layout;
    const std::size_t D    = h.head_dim;
    const std::size_t krow = static_cast<std::size_t>(l.kv_heads) * D;
    const std::size_t qrow = static_cast<std::size_t>(l.q_heads) * D;
    std::vector<float> o_float(o.size());
    std::transform(o.begin(), o.end(), o_float.begin(), [](Data x) {
        return ck_tile::type_convert<float>(x);
    });
    std::vector<std::vector<float>> keys(l.batch()), values(l.batch());
    std::vector<reference::Sequence> sequences;
    std::vector<reference::Output> outputs;
    for(int b = 0; b < l.batch(); ++b)
    {
        const int k_len = l.seqlen_k[b], q_len = l.q_length(b);
        for(int key = 0; key < k_len; ++key)
        {
            const std::size_t src =
                (static_cast<std::size_t>(l.block_table[b * l.table_stride + key / kPage]) * kPage +
                 key % kPage) *
                krow;
            for(std::size_t x = src; x < src + krow; ++x)
            {
                keys[b].push_back(ck_tile::type_convert<float>(h.k[x]));
                values[b].push_back(ck_tile::type_convert<float>(h.v[x]));
            }
        }
        const std::size_t q_offset = static_cast<std::size_t>(l.seqstart_q[b]) * qrow;
        reference::Sequence s;
        s.q               = {h.q_float.data(), h.q_float.size(), q_offset, D, qrow, 1};
        s.k               = {keys[b].data(), keys[b].size(), 0, D, krow, 1};
        s.v               = {values[b].data(), values[b].size(), 0, D, krow, 1};
        s.query_length    = q_len;
        s.key_length      = k_len;
        s.query_heads     = l.q_heads;
        s.kv_heads        = l.kv_heads;
        s.query_dimension = D;
        s.value_dimension = D;
        s.score_scale     = 1.0f / std::sqrt(static_cast<float>(D));
        if(causal)
        {
            // Key k is visible to query q when k <= q + alignment.
            const int alignment = top_left ? 0 : k_len - q_len;
            s.is_masked         = [alignment](std::size_t query, std::size_t key) {
                return static_cast<int64_t>(key) > static_cast<int64_t>(query) + alignment;
            };
        }
        for(auto r : sampled_rows)
            if(r < static_cast<std::size_t>(q_len))
                s.sampled_query_rows.push_back(r);
        if(!sampled_rows.empty() && s.sampled_query_rows.empty())
            s.sampled_query_rows.push_back(0);
        sequences.push_back(std::move(s));
        std::optional<reference::FloatView> lse_view;
        if(lse != nullptr)
            lse_view = reference::FloatView{lse->data(),
                                            lse->size(),
                                            static_cast<std::size_t>(l.seqstart_q[b]),
                                            static_cast<std::size_t>(l.seqstart_q.back()),
                                            1,
                                            1};
        outputs.push_back({{o_float.data(), o_float.size(), q_offset, D, qrow, 1}, lse_view});
    }
    const auto expected = reference::Compute(sequences);
    return reference::Compare(
        sequences, expected, outputs, {}, reference::AccuracyContract{kMantissaBits, 128});
}

template <typename M>
class PageTileForward : public ck_tile::test::Gfx125FmhaTest
{
};
TYPED_TEST_SUITE(PageTileForward, Models, ModelNames);

std::size_t LogicalRows(const Layout& l)
{
    std::size_t rows = 0;
    for(int b = 0; b < l.batch(); ++b)
        rows += static_cast<std::size_t>(l.q_length(b)) * l.q_heads;
    return rows;
}

TYPED_TEST(PageTileForward, MatchesReference)
{
    // Lengths are relative to the page size so that every binary sees full pages, a partial
    // last page, sequences shorter than one page and sequences that are not a multiple of the
    // 128-key tile. Causal query lengths are clipped to the key length (bottom-right mask).
    const std::vector<Case> shapes = {
        {"gqa2", 4, 2, {130, 13, 129, 64}, {2 * kPage + 37, kPage - 3, 129, 384}},
        {"mqa", 8, 1, {256}, {2 * kPage + 300}},
        // Single-tile and shorter-than-a-tile sequences at every page size, including q = 1.
        {"tiny", 4, 2, {1, 7, 128, 100}, {1, 100, 128, 100}},
    };
    std::uint32_t seed = 1;
    for(const auto& shape : shapes)
        for(auto values : {Values::Random, Values::Marker})
        {
            // Identity and permuted pools hold the same logical data: the outputs must be
            // bitwise equal, which exposes a page-indexing fault of any size. The aliased table
            // has its own data and is checked against the reference only.
            std::vector<Data> identity_output;
            for(auto mapping : {Mapping::Identity, Mapping::Permuted, Mapping::Aliased})
            {
                const std::string name = std::string(shape.name) + " " + MappingName(mapping) +
                                         (values == Values::Marker ? " marker" : " random");
                SCOPED_TRACE(name);
                const auto h = MakeProblem(
                    shape, mapping, TypeParam::kHeadDim, TypeParam::kCausal, seed, values);
                DeviceProblem d(h);
                std::vector<float> lse;
                const auto o       = Run<TypeParam>(h, d, false, &lse);
                const auto metrics = CompareWithReference(
                    h, o, TypeParam::kCausal, {}, false, kLse ? &lse : nullptr);
                std::cout << "page_tile P" << kPage << " D" << TypeParam::kHeadDim
                          << (TypeParam::kCausal ? " causal " : " dense ")
                          << (kSglang ? "sglang " : "") << name << " rows=" << metrics.checked_rows
                          << " max_abs=" << metrics.max_abs_error
                          << " min_cos=" << metrics.min_row_cosine
                          << " nonfinite=" << metrics.nonfinite_output_count
                          << " passed=" << metrics.passed << '\n';
                EXPECT_EQ(metrics.checked_rows, LogicalRows(h.layout));
                EXPECT_EQ(metrics.nonfinite_output_count, 0u);
                EXPECT_TRUE(metrics.passed);
                if(mapping == Mapping::Identity)
                    identity_output = o;
                if(mapping == Mapping::Permuted)
                {
                    ASSERT_EQ(o.size(), identity_output.size());
                    EXPECT_EQ(
                        std::memcmp(o.data(), identity_output.data(), o.size() * sizeof(Data)), 0)
                        << "the permuted table changed the output of the same logical problem";
                }
            }
            ++seed;
        }
}

// Top-left aligned causal mask (causal binaries only): query q sees keys 0..q. Query lengths are
// not clipped, so sequences with more queries than keys (rows past the keys see every key) and
// fewer queries than keys both occur; the bitwise identity/permuted check applies as above.
TYPED_TEST(PageTileForward, MatchesReferenceTopLeft)
{
    if constexpr(!TypeParam::kCausal)
        GTEST_SKIP() << "top-left alignment only applies to the causal mask";
    else
    {
        const std::vector<Case> shapes = {
            {"gqa2", 4, 2, {300, 13, 129, 64}, {2 * kPage + 37, kPage - 3, 129, 384}},
            {"tiny", 4, 2, {1, 130, 128, 200}, {1, 100, 128, 100}},
        };
        std::uint32_t seed = 51;
        for(const auto& shape : shapes)
        {
            std::vector<Data> identity_output;
            for(auto mapping : {Mapping::Identity, Mapping::Permuted})
            {
                const std::string name =
                    std::string(shape.name) + " top-left " + MappingName(mapping);
                SCOPED_TRACE(name);
                const auto h = MakeProblem(shape, mapping, TypeParam::kHeadDim, false, seed);
                DeviceProblem d(h);
                const auto o       = Run<TypeParam>(h, d, true);
                const auto metrics = CompareWithReference(h, o, true, {}, true);
                std::cout << "page_tile P" << kPage << " D" << TypeParam::kHeadDim << " causal "
                          << (kSglang ? "sglang " : "") << name << " rows=" << metrics.checked_rows
                          << " max_abs=" << metrics.max_abs_error
                          << " min_cos=" << metrics.min_row_cosine
                          << " nonfinite=" << metrics.nonfinite_output_count
                          << " passed=" << metrics.passed << '\n';
                EXPECT_EQ(metrics.checked_rows, LogicalRows(h.layout));
                EXPECT_EQ(metrics.nonfinite_output_count, 0u);
                EXPECT_TRUE(metrics.passed);
                if(mapping == Mapping::Identity)
                    identity_output = o;
                else
                {
                    ASSERT_EQ(o.size(), identity_output.size());
                    EXPECT_EQ(
                        std::memcmp(o.data(), identity_output.data(), o.size() * sizeof(Data)), 0)
                        << "the permuted table changed the output of the same logical problem";
                }
            }
            ++seed;
        }
    }
}

// The kernel must not read block-table entries past a sequence's last page. Poisoned spare
// entries make such a read trap; a tight stride (no spare entries) makes the longest row end at
// the end of its table row. Both change only entries the kernel must not read, so the output must
// be bitwise equal to the default table's.
TYPED_TEST(PageTileForward, SpareTableEntries)
{
    const std::vector<Case> shapes = {
        {"gqa2", 4, 2, {130, 13, 129, 64}, {2 * kPage + 37, kPage - 3, 129, 384}},
        {"tiny", 4, 2, {1, 7, 128, 100}, {1, 100, 128, 100}},
    };
    const std::uint32_t seed = 31;
    for(const auto& shape : shapes)
    {
        const auto base =
            MakeProblem(shape, Mapping::Permuted, TypeParam::kHeadDim, TypeParam::kCausal, seed);
        DeviceProblem base_device(base);
        const auto base_output = Run<TypeParam>(base, base_device);
        for(const auto& [table_name, table] : {std::pair{"poisoned spares", TableOptions{2, true}},
                                               std::pair{"tight stride", TableOptions{0, false}}})
        {
            const std::string name = std::string(shape.name) + " " + table_name;
            SCOPED_TRACE(name);
            const auto h = MakeProblem(shape,
                                       Mapping::Permuted,
                                       TypeParam::kHeadDim,
                                       TypeParam::kCausal,
                                       seed,
                                       Values::Random,
                                       table);
            DeviceProblem d(h);
            const auto o       = Run<TypeParam>(h, d);
            const auto metrics = CompareWithReference(h, o, TypeParam::kCausal);
            std::cout << "page_tile P" << kPage << " D" << TypeParam::kHeadDim
                      << (TypeParam::kCausal ? " causal " : " dense ") << (kSglang ? "sglang " : "")
                      << name << " rows=" << metrics.checked_rows
                      << " max_abs=" << metrics.max_abs_error
                      << " min_cos=" << metrics.min_row_cosine
                      << " nonfinite=" << metrics.nonfinite_output_count
                      << " passed=" << metrics.passed << '\n';
            EXPECT_EQ(metrics.checked_rows, LogicalRows(h.layout));
            EXPECT_EQ(metrics.nonfinite_output_count, 0u);
            EXPECT_TRUE(metrics.passed);
            ASSERT_EQ(o.size(), base_output.size());
            EXPECT_EQ(std::memcmp(o.data(), base_output.data(), o.size() * sizeof(Data)), 0)
                << "changing only the spare table entries changed the output";
        }
    }
}

// Repeated launches of a device-filling grid must give bitwise equal outputs, and equal to the
// same logical problem in an identity-mapped pool. This guards the residency limit of the launch
// contract: at this shape a P16/D64 launch that requests no dynamic LDS (three resident
// workgroups per WGP) gave a different output on every launch in out-of-tree measurements. A
// sampled set of rows is also checked against the reference.
TYPED_TEST(PageTileForward, Deterministic)
{
    const Case shape{"race", 32, 8, {8192}, {32768}};
    const auto h =
        MakeProblem(shape, Mapping::Permuted, TypeParam::kHeadDim, TypeParam::kCausal, 99);
    std::vector<Data> first;
    {
        DeviceProblem d(h);
        first = Run<TypeParam>(h, d);
        for(int repeat = 1; repeat < 8; ++repeat)
        {
            const auto again = Run<TypeParam>(h, d);
            ASSERT_EQ(std::memcmp(first.data(), again.data(), first.size() * sizeof(Data)), 0)
                << "repeat " << repeat << " differs from the first launch";
        }
    }
    {
        const auto identity =
            MakeProblem(shape, Mapping::Identity, TypeParam::kHeadDim, TypeParam::kCausal, 99);
        DeviceProblem d(identity);
        const auto o = Run<TypeParam>(identity, d);
        EXPECT_EQ(std::memcmp(first.data(), o.data(), first.size() * sizeof(Data)), 0)
            << "the permuted table changed the output of the same logical problem";
    }
    const std::vector<std::size_t> sampled = {0, 1, 127, 128, 129, 4095, 8191};
    const auto metrics = CompareWithReference(h, first, TypeParam::kCausal, sampled);
    EXPECT_EQ(metrics.checked_rows, sampled.size() * shape.q_heads);
    EXPECT_EQ(metrics.nonfinite_output_count, 0u);
    EXPECT_TRUE(metrics.passed);
    {
        // At this KV length a single wrong page is invisible under random V; with V nonzero
        // only on a few spread pages, a fault involving one of them is not.
        const auto marker = MakeProblem(shape,
                                        Mapping::Permuted,
                                        TypeParam::kHeadDim,
                                        TypeParam::kCausal,
                                        99,
                                        Values::SparseMarker);
        DeviceProblem d(marker);
        const auto o              = Run<TypeParam>(marker, d);
        const auto marker_metrics = CompareWithReference(marker, o, TypeParam::kCausal, sampled);
        // Non-vacuity: the marked pages must reach the output. V is zero on every other page, so
        // a row that sees one marked page has outputs of about kPage / 32768 or more (about
        // 0.0005 at P16); require half of that somewhere.
        const float floor = 0.5f * static_cast<float>(kPage) / 32768.0f;
        EXPECT_TRUE(std::any_of(o.begin(), o.end(), [floor](Data x) {
            return ck_tile::type_convert<float>(x) > floor;
        })) << "no marked page reached the output";
        EXPECT_EQ(marker_metrics.checked_rows, sampled.size() * shape.q_heads);
        EXPECT_EQ(marker_metrics.nonfinite_output_count, 0u);
        EXPECT_TRUE(marker_metrics.passed);
    }
}

} // namespace
