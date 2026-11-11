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
// KV-split binaries (CK_TILE_FMHA_PAGE_TILE_TEST_KV_SPLITS 2) run the same checks on the split
// kernel and its merge, launched as the generated instances launch them, on a NaN-filled workspace;
// query-tail binaries (CK_TILE_FMHA_PAGE_TILE_TEST_QUERY_TAIL 1) with the M128 kernel in front.
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
#include "ck_tile/ops/fmha/kernel/fmha_batch_prefill_split_merge_kernel.hpp"
#include "fmha_reference.hpp"
#include "fmha_test_common.hpp"

#if !defined(CK_TILE_FMHA_PAGE_TILE_TEST_PAGE) || \
    !defined(CK_TILE_FMHA_PAGE_TILE_TEST_HEAD_DIM) || !defined(CK_TILE_FMHA_PAGE_TILE_TEST_CAUSAL)
#error "the page size, head dimension and mask of the tested tuple must be defined"
#endif
// 0: vLLM 2D block table (default); 1: SGLang 1D page list (kv_indptr / kv_page_indices /
// kv_last_page_lens), built from the same per-sequence pages.
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_VECTORIZED
#define CK_TILE_FMHA_PAGE_TILE_TEST_VECTORIZED 0
#endif
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
// 1: the sink instance (sink tokens via sink_size and per-head sink logits via sink_ptr); causal
// models only.
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_SINK
#define CK_TILE_FMHA_PAGE_TILE_TEST_SINK 0
#endif
// 1: the tanh logits soft cap instance (every forward test runs with the cap below).
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_SOFTCAP
#define CK_TILE_FMHA_PAGE_TILE_TEST_SOFTCAP 0
#endif
// 1: elementwise bias [Hq, total_q, max seqlen_k]; 2: ALiBi with per-head slopes. Every forward
// test runs with it.
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_BIAS
#define CK_TILE_FMHA_PAGE_TILE_TEST_BIAS 0
#endif
// 1: the M64 small-grid tile (D64: K64 PV stages, kK1 64) instead of M128.
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_M64
#define CK_TILE_FMHA_PAGE_TILE_TEST_M64 0
#endif
// 2: the KV split (the kernel with KvSplits 2 writes FP32 partial outputs and their LSE to a
// workspace, FmhaBatchPrefillSplitMergeKernel merges them into O); 1: one kernel.
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_KV_SPLITS
#define CK_TILE_FMHA_PAGE_TILE_TEST_KV_SPLITS 1
#endif
// 1 (KV split with M64 tiles): the query tail. An M128 kernel runs each sequence's full 128-row
// query tiles, the KV split the rows past them, then the merge (fmha_batch_prefill_tail_launch).
#ifndef CK_TILE_FMHA_PAGE_TILE_TEST_QUERY_TAIL
#define CK_TILE_FMHA_PAGE_TILE_TEST_QUERY_TAIL 0
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
static_assert(kPage == 16 || kPage == 64 || kPage == 128 || kPage == 256 || kPage == 1024);
constexpr bool kSglang = CK_TILE_FMHA_PAGE_TILE_TEST_SGLANG != 0;
// Vectorized KV cache (K page-head [D/8][P][8], V page-head [P/8][D][8]). The host problem and
// the reference keep the linear pool; the device pools are its vectorized permutation.
constexpr bool kVectorized = CK_TILE_FMHA_PAGE_TILE_TEST_VECTORIZED != 0;
constexpr auto kKVLayout   = kVectorized
                                 ? ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT
                                 : ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT;
constexpr bool kLse        = CK_TILE_FMHA_PAGE_TILE_TEST_LSE != 0;
constexpr bool kSink       = CK_TILE_FMHA_PAGE_TILE_TEST_SINK != 0;
constexpr bool kSoftcap    = CK_TILE_FMHA_PAGE_TILE_TEST_SOFTCAP != 0;
// Scaled scores of the random inputs are about N(0, 1), so a cap of 1.5 bends most of them.
constexpr float kSoftcapValue = 1.5f;
constexpr int kBias           = CK_TILE_FMHA_PAGE_TILE_TEST_BIAS;
static_assert(kBias >= 0 && kBias <= 2);
constexpr auto kBiasEnum = kBias == 1   ? ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS
                           : kBias == 2 ? ck_tile::BlockAttentionBiasEnum::ALIBI
                                        : ck_tile::BlockAttentionBiasEnum::NO_BIAS;
constexpr auto kLookupTable =
    kSglang ? ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D
            : ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D;
constexpr bool kM64         = CK_TILE_FMHA_PAGE_TILE_TEST_M64 != 0;
constexpr index_t kTileM    = kM64 ? 64 : 128;
constexpr index_t kKvSplits = CK_TILE_FMHA_PAGE_TILE_TEST_KV_SPLITS;
static_assert(kKvSplits == 1 ||
                  (kKvSplits == 2 && kVectorized && !kSglang && !kFp16 && !kLse && !kSink &&
                   !kSoftcap && kBias == 0 && (kPage == 64 || kPage == 128)),
              "the KV split is tested in the configurations of the generated split instances");
constexpr index_t kQueryTailRows = CK_TILE_FMHA_PAGE_TILE_TEST_QUERY_TAIL != 0 ? 128 : 0;
static_assert(kQueryTailRows == 0 || (kKvSplits == 2 && kM64),
              "the query tail is a KV split with M64 tiles");

// Pipeline and epilogue of one page-tile kernel with an M TileM, PV stage depth TileK1 tile.
// Split: the KV split's kernel, which stores FP32 partial outputs and their LSE (the merge
// writes O).
template <index_t HeadDim, bool Causal, index_t TileM, index_t TileK1, bool Split>
struct KernelTypes
{
    using TypeConfig = FmhaFwdTypeConfig<DataConfig>;
    using OData      = std::conditional_t<Split, float, typename TypeConfig::ODataType>;
    using Shape =
        ck_tile::TileFmhaShape<ck_tile::sequence<TileM, 128, 32, HeadDim, TileK1, HeadDim>,
                               ck_tile::sequence<4, 1, 1>,
                               ck_tile::sequence<16, 16, 32>,
                               ck_tile::sequence<4, 1, 1>,
                               ck_tile::sequence<16, 16, 32>,
                               true,
                               false>;
    using Traits = ck_tile::TileFmhaBatchPrefillTraits<
        true,
        true,
        false,
        false,
        kSoftcap,
        kBiasEnum,
        false,
        kLse || Split,
        false,
        ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
        1,
        false,
        kSink,
        static_cast<int>(kPage),
        kKVLayout,
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
        OData,
        Shape,
        true,
        ck_tile::ComposedAttention<kSoftcap ? ck_tile::LOGITS_SOFT_CAP : 0u,
                                   CK_TILE_FMHA_FWD_FAST_EXP2>,
        ck_tile::SimplifiedGenericAttentionMask<Causal>,
        false,
        static_cast<int>(kPage),
        Traits>;
    using Pipeline = ck_tile::BlockFmhaBatchPrefillPipelineQRKSVSTdmSchedPageTile<Problem>;
    using Epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<typename TypeConfig::OaccDataType, OData, true, false>>;
};

// BF16 or FP16, vLLM 2D or SGLang 1D page table, linear [num_pages, P, Hkv, D] K/V pool, group
// mode, M128/N128 (or M64/N128).
template <index_t HeadDim, bool Causal>
struct Model
{
    using TypeConfig = FmhaFwdTypeConfig<DataConfig>;
    // PV stage depth: K64 for D64 M64 (64-column QK groups), K32 otherwise.
    static constexpr index_t kTileK1 = kM64 && HeadDim == 64 ? 64 : 32;
    using Types  = KernelTypes<HeadDim, Causal, kTileM, kTileK1, kKvSplits == 2>;
    using Kernel = ck_tile::FmhaBatchPrefillQrTdmPageTileKernel<typename Types::Pipeline,
                                                                typename Types::Epilogue,
                                                                kKvSplits,
                                                                kQueryTailRows>;
    using Merge  = ck_tile::
        FmhaBatchPrefillSplitMergeKernel<typename TypeConfig::ODataType, HeadDim, kQueryTailRows>;
    // Query tail: the M128 kernel that runs the full query tiles.
    using MainTypes  = KernelTypes<HeadDim, Causal, 128, 32, false>;
    using MainKernel = ck_tile::FmhaBatchPrefillQrTdmPageTileKernel<typename MainTypes::Pipeline,
                                                                    typename MainTypes::Epilogue>;

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
               (kFp16 ? "Fp16" : "") + (kSink ? "Sink" : "") + (kSoftcap ? "Softcap" : "") +
               (kBias == 1   ? "Bias"
                : kBias == 2 ? "Alibi"
                             : "") +
               (kM64 ? "M64" : "") + (kKvSplits == 2 ? "Split" : "") +
               (kQueryTailRows > 0 ? "Tail" : "");
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
    const void* sink              = nullptr; // per-head sink logits [Hq]
    const void* bias              = nullptr; // elementwise bias, or the ALiBi slopes [Hq]
    void* o_acc                   = nullptr; // KV-split workspace: partial O and LSE planes
    void* lse_acc                 = nullptr;
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

// Mask window of a masked model: key k is visible to query q when
// q + offset - left <= k <= q + offset + right, with offset 0 (top-left) or
// seqlen_k - seqlen_q (bottom-right); a negative bound is unbounded. The default is the
// bottom-right causal mask (left unbounded, right 0). Dense models ignore it.
struct Window
{
    bool top_left = false;
    int left      = -1;
    int right     = 0;
    // Sink instance only: the first `sink` keys stay visible (sink_size, StreamingLLM), and
    // sink_logits passes the per-head sink logits (sink_ptr).
    int sink         = 0;
    bool sink_logits = false;
};

template <typename M>
fmha_batch_prefill_args MakeArgs(const Layout& l, const Pointers& d, Window w = {})
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
    a.kv_memory_layout         = kKVLayout;
    a.kv_lookup_table          = kLookupTable;
    a.batch_stride_block_table = l.table_stride;
    a.scale_s                  = 1.0f / std::sqrt(static_cast<float>(D));
    a.stride_q                 = l.q_heads * D;
    a.stride_k                 = l.kv_heads * D;
    a.stride_v                 = l.kv_heads * D;
    a.stride_o                 = l.q_heads * D;
    a.nhead_stride_q           = D;
    a.nhead_stride_k           = kVectorized ? kPage * D : D;
    a.nhead_stride_v           = kVectorized ? kPage * D : D;
    a.nhead_stride_o           = D;
    a.batch_stride_k           = kPage * l.kv_heads * D;
    a.batch_stride_v           = kPage * l.kv_heads * D;
    a.lse_ptr                  = d.lse;
    a.nhead_stride_lse         = l.seqstart_q.back(); // LSE is [Hq, total_q]
    a.batch_stride_lse         = 0;
    a.logits_soft_cap          = kSoftcap ? kSoftcapValue : 0.f;
    // Elementwise bias [Hq, total_q, max_k]; ALiBi slopes [Hq] shared by every batch.
    a.bias_ptr          = d.bias;
    a.stride_bias       = kBias == 1 ? max_k : 0;
    a.nhead_stride_bias = kBias == 1 ? l.seqstart_q.back() * max_k : 0;
    a.batch_stride_bias = 0;
    a.sink_size         = M::kCausal ? w.sink : 0;
    a.sink_ptr          = w.sink_logits ? d.sink : nullptr;
    a.window_size_left  = M::kCausal ? w.left : -1;
    a.window_size_right = M::kCausal ? w.right : -1;
    a.mask_type         = static_cast<int>(
        !M::kCausal ? mask_enum::no_mask
                    : (w.top_left ? mask_enum::mask_top_left : mask_enum::mask_bottom_right));
    // KV-split workspace in AITER's split-KV layout: O [Hq][split][total_q][D], LSE
    // [Hq][split][total_q].
    const index_t total_q  = l.seqstart_q.back();
    a.num_splits           = kKvSplits;
    a.o_acc_ptr            = d.o_acc;
    a.lse_acc_ptr          = d.lse_acc;
    a.stride_o_acc         = D;
    a.split_stride_o_acc   = total_q * D;
    a.nhead_stride_o_acc   = kKvSplits * total_q * D;
    a.split_stride_lse_acc = total_q;
    a.nhead_stride_lse_acc = kKvSplits * total_q;
    return a;
}

// Kernel arguments and grid as the generated instances build them (a KV split's kernel writes
// the workspace planes).
template <typename Kernel>
auto MakeKargsAndGrids(const fmha_batch_prefill_args& args)
{
    if constexpr(kKvSplits == 2)
        return fmha_batch_prefill_create_split_kargs_and_grids<Kernel>(args);
    else
        return fmha_batch_prefill_create_kargs_and_grids<Kernel>(args);
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
                     fake(0xa0000),
                     nullptr,
                     fake(0xb0000),
                     fake(0xc0000),
                     fake(0xd0000)};
    const auto [base, grids] = MakeKargsAndGrids<Kernel>(MakeArgs<TypeParam>(l, d));
    const index_t need       = Kernel::GetLaunchDynamicSmemSize();

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
    expect_refused(
        "num_total_pages must be positive", need, [](Kargs& k) { k.num_total_pages = 0; });
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
        expect_refused("batch_stride_block_table must be positive", need, [](Kargs& k) {
            k.page_table.batch_stride_block_table = 0;
        });
    }
    expect_refused("seqstart_q_ptr is null", need, [](Kargs& k) { k.seqstart_q_ptr = nullptr; });
    if constexpr(kLse || kKvSplits == 2)
        expect_refused("lse_ptr is null", need, [](Kargs& k) { k.lse_ptr = nullptr; });
    if constexpr(kKvSplits == 2)
    {
        const char* rule = "the KV split needs the workspace's split strides";
        expect_refused(rule, need, [](Kargs& k) { k.split_stride_o = 0; });
        expect_refused(rule, need, [](Kargs& k) { k.split_stride_lse = 0; });
    }
    expect_refused("page stride smaller", need, [](Kargs& k) { k.batch_stride_k = k.stride_k; });
    expect_refused("page stride smaller", need, [](Kargs& k) { k.batch_stride_v = k.stride_v; });
    if constexpr(kVectorized)
    {
        // The token stride is implied by the layout; page and head strides keep 8-wide vectors.
        expect_refused("stride not a multiple of 8", need, [](Kargs& k) { k.nhead_stride_k += 1; });
        expect_refused("stride not a multiple of 8", need, [](Kargs& k) { k.batch_stride_v += 4; });
        expect_refused("token stride not positive", need, [](Kargs& k) { k.stride_k = 0; });
    }
    else
    {
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
    }
    if constexpr(TypeParam::kCausal)
    {
        if constexpr(kKvSplits == 2)
        {
            // Each KV split shifts a bottom-right window to its keys: the top-left corner and a
            // left bound are refused.
            const char* rule = "the KV split takes bottom-right masks without a left window";
            expect_refused(rule, need, [](Kargs& k) { SetTopLeft(k); });
            expect_refused(rule, need, [](Kargs& k) { k.window_size_left = 64; });
        }
        else
        {
            // Any window and corner is accepted; only a missing mask type is refused.
            EXPECT_EQ(refusal(need, [](Kargs& k) { SetTopLeft(k); }), "");
            EXPECT_EQ(refusal(need, [](Kargs& k) { k.window_size_left = 64; }), "");
        }
        EXPECT_EQ(refusal(need,
                          [](Kargs& k) {
                              k.mask_type = ck_tile::GenericAttentionMaskEnum::MASK_GENERIC;
                          }),
                  "");
        EXPECT_EQ(refusal(need, [](Kargs& k) { k.window_size_right = 1; }), "");
        expect_refused("a masked instance needs a mask type", need, [](Kargs& k) {
            k.mask_type = ck_tile::GenericAttentionMaskEnum::NO_MASK;
        });
        if constexpr(kSink)
            EXPECT_EQ(refusal(need, [](Kargs& k) { k.sink_size = 4; }), "");
        else
            expect_refused(
                "sink tokens need a sink instance", need, [](Kargs& k) { k.sink_size = 4; });
    }
    if constexpr(kBias == 1)
        expect_refused("an elementwise-bias instance needs bias_ptr", need, [](Kargs& k) {
            k.bias_ptr = nullptr;
        });
    if constexpr(kBias == 2)
        expect_refused("an ALiBi instance needs the slopes", need, [](Kargs& k) {
            k.alibi_slope_ptr = nullptr;
        });
    if constexpr(kSink)
        EXPECT_EQ(refusal(need, [&](Kargs& k) { k.sink_ptr = fake(0xa0000); }), "");
    else
        expect_refused("sink logits need a sink instance", need, [&](Kargs& k) {
            k.sink_ptr = fake(0xa0000);
        });

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
    std::vector<float> sink_logits; // per query head, in the scaled score domain
    // Elementwise bias [Hq, total_q, bias_stride] (kBias == 1) and ALiBi slopes [Hq] (kBias == 2).
    std::vector<Data> bias;
    int bias_stride = 0;
    std::vector<float> alibi_slopes;
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
// instead, so a read through a spare entry reads outside the pool (a wrong output or a fault).
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
    if constexpr(kBias == 1)
    {
        h.bias_stride = *std::max_element(l.seqlen_k.begin(), l.seqlen_k.end());
        std::uniform_real_distribution<float> bias_value(-2.0f, 2.0f);
        h.bias.resize(static_cast<std::size_t>(l.q_heads) * l.seqstart_q.back() * h.bias_stride);
        for(auto& x : h.bias)
            x = ck_tile::type_convert<Data>(bias_value(rng));
    }
    if constexpr(kBias == 2)
        h.alibi_slopes = ck_tile::get_alibi_slopes<float>(l.q_heads);
    // Sink logits around the score range, with one head at -inf (that head has no sink).
    for(int head = 0; head < h.layout.q_heads; ++head)
        h.sink_logits.push_back(head % 4 == 3 ? -std::numeric_limits<float>::infinity()
                                              : 0.75f * static_cast<float>(head % 5) - 1.0f);
    return h;
}

struct DeviceProblem
{
    ck_tile::DeviceMem q, k, v, o, seqstart_q, seqlen_k, block_table, kv_indptr, kv_last_page_lens,
        lse, sink, bias, o_acc, lse_acc;

    // (head, split, query) rows of the KV-split workspace (see MakeArgs; one row without a
    // split).
    static std::size_t WorkspaceRows(const Layout& l)
    {
        return kKvSplits == 2 ? static_cast<std::size_t>(l.q_heads) * 2 * l.seqstart_q.back() : 1;
    }

    // The block_table buffer holds the vLLM table, or the SGLang kv_page_indices.
    static const std::vector<int32_t>& Pages(const Layout& l)
    {
        return kSglang ? l.kv_page_indices : l.block_table;
    }

    // Linear pool [page][token][head][D] -> vectorized page-heads: K [page][head][D/8][P][8],
    // V [page][head][P/8][D][8]. The page stride is unchanged. V rows that no sequence reads
    // (kUnusedRowValue, including the tail of each sequence's last page) become NaN: a partial
    // group of 8 tokens is read past seqlen_k unless the kernel keeps those rows out of LDS, and
    // a zero probability does not hide a NaN.
    static std::vector<Data>
    Vectorize(const HostProblem& h, const std::vector<Data>& linear, bool is_k)
    {
        const std::size_t H = h.layout.kv_heads, D = h.head_dim, P = kPage;
        std::vector<Data> out(linear.size());
        for(std::size_t page = 0; page < static_cast<std::size_t>(h.layout.num_total_pages); ++page)
            for(std::size_t t = 0; t < P; ++t)
                for(std::size_t head = 0; head < H; ++head)
                    for(std::size_t d = 0; d < D; ++d)
                    {
                        const std::size_t base = (page * H + head) * P * D;
                        const std::size_t dst  = is_k ? base + ((d / 8) * P + t) * 8 + d % 8
                                                      : base + ((t / 8) * D + d) * 8 + t % 8;
                        const Data x           = linear[((page * P + t) * H + head) * D + d];
                        out[dst] = !is_k && ck_tile::type_convert<float>(x) == kUnusedRowValue
                                       ? ck_tile::type_convert<Data>(
                                             std::numeric_limits<float>::quiet_NaN())
                                       : x;
                    }
        return out;
    }

    // page_shift > 0 places every page page_shift pages further into a larger pool (and shifts the
    // page ids to match), so the same logical problem is addressed through large page offsets.
    static std::size_t PoolBytes(const HostProblem& h, std::int64_t page_shift)
    {
        const std::size_t page_bytes =
            static_cast<std::size_t>(kPage) * h.layout.kv_heads * h.head_dim * sizeof(Data);
        return h.k.size() * sizeof(Data) + static_cast<std::size_t>(page_shift) * page_bytes;
    }

    explicit DeviceProblem(const HostProblem& h, std::int64_t page_shift = 0)
        : q(h.q.size() * sizeof(Data)),
          k(PoolBytes(h, page_shift)),
          v(PoolBytes(h, page_shift)),
          o(h.q.size() * sizeof(Data)),
          seqstart_q(h.layout.seqstart_q.size() * sizeof(int32_t)),
          seqlen_k(h.layout.seqlen_k.size() * sizeof(int32_t)),
          block_table(std::max<std::size_t>(Pages(h.layout).size(), 1) * sizeof(int32_t)),
          kv_indptr(h.layout.kv_indptr.size() * sizeof(int32_t)),
          kv_last_page_lens(std::max<std::size_t>(h.layout.kv_last_page_lens.size(), 1) *
                            sizeof(int32_t)),
          lse(std::max<std::size_t>(
                  static_cast<std::size_t>(h.layout.q_heads) * h.layout.seqstart_q.back(), 1) *
              sizeof(float)),
          sink(std::max<std::size_t>(h.sink_logits.size(), 1) * sizeof(float)),
          bias(std::max<std::size_t>(
              std::max(h.bias.size() * sizeof(Data), h.alibi_slopes.size() * sizeof(float)), 4)),
          o_acc(WorkspaceRows(h.layout) * h.layout.q_heads * h.head_dim * sizeof(float)),
          lse_acc(WorkspaceRows(h.layout) * h.layout.q_heads * sizeof(float))
    {
        if(!h.bias.empty())
            bias.ToDevice(h.bias.data());
        if(!h.alibi_slopes.empty())
            bias.ToDevice(h.alibi_slopes.data());
        q.ToDevice(h.q.data());
        if(!h.sink_logits.empty())
            sink.ToDevice(h.sink_logits.data());
        const std::size_t shift_bytes = PoolBytes(h, page_shift) - h.k.size() * sizeof(Data);
        const std::vector<Data> k_pool =
            kVectorized ? Vectorize(h, h.k, true) : std::vector<Data>{};
        const std::vector<Data> v_pool =
            kVectorized ? Vectorize(h, h.v, false) : std::vector<Data>{};
        for(auto [pool, host] : {std::pair{&k, kVectorized ? &k_pool : &h.k},
                                 std::pair{&v, kVectorized ? &v_pool : &h.v}})
            EXPECT_EQ(hipMemcpy(static_cast<char*>(pool->GetDeviceBuffer()) + shift_bytes,
                                host->data(),
                                host->size() * sizeof(Data),
                                hipMemcpyHostToDevice),
                      hipSuccess);
        seqstart_q.ToDevice(h.layout.seqstart_q.data());
        seqlen_k.ToDevice(h.layout.seqlen_k.data());
        // Valid ids move with the pages; out-of-range (poisoned) ids stay out of range.
        std::vector<int32_t> pages = Pages(h.layout);
        for(auto& id : pages)
            if(id >= 0)
                id = static_cast<int32_t>(id + page_shift);
        if(!pages.empty())
            block_table.ToDevice(pages.data());
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
                lse.GetDeviceBuffer(),
                sink.GetDeviceBuffer(),
                bias.GetDeviceBuffer(),
                o_acc.GetDeviceBuffer(),
                lse_acc.GetDeviceBuffer()};
    }
};

// Runs one checked launch on a NaN-filled output (and LSE) and returns the output; with an LSE
// kernel the [Hq, total_q] LSE is returned through lse_out when given.
template <typename M>
std::vector<Data> Run(const HostProblem& h,
                      DeviceProblem& d,
                      Window window               = {},
                      std::vector<float>* lse_out = nullptr,
                      std::int64_t page_shift     = 0)
{
    using Kernel = typename M::Kernel;
    std::vector<Data> o(h.q.size(),
                        ck_tile::type_convert<Data>(std::numeric_limits<float>::quiet_NaN()));
    d.o.ToDevice(o.data());
    std::vector<float> lse(static_cast<std::size_t>(h.layout.q_heads) * h.layout.seqstart_q.back(),
                           std::numeric_limits<float>::quiet_NaN());
    if(kLse && !lse.empty())
        d.lse.ToDevice(lse.data());
    Layout layout = h.layout;
    layout.num_total_pages += static_cast<int>(page_shift);
    const auto args = MakeArgs<M>(layout, d.pointers(), window);
    const ck_tile::stream_config once{nullptr, false, 0, 0, 1};
    if constexpr(kKvSplits == 2)
    {
        // All-ones bytes are NaN: a row that the merge reads but no split wrote reaches O.
        EXPECT_EQ(hipMemset(d.o_acc.GetDeviceBuffer(), 0xff, d.o_acc.GetBufferSize()), hipSuccess);
        EXPECT_EQ(hipMemset(d.lse_acc.GetDeviceBuffer(), 0xff, d.lse_acc.GetBufferSize()),
                  hipSuccess);
        using Merge = typename M::Merge;
        if constexpr(kQueryTailRows > 0)
            fmha_batch_prefill_tail_launch<typename M::MainKernel, Kernel, Merge>(once, args);
        else
            fmha_batch_prefill_split_launch<Kernel, Merge>(once, args);
    }
    else
    {
        const auto [kargs, grids] = fmha_batch_prefill_create_kargs_and_grids<Kernel>(args);
        ck_tile::launch_kernel(once, Kernel::MakeCheckedKernel(kargs, grids));
    }
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
                                        Window window                                = {},
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
        if(kSoftcap)
            s.score_transform = [](std::size_t, std::size_t, std::size_t, float x) {
                return kSoftcapValue * std::tanh(x / kSoftcapValue);
            };
        if constexpr(kBias == 1)
        {
            const Data* bias       = h.bias.data();
            const std::size_t rows = static_cast<std::size_t>(l.seqstart_q.back());
            const std::size_t q0   = static_cast<std::size_t>(l.seqstart_q[b]);
            const std::size_t ld   = static_cast<std::size_t>(h.bias_stride);
            s.score_transform      = [bias, rows, q0, ld](
                                    std::size_t head, std::size_t query, std::size_t key, float x) {
                return x +
                       ck_tile::type_convert<float>(bias[(head * rows + q0 + query) * ld + key]);
            };
        }
        if constexpr(kBias == 2)
        {
            // CK's ALiBi: + slope * key under the causal window (vertical mode); otherwise
            // - slope * |diagonal - key| from the mask corner (bottom-right without a mask).
            const auto slopes    = h.alibi_slopes;
            const int64_t offset = causal && window.top_left ? 0 : k_len - q_len;
            const bool vertical  = causal && window.left < 0 && window.right == 0;
            s.score_transform    = [slopes, offset, vertical](
                                    std::size_t head, std::size_t query, std::size_t key, float x) {
                const float slope = slopes[head];
                if(vertical)
                    return x + slope * static_cast<float>(key);
                const int64_t distance =
                    static_cast<int64_t>(query) + offset - static_cast<int64_t>(key);
                return x - slope * static_cast<float>(distance < 0 ? -distance : distance);
            };
        }
        if(causal)
        {
            // Key k is visible to query q when q + offset - left <= k <= q + offset + right.
            const int64_t offset = window.top_left ? 0 : k_len - q_len;
            // Sink keys (key < sink) stay visible below the right bound when the window has a
            // left bound that does not already reach key 0, for query rows below k_len - the
            // sink rule of the existing CK mask (SimplifiedGenericAttentionMask::IsOutOfSinkBound).
            const int64_t q_rows = q_len;
            const int64_t k_rows = k_len;
            s.is_masked = [offset, window, q_rows, k_rows](std::size_t query, std::size_t key) {
                const int64_t diagonal = static_cast<int64_t>(query) + offset;
                const int64_t k        = static_cast<int64_t>(key);
                const bool past_right  = window.right >= 0 && k > diagonal + window.right;
                const bool in_window =
                    !past_right && !(window.left >= 0 && k < diagonal - window.left);
                const int64_t y    = 1 + window.left + (window.top_left ? 0 : q_rows - k_rows);
                const bool in_sink = k < window.sink && !past_right && window.left >= 0 &&
                                     y < q_rows && static_cast<int64_t>(query) < k_rows;
                return !(in_window || in_sink);
            };
            if(window.sink_logits)
                s.sink_logits = h.sink_logits;
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
                const auto o       = Run<TypeParam>(h, d, Window{}, &lse);
                const auto metrics = CompareWithReference(
                    h, o, TypeParam::kCausal, {}, Window{}, kLse ? &lse : nullptr);
                std::cout << "page_tile P" << kPage << " D" << TypeParam::kHeadDim
                          << (TypeParam::kCausal ? " causal " : " dense ")
                          << (kSglang ? "sglang " : "") << (kKvSplits == 2 ? "split " : "") << name
                          << " rows=" << metrics.checked_rows
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
    else if constexpr(kKvSplits == 2)
        GTEST_SKIP() << "the KV split takes bottom-right masks (see CheckLaunch)";
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
                const auto o       = Run<TypeParam>(h, d, Window{true});
                const auto metrics = CompareWithReference(h, o, true, {}, Window{true});
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

// Sliding and local windows on the masked model, from either corner: windows smaller than,
// equal to and larger than the N128 tile, a two-sided window, a right lookahead without a left
// bound and a left bound without a right one. Long keys against short queries make the first
// key tile of a query tile start past zero (kv_load_start > 0); queries longer than keys leave
// rows with no visible key.
TYPED_TEST(PageTileForward, MatchesReferenceWindow)
{
    if constexpr(!TypeParam::kCausal)
        GTEST_SKIP() << "windows apply to the masked model";
    else if constexpr(kKvSplits == 2)
        GTEST_SKIP() << "the KV split takes no left window (see CheckLaunch)";
    else
    {
        const std::vector<Case> shapes = {
            {"gqa2", 4, 2, {300, 13, 129, 64}, {2 * kPage + 37, kPage - 3, 129, 384}},
            {"long", 4, 2, {300, 129}, {3000, 1500}},
            {"tiny", 4, 2, {1, 130, 128, 200}, {1, 100, 128, 100}},
        };
        const std::vector<std::pair<const char*, Window>> windows = {
            {"diagonal", {false, 0, 0}},
            {"swa37", {false, 37, 0}},
            {"swa128", {false, 128, 0}},
            {"swa200", {false, 200, 0}},
            {"swa64 top-left", {true, 64, 0}},
            {"local100+30", {false, 100, 30}},
            {"lookahead129 top-left", {true, -1, 129}},
            {"left1000", {false, 1000, -1}},
        };
        std::uint32_t seed = 61;
        for(const auto& shape : shapes)
        {
            const auto h =
                MakeProblem(shape, Mapping::Permuted, TypeParam::kHeadDim, false, seed++);
            DeviceProblem d(h);
            for(const auto& [window_name, window] : windows)
            {
                const std::string name = std::string(shape.name) + " " + window_name;
                SCOPED_TRACE(name);
                const auto o       = Run<TypeParam>(h, d, window);
                const auto metrics = CompareWithReference(h, o, true, {}, window);
                std::cout << "page_tile P" << kPage << " D" << TypeParam::kHeadDim << " window "
                          << (kSglang ? "sglang " : "") << name << " rows=" << metrics.checked_rows
                          << " max_abs=" << metrics.max_abs_error
                          << " min_cos=" << metrics.min_row_cosine
                          << " nonfinite=" << metrics.nonfinite_output_count
                          << " passed=" << metrics.passed << '\n';
                EXPECT_EQ(metrics.checked_rows, LogicalRows(h.layout));
                EXPECT_EQ(metrics.nonfinite_output_count, 0u);
                EXPECT_TRUE(metrics.passed);
            }
        }
    }
}

// Sink instance: sink tokens (sink_size) with sliding windows, so the key loop visits the sink
// tiles first and then jumps to the window (the page lookup must follow the jump), sinks spanning
// one and two key tiles, and per-head sink logits (one head at -inf) with and without sink tokens.
TYPED_TEST(PageTileForward, MatchesReferenceSink)
{
    if constexpr(!kSink)
        GTEST_SKIP() << "sink instance only";
    else
    {
        const std::vector<Case> shapes = {
            {"gqa2", 4, 2, {300, 13, 129, 64}, {2 * kPage + 37, kPage - 3, 129, 384}},
            {"long", 4, 2, {300, 129}, {3000, 1500}},
            {"tiny", 4, 2, {1, 130, 128, 200}, {1, 100, 128, 100}},
        };
        const std::vector<std::pair<const char*, Window>> windows = {
            {"swa100 sink4", {false, 100, 0, 4, false}},
            {"swa300 sink1", {false, 300, 0, 1, false}},
            {"swa37 sink200", {false, 37, 0, 200, false}},
            {"causal sink4", {false, -1, 0, 4, false}},
            {"swa64 top-left sink16", {true, 64, 0, 16, false}},
            {"local100+30 sink4", {false, 100, 30, 4, false}},
            {"swa100 logits", {false, 100, 0, 0, true}},
            {"causal logits", {false, -1, 0, 0, true}},
            {"swa100 sink4 logits", {false, 100, 0, 4, true}},
        };
        std::uint32_t seed = 71;
        for(const auto& shape : shapes)
        {
            const auto h =
                MakeProblem(shape, Mapping::Permuted, TypeParam::kHeadDim, false, seed++);
            DeviceProblem d(h);
            for(const auto& [window_name, window] : windows)
            {
                const std::string name = std::string(shape.name) + " " + window_name;
                SCOPED_TRACE(name);
                const auto o       = Run<TypeParam>(h, d, window);
                const auto metrics = CompareWithReference(h, o, true, {}, window);
                std::cout << "page_tile P" << kPage << " D" << TypeParam::kHeadDim << " sink "
                          << (kSglang ? "sglang " : "") << name << " rows=" << metrics.checked_rows
                          << " max_abs=" << metrics.max_abs_error
                          << " min_cos=" << metrics.min_row_cosine
                          << " nonfinite=" << metrics.nonfinite_output_count
                          << " passed=" << metrics.passed << '\n';
                EXPECT_EQ(metrics.checked_rows, LogicalRows(h.layout));
                EXPECT_EQ(metrics.nonfinite_output_count, 0u);
                EXPECT_TRUE(metrics.passed);
            }
        }
    }
}

// Sink instance: no attended key falls in the pages between the sink tile and the first window
// tile, so their table entries may hold anything. Out-of-range ids there must leave the output
// bitwise equal to the plain table's, and correct.
TYPED_TEST(PageTileForward, SkippedSinkPages)
{
    if constexpr(!kSink)
        GTEST_SKIP() << "sink instance only";
    else
    {
        // A copy of h whose table entries of the pages lying entirely in keys [begin, end) of
        // every sequence hold `id` (the vLLM table and the SGLang list alike). The pool, the
        // logical data and h's own table stay as they are, so the reference of h still describes
        // the problem.
        const auto with_table_pages = [](HostProblem h, int begin, int end, int32_t id) {
            auto& l = h.layout;
            for(int b = 0; b < l.batch(); ++b)
                for(int page = 0; page * kPage < l.seqlen_k[b]; ++page)
                    if(page * kPage >= begin && (page + 1) * kPage <= end)
                    {
                        l.block_table[b * l.table_stride + page] = id;
                        l.kv_page_indices[l.kv_indptr[b] + page] = id;
                    }
            return h;
        };
        // One sequence, bottom-right: the first query row attends keys from
        // (3000 - 300) - 300 = 2400 on, so the key tiles between the sink tile (keys 0..127) and
        // the tile of key 2400 are never visited.
        const Case shape{"long", 4, 2, {300}, {3000}};
        const Window window{false, 300, 0, 4, false};
        const int first_window_key = (3000 - 300) - 300;
        const int skipped_begin    = 128;
        const int skipped_end      = first_window_key / 128 * 128;
        const auto h = MakeProblem(shape, Mapping::Permuted, TypeParam::kHeadDim, false, 73);
        DeviceProblem base_device(h);
        const auto base_output = Run<TypeParam>(h, base_device, window);
        const auto poisoned =
            with_table_pages(h, skipped_begin, skipped_end, std::numeric_limits<int32_t>::min());
        ASSERT_NE(poisoned.layout.block_table, h.layout.block_table) << "no page was skipped";
        DeviceProblem d(poisoned);
        const auto o       = Run<TypeParam>(poisoned, d, window);
        const auto metrics = CompareWithReference(h, o, true, {}, window);
        std::cout << "page_tile P" << kPage << " D" << TypeParam::kHeadDim << " sink "
                  << (kSglang ? "sglang " : "") << "skipped pages rows=" << metrics.checked_rows
                  << " max_abs=" << metrics.max_abs_error << " passed=" << metrics.passed << '\n';
        EXPECT_EQ(metrics.checked_rows, LogicalRows(h.layout));
        EXPECT_EQ(metrics.nonfinite_output_count, 0u);
        EXPECT_TRUE(metrics.passed);
        ASSERT_EQ(o.size(), base_output.size());
        EXPECT_EQ(std::memcmp(o.data(), base_output.data(), o.size() * sizeof(Data)), 0)
            << "the table entries of skipped pages changed the output";
    }
}

// An elementwise bias can carry a materialized mask. Here it is -inf on the first key tile (keys
// 0..127) of every row, so that tile has no finite score: the softmax must keep a finite exponent
// base, as with a mask, instead of forming -inf - (-inf). V is 0 on those keys and 1 on the
// others, so each output component is the softmax mass past the first tile: 1, or 0 for a row
// that sees no key past it (every row of the last sequence).
TYPED_TEST(PageTileForward, MaskedByBias)
{
    if constexpr(kBias != 1)
        GTEST_SKIP() << "elementwise-bias instances only";
    else
    {
        constexpr int kMaskedKeys = 128;
        const Case shape{"bias_mask",
                         4,
                         2,
                         {64, 1, 128, 7},
                         {2 * kMaskedKeys + 37, kMaskedKeys + 1, 3 * kMaskedKeys, 100}};
        auto h = MakeProblem(shape, Mapping::Permuted, TypeParam::kHeadDim, TypeParam::kCausal, 11);
        const auto& l         = h.layout;
        const std::size_t row = static_cast<std::size_t>(l.kv_heads) * h.head_dim;
        for(int b = 0; b < l.batch(); ++b)
            for(int key = 0; key < l.seqlen_k[b]; ++key)
            {
                const auto page =
                    static_cast<std::size_t>(l.block_table[b * l.table_stride + key / kPage]);
                std::fill_n(h.v.begin() + (page * kPage + key % kPage) * row,
                            row,
                            ck_tile::type_convert<Data>(key < kMaskedKeys ? 0.0f : 1.0f));
            }
        const std::size_t bias_rows = static_cast<std::size_t>(l.q_heads) * l.seqstart_q.back();
        for(std::size_t r = 0; r < bias_rows; ++r)
            std::fill_n(h.bias.begin() + r * h.bias_stride,
                        kMaskedKeys,
                        ck_tile::type_convert<Data>(-std::numeric_limits<float>::infinity()));
        DeviceProblem d(h);
        const auto o = Run<TypeParam>(h, d);
        // P is rounded to the 16-bit type, and so is the output.
        const float tolerance  = std::ldexp(2.0f, -kMantissaBits);
        std::size_t mismatches = 0;
        for(int b = 0; b < l.batch(); ++b)
            for(int q = 0; q < l.q_length(b); ++q)
            {
                // Bottom-right causal: query q sees the keys up to q + seqlen_k - seqlen_q.
                const int last_key =
                    TypeParam::kCausal ? q + l.seqlen_k[b] - l.q_length(b) : l.seqlen_k[b] - 1;
                const float expected = last_key >= kMaskedKeys ? 1.0f : 0.0f;
                const std::size_t first =
                    (static_cast<std::size_t>(l.seqstart_q[b]) + q) * l.q_heads * h.head_dim;
                for(std::size_t x = first; x < first + l.q_heads * h.head_dim; ++x)
                    mismatches +=
                        !(std::abs(ck_tile::type_convert<float>(o[x]) - expected) <= tolerance);
            }
        std::cout << "page_tile P" << kPage << " D" << TypeParam::kHeadDim
                  << (TypeParam::kCausal ? " causal" : " dense")
                  << " bias mask mismatches=" << mismatches << '\n';
        EXPECT_EQ(mismatches, 0u);
    }
}

// The kernel must not read block-table entries past a sequence's last page. Poisoned spare entries
// make such a read go outside the pool; a tight stride (no spare entries) makes the longest row end
// at the end of its table row. Both change only entries the kernel must not read, so the output
// must be bitwise equal to the default table's.
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

// The same logical problem in a pool whose pages start past 5 GiB (page offsets beyond the 32-bit
// byte range) must give a bitwise equal output: page addressing is 64-bit end to end.
TYPED_TEST(PageTileForward, LargePool)
{
    const Case shape{"large", 4, 2, {130, 64}, {2 * kPage + 37, 384}};
    const auto h =
        MakeProblem(shape, Mapping::Permuted, TypeParam::kHeadDim, TypeParam::kCausal, 77);
    DeviceProblem small(h);
    const auto base = Run<TypeParam>(h, small);
    const std::size_t page_bytes =
        static_cast<std::size_t>(kPage) * h.layout.kv_heads * TypeParam::kHeadDim * sizeof(Data);
    const std::int64_t shift = static_cast<std::int64_t>((std::size_t{5} << 30) / page_bytes) + 1;
    DeviceProblem large(h, shift);
    const auto o = Run<TypeParam>(h, large, Window{}, nullptr, shift);
    std::cout << "page_tile P" << kPage << " D" << TypeParam::kHeadDim
              << (TypeParam::kCausal ? " causal" : " dense") << " large pool: page shift " << shift
              << " (" << static_cast<double>(shift) * page_bytes / (1u << 30) << " GiB)\n";
    ASSERT_EQ(o.size(), base.size());
    EXPECT_EQ(std::memcmp(o.data(), base.data(), o.size() * sizeof(Data)), 0)
        << "pages past 5 GiB changed the output of the same logical problem";
}

// Repeated launches of a device-filling grid must give bitwise equal outputs, and equal to the
// same logical problem in an identity-mapped pool. This guards the residency limit of the launch
// contract: at this shape a P16/D64 launch that requests no dynamic LDS (three resident
// workgroups per WGP) gave a different output on every launch in out-of-tree measurements. A
// sampled set of rows is also checked against the reference.
TYPED_TEST(PageTileForward, Deterministic)
{
    // The bias would be [32, 8192, 32768] (17 GB); the bias does not change the page addressing or
    // the TDM timing this test exercises, which the unbiased models cover.
    if constexpr(kBias == 1)
        GTEST_SKIP() << "elementwise bias: covered by the unbiased models";
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
