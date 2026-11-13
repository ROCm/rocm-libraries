// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "fmha_reference.hpp"
#include "fmha_test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using namespace ck_tile;

// Instantiate both pipelines on the same problem so their names can be
// compared. The geometry is the only one this release supports:
// hdim=128, hdim_v=128, kM0=128, kN0=128, kK0=32, kN1=128, kK1=32.
using TestProblem = ck_tile::test::Model<128, false, false, false, bf16_t, 128, 128>::Problem;
using TestPolicy  = ck_tile::test::Policy<TestProblem>;

using DensePipelineForTest = BlockFmhaPipelineQRKSVSTdmSched<TestProblem, TestPolicy>;
using PagedPipelineForTest =
    BlockFmhaFwdPagedKVPipelineQRKSVSTdmSched<TestProblem, false, TestPolicy>;

TEST(QrTdmSchedPaged, PipelineNameDiffersFromDense)
{
    constexpr const char* paged = PagedPipelineForTest::name;
    constexpr const char* dense = DensePipelineForTest::name;
    ASSERT_NE(paged, nullptr);
    ASSERT_NE(dense, nullptr);
    EXPECT_STRNE(paged, dense)
        << "paged and dense qr_tdm_sched share a name; codegen blobs will collide";
}

// ---------------------------------------------------------------------------
// Shared harness for Task 4/5/6/7/8: run the dense and paged qr_tdm_sched
// kernels on identical logical Q/K/V content and compare the outputs.
// ---------------------------------------------------------------------------

using Data              = ck_tile::bf16_t;
constexpr int kHeadDim  = 128;
constexpr int kHeadDimV = 128;

void CheckHip(hipError_t rc)
{
    if(rc != hipSuccess)
        throw std::runtime_error(hipGetErrorString(rc));
}

template <typename T>
struct Buffer
{
    T* ptr = nullptr;
    explicit Buffer(std::size_t count)
    {
        CheckHip(hipMalloc(&ptr, std::max(count, std::size_t{1}) * sizeof(T)));
    }
    ~Buffer()
    {
        if(ptr)
            (void)hipFree(ptr);
    }
    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;
    void Upload(const std::vector<T>& values)
    {
        if(!values.empty())
            CheckHip(
                hipMemcpy(ptr, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    void Download(std::vector<T>& values)
    {
        if(!values.empty())
            CheckHip(
                hipMemcpy(values.data(), ptr, values.size() * sizeof(T), hipMemcpyDeviceToHost));
    }
};

// Plain (batch, head, seq, dim) contiguous layout: heads do not interleave
// within a row. Used for both the dense tensors and the logical (pre-paging)
// view of K/V -- the paged side scatters this same logical content into
// per-page-per-head physical blocks using the same row/dim convention.
std::size_t Offset(int b, int h, int s, int d, int heads, int rows, int dim)
{
    return ((static_cast<std::size_t>(b) * heads + h) * rows + s) * dim + d;
}

// Deterministic sin/cos based generator so dense and paged runs can be given
// bit-identical logical content from independent calls with the same
// (batch, heads, rows, dim, phase) arguments.
std::vector<Data> make_input(int batch, int heads, int rows, int dim, int phase)
{
    std::vector<Data> values(static_cast<std::size_t>(batch) * heads * rows * dim);
    for(int b = 0; b < batch; ++b)
        for(int h = 0; h < heads; ++h)
            for(int s = 0; s < rows; ++s)
                for(int d = 0; d < dim; ++d)
                {
                    const float value =
                        0.35f * std::sin((s + 1) * (0.061f + phase * 0.017f) + (d + 1) * 0.137f +
                                         h * 0.53f + b * 0.29f) +
                        0.11f * std::cos((s + 1) * (d % 7 + 1) * 0.031f + phase * 0.4f);
                    values[Offset(b, h, s, d, heads, rows, dim)] =
                        ck_tile::type_convert<Data>(value);
                }
    return values;
}

std::vector<float> Decode(const std::vector<Data>& input)
{
    std::vector<float> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](Data value) {
        return ck_tile::type_convert<float>(value);
    });
    return output;
}

std::vector<std::int32_t> make_shuffled_block_table(int num_pages)
{
    if(num_pages < 2)
        throw std::invalid_argument(
            "make_shuffled_block_table requires num_pages >= 2: no non-identity "
            "permutation exists for fewer than 2 pages, and an identity table "
            "would make every paged-vs-dense comparison in this file vacuous");

    std::vector<std::int32_t> table(static_cast<std::size_t>(num_pages));
    std::iota(table.begin(), table.end(), 0);

    auto is_identity = [&]() {
        for(std::size_t p = 0; p < table.size(); ++p)
            if(table[p] != static_cast<std::int32_t>(p))
                return false;
        return true;
    };

    // A fixed-seed shuffle can land on the identity permutation for small n
    // (confirmed at n == 2, where {0, 1} and {1, 0} are equally likely). Keep
    // drawing from the same generator (still fully deterministic from the
    // fixed seed) until the result moves something; fall back to a rotate by
    // 1, which is non-identity for every n >= 2, if that still fails to move
    // anything within a handful of attempts (defensive; not expected to
    // trigger in practice).
    std::mt19937 gen{0xC0FFEE};
    int attempt = 0;
    do
    {
        std::shuffle(table.begin(), table.end(), gen);
    } while(is_identity() && ++attempt < 8);
    if(is_identity())
        std::rotate(table.begin(), table.begin() + 1, table.end());

    return table;
}

enum class MaskKind
{
    None,
    Causal,
    SlidingWindow
};

// (window_size_left, window_size_right) for a MaskKind, same convention as LaunchBatch.
std::pair<ck_tile::index_t, ck_tile::index_t> WindowFor(MaskKind kind, int window)
{
    if(kind == MaskKind::Causal)
        return {-1, 0};
    if(kind == MaskKind::SlidingWindow)
        return {window, 0};
    return {-1, -1};
}

struct FwdResult
{
    std::vector<float> out;
    std::vector<float> lse;
    // False if a store reached the NaN-filled O padding (see OLayout).
    bool o_padding_intact = true;
};

// O buffer layout for LaunchBatch: row_pad extra elements per row and spare_rows extra rows per
// head, both NaN-filled so stores outside the logical rows/columns are detected. out is always
// returned compact as [batch][nhead][seqlen_q][hdim_v].
struct OLayout
{
    int row_pad    = 0;
    int spare_rows = 0;
};

// Guards against a comparison test passing vacuously because both kernels
// wrote nothing (e.g. an unlaunched kernel, an all-zero-initialized output
// buffer never actually written to): every ASSERT_EQ(dense.out[i],
// paged.out[i]) in this file would pass on all-zero buffers just as readily
// as on a genuine match.
void AssertNotAllZero(const std::vector<float>& values, const char* what)
{
    const bool any_nonzero =
        std::any_of(values.begin(), values.end(), [](float v) { return v != 0.0f; });
    ASSERT_TRUE(any_nonzero) << what << " is all zero; comparison would be vacuous";
}

void AssertAllFinite(const std::vector<float>& values, const char* what)
{
    const bool all_finite =
        std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
    ASSERT_TRUE(all_finite) << what << " contains a non-finite value";
}

using Epilogue =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<float, Data, true, true>>;

template <bool Group, bool Mask, bool Lse>
using BaseModelFor =
    typename ck_tile::test::Model<kHeadDim, Group, Mask, Lse, Data, kHeadDim, kHeadDimV>::Problem;

// ck_tile::test::Model has no Sink parameter of its own (every call site in
// this file predates sink support) and always builds its Traits with
// kHasSink=false. Reuse every other field off BaseModelFor via inheritance
// and override only kHasSink, so the dense problem's own Traits (not just
// the paged specialization below) actually compiles in the kHasSink==true
// branches instead of silently leaving them dead code, which is exactly the
// gap Task 6 found for masking one level removed.
template <bool Group, bool Mask, bool Sink, bool Lse>
struct TraitsFor : BaseModelFor<Group, Mask, Lse>::Traits
{
    static constexpr bool kHasSink = Sink;
};

template <bool Group, bool Mask, bool Sink, bool Lse>
using ProblemFor = ck_tile::BlockFmhaPipelineProblem<
    typename BaseModelFor<Group, Mask, Lse>::QDataType,
    typename BaseModelFor<Group, Mask, Lse>::KDataType,
    typename BaseModelFor<Group, Mask, Lse>::VDataType,
    typename BaseModelFor<Group, Mask, Lse>::SaccDataType,
    typename BaseModelFor<Group, Mask, Lse>::SMPLComputeDataType,
    typename BaseModelFor<Group, Mask, Lse>::BiasDataType,
    typename BaseModelFor<Group, Mask, Lse>::RandValOutputDataType,
    typename BaseModelFor<Group, Mask, Lse>::LSEDataType,
    typename BaseModelFor<Group, Mask, Lse>::PDataType,
    typename BaseModelFor<Group, Mask, Lse>::OaccDataType,
    typename BaseModelFor<Group, Mask, Lse>::ODataType,
    typename BaseModelFor<Group, Mask, Lse>::BlockFmhaShape,
    Group,
    typename BaseModelFor<Group, Mask, Lse>::AttentionVariant,
    typename BaseModelFor<Group, Mask, Lse>::FmhaMask,
    BaseModelFor<Group, Mask, Lse>::kUseTrLoad,
    TraitsFor<Group, Mask, Sink, Lse>>;

template <bool Group, bool Mask, bool Sink, bool Lse>
using PolicyFor = ck_tile::test::Policy<ProblemFor<Group, Mask, Sink, Lse>>;

template <bool Group, bool Mask, bool Sink, bool Lse>
using DensePipelineFor =
    ck_tile::BlockFmhaPipelineQRKSVSTdmSched<ProblemFor<Group, Mask, Sink, Lse>,
                                             PolicyFor<Group, Mask, Sink, Lse>>;

// FmhaFwdPagedKVKernel reads FmhaPipeline::Problem::kIsPagedKV (and
// ::kDoFp8StaticQuant) directly to pick its kargs layout (CommonPageBlockTableKargs
// vs CacheBatchIdxKargs, etc). ProblemFor's Traits (TraitsFor above) does not
// define either flag.
//
// TileFmhaFwdPagedKVTraits (the traits type the *other*, PageBlockNavigator-based
// paged pipelines pair with BlockFmhaFwdPagedKVPipelineProblem) does define them,
// but it is missing kHasDropout/QScaleEnum, which the generic BlockFmhaPipelineProblem
// our dense Problem (and this pipeline) is built on reads unconditionally -- so it
// cannot be dropped in as a straight replacement here. Instead, derive from the
// dense Problem's own Traits type (which already carries kHasSink=Sink) and add
// just the two extra flags on top; plain inheritance keeps every member
// BlockFmhaPipelineProblem already relies on, and the derived class's own
// kDoFp8StaticQuant/kIsPagedKV are picked up by BlockFmhaPipelineProblem's
// detail::traits_has_* detection.
template <bool Group, bool Mask, bool Sink, bool Lse>
struct PagedTraitsFor : ProblemFor<Group, Mask, Sink, Lse>::Traits
{
    static constexpr bool kDoFp8StaticQuant = false;
    static constexpr bool kIsPagedKV        = true;
};

template <bool Group, bool Mask, bool Sink, bool Lse>
using PagedProblemFor = ck_tile::BlockFmhaPipelineProblem<
    typename ProblemFor<Group, Mask, Sink, Lse>::QDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::KDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::VDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::SaccDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::SMPLComputeDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::BiasDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::RandValOutputDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::LSEDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::PDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::OaccDataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::ODataType,
    typename ProblemFor<Group, Mask, Sink, Lse>::BlockFmhaShape,
    Group,
    typename ProblemFor<Group, Mask, Sink, Lse>::AttentionVariant,
    typename ProblemFor<Group, Mask, Sink, Lse>::FmhaMask,
    ProblemFor<Group, Mask, Sink, Lse>::kUseTrLoad,
    PagedTraitsFor<Group, Mask, Sink, Lse>>;

template <bool Group, bool Mask, bool Sink, bool Lse>
using PagedPolicyFor = ck_tile::test::Policy<PagedProblemFor<Group, Mask, Sink, Lse>>;

template <bool Group, bool Mask, bool Sink, bool Lse, bool Gappy = false>
using PagedPipelineFor =
    ck_tile::BlockFmhaFwdPagedKVPipelineQRKSVSTdmSched<PagedProblemFor<Group, Mask, Sink, Lse>,
                                                       Gappy,
                                                       PagedPolicyFor<Group, Mask, Sink, Lse>>;

template <bool Group, bool Mask, bool Sink, bool Lse>
using DenseKernelFor = ck_tile::FmhaFwdKernel<DensePipelineFor<Group, Mask, Sink, Lse>, Epilogue>;

template <bool Group, bool Mask, bool Sink, bool Lse>
using PagedKernelFor =
    ck_tile::FmhaFwdPagedKVKernel<PagedPipelineFor<Group, Mask, Sink, Lse>, Epilogue>;

// Gappy group mode runs the variant that accepts any kv_l2p_offset (no sink).
template <bool Mask>
using PagedGappyKernelFor =
    ck_tile::FmhaFwdPagedKVKernel<PagedPipelineFor<true, Mask, false, false, true>, Epilogue>;

template <typename Kernel>
void LaunchDirect(const typename Kernel::Kargs& kargs,
                  const ck_tile::stream_config& stream,
                  dim3 grids)
{
    ck_tile::launch_kernel(stream,
                           ck_tile::make_kernel<Kernel::kBlockPerCu, ck_tile::gfx125_t>(
                               Kernel{}, grids, Kernel::BlockSize(), 0, kargs));
    CheckHip(hipGetLastError());
    CheckHip(hipDeviceSynchronize());
}

// Batch-mode (fixed seqlen across the batch) dense/paged launcher, templated
// on whether it should run the paged kernel, whether a mask is active,
// whether the compile-time sink branches are present, and whether LSE is
// written out.
template <bool IsPaged, bool Mask, bool Sink, bool Lse>
FwdResult LaunchBatch(int batch,
                      int nhead,
                      int seqlen_q,
                      int seqlen_k,
                      int hdim,
                      int page_block_size,
                      bool shuffle,
                      MaskKind mask_kind,
                      int window,
                      int sink,
                      const OLayout& o_layout = {})
{
    if(hdim != kHeadDim)
        throw std::invalid_argument("harness only supports hdim=128 (fixed tile geometry)");
    constexpr int hdim_v = kHeadDimV;

    const auto q          = make_input(batch, nhead, seqlen_q, kHeadDim, 0);
    const auto k          = make_input(batch, nhead, seqlen_k, kHeadDim, 1);
    const auto v          = make_input(batch, nhead, seqlen_k, hdim_v, 2);
    const bool o_padded   = o_layout.row_pad != 0 || o_layout.spare_rows != 0;
    const int o_row_elems = hdim_v + o_layout.row_pad;
    const int o_head_rows = seqlen_q + o_layout.spare_rows;
    std::vector<Data> o(
        static_cast<std::size_t>(batch) * nhead * o_head_rows * o_row_elems,
        ck_tile::type_convert<Data>(o_padded ? std::numeric_limits<float>::quiet_NaN() : 0.0f));
    std::vector<float> lse(static_cast<std::size_t>(batch) * nhead * seqlen_q, 0.0f);

    const float scale_s = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    const ck_tile::index_t stride_q       = kHeadDim;
    const ck_tile::index_t nhead_stride_q = static_cast<ck_tile::index_t>(seqlen_q) * kHeadDim;
    const ck_tile::index_t batch_stride_q = static_cast<ck_tile::index_t>(nhead) * nhead_stride_q;
    const ck_tile::index_t stride_o       = o_row_elems;
    const ck_tile::index_t nhead_stride_o =
        static_cast<ck_tile::index_t>(o_head_rows) * o_row_elems;
    const ck_tile::index_t batch_stride_o   = static_cast<ck_tile::index_t>(nhead) * nhead_stride_o;
    const ck_tile::index_t nhead_stride_lse = seqlen_q;
    const ck_tile::index_t batch_stride_lse = static_cast<ck_tile::index_t>(nhead) * seqlen_q;

    ck_tile::index_t window_left  = -1;
    ck_tile::index_t window_right = -1;
    ck_tile::index_t sink_size    = 0;
    ck_tile::index_t mask_type =
        static_cast<ck_tile::index_t>(ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
    if constexpr(Mask)
    {
        if(mask_kind == MaskKind::Causal)
        {
            window_left  = -1;
            window_right = 0;
        }
        else // SlidingWindow
        {
            window_left  = window;
            window_right = 0;
        }
        sink_size = sink;
    }

    Buffer<Data> qb(q.size()), ob(o.size());
    qb.Upload(q);
    ob.Upload(o);
    Buffer<float> lb(Lse ? lse.size() : std::size_t{1});
    if constexpr(Lse)
        lb.Upload(lse);

    const ck_tile::stream_config stream{nullptr, false, 0, 0, 1};

    if constexpr(IsPaged)
    {
        if(seqlen_k % page_block_size != 0)
            throw std::invalid_argument("harness requires seqlen_k % page_block_size == 0");
        const int pages_per_batch = seqlen_k / page_block_size;
        const int total_pages     = batch * pages_per_batch;

        const std::vector<std::int32_t> block_table = [&] {
            if(shuffle)
                return make_shuffled_block_table(total_pages);
            std::vector<std::int32_t> identity(static_cast<std::size_t>(total_pages));
            std::iota(identity.begin(), identity.end(), 0);
            return identity;
        }();

        const std::size_t page_elems_k =
            static_cast<std::size_t>(nhead) * page_block_size * kHeadDim;
        const std::size_t page_elems_v = static_cast<std::size_t>(nhead) * page_block_size * hdim_v;
        std::vector<Data> k_phys(static_cast<std::size_t>(total_pages) * page_elems_k,
                                 ck_tile::type_convert<Data>(0.0f));
        std::vector<Data> v_phys(static_cast<std::size_t>(total_pages) * page_elems_v,
                                 ck_tile::type_convert<Data>(0.0f));

        for(int b = 0; b < batch; ++b)
            for(int h = 0; h < nhead; ++h)
                for(int lp = 0; lp < pages_per_batch; ++lp)
                {
                    const std::int32_t physical =
                        block_table[static_cast<std::size_t>(b) * pages_per_batch + lp];
                    for(int row = 0; row < page_block_size; ++row)
                    {
                        const int s = lp * page_block_size + row;
                        const std::size_t page_base_k =
                            static_cast<std::size_t>(physical) * page_elems_k +
                            static_cast<std::size_t>(h) * page_block_size * kHeadDim +
                            static_cast<std::size_t>(row) * kHeadDim;
                        const std::size_t page_base_v =
                            static_cast<std::size_t>(physical) * page_elems_v +
                            static_cast<std::size_t>(h) * page_block_size * hdim_v +
                            static_cast<std::size_t>(row) * hdim_v;
                        for(int d = 0; d < kHeadDim; ++d)
                            k_phys[page_base_k + d] =
                                k[Offset(b, h, s, d, nhead, seqlen_k, kHeadDim)];
                        for(int d = 0; d < hdim_v; ++d)
                            v_phys[page_base_v + d] =
                                v[Offset(b, h, s, d, nhead, seqlen_k, hdim_v)];
                    }
                }

        Buffer<Data> kb(k_phys.size()), vb(v_phys.size());
        kb.Upload(k_phys);
        vb.Upload(v_phys);
        Buffer<std::int32_t> table_buf(block_table.size());
        table_buf.Upload(block_table);

        fmha_fwd_pagedkv_args args{};
        args.q_ptr    = qb.ptr;
        args.k_ptr    = kb.ptr;
        args.v_ptr    = vb.ptr;
        args.bias_ptr = nullptr;
        args.lse_ptr  = Lse ? static_cast<void*>(lb.ptr) : nullptr;
        args.o_ptr    = ob.ptr;

        args.block_table_ptr          = table_buf.ptr;
        args.batch_stride_block_table = pages_per_batch;
        args.page_block_size          = page_block_size;
        args.is_gappy                 = false;
        args.cache_batch_idx          = nullptr;

        args.seqstart_q_ptr = nullptr;
        args.seqstart_k_ptr = nullptr;
        args.seqlen_k_ptr   = nullptr;
        args.sink_ptr       = nullptr;

        args.seqlen_q     = seqlen_q;
        args.seqlen_k     = seqlen_k;
        args.batch        = batch;
        args.max_seqlen_q = seqlen_q;
        args.hdim_q       = kHeadDim;
        args.hdim_v       = hdim_v;
        args.nhead_q      = nhead;
        args.nhead_k      = nhead;

        args.scale_s         = scale_s;
        args.scale_p         = 1.0f;
        args.scale_o         = 1.0f;
        args.logits_soft_cap = 0.0f;

        args.stride_q    = stride_q;
        args.stride_k    = kHeadDim;
        args.stride_v    = hdim_v;
        args.stride_bias = 0;
        args.stride_o    = stride_o;

        args.nhead_stride_q    = nhead_stride_q;
        args.nhead_stride_k    = static_cast<ck_tile::index_t>(page_block_size) * kHeadDim;
        args.nhead_stride_v    = static_cast<ck_tile::index_t>(page_block_size) * hdim_v;
        args.nhead_stride_bias = 0;
        args.nhead_stride_lse  = nhead_stride_lse;
        args.nhead_stride_o    = nhead_stride_o;

        args.batch_stride_q    = batch_stride_q;
        args.batch_stride_k    = static_cast<ck_tile::index_t>(nhead) * args.nhead_stride_k;
        args.batch_stride_v    = static_cast<ck_tile::index_t>(nhead) * args.nhead_stride_v;
        args.batch_stride_bias = 0;
        args.batch_stride_lse  = batch_stride_lse;
        args.batch_stride_o    = batch_stride_o;

        args.window_size_left  = window_left;
        args.window_size_right = window_right;
        args.sink_size         = sink_size;
        args.mask_type         = mask_type;
        args.min_seqlen_q      = 0;

        using Kernel        = PagedKernelFor<false, Mask, Sink, Lse>;
        auto [kargs, grids] = fmha_fwd_pagedkv_create_kargs_and_grids<Kernel>(args);
        LaunchDirect<Kernel>(kargs, stream, grids);
    }
    else
    {
        Buffer<Data> kb(k.size()), vb(v.size());
        kb.Upload(k);
        vb.Upload(v);

        fmha_fwd_args args{};
        args.q_ptr    = qb.ptr;
        args.k_ptr    = kb.ptr;
        args.v_ptr    = vb.ptr;
        args.bias_ptr = nullptr;
        args.lse_ptr  = Lse ? static_cast<void*>(lb.ptr) : nullptr;
        args.o_ptr    = ob.ptr;
        args.sink_ptr = nullptr;

        args.seqlen_q     = seqlen_q;
        args.seqlen_k     = seqlen_k;
        args.batch        = batch;
        args.max_seqlen_q = seqlen_q;
        args.hdim_q       = kHeadDim;
        args.hdim_v       = hdim_v;
        args.nhead_q      = nhead;
        args.nhead_k      = nhead;

        args.scale_s         = scale_s;
        args.logits_soft_cap = 0.0f;

        args.stride_q = stride_q;
        args.stride_k = kHeadDim;
        args.stride_v = hdim_v;
        args.stride_o = stride_o;

        args.nhead_stride_q   = nhead_stride_q;
        args.nhead_stride_k   = static_cast<ck_tile::index_t>(seqlen_k) * kHeadDim;
        args.nhead_stride_v   = static_cast<ck_tile::index_t>(seqlen_k) * hdim_v;
        args.nhead_stride_lse = nhead_stride_lse;
        args.nhead_stride_o   = nhead_stride_o;

        args.batch_stride_q   = batch_stride_q;
        args.batch_stride_k   = static_cast<ck_tile::index_t>(nhead) * args.nhead_stride_k;
        args.batch_stride_v   = static_cast<ck_tile::index_t>(nhead) * args.nhead_stride_v;
        args.batch_stride_lse = batch_stride_lse;
        args.batch_stride_o   = batch_stride_o;

        args.window_size_left  = window_left;
        args.window_size_right = window_right;
        args.sink_size         = sink_size;
        args.mask_type         = mask_type;

        using Kernel        = DenseKernelFor<false, Mask, Sink, Lse>;
        auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<Kernel>(args);
        LaunchDirect<Kernel>(kargs, stream, grids);
    }

    std::vector<Data> o_result(o.size());
    ob.Download(o_result);
    if constexpr(Lse)
        lb.Download(lse);

    FwdResult result;
    if(o_padded)
    {
        std::vector<Data> compact(static_cast<std::size_t>(batch) * nhead * seqlen_q * hdim_v);
        for(int bh = 0; bh < batch * nhead; ++bh)
            for(int r = 0; r < o_head_rows; ++r)
                for(int c = 0; c < o_row_elems; ++c)
                {
                    const Data x =
                        o_result[(static_cast<std::size_t>(bh) * o_head_rows + r) * o_row_elems +
                                 c];
                    if(r < seqlen_q && c < hdim_v)
                        compact[(static_cast<std::size_t>(bh) * seqlen_q + r) * hdim_v + c] = x;
                    else if(!std::isnan(ck_tile::type_convert<float>(x)))
                        result.o_padding_intact = false;
                }
        result.out = Decode(compact);
    }
    else
    {
        result.out = Decode(o_result);
    }
    if constexpr(Lse)
        result.lse = lse;
    return result;
}

// Group-mode (variable seqlen per batch) dense/paged launcher. Fixed to a
// single head (nhead=1) since the brief's group signatures take no nhead
// argument; not exercised by any Task 4 test but kept functional for later
// tasks.
template <bool IsPaged, bool Mask = false>
FwdResult LaunchGroup(const std::vector<int>& seqlens_q,
                      const std::vector<int>& seqlens_k,
                      int hdim,
                      int page_block_size,
                      bool shuffle,
                      MaskKind mask_kind = MaskKind::None,
                      int window         = 0)
{
    if(hdim != kHeadDim)
        throw std::invalid_argument("harness only supports hdim=128 (fixed tile geometry)");
    if(seqlens_q.size() != seqlens_k.size())
        throw std::invalid_argument("seqlens_q/seqlens_k size mismatch");

    const int batch      = static_cast<int>(seqlens_q.size());
    constexpr int nhead  = 1;
    constexpr int hdim_v = kHeadDimV;

    std::vector<int> seqstart_q(static_cast<std::size_t>(batch) + 1, 0);
    std::vector<int> seqstart_k(static_cast<std::size_t>(batch) + 1, 0);
    for(int b = 0; b < batch; ++b)
    {
        seqstart_q[b + 1] = seqstart_q[b] + seqlens_q[b];
        seqstart_k[b + 1] = seqstart_k[b] + seqlens_k[b];
    }
    const int total_q      = seqstart_q[batch];
    const int total_k      = seqstart_k[batch];
    const int max_seqlen_q = *std::max_element(seqlens_q.begin(), seqlens_q.end());

    std::vector<Data> q(static_cast<std::size_t>(total_q) * kHeadDim);
    std::vector<Data> k(static_cast<std::size_t>(total_k) * kHeadDim);
    std::vector<Data> v(static_cast<std::size_t>(total_k) * hdim_v);
    for(int b = 0; b < batch; ++b)
    {
        const auto q_local = make_input(1, 1, seqlens_q[b], kHeadDim, b * 3 + 0);
        const auto k_local = make_input(1, 1, seqlens_k[b], kHeadDim, b * 3 + 1);
        const auto v_local = make_input(1, 1, seqlens_k[b], hdim_v, b * 3 + 2);
        std::copy(q_local.begin(),
                  q_local.end(),
                  q.begin() + static_cast<std::size_t>(seqstart_q[b]) * kHeadDim);
        std::copy(k_local.begin(),
                  k_local.end(),
                  k.begin() + static_cast<std::size_t>(seqstart_k[b]) * kHeadDim);
        std::copy(v_local.begin(),
                  v_local.end(),
                  v.begin() + static_cast<std::size_t>(seqstart_k[b]) * hdim_v);
    }
    std::vector<Data> o(static_cast<std::size_t>(total_q) * hdim_v,
                        ck_tile::type_convert<Data>(0.0f));

    const float scale_s = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    Buffer<Data> qb(q.size()), ob(o.size());
    qb.Upload(q);
    ob.Upload(o);
    std::vector<std::int32_t> seqstart_q32(seqstart_q.begin(), seqstart_q.end());
    std::vector<std::int32_t> seqstart_k32(seqstart_k.begin(), seqstart_k.end());
    Buffer<std::int32_t> seqstart_q_buf(seqstart_q32.size()), seqstart_k_buf(seqstart_k32.size());
    seqstart_q_buf.Upload(seqstart_q32);
    seqstart_k_buf.Upload(seqstart_k32);

    const ck_tile::index_t mask_type =
        static_cast<ck_tile::index_t>(ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
    const ck_tile::stream_config stream{nullptr, false, 0, 0, 1};

    if constexpr(IsPaged)
    {
        std::vector<int> pages_per_batch(static_cast<std::size_t>(batch));
        for(int b = 0; b < batch; ++b)
        {
            if(seqlens_k[b] % page_block_size != 0)
                throw std::invalid_argument(
                    "harness requires seqlen_k % page_block_size == 0 per batch");
            pages_per_batch[b] = seqlens_k[b] / page_block_size;
        }
        const int max_pages_per_batch =
            *std::max_element(pages_per_batch.begin(), pages_per_batch.end());
        std::vector<int> page_row_start(static_cast<std::size_t>(batch) + 1, 0);
        for(int b = 0; b < batch; ++b)
            page_row_start[b + 1] = page_row_start[b] + pages_per_batch[b];
        const int total_pages = page_row_start[batch];

        const std::vector<std::int32_t> flat_table = [&] {
            if(shuffle)
                return make_shuffled_block_table(total_pages);
            std::vector<std::int32_t> identity(static_cast<std::size_t>(total_pages));
            std::iota(identity.begin(), identity.end(), 0);
            return identity;
        }();

        std::vector<std::int32_t> block_table(static_cast<std::size_t>(batch) * max_pages_per_batch,
                                              0);
        for(int b = 0; b < batch; ++b)
            for(int lp = 0; lp < pages_per_batch[b]; ++lp)
                block_table[static_cast<std::size_t>(b) * max_pages_per_batch + lp] =
                    flat_table[static_cast<std::size_t>(page_row_start[b]) + lp];

        const std::size_t page_elems_k = static_cast<std::size_t>(page_block_size) * kHeadDim;
        const std::size_t page_elems_v = static_cast<std::size_t>(page_block_size) * hdim_v;
        std::vector<Data> k_phys(static_cast<std::size_t>(total_pages) * page_elems_k,
                                 ck_tile::type_convert<Data>(0.0f));
        std::vector<Data> v_phys(static_cast<std::size_t>(total_pages) * page_elems_v,
                                 ck_tile::type_convert<Data>(0.0f));

        for(int b = 0; b < batch; ++b)
            for(int lp = 0; lp < pages_per_batch[b]; ++lp)
            {
                const std::int32_t physical =
                    block_table[static_cast<std::size_t>(b) * max_pages_per_batch + lp];
                for(int row = 0; row < page_block_size; ++row)
                {
                    const int s = lp * page_block_size + row;
                    const std::size_t logical_k =
                        (static_cast<std::size_t>(seqstart_k[b]) + s) * kHeadDim;
                    const std::size_t logical_v =
                        (static_cast<std::size_t>(seqstart_k[b]) + s) * hdim_v;
                    std::copy(k.begin() + logical_k,
                              k.begin() + logical_k + kHeadDim,
                              k_phys.begin() + static_cast<std::size_t>(physical) * page_elems_k +
                                  static_cast<std::size_t>(row) * kHeadDim);
                    std::copy(v.begin() + logical_v,
                              v.begin() + logical_v + hdim_v,
                              v_phys.begin() + static_cast<std::size_t>(physical) * page_elems_v +
                                  static_cast<std::size_t>(row) * hdim_v);
                }
            }

        Buffer<Data> kb(k_phys.size()), vb(v_phys.size());
        kb.Upload(k_phys);
        vb.Upload(v_phys);
        Buffer<std::int32_t> table_buf(block_table.size());
        table_buf.Upload(block_table);

        fmha_fwd_pagedkv_args args{};
        args.q_ptr    = qb.ptr;
        args.k_ptr    = kb.ptr;
        args.v_ptr    = vb.ptr;
        args.bias_ptr = nullptr;
        args.lse_ptr  = nullptr;
        args.o_ptr    = ob.ptr;

        args.block_table_ptr          = table_buf.ptr;
        args.batch_stride_block_table = max_pages_per_batch;
        args.page_block_size          = page_block_size;
        args.is_gappy                 = false;
        args.cache_batch_idx          = nullptr;

        args.seqstart_q_ptr = seqstart_q_buf.ptr;
        args.seqstart_k_ptr = seqstart_k_buf.ptr;
        args.seqlen_k_ptr   = nullptr;
        args.sink_ptr       = nullptr;

        args.seqlen_q     = max_seqlen_q;
        args.seqlen_k     = total_k;
        args.batch        = batch;
        args.max_seqlen_q = max_seqlen_q;
        args.hdim_q       = kHeadDim;
        args.hdim_v       = hdim_v;
        args.nhead_q      = nhead;
        args.nhead_k      = nhead;

        args.scale_s         = scale_s;
        args.scale_p         = 1.0f;
        args.scale_o         = 1.0f;
        args.logits_soft_cap = 0.0f;

        args.stride_q    = kHeadDim;
        args.stride_k    = kHeadDim;
        args.stride_v    = hdim_v;
        args.stride_bias = 0;
        args.stride_o    = hdim_v;

        args.nhead_stride_q    = static_cast<ck_tile::index_t>(total_q) * kHeadDim;
        args.nhead_stride_k    = static_cast<ck_tile::index_t>(page_block_size) * kHeadDim;
        args.nhead_stride_v    = static_cast<ck_tile::index_t>(page_block_size) * hdim_v;
        args.nhead_stride_bias = 0;
        args.nhead_stride_lse  = 0;
        args.nhead_stride_o    = static_cast<ck_tile::index_t>(total_q) * hdim_v;

        args.batch_stride_q    = 0; // unused in group mode (addressed via seqstart_q_ptr)
        args.batch_stride_k    = static_cast<ck_tile::index_t>(nhead) * args.nhead_stride_k;
        args.batch_stride_v    = static_cast<ck_tile::index_t>(nhead) * args.nhead_stride_v;
        args.batch_stride_bias = 0;
        args.batch_stride_lse  = 0;
        args.batch_stride_o    = 0;

        args.window_size_left  = WindowFor(mask_kind, window).first;
        args.window_size_right = WindowFor(mask_kind, window).second;
        args.sink_size         = 0;
        args.mask_type         = mask_type;
        args.min_seqlen_q      = 0;

        using Kernel        = PagedKernelFor<true, Mask, false, false>;
        auto [kargs, grids] = fmha_fwd_pagedkv_create_kargs_and_grids<Kernel>(args);
        LaunchDirect<Kernel>(kargs, stream, grids);
    }
    else
    {
        Buffer<Data> kb(k.size()), vb(v.size());
        kb.Upload(k);
        vb.Upload(v);

        fmha_fwd_args args{};
        args.q_ptr    = qb.ptr;
        args.k_ptr    = kb.ptr;
        args.v_ptr    = vb.ptr;
        args.bias_ptr = nullptr;
        args.lse_ptr  = nullptr;
        args.o_ptr    = ob.ptr;
        args.sink_ptr = nullptr;

        args.seqstart_q_ptr = seqstart_q_buf.ptr;
        args.seqstart_k_ptr = seqstart_k_buf.ptr;

        args.seqlen_q     = max_seqlen_q;
        args.seqlen_k     = total_k;
        args.batch        = batch;
        args.max_seqlen_q = max_seqlen_q;
        args.hdim_q       = kHeadDim;
        args.hdim_v       = hdim_v;
        args.nhead_q      = nhead;
        args.nhead_k      = nhead;

        args.scale_s         = scale_s;
        args.logits_soft_cap = 0.0f;

        args.stride_q = kHeadDim;
        args.stride_k = kHeadDim;
        args.stride_v = hdim_v;
        args.stride_o = hdim_v;

        args.nhead_stride_q   = static_cast<ck_tile::index_t>(total_q) * kHeadDim;
        args.nhead_stride_k   = static_cast<ck_tile::index_t>(total_k) * kHeadDim;
        args.nhead_stride_v   = static_cast<ck_tile::index_t>(total_k) * hdim_v;
        args.nhead_stride_lse = 0;
        args.nhead_stride_o   = static_cast<ck_tile::index_t>(total_q) * hdim_v;

        args.batch_stride_q   = 0;
        args.batch_stride_k   = 0;
        args.batch_stride_v   = 0;
        args.batch_stride_lse = 0;
        args.batch_stride_o   = 0;

        args.window_size_left  = WindowFor(mask_kind, window).first;
        args.window_size_right = WindowFor(mask_kind, window).second;
        args.sink_size         = 0;
        args.mask_type         = mask_type;

        using Kernel        = DenseKernelFor<true, Mask, false, false>;
        auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<Kernel>(args);
        LaunchDirect<Kernel>(kargs, stream, grids);
    }

    std::vector<Data> o_result(o.size());
    ob.Download(o_result);

    FwdResult result;
    result.out = Decode(o_result);
    return result;
}

FwdResult run_dense_tdm_sched(int batch,
                              int nhead,
                              int seqlen_q,
                              int seqlen_k,
                              int hdim,
                              MaskKind mask = MaskKind::None,
                              int window    = 0,
                              int sink      = 0)
{
    // LaunchBatch only wires sink_size into kargs under if constexpr(Mask), so
    // sink > 0 with mask == MaskKind::None would be silently dropped: the call
    // would compile, run, and test nothing. Fail loudly instead, so the next
    // caller gets an immediate error rather than a green but vacuous test.
    if(sink > 0 && mask == MaskKind::None)
        throw std::invalid_argument(
            "run_dense_tdm_sched: sink > 0 requires a mask (sink_size is only "
            "assigned under if constexpr(Mask) in LaunchBatch)");
    if(mask == MaskKind::None)
        return LaunchBatch<false, false, false, false>(
            batch, nhead, seqlen_q, seqlen_k, hdim, 0, false, mask, window, sink);
    if(sink > 0)
        return LaunchBatch<false, true, true, false>(
            batch, nhead, seqlen_q, seqlen_k, hdim, 0, false, mask, window, sink);
    return LaunchBatch<false, true, false, false>(
        batch, nhead, seqlen_q, seqlen_k, hdim, 0, false, mask, window, sink);
}

FwdResult run_paged_tdm_sched(int batch,
                              int nhead,
                              int seqlen_q,
                              int seqlen_k,
                              int hdim,
                              int page_block_size,
                              bool shuffle,
                              MaskKind mask = MaskKind::None,
                              int window    = 0,
                              int sink      = 0)
{
    // See run_dense_tdm_sched above: same vacuity trap, same guard.
    if(sink > 0 && mask == MaskKind::None)
        throw std::invalid_argument(
            "run_paged_tdm_sched: sink > 0 requires a mask (sink_size is only "
            "assigned under if constexpr(Mask) in LaunchBatch)");
    if(mask == MaskKind::None)
        return LaunchBatch<true, false, false, false>(
            batch, nhead, seqlen_q, seqlen_k, hdim, page_block_size, shuffle, mask, window, sink);
    if(sink > 0)
        return LaunchBatch<true, true, true, false>(
            batch, nhead, seqlen_q, seqlen_k, hdim, page_block_size, shuffle, mask, window, sink);
    return LaunchBatch<true, true, false, false>(
        batch, nhead, seqlen_q, seqlen_k, hdim, page_block_size, shuffle, mask, window, sink);
}

FwdResult run_dense_tdm_sched_group(const std::vector<int>& seqlens_q,
                                    const std::vector<int>& seqlens_k,
                                    int hdim)
{
    return LaunchGroup<false>(seqlens_q, seqlens_k, hdim, /*page_block_size=*/0, /*shuffle=*/false);
}

FwdResult run_paged_tdm_sched_group(const std::vector<int>& seqlens_q,
                                    const std::vector<int>& seqlens_k,
                                    int hdim,
                                    int page_block_size,
                                    bool shuffle)
{
    return LaunchGroup<true>(seqlens_q, seqlens_k, hdim, page_block_size, shuffle);
}

// Gappy group-mode paged launcher: the only path that drives kv_l2p_offset
// nonzero (see the kv_l2p_offset computation in fmha_fwd_pagedkv_kernel.hpp,
// gated on kargs.is_gappy). Under is_gappy, seqstart_k_ptr[b] is repurposed
// (per the fmha_fwd_pagedkv_args contract in fmha_fwd.hpp) to hold each
// batch's own physical start offset into ITS OWN page pool -- not a running
// cross-batch prefix sum like the non-gappy convention -- and seqlen_k_ptr[b]
// carries the real per-batch token count. Each batch's real K/V content is
// placed starting at physical row l2p_offsets[b] within its own page set; the
// rows before that (the "gap") carry a NaN sentinel and are provably never
// read. The TDM affine K/V view is sized by the LOGICAL seqlen_k alone (the
// l2p shift never widens the descriptor); it is the tracker's reverse offset,
// folded into the base pointer, that maps a GLOBAL (gap-unaware) window
// origin in [0, seqlen_k) onto physical rows
// [l2p_offsets[b], l2p_offsets[b]+seqlen_k) -- see tdm_page_tracker.hpp. A
// descriptor that instead over-reads into the gap (or the next page) would
// pull the NaN sentinel through P*V instead of silently agreeing with a zero
// fill, which is why the gap rows are NaN and not zero.
//
// l2p_offsets[b] must be a multiple of kN0 (128, this file's fixed TileRows):
// the tracker's tile-advance countdown assumes the first token inside a page
// is tile aligned, same requirement as page_block_size being a TileRows
// multiple.
template <bool Mask = false>
FwdResult run_paged_tdm_sched_group_gappy(const std::vector<int>& seqlens_q,
                                          const std::vector<int>& seqlens_k,
                                          const std::vector<int>& l2p_offsets,
                                          int hdim,
                                          int page_block_size,
                                          bool shuffle,
                                          MaskKind mask_kind = MaskKind::None,
                                          int window         = 0)
{
    if(hdim != kHeadDim)
        throw std::invalid_argument("harness only supports hdim=128 (fixed tile geometry)");
    if(seqlens_q.size() != seqlens_k.size() || seqlens_q.size() != l2p_offsets.size())
        throw std::invalid_argument("seqlens_q/seqlens_k/l2p_offsets size mismatch");

    const int batch      = static_cast<int>(seqlens_q.size());
    constexpr int nhead  = 1;
    constexpr int hdim_v = kHeadDimV;

    for(int b = 0; b < batch; ++b)
    {
        if(l2p_offsets[b] < 0)
            throw std::invalid_argument("l2p_offsets[b] must be non-negative");
    }

    std::vector<int> seqstart_q(static_cast<std::size_t>(batch) + 1, 0);
    for(int b = 0; b < batch; ++b)
        seqstart_q[b + 1] = seqstart_q[b] + seqlens_q[b];
    const int total_q      = seqstart_q[batch];
    const int max_seqlen_q = *std::max_element(seqlens_q.begin(), seqlens_q.end());

    // Q (and O) stay packed in the ordinary running-prefix-sum convention --
    // gappy only changes how K/V address into the page pool.
    std::vector<Data> q(static_cast<std::size_t>(total_q) * kHeadDim);
    for(int b = 0; b < batch; ++b)
    {
        const auto q_local = make_input(1, 1, seqlens_q[b], kHeadDim, b * 3 + 0);
        std::copy(q_local.begin(),
                  q_local.end(),
                  q.begin() + static_cast<std::size_t>(seqstart_q[b]) * kHeadDim);
    }
    std::vector<Data> o(static_cast<std::size_t>(total_q) * hdim_v,
                        ck_tile::type_convert<Data>(0.0f));

    const float scale_s = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    Buffer<Data> qb(q.size()), ob(o.size());
    qb.Upload(q);
    ob.Upload(o);
    std::vector<std::int32_t> seqstart_q32(seqstart_q.begin(), seqstart_q.end());
    Buffer<std::int32_t> seqstart_q_buf(seqstart_q32.size());
    seqstart_q_buf.Upload(seqstart_q32);

    std::vector<std::int32_t> l2p_offset32(l2p_offsets.begin(), l2p_offsets.end());
    std::vector<std::int32_t> seqlen_k32(seqlens_k.begin(), seqlens_k.end());
    Buffer<std::int32_t> seqstart_k_buf(l2p_offset32.size()), seqlen_k_buf(seqlen_k32.size());
    seqstart_k_buf.Upload(l2p_offset32);
    seqlen_k_buf.Upload(seqlen_k32);

    std::vector<int> pages_per_batch(static_cast<std::size_t>(batch));
    for(int b = 0; b < batch; ++b)
        pages_per_batch[b] =
            (l2p_offsets[b] + seqlens_k[b] + page_block_size - 1) / page_block_size;
    const int max_pages_per_batch =
        *std::max_element(pages_per_batch.begin(), pages_per_batch.end());
    std::vector<int> page_row_start(static_cast<std::size_t>(batch) + 1, 0);
    for(int b = 0; b < batch; ++b)
        page_row_start[b + 1] = page_row_start[b] + pages_per_batch[b];
    const int total_pages = page_row_start[batch];

    const std::vector<std::int32_t> flat_table = [&] {
        if(shuffle)
            return make_shuffled_block_table(total_pages);
        std::vector<std::int32_t> identity(static_cast<std::size_t>(total_pages));
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }();

    std::vector<std::int32_t> block_table(static_cast<std::size_t>(batch) * max_pages_per_batch, 0);
    for(int b = 0; b < batch; ++b)
        for(int lp = 0; lp < pages_per_batch[b]; ++lp)
            block_table[static_cast<std::size_t>(b) * max_pages_per_batch + lp] =
                flat_table[static_cast<std::size_t>(page_row_start[b]) + lp];

    const std::size_t page_elems_k = static_cast<std::size_t>(page_block_size) * kHeadDim;
    const std::size_t page_elems_v = static_cast<std::size_t>(page_block_size) * hdim_v;
    // Rows past the end of a batch's keys hold NaN: they must never be read (the TDM view
    // ends at l2p_offset + seqlen_k), and 0 * NaN == NaN would expose an over-read even
    // where the score is masked. Gap rows before l2p_offset hold finite data unrelated to
    // the keys: the first K/V tile may read them and must mask them out, so finite values
    // are required (as for the generic qr_pagedkv), and leaking them changes the output.
    const Data sentinel = ck_tile::type_convert<Data>(std::numeric_limits<float>::quiet_NaN());
    std::vector<Data> k_phys(static_cast<std::size_t>(total_pages) * page_elems_k, sentinel);
    std::vector<Data> v_phys(static_cast<std::size_t>(total_pages) * page_elems_v, sentinel);

    for(int b = 0; b < batch; ++b)
    {
        // Same deterministic generator and phase indices as LaunchGroup/the
        // dense reference, so logical K/V content is bit-identical -- only
        // its placement in the (gappy) page pool differs.
        const auto k_local = make_input(1, 1, seqlens_k[b], kHeadDim, b * 3 + 1);
        const auto v_local = make_input(1, 1, seqlens_k[b], hdim_v, b * 3 + 2);
        const int gap_rows = std::max(l2p_offsets[b], 1);
        const auto k_gap   = make_input(1, 1, gap_rows, kHeadDim, 1000 + b * 3 + 1);
        // A large V makes any leaked gap column visible far above the comparison tolerance.
        const std::vector<Data> v_gap(static_cast<std::size_t>(gap_rows) * hdim_v,
                                      ck_tile::type_convert<Data>(64.0f));
        for(int lp = 0; lp < pages_per_batch[b]; ++lp)
        {
            const std::int32_t physical =
                block_table[static_cast<std::size_t>(b) * max_pages_per_batch + lp];
            for(int row = 0; row < page_block_size; ++row)
            {
                const int phys_row = lp * page_block_size + row;
                if(phys_row >= l2p_offsets[b] + seqlens_k[b])
                    continue; // trailing pad: must never be read, left as NaN
                const bool gap    = phys_row < l2p_offsets[b];
                const int s       = gap ? phys_row : phys_row - l2p_offsets[b];
                const auto& k_src = gap ? k_gap : k_local;
                const auto& v_src = gap ? v_gap : v_local;
                std::copy(k_src.begin() + static_cast<std::size_t>(s) * kHeadDim,
                          k_src.begin() + static_cast<std::size_t>(s + 1) * kHeadDim,
                          k_phys.begin() + static_cast<std::size_t>(physical) * page_elems_k +
                              static_cast<std::size_t>(row) * kHeadDim);
                std::copy(v_src.begin() + static_cast<std::size_t>(s) * hdim_v,
                          v_src.begin() + static_cast<std::size_t>(s + 1) * hdim_v,
                          v_phys.begin() + static_cast<std::size_t>(physical) * page_elems_v +
                              static_cast<std::size_t>(row) * hdim_v);
            }
        }
    }

    Buffer<Data> kb(k_phys.size()), vb(v_phys.size());
    kb.Upload(k_phys);
    vb.Upload(v_phys);
    Buffer<std::int32_t> table_buf(block_table.size());
    table_buf.Upload(block_table);

    const ck_tile::index_t mask_type =
        static_cast<ck_tile::index_t>(ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
    const ck_tile::stream_config stream{nullptr, false, 0, 0, 1};

    fmha_fwd_pagedkv_args args{};
    args.q_ptr    = qb.ptr;
    args.k_ptr    = kb.ptr;
    args.v_ptr    = vb.ptr;
    args.bias_ptr = nullptr;
    args.lse_ptr  = nullptr;
    args.o_ptr    = ob.ptr;

    args.block_table_ptr          = table_buf.ptr;
    args.batch_stride_block_table = max_pages_per_batch;
    args.page_block_size          = page_block_size;
    args.is_gappy                 = true; // seqstart_k_ptr[b] is now a per-batch l2p_offset
    args.cache_batch_idx          = nullptr;

    args.seqstart_q_ptr = seqstart_q_buf.ptr;
    args.seqstart_k_ptr = seqstart_k_buf.ptr; // per-batch l2p_offset, NOT a prefix sum
    args.seqlen_k_ptr   = seqlen_k_buf.ptr;   // real per-batch key length
    args.sink_ptr       = nullptr;

    args.seqlen_q     = max_seqlen_q;
    args.seqlen_k     = 0; // unused: is_gappy routes seqlen_k through seqlen_k_ptr
    args.batch        = batch;
    args.max_seqlen_q = max_seqlen_q;
    args.hdim_q       = kHeadDim;
    args.hdim_v       = hdim_v;
    args.nhead_q      = nhead;
    args.nhead_k      = nhead;

    args.scale_s         = scale_s;
    args.scale_p         = 1.0f;
    args.scale_o         = 1.0f;
    args.logits_soft_cap = 0.0f;

    args.stride_q    = kHeadDim;
    args.stride_k    = kHeadDim;
    args.stride_v    = hdim_v;
    args.stride_bias = 0;
    args.stride_o    = hdim_v;

    args.nhead_stride_q    = static_cast<ck_tile::index_t>(total_q) * kHeadDim;
    args.nhead_stride_k    = static_cast<ck_tile::index_t>(page_block_size) * kHeadDim;
    args.nhead_stride_v    = static_cast<ck_tile::index_t>(page_block_size) * hdim_v;
    args.nhead_stride_bias = 0;
    args.nhead_stride_lse  = 0;
    args.nhead_stride_o    = static_cast<ck_tile::index_t>(total_q) * hdim_v;

    args.batch_stride_q    = 0; // unused in group mode (addressed via seqstart_q_ptr)
    args.batch_stride_k    = static_cast<ck_tile::index_t>(nhead) * args.nhead_stride_k;
    args.batch_stride_v    = static_cast<ck_tile::index_t>(nhead) * args.nhead_stride_v;
    args.batch_stride_bias = 0;
    args.batch_stride_lse  = 0;
    args.batch_stride_o    = 0;

    args.window_size_left  = WindowFor(mask_kind, window).first;
    args.window_size_right = WindowFor(mask_kind, window).second;
    args.sink_size         = 0;
    args.mask_type         = mask_type;
    args.min_seqlen_q      = 0;

    using Kernel        = PagedGappyKernelFor<Mask>;
    auto [kargs, grids] = fmha_fwd_pagedkv_create_kargs_and_grids<Kernel>(args);
    LaunchDirect<Kernel>(kargs, stream, grids);

    std::vector<Data> o_result(o.size());
    ob.Download(o_result);

    FwdResult result;
    result.out = Decode(o_result);
    return result;
}

FwdResult run_dense_tdm_sched_with_lse(int batch, int nhead, int seqlen_q, int seqlen_k, int hdim)
{
    return LaunchBatch<false, false, false, true>(
        batch, nhead, seqlen_q, seqlen_k, hdim, 0, false, MaskKind::None, 0, 0);
}

FwdResult run_paged_tdm_sched_with_lse(
    int batch, int nhead, int seqlen_q, int seqlen_k, int hdim, int page_block_size, bool shuffle)
{
    return LaunchBatch<true, false, false, true>(
        batch, nhead, seqlen_q, seqlen_k, hdim, page_block_size, shuffle, MaskKind::None, 0, 0);
}

TEST(QrTdmSchedPagedHarness, ShuffledBlockTableIsNotIdentity)
{
    // A representative sample of page counts, not an exhaustive list of every
    // value the run_paged_* tests in this file request (that set keeps changing
    // as tests are added/resized, e.g. SlidingWindowMaskMatchesDense alone can
    // request 32). The guarantee this is actually checking --
    // make_shuffled_block_table producing a non-identity permutation -- holds
    // for any n >= 2, so a few representative sizes are sufficient; the small
    // ones (2, 3) are kept because a fixed-seed single shuffle is most likely to
    // land on the identity by chance there (confirmed at n == 2 before
    // make_shuffled_block_table was fixed to guard against it).
    for(const int num_pages : {2, 3, 4, 6, 8, 12, 16})
    {
        SCOPED_TRACE(num_pages);
        const auto table = make_shuffled_block_table(num_pages);
        bool identity    = true;
        for(std::size_t i = 0; i < table.size(); ++i)
            identity &= (table[i] == static_cast<std::int32_t>(i));
        EXPECT_FALSE(identity) << "shuffled block table must not be the identity permutation";
    }
}

TEST(QrTdmSchedPaged, MatchesDenseForIdenticalLogicalContent)
{
    constexpr int batch    = 2;
    constexpr int nhead    = 4;
    constexpr int seqlen_q = 256;
    constexpr int seqlen_k = 1024;
    constexpr int hdim     = 128;

    for(const int page_block_size : {128, 256, 512})
    {
        SCOPED_TRACE(page_block_size);
        const auto dense = run_dense_tdm_sched(batch, nhead, seqlen_q, seqlen_k, hdim);
        const auto paged = run_paged_tdm_sched(
            batch, nhead, seqlen_q, seqlen_k, hdim, page_block_size, /*shuffle=*/true);
        AssertNotAllZero(dense.out, "dense.out");
        ASSERT_EQ(dense.out.size(), paged.out.size());
        for(std::size_t i = 0; i < dense.out.size(); ++i)
            ASSERT_EQ(dense.out[i], paged.out[i]) << "mismatch at element " << i;
    }
}

// Masking reads the window origin as a global seqlen_k coordinate. The paged
// path must not disturb that, so these should pass with no pipeline changes.
TEST(QrTdmSchedPaged, CausalMaskMatchesDense)
{
    // seqlen_q == seqlen_k so the last query tile's causal range reaches all the
    // way to x_total under TOP_LEFT alignment (GetTileRangeAlongX: x_end = min(i_y
    // + YTile - 1 + x, x_total), and at i_y = seqlen_q - YTile this hits x_total
    // exactly when seqlen_q == seqlen_k). That forces every page to actually be
    // read, not just the first half -- otherwise seqlen_k's upper pages would be
    // populated but never touched by the kernel, making the paged-vs-dense
    // comparison weaker than it looks.
    constexpr int batch = 2, nhead = 4, seqlen_q = 1024, seqlen_k = 1024, hdim = 128;

    for(int page_block_size : {128, 256})
    {
        auto dense = run_dense_tdm_sched(batch, nhead, seqlen_q, seqlen_k, hdim, MaskKind::Causal);
        auto paged = run_paged_tdm_sched(batch,
                                         nhead,
                                         seqlen_q,
                                         seqlen_k,
                                         hdim,
                                         page_block_size,
                                         /*shuffle=*/true,
                                         MaskKind::Causal);

        AssertNotAllZero(dense.out, "dense.out");
        ASSERT_EQ(dense.out.size(), paged.out.size());
        for(size_t i = 0; i < dense.out.size(); ++i)
            ASSERT_EQ(dense.out[i], paged.out[i])
                << "page_block_size=" << page_block_size << " elem=" << i;
    }
}

TEST(QrTdmSchedPaged, SlidingWindowMaskMatchesDense)
{
    // seqlen_q == seqlen_k for the same reason as CausalMaskMatchesDense: with
    // window_right == 0 (both window values below), x_start/x_end are contiguous
    // across consecutive query tiles regardless of window size, so the union of
    // tiles read reaches x_total only once the last query tile's range does --
    // which requires seqlen_q == seqlen_k. Otherwise every page past seqlen_q
    // would be populated but never read by either kernel.
    constexpr int batch = 2, nhead = 4, seqlen_q = 2048, seqlen_k = 2048, hdim = 128;

    // A window shorter than a page exercises the case where kv_load_start lands
    // mid-page and the first tracker seek is not page-aligned.
    for(int window : {128, 384})
    {
        for(int page_block_size : {128, 256})
        {
            auto dense = run_dense_tdm_sched(
                batch, nhead, seqlen_q, seqlen_k, hdim, MaskKind::SlidingWindow, window);
            auto paged = run_paged_tdm_sched(batch,
                                             nhead,
                                             seqlen_q,
                                             seqlen_k,
                                             hdim,
                                             page_block_size,
                                             /*shuffle=*/true,
                                             MaskKind::SlidingWindow,
                                             window);

            AssertNotAllZero(dense.out, "dense.out");
            ASSERT_EQ(dense.out.size(), paged.out.size());
            for(size_t i = 0; i < dense.out.size(); ++i)
                ASSERT_EQ(dense.out[i], paged.out[i])
                    << "window=" << window << " page_block_size=" << page_block_size
                    << " elem=" << i;
        }
    }
}

// Sink is the only path that jumps the window an arbitrary distance, crossing
// several pages at once. K and V jump on different iterations (K two tiles
// ahead, V one), so both seek sites must be correct independently.
TEST(QrTdmSchedPaged, AttentionSinkMatchesDense)
{
    // seqlen_q == seqlen_k on purpose. Under MASK_FROM_TOP_LEFT a query tile
    // reads K only up to x_end = min(i_y + kM0, seqlen_k), so with a smaller
    // seqlen_q the mask caps the K range well below seqlen_k and most pages are
    // never read -- which would defeat the point of testing a paged walk. Task 6
    // hit exactly that and had to be corrected.
    constexpr int batch = 2, nhead = 4, seqlen_q = 2048, seqlen_k = 2048, hdim = 128;

    // sink=128 gives num_sink_loop==1 (prologue jump site);
    // sink=256 gives num_sink_loop==2, the only case where K's main-loop jump
    // fires on the very first main-loop iteration
    // (i_total_loops == num_sink_loop - 2 == 0);
    // sink=384 gives num_sink_loop==3 (main-loop jump site, later iteration).
    for(int sink : {128, 256, 384})
    {
        for(int page_block_size : {128, 256})
        {
            auto dense = run_dense_tdm_sched(batch,
                                             nhead,
                                             seqlen_q,
                                             seqlen_k,
                                             hdim,
                                             MaskKind::SlidingWindow,
                                             /*window=*/512,
                                             sink);
            auto paged = run_paged_tdm_sched(batch,
                                             nhead,
                                             seqlen_q,
                                             seqlen_k,
                                             hdim,
                                             page_block_size,
                                             /*shuffle=*/true,
                                             MaskKind::SlidingWindow,
                                             /*window=*/512,
                                             sink);

            AssertNotAllZero(dense.out, "dense.out");
            ASSERT_EQ(dense.out.size(), paged.out.size());
            for(size_t i = 0; i < dense.out.size(); ++i)
                ASSERT_EQ(dense.out[i], paged.out[i])
                    << "sink=" << sink << " page_block_size=" << page_block_size << " elem=" << i;
        }
    }
}

// Additive rigor beyond the brief: exercise the group-mode and LSE-output
// harness paths too (the brief's Step 5/6 tests only cover batch mode without
// LSE), so run_*_group/run_*_with_lse are not left as untested dead code.
TEST(QrTdmSchedPaged, GroupModeMatchesDenseForIdenticalLogicalContent)
{
    const std::vector<int> seqlens_q{64, 192, 128};
    // Every entry must be divisible by every page_block_size exercised below
    // (128 and 256); the harness requires seqlen_k % page_block_size == 0.
    const std::vector<int> seqlens_k{256, 512, 768};
    constexpr int hdim = 128;

    for(const int page_block_size : {128, 256})
    {
        SCOPED_TRACE(page_block_size);
        const auto dense = run_dense_tdm_sched_group(seqlens_q, seqlens_k, hdim);
        const auto paged = run_paged_tdm_sched_group(
            seqlens_q, seqlens_k, hdim, page_block_size, /*shuffle=*/true);
        AssertNotAllZero(dense.out, "dense.out");
        ASSERT_EQ(dense.out.size(), paged.out.size());
        for(std::size_t i = 0; i < dense.out.size(); ++i)
            ASSERT_EQ(dense.out[i], paged.out[i]) << "mismatch at element " << i;
    }
}

// Group mode: variable seqlen per batch via seqstart_k_ptr. This mirrors the
// task brief's Step 1 test, with one deviation: the brief's seqlens_k =
// {128, 1024, 256, 768} includes 128, which is not divisible by
// page_block_size=256, and this harness (deliberately, see LaunchGroup)
// throws on that precondition rather than silently mis-sizing pages. Bumped
// the first entry to 256 so every length is divisible by both page sizes
// exercised below; the rest is unchanged from the brief.
TEST(QrTdmSchedPaged, GroupModeMatchesDense)
{
    const std::vector<int> seqlens_q = {64, 300, 128, 512};
    const std::vector<int> seqlens_k = {256, 1024, 256, 768};

    for(int page_block_size : {128, 256})
    {
        SCOPED_TRACE(page_block_size);
        auto dense = run_dense_tdm_sched_group(seqlens_q, seqlens_k, /*hdim=*/128);
        auto paged = run_paged_tdm_sched_group(
            seqlens_q, seqlens_k, /*hdim=*/128, page_block_size, /*shuffle=*/true);

        AssertNotAllZero(dense.out, "dense.out");
        ASSERT_EQ(dense.out.size(), paged.out.size());
        for(size_t i = 0; i < dense.out.size(); ++i)
            ASSERT_EQ(dense.out[i], paged.out[i])
                << "page_block_size=" << page_block_size << " elem=" << i;
    }
}

// Group mode, gappy variant: this is the only path in the whole suite that
// drives kv_l2p_offset to a nonzero value (GroupModeMatchesDense above and
// GroupModeMatchesDenseForIdenticalLogicalContent both hard-code
// is_gappy=false, so kv_l2p_offset stays 0 for them and Task 5's new
// seek()/refresh() shift is never actually exercised by either). Here
// seqstart_k_ptr carries a nonzero per-batch l2p_offset -- the sequence
// starts partway (0, 128, or 384 tokens) into its own page pool -- and
// seqlen_k_ptr carries the real per-batch length. See
// run_paged_tdm_sched_group_gappy for how the physical layout is built and
// why the "gap" rows before the offset are provably never read.
//
// If kv_l2p_offset were silently dropped back to 0 (the pre-Task-5 bug this
// test exists to catch), the tracker would read starting at physical row 0
// of each batch's page pool instead of l2p_offsets[b], landing on the
// zero-filled gap rows for the batches with a nonzero offset and producing
// output that does not match dense -- so this test fails loudly rather than
// vacuously passing on an untested zero offset.
TEST(QrTdmSchedPaged, GroupModeGappyMatchesDenseWithNonzeroL2pOffset)
{
    const std::vector<int> seqlens_q   = {64, 128, 96};
    const std::vector<int> seqlens_k   = {200, 384, 150};
    const std::vector<int> l2p_offsets = {0, 128, 384}; // multiples of kN0=128; not all zero
    constexpr int hdim                 = 128;

    for(int page_block_size : {128, 256})
    {
        SCOPED_TRACE(page_block_size);
        auto dense = run_dense_tdm_sched_group(seqlens_q, seqlens_k, hdim);
        auto paged = run_paged_tdm_sched_group_gappy(
            seqlens_q, seqlens_k, l2p_offsets, hdim, page_block_size, /*shuffle=*/true);

        AssertNotAllZero(dense.out, "dense.out");
        ASSERT_EQ(dense.out.size(), paged.out.size());
        for(size_t i = 0; i < dense.out.size(); ++i)
            ASSERT_EQ(dense.out[i], paged.out[i])
                << "page_block_size=" << page_block_size << " elem=" << i;
    }
}

// Gappy offsets are arbitrary in practice (xformers only requires 0 <= start < row length), so
// the first key of a batch usually sits mid-tile. Covers offsets that are not multiples of
// kN0 and all mask kinds. With a misaligned offset the KV tiles cover different key ranges
// than dense, so the softmax accumulation order differs and the outputs match to bf16
// rounding, not bitwise.
template <bool Mask>
void CheckGappyUnaligned(MaskKind mask_kind, int window)
{
    // l2p 37: the keys [37, 237) cross a page boundary (page 128) mid-tile.
    // l2p 129: the first key sits past page 0 (page 128), or crosses 256 (page 256).
    // l2p 255: one key row in the first page at page 256. 1 and 300: smallest and a
    // multi-tile offset.
    const std::vector<int> seqlens_q   = {64, 128, 96, 200, 33, 50};
    const std::vector<int> seqlens_k   = {200, 384, 150, 700, 130, 90};
    const std::vector<int> l2p_offsets = {37, 129, 300, 1, 255, 64};
    constexpr int hdim                 = 128;

    for(int page_block_size : {128, 256})
    {
        SCOPED_TRACE(page_block_size);
        auto dense = LaunchGroup<false, Mask>(
            seqlens_q, seqlens_k, hdim, page_block_size, false, mask_kind, window);
        auto paged = run_paged_tdm_sched_group_gappy<Mask>(seqlens_q,
                                                           seqlens_k,
                                                           l2p_offsets,
                                                           hdim,
                                                           page_block_size,
                                                           /*shuffle=*/true,
                                                           mask_kind,
                                                           window);
        AssertNotAllZero(dense.out, "dense.out");
        AssertAllFinite(paged.out, "paged.out");
        ASSERT_EQ(dense.out.size(), paged.out.size());
        constexpr float atol = 2e-3f;
        constexpr float rtol = 1e-2f;
        float max_abs_diff   = 0.0f;
        for(size_t i = 0; i < dense.out.size(); ++i)
        {
            const float diff = std::abs(dense.out[i] - paged.out[i]);
            max_abs_diff     = std::max(max_abs_diff, diff);
            ASSERT_LE(diff, atol + rtol * std::abs(dense.out[i]))
                << "page_block_size=" << page_block_size << " elem=" << i
                << " dense=" << dense.out[i] << " paged=" << paged.out[i];
        }
        std::cout << "gappy unaligned page_block_size=" << page_block_size
                  << " max_abs_diff=" << max_abs_diff << std::endl;
    }
}

TEST(QrTdmSchedPaged, GroupModeGappyUnalignedL2pOffsetMatchesDense)
{
    CheckGappyUnaligned<false>(MaskKind::None, 0);
}

TEST(QrTdmSchedPaged, GroupModeGappyUnalignedL2pOffsetCausalMatchesDense)
{
    CheckGappyUnaligned<true>(MaskKind::Causal, 0);
}

TEST(QrTdmSchedPaged, GroupModeGappyUnalignedL2pOffsetSlidingWindowMatchesDense)
{
    CheckGappyUnaligned<true>(MaskKind::SlidingWindow, 100);
}

TEST(QrTdmSchedPaged, LseMatchesDenseForIdenticalLogicalContent)
{
    constexpr int batch    = 2;
    constexpr int nhead    = 2;
    constexpr int seqlen_q = 128;
    constexpr int seqlen_k = 512;
    constexpr int hdim     = 128;

    for(const int page_block_size : {128, 256})
    {
        SCOPED_TRACE(page_block_size);
        const auto dense = run_dense_tdm_sched_with_lse(batch, nhead, seqlen_q, seqlen_k, hdim);
        const auto paged = run_paged_tdm_sched_with_lse(
            batch, nhead, seqlen_q, seqlen_k, hdim, page_block_size, /*shuffle=*/true);
        AssertNotAllZero(dense.out, "dense.out");
        ASSERT_EQ(dense.out.size(), paged.out.size());
        for(std::size_t i = 0; i < dense.out.size(); ++i)
            ASSERT_EQ(dense.out[i], paged.out[i]) << "mismatch at element " << i;
        AssertAllFinite(dense.lse, "dense.lse");
        AssertNotAllZero(dense.lse, "dense.lse");
        ASSERT_EQ(dense.lse.size(), paged.lse.size());
        for(std::size_t i = 0; i < dense.lse.size(); ++i)
            ASSERT_EQ(dense.lse[i], paged.lse[i]) << "lse mismatch at element " << i;
    }
}

// LseMatchesDenseForIdenticalLogicalContent above only exercises the no-mask
// path (run_dense_tdm_sched_with_lse/run_paged_tdm_sched_with_lse take no mask
// argument). Masking changes which logits enter the softmax normalization --
// masked-out entries are excluded from the LSE sum -- so that test cannot
// prove paging leaves LSE untouched once a mask is active. Call LaunchBatch
// directly (Mask=true, Lse=true) to close that gap instead of duplicating the
// brief's identical-content LSE test under a new name with different shape
// numbers, which would add no new coverage.
TEST(QrTdmSchedPaged, LseMatchesDenseUnderCausalMask)
{
    // seqlen_q == seqlen_k for the same reason as CausalMaskMatchesDense: the
    // last query tile's causal range must reach x_total under TOP_LEFT
    // alignment so every page is actually read, not just the first half.
    constexpr int batch = 2, nhead = 4, seqlen_q = 1024, seqlen_k = 1024, hdim = 128;

    for(int page_block_size : {128, 256})
    {
        SCOPED_TRACE(page_block_size);
        auto dense = LaunchBatch<false, true, false, true>(
            batch, nhead, seqlen_q, seqlen_k, hdim, 0, false, MaskKind::Causal, 0, 0);
        auto paged = LaunchBatch<true, true, false, true>(batch,
                                                          nhead,
                                                          seqlen_q,
                                                          seqlen_k,
                                                          hdim,
                                                          page_block_size,
                                                          /*shuffle=*/true,
                                                          MaskKind::Causal,
                                                          0,
                                                          0);

        AssertNotAllZero(dense.out, "dense.out");
        ASSERT_EQ(dense.out.size(), paged.out.size());
        for(size_t i = 0; i < dense.out.size(); ++i)
            ASSERT_EQ(dense.out[i], paged.out[i]) << "out elem " << i;

        AssertAllFinite(dense.lse, "dense.lse");
        AssertNotAllZero(dense.lse, "dense.lse");
        ASSERT_EQ(dense.lse.size(), paged.lse.size());
        for(size_t i = 0; i < dense.lse.size(); ++i)
            ASSERT_EQ(dense.lse[i], paged.lse[i]) << "lse elem " << i;
    }
}

// The O store must not write outside the logical rows: the last query tile is partial
// (seqlen_q % kM0 != 0) and every head is followed by NaN-filled spare rows. Covers the async
// LDS-to-global O store, which is taken for this aligned layout.
TEST(QrTdmSchedPaged, OutputStoreStaysInsideLogicalRows)
{
    constexpr int batch = 2, nhead = 3, seqlen_k = 512, hdim = 128, page_block_size = 128;

    for(const int seqlen_q : {100, 200, 256})
    {
        SCOPED_TRACE(seqlen_q);
        const auto compact = LaunchBatch<true, true, false, false>(batch,
                                                                   nhead,
                                                                   seqlen_q,
                                                                   seqlen_k,
                                                                   hdim,
                                                                   page_block_size,
                                                                   /*shuffle=*/true,
                                                                   MaskKind::Causal,
                                                                   0,
                                                                   0);
        const auto guarded =
            LaunchBatch<true, true, false, false>(batch,
                                                  nhead,
                                                  seqlen_q,
                                                  seqlen_k,
                                                  hdim,
                                                  page_block_size,
                                                  /*shuffle=*/true,
                                                  MaskKind::Causal,
                                                  0,
                                                  0,
                                                  OLayout{/*row_pad=*/0, /*spare_rows=*/128});
        AssertNotAllZero(compact.out, "compact.out");
        AssertAllFinite(guarded.out, "guarded.out");
        ASSERT_TRUE(guarded.o_padding_intact) << "a store reached the spare O rows";
        ASSERT_EQ(compact.out.size(), guarded.out.size());
        for(std::size_t i = 0; i < compact.out.size(); ++i)
            ASSERT_EQ(compact.out[i], guarded.out[i]) << "elem " << i;
    }
}

// A row stride that is not a multiple of 16 bytes cannot use the async O store; the kernel
// must fall back to the epilogue and still produce the same values without touching the pad.
TEST(QrTdmSchedPaged, UnalignedOutputStrideMatchesAlignedLayout)
{
    constexpr int batch = 2, nhead = 2, seqlen_q = 200, seqlen_k = 512, hdim = 128;
    constexpr int page_block_size = 128;

    const auto aligned = LaunchBatch<true, false, false, false>(batch,
                                                                nhead,
                                                                seqlen_q,
                                                                seqlen_k,
                                                                hdim,
                                                                page_block_size,
                                                                /*shuffle=*/true,
                                                                MaskKind::None,
                                                                0,
                                                                0);
    const auto unaligned =
        LaunchBatch<true, false, false, false>(batch,
                                               nhead,
                                               seqlen_q,
                                               seqlen_k,
                                               hdim,
                                               page_block_size,
                                               /*shuffle=*/true,
                                               MaskKind::None,
                                               0,
                                               0,
                                               OLayout{/*row_pad=*/4, /*spare_rows=*/128});
    AssertNotAllZero(aligned.out, "aligned.out");
    AssertAllFinite(unaligned.out, "unaligned.out");
    ASSERT_TRUE(unaligned.o_padding_intact) << "a store reached the O row pad or spare rows";
    ASSERT_EQ(aligned.out.size(), unaligned.out.size());
    for(std::size_t i = 0; i < aligned.out.size(); ++i)
        ASSERT_EQ(aligned.out[i], unaligned.out[i]) << "elem " << i;
}

// Additive rigor beyond the brief: confirm the dense baseline itself is
// numerically correct against the independent CPU reference oracle, so a
// bitwise dense==paged match above cannot hide both sides being equally
// wrong. Uses -v=1 (CPU reference) semantics per project convention.
TEST(QrTdmSchedPaged, DenseMatchesCpuReference)
{
    constexpr int batch    = 1;
    constexpr int nhead    = 1;
    constexpr int seqlen_q = 128;
    constexpr int seqlen_k = 128;
    constexpr int hdim     = 128;

    const auto q = make_input(batch, nhead, seqlen_q, hdim, 0);
    const auto k = make_input(batch, nhead, seqlen_k, hdim, 1);
    const auto v = make_input(batch, nhead, seqlen_k, hdim, 2);

    const auto dense = run_dense_tdm_sched(batch, nhead, seqlen_q, seqlen_k, hdim);

    const auto q_float = Decode(q);
    const auto k_float = Decode(k);
    const auto v_float = Decode(v);

    namespace reference = ck_tile::test::reference;
    // FloatView field order is {data, size, offset, head_stride, row_stride,
    // dim_stride}. Our buffers are row-major (batch, head, row, dim) with a
    // single batch/head here, so head_stride is unused (0) and row_stride is
    // the per-row element count (dim), with dim_stride = 1 for contiguous dims.
    auto view = [&](const std::vector<float>& values, int dim) {
        return reference::FloatView{
            values.data(), values.size(), 0, 0, static_cast<std::size_t>(dim), 1};
    };

    reference::Sequence sequence;
    sequence.q               = view(q_float, hdim);
    sequence.k               = view(k_float, hdim);
    sequence.v               = view(v_float, hdim);
    sequence.query_length    = seqlen_q;
    sequence.key_length      = seqlen_k;
    sequence.query_heads     = nhead;
    sequence.kv_heads        = nhead;
    sequence.query_dimension = hdim;
    sequence.value_dimension = hdim;
    sequence.score_scale     = 1.0f / std::sqrt(static_cast<float>(hdim));

    reference::Output output;
    output.o = reference::FloatView{
        dense.out.data(), dense.out.size(), 0, 0, static_cast<std::size_t>(hdim), 1};

    std::vector<reference::Sequence> sequences{sequence};
    std::vector<reference::Output> outputs{output};

    const auto expected = reference::Compute(sequences);
    const reference::AccuracyContract contract{7, 128};
    const auto metrics = reference::Compare(sequences, expected, outputs, {}, contract);
    EXPECT_TRUE(metrics.passed) << "max_abs_error=" << metrics.max_abs_error
                                << " min_row_cosine=" << metrics.min_row_cosine;
}

} // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
