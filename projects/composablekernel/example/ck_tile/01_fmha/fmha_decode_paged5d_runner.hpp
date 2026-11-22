// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host side of FmhaDecodePaged5dKernel for the example and the tests: launch dispatch over the
// compile-time split count and cache page size, random inputs in the 5D layout, a CPU reference
// and timing.

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_decode_paged5d_kernel.hpp"

namespace fmha_decode_paged5d {

using ck_tile::index_t;
constexpr index_t kHdim = 128;
constexpr index_t kPage = 16;

template <bool kWaveSplit, index_t kNumSplits, bool kGroups, index_t kPageSize, bool kQoFp16>
hipError_t launch_one(const ck_tile::FmhaDecodePaged5dHostArgs& h, hipStream_t stream)
{
    using K = ck_tile::FmhaDecodePaged5dKernel<kWaveSplit, kNumSplits, kGroups, kPageSize, kQoFp16>;
    auto kargs = K::MakeKargs(h);
    hipLaunchAttribute attr[1];
    attr[0].id               = hipLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = kNumSplits;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    if constexpr(kNumSplits > 15)
    {
        // see fmha_decode_paged5d_entry_launch_cluster
        static const hipFunction_t fn = [] {
            hipFunction_t f = nullptr;
            HIP_CHECK_ERROR(
                hipGetFuncBySymbol(&f,
                                   reinterpret_cast<const void*>(
                                       &ck_tile::fmha_decode_paged5d_entry_launch_cluster<K>)));
            return f;
        }();
        const dim3 grid = K::GridSize(h.batch, h.nhead_k, h.num_groups);
        HIP_LAUNCH_CONFIG cfg{};
        cfg.gridDimX    = grid.x;
        cfg.gridDimY    = grid.y;
        cfg.gridDimZ    = grid.z;
        cfg.blockDimX   = K::kBlockSize;
        cfg.blockDimY   = 1;
        cfg.blockDimZ   = 1;
        cfg.hStream     = stream;
        cfg.attrs       = attr;
        cfg.numAttrs    = 1;
        void* params[1] = {&kargs};
        return hipDrvLaunchKernelEx(&cfg, fn, params, nullptr);
    }
    else
    {
        hipLaunchConfig_t cfg{};
        cfg.gridDim  = K::GridSize(h.batch, h.nhead_k, h.num_groups);
        cfg.blockDim = K::BlockSize();
        cfg.stream   = stream;
        // must match the cluster_dims of the entry
        cfg.attrs    = attr;
        cfg.numAttrs = 1;
        return hipLaunchKernelEx(&cfg, ck_tile::fmha_decode_paged5d_entry<K>, kargs);
    }
}

template <bool kWaveSplit, bool kGroups, index_t kPageSize, bool kQoFp16, index_t... Ns>
hipError_t dispatch(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                    index_t splits,
                    hipStream_t stream,
                    std::integer_sequence<index_t, Ns...>)
{
    hipError_t e       = hipErrorInvalidValue;
    const auto try_one = [&](auto ns) {
        if(splits == decltype(ns)::value)
            e = launch_one<kWaveSplit, decltype(ns)::value, kGroups, kPageSize, kQoFp16>(h, stream);
    };
    (try_one(ck_tile::number<Ns + 1>{}), ...);
    return e;
}

template <index_t kPageSize, bool kQoFp16>
hipError_t launch_page(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                       const ck_tile::FmhaDecodePaged5dConfig& c,
                       hipStream_t stream)
{
    const auto seq = std::make_integer_sequence<index_t, ck_tile::kMaxClusterSplits>{};
    if(c.num_groups > 1)
        return c.wave_split
                   ? dispatch<true, true, kPageSize, kQoFp16>(h, c.num_splits, stream, seq)
                   : dispatch<false, true, kPageSize, kQoFp16>(h, c.num_splits, stream, seq);
    return c.wave_split ? dispatch<true, false, kPageSize, kQoFp16>(h, c.num_splits, stream, seq)
                        : dispatch<false, false, kPageSize, kQoFp16>(h, c.num_splits, stream, seq);
}

template <bool kQoFp16>
hipError_t launch_type(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                       const ck_tile::FmhaDecodePaged5dConfig& c,
                       hipStream_t stream)
{
    switch(h.page_size)
    {
    case 16: return launch_page<16, kQoFp16>(h, c, stream);
    case 64: return launch_page<64, kQoFp16>(h, c, stream);
    case 128: return launch_page<128, kQoFp16>(h, c, stream);
    default: return hipErrorInvalidValue;
    }
}

inline hipError_t launch(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                         const ck_tile::FmhaDecodePaged5dConfig& c,
                         hipStream_t stream)
{
    // larger cache pages need whole 4-page groups per split (see FmhaDecodePaged5dKernel::Run)
    if(h.page_size != 16 && c.pages_per_split % 4 != 0)
        return hipErrorInvalidValue;
    return h.qo_fp16 ? launch_type<true>(h, c, stream) : launch_type<false>(h, c, stream);
}

inline float bf16_to_float(uint16_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::bf16_t>(x));
}
inline float fp16_to_float(uint16_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::fp16_t>(x));
}
inline float fp8_to_float(uint8_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::fp8_t>(x));
}

struct Case
{
    index_t batch   = 1;
    index_t nhead   = 16;
    index_t nhead_k = 1;
    std::vector<index_t> seqlen_k{8192}; // one per batch, or one for all
    index_t window          = 2048;      // 0 = no window
    index_t splits          = 0;         // 0 = heuristic
    index_t pages_per_split = 0;         // 0 = heuristic
    std::string layout      = "auto";    // auto, coop, wave
    index_t max_seqlen_k    = 0;         // the bound the grid is sized for; 0 = longest seqlen_k
    index_t page_size       = 16;        // tokens per cache page: 16, 64 or 128
    bool fp16               = false;     // fp16 Q and output instead of bf16
    unsigned seed           = 1;
};

struct Result
{
    bool valid     = false; // set only when validated
    double max_err = 0, max_ref = 0, cosine = 0;
    ck_tile::FmhaDecodePaged5dConfig config{};
    float us_stream = 0, us_graph = 0; // set only when timed
    bool graph_same = true;            // graph replay wrote the output of a stream launch
    std::string error;
};

// Runs one case: random Q (bf16 or fp16), K/V (finite fp8 in [-1, 1], pages in a random order, the
// unused tail of each sequence's last page filled with NaN codes), a workspace filled with NaN
// codes, then validates against a CPU reference and/or times stream and HIP-graph launches.
inline Result run(const Case& c, bool validate, int warmup = 0, int repeat = 0)
{
    Result res;
    const index_t batch = c.batch;
    std::vector<index_t> seqlens(batch);
    for(index_t b = 0; b < batch; ++b)
        seqlens[b] = c.seqlen_k[std::min<size_t>(b, c.seqlen_k.size() - 1)];
    const index_t max_seqlen =
        std::max(c.max_seqlen_k, *std::max_element(seqlens.begin(), seqlens.end()));
    if(c.nhead % c.nhead_k != 0 || c.nhead / c.nhead_k > 16)
    {
        res.error = "h / h_k must be an integer <= 16";
        return res;
    }

    const index_t ps = c.page_size;
    if(ps != 16 && ps != 64 && ps != 128)
    {
        res.error = "page_size must be 16, 64 or 128";
        return res;
    }
    auto cfg = ck_tile::fmha_decode_paged5d_config(batch, c.nhead_k, max_seqlen, c.window, ps);
    if(c.layout == "coop" || c.layout == "wave")
        cfg.wave_split = c.layout == "wave";
    const index_t lo_align = ps > kPage ? 4 : 1;
    const index_t span     = ck_tile::fmha_decode_paged5d_max_span(max_seqlen, c.window, ps);
    if(c.pages_per_split > 0)
        cfg.pages_per_split = c.pages_per_split;
    if(c.splits > 0)
        cfg.pages_per_split = (span + c.splits - 1) / c.splits;
    if(c.pages_per_split > 0 || c.splits > 0)
    {
        cfg.pages_per_split = (cfg.pages_per_split + lo_align - 1) / lo_align * lo_align;
        const index_t total = (span + cfg.pages_per_split - 1) / cfg.pages_per_split;
        cfg.num_groups      = (total + ck_tile::kMaxClusterSplits - 1) / ck_tile::kMaxClusterSplits;
        cfg.num_splits      = (total + cfg.num_groups - 1) / cfg.num_groups;
    }
    res.config = cfg;

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    const index_t max_blocks = (max_seqlen + ps - 1) / ps;
    index_t num_blocks       = 1;
    for(index_t b = 0; b < batch; ++b)
        num_blocks += (seqlens[b] + ps - 1) / ps;
    // one cache page of one kv head, K and V alike
    const index_t page_bytes = ps * kHdim;
    const auto k_off         = [&](index_t tok, index_t d) {
        return (d / 16) * ps * 16 + tok * 16 + d % 16;
    };
    const auto v_off = [&](index_t tok, index_t d) {
        return (tok / 16) * kHdim * 16 + d * 16 + tok % 16;
    };
    std::vector<uint16_t> q(static_cast<size_t>(batch) * c.nhead * kHdim);
    for(auto& x : q)
        x = c.fp16 ? ck_tile::bit_cast<uint16_t>(
                         ck_tile::type_convert<ck_tile::fp16_t>(2.f * uni(rng)))
                   : ck_tile::bit_cast<uint16_t>(
                         ck_tile::type_convert<ck_tile::bf16_t>(2.f * uni(rng)));
    const auto qo_to_float = [&](uint16_t x) {
        return c.fp16 ? fp16_to_float(x) : bf16_to_float(x);
    };
    std::vector<uint8_t> finite;
    for(int i = 0; i < 256; ++i)
        if((i & 0x7f) != 0x7f && std::fabs(fp8_to_float(static_cast<uint8_t>(i))) <= 1.f)
            finite.push_back(static_cast<uint8_t>(i));
    std::uniform_int_distribution<size_t> pick(0, finite.size() - 1);
    std::vector<uint8_t> k(static_cast<size_t>(num_blocks) * c.nhead_k * page_bytes);
    std::vector<uint8_t> v(k.size());
    for(auto& x : k)
        x = finite[pick(rng)];
    for(auto& x : v)
        x = finite[pick(rng)];
    std::vector<int32_t> perm(num_blocks - 1);
    std::iota(perm.begin(), perm.end(), 1);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> block_table(static_cast<size_t>(batch) * max_blocks, 0);
    std::vector<int32_t> seqlens32(seqlens.begin(), seqlens.end());
    {
        size_t next = 0;
        for(index_t b = 0; b < batch; ++b)
        {
            const index_t pages = (seqlens[b] + ps - 1) / ps;
            for(index_t p = 0; p < pages; ++p)
                block_table[b * max_blocks + p] = perm[next++];
            if(pages == 0)
                continue;
            const int32_t last = block_table[b * max_blocks + pages - 1];
            for(index_t hk = 0; hk < c.nhead_k; ++hk)
                for(index_t tok = seqlens[b] - (pages - 1) * ps; tok < ps; ++tok)
                    for(index_t d = 0; d < kHdim; ++d)
                    {
                        const size_t base =
                            (static_cast<size_t>(last) * c.nhead_k + hk) * page_bytes;
                        k[base + k_off(tok, d)] = 0x7f;
                        v[base + v_off(tok, d)] = 0x7f;
                    }
        }
    }
    const float k_descale = 0.75f, v_descale = 1.25f;
    const float scale_s = 1.f / std::sqrt(static_cast<float>(kHdim));

    ck_tile::DeviceMem q_buf(q.size() * 2), k_buf(k.size()), v_buf(v.size());
    ck_tile::DeviceMem bt_buf(block_table.size() * 4), sl_buf(seqlens32.size() * 4);
    ck_tile::DeviceMem o_buf(q.size() * 2), scale_buf(8);
    ck_tile::DeviceMem ws_buf(ck_tile::fmha_decode_paged5d_workspace_bytes(
        batch, c.nhead_k, cfg.num_splits, cfg.num_groups));
    ck_tile::DeviceMem cnt_buf(ck_tile::fmha_decode_paged5d_num_counters(batch, c.nhead_k) * 4);
    HIP_CHECK_ERROR(hipMemset(ws_buf.GetDeviceBuffer(), 0xff, ws_buf.GetBufferSize()));
    cnt_buf.SetZero();
    q_buf.ToDevice(q.data());
    k_buf.ToDevice(k.data());
    v_buf.ToDevice(v.data());
    bt_buf.ToDevice(block_table.data());
    sl_buf.ToDevice(seqlens32.data());
    const float scales[2] = {k_descale, v_descale};
    scale_buf.ToDevice(scales);

    const ck_tile::FmhaDecodePaged5dHostArgs args{
        q_buf.GetDeviceBuffer(),
        k_buf.GetDeviceBuffer(),
        v_buf.GetDeviceBuffer(),
        static_cast<const int32_t*>(bt_buf.GetDeviceBuffer()),
        static_cast<const int32_t*>(sl_buf.GetDeviceBuffer()),
        static_cast<const float*>(scale_buf.GetDeviceBuffer()),
        static_cast<const float*>(scale_buf.GetDeviceBuffer()) + 1,
        o_buf.GetDeviceBuffer(),
        ws_buf.GetDeviceBuffer(),
        static_cast<int32_t*>(cnt_buf.GetDeviceBuffer()),
        batch,
        c.nhead,
        c.nhead_k,
        max_blocks,
        ps,
        c.window,
        cfg.pages_per_split,
        cfg.num_groups,
        scale_s,
        c.fp16};

    hipStream_t stream;
    HIP_CHECK_ERROR(hipDeviceSynchronize()); // the fills above, before a non-blocking stream
    HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK_ERROR(launch(args, cfg, stream));
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));

    if(validate)
    {
        std::vector<uint16_t> out(q.size());
        o_buf.FromDevice(out.data());
        const index_t gqa = c.nhead / c.nhead_k;
        double dot = 0, nr = 0, ng = 0;
        for(index_t b = 0; b < batch; ++b)
        {
            const index_t sk = seqlens[b];
            const index_t lo = c.window > 0 ? std::max(0, sk - c.window) : 0;
            for(index_t hq = 0; hq < c.nhead; ++hq)
            {
                const index_t hk     = hq / gqa;
                const auto page_base = [&](index_t t) {
                    return (static_cast<size_t>(block_table[b * max_blocks + t / ps]) * c.nhead_k +
                            hk) *
                           page_bytes;
                };
                std::vector<double> sc(std::max(0, sk - lo));
                double mx = -1e300;
                for(index_t t = lo; t < sk; ++t)
                {
                    double acc = 0;
                    for(index_t d = 0; d < kHdim; ++d)
                        acc += qo_to_float(q[(static_cast<size_t>(b) * c.nhead + hq) * kHdim + d]) *
                               fp8_to_float(k[page_base(t) + k_off(t % ps, d)]);
                    sc[t - lo] = acc * k_descale * scale_s;
                    mx         = std::max(mx, sc[t - lo]);
                }
                double l = 0;
                std::vector<double> o(kHdim, 0.0);
                for(index_t t = lo; t < sk; ++t)
                {
                    const double p = std::exp(sc[t - lo] - mx);
                    l += p;
                    for(index_t d = 0; d < kHdim; ++d)
                        o[d] += p * fp8_to_float(v[page_base(t) + v_off(t % ps, d)]);
                }
                for(index_t d = 0; d < kHdim; ++d)
                {
                    const double ref = l > 0 ? o[d] / l * v_descale : 0.0;
                    const double got =
                        qo_to_float(out[(static_cast<size_t>(b) * c.nhead + hq) * kHdim + d]);
                    res.max_err = std::max(res.max_err, std::fabs(ref - got));
                    res.max_ref = std::max(res.max_ref, std::fabs(ref));
                    dot += ref * got;
                    nr += ref * ref;
                    ng += got * got;
                }
            }
        }
        res.cosine = nr + ng <= 0.0 ? 1.0 : dot / std::sqrt(nr * ng + 1e-300);
        // fp8 Q and P: the output follows the reference direction tightly, element errors stay
        // a few percent of the largest output
        res.valid = std::isfinite(res.max_err) && res.cosine > 0.999 &&
                    res.max_err < 0.05 * std::max(res.max_ref, 1e-3);
    }

    if(repeat > 0)
    {
        for(int i = 0; i < warmup; ++i)
            HIP_CHECK_ERROR(launch(args, cfg, stream));
        hipEvent_t e0, e1;
        HIP_CHECK_ERROR(hipEventCreate(&e0));
        HIP_CHECK_ERROR(hipEventCreate(&e1));
        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
        HIP_CHECK_ERROR(hipEventRecord(e0, stream));
        for(int i = 0; i < repeat; ++i)
            HIP_CHECK_ERROR(launch(args, cfg, stream));
        HIP_CHECK_ERROR(hipEventRecord(e1, stream));
        HIP_CHECK_ERROR(hipEventSynchronize(e1));
        float ms = 0;
        HIP_CHECK_ERROR(hipEventElapsedTime(&ms, e0, e1));
        res.us_stream = ms * 1e3f / repeat;

        // HIP graph replay, how a serving engine runs decode
        hipGraph_t graph;
        hipGraphExec_t graph_exec;
        HIP_CHECK_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        for(int i = 0; i < repeat; ++i)
            HIP_CHECK_ERROR(launch(args, cfg, stream));
        HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph));
        HIP_CHECK_ERROR(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        {
            std::vector<uint16_t> ref_out(q.size()), graph_out(q.size());
            o_buf.FromDevice(ref_out.data());
            // stale partials of the stream launches must not be able to stand in
            HIP_CHECK_ERROR(
                hipMemsetAsync(o_buf.GetDeviceBuffer(), 0xff, o_buf.GetBufferSize(), stream));
            HIP_CHECK_ERROR(
                hipMemsetAsync(ws_buf.GetDeviceBuffer(), 0xff, ws_buf.GetBufferSize(), stream));
            HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, stream));
            HIP_CHECK_ERROR(hipStreamSynchronize(stream));
            o_buf.FromDevice(graph_out.data());
            res.graph_same = ref_out == graph_out;
        }
        float best = 1e30f;
        for(int r = 0; r < 5; ++r)
        {
            HIP_CHECK_ERROR(hipEventRecord(e0, stream));
            HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, stream));
            HIP_CHECK_ERROR(hipEventRecord(e1, stream));
            HIP_CHECK_ERROR(hipEventSynchronize(e1));
            HIP_CHECK_ERROR(hipEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
        }
        res.us_graph = best * 1e3f / repeat;
        HIP_CHECK_ERROR(hipGraphExecDestroy(graph_exec));
        HIP_CHECK_ERROR(hipGraphDestroy(graph));
        HIP_CHECK_ERROR(hipEventDestroy(e0));
        HIP_CHECK_ERROR(hipEventDestroy(e1));
    }
    HIP_CHECK_ERROR(hipStreamDestroy(stream));
    return res;
}

} // namespace fmha_decode_paged5d
