// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host side of FmhaDecodePaged5dKernel for the example and the tests: launch dispatch over the
// compile-time split count, random inputs in the 5D layout, a CPU reference and timing.

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

template <bool kWaveSplit, index_t kNumSplits, bool kGroups>
hipError_t launch_one(const ck_tile::FmhaDecodePaged5dHostArgs& h, hipStream_t stream)
{
    using K    = ck_tile::FmhaDecodePaged5dKernel<kWaveSplit, kNumSplits, kGroups>;
    auto kargs = K::MakeKargs(h);
    hipLaunchConfig_t cfg{};
    cfg.gridDim  = K::GridSize(h.batch, h.nhead_k, h.num_groups);
    cfg.blockDim = K::BlockSize();
    cfg.stream   = stream;
    // must match the cluster_dims of the entry
    hipLaunchAttribute attr[1];
    attr[0].id               = hipLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = kNumSplits;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs                = attr;
    cfg.numAttrs             = 1;
    return hipLaunchKernelEx(&cfg, ck_tile::fmha_decode_paged5d_entry<K>, kargs);
}

template <bool kWaveSplit, bool kGroups, index_t... Ns>
hipError_t dispatch(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                    index_t splits,
                    hipStream_t stream,
                    std::integer_sequence<index_t, Ns...>)
{
    hipError_t e       = hipErrorInvalidValue;
    const auto try_one = [&](auto ns) {
        if(splits == decltype(ns)::value)
            e = launch_one<kWaveSplit, decltype(ns)::value, kGroups>(h, stream);
    };
    (try_one(ck_tile::number<Ns + 1>{}), ...);
    return e;
}

inline hipError_t launch(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                         const ck_tile::FmhaDecodePaged5dConfig& c,
                         hipStream_t stream)
{
    const auto seq = std::make_integer_sequence<index_t, 15>{};
    if(c.num_groups > 1)
        return c.wave_split ? dispatch<true, true>(h, c.num_splits, stream, seq)
                            : dispatch<false, true>(h, c.num_splits, stream, seq);
    return c.wave_split ? dispatch<true, false>(h, c.num_splits, stream, seq)
                        : dispatch<false, false>(h, c.num_splits, stream, seq);
}

inline float bf16_to_float(uint16_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::bf16_t>(x));
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
    unsigned seed           = 1;
};

struct Result
{
    bool valid     = false; // set only when validated
    double max_err = 0, max_ref = 0, cosine = 0;
    ck_tile::FmhaDecodePaged5dConfig config{};
    float us_stream = 0, us_graph = 0; // set only when timed
    std::string error;
};

// Runs one case: random Q (bf16), K/V (finite fp8 in [-1, 1], pages in a random order, the
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

    auto cfg = ck_tile::fmha_decode_paged5d_config(batch, c.nhead_k, max_seqlen, c.window);
    if(c.layout == "coop" || c.layout == "wave")
        cfg.wave_split = c.layout == "wave";
    const index_t lo_pages = c.window > 0 ? std::max(0, max_seqlen - c.window) / kPage : 0;
    const index_t span     = std::max(1, (max_seqlen + kPage - 1) / kPage - lo_pages);
    if(c.pages_per_split > 0)
        cfg.pages_per_split = c.pages_per_split;
    if(c.splits > 0)
        cfg.pages_per_split = (span + c.splits - 1) / c.splits;
    if(c.pages_per_split > 0 || c.splits > 0)
    {
        const index_t total = (span + cfg.pages_per_split - 1) / cfg.pages_per_split;
        cfg.num_groups      = (total + 14) / 15;
        cfg.num_splits      = (total + cfg.num_groups - 1) / cfg.num_groups;
    }
    res.config = cfg;

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    const index_t max_blocks = (max_seqlen + kPage - 1) / kPage;
    index_t num_blocks       = 1;
    for(index_t b = 0; b < batch; ++b)
        num_blocks += (seqlens[b] + kPage - 1) / kPage;
    const index_t page_bytes = kPage * kHdim;
    std::vector<uint16_t> q(static_cast<size_t>(batch) * c.nhead * kHdim);
    for(auto& x : q)
        x = ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<ck_tile::bf16_t>(2.f * uni(rng)));
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
            const index_t pages = (seqlens[b] + kPage - 1) / kPage;
            for(index_t p = 0; p < pages; ++p)
                block_table[b * max_blocks + p] = perm[next++];
            if(pages == 0)
                continue;
            const int32_t last = block_table[b * max_blocks + pages - 1];
            for(index_t hk = 0; hk < c.nhead_k; ++hk)
                for(index_t tok = seqlens[b] - (pages - 1) * kPage; tok < kPage; ++tok)
                    for(index_t d = 0; d < kHdim; ++d)
                    {
                        const size_t base =
                            (static_cast<size_t>(last) * c.nhead_k + hk) * page_bytes;
                        k[base + (d / 16) * 256 + tok * 16 + d % 16] = 0x7f;
                        v[base + d * 16 + tok]                       = 0x7f;
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
        c.window,
        cfg.pages_per_split,
        cfg.num_groups,
        scale_s};

    hipStream_t stream;
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
                    return (static_cast<size_t>(block_table[b * max_blocks + t / kPage]) *
                                c.nhead_k +
                            hk) *
                           page_bytes;
                };
                std::vector<double> sc(std::max(0, sk - lo));
                double mx = -1e300;
                for(index_t t = lo; t < sk; ++t)
                {
                    double acc = 0;
                    for(index_t d = 0; d < kHdim; ++d)
                        acc +=
                            bf16_to_float(q[(static_cast<size_t>(b) * c.nhead + hq) * kHdim + d]) *
                            fp8_to_float(
                                k[page_base(t) + (d / 16) * 256 + (t % kPage) * 16 + d % 16]);
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
                        o[d] += p * fp8_to_float(v[page_base(t) + d * 16 + t % kPage]);
                }
                for(index_t d = 0; d < kHdim; ++d)
                {
                    const double ref = l > 0 ? o[d] / l * v_descale : 0.0;
                    const double got =
                        bf16_to_float(out[(static_cast<size_t>(b) * c.nhead + hq) * kHdim + d]);
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
        HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, stream));
        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
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
