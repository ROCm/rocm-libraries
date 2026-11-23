// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host side of fmha_fwd_decode_pagedkv for the example and the tests: random inputs in the
// vectorized cache layout, a CPU reference and timing.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "fmha_fwd_decode_pagedkv.hpp"

namespace fmha_fwd_decode_pagedkv_runner {

using ck_tile::index_t;
constexpr index_t kHdim = 128;
constexpr index_t kPage = 16;
// needle keys (Case::needles): fp8 codes of 1 and 32, dims of K one query row looks at
constexpr uint8_t kFp8One       = 0x38;
constexpr uint8_t kFp8ThirtyTwo = 0x60;
constexpr index_t kNeedleDims   = kHdim / 16;

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
    bool needles            = false; // needle keys instead of random inputs, see run()
    index_t max_cluster     = 0;     // cap on the splits per cluster; 0 = what the device takes
};

struct Result
{
    bool valid     = false; // set only when validated
    double max_err = 0, max_ref = 0, cosine = 0;
    double row_err = 0; // largest error of an output row over the row's tolerance
    ck_tile::FmhaFwdDecodePagedKVConfig config{};
    float us_stream = 0, us_graph = 0; // set only when timed
    bool graph_same = true;            // graph replay wrote the output of a stream launch
    std::string error;
};

// Runs one case: random Q (bf16 or fp16), K/V (finite fp8 in [-1, 1], pages in a random order, the
// unused tail of each sequence's last page filled with NaN codes), a workspace filled with NaN
// codes, then validates against a CPU reference and/or times stream and HIP-graph launches.
//
// Random keys cannot show one key lost or counted twice: among thousands it moves the output less
// than the fp8 rounding of P does. Case::needles makes single keys visible. The keys on both
// sides of every 16-token page boundary, the window's first key, the key before it and the newest
// key become needles, handed to the query rows of a kv head in turn: Q of a row is nonzero in 8
// dims of its own, K of its needles is large in those dims and zero elsewhere, V of a needle is
// one-hot. The output of a row is then the mean of the V of its needles inside the window, and
// each of them accounts for 1 / count of one output element.
inline Result run(const Case& c, bool validate, int warmup = 0, int repeat = 0)
{
    Result res;
    const index_t batch = c.batch;
    std::vector<index_t> seqlens(batch);
    for(index_t b = 0; b < batch; ++b)
        seqlens[b] = c.seqlen_k[std::min<size_t>(b, c.seqlen_k.size() - 1)];
    const index_t max_seqlen =
        std::max(c.max_seqlen_k, *std::max_element(seqlens.begin(), seqlens.end()));
    if(c.nhead_k <= 0 || c.nhead % c.nhead_k != 0 || c.nhead / c.nhead_k > 16)
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
    index_t max_cluster = ck_tile::get_max_cluster_size();
    if(c.max_cluster > 0)
        max_cluster = std::min(max_cluster, c.max_cluster);
    auto cfg = ck_tile::fmha_fwd_decode_pagedkv_config(
        batch, c.nhead_k, max_seqlen, c.window, ps, max_cluster);
    if(c.layout == "coop" || c.layout == "wave")
        cfg.wave_split = c.layout == "wave";
    const index_t lo_align = ps > kPage ? 4 : 1;
    const index_t span     = ck_tile::fmha_fwd_decode_pagedkv_max_span(max_seqlen, c.window, ps);
    if(c.pages_per_split > 0)
        cfg.pages_per_split = c.pages_per_split;
    if(c.splits > 0)
        cfg.pages_per_split = (span + c.splits - 1) / c.splits;
    if(c.pages_per_split > 0 || c.splits > 0)
    {
        cfg.pages_per_split = (cfg.pages_per_split + lo_align - 1) / lo_align * lo_align;
        const index_t total = (span + cfg.pages_per_split - 1) / cfg.pages_per_split;
        cfg.num_groups      = (total + max_cluster - 1) / max_cluster;
        cfg.num_splits      = (total + cfg.num_groups - 1) / cfg.num_groups;
    }
    // what the kernel uses for the bound after the overrides above
    cfg.pages_per_split = ck_tile::fmha_fwd_decode_pagedkv_split_pages(
        span, cfg.num_splits, cfg.num_groups, cfg.wave_split);
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
    const index_t gqa = c.nhead / c.nhead_k;
    std::vector<uint16_t> q(static_cast<size_t>(batch) * c.nhead * kHdim);
    for(size_t i = 0; i < q.size(); ++i)
    {
        float x = 0.f;
        if(!c.needles)
            x = 2.f * uni(rng);
        else if((i % kHdim) / kNeedleDims == (i / kHdim) % gqa)
            x = 2.f;
        q[i] = c.fp16 ? ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<ck_tile::fp16_t>(x))
                      : ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<ck_tile::bf16_t>(x));
    }
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
    const auto window_lo = [&](index_t sk) {
        return c.window > 0 ? std::max(0, sk - c.window) : 0;
    };
    // needle tokens of each sequence, ascending; needle i belongs to row i % gqa of the kv head
    std::vector<std::vector<index_t>> needles(batch);
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
            if(!c.needles)
                continue;
            auto& n = needles[b];
            n       = {window_lo(seqlens[b]) - 1, window_lo(seqlens[b]), seqlens[b] - 1};
            for(index_t t = kPage; t < seqlens[b]; t += kPage)
            {
                n.push_back(t - 1);
                n.push_back(t);
            }
            std::sort(n.begin(), n.end());
            n.erase(std::unique(n.begin(), n.end()), n.end());
            n.erase(n.begin(), std::lower_bound(n.begin(), n.end(), 0));
            for(size_t i = 0; i < n.size(); ++i)
                for(index_t hk = 0; hk < c.nhead_k; ++hk)
                    for(index_t d = 0; d < kHdim; ++d)
                    {
                        const size_t base =
                            (static_cast<size_t>(block_table[b * max_blocks + n[i] / ps]) *
                                 c.nhead_k +
                             hk) *
                            page_bytes;
                        const bool row_dim = d / kNeedleDims == static_cast<index_t>(i % gqa);
                        const bool v_dim   = d == static_cast<index_t>(i / gqa % kHdim);
                        k[base + k_off(n[i] % ps, d)] = row_dim ? kFp8ThirtyTwo : 0;
                        v[base + v_off(n[i] % ps, d)] = v_dim ? kFp8One : 0;
                    }
        }
    }
    const float k_descale = 0.75f, v_descale = 1.25f;
    const float scale_s = 1.f / std::sqrt(static_cast<float>(kHdim));

    ck_tile::DeviceMem q_buf(q.size() * 2), k_buf(k.size()), v_buf(v.size());
    ck_tile::DeviceMem bt_buf(block_table.size() * 4), sl_buf(seqlens32.size() * 4);
    ck_tile::DeviceMem o_buf(q.size() * 2), scale_buf(8);
    ck_tile::DeviceMem ws_buf(ck_tile::fmha_fwd_decode_pagedkv_workspace_bytes(
        batch, c.nhead_k, cfg.num_splits, cfg.num_groups));
    ck_tile::DeviceMem cnt_buf(ck_tile::fmha_fwd_decode_pagedkv_num_counters(batch, c.nhead_k) * 4);
    HIP_CHECK_ERROR(hipMemset(ws_buf.GetDeviceBuffer(), 0xff, ws_buf.GetBufferSize()));
    cnt_buf.SetZero();
    q_buf.ToDevice(q.data());
    k_buf.ToDevice(k.data());
    v_buf.ToDevice(v.data());
    bt_buf.ToDevice(block_table.data());
    sl_buf.ToDevice(seqlens32.data());
    const float scales[2] = {k_descale, v_descale};
    scale_buf.ToDevice(scales);

    fmha_fwd_decode_pagedkv_args args{};
    args.q_ptr              = q_buf.GetDeviceBuffer();
    args.k_ptr              = k_buf.GetDeviceBuffer();
    args.v_ptr              = v_buf.GetDeviceBuffer();
    args.o_ptr              = o_buf.GetDeviceBuffer();
    args.block_table_ptr    = static_cast<const int32_t*>(bt_buf.GetDeviceBuffer());
    args.block_table_stride = max_blocks;
    args.seqlen_k_ptr       = static_cast<const int32_t*>(sl_buf.GetDeviceBuffer());
    args.k_descale_ptr      = static_cast<const float*>(scale_buf.GetDeviceBuffer());
    args.v_descale_ptr      = static_cast<const float*>(scale_buf.GetDeviceBuffer()) + 1;
    args.workspace_ptr      = ws_buf.GetDeviceBuffer();
    args.counters_ptr       = static_cast<int32_t*>(cnt_buf.GetDeviceBuffer());
    args.batch              = batch;
    args.nhead_q            = c.nhead;
    args.nhead_k            = c.nhead_k;
    args.window_size        = c.window;
    args.num_groups         = cfg.num_groups;
    args.scale_s            = scale_s;

    fmha_fwd_decode_pagedkv_traits traits{};
    traits.hdim_q          = kHdim;
    traits.data_type       = c.fp16 ? "fp16" : "bf16";
    traits.page_block_size = ps;
    traits.wave_split      = cfg.wave_split;
    traits.num_splits      = cfg.num_splits;
    if(!fmha_fwd_decode_pagedkv_is_valid(traits, args))
    {
        res.error = "invalid arguments";
        return res;
    }

    hipStream_t stream;
    HIP_CHECK_ERROR(hipDeviceSynchronize()); // the fills above, before a non-blocking stream
    HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    const ck_tile::stream_config stream_cfg{stream};
    // throws on a HIP error
    const auto launch = [&]() { return fmha_fwd_decode_pagedkv(traits, args, stream_cfg) >= 0; };
    if(!launch())
    {
        HIP_CHECK_ERROR(hipStreamDestroy(stream));
        res.error = "no instance for this configuration";
        return res;
    }
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));

    if(validate)
    {
        std::vector<uint16_t> out(q.size());
        o_buf.FromDevice(out.data());
        double dot = 0, nr = 0, ng = 0;
        for(index_t b = 0; b < batch; ++b)
        {
            const index_t sk = seqlens[b];
            const index_t lo = window_lo(sk);
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
                double row_err = 0, row_ref = 0;
                bool row_finite = true;
                for(index_t d = 0; d < kHdim; ++d)
                {
                    const double ref = l > 0 ? o[d] / l * v_descale : 0.0;
                    const double got =
                        qo_to_float(out[(static_cast<size_t>(b) * c.nhead + hq) * kHdim + d]);
                    row_finite = row_finite && std::isfinite(got);
                    row_err    = std::max(row_err, std::fabs(ref - got));
                    row_ref    = std::max(row_ref, std::fabs(ref));
                    dot += ref * got;
                    nr += ref * ref;
                    ng += got * got;
                }
                index_t seen = 0;
                for(size_t i = hq % gqa; i < needles[b].size(); i += gqa)
                    seen += needles[b][i] >= lo ? 1 : 0;
                // A row of needles: a quarter of what one key lost or counted twice moves an
                // element. Other rows carry the fp8 rounding of Q and P, a few percent of the
                // row's largest output.
                const double tol =
                    seen > 0 ? 0.25 * v_descale / seen : 0.1 * std::max(row_ref, 1e-3);
                res.row_err =
                    std::max(res.row_err,
                             row_finite ? row_err / tol : std::numeric_limits<double>::infinity());
                res.max_err = std::max(res.max_err, row_err);
                res.max_ref = std::max(res.max_ref, row_ref);
            }
        }
        res.cosine = nr + ng <= 0.0 ? 1.0 : dot / std::sqrt(nr * ng + 1e-300);
        res.valid  = res.row_err < 1.0 && res.cosine > 0.999;
    }

    if(repeat > 0)
    {
        for(int i = 0; i < warmup; ++i)
            launch();
        hipEvent_t e0, e1;
        HIP_CHECK_ERROR(hipEventCreate(&e0));
        HIP_CHECK_ERROR(hipEventCreate(&e1));
        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
        HIP_CHECK_ERROR(hipEventRecord(e0, stream));
        for(int i = 0; i < repeat; ++i)
            launch();
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
            launch();
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

} // namespace fmha_fwd_decode_pagedkv_runner
