// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host side of fmha_fwd_decode_pagedkv for the example and the tests: random inputs in the
// vectorized cache layout, a CPU reference and timing.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
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
// largest |Q|, and |K|, |V| after the descales, of the random inputs
constexpr float kQkvMax = 3.f;
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
    // added to the context lengths the device reads: lengths beyond the block table, which the
    // kernel must not follow (for lengths that fill the table, without a window)
    index_t over_report = 0;
    // elements left unused after every query head and, three times that, after every sequence
    // of Q and of O (strided tensors); the kernel must leave the gaps of O alone
    index_t qo_gap = 0;
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

// Runs one case: random Q in [-3, 3], K/V over the whole fp8 range with descales that map it to
// [-3, 3] (the image of quantized tensors), pages in a random order, NaN codes in the
// unused tail of each last page, the workspace and the output; then validates against a CPU
// reference and/or times stream and HIP-graph launches. The reference quantizes Q to fp8 per row
// as the kernel does: against the unquantized Q the 3-bit mantissa of fp8 Q moves the outputs by
// several percent of the largest one, which would hide everything else.
//
// Case::needles shows single keys lost or counted twice, which random keys hide below the fp8
// rounding of P. The keys on both sides of every 16-token page boundary, one inside every page,
// the window's first key, the key before it and the newest key become needles, handed to the
// query rows of a kv head in turn: Q of a row is nonzero in 8 dims of its own, K of its needles
// is large in those dims and zero elsewhere, V of a needle is one-hot. A row's output is the mean
// of the V of its needles inside the window.
inline Result run(const Case& c, bool validate, int warmup = 0, int repeat = 0)
{
    Result res;
    const index_t batch = c.batch;
    std::vector<index_t> seqlens(batch);
    for(index_t b = 0; b < batch; ++b)
        seqlens[b] = c.seqlen_k[std::min<size_t>(b, c.seqlen_k.size() - 1)];
    const index_t max_seqlen =
        std::max(c.max_seqlen_k, *std::max_element(seqlens.begin(), seqlens.end()));
    if(c.nhead_k <= 0 || c.nhead <= 0 || c.nhead % c.nhead_k != 0)
    {
        res.error = "h / h_k must be an integer";
        return res;
    }

    const index_t ps = c.page_size;
    if(ps != 16 && ps != 64 && ps != 128)
    {
        res.error = "page_size must be 16, 64 or 128";
        return res;
    }
    // workgroup rows of the grid per sequence: one per kv head and 16 of its query heads
    const index_t grid_rows =
        c.nhead_k * ck_tile::fmha_fwd_decode_pagedkv_q_blocks(c.nhead, c.nhead_k);
    index_t max_cluster = ck_tile::get_max_cluster_size();
    if(c.max_cluster > 0)
        max_cluster = std::min(max_cluster, c.max_cluster);
    auto cfg = ck_tile::fmha_fwd_decode_pagedkv_config(
        batch, grid_rows, max_seqlen, c.window, ps, max_cluster);
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
    // needles are dealt to 16 classes of query rows at most (16 groups of 8 head dims); rows 16
    // apart share a class
    const index_t needle_classes = std::min<index_t>(gqa, kHdim / kNeedleDims);
    std::vector<uint16_t> q(static_cast<size_t>(batch) * c.nhead * kHdim);
    for(size_t i = 0; i < q.size(); ++i)
    {
        float x = 0.f;
        if(!c.needles)
            x = kQkvMax * uni(rng);
        else if((i % kHdim) / kNeedleDims == (i / kHdim) % gqa % needle_classes)
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
    // random inputs: K and V fill the fp8 range, the image of tensors with values up to kQkvMax;
    // needle inputs: small codes around the needles
    const float fp8_max = ck_tile::type_convert<float>(ck_tile::numeric<ck_tile::fp8_t>::max());
    const auto fill     = [&](std::vector<uint8_t>& t) {
        for(auto& x : t)
            x = c.needles ? finite[pick(rng)]
                              : ck_tile::bit_cast<uint8_t>(
                                ck_tile::type_convert<ck_tile::fp8_t>(fp8_max * uni(rng)));
    };
    fill(k);
    fill(v);
    std::vector<int32_t> perm(num_blocks - 1);
    std::iota(perm.begin(), perm.end(), 1);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> block_table(static_cast<size_t>(batch) * max_blocks, 0);
    std::vector<int32_t> seqlens32(seqlens.begin(), seqlens.end());
    for(auto& s : seqlens32)
        s += c.over_report;
    const auto window_lo = [&](index_t sk) {
        return c.window > 0 ? std::max(0, sk - c.window) : 0;
    };
    // needle tokens of each sequence, ascending; needle i belongs to the rows of class i %
    // needle_classes of the kv head
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
            // keys before the window as well: a cache may have freed their slots
            for(index_t hk = 0; hk < c.nhead_k; ++hk)
                for(index_t tok = 0; tok < window_lo(seqlens[b]); ++tok)
                    for(index_t d = 0; d < kHdim; ++d)
                    {
                        const size_t base =
                            (static_cast<size_t>(block_table[b * max_blocks + tok / ps]) *
                                 c.nhead_k +
                             hk) *
                            page_bytes;
                        k[base + k_off(tok % ps, d)] = 0x7f;
                        v[base + v_off(tok % ps, d)] = 0x7f;
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
            // inside each page, at an offset that goes through all of 1 .. 14
            for(index_t t = 0; t < seqlens[b]; t += kPage)
                if(const index_t mid = t + 1 + t / kPage * 5 % 14; mid < seqlens[b])
                    n.push_back(mid);
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
                        const bool row_dim =
                            d / kNeedleDims == static_cast<index_t>(i % needle_classes);
                        const bool v_dim = d == static_cast<index_t>(i / needle_classes % kHdim);
                        k[base + k_off(n[i] % ps, d)] = row_dim ? kFp8ThirtyTwo : 0;
                        v[base + v_off(n[i] % ps, d)] = v_dim ? kFp8One : 0;
                    }
        }
    }
    const float k_descale = c.needles ? 0.75f : kQkvMax / fp8_max;
    const float v_descale = c.needles ? 1.25f : kQkvMax / fp8_max;
    const float scale_s   = 1.f / std::sqrt(static_cast<float>(kHdim));

    // Q and O on the device: head h of sequence b at b * batch_stride + h * nhead_stride
    const index_t nhead_stride = kHdim + c.qo_gap;
    const index_t batch_stride = c.nhead * nhead_stride + 3 * c.qo_gap;
    const size_t qo_elems      = static_cast<size_t>(batch) * batch_stride;
    const auto qo_at           = [&](size_t i) {
        return i / kHdim / c.nhead * batch_stride + i / kHdim % c.nhead * nhead_stride + i % kHdim;
    };
    // the contiguous [batch][nhead][hdim] elements of a device image, and whether its gaps
    // still hold the 0xffff they were filled with
    const auto qo_read = [&](const ck_tile::DeviceMem& buf, std::vector<uint16_t>& dst) {
        std::vector<uint16_t> image(qo_elems);
        buf.FromDevice(image.data());
        size_t kept = 0;
        for(size_t i = 0; i < dst.size(); ++i)
        {
            dst[i]          = image[qo_at(i)];
            image[qo_at(i)] = 0xffff;
        }
        for(const uint16_t x : image)
            kept += x == 0xffff ? 1 : 0;
        return kept == image.size();
    };
    std::vector<uint16_t> q_image(qo_elems, 0xffff);
    for(size_t i = 0; i < q.size(); ++i)
        q_image[qo_at(i)] = q[i];
    ck_tile::DeviceMem q_buf(qo_elems * 2), k_buf(k.size()), v_buf(v.size());
    ck_tile::DeviceMem bt_buf(block_table.size() * 4), sl_buf(seqlens32.size() * 4);
    ck_tile::DeviceMem o_buf(qo_elems * 2), scale_buf(8);
    ck_tile::DeviceMem ws_buf(ck_tile::fmha_fwd_decode_pagedkv_workspace_bytes(
        batch, grid_rows, cfg.num_splits, cfg.num_groups));
    ck_tile::DeviceMem cnt_buf(ck_tile::fmha_fwd_decode_pagedkv_num_counters(batch, grid_rows) * 4);
    HIP_CHECK_ERROR(hipMemset(ws_buf.GetDeviceBuffer(), 0xff, ws_buf.GetBufferSize()));
    HIP_CHECK_ERROR(hipMemset(o_buf.GetDeviceBuffer(), 0xff, o_buf.GetBufferSize()));
    cnt_buf.SetZero();
    q_buf.ToDevice(q_image.data());
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
    args.nhead_stride_q     = nhead_stride;
    args.batch_stride_q     = batch_stride;
    args.nhead_stride_o     = nhead_stride;
    args.batch_stride_o     = batch_stride;
    args.window_size        = c.window;
    args.num_groups         = cfg.num_groups;
    args.pages_per_split    = cfg.pages_per_split;
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
        if(!qo_read(o_buf, out))
        {
            HIP_CHECK_ERROR(hipStreamDestroy(stream));
            res.error = "the kernel wrote between the output rows";
            return res;
        }
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
                // Q as the kernel uses it: the row in fp8, its largest magnitude at the fp8 maximum
                float q_row[kHdim];
                float q_max = 0.f;
                for(index_t d = 0; d < kHdim; ++d)
                {
                    q_row[d] = qo_to_float(q[(static_cast<size_t>(b) * c.nhead + hq) * kHdim + d]);
                    q_max    = std::max(q_max, std::fabs(q_row[d]));
                }
                for(index_t d = 0; d < kHdim && q_max > 0.f; ++d)
                    q_row[d] = ck_tile::type_convert<float>(ck_tile::type_convert<ck_tile::fp8_t>(
                                   q_row[d] / q_max * fp8_max)) *
                               q_max / fp8_max;
                std::vector<double> sc(std::max(0, sk - lo));
                double mx = -1e300;
                for(index_t t = lo; t < sk; ++t)
                {
                    double acc = 0;
                    for(index_t d = 0; d < kHdim; ++d)
                        acc += q_row[d] * fp8_to_float(k[page_base(t) + k_off(t % ps, d)]);
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
                for(size_t i = hq % gqa % needle_classes; i < needles[b].size();
                    i += needle_classes)
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
            qo_read(o_buf, ref_out);
            // stale partials of the stream launches must not be able to stand in
            HIP_CHECK_ERROR(
                hipMemsetAsync(o_buf.GetDeviceBuffer(), 0xff, o_buf.GetBufferSize(), stream));
            HIP_CHECK_ERROR(
                hipMemsetAsync(ws_buf.GetDeviceBuffer(), 0xff, ws_buf.GetBufferSize(), stream));
            HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, stream));
            HIP_CHECK_ERROR(hipStreamSynchronize(stream));
            res.graph_same = qo_read(o_buf, graph_out) && ref_out == graph_out;
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

// The -decode_paged=1 mode of tile_example_fmha_fwd. Of that example's options it reads b, h, h_k,
// s_k (one length, or one per sequence), d (128), mask (0, n, 2, b: every key; b:l,0: the l + 1
// newest keys), page_block_size (16, 64 or 128; 0 = 16), prec (bf16 or fp16, of Q and the output;
// K and V are fp8), num_splits (above 1: workgroups per (batch, kv head) instead of the
// heuristic), max_s_k, seed, v, warmup and repeat. One query token per sequence: s is not read.
inline int run_example(const ck_tile::ArgParser& arg_parser)
{
    const auto fail = [](const char* what) {
        std::cerr << "decode_paged: " << what << std::endl;
        return -1;
    };
    Case c;
    c.batch   = arg_parser.get_int("b");
    c.nhead   = arg_parser.get_int("h");
    c.nhead_k = arg_parser.get_int("h_k") < 0 ? c.nhead : arg_parser.get_int("h_k");
    c.seqlen_k.clear();
    for(const auto x : arg_parser.get_int_vec("s_k"))
    {
        if(x < 0)
            return fail("needs -s_k, the context lengths");
        c.seqlen_k.push_back(x);
    }
    if(c.batch <= 0 || c.seqlen_k.empty())
        return fail("needs -b >= 1 and -s_k");
    if(arg_parser.get_int("d") != kHdim)
        return fail("supports -d=128 only");

    const std::string mask = arg_parser.get_str("mask");
    if(mask == "0" || mask == "n" || mask == "2" || mask == "b")
        c.window = 0;
    else
    {
        int left = -1, right = -1;
        if(std::sscanf(mask.c_str(), "b:%d,%d", &left, &right) != 2 || left < 0 || right != 0)
            return fail("supports -mask=0, b (every key) or b:l,0 (the l + 1 newest keys)");
        c.window = left + 1;
    }

    c.page_size =
        arg_parser.get_int("page_block_size") == 0 ? kPage : arg_parser.get_int("page_block_size");
    const std::string prec = arg_parser.get_str("prec");
    if(prec != "bf16" && prec != "fp16")
        return fail("supports -prec=bf16 or fp16 (Q and the output; K and V are fp8)");
    c.fp16 = prec == "fp16";
    if(arg_parser.get_int("num_splits") > 1)
        c.splits = arg_parser.get_int("num_splits");
    c.max_seqlen_k      = arg_parser.get_int("max_s_k");
    c.seed              = arg_parser.get_int("seed") < 0 ? 1 : arg_parser.get_int("seed");
    const bool validate = arg_parser.get_int("v") != 0;

    const auto r = run(c, validate, arg_parser.get_int("warmup"), arg_parser.get_int("repeat"));
    if(!r.error.empty())
        return fail(r.error.c_str());
    if(validate)
        std::printf("max abs err %.4g (max |ref| %.4g), cos %.6f, row err/tol %.3f, ",
                    r.max_err,
                    r.max_ref,
                    r.cosine,
                    r.row_err);
    std::printf(
        "%s b:%d h:%d/%d s_k:%d page:%d window:%d layout:%s groups:%d splits:%d pages/split:%d, "
        "stream %.2f us, graph %.2f us%s\n",
        c.fp16 ? "fp16" : "bf16",
        c.batch,
        c.nhead,
        c.nhead_k,
        c.seqlen_k.front(),
        c.page_size,
        c.window,
        r.config.wave_split ? "wave" : "coop",
        r.config.num_groups,
        r.config.num_splits,
        r.config.pages_per_split,
        r.us_stream,
        r.us_graph,
        validate ? (r.valid ? ", valid:y" : ", valid:n") : "");
    if(!r.graph_same)
        std::printf("graph replay output differs from a stream launch\n");
    return (!validate || r.valid) && r.graph_same ? 0 : 1;
}

} // namespace fmha_fwd_decode_pagedkv_runner
