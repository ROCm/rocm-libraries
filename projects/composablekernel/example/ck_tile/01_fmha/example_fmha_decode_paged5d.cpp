// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// gfx1250 fp8 decode attention over the SHUFFLE 5D paged KV cache (see
// ck_tile/ops/fmha/kernel/fmha_fwd_decode_paged5d_kernel.hpp), with a CPU reference.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_decode_paged5d_kernel.hpp"

namespace {

constexpr ck_tile::index_t kHdim = 128;
constexpr ck_tile::index_t kPage = 16;

template <bool kWaveSplit, ck_tile::index_t kNumSplits>
hipError_t launch_one(const ck_tile::FmhaDecodePaged5dHostArgs& h, hipStream_t stream)
{
    using K    = ck_tile::FmhaDecodePaged5dKernel<kWaveSplit, kNumSplits>;
    auto kargs = K::MakeKargs(h);
    hipLaunchConfig_t cfg{};
    cfg.gridDim  = K::GridSize(h.batch, h.nhead_k);
    cfg.blockDim = K::BlockSize();
    cfg.stream   = stream;
    hipLaunchAttribute attr[1];
    attr[0].id               = hipLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = kNumSplits;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs                = attr;
    cfg.numAttrs             = 1;
    return hipLaunchKernelEx(&cfg, ck_tile::fmha_decode_paged5d_entry<K>, kargs);
}

template <bool kWaveSplit, ck_tile::index_t... Ns>
hipError_t dispatch(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                    ck_tile::index_t splits,
                    hipStream_t stream,
                    std::integer_sequence<ck_tile::index_t, Ns...>)
{
    hipError_t e       = hipErrorInvalidValue;
    const auto try_one = [&](auto ns) {
        if(splits == decltype(ns)::value)
            e = launch_one<kWaveSplit, decltype(ns)::value>(h, stream);
    };
    (try_one(ck_tile::number<Ns + 1>{}), ...);
    return e;
}

hipError_t fmha_decode_paged5d(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                               const ck_tile::FmhaDecodePaged5dConfig& c,
                               hipStream_t stream)
{
    const auto seq = std::make_integer_sequence<ck_tile::index_t, 15>{};
    return c.wave_split ? dispatch<true>(h, c.num_splits, stream, seq)
                        : dispatch<false>(h, c.num_splits, stream, seq);
}

float bf16_to_float(uint16_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::bf16_t>(x));
}
float fp8_to_float(uint8_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::fp8_t>(x));
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "1", "batch size")
        .insert("h", "16", "number of query heads")
        .insert("h_k", "1", "number of kv heads (h / h_k <= 16)")
        .insert("s_k", "8192", "context length of every sequence")
        .insert("window", "2048", "keys attended, the query's own included; 0 = no window")
        .insert("splits", "0", "workgroups per (batch, kv head); 0 = heuristic")
        .insert("pages_per_split", "0", "pages per split; 0 = heuristic")
        .insert("layout", "auto", "auto, coop or wave")
        .insert("v", "1", "validate against the CPU reference")
        .insert("seed", "1", "random seed")
        .insert("warmup", "10", "warmup launches")
        .insert("repeat", "100", "timed launches");
    if(!arg_parser.parse(argc, argv))
        return -1;

    const ck_tile::index_t batch    = arg_parser.get_int("b");
    const ck_tile::index_t nhead    = arg_parser.get_int("h");
    const ck_tile::index_t nhead_k  = arg_parser.get_int("h_k");
    const ck_tile::index_t seqlen_k = arg_parser.get_int("s_k");
    const ck_tile::index_t window   = arg_parser.get_int("window");
    const std::string layout        = arg_parser.get_str("layout");
    const bool validate             = arg_parser.get_bool("v");
    const int repeat                = arg_parser.get_int("repeat");
    const int warmup                = arg_parser.get_int("warmup");
    if(nhead % nhead_k != 0 || nhead / nhead_k > 16)
    {
        std::cerr << "h / h_k must be an integer <= 16" << std::endl;
        return -1;
    }

    auto cfg = ck_tile::fmha_decode_paged5d_config(batch, nhead_k, seqlen_k, window);
    if(layout == "coop" || layout == "wave")
        cfg.wave_split = layout == "wave";
    const ck_tile::index_t lo_pages = window > 0 ? std::max(0, seqlen_k - window) / kPage : 0;
    const ck_tile::index_t span     = std::max(1, (seqlen_k + kPage - 1) / kPage - lo_pages);
    if(arg_parser.get_int("pages_per_split") > 0)
        cfg.pages_per_split = arg_parser.get_int("pages_per_split");
    if(arg_parser.get_int("splits") > 0)
        cfg.pages_per_split =
            (span + arg_parser.get_int("splits") - 1) / arg_parser.get_int("splits");
    cfg.num_splits = (span + cfg.pages_per_split - 1) / cfg.pages_per_split;
    if(cfg.num_splits > 15)
    {
        std::cerr << "more than 15 splits; raise -pages_per_split" << std::endl;
        return -1;
    }

    // inputs: Q bf16, K/V fp8 from finite e4m3 values in [-1, 1], pages in a random order;
    // the bytes of the last page past the context are NaN codes
    std::mt19937 rng(arg_parser.get_int("seed"));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    const ck_tile::index_t pages_per_seq = (seqlen_k + kPage - 1) / kPage;
    const ck_tile::index_t num_blocks    = batch * pages_per_seq + 1;
    const ck_tile::index_t page_bytes    = kPage * kHdim;
    std::vector<uint16_t> q(static_cast<size_t>(batch) * nhead * kHdim);
    for(auto& x : q)
        x = ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<ck_tile::bf16_t>(2.f * uni(rng)));
    std::vector<uint8_t> finite;
    for(int i = 0; i < 256; ++i)
        if((i & 0x7f) != 0x7f && std::fabs(fp8_to_float(i)) <= 1.f)
            finite.push_back(i);
    std::uniform_int_distribution<size_t> pick(0, finite.size() - 1);
    std::vector<uint8_t> k(static_cast<size_t>(num_blocks) * nhead_k * page_bytes);
    std::vector<uint8_t> v(k.size());
    for(auto& x : k)
        x = finite[pick(rng)];
    for(auto& x : v)
        x = finite[pick(rng)];
    std::vector<int32_t> perm(num_blocks - 1);
    std::iota(perm.begin(), perm.end(), 1);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> block_table(static_cast<size_t>(batch) * pages_per_seq);
    std::vector<int32_t> seqlens(batch, seqlen_k);
    for(ck_tile::index_t b = 0; b < batch; ++b)
        for(ck_tile::index_t p = 0; p < pages_per_seq; ++p)
            block_table[b * pages_per_seq + p] = perm[b * pages_per_seq + p];
    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        const int32_t last = block_table[b * pages_per_seq + pages_per_seq - 1];
        for(ck_tile::index_t hk = 0; hk < nhead_k; ++hk)
            for(ck_tile::index_t tok = seqlen_k - (pages_per_seq - 1) * kPage; tok < kPage; ++tok)
                for(ck_tile::index_t d = 0; d < kHdim; ++d)
                {
                    const size_t base = (static_cast<size_t>(last) * nhead_k + hk) * page_bytes;
                    k[base + (d / 16) * 256 + tok * 16 + d % 16] = 0x7f;
                    v[base + d * 16 + tok]                       = 0x7f;
                }
    }
    const float k_descale = 0.75f, v_descale = 1.25f;
    const float scale_s = 1.f / std::sqrt(static_cast<float>(kHdim));

    ck_tile::DeviceMem q_buf(q.size() * 2), k_buf(k.size()), v_buf(v.size());
    ck_tile::DeviceMem bt_buf(block_table.size() * 4), sl_buf(seqlens.size() * 4);
    ck_tile::DeviceMem o_buf(q.size() * 2), scale_buf(8);
    ck_tile::DeviceMem ws_buf(
        ck_tile::fmha_decode_paged5d_workspace_bytes(batch, nhead_k, cfg.num_splits));
    q_buf.ToDevice(q.data());
    k_buf.ToDevice(k.data());
    v_buf.ToDevice(v.data());
    bt_buf.ToDevice(block_table.data());
    sl_buf.ToDevice(seqlens.data());
    const float scales[2] = {k_descale, v_descale};
    scale_buf.ToDevice(scales);

    ck_tile::FmhaDecodePaged5dHostArgs args{q_buf.GetDeviceBuffer(),
                                            k_buf.GetDeviceBuffer(),
                                            v_buf.GetDeviceBuffer(),
                                            static_cast<const int32_t*>(bt_buf.GetDeviceBuffer()),
                                            static_cast<const int32_t*>(sl_buf.GetDeviceBuffer()),
                                            static_cast<const float*>(scale_buf.GetDeviceBuffer()),
                                            static_cast<const float*>(scale_buf.GetDeviceBuffer()) +
                                                1,
                                            o_buf.GetDeviceBuffer(),
                                            ws_buf.GetDeviceBuffer(),
                                            batch,
                                            nhead,
                                            nhead_k,
                                            pages_per_seq,
                                            window,
                                            cfg.pages_per_split,
                                            scale_s};

    hipStream_t stream;
    HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK_ERROR(fmha_decode_paged5d(args, cfg, stream));
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));

    bool pass = true;
    if(validate)
    {
        std::vector<uint16_t> out(q.size());
        o_buf.FromDevice(out.data());
        const ck_tile::index_t lo  = window > 0 ? std::max(0, seqlen_k - window) : 0;
        const ck_tile::index_t gqa = nhead / nhead_k;
        double max_err = 0, max_ref = 0, dot = 0, nr = 0, ng = 0;
        for(ck_tile::index_t b = 0; b < batch; ++b)
            for(ck_tile::index_t hq = 0; hq < nhead; ++hq)
            {
                const ck_tile::index_t hk = hq / gqa;
                std::vector<double> sc(seqlen_k - lo);
                double mx = -1e300;
                for(ck_tile::index_t t = lo; t < seqlen_k; ++t)
                {
                    const size_t base =
                        (static_cast<size_t>(block_table[b * pages_per_seq + t / kPage]) * nhead_k +
                         hk) *
                        page_bytes;
                    double acc = 0;
                    for(ck_tile::index_t d = 0; d < kHdim; ++d)
                        acc += bf16_to_float(q[(static_cast<size_t>(b) * nhead + hq) * kHdim + d]) *
                               fp8_to_float(k[base + (d / 16) * 256 + (t % kPage) * 16 + d % 16]);
                    sc[t - lo] = acc * k_descale * scale_s;
                    mx         = std::max(mx, sc[t - lo]);
                }
                double l = 0;
                std::vector<double> o(kHdim, 0.0);
                for(ck_tile::index_t t = lo; t < seqlen_k; ++t)
                {
                    const size_t base =
                        (static_cast<size_t>(block_table[b * pages_per_seq + t / kPage]) * nhead_k +
                         hk) *
                        page_bytes;
                    const double p = std::exp(sc[t - lo] - mx);
                    l += p;
                    for(ck_tile::index_t d = 0; d < kHdim; ++d)
                        o[d] += p * fp8_to_float(v[base + d * 16 + t % kPage]);
                }
                for(ck_tile::index_t d = 0; d < kHdim; ++d)
                {
                    const double ref = o[d] / l * v_descale;
                    const double got =
                        bf16_to_float(out[(static_cast<size_t>(b) * nhead + hq) * kHdim + d]);
                    max_err = std::max(max_err, std::fabs(ref - got));
                    max_ref = std::max(max_ref, std::fabs(ref));
                    dot += ref * got;
                    nr += ref * ref;
                    ng += got * got;
                }
            }
        const double cosine = dot / std::sqrt(nr * ng + 1e-300);
        // fp8 Q and P: the output tracks the reference direction tightly, element errors are a
        // few percent of the largest output
        pass = std::isfinite(max_err) && cosine > 0.999 && max_err < 0.05 * std::max(max_ref, 1e-3);
        std::printf("max abs err %.4g (max |ref| %.4g), cos %.6f, ", max_err, max_ref, cosine);
    }

    for(int i = 0; i < warmup; ++i)
        HIP_CHECK_ERROR(fmha_decode_paged5d(args, cfg, stream));
    hipEvent_t e0, e1;
    HIP_CHECK_ERROR(hipEventCreate(&e0));
    HIP_CHECK_ERROR(hipEventCreate(&e1));
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));
    HIP_CHECK_ERROR(hipEventRecord(e0, stream));
    for(int i = 0; i < repeat; ++i)
        HIP_CHECK_ERROR(fmha_decode_paged5d(args, cfg, stream));
    HIP_CHECK_ERROR(hipEventRecord(e1, stream));
    HIP_CHECK_ERROR(hipEventSynchronize(e1));
    float ms_stream = 0;
    HIP_CHECK_ERROR(hipEventElapsedTime(&ms_stream, e0, e1));

    // HIP graph replay, how a serving engine runs decode
    hipGraph_t graph;
    hipGraphExec_t graph_exec;
    HIP_CHECK_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for(int i = 0; i < repeat; ++i)
        HIP_CHECK_ERROR(fmha_decode_paged5d(args, cfg, stream));
    HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph));
    HIP_CHECK_ERROR(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, stream));
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));
    float ms_graph = 1e30f;
    for(int r = 0; r < 5; ++r)
    {
        HIP_CHECK_ERROR(hipEventRecord(e0, stream));
        HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, stream));
        HIP_CHECK_ERROR(hipEventRecord(e1, stream));
        HIP_CHECK_ERROR(hipEventSynchronize(e1));
        float ms = 0;
        HIP_CHECK_ERROR(hipEventElapsedTime(&ms, e0, e1));
        ms_graph = std::min(ms_graph, ms);
    }
    HIP_CHECK_ERROR(hipGraphExecDestroy(graph_exec));
    HIP_CHECK_ERROR(hipGraphDestroy(graph));

    std::printf("b:%d h:%d/%d s_k:%d window:%d layout:%s splits:%d pages/split:%d, "
                "stream %.2f us, graph %.2f us%s\n",
                batch,
                nhead,
                nhead_k,
                seqlen_k,
                window,
                cfg.wave_split ? "wave" : "coop",
                cfg.num_splits,
                cfg.pages_per_split,
                ms_stream * 1e3f / repeat,
                ms_graph * 1e3f / repeat,
                validate ? (pass ? ", valid:y" : ", valid:n") : "");
    return pass ? 0 : 1;
}
