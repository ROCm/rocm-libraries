// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Decode / verify attention of the target model (one TP8 rank: 16 q heads, 2 kv heads, head_dim
// 64) as a ck_tile::Megakernel phase (DecodeAttentionPhase), on a paged bf16 or fp8 KV cache,
// checked against a double reference.
//   -window=2048 -ctx=4096   sliding layer;   -window=0 -ctx=65536   global layer
//   -kv=bf16|fp8  -epilogue=store|normgate  -page=16|64  -running_max=0|1
// Checks, in order:
//   1. The host emulation of the phase's split / mask / combine decomposition against the
//      reference (needs no GPU; -emulate=0 skips it).
//   2. The GPU output against the reference, over -launches launches sharing one counter
//      buffer (the split counters must come back to zero).
// Exit code: 0 pass, 1 wrong result, 2 unsupported arguments, 3 error.

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/megakernel.hpp"
#include "ck_tile/ops/megakernel/attention/decode_attention_phase.hpp"

#include "attention_reference.hpp"

namespace {

using index_t = ck_tile::index_t;

#define HIP_CHECK(expr)                                                                    \
    do                                                                                     \
    {                                                                                      \
        const hipError_t err_ = (expr);                                                    \
        if(err_ != hipSuccess)                                                             \
        {                                                                                  \
            throw std::runtime_error(std::string("HIP error ") + hipGetErrorString(err_) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));  \
        }                                                                                  \
    } while(0)

struct DeviceBuffer
{
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { HIP_CHECK(hipMalloc(&p, std::max<size_t>(bytes, 4))); }
    template <typename T>
    explicit DeviceBuffer(const std::vector<T>& v) : DeviceBuffer(v.size() * sizeof(T))
    {
        HIP_CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    ~DeviceBuffer() { (void)hipFree(p); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* as() const
    {
        return static_cast<T*>(p);
    }
};

template <typename Out>
struct OutArgs;
template <>
struct OutArgs<ck_tile::AttentionOutputStore>
{
    static auto make(void* out, index_t stride, const void*, index_t)
    {
        return ck_tile::AttentionOutputStore::Args{static_cast<ck_tile::bf16_t*>(out), stride};
    }
};
template <>
struct OutArgs<ck_tile::AttentionOutputNormGate>
{
    static auto make(void* out, index_t stride, const void* gate, index_t gate_stride)
    {
        return ck_tile::AttentionOutputNormGate::Args{static_cast<ck_tile::bf16_t*>(out),
                                                      stride,
                                                      static_cast<const ck_tile::bf16_t*>(gate),
                                                      gate_stride,
                                                      1e-5f};
    }
};

template <typename KV, typename Out, index_t kPage, bool kRunningMax>
int run(const ck_tile::ArgParser& arg)
{
    using Phase              = ck_tile::DecodeAttentionPhase<KV, Out, kPage, kRunningMax>;
    using MK                 = ck_tile::Megakernel<ck_tile::megakernel_stage<Phase>>;
    constexpr bool kNormGate = std::is_same_v<Out, ck_tile::AttentionOutputNormGate>;

    attn_ref::Config cfg;
    cfg.num_requests = arg.get_int("b");
    cfg.q_len        = arg.get_int("q_len");
    cfg.ctx          = arg.get_int("ctx");
    cfg.window       = arg.get_int("window");
    cfg.page_size    = kPage;
    cfg.fp8          = std::is_same_v<KV, ck_tile::fp8_t>;
    if(cfg.kv_len(cfg.num_requests - 1) < cfg.q_len)
    {
        std::printf("context too short for %d requests\n", cfg.num_requests);
        return 2;
    }
    const auto D    = attn_ref::make_data(cfg);
    const index_t M = cfg.M();

    DeviceBuffer d_q(D.q), d_gate(D.gate), d_k(D.k_cache), d_v(D.v_cache);
    DeviceBuffer d_bt(D.block_table), d_lens(D.kv_lens);
    DeviceBuffer d_out(size_t(M) * D.stride_q * 2);

    typename Phase::Args a{};
    a.q                  = d_q.as<ck_tile::bf16_t>();
    a.stride_q           = D.stride_q;
    a.k_cache            = d_k.as<KV>();
    a.v_cache            = d_v.as<KV>();
    a.page_stride        = D.page_stride;
    a.head_stride        = D.head_stride;
    a.token_stride       = D.token_stride;
    a.block_table        = d_bt.as<int32_t>();
    a.block_table_stride = D.pages_per_req;
    a.kv_lens            = d_lens.as<int32_t>();
    a.num_requests       = cfg.num_requests;
    a.q_len              = cfg.q_len;
    a.num_kv_heads       = cfg.kv_heads;
    a.window             = cfg.window;
    a.scale              = float(attn_ref::kScale);
    a.k_scale            = D.k_scale;
    a.v_scale            = D.v_scale;
    a.out                = OutArgs<Out>::make(d_out.p, D.stride_q, d_gate.p, D.stride_gate);

    const ck_tile::stream_config s{nullptr, false};
    const index_t resident = MK::MaxResidentGridSize(s);
    a.num_splits           = arg.get_int("splits");
    if(a.num_splits <= 0)
        a.num_splits = Phase::ChooseNumSplits(
            Phase::NumTiles(a), Phase::MaxSpanKeys(cfg.ctx, cfg.q_len, cfg.window), resident);

    DeviceBuffer d_opart(Phase::PartialOFloats(a) * sizeof(float));
    DeviceBuffer d_lpart(Phase::PartialLseFloats(a) * sizeof(float));
    DeviceBuffer d_split_cnt(size_t(Phase::NumTiles(a)) * sizeof(uint32_t));
    HIP_CHECK(hipMemset(d_split_cnt.p, 0, size_t(Phase::NumTiles(a)) * sizeof(uint32_t)));
    a.o_partial      = d_opart.as<float>();
    a.lse_partial    = d_lpart.as<float>();
    a.split_counters = d_split_cnt.as<uint32_t>();

    const typename MK::ArgsTuple args{a};
    const index_t n_counters = MK::NumCounters(args);
    DeviceBuffer d_counters(size_t(n_counters) * sizeof(uint32_t));
    HIP_CHECK(hipMemset(d_counters.p, 0, size_t(n_counters) * sizeof(uint32_t)));
    const auto kargs = MK::MakeKargs(args, d_counters.as<uint32_t>());
    if(!MK::IsSupportedArgument(kargs))
    {
        std::printf("arguments not supported\n");
        return 2;
    }

    index_t grid = arg.get_int("grid");
    grid         = std::min(grid > 0 ? grid : Phase::NumWork(a), resident);
    std::printf("B %d q_len %d ctx %d window %d kv %s page %d: %d tiles x %d splits, grid %d "
                "(resident %d), LDS %d B\n",
                cfg.num_requests,
                cfg.q_len,
                cfg.ctx,
                cfg.window,
                cfg.fp8 ? "fp8" : "bf16",
                kPage,
                Phase::NumTiles(a),
                a.num_splits,
                grid,
                resident,
                static_cast<int>(MK::kSmemSize));

    auto ref = attn_ref::reference(D);
    long bad = 0;
    if(arg.get_int("emulate"))
    {
        const auto emu = attn_ref::emulate<Phase>(D, a);
        bad += emu.empty() ? 1 : attn_ref::compare("emulation", emu, ref, 1e-4, 1e-3);
    }
    if constexpr(kNormGate)
        for(index_t t = 0; t < M; ++t)
            for(index_t h = 0; h < cfg.q_heads; ++h)
                attn_ref::norm_gate(&ref[(size_t(t) * cfg.q_heads + h) * attn_ref::kHeadDim],
                                    &D.gate[size_t(t) * D.stride_gate + h * attn_ref::kHeadDim]);

    std::vector<uint16_t> out(size_t(M) * D.stride_q);
    std::vector<double> got(out.size());
    std::vector<uint32_t> cnt(Phase::NumTiles(a));
    const int launches = arg.get_int("launches");
    for(int l = 0; l < launches; ++l)
    {
        HIP_CHECK(hipMemset(d_out.p, 0xFF, out.size() * 2));
        ck_tile::launch_kernel(
            s, ck_tile::make_kernel<1>(MK{}, dim3(grid), MK::BlockSize(), 0, kargs));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(out.data(), d_out.p, out.size() * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(
            cnt.data(), d_split_cnt.p, cnt.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
        for(size_t i = 0; i < out.size(); ++i)
            got[i] = attn_ref::bf16_val(out[i]);
        // bf16 P and output: 2^-7 relative plus 2^-8 of |v| ~ 1 (fp8 V: one more e4m3 ulp).
        const double atol = kNormGate ? 1.0 / 32 : (cfg.fp8 ? 1.0 / 64 : 1.0 / 256);
        bad += attn_ref::compare("gpu", got, ref, atol, 1.0 / 128);
        for(uint32_t c : cnt)
            bad += c != 0;
    }

    std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}

template <typename KV, typename Out>
int dispatch_page(const ck_tile::ArgParser& arg)
{
    const int page = arg.get_int("page");
    const bool rm  = arg.get_int("running_max") != 0;
    if(page == 16 && !rm)
        return run<KV, Out, 16, false>(arg);
    if(page == 16 && rm)
        return run<KV, Out, 16, true>(arg);
    if(page == 64 && !rm)
        return run<KV, Out, 64, false>(arg);
    std::printf("unsupported -page=%d -running_max=%d\n", page, int(rm));
    return 2;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("b", "1", "requests")
        .insert("q_len", "16", "query rows per request (DFlash2 verify window)")
        .insert("ctx", "4096", "context of request 0 incl. the q_len rows; request b: ctx - 37b")
        .insert("window", "2048", "sliding window, 0 = global layer")
        .insert("kv", "bf16", "KV cache type: bf16 | fp8")
        .insert("page", "16", "tokens per page: 16 | 64")
        .insert("running_max", "0", "1: online softmax with a running max")
        .insert("epilogue", "store", "store | normgate (o_norm + sigmoid gate)")
        .insert("splits", "0", "KV splits, 0 = heuristic")
        .insert("grid", "0", "workgroups; 0 = min(work, resident)")
        .insert("launches", "3", "launches sharing one counter buffer")
        .insert("emulate", "1", "also check the host emulation of the decomposition");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        const std::string kv = arg.get_str("kv"), ep = arg.get_str("epilogue");
        const bool ng = ep == "normgate";
        if(ep != "store" && !ng)
            return 2;
        if(kv == "bf16")
            return ng ? dispatch_page<ck_tile::bf16_t, ck_tile::AttentionOutputNormGate>(arg)
                      : dispatch_page<ck_tile::bf16_t, ck_tile::AttentionOutputStore>(arg);
        if(kv == "fp8")
            return ng ? dispatch_page<ck_tile::fp8_t, ck_tile::AttentionOutputNormGate>(arg)
                      : dispatch_page<ck_tile::fp8_t, ck_tile::AttentionOutputStore>(arg);
        std::printf("unknown -kv=%s\n", kv.c_str());
        return 2;
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
