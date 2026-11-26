// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Cross-GPU phases (include/ck_tile/ops/megakernel/megakernel_xgpu.hpp) in persistent
// ck_tile::Megakernel launches, one process driving N GPUs (TP over XGMI peer memory; N = 1 is a
// self loop). Each device gets its own exchange buffers (uncached, filled once with the empty
// pattern), its own counter buffer and its own stream; -launches replays the launch with no host
// reset of either, with new inputs every launch, and checks every rank against a host reference.
//
//   -test=ar     two all-reduces sharing one exchange (uses 0 and 1); the second reduces the
//                first's output (after_rows), so out1 = sum over ranks of out0
//   -test=split  one all-reduce as two stages: push all, then consume
//   -test=ag     bf16 all-gather (sentinel codec) + raw-byte all-gather (LL codec, fp8-like bytes)
//   -test=barrier the system-scope rank barrier stage; afterwards every flag holds the launch count
//   -test=norm   all-reduce fused with MkResidualNormPhase (post_attention_layernorm + residual
//                gate + next CenteredRMSNorm); reference: the same row phase alone on device 0
//                reading every rank's partial through peer pointers

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/megakernel.hpp"

namespace {

using index_t = ck_tile::index_t;
using bf16_t  = ck_tile::bf16_t;

constexpr index_t kWarps    = 8;
constexpr index_t kMaxWorld = 8;
constexpr index_t kHidden   = 8192;

#define HIP_CHECK(cmd)                                                                  \
    do                                                                                  \
    {                                                                                   \
        const hipError_t e_ = (cmd);                                                    \
        if(e_ != hipSuccess)                                                            \
        {                                                                               \
            throw std::runtime_error(std::string(#cmd) + ": " + hipGetErrorString(e_)); \
        }                                                                               \
    } while(0)

// ---- phases under test
using ArStore = ck_tile::xgpu_store_epilogue<bf16_t>;
using Ar      = ck_tile::XgpuAllReducePhase<bf16_t, kWarps, ArStore, kMaxWorld>;
using ArPush =
    ck_tile::XgpuAllReducePhase<bf16_t, kWarps, ArStore, kMaxWorld, 2, ck_tile::xgpu_part::kPush>;
using ArReduce = ck_tile::
    XgpuAllReducePhase<bf16_t, kWarps, ArStore, kMaxWorld, 2, ck_tile::xgpu_part::kConsume>;
using AgBf16 = ck_tile::XgpuAllGatherPhase<ck_tile::xgpu_sentinel_codec<bf16_t>, kWarps, kMaxWorld>;
using AgBytes = ck_tile::XgpuAllGatherPhase<ck_tile::xgpu_ll_codec, kWarps, kMaxWorld>;
using NormRow = ck_tile::
    MkResidualNormPhase<kHidden, kWarps, ck_tile::MkFirstNorm::kRms, bf16_t, kMaxWorld>;
using ArNorm = ck_tile::
    XgpuAllReducePhase<bf16_t, kWarps, ck_tile::xgpu_row_phase_epilogue<NormRow>, kMaxWorld>;

using MkAr = ck_tile::Megakernel<ck_tile::megakernel_stage<Ar>,
                                 ck_tile::megakernel_stage<Ar, ck_tile::after_rows<0>>>;
using MkSplit =
    ck_tile::Megakernel<ck_tile::megakernel_stage<ArPush>, ck_tile::megakernel_stage<ArReduce>>;
using MkAg =
    ck_tile::Megakernel<ck_tile::megakernel_stage<AgBf16>, ck_tile::megakernel_stage<AgBytes>>;
using MkNorm    = ck_tile::Megakernel<ck_tile::megakernel_stage<ArNorm>>;
using MkNormRef = ck_tile::Megakernel<ck_tile::megakernel_stage<NormRow>>;
using Barrier   = ck_tile::XgpuRankBarrierPhase<kWarps, kMaxWorld>;
using MkBarrier = ck_tile::Megakernel<ck_tile::megakernel_stage<Barrier>>;

// ---- host helpers
uint16_t bf16_rne(float f)
{
    const uint32_t u = ck_tile::bit_cast<uint32_t>(f);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float bf16_to_float(uint16_t b) { return ck_tile::bit_cast<float>(static_cast<uint32_t>(b) << 16); }
uint16_t canon(uint16_t b) { return b == 0x8000u ? 0 : b; }

/// Random bf16 with some exact +0 and -0 (the sentinel bit pattern must round-trip as +0).
std::vector<uint16_t> random_bf16(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<uint16_t> v(n);
    for(size_t i = 0; i < n; ++i)
    {
        const uint64_t r = rng() % 64;
        v[i]             = r == 0 ? uint16_t{0x8000} : r == 1 ? uint16_t{0} : bf16_rne(dist(rng));
    }
    return v;
}

std::vector<uint8_t> random_bytes(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::vector<uint8_t> v(n);
    for(auto& x : v)
        x = static_cast<uint8_t>(rng());
    return v;
}

struct DeviceBuffer
{
    int dev      = 0;
    void* p      = nullptr;
    size_t bytes = 0;
    DeviceBuffer(int d, size_t n, bool uncached = false) : dev(d), bytes(n)
    {
        HIP_CHECK(hipSetDevice(dev));
        if(uncached)
            HIP_CHECK(hipExtMallocWithFlags(&p, n, hipDeviceMallocUncached));
        else
            HIP_CHECK(hipMalloc(&p, n));
    }
    ~DeviceBuffer()
    {
        if(p != nullptr)
        {
            (void)hipSetDevice(dev);
            (void)hipFree(p);
        }
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : dev(o.dev), p(o.p), bytes(o.bytes) { o.p = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    template <typename T>
    T* as() const
    {
        return static_cast<T*>(p);
    }
    void upload(const void* src, size_t n) const
    {
        HIP_CHECK(hipSetDevice(dev));
        HIP_CHECK(hipMemcpy(p, src, n, hipMemcpyHostToDevice));
    }
    template <typename T>
    std::vector<T> download() const
    {
        std::vector<T> v(bytes / sizeof(T));
        HIP_CHECK(hipSetDevice(dev));
        HIP_CHECK(hipMemcpy(v.data(), p, bytes, hipMemcpyDeviceToHost));
        return v;
    }
    void fill(uint8_t b) const
    {
        HIP_CHECK(hipSetDevice(dev));
        HIP_CHECK(hipMemset(p, b, bytes));
    }
};

/// One exchange: a buffer per rank, peers map each other's (peer access is enabled up front), and
/// per rank a device table of the peer pointers.
struct Exchange
{
    std::vector<DeviceBuffer> bufs, tables;
    uint32_t slot_bytes   = 0;
    uint32_t buffer_bytes = 0;

    Exchange(int world, uint32_t slot, uint32_t total, uint32_t fill_word)
        : slot_bytes(slot), buffer_bytes(total)
    {
        std::vector<void*> ptrs;
        for(int d = 0; d < world; ++d)
        {
            bufs.emplace_back(d, total, true);
            HIP_CHECK(hipMemsetD32(
                static_cast<hipDeviceptr_t>(bufs.back().p), fill_word, total / sizeof(uint32_t)));
            ptrs.push_back(bufs.back().p);
        }
        // With peer access enabled a device pointer is valid on every peer; with IPC handles each
        // rank would put its own mappings here.
        ptrs.resize(kMaxWorld, nullptr); // padded: no table read can leave the allocation
        for(int d = 0; d < world; ++d)
        {
            tables.emplace_back(d, ptrs.size() * sizeof(void*));
            tables.back().upload(ptrs.data(), ptrs.size() * sizeof(void*));
        }
    }

    ck_tile::xgpu_exchange<kMaxWorld>
    view(int rank, const uint32_t* launch_state, uint32_t uses, uint32_t use, uint32_t spin) const
    {
        ck_tile::xgpu_exchange<kMaxWorld> x{};
        x.peers           = tables[static_cast<size_t>(rank)].as<void* const>();
        x.rank            = rank;
        x.world           = static_cast<index_t>(bufs.size());
        x.slot_bytes      = slot_bytes;
        x.buffer_bytes    = buffer_bytes;
        x.launch_state    = launch_state;
        x.uses_per_launch = uses;
        x.use_index       = use;
        x.spin_limit      = spin;
        return x;
    }
};

struct Options
{
    int world;
    index_t m;
    index_t width;
    index_t shard;
    int launches;
    int repeat;
    index_t grid;
    uint32_t spin;
};

/// Per-device megakernel state: counters, kernel arguments, grid.
template <typename MK>
struct Launcher
{
    std::vector<DeviceBuffer> counters;
    std::vector<typename MK::Kargs> kargs;
    std::vector<hipStream_t> streams;
    index_t grid = 0;

    /// make_args(rank, launch_state) -> ArgsTuple.
    template <typename F>
    Launcher(const Options& o, const std::vector<hipStream_t>& s, F&& make_args) : streams(s)
    {
        index_t work = 1;
        for(int r = 0; r < o.world; ++r)
        {
            // NumCounters does not depend on the pointers: size first, then bind launch_state.
            const auto probe = make_args(r, nullptr);
            const index_t n  = MK::NumCounters(probe);
            counters.emplace_back(r, size_t(n) * sizeof(uint32_t));
            counters.back().fill(0);
            const auto args = make_args(r, counters.back().template as<uint32_t>());
            kargs.push_back(MK::MakeKargs(args, counters.back().template as<uint32_t>()));
            if(!MK::IsSupportedArgument(kargs.back()))
                throw std::runtime_error("megakernel arguments not supported");
            ck_tile::static_for<0, MK::kNumStages, 1>{}([&](auto i) {
                work = std::max(
                    work,
                    MK::template phase_t<i.value>::NumWork(args.at(ck_tile::number<i.value>{})));
            });
        }
        HIP_CHECK(hipSetDevice(0));
        const index_t resident = MK::MaxResidentGridSize(ck_tile::stream_config{streams[0]});
        // Ranks may use different grids; within a launch, the stages sharing an exchange have
        // the same NumWork, so one workgroup owns each cell (ordering note in megakernel_xgpu.hpp).
        grid = o.grid > 0 ? std::min(o.grid, resident) : std::min(work, resident);
        std::printf("  %d stage(s), %d work items max, grid %d (resident %d), LDS %d B\n",
                    static_cast<int>(MK::kNumStages),
                    work,
                    grid,
                    resident,
                    static_cast<int>(MK::kSmemSize));
    }

    void launch_all() const
    {
        for(size_t r = 0; r < kargs.size(); ++r)
        {
            HIP_CHECK(hipSetDevice(static_cast<int>(r)));
            ck_tile::launch_kernel(
                ck_tile::stream_config{streams[r]},
                ck_tile::make_kernel<1>(MK{}, dim3(grid), MK::BlockSize(), 0, kargs[r]));
        }
    }

    void sync_all() const
    {
        for(size_t r = 0; r < streams.size(); ++r)
        {
            HIP_CHECK(hipSetDevice(static_cast<int>(r)));
            HIP_CHECK(hipStreamSynchronize(streams[r]));
        }
    }

    int check_epochs(uint32_t expected) const
    {
        int bad = 0;
        for(size_t r = 0; r < counters.size(); ++r)
        {
            uint32_t e = 0;
            HIP_CHECK(hipSetDevice(static_cast<int>(r)));
            HIP_CHECK(hipMemcpy(&e, counters[r].p, sizeof(e), hipMemcpyDeviceToHost));
            if(e != expected)
            {
                std::printf("  rank %zu: device epoch %u, expected %u\n", r, e, expected);
                ++bad;
            }
        }
        return bad;
    }

    /// Back-to-back launches with no host sync; returns us per launch (max over devices).
    double time(int repeat) const
    {
        std::vector<hipEvent_t> start(streams.size()), stop(streams.size());
        for(size_t r = 0; r < streams.size(); ++r)
        {
            HIP_CHECK(hipSetDevice(static_cast<int>(r)));
            HIP_CHECK(hipEventCreate(&start[r]));
            HIP_CHECK(hipEventCreate(&stop[r]));
        }
        launch_all(); // warm up
        sync_all();
        for(size_t r = 0; r < streams.size(); ++r)
        {
            HIP_CHECK(hipSetDevice(static_cast<int>(r)));
            HIP_CHECK(hipEventRecord(start[r], streams[r]));
        }
        for(int i = 0; i < repeat; ++i)
            launch_all();
        float worst = 0.f;
        for(size_t r = 0; r < streams.size(); ++r)
        {
            HIP_CHECK(hipSetDevice(static_cast<int>(r)));
            HIP_CHECK(hipEventRecord(stop[r], streams[r]));
        }
        for(size_t r = 0; r < streams.size(); ++r)
        {
            HIP_CHECK(hipSetDevice(static_cast<int>(r)));
            HIP_CHECK(hipEventSynchronize(stop[r]));
            float ms = 0.f;
            HIP_CHECK(hipEventElapsedTime(&ms, start[r], stop[r]));
            worst = std::max(worst, ms);
            HIP_CHECK(hipEventDestroy(start[r]));
            HIP_CHECK(hipEventDestroy(stop[r]));
        }
        return 1e3 * worst / std::max(1, repeat);
    }
};

int compare_u16(const char* what,
                int rank,
                const std::vector<uint16_t>& got,
                const std::vector<uint16_t>& ref)
{
    int bad = 0;
    for(size_t i = 0; i < ref.size(); ++i)
    {
        if(got[i] != ref[i])
        {
            if(bad < 4)
                std::printf(
                    "  rank %d %s[%zu] = 0x%04x, expected 0x%04x\n", rank, what, i, got[i], ref[i]);
            ++bad;
        }
    }
    return bad;
}

/// Host all-reduce: fp32 sum in rank order of the canonical (-0 -> +0) values, then bf16.
std::vector<uint16_t> host_allreduce(const std::vector<std::vector<uint16_t>>& in)
{
    std::vector<uint16_t> out(in[0].size());
    for(size_t i = 0; i < out.size(); ++i)
    {
        float s = 0.f;
        for(const auto& v : in)
            s += bf16_to_float(canon(v[i]));
        out[i] = bf16_rne(s);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------

int run_ar(const Options& o, const std::vector<hipStream_t>& streams)
{
    const index_t M = o.m, W = o.width;
    Exchange ex(o.world,
                Ar::SlotBytes(M, W),
                Ar::ExchangeBytes(o.world, Ar::SlotBytes(M, W)),
                Ar::kFillWord);
    std::vector<DeviceBuffer> in, out0, out1;
    for(int r = 0; r < o.world; ++r)
    {
        in.emplace_back(r, size_t(M) * W * 2);
        out0.emplace_back(r, size_t(M) * W * 2);
        out1.emplace_back(r, size_t(M) * W * 2);
    }
    Launcher<MkAr> L(o, streams, [&](int r, const uint32_t* state) {
        const Ar::Args a0{ex.view(r, state, 2, 0, o.spin),
                          in[r].as<bf16_t>(),
                          W,
                          M,
                          W,
                          {out0[r].as<bf16_t>(), W}};
        const Ar::Args a1{ex.view(r, state, 2, 1, o.spin),
                          out0[r].as<bf16_t>(),
                          W,
                          M,
                          W,
                          {out1[r].as<bf16_t>(), W}};
        return MkAr::ArgsTuple{a0, a1};
    });

    int failures = 0;
    for(int l = 0; l < o.launches; ++l)
    {
        std::vector<std::vector<uint16_t>> x(o.world);
        for(int r = 0; r < o.world; ++r)
        {
            x[r] = random_bf16(size_t(M) * W, 1000u * l + r);
            in[r].upload(x[r].data(), x[r].size() * 2);
            out0[r].fill(0xFF);
            out1[r].fill(0xFF);
        }
        const auto ref0 = host_allreduce(x);
        const auto ref1 = host_allreduce(std::vector<std::vector<uint16_t>>(o.world, ref0));
        L.launch_all();
        L.sync_all();
        int bad = 0;
        for(int r = 0; r < o.world; ++r)
        {
            bad += compare_u16("out0", r, out0[r].download<uint16_t>(), ref0);
            bad += compare_u16("out1", r, out1[r].download<uint16_t>(), ref1);
        }
        if(bad != 0)
        {
            ++failures;
            std::printf("  launch %d: %d mismatches\n", l, bad);
        }
    }
    failures += L.check_epochs(static_cast<uint32_t>(o.launches));
    if(o.repeat > 0)
        std::printf("  %.2f us per launch (2 all-reduces of %d x %d bf16, %d ranks)\n",
                    L.time(o.repeat),
                    M,
                    W,
                    o.world);
    return failures;
}

int run_split(const Options& o, const std::vector<hipStream_t>& streams)
{
    const index_t M = o.m, W = o.width;
    Exchange ex(o.world,
                Ar::SlotBytes(M, W),
                Ar::ExchangeBytes(o.world, Ar::SlotBytes(M, W)),
                Ar::kFillWord);
    std::vector<DeviceBuffer> in, out;
    for(int r = 0; r < o.world; ++r)
    {
        in.emplace_back(r, size_t(M) * W * 2);
        out.emplace_back(r, size_t(M) * W * 2);
    }
    Launcher<MkSplit> L(o, streams, [&](int r, const uint32_t* state) {
        const auto x = ex.view(r, state, 1, 0, o.spin);
        const ArPush::Args p{x, in[r].as<bf16_t>(), W, M, W, {out[r].as<bf16_t>(), W}};
        const ArReduce::Args c{x, in[r].as<bf16_t>(), W, M, W, {out[r].as<bf16_t>(), W}};
        return MkSplit::ArgsTuple{p, c};
    });
    int failures = 0;
    for(int l = 0; l < o.launches; ++l)
    {
        std::vector<std::vector<uint16_t>> x(o.world);
        for(int r = 0; r < o.world; ++r)
        {
            x[r] = random_bf16(size_t(M) * W, 2000u * l + r);
            in[r].upload(x[r].data(), x[r].size() * 2);
            out[r].fill(0xFF);
        }
        const auto ref = host_allreduce(x);
        L.launch_all();
        L.sync_all();
        int bad = 0;
        for(int r = 0; r < o.world; ++r)
            bad += compare_u16("out", r, out[r].download<uint16_t>(), ref);
        if(bad != 0)
        {
            ++failures;
            std::printf("  launch %d: %d mismatches\n", l, bad);
        }
    }
    failures += L.check_epochs(static_cast<uint32_t>(o.launches));
    if(o.repeat > 0)
        std::printf(
            "  %.2f us per launch (1 all-reduce of %d x %d bf16 as push + consume, %d ranks)\n",
            L.time(o.repeat),
            M,
            W,
            o.world);
    return failures;
}

int run_ag(const Options& o, const std::vector<hipStream_t>& streams)
{
    const index_t M  = o.m;
    const index_t S  = o.shard;     // bf16 columns per rank
    const index_t SB = o.shard / 2; // raw bytes per rank and row (e.g. fp8 [M, 512])
    const int Wd     = o.world;
    Exchange ex_h(Wd,
                  AgBf16::SlotBytes(M, S * 2),
                  AgBf16::ExchangeBytes(Wd, AgBf16::SlotBytes(M, S * 2)),
                  AgBf16::kFillWord);
    Exchange ex_b(Wd,
                  AgBytes::SlotBytes(M, SB),
                  AgBytes::ExchangeBytes(Wd, AgBytes::SlotBytes(M, SB)),
                  AgBytes::kFillWord);
    std::vector<DeviceBuffer> in_h, in_b, out_h, out_b;
    for(int r = 0; r < Wd; ++r)
    {
        in_h.emplace_back(r, size_t(M) * S * 2);
        in_b.emplace_back(r, size_t(M) * SB);
        out_h.emplace_back(r, size_t(M) * S * 2 * Wd);
        out_b.emplace_back(r, size_t(M) * SB * Wd);
    }
    Launcher<MkAg> L(o, streams, [&](int r, const uint32_t* state) {
        // bf16: [M, world * S] row-major (N-sharded output); bytes: rank-major blocks
        // [world][M][SB].
        const AgBf16::Args h{ex_h.view(r, state, 1, 0, o.spin),
                             in_h[r].p,
                             S * 2,
                             M,
                             S * 2,
                             out_h[r].p,
                             static_cast<ck_tile::long_index_t>(Wd) * S * 2,
                             S * 2};
        const AgBytes::Args b{ex_b.view(r, state, 1, 0, o.spin),
                              in_b[r].p,
                              SB,
                              M,
                              SB,
                              out_b[r].p,
                              SB,
                              static_cast<ck_tile::long_index_t>(M) * SB};
        return MkAg::ArgsTuple{h, b};
    });
    int failures = 0;
    for(int l = 0; l < o.launches; ++l)
    {
        std::vector<uint16_t> ref_h(size_t(M) * S * Wd);
        std::vector<uint8_t> ref_b(size_t(M) * SB * Wd);
        for(int r = 0; r < Wd; ++r)
        {
            const auto xh = random_bf16(size_t(M) * S, 3000u * l + r);
            const auto xb = random_bytes(size_t(M) * SB, 4000u * l + r);
            in_h[r].upload(xh.data(), xh.size() * 2);
            in_b[r].upload(xb.data(), xb.size());
            out_h[r].fill(0xFF);
            out_b[r].fill(0xFF);
            for(index_t i = 0; i < M; ++i)
            {
                for(index_t j = 0; j < S; ++j)
                    ref_h[size_t(i) * S * Wd + size_t(r) * S + j] = canon(xh[size_t(i) * S + j]);
                for(index_t j = 0; j < SB; ++j)
                    ref_b[(size_t(r) * M + i) * SB + j] = xb[size_t(i) * SB + j];
            }
        }
        L.launch_all();
        L.sync_all();
        int bad = 0;
        for(int r = 0; r < Wd; ++r)
        {
            bad += compare_u16("bf16 gather", r, out_h[r].download<uint16_t>(), ref_h);
            const auto gb = out_b[r].download<uint8_t>();
            const int bb  = static_cast<int>(
                gb.size() -
                static_cast<size_t>(std::mismatch(gb.begin(), gb.end(), ref_b.begin()).first -
                                    gb.begin()));
            if(bb != 0)
                std::printf("  rank %d byte gather differs from byte %zu on\n", r, gb.size() - bb);
            bad += bb != 0;
        }
        if(bad != 0)
        {
            ++failures;
            std::printf("  launch %d: %d mismatches\n", l, bad);
        }
    }
    failures += L.check_epochs(static_cast<uint32_t>(o.launches));
    if(o.repeat > 0)
        std::printf(
            "  %.2f us per launch (all-gathers of %d x %d bf16 and %d x %d B per rank, %d ranks)\n",
            L.time(o.repeat),
            M,
            S,
            M,
            SB,
            Wd);
    return failures;
}

int run_norm(const Options& o, const std::vector<hipStream_t>& streams)
{
    const index_t M = o.m, W = kHidden;
    const int Wd = o.world;
    Exchange ex(Wd,
                ArNorm::SlotBytes(M, W),
                ArNorm::ExchangeBytes(Wd, ArNorm::SlotBytes(M, W)),
                ArNorm::kFillWord);

    // Weights, replicated on every rank.
    std::vector<float> gate(W), alpha(W), beta(W);
    std::mt19937_64 rng(7);
    std::normal_distribution<float> dist(0.f, 1.f);
    for(auto& g : gate)
        g = dist(rng);
    ck_tile::mk_residual_gate_coeffs(gate.data(), W, alpha.data(), beta.data());
    const auto next_w = random_bf16(W, 8);

    std::vector<DeviceBuffer> in, res, d_alpha, d_beta, d_next, x_out, h_out;
    for(int r = 0; r < Wd; ++r)
    {
        in.emplace_back(r, size_t(M) * W * 2);
        res.emplace_back(r, size_t(M) * W * 2);
        d_alpha.emplace_back(r, size_t(W) * 4);
        d_beta.emplace_back(r, size_t(W) * 4);
        d_next.emplace_back(r, size_t(W) * 2);
        x_out.emplace_back(r, size_t(M) * W * 2);
        h_out.emplace_back(r, size_t(M) * W * 2);
        d_alpha.back().upload(alpha.data(), W * 4);
        d_beta.back().upload(beta.data(), W * 4);
        d_next.back().upload(next_w.data(), W * 2);
    }
    auto norm_args = [&](int r, bf16_t* xo, bf16_t* ho) {
        NormRow::Args a{};
        a.num_inputs      = Wd;
        a.in_stride       = W;
        a.first_w         = nullptr;
        a.first_eps       = 1e-8f;
        a.gate_alpha      = d_alpha[r].as<float>();
        a.gate_beta       = d_beta[r].as<float>();
        a.residual        = res[r].as<bf16_t>();
        a.residual_stride = W;
        a.x_out           = xo;
        a.x_stride        = W;
        a.next_w          = d_next[r].as<bf16_t>();
        a.next_eps        = 1e-5f;
        a.h_out           = ho;
        a.h_stride        = W;
        a.M               = M;
        return a;
    };
    Launcher<MkNorm> L(o, streams, [&](int r, const uint32_t* state) {
        const ArNorm::Args a{ex.view(r, state, 1, 0, o.spin),
                             in[r].as<bf16_t>(),
                             W,
                             M,
                             W,
                             norm_args(r, x_out[r].as<bf16_t>(), h_out[r].as<bf16_t>())};
        return MkNorm::ArgsTuple{a};
    });

    // Reference: the row phase alone on device 0, inputs = every rank's partial via peer pointers.
    DeviceBuffer ref_x(0, size_t(M) * W * 2), ref_h(0, size_t(M) * W * 2);
    auto ref_args = norm_args(0, ref_x.as<bf16_t>(), ref_h.as<bf16_t>());
    for(int r = 0; r < Wd; ++r)
        ref_args.in[r] = in[r].as<bf16_t>();
    const MkNormRef::ArgsTuple ref_tuple{ref_args};
    DeviceBuffer ref_counters(0, size_t(MkNormRef::NumCounters(ref_tuple)) * 4);
    ref_counters.fill(0);
    const auto ref_kargs = MkNormRef::MakeKargs(ref_tuple, ref_counters.as<uint32_t>());
    if(!MkNormRef::IsSupportedArgument(ref_kargs))
        throw std::runtime_error("reference arguments not supported");

    int failures = 0;
    for(int l = 0; l < o.launches; ++l)
    {
        const auto residual = random_bf16(size_t(M) * W, 5000u * l + 99);
        for(int r = 0; r < Wd; ++r)
        {
            // No -0 here: the reference sums the raw partials, the exchange sends -0 as +0, and a
            // row of -0 sums only would then differ in sign (covered by -test=ar instead).
            auto x = random_bf16(size_t(M) * W, 5000u * l + r);
            for(auto& v : x)
                v = canon(v);
            in[r].upload(x.data(), x.size() * 2);
            res[r].upload(residual.data(), residual.size() * 2);
            x_out[r].fill(0xFF);
            h_out[r].fill(0xFF);
        }
        HIP_CHECK(hipSetDevice(0));
        ck_tile::launch_kernel(
            ck_tile::stream_config{streams[0]},
            ck_tile::make_kernel<1>(MkNormRef{}, dim3(M), MkNormRef::BlockSize(), 0, ref_kargs));
        HIP_CHECK(hipStreamSynchronize(streams[0]));
        const auto rx = ref_x.download<uint16_t>();
        const auto rh = ref_h.download<uint16_t>();
        L.launch_all();
        L.sync_all();
        int bad = 0;
        for(int r = 0; r < Wd; ++r)
        {
            bad += compare_u16("x_out", r, x_out[r].download<uint16_t>(), rx);
            bad += compare_u16("h_out", r, h_out[r].download<uint16_t>(), rh);
        }
        if(bad != 0)
        {
            ++failures;
            std::printf("  launch %d: %d mismatches\n", l, bad);
        }
    }
    failures += L.check_epochs(static_cast<uint32_t>(o.launches));
    if(o.repeat > 0)
        std::printf("  %.2f us per launch (all-reduce + residual norm of %d x %d bf16, %d ranks)\n",
                    L.time(o.repeat),
                    M,
                    W,
                    Wd);
    return failures;
}

int run_barrier(const Options& o, const std::vector<hipStream_t>& streams)
{
    std::vector<DeviceBuffer> flags;
    for(int r = 0; r < o.world; ++r)
    {
        flags.emplace_back(r, kMaxWorld * sizeof(uint32_t), true);
        flags.back().fill(0);
    }
    Launcher<MkBarrier> L(o, streams, [&](int r, const uint32_t* state) {
        Barrier::Args a{};
        for(int p = 0; p < o.world; ++p)
            a.flags[p] = flags[p].as<uint32_t>();
        a.rank            = r;
        a.world           = o.world;
        a.launch_state    = state;
        a.uses_per_launch = 1;
        a.use_index       = 0;
        a.spin_limit      = o.spin;
        return MkBarrier::ArgsTuple{a};
    });
    for(int l = 0; l < o.launches; ++l)
        L.launch_all();
    L.sync_all();
    int failures = L.check_epochs(static_cast<uint32_t>(o.launches));
    for(int r = 0; r < o.world; ++r)
    {
        const auto f = flags[r].download<uint32_t>();
        for(int q = 0; q < o.world; ++q)
        {
            if(f[q] != static_cast<uint32_t>(o.launches))
            {
                std::printf("  rank %d flag[%d] = %u, expected %d\n", r, q, f[q], o.launches);
                ++failures;
            }
        }
    }
    if(o.repeat > 0)
        std::printf("  %.2f us per launch (rank barrier, %d ranks)\n", L.time(o.repeat), o.world);
    return failures;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("test", "all", "ar | split | ag | norm | barrier | all")
        .insert("ngpu", "0", "ranks = visible GPUs used; 0 = all (max 8); 1 = self loop")
        .insert("m", "16", "rows: 16 (one DFlash2 verify window) .. 64")
        .insert("width", "8192", "all-reduce row width (bf16), a multiple of 8")
        .insert("shard", "1024", "all-gather bf16 columns per rank (bytes test: shard / 2 bytes)")
        .insert("launches", "5", "validated launches, no host reset in between")
        .insert("repeat", "100", "timed back-to-back launches (0 = no timing)")
        .insert("grid", "0", "workgroups per device; 0 = min(work, resident)")
        .insert("spin_limit", "16777216", "polls before a wait traps (0 = unbounded)");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        int count = 0;
        HIP_CHECK(hipGetDeviceCount(&count));
        int world = arg.get_int("ngpu");
        world     = world > 0 ? std::min(world, count) : std::min(count, int{kMaxWorld});
        if(world < 1)
        {
            std::printf("no GPU\n");
            return 2;
        }
        for(int i = 0; i < world; ++i)
        {
            HIP_CHECK(hipSetDevice(i));
            for(int j = 0; j < world; ++j)
            {
                if(i == j)
                    continue;
                int can = 0;
                HIP_CHECK(hipDeviceCanAccessPeer(&can, i, j));
                if(!can)
                {
                    std::printf("GPU %d cannot access GPU %d\n", i, j);
                    return 2;
                }
                const hipError_t e = hipDeviceEnablePeerAccess(j, 0);
                if(e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                    HIP_CHECK(e);
            }
        }
        std::vector<hipStream_t> streams(world);
        for(int i = 0; i < world; ++i)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));
        }

        const Options o{world,
                        arg.get_int("m"),
                        arg.get_int("width"),
                        arg.get_int("shard"),
                        arg.get_int("launches"),
                        arg.get_int("repeat"),
                        arg.get_int("grid"),
                        arg.get_uint32("spin_limit")};
        const std::string test = arg.get_str("test");
        std::printf("%d rank(s), M %d\n", world, o.m);
        int failures = 0;
        auto run     = [&](const char* name, auto&& f) {
            if(test != "all" && test != name)
                return;
            std::printf("[%s]\n", name);
            const int f_ = f(o, streams);
            std::printf("  %s\n", f_ == 0 ? "PASS" : "FAIL");
            failures += f_;
        };
        run("ar", run_ar);
        run("split", run_split);
        run("ag", run_ag);
        run("norm", run_norm);
        run("barrier", run_barrier);
        for(int i = 0; i < world; ++i)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamDestroy(streams[i]));
        }
        std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
