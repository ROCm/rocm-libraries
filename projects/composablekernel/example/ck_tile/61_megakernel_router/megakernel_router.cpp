// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The target model's MoE router in one ck_tile::Megakernel launch (gfx1250):
//   stage 0 RouterGemmPhase:     logits = (h fp32) @ W_r^T * 0.17677669529663687, split-K
//   stage 1 RouterFinalizePhase: sigmoid, + e_score_correction_bias, top-8, normalise, route table
// Checks against a double reference (router_reference.hpp), including a forced tie at the top-8
// cut (the smaller expert must win), and that repeated launches with poisoned outputs and one
// counter / ticket buffer give bitwise-identical tables.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/megakernel.hpp"
#include "ck_tile/ops/megakernel/router/router.hpp"

#include "router_reference.hpp"

namespace {

using index_t = ck_tile::index_t;

constexpr index_t kBlockSize   = 256;
constexpr index_t kNumExperts  = 256;
constexpr index_t kTopK        = 8;
constexpr float kOutputMult    = 0.17677669529663687f;
constexpr float kScoreEps      = 1e-15f;
constexpr float kRoutedScaling = 1.0f;

template <index_t kMaxM>
struct Router
{
    using Gemm     = ck_tile::RouterGemmPhase<kBlockSize, kMaxM>;
    using Finalize = ck_tile::RouterFinalizePhase<kBlockSize, kMaxM, kNumExperts, kTopK>;
    using MK       = ck_tile::Megakernel<ck_tile::megakernel_stage<Gemm>,
                                         ck_tile::megakernel_stage<Finalize, ck_tile::after_all<0>>>;
};

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

uint16_t bf16_rne(float f)
{
    const uint32_t u = ck_tile::bit_cast<uint32_t>(f);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

struct DeviceBuffer
{
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { HIP_CHECK(hipMalloc(&p, bytes > 0 ? bytes : 4)); }
    ~DeviceBuffer() { (void)hipFree(p); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* as() const
    {
        return static_cast<T*>(p);
    }
};

template <typename T>
std::vector<T> to_host(const DeviceBuffer& b, size_t n)
{
    std::vector<T> v(n);
    HIP_CHECK(hipMemcpy(v.data(), b.p, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

/// Everything one launch produces, copied back.
struct Result
{
    std::vector<float> logits;
    std::vector<int> ids;
    std::vector<float> weights;
    router_ref::RouteTable table;

    bool operator==(const Result& o) const
    {
        return std::memcmp(logits.data(), o.logits.data(), logits.size() * 4) == 0 &&
               ids == o.ids &&
               std::memcmp(weights.data(), o.weights.data(), weights.size() * 4) == 0 &&
               table.count == o.table.count && table.routes == o.table.routes &&
               table.active == o.table.active && table.num_active == o.table.num_active;
    }
};

template <index_t kMaxM>
int run(const ck_tile::ArgParser& arg)
{
    using R        = Router<kMaxM>;
    using Gemm     = typename R::Gemm;
    using Finalize = typename R::Finalize;
    using MK       = typename R::MK;

    const index_t M    = arg.get_int("m");
    const index_t K    = arg.get_int("hidden");
    const index_t E    = kNumExperts;
    const index_t maxm = kMaxM;
    const int launches = arg.get_int("launches");
    const ck_tile::stream_config s{nullptr, false};

    // Data: h ~ N(0, 1) in bf16 (the normed hidden state), W_r ~ N(0, 4 / sqrt(K)).
    std::mt19937_64 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> h(size_t(M) * K);
    for(auto& x : h)
        x = bf16_rne(nd(rng));
    std::vector<float> w(size_t(E) * K);
    for(auto& x : w)
        x = nd(rng) * 4.f / std::sqrt(float(K));
    std::vector<float> bias(E);
    for(auto& x : bias)
        x = 0.02f * nd(rng);
    // Tie case: experts 5, 50, 200 share one weight row and bias, 7 experts are far ahead, so
    // every row's 8th slot is a three-way exact tie that expert 5 must win.
    std::vector<float> bias_tie(E, 0.f);
    for(int e : {3, 9, 100, 101, 150, 254, 255})
        bias_tie[e] = 2.f;
    for(int e : {5, 50, 200})
    {
        bias_tie[e] = 1.f;
        std::copy(w.begin() + size_t(5) * K, w.begin() + size_t(6) * K, w.begin() + size_t(e) * K);
    }

    DeviceBuffer d_h(h.size() * 2), d_w(w.size() * 4), d_bias(E * 4), d_bias_tie(E * 4);
    DeviceBuffer d_logits(size_t(M) * E * 4);
    DeviceBuffer d_ids(size_t(M) * kTopK * 4), d_weights(size_t(M) * kTopK * 4);
    DeviceBuffer d_count(E * 4), d_routes(size_t(E) * maxm * 4), d_active(E * 4), d_num_active(4);
    HIP_CHECK(hipMemcpy(d_h.p, h.data(), h.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w.p, w.data(), w.size() * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias.p, bias.data(), E * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias_tie.p, bias_tie.data(), E * 4, hipMemcpyHostToDevice));

    typename Gemm::Args g{};
    g.h           = reinterpret_cast<const ck_tile::bf16_t*>(d_h.p);
    g.w           = d_w.as<float>();
    g.logits      = d_logits.as<float>();
    g.M           = M;
    g.K           = K;
    g.stride_h    = K;
    g.ld_logits   = E;
    g.num_experts = E;
    g.rank        = 0;
    g.world       = 1; // TODO: TP sharding over experts needs the item-2 all-gather of logits
    g.scale       = kOutputMult;

    // Resident grid (for the default split count) needs a valid Kargs type only.
    const index_t resident = MK::MaxResidentGridSize(s);
    g.k_splits             = arg.get_int("k_splits");
    if(g.k_splits <= 0)
        g.k_splits = std::max<index_t>(1, resident / Gemm::NumNTiles(g));
    g.k_splits = std::min<index_t>(g.k_splits, K / Gemm::kKUnit);

    DeviceBuffer d_partial(size_t(Gemm::PartialSize(g)) * 4);
    DeviceBuffer d_tickets(size_t(Gemm::NumTickets(g)) * 4);
    HIP_CHECK(hipMemset(d_tickets.p, 0, size_t(Gemm::NumTickets(g)) * 4));
    g.partial = d_partial.as<float>();
    g.tickets = d_tickets.as<uint32_t>();

    ck_tile::moe_route_table table{};
    table.topk_ids      = d_ids.as<int32_t>();
    table.topk_weights  = d_weights.as<float>();
    table.expert_count  = d_count.as<int32_t>();
    table.expert_routes = d_routes.as<int32_t>();
    table.active        = d_active.as<int32_t>();
    table.num_active    = d_num_active.as<int32_t>();
    table.M             = M;
    table.max_m         = maxm;
    table.top_k         = kTopK;
    table.num_experts   = E;

    typename Finalize::Args f{};
    f.logits                = d_logits.as<float>();
    f.ld_logits             = E;
    f.routed_scaling_factor = kRoutedScaling;
    f.eps                   = kScoreEps;
    f.table                 = table;

    const typename MK::ArgsTuple args{g, f};
    const index_t n_counters = MK::NumCounters(args);
    DeviceBuffer d_counters(size_t(n_counters) * 4);
    HIP_CHECK(hipMemset(d_counters.p, 0, size_t(n_counters) * 4));
    index_t grid = arg.get_int("grid");
    grid         = grid > 0 ? std::min(grid, resident) : std::min(Gemm::NumWork(g), resident);
    std::printf("M %d K %d E %d: %d GEMM items (%d n tiles x %d splits), grid %d (resident %d), "
                "LDS %d B\n",
                M,
                K,
                E,
                Gemm::NumWork(g),
                Gemm::NumNTiles(g),
                g.k_splits,
                grid,
                resident,
                static_cast<int>(MK::kSmemSize));

    const auto ref_logits = router_ref::router_logits(h, w, M, K, E, kOutputMult);

    int failures = 0;
    for(int tie_case = 0; tie_case < 2; ++tie_case)
    {
        f.bias           = (tie_case ? d_bias_tie : d_bias).template as<float>();
        const auto kargs = MK::MakeKargs(typename MK::ArgsTuple{g, f}, d_counters.as<uint32_t>());
        if(!MK::IsSupportedArgument(kargs))
        {
            std::printf("arguments not supported\n");
            return 2;
        }

        std::vector<Result> res;
        for(int l = 0; l < launches; ++l)
        {
            // Poison every output: the phases must rewrite all of it.
            HIP_CHECK(hipMemset(d_logits.p, 0xFF, size_t(M) * E * 4));
            HIP_CHECK(hipMemset(d_ids.p, 0xFF, size_t(M) * kTopK * 4));
            HIP_CHECK(hipMemset(d_weights.p, 0xFF, size_t(M) * kTopK * 4));
            HIP_CHECK(hipMemset(d_count.p, 0xFF, E * 4));
            HIP_CHECK(hipMemset(d_routes.p, 0xFF, size_t(E) * maxm * 4));
            HIP_CHECK(hipMemset(d_active.p, 0xFF, E * 4));
            HIP_CHECK(hipMemset(d_num_active.p, 0xFF, 4));
            ck_tile::launch_kernel(
                s, ck_tile::make_kernel<1>(MK{}, dim3(grid), MK::BlockSize(), 0, kargs));
            HIP_CHECK(hipDeviceSynchronize());

            Result r;
            r.logits           = to_host<float>(d_logits, size_t(M) * E);
            r.ids              = to_host<int>(d_ids, size_t(M) * kTopK);
            r.weights          = to_host<float>(d_weights, size_t(M) * kTopK);
            r.table.count      = to_host<int>(d_count, E);
            r.table.active     = to_host<int>(d_active, E);
            r.table.num_active = to_host<int>(d_num_active, 1)[0];
            const auto routes  = to_host<int>(d_routes, size_t(E) * maxm);
            r.table.routes.assign(size_t(E) * maxm, -1); // only the counted entries are defined
            for(index_t e = 0; e < E; ++e)
                for(int i = 0; i < r.table.count[e] && i < maxm; ++i)
                    r.table.routes[size_t(e) * maxm + i] = routes[size_t(e) * maxm + i];
            res.push_back(std::move(r));
        }

        int same = 0;
        for(const auto& r : res)
            same += r == res[0] ? 1 : 0;

        const Result& r = res[0];
        int bad_logits  = 0;
        for(size_t i = 0; i < r.logits.size(); ++i)
        {
            const double tol = ref_logits.mag[i] * 1e-5 + 1e-6;
            if(!(std::fabs(r.logits[i] - ref_logits.logits[i]) <= tol))
            {
                if(bad_logits < 5)
                    std::printf(
                        "  logits[%zu] = %g, ref %g\n", i, r.logits[i], ref_logits.logits[i]);
                ++bad_logits;
            }
        }
        const auto ref = router_ref::route_from_logits(
            ref_logits.logits, tie_case ? bias_tie : bias, M, E, kTopK, kRoutedScaling);
        int ties           = 0;
        const int bad_topk = router_ref::compare_topk(ref, r.ids, r.weights, 1e-5, 1e-5, &ties);
        int bad_tie        = 0;
        if(tie_case)
            for(index_t m = 0; m < M; ++m)
                bad_tie += r.ids[size_t(m) * kTopK + kTopK - 1] == 5 ? 0 : 1;
        bool table_ok = true;
        if(bad_topk == 0)
        {
            const auto t = router_ref::build_route_table(r.ids, M, kTopK, E, maxm);
            table_ok     = t.count == r.table.count && t.active == r.table.active &&
                       t.num_active == r.table.num_active;
            for(index_t e = 0; table_ok && e < E; ++e)
                for(int i = 0; i < t.count[e]; ++i)
                    table_ok = table_ok && t.routes[size_t(e) * maxm + i] ==
                                               r.table.routes[size_t(e) * maxm + i];
        }

        const bool pass =
            same == launches && bad_logits == 0 && bad_topk == 0 && bad_tie == 0 && table_ok;
        failures += pass ? 0 : 1;
        std::printf("%s [%s]: %d/%d launches identical, logits bad %d, top-k bad rows %d "
                    "(%d near-ties), tie rows lost by expert 5: %d, route table %s, %d active\n",
                    pass ? "PASS" : "FAIL",
                    tie_case ? "tie" : "random",
                    same,
                    launches,
                    bad_logits,
                    bad_topk,
                    ties,
                    bad_tie,
                    table_ok ? "ok" : "BAD",
                    r.table.num_active);
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("m", "16", "rows (tokens), 1..64: 16 for one DFlash2 verify window")
        .insert("hidden", "8192", "K: hidden size")
        .insert("k_splits", "0", "K splits of the router GEMM; 0 = resident grid / n tiles")
        .insert("grid", "0", "workgroups; 0 = min(GEMM items, resident)")
        .insert("launches", "3", "launches sharing one counter / ticket buffer");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        const index_t m = arg.get_int("m");
        if(m >= 1 && m <= 16)
            return run<16>(arg);
        if(m >= 1 && m <= 64)
            return run<64>(arg);
        std::printf("-m must be in 1..64\n");
        return 2;
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
