// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Weight-streaming bf16 GEMM at decode M (ck_tile::SmallMGemmPhase) in a persistent
// ck_tile::Megakernel launch, gfx1250:
//   single (default):  Y = X W^T (+ residual)          X [M, K], W [N, K], Y [M, N]
//   -chain=1:          Y = X Wp^T, Z = Y Wu^T           the target model's pooler -> unpooler;
//                      stage 1 waits (after_rows) only for the Y columns its K split reads
// Shapes per TP8 rank (notes): qkvg N 2304 K 8192; o_proj N 8192 K 1024 (or N 1024 K 8192);
// pooler N 4096 K 8192 / unpooler N 8192 K 4096 (full, unsharded; 8.4 MB per rank = N 512).
//
// Checks: every launch reproduces launch 0 bit for bit (outputs NaN-filled before each launch,
// the counter buffer reused without a reset); -check_rows rows against a double reference
// (|got - ref| <= |ref| 2^-7 + sum|x w| 2^-14; the chain's Z uses the GPU's Y); with -chain,
// prefetch on and off give identical bits. Then timing: -iters launches, weights rotated over
// copies totalling -rotate_mb so they come from HBM, -rounds interleaved rounds (prefetch on/off
// alternate), median per variant, GB/s on weight bytes.
// -selftest=1 runs only the host emulation of the phase's data path (no GPU).
// Exit code: 0 pass, 1 wrong result, 2 unsupported arguments, 3 error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/megakernel.hpp"

#include "small_m_host.hpp"

namespace {

using index_t = ck_tile::index_t;
using ck_tile::small_m::EpiPassThrough;
using ck_tile::small_m::EpiResidualAdd;

// kMaxM 16 covers M 1..16 (8 and 16 are the batch-1 verify windows), kMaxM 64 up to 4 requests.
// Ring: 8 waves x kStages x (kNTiles + kMTiles) x 2 KB = 192 KB (16) / 288 KB (64).
template <bool kPrefetch, typename Epi = EpiPassThrough>
using Phase16 = ck_tile::SmallMGemmPhase<16, 2, 4, ck_tile::bf16_t, Epi, kPrefetch>;
template <bool kPrefetch, typename Epi = EpiPassThrough>
using Phase64 = ck_tile::SmallMGemmPhase<64, 2, 3, ck_tile::bf16_t, Epi, kPrefetch>;

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
    explicit DeviceBuffer(size_t bytes) { HIP_CHECK(hipMalloc(&p, std::max<size_t>(bytes, 16))); }
    ~DeviceBuffer() { (void)hipFree(p); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* as() const
    {
        return static_cast<T*>(p);
    }
};

void upload(const DeviceBuffer& d, const std::vector<uint16_t>& h)
{
    HIP_CHECK(hipMemcpy(d.p, h.data(), h.size() * 2, hipMemcpyHostToDevice));
}

std::vector<uint16_t> download(const DeviceBuffer& d, size_t n)
{
    std::vector<uint16_t> h(n);
    HIP_CHECK(hipMemcpy(h.data(), d.p, n * 2, hipMemcpyDeviceToHost));
    return h;
}

std::vector<index_t> pick_rows(index_t M, int n)
{
    std::vector<index_t> rows;
    for(int i = 0; i < n && i < M; ++i)
        rows.push_back(n >= M ? i
                              : static_cast<index_t>(int64_t(i) * (M - 1) / std::max(1, n - 1)));
    return rows;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

/// One GEMM's device state: weights (and rotation copies), split-K scratch, phase arguments.
template <typename Phase>
struct Gemm
{
    index_t M, N, K, splits;
    std::unique_ptr<DeviceBuffer> partial, tickets;
    std::vector<std::unique_ptr<DeviceBuffer>> w; // w[0] = the checked weights

    Gemm(index_t m, index_t n, index_t k, index_t s) : M(m), N(n), K(k), splits(s) {}

    size_t WeightBytes() const { return size_t(N) * K * 2; }

    void Alloc(const std::vector<uint16_t>& host_w, size_t rotate_bytes)
    {
        const size_t copies =
            std::max<size_t>(1, (rotate_bytes + WeightBytes() - 1) / WeightBytes());
        for(size_t c = 0; c < copies; ++c)
        {
            w.push_back(std::make_unique<DeviceBuffer>(WeightBytes()));
            upload(*w.back(), host_w);
        }
        const size_t n_partial = size_t(splits) * M * N;
        const size_t n_tickets = size_t(N / Phase::kNRows);
        partial = std::make_unique<DeviceBuffer>(splits > 1 ? n_partial * 4 : 16);
        tickets = std::make_unique<DeviceBuffer>(n_tickets * 4);
        HIP_CHECK(hipMemset(tickets->p, 0, n_tickets * 4));
    }

    typename Phase::Args Args(const void* x,
                              void* y,
                              size_t copy,
                              index_t l2pf,
                              typename Phase::EpiArgs epi = {}) const
    {
        typename Phase::Args a{};
        a.x                   = static_cast<const ck_tile::bf16_t*>(x);
        a.w                   = w[copy % w.size()]->template as<const ck_tile::bf16_t>();
        a.y                   = static_cast<typename Phase::OutType*>(y);
        a.partial             = partial->template as<float>();
        a.tickets             = tickets->template as<uint32_t>();
        a.M                   = M;
        a.N                   = N;
        a.K                   = K;
        a.stride_x            = K;
        a.stride_w            = K;
        a.ld_y                = N;
        a.k_splits            = splits;
        a.l2_prefetch_windows = l2pf;
        a.epi                 = epi;
        return a;
    }
};

template <typename Phase>
index_t choose_splits(const ck_tile::ArgParser& arg, index_t N, index_t K, index_t grid)
{
    const index_t s = arg.get_int("splitk");
    if(s > 0)
        return s;
    return small_m_host::auto_splits(
        N / Phase::kNRows, K / ck_tile::small_m::kWindowK, Phase::kWaves, Phase::kStages, grid);
}

template <typename MK>
index_t resolve_grid(const ck_tile::ArgParser& arg, const ck_tile::stream_config& s)
{
    const index_t resident = MK::MaxResidentGridSize(s);
    const index_t g        = arg.get_int("grid");
    return g > 0 ? std::min(g, resident) : resident;
}

/// Launches kargs[i % kargs.size()] @p iters times; returns microseconds per launch.
template <typename MK>
double time_launches(const std::vector<typename MK::Kargs>& kargs, index_t grid, int iters)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    const ck_tile::stream_config s{nullptr, false};
    HIP_CHECK(hipEventRecord(start, nullptr));
    for(int i = 0; i < iters; ++i)
        ck_tile::launch_kernel(
            s,
            ck_tile::make_kernel<1>(MK{}, dim3(grid), MK::BlockSize(), 0, kargs[i % kargs.size()]));
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return 1e3 * ms / iters;
}

void print_rate(const char* what, double us, size_t weight_bytes, double hbm_tbps)
{
    const double gbs = weight_bytes / (us * 1e3);
    std::printf(
        "%s: %.2f us/launch, weights %.1f MB -> %.0f GB/s", what, us, weight_bytes / 1e6, gbs);
    if(hbm_tbps > 0)
        std::printf(" (%.0f%% of %.1f TB/s; roofline %.2f us)",
                    100 * gbs / (hbm_tbps * 1e3),
                    hbm_tbps,
                    weight_bytes / (hbm_tbps * 1e6));
    std::printf("\n");
}

// ------------------------------------------------------------------------------------------------
// Single GEMM
template <typename Phase>
int run_single(const ck_tile::ArgParser& arg)
{
    using MK      = ck_tile::Megakernel<ck_tile::megakernel_stage<Phase>>;
    const index_t M = arg.get_int("m"), N = arg.get_int("n"), K = arg.get_int("k");
    const bool residual = arg.get_int("residual") != 0;
    const ck_tile::stream_config s{nullptr, false};
    const index_t grid = resolve_grid<MK>(arg, s);

    Gemm<Phase> g(M, N, K, choose_splits<Phase>(arg, N, K, grid));
    const auto x   = small_m_host::random_bf16(size_t(M) * K, 1.f, 1);
    const auto w   = small_m_host::random_bf16(size_t(N) * K, 1.f / std::sqrt(float(K)), 2);
    const auto res = small_m_host::random_bf16(size_t(M) * N, 1.f, 3);
    g.Alloc(w, size_t(arg.get_int("rotate_mb")) << 20);
    DeviceBuffer d_x(x.size() * 2), d_y(size_t(M) * N * 2), d_res(res.size() * 2);
    upload(d_x, x);
    upload(d_res, res);

    typename Phase::EpiArgs epi{};
    if constexpr(std::is_same_v<typename Phase::Epilogue, EpiResidualAdd>)
        epi = {d_res.as<const ck_tile::bf16_t>(), N, 1.f};
    const index_t l2pf = arg.get_int("l2pf");

    std::vector<typename MK::Kargs> kargs;
    const typename MK::ArgsTuple args0{g.Args(d_x.p, d_y.p, 0, l2pf, epi)};
    const index_t n_counters = MK::NumCounters(args0);
    DeviceBuffer d_counters(size_t(n_counters) * 4);
    HIP_CHECK(hipMemset(d_counters.p, 0, size_t(n_counters) * 4));
    for(size_t c = 0; c < g.w.size(); ++c)
        kargs.push_back(MK::MakeKargs(typename MK::ArgsTuple{g.Args(d_x.p, d_y.p, c, l2pf, epi)},
                                      d_counters.as<uint32_t>()));
    if(!MK::IsSupportedArgument(kargs[0]))
    {
        std::printf("arguments not supported\n");
        return 2;
    }
    std::printf("single: M %d N %d K %d (kMaxM %d), splits %d, %d items, grid %d, LDS %d B, "
                "residual %d, l2pf %d, %zu weight copies\n",
                M,
                N,
                K,
                Phase::kMaxM,
                g.splits,
                Phase::NumWork(args0.at(ck_tile::number<0>{})),
                grid,
                static_cast<int>(MK::kSmemSize),
                residual ? 1 : 0,
                l2pf,
                g.w.size());

    const int launches = arg.get_int("launches");
    std::vector<uint16_t> first;
    int failures = 0;
    for(int l = 0; l < launches; ++l)
    {
        HIP_CHECK(hipMemset(d_y.p, 0xFF, size_t(M) * N * 2));
        ck_tile::launch_kernel(
            s, ck_tile::make_kernel<1>(MK{}, dim3(grid), MK::BlockSize(), 0, kargs[0]));
        HIP_CHECK(hipDeviceSynchronize());
        auto y = download(d_y, size_t(M) * N);
        if(l == 0)
            first = std::move(y);
        else if(std::memcmp(y.data(), first.data(), first.size() * 2) != 0)
        {
            ++failures;
            std::printf("launch %d differs from launch 0\n", l);
        }
    }
    uint32_t epoch = 0;
    HIP_CHECK(hipMemcpy(&epoch, d_counters.p, 4, hipMemcpyDeviceToHost));
    if(epoch != static_cast<uint32_t>(launches))
    {
        ++failures;
        std::printf("device epoch %u after %d launches\n", epoch, launches);
    }
    small_m_host::HostEpi hepi;
    if(residual)
        hepi = {res.data(), N};
    const auto rows = pick_rows(M, arg.get_int("check_rows"));
    const int bad   = small_m_host::check_rows("Y", first, x, w, N, K, rows, hepi);
    const bool pass = failures == 0 && bad == 0;
    std::printf("%s: %d/%d launches identical, %zu reference rows, %d bad\n",
                pass ? "PASS" : "FAIL",
                launches - failures,
                launches,
                rows.size(),
                bad);

    const int iters = arg.get_int("iters"), rounds = arg.get_int("rounds");
    if(iters > 0)
    {
        time_launches<MK>(kargs, grid, std::min(iters, 5)); // warm-up
        std::vector<double> t;
        for(int r = 0; r < rounds; ++r)
            t.push_back(time_launches<MK>(kargs, grid, iters));
        print_rate("single", median(t), g.WeightBytes(), arg.get_float("hbm_tbps"));
    }
    return pass ? 0 : 1;
}

// ------------------------------------------------------------------------------------------------
// Chain: pooler -> unpooler, prefetch on and off.
template <typename PhaseOn, typename PhaseOff>
int run_chain(const ck_tile::ArgParser& arg)
{
    using Dep   = ck_tile::after_rows<0>;
    using MKOn  = ck_tile::Megakernel<ck_tile::megakernel_stage<PhaseOn>,
                                      ck_tile::megakernel_stage<PhaseOn, Dep>>;
    using MKOff = ck_tile::Megakernel<ck_tile::megakernel_stage<PhaseOff>,
                                      ck_tile::megakernel_stage<PhaseOff, Dep>>;
    static_assert(PhaseOn::kNRows == PhaseOff::kNRows && PhaseOn::kWaves == PhaseOff::kWaves);

    const index_t M = arg.get_int("m"), H = arg.get_int("hidden"), P = arg.get_int("pooled");
    const int mode  = arg.get_int("prefetch"); // 0 off, 1 on, 2 both
    const ck_tile::stream_config s{nullptr, false};
    const index_t grid = std::min(resolve_grid<MKOn>(arg, s), resolve_grid<MKOff>(arg, s));

    // Splits are chosen for the pooler and unpooler separately (same rule).
    Gemm<PhaseOn> gp(M, P, H, choose_splits<PhaseOn>(arg, P, H, grid));
    Gemm<PhaseOn> gu(M, H, P, choose_splits<PhaseOn>(arg, H, P, grid));
    const auto x  = small_m_host::random_bf16(size_t(M) * H, 1.f, 1);
    const auto wp = small_m_host::random_bf16(size_t(P) * H, 1.f / std::sqrt(float(H)), 2);
    const auto wu = small_m_host::random_bf16(size_t(H) * P, 1.f / std::sqrt(float(P)), 3);
    const size_t rotate = size_t(arg.get_int("rotate_mb")) << 20;
    gp.Alloc(wp, rotate / 2);
    gu.Alloc(wu, rotate / 2);
    DeviceBuffer d_x(x.size() * 2), d_y(size_t(M) * P * 2), d_z(size_t(M) * H * 2);
    upload(d_x, x);
    const index_t l2pf = arg.get_int("l2pf");

    // The two variants share the scratch (tickets end every launch at zero) but not counters.
    const auto args_of = [&](size_t c) {
        // Phase Args types differ only in the template; the layouts are identical.
        const auto ap = gp.Args(d_x.p, d_y.p, c, l2pf);
        const auto au = gu.Args(d_y.p, d_z.p, c, l2pf);
        return std::make_pair(ap, au);
    };
    const auto convert = [](const typename PhaseOn::Args& a) {
        typename PhaseOff::Args b;
        static_assert(sizeof(a) == sizeof(b));
        std::memcpy(&b, &a, sizeof(a));
        return b;
    };
    const size_t copies = std::min(gp.w.size(), gu.w.size());

    const auto a0 = args_of(0);
    const typename MKOn::ArgsTuple t_on{a0.first, a0.second};
    const index_t n_counters = MKOn::NumCounters(t_on);
    DeviceBuffer c_on(size_t(n_counters) * 4), c_off(size_t(n_counters) * 4);
    HIP_CHECK(hipMemset(c_on.p, 0, size_t(n_counters) * 4));
    HIP_CHECK(hipMemset(c_off.p, 0, size_t(n_counters) * 4));
    std::vector<typename MKOn::Kargs> k_on;
    std::vector<typename MKOff::Kargs> k_off;
    for(size_t c = 0; c < copies; ++c)
    {
        const auto a = args_of(c);
        k_on.push_back(MKOn::MakeKargs(typename MKOn::ArgsTuple{a.first, a.second},
                                       c_on.as<uint32_t>()));
        k_off.push_back(MKOff::MakeKargs(
            typename MKOff::ArgsTuple{convert(a.first), convert(a.second)}, c_off.as<uint32_t>()));
    }
    if(!MKOn::IsSupportedArgument(k_on[0]) || !MKOff::IsSupportedArgument(k_off[0]))
    {
        std::printf("arguments not supported\n");
        return 2;
    }
    std::printf("chain: M %d H %d P %d (kMaxM %d), splits pooler %d unpooler %d, items %d + %d, "
                "grid %d, LDS %d B, l2pf %d, %zu weight copies\n",
                M,
                H,
                P,
                PhaseOn::kMaxM,
                gp.splits,
                gu.splits,
                PhaseOn::NumWork(a0.first),
                PhaseOn::NumWork(a0.second),
                grid,
                static_cast<int>(MKOn::kSmemSize),
                l2pf,
                copies);

    const int launches = arg.get_int("launches");
    std::vector<uint16_t> y0, z0;
    int failures = 0;
    const auto check_launch = [&](const char* tag, int l, auto launch) {
        HIP_CHECK(hipMemset(d_y.p, 0xFF, size_t(M) * P * 2));
        HIP_CHECK(hipMemset(d_z.p, 0xFF, size_t(M) * H * 2));
        launch();
        HIP_CHECK(hipDeviceSynchronize());
        auto y = download(d_y, size_t(M) * P);
        auto z = download(d_z, size_t(M) * H);
        if(y0.empty())
        {
            y0 = std::move(y);
            z0 = std::move(z);
            return;
        }
        const bool same = std::memcmp(y.data(), y0.data(), y0.size() * 2) == 0 &&
                          std::memcmp(z.data(), z0.data(), z0.size() * 2) == 0;
        if(!same)
        {
            ++failures;
            std::printf("%s launch %d differs from the first launch\n", tag, l);
        }
    };
    for(int l = 0; l < launches; ++l)
    {
        if(mode != 0)
            check_launch("prefetch on", l, [&] {
                ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<1>(MKOn{}, dim3(grid), MKOn::BlockSize(), 0, k_on[0]));
            });
        if(mode != 1)
            check_launch("prefetch off", l, [&] {
                ck_tile::launch_kernel(
                    s,
                    ck_tile::make_kernel<1>(MKOff{}, dim3(grid), MKOff::BlockSize(), 0, k_off[0]));
            });
    }
    const auto rows  = pick_rows(M, arg.get_int("check_rows"));
    const int bad_y  = small_m_host::check_rows("Y", y0, x, wp, P, H, rows);
    const int bad_z  = small_m_host::check_rows("Z", z0, y0, wu, H, P, rows);
    const bool pass  = failures == 0 && bad_y == 0 && bad_z == 0;
    std::printf("%s: launches identical (failures %d), reference rows %zu: Y bad %d, Z bad %d\n",
                pass ? "PASS" : "FAIL",
                failures,
                rows.size(),
                bad_y,
                bad_z);

    const int iters = arg.get_int("iters"), rounds = arg.get_int("rounds");
    if(iters > 0)
    {
        if(mode != 0)
            time_launches<MKOn>(k_on, grid, std::min(iters, 5));
        if(mode != 1)
            time_launches<MKOff>(k_off, grid, std::min(iters, 5));
        std::vector<double> us_on, us_off;
        for(int r = 0; r < rounds; ++r)
        {
            if(mode != 0)
                us_on.push_back(time_launches<MKOn>(k_on, grid, iters));
            if(mode != 1)
                us_off.push_back(time_launches<MKOff>(k_off, grid, iters));
        }
        const size_t bytes = gp.WeightBytes() + gu.WeightBytes();
        const double tbps  = arg.get_float("hbm_tbps");
        if(mode != 0)
            print_rate("chain prefetch on ", median(us_on), bytes, tbps);
        if(mode != 1)
            print_rate("chain prefetch off", median(us_off), bytes, tbps);
    }
    return pass ? 0 : 1;
}

/// Host only: splits, items and the HBM roofline for the target model's dense GEMMs per TP8 rank.
template <typename Phase>
int plan(const ck_tile::ArgParser& arg)
{
    const index_t grid = arg.get_int("grid") > 0 ? arg.get_int("grid") : 96;
    const double tbps  = arg.get_float("hbm_tbps");
    struct Shape
    {
        const char* name;
        index_t N, K;
    };
    const Shape shapes[] = {{"qkvg", 2304, 8192},
                            {"o_proj (AR, K-shard)", 8192, 1024},
                            {"o_proj (AG, N-shard)", 1024, 8192},
                            {"pooler full", 4096, 8192},
                            {"pooler per rank", 512, 8192},
                            {"unpooler full", 8192, 4096},
                            {"unpooler per rank", 1024, 4096}};
    std::printf(
        "plan: M %d (kMaxM %d), grid %d, %.1f TB/s\n", arg.get_int("m"), Phase::kMaxM, grid, tbps);
    for(const auto& sh : shapes)
    {
        const index_t nb = sh.N / Phase::kNRows, windows = sh.K / ck_tile::small_m::kWindowK;
        const index_t S =
            small_m_host::auto_splits(nb, windows, Phase::kWaves, Phase::kStages, grid);
        const double mb  = double(sh.N) * sh.K * 2 / 1e6;
        std::printf("  %-22s N %5d K %5d: %6.2f MB, splits %2d, items %4d (%.2f rounds), "
                    "windows/wave/item %.1f, roofline %.2f us\n",
                    sh.name,
                    sh.N,
                    sh.K,
                    mb,
                    S,
                    nb * S,
                    double(nb * S) / grid,
                    double(windows) / S / Phase::kWaves,
                    mb / tbps); // MB / (TB/s) = us
    }
    return 0;
}

int run(const ck_tile::ArgParser& arg)
{
    const index_t M = arg.get_int("m");
    const bool chain = arg.get_int("chain") != 0;
    const bool res   = arg.get_int("residual") != 0;
    if(M < 1 || M > 64)
    {
        std::printf("-m must be in [1, 64]\n");
        return 2;
    }
    if(arg.get_int("plan"))
        return M <= 16 ? plan<Phase16<true>>(arg) : plan<Phase64<true>>(arg);
    if(chain)
        return M <= 16 ? run_chain<Phase16<true>, Phase16<false>>(arg)
                       : run_chain<Phase64<true>, Phase64<false>>(arg);
    if(M <= 16)
        return res ? run_single<Phase16<true, EpiResidualAdd>>(arg)
                   : run_single<Phase16<true>>(arg);
    return res ? run_single<Phase64<true, EpiResidualAdd>>(arg) : run_single<Phase64<true>>(arg);
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("m", "16", "rows of X (tokens), 1..64: 8 / 16 per verify window, 64 for four")
        .insert("n", "4096", "N (weight rows), multiple of 32")
        .insert("k", "8192", "K, multiple of 64")
        .insert("splitk", "0", "K splits; 0 = auto for the grid")
        .insert("residual", "0", "single GEMM: 1 = fused residual add epilogue")
        .insert("chain", "0", "1 = pooler -> unpooler (-hidden, -pooled) as two dependent stages")
        .insert("hidden", "8192", "chain: H (pooler K, unpooler N)")
        .insert("pooled", "4096", "chain: P (pooler N, unpooler K)")
        .insert("prefetch", "2", "chain: 0 off, 1 on, 2 both (interleaved timing)")
        .insert("l2pf", "0", "windows per wave L2-prefetched after the LDS ring (Prefetch)")
        .insert("grid", "0", "workgroups; 0 = all resident")
        .insert("launches", "5", "checked launches sharing one counter buffer")
        .insert("check_rows", "8", "rows checked against the double reference")
        .insert("iters", "50", "timed launches per round; 0 = no timing")
        .insert("rounds", "7", "timing rounds (median)")
        .insert("rotate_mb", "1024", "weight copies rotated in timing, total MB (defeats L2/MALL)")
        .insert("hbm_tbps", "19.6", "HBM bandwidth for the roofline column; 0 = omit")
        .insert("plan", "0", "1: print splits and roofline for the target-model shapes (host only)")
        .insert("selftest", "0", "1: only the host emulation of the data path (no GPU)")
        .insert("seed", "7", "selftest seed");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        if(arg.get_int("selftest"))
        {
            const uint64_t seed = static_cast<uint64_t>(arg.get_int("seed"));
            const bool ok       = small_m_host::selftest<Phase16<true>>("kMaxM16", seed) &&
                            small_m_host::selftest<Phase64<true>>("kMaxM64", seed);
            return ok ? 0 : 1;
        }
        return run(arg);
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
