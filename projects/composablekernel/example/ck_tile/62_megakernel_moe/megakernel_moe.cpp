// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The target model's MoE experts at decode M in one persistent ck_tile::Megakernel launch, MXFP4
// weights with MX fp8 activations (A8W4), single GPU (world 1) or one TP rank's share:
//   stage 0  MoeGateUpPhase   z[route] = mxfp8(silu(p[row] gate[e]^T) * (p[row] up[e]^T))
//   stage 1  MoeDownPhase     y[route] = bf16(z[route] down[e]^T)       after_rows<0>: its expert
//   stage 2  MoeCombinePhase  out[row] = sum_slot w * InputScaledRMSNorm(y[route])   after_all<1>
// The route table (moe_route_table) and the quantised p are built on the host here; in the model
// they come from the router and the pooler + pre_expert_norm epilogue.
//
// Checks, against a double reference on dequantised data, each stage from the GPU's input to it:
//   z (one fp8 rounding), y (bf16), out (bf16 roundings of the norm as the HF model does).
// Every launch must reproduce launch 0 bit for bit (z, y, out are NaN-filled before each).
// -selftest runs only the host emulation of the GEMV's fragment assembly (no GPU needed).
// Exit code: 0 pass, 1 wrong result, 2 unsupported arguments, 3 error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/megakernel.hpp"
#include "ck_tile/ops/megakernel/moe/moe_expert_phases.hpp"

#include "moe_mx_host.hpp"

namespace {

using index_t = ck_tile::index_t;
using ck_tile::moe_mx::bf16_rne_bits;
using ck_tile::moe_mx::bf16_round;

constexpr index_t kDim = 4096; // P (moe_hidden) = I (expert intermediate) = 4096

using GateUp  = ck_tile::MoeGateUpPhase<ck_tile::fp8_t, kDim, 32, 4, 256>;
using Down    = ck_tile::MoeDownPhase<ck_tile::fp8_t, kDim, 32, 4, 256>;
using Combine = ck_tile::MoeCombinePhase<ck_tile::bf16_t, kDim, 8, 256>;
using MoeMK   = ck_tile::Megakernel<ck_tile::megakernel_stage<GateUp>,
                                    ck_tile::megakernel_stage<Down, ck_tile::after_rows<0>>,
                                    ck_tile::megakernel_stage<Combine, ck_tile::after_all<1>>>;

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
    DeviceBuffer(const std::vector<T>& v) : DeviceBuffer(v.size() * sizeof(T))
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

template <typename T>
std::vector<T> download(const DeviceBuffer& b, size_t n)
{
    std::vector<T> v(n);
    HIP_CHECK(hipMemcpy(v.data(), b.p, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

double bf16_to_double(uint16_t b)
{
    return ck_tile::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

/// Error bound of the GPU's MX GEMV against the exact sum. gfx1250's
/// v_wmma_scale_f32_16x16x128_f8f6f4 does not add its 128 scaled products t_k exactly: measured
/// on MI450 (random operands, wide dynamic range), the result comes out low (it truncates, so the
/// error does not cancel over K) by at most 0.44 * sum_k (t_k - floor(t_k / g) g), g = 2^(e - 13),
/// e the frexp exponent of the WMMA's largest |t_k| (up to 2^-9.5 of it); operands of narrow
/// dynamic range come out exact. Feed the products in K order; the WMMAs are the GEMV's: in each
/// 256-K window, WMMA j takes the 32-K blocks window_block(j, 0..3).
struct WmmaErr
{
    double bound = 0;
    double t[ck_tile::moe_mx::kWindowK];
    void add(index_t k, double v)
    {
        namespace mx        = ck_tile::moe_mx;
        t[k % mx::kWindowK] = v;
        if((k + 1) % mx::kWindowK != 0)
            return;
        for(index_t j = 0; j < 2; ++j)
        {
            auto at     = [&](index_t b, index_t i) { return t[mx::window_block(j, b) * mx::kScaleBlock + i]; };
            double tmax = 0;
            for(index_t b = 0; b < 4; ++b)
                for(index_t i = 0; i < mx::kScaleBlock; ++i)
                    tmax = std::max(tmax, std::fabs(at(b, i)));
            int e = 0;
            std::frexp(tmax, &e);
            const double g = std::ldexp(1.0, e - 13);
            for(index_t b = 0; b < 4; ++b)
                for(index_t i = 0; i < mx::kScaleBlock; ++i)
                    bound += at(b, i) - std::floor(at(b, i) / g) * g;
        }
    }
};

/// Host routing: each row picks top_k distinct experts from the first `pool` experts (pool < E
/// leaves experts idle, so num_active < max_active and the bound-and-exit path is exercised).
struct HostRoutes
{
    std::vector<int32_t> ids, count, routes, active, num_active;
    std::vector<float> weights;
};

HostRoutes
make_routes(index_t M, index_t top_k, index_t E, index_t pool, index_t max_m, std::mt19937_64& rng)
{
    HostRoutes h;
    h.ids.resize(size_t(M) * top_k);
    h.weights.resize(h.ids.size());
    h.count.assign(E, 0);
    h.routes.assign(size_t(E) * max_m, -1);
    std::vector<int32_t> perm(pool);
    std::uniform_real_distribution<float> u(0.05f, 1.f);
    for(index_t r = 0; r < M; ++r)
    {
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        float sum = 0;
        for(index_t s = 0; s < top_k; ++s)
        {
            h.ids[r * top_k + s]     = perm[s];
            h.weights[r * top_k + s] = u(rng);
            sum += h.weights[r * top_k + s];
        }
        for(index_t s = 0; s < top_k; ++s)
            h.weights[r * top_k + s] /= sum;
    }
    // Routes of each expert in ascending order (route = row * top_k + slot).
    for(index_t route = 0; route < M * top_k; ++route)
    {
        const int32_t e                            = h.ids[route];
        h.routes[size_t(e) * max_m + h.count[e]++] = route;
    }
    for(index_t e = 0; e < E; ++e)
        if(h.count[e] > 0)
            h.active.push_back(e);
    h.num_active = {static_cast<int32_t>(h.active.size())};
    h.active.resize(E, -1);
    return h;
}

int run(const ck_tile::ArgParser& arg)
{
    const index_t M        = arg.get_int("m");
    const index_t top_k    = arg.get_int("topk");
    const index_t E        = arg.get_int("experts");
    const index_t pool     = std::min(arg.get_int("pool") > 0 ? arg.get_int("pool") : E, E);
    const index_t rank     = arg.get_int("rank");
    const index_t world    = arg.get_int("world");
    const int launches     = arg.get_int("launches");
    const int n_check_rows = arg.get_int("check_rows");
    const index_t K = kDim, I = kDim, P = kDim;
    const index_t routes_n = M * top_k;
    if(top_k > pool)
    {
        std::printf("top_k %d > pool %d\n", top_k, pool);
        return 2;
    }

    std::mt19937_64 rng(arg.get_int("seed"));

    // p: CenteredRMSNorm output, about unit rms, MX fp8.
    std::vector<float> pf(size_t(M) * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for(auto& v : pf)
        v = nd(rng);
    std::vector<uint8_t> p(size_t(M) * K), p_scale(size_t(M) * (K / 32));
    for(index_t r = 0; r < M; ++r)
        moe_host::quantize_row_mxfp8(
            &pf[size_t(r) * K], K, &p[size_t(r) * K], &p_scale[size_t(r) * (K / 32)]);

    // Experts: full (unsharded) weights; this rank reads rows [rank * N / world, +N / world).
    const moe_host::Mxfp4Matrix Wg = moe_host::random_mxfp4(size_t(E) * I, K, rng);
    const moe_host::Mxfp4Matrix Wu = moe_host::random_mxfp4(size_t(E) * I, K, rng);
    const moe_host::Mxfp4Matrix Wd = moe_host::random_mxfp4(size_t(E) * P, I, rng);
    std::vector<uint16_t> norm_w(P);
    for(auto& v : norm_w)
        v = bf16_rne_bits(0.1f * nd(rng));

    const index_t max_m = M;
    const HostRoutes hr = make_routes(M, top_k, E, pool, max_m, rng);

    DeviceBuffer d_p(p), d_ps(p_scale), d_gate(Wg.q), d_gate_s(Wg.s), d_up(Wu.q), d_up_s(Wu.s);
    DeviceBuffer d_down(Wd.q), d_down_s(Wd.s), d_norm_w(norm_w);
    DeviceBuffer d_ids(hr.ids), d_w(hr.weights), d_count(hr.count), d_routes(hr.routes);
    DeviceBuffer d_active(hr.active), d_num_active(hr.num_active);
    DeviceBuffer d_z(size_t(routes_n) * I), d_zs(size_t(routes_n) * (I / 32));
    DeviceBuffer d_y(size_t(routes_n) * P * 2), d_out(size_t(M) * P * 2);

    const ck_tile::moe_route_table table{d_ids.as<int32_t>(),
                                         d_w.as<float>(),
                                         d_count.as<int32_t>(),
                                         d_routes.as<int32_t>(),
                                         d_active.as<int32_t>(),
                                         d_num_active.as<int32_t>(),
                                         M,
                                         max_m,
                                         top_k,
                                         E};

    const GateUp::Args gu{table,
                          d_p.as<ck_tile::fp8_t>(),
                          d_ps.as<uint8_t>(),
                          d_gate.as<ck_tile::pk_fp4_t>(),
                          d_gate_s.as<uint8_t>(),
                          d_up.as<ck_tile::pk_fp4_t>(),
                          d_up_s.as<uint8_t>(),
                          d_z.as<ck_tile::fp8_t>(),
                          d_zs.as<uint8_t>(),
                          I,
                          I,
                          rank * (I / world),
                          rank,
                          world};
    const Down::Args dn{table,
                        d_z.as<ck_tile::fp8_t>(),
                        d_zs.as<uint8_t>(),
                        d_down.as<ck_tile::pk_fp4_t>(),
                        d_down_s.as<uint8_t>(),
                        d_y.as<ck_tile::bf16_t>(),
                        P,
                        P,
                        rank * (P / world),
                        rank,
                        world};
    const Combine::Args cb{table,
                           d_y.as<ck_tile::bf16_t>(),
                           d_norm_w.as<ck_tile::bf16_t>(),
                           d_out.as<ck_tile::bf16_t>(),
                           1e-8f};

    const MoeMK::ArgsTuple args{gu, dn, cb};
    const index_t n_counters = MoeMK::NumCounters(args);
    DeviceBuffer d_counters(size_t(n_counters) * sizeof(uint32_t));
    HIP_CHECK(hipMemset(d_counters.p, 0, size_t(n_counters) * sizeof(uint32_t)));
    const auto kargs = MoeMK::MakeKargs(args, d_counters.as<uint32_t>());
    if(!MoeMK::IsSupportedArgument(kargs))
    {
        std::printf("megakernel arguments not supported\n");
        return 2;
    }

    const ck_tile::stream_config s{nullptr, false};
    const index_t resident = MoeMK::MaxResidentGridSize(s);
    const index_t work     = std::max(GateUp::NumWork(gu), Down::NumWork(dn));
    index_t grid           = arg.get_int("grid");
    grid                   = grid > 0 ? std::min(grid, resident) : std::min(work, resident);
    std::printf("M %d top_k %d experts %d (pool %d, active %d), rank %d/%d: work %d + %d + %d, "
                "%d counters, grid %d (resident %d), LDS %d B\n",
                M,
                top_k,
                E,
                pool,
                hr.num_active[0],
                rank,
                world,
                GateUp::NumWork(gu),
                Down::NumWork(dn),
                Combine::NumWork(cb),
                n_counters,
                grid,
                resident,
                static_cast<int>(MoeMK::kSmemSize));

    std::vector<uint8_t> z0, zs0;
    std::vector<uint16_t> y0, out0;
    int failures = 0;
    for(int l = 0; l < launches; ++l)
    {
        HIP_CHECK(hipMemset(d_z.p, 0xFF, size_t(routes_n) * I));
        HIP_CHECK(hipMemset(d_y.p, 0xFF, size_t(routes_n) * P * 2));
        HIP_CHECK(hipMemset(d_out.p, 0xFF, size_t(M) * P * 2));
        ck_tile::launch_kernel(
            s, ck_tile::make_kernel<1>(MoeMK{}, dim3(grid), MoeMK::BlockSize(), 0, kargs));
        HIP_CHECK(hipDeviceSynchronize());
        auto z   = download<uint8_t>(d_z, size_t(routes_n) * I);
        auto zs  = download<uint8_t>(d_zs, size_t(routes_n) * (I / 32));
        auto y   = download<uint16_t>(d_y, size_t(routes_n) * P);
        auto out = download<uint16_t>(d_out, size_t(M) * P);
        if(l == 0)
        {
            z0   = std::move(z);
            zs0  = std::move(zs);
            y0   = std::move(y);
            out0 = std::move(out);
        }
        else if(z != z0 || zs != zs0 || y != y0 || out != out0)
        {
            ++failures;
            std::printf("launch %d differs from launch 0\n", l);
        }
    }
    uint32_t epoch = 0;
    HIP_CHECK(hipMemcpy(&epoch, d_counters.p, sizeof(epoch), hipMemcpyDeviceToHost));
    if(epoch != static_cast<uint32_t>(launches))
    {
        ++failures;
        std::printf("device epoch %u after %d launches\n", epoch, launches);
    }

    // Reference, rows spread over [0, M). Columns: this rank's slice of z and y; all of out
    // (world 1 only, since out needs the gathered y).
    std::vector<index_t> rows;
    for(int i = 0; i < n_check_rows && i < M; ++i)
        rows.push_back(n_check_rows >= M ? i
                                         : static_cast<index_t>(int64_t(i) * (M - 1) /
                                                                std::max(1, n_check_rows - 1)));
    auto p_at = [&](index_t r, index_t k) {
        return moe_host::e4m3_to_double(p[size_t(r) * K + k]) *
               moe_host::e8m0_to_double(p_scale[size_t(r) * (K / 32) + k / 32]);
    };
    auto z_at = [&](index_t route, index_t k) {
        return moe_host::e4m3_to_double(z0[size_t(route) * I + k]) *
               moe_host::e8m0_to_double(zs0[size_t(route) * (I / 32) + k / 32]);
    };
    const index_t z_lo = rank * (I / world), z_hi = z_lo + I / world;
    const index_t y_lo = rank * (P / world), y_hi = y_lo + P / world;
    int bad_z = 0, bad_y = 0, bad_out = 0;
    auto report =
        [](int& bad, const char* what, index_t a, index_t b, double got, double ref, double tol) {
            if(!(std::fabs(got - ref) <= tol) && bad++ < 5)
                std::printf("  %s[%d, %d] = %g, ref %g, tol %g\n", what, a, b, got, ref, tol);
        };
    for(index_t r : rows)
    {
        for(index_t s_ = 0; s_ < top_k; ++s_)
        {
            const index_t route = r * top_k + s_;
            const index_t e     = hr.ids[route];
            // z: one fp8 rounding (half an ulp, 2^-4 relative); below the normal range the step
            // is 2^-9 of the block scale, <= 2 amax / 448 (x2 if the GPU's amax rounds up a
            // binade); plus the WMMA error of g and u (WmmaErr).
            std::vector<double> zr(I / world), mag(I / world);
            double amax_blk = 0;
            for(index_t n = z_lo; n < z_hi; ++n)
            {
                double g = 0, u = 0;
                WmmaErr eg, eu;
                for(index_t k = 0; k < K; ++k)
                {
                    const double x = p_at(r, k);
                    const double a = x * Wg.at(K, e * I + n, k), b = x * Wu.at(K, e * I + n, k);
                    g += a;
                    u += b;
                    eg.add(k, a);
                    eu.add(k, b);
                }
                const double sg = g / (1 + std::exp(-g));
                zr[n - z_lo]    = sg * u;
                // silu' <= 1.1
                mag[n - z_lo] =
                    1.1 * eg.bound * std::fabs(u) + eu.bound * std::fabs(sg) + 1.1 * eg.bound * eu.bound;
            }
            for(index_t n = z_lo; n < z_hi; ++n)
            {
                if((n - z_lo) % 32 == 0)
                {
                    amax_blk = 0;
                    for(index_t i = 0; i < 32; ++i)
                        amax_blk = std::max(amax_blk, std::fabs(zr[n - z_lo + i]));
                }
                const double ref = zr[n - z_lo];
                const double tol =
                    std::fabs(ref) * 0.0626 + amax_blk / 448 * std::ldexp(1.0, -8) + mag[n - z_lo];
                report(bad_z, "z", route, n, z_at(route, n), ref, tol);
            }
            // y from the GPU's z (gathered z when world > 1 is not available here: K = this
            // rank's slice only when world == 1).
            if(world == 1)
            {
                for(index_t n = y_lo; n < y_hi; ++n)
                {
                    double ref = 0;
                    WmmaErr hw;
                    for(index_t k = 0; k < I; ++k)
                    {
                        const double t = z_at(route, k) * Wd.at(I, e * P + n, k);
                        ref += t;
                        hw.add(k, t);
                    }
                    const double tol = std::fabs(ref) * std::ldexp(1.0, -8) + hw.bound;
                    report(
                        bad_y, "y", route, n, bf16_to_double(y0[size_t(route) * P + n]), ref, tol);
                }
            }
        }
        if(world == 1)
        {
            // out from the GPU's y, with the kernel's bf16 roundings; tolerance: one bf16 ulp per
            // rounded term plus the final rounding.
            std::vector<double> acc(P, 0.0), mag(P, 0.0);
            for(index_t s_ = 0; s_ < top_k; ++s_)
            {
                const index_t route = r * top_k + s_;
                std::vector<double> v(P);
                double ss = 0;
                for(index_t n = 0; n < P; ++n)
                {
                    const float wn = static_cast<float>(bf16_to_double(norm_w[n])) + 1.0f;
                    v[n]           = bf16_round(
                        static_cast<float>(bf16_to_double(y0[size_t(route) * P + n])) * wn);
                    ss += v[n] * v[n];
                }
                const double rs = 1.0 / std::sqrt(ss / P + 1e-8);
                for(index_t n = 0; n < P; ++n)
                {
                    acc[n] += hr.weights[route] * v[n] * rs;
                    mag[n] += hr.weights[route] * std::fabs(v[n] * rs);
                }
            }
            for(index_t n = 0; n < P; ++n)
            {
                const double tol =
                    mag[n] * std::ldexp(1.0, -7) + std::fabs(acc[n]) * std::ldexp(1.0, -8);
                report(bad_out, "out", r, n, bf16_to_double(out0[size_t(r) * P + n]), acc[n], tol);
            }
        }
    }

    const bool pass = failures == 0 && bad_z == 0 && bad_y == 0 && bad_out == 0;
    std::printf("%s: %d/%d launches identical, rows %zu: z bad %d, y bad %d, out bad %d\n",
                pass ? "PASS" : "FAIL",
                launches - failures,
                launches,
                rows.size(),
                bad_z,
                bad_y,
                bad_out);
    return pass ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("m", "16", "rows (tokens): 16 for one DFlash2 verify window, up to 64")
        .insert("topk", "8", "experts per row")
        .insert("experts", "32", "experts in the layer (the model has 256; host memory 25 MB each)")
        .insert(
            "pool", "24", "rows pick from the first `pool` experts (0 = all), so some stay idle")
        .insert("rank", "0", "TP rank")
        .insert("world", "1", "TP world; > 1 checks only this rank's z (no all-gather here)")
        .insert("grid", "0", "workgroups; 0 = min(work, resident)")
        .insert("launches", "3", "launches sharing one counter buffer")
        .insert("check_rows", "2", "rows checked against the double reference")
        .insert("seed", "1", "random seed")
        .insert("selftest", "0", "1: only the host emulation of the fragment layout (no GPU)");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        if(arg.get_int("selftest"))
            return moe_host::selftest_fragments(kDim, arg.get_int("seed")) ? 0 : 1;
        return run(arg);
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
