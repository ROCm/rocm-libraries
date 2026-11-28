// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// MX GEMM on the comp_tdm_scheduled pipeline (gfx1250): E = A @ B^T with K-major E8M0 scales
// [ceil(K/128)][MN][4]. Named instances carry the policy's tuning knobs; -list prints them,
// -kernel picks one. E is stored through LDS by TDM (MxTdmStoreEpilogue).
//
//   verify  the sparse check inputs, E bit for bit against the host reference (sampled rows
//           when the problem is large)
//   bench   timed on FP4 sin/cos codes (-data=trig; other types random) or random finite inputs
//           (-data=random); with -rotate_mb above 0 the launches cycle over copies of A, B, the
//           scales and E (cold timing); with -v=1 every element of E is then checked against the
//           host reference (check_every_element)
//   check   whether the instance runs the problem; no device work

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/reference/reference_mx_gemm_tdm_scheduled.hpp"
#include "ck_tile/host/timer.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::long_index_t;
namespace sched = ck_tile::mx_tdm_scheduled;

template <typename A,
          typename B,
          typename O,
          index_t TileM,
          index_t TileN,
          index_t TileK,
          index_t WarpM,
          index_t WarpN,
          index_t AtomM,
          typename Tuning,
          typename Traits>
using Pipeline = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduled<
    ck_tile::MxGemmTdmScheduledProblem<
        A,
        B,
        O,
        ck_tile::ClusterTileGemmShape<typename Tuning::Cluster,
                                      ck_tile::sequence<TileM, TileN, TileK>,
                                      ck_tile::sequence<WarpM, WarpN, 1>,
                                      ck_tile::sequence<AtomM, 16, 128>>,
        Traits>,
    typename Tuning::Policy>;

template <typename P>
using Kernel =
    ck_tile::MxGemmKernel<ck_tile::GemmTile2DNMajorPartitioner<typename P::BlockGemmShape>,
                          P,
                          ck_tile::MxTdmStoreEpilogue<P>>;

/// Launches K: the grid of GridSize, and with a cluster tuning the cluster of ClusterSize.
template <typename K>
float launch(const ck_tile::stream_config& s, const typename K::KernelArgs& kargs)
{
    if(!K::IsSupportedArgument(kargs))
    {
        const char* why = K::GetUnsupportedReason(kargs);
        throw std::runtime_error(K::GetName() + ": " + (why ? why : "see CK_TILE_LOGGING"));
    }
    const dim3 grids  = K::GridSize(kargs.M, kargs.N, kargs.k_batch);
    const dim3 blocks = K::BlockSize();
    if constexpr(K::ClusterLaunch)
    {
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<1>(K{}, K::ClusterSize(), grids, blocks, 0, kargs));
#else
        throw std::runtime_error(K::GetName() + ": this build has no cluster launch");
#endif
    }
    else
    {
        return ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(K{}, grids, blocks, 0, kargs));
    }
}

/// A tuning: the cluster (CX N tiles by CY M tiles), then the policy's knobs with its defaults:
/// GL2 prefetch distance, PrefetchLdsRead, SplitBLdsRead, InterleaveLdsRegions, LdsBuffers.
template <index_t CX,
          index_t CY,
          index_t Gl2        = 4,
          bool LdsPrefetch   = true,
          bool SplitB        = true,
          bool Interleave    = true,
          index_t LdsBuffers = 2>
struct Tu
{
    using Cluster = ck_tile::sequence<CY, CX, 1>;
    using Policy  = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<Gl2,
                                                                        LdsPrefetch,
                                                                        SplitB,
                                                                        Interleave,
                                                                        LdsBuffers>;
};

// Whole M and N tiles, no K tail.
using Whole = ck_tile::MxTdmScheduledGemmTraits<false, false, false, false, false, 0>;
// Any M, N (edges), no K tail.
using Any = ck_tile::MxTdmScheduledGemmTraits<true, true, false, false, false, 0>;
// Whole M and N tiles, no K tail, A past 2^31 bytes (64-bit A offsets).
using WholeWideA = ck_tile::MxTdmScheduledGemmTraits<false, false, true, false, false, 0>;

using ck_tile::bf16_t;
using ck_tile::bf8_t;
using ck_tile::fp8_t;
using ck_tile::half_t;
using ck_tile::pk_fp4_t;
using ck_tile::pk_fp6x16_t;

struct Options
{
    index_t M, N, K;
    std::string mode, data;
    int warmup, repeat, rotate_mb, v, verbose;
    uint32_t seed;
};

/// The sparse check inputs (see test_mx_gemm_tdm_scheduled_util.hpp) or random finite ones: FP4
/// any code, FP6 codes with the exponent MSB clear (magnitude at most 1.875), 8-bit codes below 2
/// in magnitude, E8M0 2^-3..2^3. The sparse scales carry an alpha of 2^-6 as 2^-3 on each
/// operand, which keeps E in FP8 range. With `trig` FP4 A and B are instead the E2M1 code, rounded
/// to nearest even, of sin (A) or cos (B) of the element's linear index, K fastest.
struct Host
{
    std::vector<uint8_t> a, b, as, bs;
};

uint8_t e2m1_rne(float x)
{
    // Midpoints between consecutive E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    static constexpr float kMidpoint[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    const float a                       = std::fabs(x);
    uint8_t code                        = 7;
    for(uint8_t c = 0; c < 7; ++c)
    {
        if(a < kMidpoint[c])
        {
            code = c;
            break;
        }
        if(!(a > kMidpoint[c]))
        {
            code = static_cast<uint8_t>(c % 2 == 0 ? c : c + 1);
            break;
        }
    }
    return static_cast<uint8_t>((std::signbit(x) ? 8 : 0) | code);
}

template <typename A, typename B>
Host make_inputs(
    index_t M, index_t N, long_index_t K, index_t tile_k, bool sparse, bool trig, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::set<long_index_t> steps;
    const long_index_t n_steps = (K + 127) / 128, per_tile = tile_k / 128;
    const long_index_t last = (n_steps - 1) / per_tile * per_tile;
    for(long_index_t x : {long_index_t{0}, n_steps - 1, per_tile - 1, per_tile, last - 1, last})
        if(x >= 0 && x < n_steps)
            steps.insert(x);
    while(static_cast<long_index_t>(steps.size()) < std::min<long_index_t>(12, n_steps))
        steps.insert(static_cast<long_index_t>(rng() % n_steps));
    auto code = [&](auto type) -> uint8_t {
        using T = decltype(type);
        if(std::is_same_v<T, pk_fp4_t>)
            return static_cast<uint8_t>(rng() % 16);
        if(std::is_same_v<T, pk_fp6x16_t>)
        {
            if(!sparse)
                return static_cast<uint8_t>(rng() & 0x2F);
            // Sparse: exponent field 0 to 2, the 48 codes of magnitude below 4. The two draws are
            // separate statements, so their order does not depend on the compiler.
            const uint32_t sign      = rng() % 2;
            const uint32_t magnitude = rng() % 24;
            return static_cast<uint8_t>((sign << 5) | magnitude);
        }
        if(!sparse)
            return static_cast<uint8_t>(rng() & 0xBF);
        const int bits = std::is_same_v<T, fp8_t> ? 3 : 2, bias = std::is_same_v<T, fp8_t> ? 7 : 15;
        return static_cast<uint8_t>(((rng() % 2) << 7) | ((bias - rng() % 2) << bits) |
                                    (rng() % (1u << bits)));
    };
    auto operand = [&](auto type, index_t rows, bool use_cos) {
        using T               = decltype(type);
        constexpr bool fp4_t  = std::is_same_v<T, pk_fp4_t>;
        const long_index_t rb = sched::row_bytes<T>(K);
        const bool trig_fp4   = trig && !sparse && fp4_t;
        std::vector<uint8_t> data(static_cast<std::size_t>(rows) * rb, 0);
        // Random FP6 rows: every bit is part of a code (K % 4 == 0); each code's exponent MSB, bits
        // 4, 10, 16 and 22 of every 3 bytes, is cleared.
        if(std::is_same_v<T, pk_fp6x16_t> && !sparse && K % 4 == 0)
        {
            static constexpr uint8_t kMask[3] = {0xEF, 0xFB, 0xBE};
            for(std::size_t i = 0; i < data.size(); ++i)
                data[i] = static_cast<uint8_t>(rng() & kMask[i % 3]);
            return data;
        }
        for(index_t r = 0; r < rows; ++r)
        {
            uint8_t* row = data.data() + r * rb;
            auto put     = [&](long_index_t k) {
                const double index = static_cast<double>(r * K + k);
                const uint8_t x =
                    trig_fp4
                            ? e2m1_rne(static_cast<float>(use_cos ? std::cos(index) : std::sin(index)))
                            : code(type);
                sched::set_element_code<T>(row, k, x);
            };
            if(sparse)
                for(long_index_t st : steps)
                    for(long_index_t k = st * 128; k < std::min(K, (st + 1) * 128); ++k)
                        put(k);
            else
                for(long_index_t k = 0; k < K; ++k)
                    put(k);
        }
        return data;
    };
    Host h;
    h.a                       = operand(A{}, M, false);
    h.b                       = operand(B{}, N, true);
    const long_index_t groups = (K + 31) / 32;
    const bool fp4            = std::is_same_v<A, pk_fp4_t>;
    auto scales               = [&](index_t rows) {
        std::vector<uint8_t> natural(static_cast<std::size_t>(rows) * groups);
        for(auto& x : natural)
            x = static_cast<uint8_t>(!sparse ? 124 + rng() % 7
                                             : (fp4 ? 122 + rng() % 4 : 123 + rng() % 2));
        return ck_tile::mx_scale_to_kmajor(natural.data(), rows, K, groups);
    };
    h.as = scales(M);
    h.bs = scales(N);
    return h;
}

/// Every element of E (M x N of O, row-major, on the device) against the host reference on h's
/// inputs. The products of A, B and the scales are exact in f32, so E and the reference (summed in
/// double) differ only by the rounding of the kernel's f32 sums, which the rounding to O mostly
/// hides. An element passes when it equals the reference, is one step of O from it (the two rounded
/// to neighbouring values of O), or is within sqrt(K) f32 steps at the largest |E| of it (f32
/// rounding left visible in a small element after cancellation); NaN fails. Prints the first
/// failures and how many elements pass each way; returns whether every element passes.
template <typename A, typename B, typename O>
bool check_every_element(
    const Host& h, const ck_tile::DeviceMem& e, index_t M, index_t N, long_index_t K)
{
    const std::size_t count = static_cast<std::size_t>(M) * N;
    const index_t ob        = sched::out_bytes<O>();
    std::vector<uint8_t> got(count * ob), want(got.size());
    e.FromDevice(got.data());
    ck_tile::reference_mx_gemm_tdm_scheduled<A, B, O>(h.a.data(),
                                                      sched::row_bytes<A>(K),
                                                      h.b.data(),
                                                      sched::row_bytes<B>(K),
                                                      h.as.data(),
                                                      h.bs.data(),
                                                      want.data(),
                                                      N,
                                                      M,
                                                      N,
                                                      K);
    // Element i of v as a signed count of steps of O: the sign and magnitude of its bits, so that
    // -0 and +0 are 0 steps apart and neighbouring values of O 1 step.
    auto steps_of = [&](const std::vector<uint8_t>& v, std::size_t i) -> long {
        uint32_t bits = 0;
        std::memcpy(&bits, v.data() + i * ob, ob);
        const uint32_t sign = 1u << (8 * ob - 1);
        return bits & sign ? -static_cast<long>(bits & (sign - 1)) : static_cast<long>(bits);
    };
    float max_abs = 1.0f;
    for(std::size_t i = 0; i < count; ++i)
        max_abs = std::max(max_abs, std::fabs(sched::load_out<O>(want.data(), i)));
    const double allowance = std::sqrt(static_cast<double>(K)) *
                             std::ldexp(1.0, static_cast<int>(std::floor(std::log2(max_abs))) - 23);
    std::size_t identical = 0, one_step = 0, within = 0, bad = 0;
    for(std::size_t i = 0; i < count; ++i)
    {
        const long steps = std::labs(steps_of(got, i) - steps_of(want, i));
        const float out  = sched::load_out<O>(got.data(), i);
        const float ref  = sched::load_out<O>(want.data(), i);
        if(steps == 0)
            ++identical;
        else if(steps == 1)
            ++one_step;
        else if(std::fabs(static_cast<double>(out) - ref) <= allowance)
            ++within;
        else if(bad++ < 8)
            std::printf("E(%zu, %zu): got %g, want %g, %ld steps of O apart\n",
                        i / N,
                        i % N,
                        out,
                        ref,
                        steps);
    }
    std::printf(
        "%s: every element of E (%d x %d): %zu identical to the reference, %zu one step of O "
        "apart, %zu within the f32 allowance %g, %zu beyond\n",
        bad ? "FAIL" : "PASS",
        M,
        N,
        identical,
        one_step,
        within,
        allowance,
        bad);
    return bad == 0;
}

/// With -verbose: the top-left corners, up to 8 x 8, of A and of B as stored (N rows of K values),
/// with each row's E8M0 scales of its first K groups as powers of two, and of E from the device and
/// from the host reference.
template <typename A, typename B, typename O>
void print_corners(const Host& h, const ck_tile::DeviceMem& e, index_t M, index_t N, long_index_t K)
{
    const index_t rows_m = std::min<index_t>(M, 8), rows_n = std::min<index_t>(N, 8);
    const index_t cols_k = static_cast<index_t>(std::min<long_index_t>(K, 8));
    const long_index_t groups =
        std::min<long_index_t>((K + sched::kScaleGroupK - 1) / sched::kScaleGroupK, 4);
    auto operand = [&](const char* name,
                       auto type,
                       const std::vector<uint8_t>& data,
                       const std::vector<uint8_t>& scales,
                       index_t rows,
                       index_t extent) {
        using T               = decltype(type);
        const long_index_t rb = sched::row_bytes<T>(K);
        std::printf("%s rows 0..%d, K 0..%d, then the scales of K groups 0..%lld\n",
                    name,
                    rows - 1,
                    cols_k - 1,
                    static_cast<long long>(groups - 1));
        for(index_t r = 0; r < rows; ++r)
        {
            std::printf("  %4d:", r);
            for(index_t k = 0; k < cols_k; ++k)
                std::printf(" %8.4g", sched::decode_element<T>(data.data() + r * rb, k));
            std::printf("   scales");
            for(long_index_t g = 0; g < groups; ++g)
            {
                const long_index_t k = g * sched::kScaleGroupK;
                const uint8_t code   = scales[((k / sched::kScaleStepK) * extent + r) * 4 +
                                            (k % sched::kScaleStepK) / sched::kScaleGroupK];
                std::printf(" 2^%d", static_cast<int>(code) - 127);
            }
            std::printf("\n");
        }
    };
    operand("A", A{}, h.a, h.as, rows_m, M);
    operand("B (row n is column n of B)", B{}, h.b, h.bs, rows_n, N);
    const index_t ob = sched::out_bytes<O>();
    std::vector<uint8_t> got(static_cast<std::size_t>(M) * N * ob);
    e.FromDevice(got.data());
    std::vector<index_t> rows(rows_m);
    for(index_t r = 0; r < rows_m; ++r)
        rows[r] = r;
    std::vector<uint8_t> want(static_cast<std::size_t>(rows_m) * N * ob);
    ck_tile::reference_mx_gemm_tdm_scheduled<A, B, O>(h.a.data(),
                                                      sched::row_bytes<A>(K),
                                                      h.b.data(),
                                                      sched::row_bytes<B>(K),
                                                      h.as.data(),
                                                      h.bs.data(),
                                                      want.data(),
                                                      N,
                                                      M,
                                                      N,
                                                      K,
                                                      0,
                                                      rows.data(),
                                                      rows_m);
    for(const bool device : {true, false})
    {
        std::printf("E from the %s, rows 0..%d, columns 0..%d\n",
                    device ? "device" : "host reference",
                    rows_m - 1,
                    std::min<index_t>(N, 8) - 1);
        for(index_t r = 0; r < rows_m; ++r)
        {
            std::printf("  %4d:", r);
            for(index_t n = 0; n < std::min<index_t>(N, 8); ++n)
                std::printf(" %10.5g",
                            sched::load_out<O>(device ? got.data() : want.data(),
                                               static_cast<std::size_t>(r) * N + n));
            std::printf("\n");
        }
    }
}

/// The number of copies of A, B, the scales and E that the launches cycle over with -rotate_mb:
/// enough copies of `bytes` to fill rotate_mb MiB, at least one and at most `cap`.
long_index_t rotating_copies(std::size_t bytes, int rotate_mb, long_index_t cap)
{
    const long_index_t budget = static_cast<long_index_t>(rotate_mb) << 20;
    const long_index_t per    = std::max<long_index_t>(static_cast<long_index_t>(bytes), 1);
    return std::max<long_index_t>(1, std::min(cap, (budget + per - 1) / per));
}

/// Mean time in ms of s.nrepeat_ launches of K, launch i on kargs[i % kargs.size()], after
/// max(s.cold_niters_, 1) untimed launches and then s.nrepeat_ more untimed ones; each phase starts
/// at kargs[0] on an idle device. Every launch is prepared (make_kernel) before the timing, and one
/// GPU timer spans the timed launches.
template <typename K>
float time_rotating(const ck_tile::stream_config& s,
                    const std::vector<typename K::KernelArgs>& kargs)
{
    // The first launch also runs launch's checks (supported arguments, cluster launch).
    launch<K>(ck_tile::stream_config{s.stream_id_, false}, kargs[0]);
    const dim3 grids  = K::GridSize(kargs[0].M, kargs[0].N, kargs[0].k_batch);
    const dim3 blocks = K::BlockSize();
    auto prepare      = [&](const typename K::KernelArgs& ka) {
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
        if constexpr(K::ClusterLaunch)
            return ck_tile::make_kernel<1>(K{}, K::ClusterSize(), grids, blocks, 0, ka);
        else
#endif
            return ck_tile::make_kernel<1>(K{}, grids, blocks, 0, ka);
    };
    std::vector<decltype(prepare(kargs[0]))> calls;
    for(const auto& ka : kargs)
        calls.push_back(prepare(ka));
    for(int i = 1; i < s.cold_niters_; ++i)
        ck_tile::launch_and_check(s, calls[i % calls.size()]);
    HIP_CHECK_ERROR(hipStreamSynchronize(s.stream_id_));
    for(int i = 0; i < s.nrepeat_; ++i)
        ck_tile::launch_and_check(s, calls[i % calls.size()]);
    // start waits for the stream, so the timed batch also starts on an idle device.
    ck_tile::gpu_timer timer;
    timer.start(s.stream_id_);
    for(int i = 0; i < s.nrepeat_; ++i)
        ck_tile::launch_and_check(s, calls[i % calls.size()]);
    timer.stop(s.stream_id_);
    return s.nrepeat_ > 0 ? timer.duration() / s.nrepeat_ : 0.0f;
}

template <typename K>
int run_kernel(const Options& o)
{
    using P = typename K::GemmPipeline;
    using A = typename P::ADataType;
    using B = typename P::BDataType;
    using O = typename P::ODataType;
    {
        uint8_t dummy = 0;
        const ck_tile::MxGemmHostArgs<> probe{{&dummy},
                                              {&dummy},
                                              {&dummy},
                                              {&dummy},
                                              {},
                                              &dummy,
                                              1,
                                              o.M,
                                              o.N,
                                              o.K,
                                              {o.K},
                                              {o.K},
                                              {},
                                              o.N};
        if(const char* why = K::GetUnsupportedReason(K::MakeKernelArgs(probe)))
        {
            std::cerr << K::GetName() << " does not run M " << o.M << " N " << o.N << " K " << o.K
                      << ": " << why << "\n";
            return 2;
        }
    }
    std::cout << K::GetName() << " M " << o.M << " N " << o.N << " K " << o.K << " grid "
              << K::GridSize(o.M, o.N, 1).x << "x" << K::GridSize(o.M, o.N, 1).y << " lds "
              << K::GetSmemSize() << "\n";
    if(o.mode == "check")
        return 0;
    const Host h =
        make_inputs<A, B>(o.M, o.N, o.K, P::kTileK, o.mode == "verify", o.data == "trig", o.seed);

    const std::size_t e_bytes = static_cast<std::size_t>(o.M) * o.N * sched::out_bytes<O>();
    ck_tile::DeviceMem a(h.a.size()), b(h.b.size()), as(h.as.size()), bs(h.bs.size()), e(e_bytes);
    a.ToDevice(h.a.data());
    b.ToDevice(h.b.data());
    as.ToDevice(h.as.data());
    bs.ToDevice(h.bs.data());
    e.SetZero();
    const ck_tile::MxGemmHostArgs<> host_args{{a.GetDeviceBuffer()},
                                              {as.GetDeviceBuffer()},
                                              {b.GetDeviceBuffer()},
                                              {bs.GetDeviceBuffer()},
                                              {},
                                              e.GetDeviceBuffer(),
                                              1,
                                              o.M,
                                              o.N,
                                              o.K,
                                              {o.K},
                                              {o.K},
                                              {},
                                              o.N};
    const auto args = K::MakeKernelArgs(host_args);

    if(o.mode == "bench")
    {
        const double flops = 2.0 * o.M * o.N * static_cast<double>(o.K);
        // The bytes a launch moves: A, B, both K-major scales and E (no C is read).
        const std::size_t copy_bytes =
            h.a.size() + h.b.size() + h.as.size() + h.bs.size() + e_bytes;
        const auto report = [&](float ms) {
            std::printf("%.3f us, %.1f TFLOPS, %.3f TB/s",
                        ms * 1e3,
                        flops / (ms * 1e-3) / 1e12,
                        static_cast<double>(copy_bytes) / (ms * 1e-3) / 1e12);
        };
        if(o.rotate_mb == 0)
        {
            const float ms =
                launch<K>(ck_tile::stream_config{nullptr, true, 0, o.warmup, o.repeat}, args);
            report(ms);
            std::printf("\n");
            if(o.verbose)
                print_corners<A, B, O>(h, e, o.M, o.N, o.K);
            return o.v && !check_every_element<A, B, O>(h, e, o.M, o.N, o.K) ? 1 : 0;
        }
        // Copy 0 is the buffers above, its E zeroed; copies 1 and up hold the same inputs and their
        // E starts as 0xFF bytes, so an element no launch writes differs from copy 0's.
        const long_index_t copies = rotating_copies(
            copy_bytes, o.rotate_mb, std::max<long_index_t>({o.warmup, o.repeat, 1}));
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> extra;
        std::vector<typename K::KernelArgs> kargs{args};
        std::vector<const ck_tile::DeviceMem*> outs{&e};
        for(long_index_t c = 1; c < copies; ++c)
        {
            auto buffer = [&](std::size_t bytes, const void* host) {
                extra.push_back(std::make_unique<ck_tile::DeviceMem>(bytes));
                if(host)
                    extra.back()->ToDevice(host);
                else
                    HIP_CHECK_ERROR(hipMemset(extra.back()->GetDeviceBuffer(), 0xFF, bytes));
                return extra.back()->GetDeviceBuffer();
            };
            void* ca  = buffer(h.a.size(), h.a.data());
            void* cb  = buffer(h.b.size(), h.b.data());
            void* cas = buffer(h.as.size(), h.as.data());
            void* cbs = buffer(h.bs.size(), h.bs.data());
            void* ce  = buffer(e_bytes, nullptr);
            outs.push_back(extra.back().get());
            kargs.push_back(K::MakeKernelArgs(ck_tile::MxGemmHostArgs<>{
                {ca}, {cas}, {cb}, {cbs}, {}, ce, 1, o.M, o.N, o.K, {o.K}, {o.K}, {}, o.N}));
        }
        const float ms =
            time_rotating<K>(ck_tile::stream_config{nullptr, true, 0, o.warmup, o.repeat}, kargs);
        HIP_CHECK_ERROR(hipDeviceSynchronize());
        // Every copy holds the same inputs, so E of the middle and of the last copy must equal E
        // of copy 0, byte for byte.
        std::vector<uint8_t> first(e_bytes), other(e_bytes);
        e.FromDevice(first.data());
        for(long_index_t c : std::set<long_index_t>{copies / 2, copies - 1})
        {
            if(c == 0)
                continue;
            outs[c]->FromDevice(other.data());
            if(other != first)
            {
                std::printf("FAIL: E of rotating copy %lld differs from copy 0\n",
                            static_cast<long long>(c));
                return 1;
            }
        }
        report(ms);
        std::printf(", %lld rotating copies of %.1f MiB\n",
                    static_cast<long long>(copies),
                    static_cast<double>(copy_bytes) / (1 << 20));
        if(o.verbose)
            print_corners<A, B, O>(h, e, o.M, o.N, o.K);
        return o.v && !check_every_element<A, B, O>(h, e, o.M, o.N, o.K) ? 1 : 0;
    }

    launch<K>(ck_tile::stream_config{nullptr, false}, args);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    if(o.verbose)
        print_corners<A, B, O>(h, e, o.M, o.N, o.K);
    std::vector<uint8_t> got(e_bytes);
    e.FromDevice(got.data());
    // Full check up to 2^32 multiply-adds, else the first and last tile of rows and 64 random.
    std::vector<index_t> rows;
    if(static_cast<double>(o.M) * o.N * o.K <= 4294967296.0)
        for(index_t r = 0; r < o.M; ++r)
            rows.push_back(r);
    else
    {
        std::set<index_t> pick;
        std::mt19937 rng(0xB10C);
        for(index_t r = 0; r < std::min(o.M, P::kTileM); ++r)
        {
            pick.insert(r);
            pick.insert(o.M - 1 - r);
        }
        for(int i = 0; i < 64; ++i)
            pick.insert(static_cast<index_t>(rng() % o.M));
        rows.assign(pick.begin(), pick.end());
    }
    const index_t ob = sched::out_bytes<O>();
    std::vector<uint8_t> want(rows.size() * o.N * ob);
    ck_tile::reference_mx_gemm_tdm_scheduled<A, B, O>(h.a.data(),
                                                      sched::row_bytes<A>(o.K),
                                                      h.b.data(),
                                                      sched::row_bytes<B>(o.K),
                                                      h.as.data(),
                                                      h.bs.data(),
                                                      want.data(),
                                                      o.N,
                                                      o.M,
                                                      o.N,
                                                      o.K,
                                                      0,
                                                      rows.data(),
                                                      static_cast<index_t>(rows.size()));
    std::size_t bad = 0;
    for(std::size_t i = 0; i < rows.size(); ++i)
        for(index_t n = 0; n < o.N; ++n)
        {
            const std::size_t g = static_cast<std::size_t>(rows[i]) * o.N + n, w = i * o.N + n;
            if(std::memcmp(got.data() + g * ob, want.data() + w * ob, ob) != 0 && bad++ < 8)
                std::printf("E(%d, %d): got %g, want %g\n",
                            rows[i],
                            n,
                            sched::load_out<O>(got.data(), g),
                            sched::load_out<O>(want.data(), w));
        }
    std::printf("%s: %zu of %zu rows checked, %zu mismatches\n",
                bad ? "FAIL" : "PASS",
                rows.size(),
                static_cast<std::size_t>(o.M),
                bad);
    return bad ? 1 : 0;
}

/// Runs pipeline P in the variant of the loop's tile count.
template <typename P>
int run(const Options& o)
{
    const index_t full_tiles = P::GetFullTiles(o.K);
    return P::TailHandler(
        [&](auto, auto tail_num) {
            using PT = typename P::template WithTailNum<decltype(tail_num)::value>;
            return run_kernel<Kernel<PT>>(o);
        },
        P::BlockHasHotloop(full_tiles),
        P::GetBlockLoopTailNum(full_tiles));
}

/// The named instances: the tuned instances of the maf, medium, mab and sm problems and variants
/// of each knob.
const std::map<std::string, std::pair<const char*, std::function<int(const Options&)>>>& instances()
{
    static const std::map<std::string, std::pair<const char*, std::function<int(const Options&)>>>
        m{
            {"maf",
             {"mxfp4 t256x256x512 c4x4, fp8 out (MAF winner)",
              run<Pipeline<pk_fp4_t, pk_fp4_t, fp8_t, 256, 256, 512, 2, 2, 32, Tu<4, 4>, Whole>>}},
            {"maf_fp16",
             {"maf with fp16 out",
              run<Pipeline<pk_fp4_t, pk_fp4_t, half_t, 256, 256, 512, 2, 2, 32, Tu<4, 4>, Whole>>}},
            {"maf_bf16",
             {"maf with bf16 out",
              run<Pipeline<pk_fp4_t, pk_fp4_t, bf16_t, 256, 256, 512, 2, 2, 32, Tu<4, 4>, Whole>>}},
            {"maf_c2x2",
             {"maf on a 2x2 cluster",
              run<Pipeline<pk_fp4_t, pk_fp4_t, fp8_t, 256, 256, 512, 2, 2, 32, Tu<2, 2>, Whole>>}},
            {"medium_t128",
             {"mxfp4 t128x128x1024 c2x2",
              run<Pipeline<pk_fp4_t, pk_fp4_t, fp8_t, 128, 128, 1024, 2, 2, 32, Tu<2, 2>, Whole>>}},
            {"medium",
             {"mxfp4 t128x128x1024 16x16 atom, c4x4, no SplitBLdsRead (medium winner)",
              run<Pipeline<pk_fp4_t,
                           pk_fp4_t,
                           fp8_t,
                           128,
                           128,
                           1024,
                           2,
                           2,
                           16,
                           Tu<4, 4, 4, true, false>,
                           Whole>>}},
            {"mab",
             {"mxfp4 t128x16x512 wave 4x1, 16x16 atom, c1x4, no PrefetchLdsRead (MAB)",
              run<Pipeline<pk_fp4_t,
                           pk_fp4_t,
                           fp8_t,
                           128,
                           16,
                           512,
                           4,
                           1,
                           16,
                           Tu<1, 4, 0, false, false>,
                           Whole>>}},
            {"mab256",
             {"mab with a t256x16x512 tile (MAB winner)",
              run<Pipeline<pk_fp4_t,
                           pk_fp4_t,
                           fp8_t,
                           256,
                           16,
                           512,
                           4,
                           1,
                           16,
                           Tu<1, 4, 0, false, false>,
                           Whole>>}},
            {"any",
             {"mxfp4 t256x256x512 c1x1 for any M, N and tile count (no K tail)",
              run<Pipeline<pk_fp4_t, pk_fp4_t, fp8_t, 256, 256, 512, 2, 2, 32, Tu<1, 1>, Any>>}},
            {"fp8",
             {"mxfp8 t128x128x256 16x16 atom c2x2",
              run<Pipeline<fp8_t, fp8_t, fp8_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, Whole>>}},
            {"fp8_bf8_bf16",
             {"fp8 x bf8, bf16 out, t128x128x256 c2x2",
              run<Pipeline<fp8_t, bf8_t, bf16_t, 128, 128, 256, 2, 2, 16, Tu<2, 2>, Whole>>}},
            {"sm_fp6",
             {"mxfp6 t64x128x512 16x16 atom, c4x4, GL2 prefetch 2, 3 LDS buffers, no "
              "SplitBLdsRead or InterleaveLdsRegions, bf16 out (SM)",
              run<Pipeline<pk_fp6x16_t,
                           pk_fp6x16_t,
                           bf16_t,
                           64,
                           128,
                           512,
                           2,
                           2,
                           16,
                           Tu<4, 4, 2, true, false, false, 3>,
                           Whole>>}},
            {"mab_fp6",
             {"mxfp6 t256x16x512 wave 4x1, 16x16 atom, c1x2, no GL2 prefetch or "
              "PrefetchLdsRead, bf16 out (MAB)",
              run<Pipeline<pk_fp6x16_t,
                           pk_fp6x16_t,
                           bf16_t,
                           256,
                           16,
                           512,
                           4,
                           1,
                           16,
                           Tu<1, 2, 0, false, false>,
                           Whole>>}},
            {"mab_fp6_wide",
             {"mab_fp6 for A past 2^31 bytes (WideA)",
              run<Pipeline<pk_fp6x16_t,
                           pk_fp6x16_t,
                           bf16_t,
                           256,
                           16,
                           512,
                           4,
                           1,
                           16,
                           Tu<1, 2, 0, false, false>,
                           WholeWideA>>}},
        };
    return m;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("kernel", "maf", "named instance (-list=1)")
        .insert("list", "0", "1: print the instances and exit")
        .insert("m", "1024", "M")
        .insert("n", "1024", "N")
        .insert("k", "2048", "K (a multiple of 32)")
        .insert("mode", "verify", "verify | bench | check")
        .insert("data", "trig", "bench data: trig (FP4 sin/cos codes) | random")
        .insert("v", "0", "bench: 1 to check every element of E against the host reference")
        .insert("verbose",
                "0",
                "1: print the top-left corners of A, B (values and scales) and E (device and host "
                "reference)")
        .insert("warmup", "50", "bench: untimed launches (with -rotate_mb, -repeat more follow)")
        .insert("repeat", "200", "bench: timed launches")
        .insert("rotate_mb",
                "0",
                "bench: MiB of copies of A, B, the scales and E that the launches cycle over, "
                "rounded up to whole copies, at most max(warmup, repeat) (0: one set of buffers)")
        .insert("seed", "23206", "input RNG seed");
    if(!arg.parse(argc, argv))
        return 1;
    if(arg.get_int("list"))
    {
        for(const auto& [name, entry] : instances())
            std::printf("%-14s %s\n", name.c_str(), entry.first);
        return 0;
    }
    const auto it = instances().find(arg.get_str("kernel"));
    if(it == instances().end())
    {
        std::cerr << "unknown -kernel " << arg.get_str("kernel") << " (see -list=1)\n";
        return 1;
    }
    const Options o{arg.get_int("m"),
                    arg.get_int("n"),
                    arg.get_int("k"),
                    arg.get_str("mode"),
                    arg.get_str("data"),
                    arg.get_int("warmup"),
                    arg.get_int("repeat"),
                    arg.get_int("rotate_mb"),
                    arg.get_int("v"),
                    arg.get_int("verbose"),
                    static_cast<uint32_t>(arg.get_int("seed"))};
    if(o.mode != "verify" && o.mode != "bench" && o.mode != "check")
    {
        std::cerr << "-mode must be verify, bench or check\n";
        return 1;
    }
    if(o.rotate_mb < 0)
    {
        std::cerr << "-rotate_mb must be 0 or more\n";
        return 1;
    }
    if(o.data != "trig" && o.data != "random")
    {
        std::cerr << "-data must be trig or random\n";
        return 1;
    }
    return it->second.second(o);
}
