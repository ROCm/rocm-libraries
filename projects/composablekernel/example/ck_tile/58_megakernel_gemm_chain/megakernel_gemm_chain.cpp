// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Two chained GEMMs in one persistent launch (ck_tile::Megakernel), checked against the same two
// GEMMs launched separately. The shapes default to the target model's MoE pooler and unpooler at
// decode M:
//   Y = X  * Wp^T   X [M, H] row-major, Wp [P, H] (column-major B), Y [M, P]
//   Z = Y  * Wu^T   Wu [H, P], Z [M, H]
// with H = 8192 and P = 4096. Stage 1 waits on stage 0 through completion counters, either for
// the M tile it reads (-dep=rows) or for all of stage 0 (-dep=all).
//
// Checks, in order:
//   1. Z and Y from the megakernel equal those of two ordinary launches of the same GEMM kernels,
//      bit for bit (same tiles, same K order).
//   2. -check_rows rows of Y and Z against a double reference, |got - ref| <= |ref| * 2^-7 +
//      sum|a*b| * 2^-16 (Z's reference uses the GPU's Y).
//   3. -launches launches reuse one counter buffer without a reset. Y is filled with NaN before
//      each, so a stage-1 tile that ran before its rows of Y were written shows up in Z.
// Exit code: 0 pass, 1 wrong result, 2 unsupported arguments, 3 error.

#include <algorithm>
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
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/megakernel.hpp"

namespace {

using DataType    = ck_tile::bf16_t;
using AccDataType = float;
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using index_t     = ck_tile::index_t;

constexpr index_t kWarpM = 16;
constexpr index_t kWarpN = 16;
constexpr index_t kWarpK = 32;

// comp_tdm_v3, A row-major, B column-major (weights [N, K]), C row-major through TdmEpilogue,
// which clips to the real M x N, so the decode M of 16-64 rows needs no padding.
template <index_t TileM, index_t TileN, index_t TileK, index_t WarpsM, index_t WarpsN>
struct GemmInstance
{
    using TileShape   = ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                               ck_tile::sequence<WarpsM, WarpsN, 1>,
                                               ck_tile::sequence<kWarpM, kWarpN, kWarpK>>;
    using Partitioner = ck_tile::GemmTile1DPartitioner<TileShape>;
    using Traits      = ck_tile::TileGemmUniversalTraits<false, // kPadM
                                                         false, // kPadN
                                                         false, // kPadK
                                                         true,  // DoubleSmemBuffer
                                                         RowMajor,
                                                         ColumnMajor,
                                                         RowMajor>;
    using Problem =
        ck_tile::UniversalGemmPipelineProblem<DataType,
                                              DataType,
                                              AccDataType,
                                              TileShape,
                                              Traits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;
    using Pipeline        = ck_tile::GemmPipelineAgBgCrCompTDMV3<Problem>;
    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<DataType,
                                                             DataType,
                                                             ck_tile::tuple<>,
                                                             AccDataType,
                                                             DataType,
                                                             ck_tile::tuple<>,
                                                             RowMajor,
                                                             ck_tile::element_wise::PassThrough,
                                                             TileM,
                                                             TileN,
                                                             WarpsM,
                                                             WarpsN,
                                                             kWarpM,
                                                             kWarpN,
                                                             kWarpK,
                                                             false, // isCTransposed
                                                             1,     // kNumWaveGroups
                                                             false, // FixedVectorSize
                                                             1,
                                                             1,     // BlockedXDLN_PerWarp
                                                             true>; // DoubleSmemBuffer
    using Epilogue        = ck_tile::TdmEpilogue<EpilogueProblem>;
    using Kernel          = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;
};

using Gemm = GemmInstance<128, 128, 64, 2, 2>;

using PoolerPhase   = ck_tile::GemmPhase<Gemm::Kernel>;
using UnpoolerPhase = ck_tile::GemmPhase<Gemm::Kernel>;

template <typename Dependency>
using Chain     = ck_tile::Megakernel<ck_tile::megakernel_stage<PoolerPhase>,
                                      ck_tile::megakernel_stage<UnpoolerPhase, Dependency>>;
using ChainRows = Chain<ck_tile::after_rows<0>>;
using ChainAll  = Chain<ck_tile::after_all<0>>;

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

float bf16_to_float(uint16_t b) { return ck_tile::bit_cast<float>(static_cast<uint32_t>(b) << 16); }

std::vector<uint16_t> random_bf16(size_t n, float sigma, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, sigma);
    std::vector<uint16_t> v(n);
    for(auto& x : v)
        x = bf16_rne(dist(rng));
    return v;
}

struct DeviceBuffer
{
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { HIP_CHECK(hipMalloc(&p, bytes)); }
    ~DeviceBuffer() { (void)hipFree(p); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// out[r, n] = sum_k a[r, k] * w[n, k], both bf16, rows [0, rows) of a.
int check_rows(const char* name,
               const std::vector<uint16_t>& got,
               const std::vector<uint16_t>& a,
               const std::vector<uint16_t>& w,
               index_t N,
               index_t K,
               const std::vector<index_t>& rows)
{
    int bad = 0;
    for(index_t r : rows)
    {
        for(index_t n = 0; n < N; ++n)
        {
            double ref = 0, mag = 0;
            for(index_t k = 0; k < K; ++k)
            {
                const double p = static_cast<double>(bf16_to_float(a[size_t(r) * K + k])) *
                                 bf16_to_float(w[size_t(n) * K + k]);
                ref += p;
                mag += std::fabs(p);
            }
            const double g   = bf16_to_float(got[size_t(r) * N + n]);
            const double tol = std::fabs(ref) * std::ldexp(1.0, -7) + mag * std::ldexp(1.0, -16);
            if(!(std::fabs(g - ref) <= tol))
            {
                if(bad < 5)
                    std::printf("  %s[%d, %d] = %g, ref %g, tol %g\n", name, r, n, g, ref, tol);
                ++bad;
            }
        }
    }
    return bad;
}

template <typename MK>
int run(const ck_tile::ArgParser& arg)
{
    const index_t M        = arg.get_int("m");
    const index_t H        = arg.get_int("hidden");
    const index_t P        = arg.get_int("pooled");
    const int launches     = arg.get_int("launches");
    const int n_check_rows = arg.get_int("check_rows");

    const auto x  = random_bf16(size_t(M) * H, 1.f, 1);
    const auto wp = random_bf16(size_t(P) * H, 1.f / std::sqrt(float(H)), 2);
    const auto wu = random_bf16(size_t(H) * P, 1.f / std::sqrt(float(P)), 3);

    DeviceBuffer d_x(x.size() * 2), d_wp(wp.size() * 2), d_wu(wu.size() * 2);
    DeviceBuffer d_y(size_t(M) * P * 2), d_z(size_t(M) * H * 2);
    DeviceBuffer d_y_ref(size_t(M) * P * 2), d_z_ref(size_t(M) * H * 2);
    HIP_CHECK(hipMemcpy(d_x.p, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wp.p, wp.data(), wp.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wu.p, wu.data(), wu.size() * 2, hipMemcpyHostToDevice));

    auto gemm_args = [](const void* a, const void* b, void* c, index_t m, index_t n, index_t k) {
        return Gemm::Kernel::MakeKernelArgs(ck_tile::GemmHostArgs(a, b, c, 1, m, n, k, k, k, n));
    };
    const auto pooler_ref   = gemm_args(d_x.p, d_wp.p, d_y_ref.p, M, P, H);
    const auto unpooler_ref = gemm_args(d_y_ref.p, d_wu.p, d_z_ref.p, M, H, P);
    const auto pooler       = gemm_args(d_x.p, d_wp.p, d_y.p, M, P, H);
    const auto unpooler     = gemm_args(d_y.p, d_wu.p, d_z.p, M, H, P);
    if(!Gemm::Kernel::IsSupportedArgument(pooler) || !Gemm::Kernel::IsSupportedArgument(unpooler))
    {
        std::printf("GEMM arguments not supported\n");
        return 2;
    }

    // Reference: the same kernel, two launches.
    const ck_tile::stream_config s{nullptr, false};
    for(const auto& k : {pooler_ref, unpooler_ref})
    {
        ck_tile::launch_kernel(s,
                               ck_tile::make_kernel<1>(Gemm::Kernel{},
                                                       Gemm::Kernel::GridSize(k.M, k.N, 1),
                                                       Gemm::Kernel::BlockSize(),
                                                       0,
                                                       k));
    }

    // Megakernel.
    const typename MK::ArgsTuple args{pooler, unpooler};
    const index_t n_counters = MK::NumCounters(args);
    DeviceBuffer d_counters(size_t(n_counters) * sizeof(uint32_t));
    HIP_CHECK(hipMemset(d_counters.p, 0, size_t(n_counters) * sizeof(uint32_t)));
    const auto kargs = MK::MakeKargs(args, static_cast<uint32_t*>(d_counters.p));
    if(!MK::IsSupportedArgument(kargs))
    {
        std::printf("megakernel arguments not supported\n");
        return 2;
    }

    const index_t resident = MK::MaxResidentGridSize(s);
    const index_t work = std::max(PoolerPhase::NumWork(pooler), UnpoolerPhase::NumWork(unpooler));
    index_t grid       = arg.get_int("grid");
    grid               = grid > 0 ? std::min(grid, resident) : std::min(work, resident);
    std::printf("M %d H %d P %d: %d + %d tiles, %d counters, grid %d (resident %d), LDS %d B\n",
                M,
                H,
                P,
                PoolerPhase::NumWork(pooler),
                UnpoolerPhase::NumWork(unpooler),
                n_counters,
                grid,
                resident,
                static_cast<int>(MK::kSmemSize));

    std::vector<uint16_t> y_ref(size_t(M) * P), z_ref(size_t(M) * H);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(y_ref.data(), d_y_ref.p, y_ref.size() * 2, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(z_ref.data(), d_z_ref.p, z_ref.size() * 2, hipMemcpyDeviceToHost));

    std::vector<uint16_t> y(y_ref.size()), z(z_ref.size());
    int failures = 0;
    for(int l = 0; l < launches; ++l)
    {
        // NaN in Y and Z: anything not overwritten, or read before it was written, stays visible.
        HIP_CHECK(hipMemset(d_y.p, 0xFF, y.size() * 2));
        HIP_CHECK(hipMemset(d_z.p, 0xFF, z.size() * 2));
        ck_tile::launch_kernel(
            s, ck_tile::make_kernel<1>(MK{}, dim3(grid), MK::BlockSize(), 0, kargs));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(y.data(), d_y.p, y.size() * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(z.data(), d_z.p, z.size() * 2, hipMemcpyDeviceToHost));
        const bool y_same = std::memcmp(y.data(), y_ref.data(), y.size() * 2) == 0;
        const bool z_same = std::memcmp(z.data(), z_ref.data(), z.size() * 2) == 0;
        if(!y_same || !z_same)
        {
            ++failures;
            std::printf("launch %d: Y %s, Z %s the two-launch result\n",
                        l,
                        y_same ? "matches" : "DIFFERS from",
                        z_same ? "matches" : "DIFFERS from");
        }
    }

    uint32_t epoch = 0;
    HIP_CHECK(hipMemcpy(&epoch, d_counters.p, sizeof(epoch), hipMemcpyDeviceToHost));
    if(epoch != static_cast<uint32_t>(launches))
    {
        ++failures;
        std::printf("device epoch %u after %d launches\n", epoch, launches);
    }

    std::vector<index_t> rows;
    for(int i = 0; i < n_check_rows && i < M; ++i)
        rows.push_back(n_check_rows >= M ? i
                                         : static_cast<index_t>(int64_t(i) * (M - 1) /
                                                                std::max(1, n_check_rows - 1)));
    const int bad_y = check_rows("Y", y_ref, x, wp, P, H, rows);
    const int bad_z = check_rows("Z", z_ref, y_ref, wu, H, P, rows);

    const bool pass = failures == 0 && bad_y == 0 && bad_z == 0;
    std::printf("%s: %d/%d launches identical to two launches, reference rows %zu: Y bad %d, Z "
                "bad %d\n",
                pass ? "PASS" : "FAIL",
                launches - failures,
                launches,
                rows.size(),
                bad_y,
                bad_z);
    return pass ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("m", "64", "rows (tokens): 16 for one DFlash2 verify window, 64 for four")
        .insert("hidden", "8192", "H: pooler K, unpooler N")
        .insert("pooled", "4096", "P: pooler N, unpooler K")
        .insert("dep", "rows", "stage 1 waits for: rows (its M tile of Y) | all (all of Y)")
        .insert("grid", "0", "workgroups; 0 = min(tiles, resident), capped at resident")
        .insert("launches", "5", "launches sharing one counter buffer")
        .insert("check_rows", "4", "rows checked against the double reference");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        const std::string dep = arg.get_str("dep");
        if(dep == "rows")
            return run<ChainRows>(arg);
        if(dep == "all")
            return run<ChainAll>(arg);
        std::printf("unknown -dep=%s\n", dep.c_str());
        return 2;
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
