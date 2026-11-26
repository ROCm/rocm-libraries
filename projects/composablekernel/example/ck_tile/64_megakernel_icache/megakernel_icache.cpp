// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Example 58's two-GEMM megakernel (the target model's pooler + unpooler at decode M), launched
// through an entry point that first warms the kernel's own code
// (ck_tile/ops/megakernel/icache.hpp):
//
//   __global__ CK_TILE_ICACHE_SECTION(ck_mk64_chain) void entry(Kargs k, warm_policy w, int on)
//   {   if(on) icache::warm_code(CK_TILE_ICACHE_SECTION_RANGE(ck_mk64_chain), w);   Chain{}(k);   }
//
// The entry is the only function in ELF section ck_mk64_chain, so the linker's __start_/__stop_
// symbols bound exactly its code. A probe kernel reports that range, which should equal the size
// megakernel_code_size.py prints for the entry.
//
// Per warm mode (none, l2_prefetch, l2_load, inst_prefetch) it runs -iters launches, interleaved
// round-robin, and checks that Y and Z are bit-identical to two plain GEMM launches. It prints the
// median time per mode. -flush_mb > 0 writes a buffer that large between launches, to push the
// code out of L2 the way an expert-weight stream would.
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
#include "ck_tile/ops/megakernel/icache.hpp"

namespace {

using DataType    = ck_tile::bf16_t;
using AccDataType = float;
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using index_t     = ck_tile::index_t;
namespace icache  = ck_tile::icache;

// Same GEMM as example 58: comp_tdm_v3, 128x128x64 tile, 2x2 waves, TdmEpilogue.
using TileShape = ck_tile::TileGemmShape<ck_tile::sequence<128, 128, 64>,
                                         ck_tile::sequence<2, 2, 1>,
                                         ck_tile::sequence<16, 16, 32>>;
using Traits =
    ck_tile::TileGemmUniversalTraits<false, false, false, true, RowMajor, ColumnMajor, RowMajor>;
using Problem         = ck_tile::UniversalGemmPipelineProblem<DataType,
                                                              DataType,
                                                              AccDataType,
                                                              TileShape,
                                                              Traits,
                                                              ck_tile::GemmPipelineScheduler::Intrawave>;
using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<DataType,
                                                         DataType,
                                                         ck_tile::tuple<>,
                                                         AccDataType,
                                                         DataType,
                                                         ck_tile::tuple<>,
                                                         RowMajor,
                                                         ck_tile::element_wise::PassThrough,
                                                         128,
                                                         128,
                                                         2,
                                                         2,
                                                         16,
                                                         16,
                                                         32,
                                                         false,
                                                         1,
                                                         false,
                                                         1,
                                                         1,
                                                         true>;
using GemmKernel      = ck_tile::GemmKernel<ck_tile::GemmTile1DPartitioner<TileShape>,
                                            ck_tile::GemmPipelineAgBgCrCompTDMV3<Problem>,
                                            ck_tile::TdmEpilogue<EpilogueProblem>>;
using GemmPhase       = ck_tile::GemmPhase<GemmKernel>;
using Chain           = ck_tile::Megakernel<ck_tile::megakernel_stage<GemmPhase>,
                                            ck_tile::megakernel_stage<GemmPhase, ck_tile::after_rows<0>>>;

} // namespace

CK_TILE_ICACHE_DECLARE_SECTION(ck_mk64_chain);

/// The megakernel with the warm hook. @p warm_on selects at run time, so one binary times both.
__global__ __launch_bounds__(Chain::kBlockSize)
    CK_TILE_ICACHE_SECTION(ck_mk64_chain) void megakernel_icache_entry(Chain::Kargs kargs,
                                                                       icache::warm_policy warm,
                                                                       int warm_on)
{
#if defined(__HIP_DEVICE_COMPILE__)
    if(warm_on != 0)
        icache::warm_code(CK_TILE_ICACHE_SECTION_RANGE(ck_mk64_chain), warm);
    Chain{}(kargs);
#else
    ck_tile::ignore = kargs;
    ck_tile::ignore = warm;
    ck_tile::ignore = warm_on;
#endif
}

/// out[0] = section begin, out[1] = section bytes, out[2] = entry address (device view).
__global__ void megakernel_icache_probe(uint64_t* out)
{
#if defined(__HIP_DEVICE_COMPILE__)
    const auto r = CK_TILE_ICACHE_SECTION_RANGE(ck_mk64_chain);
    out[0]       = r.begin;
    out[1]       = r.bytes;
    out[2]       = reinterpret_cast<uint64_t>(&megakernel_icache_entry);
#else
    ck_tile::ignore = out;
#endif
}

namespace {

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
    explicit DeviceBuffer(size_t bytes) { HIP_CHECK(hipMalloc(&p, std::max<size_t>(bytes, 1))); }
    ~DeviceBuffer() { (void)hipFree(p); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

std::vector<uint16_t> random_bf16(size_t n, float sigma, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, sigma);
    std::vector<uint16_t> v(n);
    for(auto& x : v)
        x = ck_tile::bit_cast<uint16_t>(ck_tile::type_convert<DataType>(dist(rng)));
    return v;
}

struct Mode
{
    const char* name;
    int on;
    icache::warm_mode mode;
};

int run(const ck_tile::ArgParser& arg)
{
    const index_t M      = arg.get_int("m");
    const index_t H      = arg.get_int("hidden");
    const index_t P      = arg.get_int("pooled");
    const int iters      = arg.get_int("iters");
    const int warm_wgs   = arg.get_int("warm_wgs");
    const size_t flush_b = size_t(arg.get_int("flush_mb")) << 20;

    // Where the warm routine will read: the entry's own section.
    DeviceBuffer d_probe(3 * sizeof(uint64_t));
    megakernel_icache_probe<<<1, 1>>>(static_cast<uint64_t*>(d_probe.p));
    uint64_t probe[3] = {};
    HIP_CHECK(hipMemcpy(probe, d_probe.p, sizeof(probe), hipMemcpyDeviceToHost));
    std::printf(
        "code section ck_mk64_chain: %llu B (%llu lines of %u B), entry %s section start\n",
        static_cast<unsigned long long>(probe[1]),
        static_cast<unsigned long long>((probe[1] + icache::kLineBytes - 1) / icache::kLineBytes),
        icache::kLineBytes,
        probe[2] == probe[0] ? "at" : "NOT at");

    const auto x  = random_bf16(size_t(M) * H, 1.f, 1);
    const auto wp = random_bf16(size_t(P) * H, 1.f / std::sqrt(float(H)), 2);
    const auto wu = random_bf16(size_t(H) * P, 1.f / std::sqrt(float(P)), 3);
    DeviceBuffer d_x(x.size() * 2), d_wp(wp.size() * 2), d_wu(wu.size() * 2);
    DeviceBuffer d_y(size_t(M) * P * 2), d_z(size_t(M) * H * 2);
    DeviceBuffer d_y_ref(size_t(M) * P * 2), d_z_ref(size_t(M) * H * 2);
    DeviceBuffer d_flush(flush_b);
    HIP_CHECK(hipMemcpy(d_x.p, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wp.p, wp.data(), wp.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wu.p, wu.data(), wu.size() * 2, hipMemcpyHostToDevice));

    auto gemm_args = [](const void* a, const void* b, void* c, index_t m, index_t n, index_t k) {
        return GemmKernel::MakeKernelArgs(ck_tile::GemmHostArgs(a, b, c, 1, m, n, k, k, k, n));
    };
    const auto pooler_ref   = gemm_args(d_x.p, d_wp.p, d_y_ref.p, M, P, H);
    const auto unpooler_ref = gemm_args(d_y_ref.p, d_wu.p, d_z_ref.p, M, H, P);
    const auto pooler       = gemm_args(d_x.p, d_wp.p, d_y.p, M, P, H);
    const auto unpooler     = gemm_args(d_y.p, d_wu.p, d_z.p, M, H, P);
    if(!GemmKernel::IsSupportedArgument(pooler) || !GemmKernel::IsSupportedArgument(unpooler))
    {
        std::printf("GEMM arguments not supported\n");
        return 2;
    }

    const ck_tile::stream_config s{nullptr, false};
    for(const auto& k : {pooler_ref, unpooler_ref})
    {
        ck_tile::launch_kernel(
            s,
            ck_tile::make_kernel<1>(
                GemmKernel{}, GemmKernel::GridSize(k.M, k.N, 1), GemmKernel::BlockSize(), 0, k));
    }

    const Chain::ArgsTuple args{pooler, unpooler};
    const index_t n_counters = Chain::NumCounters(args);
    DeviceBuffer d_counters(size_t(n_counters) * sizeof(uint32_t));
    HIP_CHECK(hipMemset(d_counters.p, 0, size_t(n_counters) * sizeof(uint32_t)));
    const auto kargs = Chain::MakeKargs(args, static_cast<uint32_t*>(d_counters.p));
    if(!Chain::IsSupportedArgument(kargs))
    {
        std::printf("megakernel arguments not supported\n");
        return 2;
    }

    // Launch with Chain::BlockSize(), not Chain::kBlockSize: kBlockSize follows the compile pass,
    // and the host pass sees 64-lane waves (256 threads), twice the device's __launch_bounds__.
    const dim3 block = Chain::BlockSize();
    int occupancy    = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, megakernel_icache_entry, block.x, 0));
    const index_t resident = ck_tile::get_available_compute_units(s) * occupancy;
    const index_t work     = std::max(GemmPhase::NumWork(pooler), GemmPhase::NumWork(unpooler));
    const index_t grid     = std::min(work, resident);

    std::vector<uint16_t> y_ref(size_t(M) * P), z_ref(size_t(M) * H), y(y_ref.size()),
        z(z_ref.size());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(y_ref.data(), d_y_ref.p, y_ref.size() * 2, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(z_ref.data(), d_z_ref.p, z_ref.size() * 2, hipMemcpyDeviceToHost));

    const Mode modes[]   = {{"none", 0, icache::warm_mode::l2_prefetch},
                            {"l2_prefetch", 1, icache::warm_mode::l2_prefetch},
                            {"l2_load", 1, icache::warm_mode::l2_load},
                            {"inst_prefetch", 1, icache::warm_mode::inst_prefetch}};
    constexpr int kModes = sizeof(modes) / sizeof(modes[0]);
    std::vector<std::vector<float>> ms(kModes);

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    int failures = 0;
    for(int it = 0; it < iters; ++it)
    {
        for(int mi = 0; mi < kModes; ++mi)
        {
            icache::warm_policy w;
            w.mode           = modes[mi].mode;
            w.num_workgroups = modes[mi].mode == icache::warm_mode::inst_prefetch ? grid : warm_wgs;
            HIP_CHECK(hipMemset(d_y.p, 0xFF, y.size() * 2));
            HIP_CHECK(hipMemset(d_z.p, 0xFF, z.size() * 2));
            if(flush_b != 0)
                HIP_CHECK(hipMemsetAsync(d_flush.p, it & 0xFF, flush_b, nullptr));
            HIP_CHECK(hipEventRecord(e0, nullptr));
            megakernel_icache_entry<<<grid, block, 0, nullptr>>>(
                kargs, w, modes[mi].on);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipEventRecord(e1, nullptr));
            HIP_CHECK(hipEventSynchronize(e1));
            float t = 0;
            HIP_CHECK(hipEventElapsedTime(&t, e0, e1));
            ms[mi].push_back(t);
            HIP_CHECK(hipMemcpy(y.data(), d_y.p, y.size() * 2, hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(z.data(), d_z.p, z.size() * 2, hipMemcpyDeviceToHost));
            if(std::memcmp(y.data(), y_ref.data(), y.size() * 2) != 0 ||
               std::memcmp(z.data(), z_ref.data(), z.size() * 2) != 0)
            {
                ++failures;
                std::printf("iter %d mode %s: result differs from two plain launches\n",
                            it,
                            modes[mi].name);
            }
        }
    }
    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));

    std::printf("M %d H %d P %d grid %d, %d iterations, flush %zu MB between launches\n",
                M,
                H,
                P,
                grid,
                iters,
                flush_b >> 20);
    for(int mi = 0; mi < kModes; ++mi)
    {
        auto v = ms[mi];
        std::sort(v.begin(), v.end());
        std::printf("  warm %-14s median %8.2f us  min %8.2f us\n",
                    modes[mi].name,
                    v.empty() ? 0.0 : 1e3 * v[v.size() / 2],
                    v.empty() ? 0.0 : 1e3 * v.front());
    }
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("m", "16", "rows (tokens)")
        .insert("hidden", "8192", "H: pooler K, unpooler N")
        .insert("pooled", "4096", "P: pooler N, unpooler K")
        .insert("iters", "20", "launches per warm mode (interleaved)")
        .insert("warm_wgs", "8", "workgroups that warm L2 (l2_prefetch / l2_load)")
        .insert("flush_mb", "0", "MB written between launches to evict L2 (0 = none)");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        return run(arg);
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
