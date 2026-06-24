// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Single-config launcher for the M<=8 headroom diagnosis (design doc §15.F).
// Unlike bench_msweep (which sweeps every cell and reports the best), this
// fires ONE fixed GemmDecodeUniversalKernel config in a tight loop so rocprofv3
// can attach hardware counters to a clean, repeated dispatch. Pick the case by
// name; the three cases are the diagnosis points:
//
//   m1n4096  M=1 N=4096 K=7168  mp1/np2/v16/swz8   (the ~73% HBM outlier)
//   m1n8192  M=1 N=8192 K=7168  mp1/np2/v16/swz64  (large-N M=1 reference)
//   m8n8192  M=8 N=8192 K=7168  mp4/np4/v8/swz32   (the mp4 ceil(M/mp)=2 step)
//
// Usage:  prof_gemm_decode_one <case> [iters]
//
// Build target: prof_gemm_decode_one.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm_decode/gemm_decode.hpp"

using namespace ck_tile;

namespace {

template <typename T>
void fill_random(HostTensor<T>& t, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(index_t i = 0; i < static_cast<index_t>(t.get_element_space_size()); ++i)
        t.mData[i] = type_convert<T>(dist(gen));
}

// Launch one fully-resolved config `iters` times. No per-iter timing: the loop
// just gives rocprof a stream of identical dispatches to sample. A single wall
// clock over the loop sanity-checks that we are profiling the intended config.
template <index_t kMPerWarp, index_t kNPerWarp, index_t kVector, bool kChipletSwizzle,
          index_t kChipletNumXcds, index_t kChipletChunk>
void run_case(index_t M, index_t N, index_t K, index_t k_batch, int iters)
{
    using ADataType = bf16_t;
    using BDataType = bf16_t;
    using CDataType = bf16_t;
    using Problem   = GemmDecodeProblem<ADataType,
                                        BDataType,
                                        /*ComputeDataType=*/float,
                                        CDataType,
                                        /*XScaleDataType=*/float,
                                        /*WScaleDataType=*/float,
                                        /*XScaleLayout=*/void,
                                        /*WScaleLayout=*/void,
                                        kVector,
                                        /*kUseDot2=*/false,
                                        /*kUsePackedFp32=*/false,
                                        kMPerWarp,
                                        kNPerWarp,
                                        GemmDecodeOutputAxis::SmallM,
                                        /*kHasBias=*/false,
                                        /*kWarpsPerBlock=*/1,
                                        /*kBPreshuffle=*/false,
                                        kChipletSwizzle,
                                        kChipletNumXcds,
                                        kChipletChunk>;
    using Kernel = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    HostTensor<bf16_t> a({M, K});
    HostTensor<bf16_t> b({N, K});
    HostTensor<bf16_t> c({M, N});
    fill_random(a, 0xA1u);
    fill_random(b, 0xB2u);

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c.get_element_space_size_in_bytes());
    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    c_buf.SetZero();

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        M, N, K,
                                        /*stride_a=*/K,
                                        /*stride_b=*/K,
                                        /*stride_c=*/N,
                                        k_batch);

    if(!Kernel::IsSupportedArgument(kargs))
    {
        std::fprintf(stderr, "run_case: unsupported config\n");
        std::exit(2);
    }

    const stream_config sc{nullptr, /*time_kernel=*/false};
    for(int i = 0; i < 5; ++i) // warmup
        launch_gemm_decode_universal<Kernel>(kargs, sc);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    const auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iters; ++i)
        launch_gemm_decode_universal<Kernel>(kargs, sc);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / double(iters);
    const double bytes = (double(M) * K + double(N) * K + double(M) * N) * sizeof(bf16_t);
    std::fprintf(stderr,
                 "case M=%d N=%d K=%d mp%d np%d v%d kb%d swz%d/c%d : %d iters, "
                 "%.2f us/iter, %.1f GB/s\n",
                 int(M), int(N), int(K), int(kMPerWarp), int(kNPerWarp), int(kVector),
                 int(k_batch), int(kChipletSwizzle), int(kChipletChunk), iters, us,
                 bytes / (us * 1.0e-6) / 1.0e9);
}

// FP8 PerTensor variant (dot2 path) with a kWarpsPerBlock template param so the
// §15.F multi-warp occupancy probe can be profiled head-to-head against the
// single-warp baseline at the N=7168 M=1 blemish point.
template <index_t kMPerWarp, index_t kNPerWarp, index_t kVector, bool kChipletSwizzle,
          index_t kChipletNumXcds, index_t kChipletChunk, index_t kWarpsPerBlock>
void run_case_fp8(index_t M, index_t N, index_t K, index_t k_batch, int iters)
{
    using ADataType = fp8_t;
    using BDataType = fp8_t;
    using CDataType = bf16_t;
    using Problem   = GemmDecodeProblem<ADataType,
                                        BDataType,
                                        /*ComputeDataType=*/float,
                                        CDataType,
                                        /*XScaleDataType=*/float,
                                        /*WScaleDataType=*/float,
                                        GemmDecodeScaleLayout::PerTensor,
                                        GemmDecodeScaleLayout::PerTensor,
                                        kVector,
                                        /*kUseDot2=*/true,
                                        /*kUsePackedFp32=*/false,
                                        kMPerWarp,
                                        kNPerWarp,
                                        GemmDecodeOutputAxis::SmallM,
                                        /*kHasBias=*/false,
                                        kWarpsPerBlock,
                                        /*kBPreshuffle=*/false,
                                        kChipletSwizzle,
                                        kChipletNumXcds,
                                        kChipletChunk>;
    using Kernel = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    HostTensor<fp8_t>  a({M, K});
    HostTensor<fp8_t>  b({N, K});
    HostTensor<bf16_t> c({M, N});
    fill_random(a, 0xA1u);
    fill_random(b, 0xB2u);

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c.get_element_space_size_in_bytes());
    DeviceMem sa_buf(sizeof(float));
    DeviceMem sb_buf(sizeof(float));
    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());
    const float sA = 0.125f, sB = 0.0625f;
    sa_buf.ToDevice(&sA);
    sb_buf.ToDevice(&sB);
    c_buf.SetZero();

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        sa_buf.GetDeviceBuffer(),
                                        sb_buf.GetDeviceBuffer(),
                                        /*p_bias=*/nullptr,
                                        M, N, K,
                                        /*stride_a=*/K,
                                        /*stride_b=*/K,
                                        /*stride_c=*/N,
                                        k_batch);

    if(!Kernel::IsSupportedArgument(kargs))
    {
        std::fprintf(stderr, "run_case_fp8: unsupported config\n");
        std::exit(2);
    }

    const stream_config sc{nullptr, /*time_kernel=*/false};
    for(int i = 0; i < 5; ++i)
        launch_gemm_decode_universal<Kernel>(kargs, sc);
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    const auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iters; ++i)
        launch_gemm_decode_universal<Kernel>(kargs, sc);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / double(iters);
    const double bytes = double(M) * K * sizeof(fp8_t) + double(N) * K * sizeof(fp8_t) +
                         double(M) * N * sizeof(bf16_t);
    std::fprintf(stderr,
                 "case FP8 M=%d N=%d K=%d mp%d np%d v%d kb%d wpb%d : %d iters, "
                 "%.2f us/iter, %.1f GB/s\n",
                 int(M), int(N), int(K), int(kMPerWarp), int(kNPerWarp), int(kVector),
                 int(k_batch), int(kWarpsPerBlock), iters, us, bytes / (us * 1.0e-6) / 1.0e9);
}

} // namespace

int main(int argc, char** argv)
{
    const std::string name = argc > 1 ? argv[1] : "m1n4096";
    const int         iters = argc > 2 ? std::atoi(argv[2]) : 200;
    constexpr index_t K     = 7168;

    if(name == "m1n4096")
        run_case</*mp=*/1, /*np=*/2, /*v=*/16, /*swz=*/true, /*xcd=*/8, /*chunk=*/8>(
            /*M=*/1, /*N=*/4096, K, /*k_batch=*/1, iters);
    else if(name == "m1n8192")
        run_case</*mp=*/1, /*np=*/2, /*v=*/16, /*swz=*/true, /*xcd=*/8, /*chunk=*/64>(
            /*M=*/1, /*N=*/8192, K, /*k_batch=*/1, iters);
    else if(name == "m8n8192")
        run_case</*mp=*/4, /*np=*/4, /*v=*/8, /*swz=*/true, /*xcd=*/8, /*chunk=*/32>(
            /*M=*/8, /*N=*/8192, K, /*k_batch=*/1, iters);
    else if(name == "m1n7168") // FP8 single-warp baseline (the −23% blemish point)
        run_case_fp8</*mp=*/1, /*np=*/1, /*v=*/16, /*swz=*/false, /*xcd=*/8, /*chunk=*/8,
                     /*wpb=*/1>(/*M=*/1, /*N=*/7168, K, /*k_batch=*/1, iters);
    else if(name == "m1n7168mw") // FP8 multi-warp probe (kWarpsPerBlock=4)
        run_case_fp8</*mp=*/1, /*np=*/1, /*v=*/16, /*swz=*/false, /*xcd=*/8, /*chunk=*/8,
                     /*wpb=*/4>(/*M=*/1, /*N=*/7168, K, /*k_batch=*/1, iters);
    else
    {
        std::fprintf(stderr,
                     "unknown case '%s' (m1n4096 | m1n8192 | m8n8192 | m1n7168 | "
                     "m1n7168mw)\n",
                     name.c_str());
        return 1;
    }
    return 0;
}
