// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Standalone microbench for the two A1/A3 levers on GemmDecodeUniversalKernel:
//   - A1: kNPerWarp N-tile register reuse (load the shared A row once, reuse
//         across kNPerWarp B rows).
//   - A3: the XCD-aware workgroup swizzle (chiplet chunk_size sweep).
// For each shape it times the (kNPerWarp, chiplet chunk_size) grid against the
// kNPerWarp=1 / swizzle-off baseline and reports the best cell.
// Build target: bench_gemm_decode_chiplet_swizzle.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm_decode/gemm_decode.hpp"

using namespace ck_tile;

namespace {

struct Shape
{
    index_t M;
    index_t N;
    index_t K;
};

template <typename T>
void fill_random(HostTensor<T>& t, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(index_t i = 0; i < static_cast<index_t>(t.get_element_space_size()); ++i)
    {
        t.mData[i] = type_convert<T>(dist(gen));
    }
}

template <index_t kNPerWarp, bool kChipletSwizzle, index_t kChipletNumXcds, index_t kChipletChunk>
float bench_one(const Shape& s, int warmup, int repeat,
                const DeviceMem& a_buf, const DeviceMem& b_buf, DeviceMem& c_buf)
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
                                        /*kVector=*/8,
                                        /*kUseDot2=*/false,
                                        /*kUsePackedFp32=*/false,
                                        /*kMPerWarp=*/1,
                                        kNPerWarp,
                                        GemmDecodeOutputAxis::SmallM,
                                        /*kHasBias=*/false,
                                        /*kWarpsPerBlock=*/1,
                                        /*kBPreshuffle=*/false,
                                        kChipletSwizzle,
                                        kChipletNumXcds,
                                        kChipletChunk>;
    using Kernel = GemmDecodeUniversalKernel<Problem, GemmDecodePolicy>;

    auto kargs = Kernel::MakeKernelArgs(a_buf.GetDeviceBuffer(),
                                        b_buf.GetDeviceBuffer(),
                                        c_buf.GetDeviceBuffer(),
                                        s.M, s.N, s.K,
                                        /*stride_a=*/s.K,
                                        /*stride_b=*/s.K,
                                        /*stride_c=*/s.N,
                                        /*k_batch=*/1);

    {
        const stream_config sc{nullptr, /*time_kernel=*/false};
        for(int i = 0; i < warmup; ++i)
            launch_gemm_decode_universal<Kernel>(kargs, sc);
    }
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    const stream_config sc_timed{nullptr,
                                 /*time_kernel=*/true,
                                 /*log_level=*/0,
                                 /*cold_niters=*/0,
                                 /*nrepeat=*/repeat};
    return launch_gemm_decode_universal<Kernel>(kargs, sc_timed);
}

struct Best
{
    float    t_us      = 1.0e30f;
    index_t  n_per_warp = 1;
    bool     swizzle    = false;
    index_t  chunk      = 8;
};

// Sweep the five chiplet chunk sizes for a fixed compile-time kNPerWarp and
// fold the results (plus the swizzle-off point) into `best`. Returns the
// swizzle-off time for this kNPerWarp so the caller can show N-reuse alone.
template <index_t kNPerWarp>
float sweep_np(const Shape& s, int warmup, int repeat,
               const DeviceMem& a_buf, const DeviceMem& b_buf, DeviceMem& c_buf,
               Best& best)
{
    auto consider = [&](float t, bool swz, index_t chunk) {
        if(t < best.t_us)
            best = Best{t, kNPerWarp, swz, chunk};
    };

    const float t_off = bench_one<kNPerWarp, false, 8, 8>(s, warmup, repeat, a_buf, b_buf, c_buf);
    consider(t_off, /*swz=*/false, /*chunk=*/8);

    consider(bench_one<kNPerWarp, true, 8, 4>(s, warmup, repeat, a_buf, b_buf, c_buf), true, 4);
    consider(bench_one<kNPerWarp, true, 8, 8>(s, warmup, repeat, a_buf, b_buf, c_buf), true, 8);
    consider(bench_one<kNPerWarp, true, 8, 16>(s, warmup, repeat, a_buf, b_buf, c_buf), true, 16);
    consider(bench_one<kNPerWarp, true, 8, 32>(s, warmup, repeat, a_buf, b_buf, c_buf), true, 32);
    consider(bench_one<kNPerWarp, true, 8, 64>(s, warmup, repeat, a_buf, b_buf, c_buf), true, 64);
    return t_off;
}

void run_shape(const Shape& s, int warmup, int repeat)
{
    HostTensor<bf16_t> a({s.M, s.K});
    HostTensor<bf16_t> b({s.N, s.K});
    HostTensor<bf16_t> c({s.M, s.N});
    fill_random(a, 0xA1u);
    fill_random(b, 0xB2u);

    DeviceMem a_buf(a.get_element_space_size_in_bytes());
    DeviceMem b_buf(b.get_element_space_size_in_bytes());
    DeviceMem c_buf(c.get_element_space_size_in_bytes());
    a_buf.ToDevice(a.mData.data());
    b_buf.ToDevice(b.mData.data());

    const auto bw = [&](float t_us) {
        return 2.0f * float(s.M) * float(s.N) * float(s.K) / (t_us * 1.0e-6f) / 1.0e12f;
    };

    // Baseline: kNPerWarp=1, swizzle off.
    const float t_base =
        bench_one<1, false, 8, 8>(s, warmup, repeat, a_buf, b_buf, c_buf) * 1000.0f;

    Best best;
    // N must be divisible by kNPerWarp; the bench shapes are all multiples of
    // 4, so the {1,2,4} fan-out is always legal here.
    const float np1_off = sweep_np<1>(s, warmup, repeat, a_buf, b_buf, c_buf, best) * 1000.0f;
    const float np2_off = sweep_np<2>(s, warmup, repeat, a_buf, b_buf, c_buf, best) * 1000.0f;
    const float np4_off = sweep_np<4>(s, warmup, repeat, a_buf, b_buf, c_buf, best) * 1000.0f;
    const float best_us = best.t_us * 1000.0f;

    std::printf("M=%4d N=%5d K=%5d  base=%7.2fus  Nreuse-only[np2=%7.2f np4=%7.2f]  "
                "best=%7.2fus (np%d %s chunk=%d) spd=%5.3fx  %5.2fTF/s\n",
                s.M, s.N, s.K, t_base, np2_off, np4_off, best_us,
                int(best.n_per_warp), best.swizzle ? "swz" : "off", int(best.chunk),
                t_base / best_us, bw(best_us));
    (void)np1_off;
}

} // namespace

int main(int argc, char** argv)
{
    int warmup = 5;
    int repeat = 50;
    if(argc > 1) warmup = std::atoi(argv[1]);
    if(argc > 2) repeat = std::atoi(argv[2]);

    const std::vector<Shape> shapes{
        {1, 1024, 7168},
        {1, 2048, 7168},
        {1, 4096, 7168},
        {1, 8192, 7168},
        {2, 4096, 7168},
        {4, 4096, 7168},
        {8, 8192, 7168},
    };

    std::printf("--- gemm_decode A1+A3 sweep: kNPerWarp x chiplet chunk_size, "
                "BF16 unscaled (num_xcds=8) ---\n");
    for(const auto& s : shapes)
        run_shape(s, warmup, repeat);

    return 0;
}
