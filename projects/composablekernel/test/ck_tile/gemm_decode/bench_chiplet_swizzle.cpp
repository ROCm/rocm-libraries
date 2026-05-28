// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Standalone microbench: time GemmDecodeUniversalKernel with and without
// the XCD-aware workgroup swizzle on the same inputs, across a small
// shape grid. Build target: bench_gemm_decode_chiplet_swizzle.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
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

template <bool kChipletSwizzle, index_t kChipletNumXcds, index_t kChipletChunk>
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
                                        /*kNPerWarp=*/1,
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

    // Warmup.
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

    const float t_off =
        bench_one</*kChipletSwizzle=*/false, 8, 8>(s, warmup, repeat, a_buf, b_buf, c_buf);

    struct Cfg { index_t num_xcds; index_t chunk; };
    const std::vector<Cfg> cfgs{
        {8, 4}, {8, 8}, {8, 16}, {8, 32}, {8, 64},
    };

    std::printf("M=%4d N=%5d K=%5d  off=%7.2f us",
                s.M, s.N, s.K, t_off * 1000.0f);

    // num_xcds is compile-time, so we iterate via if/else on chunk.
    auto run_cfg = [&](index_t chunk_size, float t) {
        const float bw   = 2.0f * float(s.M) * float(s.N) * float(s.K) /
                           (t * 1.0e-3f) / 1.0e12f;
        const float spd  = t_off / t;
        std::printf("   chunk=%2d  on=%7.2f us  spd=%5.3fx  (%5.2f TF/s)",
                    int(chunk_size), t * 1000.0f, spd, bw);
    };

    for(const auto& cfg : cfgs)
    {
        float t = 0.0f;
        if(cfg.chunk == 4)
            t = bench_one<true, 8, 4>(s, warmup, repeat, a_buf, b_buf, c_buf);
        else if(cfg.chunk == 8)
            t = bench_one<true, 8, 8>(s, warmup, repeat, a_buf, b_buf, c_buf);
        else if(cfg.chunk == 16)
            t = bench_one<true, 8, 16>(s, warmup, repeat, a_buf, b_buf, c_buf);
        else if(cfg.chunk == 32)
            t = bench_one<true, 8, 32>(s, warmup, repeat, a_buf, b_buf, c_buf);
        else if(cfg.chunk == 64)
            t = bench_one<true, 8, 64>(s, warmup, repeat, a_buf, b_buf, c_buf);
        run_cfg(cfg.chunk, t);
    }
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv)
{
    int warmup = 5;
    int repeat = 50;
    if(argc > 1) warmup = std::atoi(argv[1]);
    if(argc > 2) repeat = std::atoi(argv[2]);

    const std::vector<Shape> shapes{
        // (1, N, 7168) sweep across N to span chunk*num_xcds boundaries.
        {1, 1024, 7168},
        {1, 2048, 7168},
        {1, 4096, 7168},
        {1, 8192, 7168},
        // Multi-row M to exercise (m, n_block) flatten/unflatten.
        {2, 4096, 7168},
        {4, 4096, 7168},
    };

    std::printf("--- gemm_decode chiplet swizzle bench, BF16 unscaled "
                "(num_xcds=8 fixed) ---\n");
    for(const auto& s : shapes)
        run_shape(s, warmup, repeat);

    return 0;
}
