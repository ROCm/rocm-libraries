// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Example 08: RDNA4 (gfx1201) GEMM Benchmark
 *
 * Demonstrates the dispatcher working with gfx1201-specific kernels:
 *
 *  - FP16 and BF16 GEMM using 16x16x16 WMMA tiles (wave32)
 *  - Multiple tile configs optimized for RDNA4's 128 AI accelerators
 *  - 64KB LDS per workgroup
 *
 * Key differences from CDNA (gfx9):
 *  - Wave32 (not wave64): warp tiles are 16x16x16 (not 32x32x16)
 *  - Valid wave configs: [2,4,1], [4,2,1], [1,8,1], [8,1,1]
 *  - 64KB LDS (gfx942 has 64KB, gfx950 has 160KB)
 *
 * Build: cd dispatcher/build && cmake .. -DGPU_TARGETS=gfx1201 && make gemm_08_gfx1201_rdna4
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/kernel_decl.hpp"
#include "ck_tile/dispatcher/example_args.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using namespace ck_tile::dispatcher::utils;
using Signature = decl::Signature;
using Algorithm = decl::Algorithm;

// =============================================================================
// gfx1201-targeted kernel declarations
//
// RDNA4 WMMA: 16x16x16 warp tiles, wave32
// Valid wave configs: [2,4,1], [4,2,1], [1,8,1], [8,1,1]
// =============================================================================

DECL_KERNEL_SET(gfx1201_gemm_kernels,

                // --- FP16 kernels ---

                // fp16 128x128x32 -- large tile, wave(2,4,1)
                // M-Repeat=128/(2*16)=4, N-Repeat=128/(4*16)=2
                // LDS: 128*32*2 + 128*32*2 = 16KB
                .add(Signature().dtype("fp16").layout("rcr"),
                     Algorithm()
                         .tile(128, 128, 32)
                         .wave(2, 4, 1)
                         .warp(16, 16, 16)
                         .pipeline("compv3")
                         .scheduler("intrawave")
                         .epilogue("cshuffle"),
                     "gfx1201")

                    // fp16 64x64x32 -- smaller tile, wave(2,4,1)
                    // M-Repeat=64/(2*16)=2, N-Repeat=64/(4*16)=1
                    // LDS: 64*32*2 + 64*32*2 = 8KB
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(64, 64, 32)
                             .wave(2, 4, 1)
                             .warp(16, 16, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx1201")

                    // fp16 128x128x64 -- deeper K tile for compute-bound
                    // LDS: 128*64*2 + 128*64*2 = 32KB
                    .add(Signature().dtype("fp16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 64)
                             .wave(2, 4, 1)
                             .warp(16, 16, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx1201")

                    // --- BF16 kernels ---
                    // BF16 was previously BLOCKED on gfx1201 due to arch_filter.hpp bug

                    // bf16 128x128x32 -- same tile config as fp16
                    .add(Signature().dtype("bf16").layout("rcr"),
                         Algorithm()
                             .tile(128, 128, 32)
                             .wave(2, 4, 1)
                             .warp(16, 16, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx1201")

                    // bf16 64x64x32
                    .add(Signature().dtype("bf16").layout("rcr"),
                         Algorithm()
                             .tile(64, 64, 32)
                             .wave(2, 4, 1)
                             .warp(16, 16, 16)
                             .pipeline("compv3")
                             .scheduler("intrawave")
                             .epilogue("cshuffle"),
                         "gfx1201"));

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[])
{
    ExampleArgs args("Example 08: gfx1201 RDNA4 GEMM",
                     "Benchmark GEMM on RDNA4 (RX 9070 XT) with FP16/BF16 WMMA");
    args.add_flag("--list", "List registered kernels");
    args.add_flag("--list-verbose", "List registered kernels with full details");
    args.add_option("--M", "4096", "Problem M dimension");
    args.add_option("--N", "4096", "Problem N dimension");
    args.add_option("--K", "4096", "Problem K dimension");
    args.add_option("--arch", "gfx1201", "GPU architecture");
    args.add_option("--warmup", "10", "Warmup iterations");
    args.add_option("--repeat", "50", "Benchmark iterations");

    if(!args.parse(argc, argv))
        return 0;

    std::string gfx_arch = args.get("--arch", "gfx1201");

    print_header("Example 08: gfx1201 (RDNA4) GEMM Benchmark");

    // =========================================================================
    // Architecture info
    // =========================================================================
    std::cout << "\ngfx1201 (RDNA4 / RX 9070 XT) highlights:\n";
    std::cout << "  - 128 AI Accelerators (WMMA units)\n";
    std::cout << "  - Wave32 (not wave64): warp tiles 16x16x16\n";
    std::cout << "  - 64KB LDS per workgroup\n";
    std::cout << "  - 64 CUs, ~605 GB/s VRAM bandwidth\n";
    std::cout << "  - FP16/BF16/FP8 WMMA support\n";
    std::cout << "  - Valid wave configs: [2,4,1], [4,2,1], [1,8,1], [8,1,1]\n\n";

    // =========================================================================
    // Register kernels
    // =========================================================================
    std::cout << "Registering kernels for " << gfx_arch << "...\n";

    Registry registry;
    registry.set_name("gfx1201_gemm");
    REGISTER_GENERATED_KERNELS(registry, gfx_arch);

    std::cout << "  Registered " << registry.size() << " kernel(s)\n";

    if(args.has("--list") || args.has("--list-verbose"))
    {
        std::cout << "\n";
        print_registered_kernels(registry, std::cout, args.has("--list-verbose"));
        return 0;
    }

    if(registry.size() == 0)
    {
        std::cerr << "ERROR: No kernels registered for " << gfx_arch << "!\n";
        std::cerr << "  Did you build with -DGPU_TARGETS=gfx1201?\n";
        return 1;
    }

    // =========================================================================
    // Benchmark
    // =========================================================================
    Dispatcher dispatcher(&registry);

    const int M      = args.get_int("--M", 4096);
    const int N      = args.get_int("--N", 4096);
    const int K      = args.get_int("--K", 4096);
    int warmup       = args.get_int("--warmup", 10);
    int repeat       = args.get_int("--repeat", 50);

    std::cout << "\nProblem: " << M << " x " << N << " x " << K << "\n";

    Problem problem(M, N, K);

    using DataType = ck_tile::fp16_t;
    GpuBuffer<DataType> a_dev(M * K);
    GpuBuffer<DataType> b_dev(K * N);
    GpuBuffer<DataType> c_dev(M * N);

    std::vector<DataType> a_host(M * K, DataType(0.01f));
    std::vector<DataType> b_host(K * N, DataType(0.01f));
    a_dev.copy_from_host(a_host.data());
    b_dev.copy_from_host(b_host.data());
    c_dev.zero();

    // =========================================================================
    // Benchmark ALL registered kernels
    // =========================================================================
    double flops   = 2.0 * M * N * K;
    float best_t   = 1e9f;
    std::string best_name;
    bool all_passed = true;

    auto all_kernels = registry.get_all_instances();
    std::cout << "\nBenchmarking " << all_kernels.size() << " kernel(s)...\n";

    for(size_t ki = 0; ki < all_kernels.size(); ++ki)
    {
        const auto& kernel = all_kernels[ki];
        const auto& name   = kernel->get_name();

        // Skip BF16 kernels (we allocated FP16 buffers)
        if(name.find("bf16") != std::string::npos)
            continue;

        print_separator();
        std::cout << "[" << (ki + 1) << "/" << all_kernels.size() << "] " << name << "\n";

        c_dev.zero();

        // Warmup
        bool launch_ok = true;
        for(int i = 0; i < warmup; ++i)
        {
            try {
                (void)dispatcher.run_explicit(name,
                    a_dev.get(), b_dev.get(), c_dev.get(), nullptr, problem, nullptr);
            } catch(...) { launch_ok = false; break; }
        }
        if(!launch_ok)
        {
            std::cout << "  SKIP (launch failed)\n";
            continue;
        }

        // Benchmark
        std::vector<float> times;
        times.reserve(repeat);
        for(int i = 0; i < repeat; ++i)
        {
            float t = dispatcher.run_explicit(name,
                          a_dev.get(), b_dev.get(), c_dev.get(), nullptr, problem, nullptr);
            times.push_back(t);
        }

        std::sort(times.begin(), times.end());
        float min_t    = times.front();
        float median_t = times[times.size() / 2];
        float mean_t   = std::accumulate(times.begin(), times.end(), 0.0f)
                         / static_cast<float>(times.size());

        double tflops_peak   = flops / (min_t * 1e9);
        double tflops_median = flops / (median_t * 1e9);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Min:    " << min_t << " ms  ("
                  << std::setprecision(2) << tflops_peak << " TFLOPS)\n";
        std::cout << std::setprecision(4);
        std::cout << "  Mean:   " << mean_t << " ms\n";
        std::cout << "  Median: " << median_t << " ms  ("
                  << std::setprecision(2) << tflops_median << " TFLOPS)\n";
        std::cout << "  Efficiency: " << std::setprecision(1)
                  << (100.0 * tflops_peak / 195.0) << "% of WMMA peak\n";

        if(min_t < best_t) { best_t = min_t; best_name = name; }

        // Verification
        std::vector<DataType> c_host(M * N);
        c_dev.copy_to_host(c_host.data());
        const float expected = static_cast<float>(K) * 0.01f * 0.01f;
        int errors = 0;
        for(int i = 0; i < std::min(M * N, 1024); ++i)
        {
            float val = static_cast<float>(c_host[i]);
            if(std::abs(val - expected) > 0.1f * std::abs(expected) + 0.01f)
                ++errors;
        }
        std::cout << "  Verify: " << (errors == 0 ? "PASS" : "FAIL")
                  << " (errors=" << errors << ")\n";
        if(errors > 0) all_passed = false;
    }

    // =========================================================================
    // Summary
    // =========================================================================
    print_separator();
    std::cout << "gfx1201 RDNA4 GEMM Summary (" << M << "x" << N << "x" << K << "):\n";
    std::cout << "  Best kernel: " << best_name << "\n";
    std::cout << std::setprecision(2);
    std::cout << "  Peak:        " << (flops / (best_t * 1e9)) << " TFLOPS\n";
    std::cout << "  Efficiency:  " << std::setprecision(1)
              << (100.0 * (flops / (best_t * 1e9)) / 195.0) << "% of WMMA peak\n";
    print_separator();

    return all_passed ? 0 : 1;
}
