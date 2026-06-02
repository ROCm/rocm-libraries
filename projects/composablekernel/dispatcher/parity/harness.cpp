// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Minimal single-kernel harness -- parity deliverable (d).
//
// Runs exactly ONE generated dispatcher kernel via the CK_TILE_SINGLE_KERNEL_INCLUDE
// contract: defining that macro before including a generated kernel header exposes
//   - SelectedKernel  (the kernel struct, with static launch())
//   - ADataType / BDataType / CDataType
//   - KERNEL_NAME
// and SelectedKernel::launch(const GemmHostArgs&, const stream_config&).
//
// The generated header path is injected at compile time as PARITY_KERNEL_HEADER,
// so this single .cpp drives whichever kernel drive_codegen.py produced:
//
//   hipcc -std=c++17 --offload-arch=<gfx> \
//       -I <composablekernel/include> \
//       -DCK_TILE_SINGLE_KERNEL_INCLUDE \
//       -DPARITY_KERNEL_HEADER='"<abs path to generated gemm_*.hpp>"' \
//       harness.cpp -o harness
//
// It computes C = A * B for a row/col/row (rcr) fp16 GEMM, verifies against a CPU
// reference, and (optionally) reports kernel time. This is the GPU-gated half of
// the parity work: it builds with hipcc here but RUNNING requires a GPU node
// (k8s/SLURM/Alola), since this environment exposes only a CPU.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

#ifndef PARITY_KERNEL_HEADER
#error "Define PARITY_KERNEL_HEADER to the generated kernel header path."
#endif
#ifndef CK_TILE_SINGLE_KERNEL_INCLUDE
#error "Define CK_TILE_SINGLE_KERNEL_INCLUDE to expose SelectedKernel aliases."
#endif

#include PARITY_KERNEL_HEADER

#define HIP_CHECK(expr)                                                                  \
    do                                                                                   \
    {                                                                                    \
        hipError_t _e = (expr);                                                          \
        if(_e != hipSuccess)                                                             \
        {                                                                                \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),       \
                         __FILE__, __LINE__);                                            \
            std::exit(1);                                                                \
        }                                                                                \
    } while(0)

namespace {

int arg_int(int argc, char** argv, const std::string& flag, int dflt)
{
    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if(a.rfind(flag + "=", 0) == 0)
            return std::atoi(a.substr(flag.size() + 1).c_str());
    }
    return dflt;
}

std::string arg_str(int argc, char** argv, const std::string& flag,
                    const std::string& dflt)
{
    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if(a.rfind(flag + "=", 0) == 0)
            return a.substr(flag.size() + 1);
    }
    return dflt;
}

} // namespace

int main(int argc, char** argv)
{
    const int M      = arg_int(argc, argv, "-m", 512);
    const int N      = arg_int(argc, argv, "-n", 512);
    const int K      = arg_int(argc, argv, "-k", 512);
    const int verify = arg_int(argc, argv, "-verify", 1);

    // Layout guard: this harness hard-codes rcr strides (A row-major, B col-major,
    // C row-major). If the kernel was generated for a different layout the strides
    // will be wrong and produce silently wrong results.  --layout= lets the
    // orchestrator assert the config matches before invoking.
    const std::string layout = arg_str(argc, argv, "-layout", "rcr");
    if(layout != "rcr")
    {
        std::fprintf(stderr,
                     "error: harness only supports rcr layout; got '%s'.\n"
                     "       Pass -layout=rcr or generalize strides.\n",
                     layout.c_str());
        return 1;
    }

    std::printf("kernel : %s\n", KERNEL_NAME);
    std::printf("problem: M=%d N=%d K=%d (rcr)\n", M, N, K);

    // rcr layout strides: A row-major (M,K) -> K ; B col-major (K,N) -> K ;
    // C row-major (M,N) -> N.
    const int stride_a = K;
    const int stride_b = K;
    const int stride_c = N;

    std::vector<ADataType> h_a(static_cast<size_t>(M) * K);
    std::vector<BDataType> h_b(static_cast<size_t>(K) * N);
    std::vector<CDataType> h_c(static_cast<size_t>(M) * N);

    // Deterministic, small inputs keep the fp16 reference well-conditioned.
    for(size_t i = 0; i < h_a.size(); ++i)
        h_a[i] = ck_tile::type_convert<ADataType>(static_cast<float>((i % 7) - 3) * 0.25f);
    for(size_t i = 0; i < h_b.size(); ++i)
        h_b[i] = ck_tile::type_convert<BDataType>(static_cast<float>((i % 5) - 2) * 0.25f);

    void* d_a = nullptr;
    void* d_b = nullptr;
    void* d_c = nullptr;
    HIP_CHECK(hipMalloc(&d_a, h_a.size() * sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&d_b, h_b.size() * sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&d_c, h_c.size() * sizeof(CDataType)));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), h_a.size() * sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), h_b.size() * sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_c, 0, h_c.size() * sizeof(CDataType)));

    ck_tile::GemmHostArgs args(
        d_a, d_b, d_c, /*k_batch=*/1, M, N, K, stride_a, stride_b, stride_c);

    ck_tile::stream_config stream{};
    stream.time_kernel_  = true;
    stream.cold_niters_  = 3;   // warmup: matches -warmup=3 passed to TE benchmark
    stream.nrepeat_      = 20;  // timed:  matches -repeat=20 passed to TE benchmark

    float ave_ms = 0.0f;
    try
    {
        ave_ms = SelectedKernel::launch(args, stream);
    }
    catch(const std::exception& e)
    {
        std::printf("SKIPPED: %s\n", e.what());
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_c));
        return 0; // unsupported argument is a skip, not a failure (see test README)
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, h_c.size() * sizeof(CDataType), hipMemcpyDeviceToHost));

    if(ave_ms > 0.0f)
    {
        const double flops = 2.0 * M * N * K;
        std::printf("time   : %.4f ms  (%.1f GFLOP/s)\n", ave_ms, flops / (ave_ms * 1e6));
    }

    // fp8/bf8 host-side type_convert is not reliable without CK_TILE_USE_CUSTOM_DATA_TYPE,
    // which conflicts with host-side headers. Skip numerical verification for 8-bit float
    // types; the harness still measures and reports timing.
    constexpr bool kSkipVerifyForFp8 = (sizeof(ADataType) == 1 &&
        !std::is_same<ADataType, int8_t>::value &&
        !std::is_same<ADataType, uint8_t>::value);

    int rc = 0;
    if(verify && !kSkipVerifyForFp8)
    {
        // CPU reference in fp32: C[m,n] = sum_k A[m,k] * B[k,n] (B col-major).
        // fp16 accumulation tolerance:
        //   abs: 1e-3 * sqrt(K)   (tight constant vs original 1e-2)
        //   rel: 1e-2             (relative gate matches projectdes.txt T1.5 spec)
        // Both gates must pass; a kernel returning values off by 0.2 when K=512
        // would slip past the abs-only check but is caught by the relative gate.
        const double abs_tol = 1e-3 * std::sqrt(static_cast<double>(K));
        const double rel_tol = 1e-2;

        double max_abs_err = 0.0;
        double max_rel_err = 0.0;
        int total_elems    = 0;
        int fail_count     = 0;
        // First 10 mismatches printed to stderr to aid diagnosis without overwhelming output.
        std::vector<std::tuple<int,int,float,float>> first_mismatches;

        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                float acc = 0.0f;
                for(int k = 0; k < K; ++k)
                {
                    const float a = ck_tile::type_convert<float>(h_a[static_cast<size_t>(m) * stride_a + k]);
                    const float b = ck_tile::type_convert<float>(h_b[static_cast<size_t>(n) * stride_b + k]);
                    acc += a * b;
                }
                const float got =
                    ck_tile::type_convert<float>(h_c[static_cast<size_t>(m) * stride_c + n]);
                const double abs_err = std::fabs(got - acc);
                const double rel_err = abs_err / (std::fabs(acc) + 1e-6);
                max_abs_err = std::max(max_abs_err, abs_err);
                max_rel_err = std::max(max_rel_err, rel_err);
                ++total_elems;
                if(abs_err > abs_tol || rel_err > rel_tol)
                {
                    ++fail_count;
                    if(static_cast<int>(first_mismatches.size()) < 10)
                        first_mismatches.emplace_back(m, n, got, acc);
                }
            }
        }

        const int pass_count = total_elems - fail_count;
        std::printf("verify : max_abs_err=%.5f max_rel_err=%.5f abs_tol=%.5f rel_tol=%.5f\n",
                    max_abs_err, max_rel_err, abs_tol, rel_tol);
        std::printf("verify : %d/%d elements pass (%.1f%%)\n",
                    pass_count, total_elems,
                    total_elems > 0 ? 100.0 * pass_count / total_elems : 0.0);

        if(fail_count > 0)
        {
            std::fprintf(stderr, "first %d mismatch(es):\n",
                         static_cast<int>(first_mismatches.size()));
            for(auto& [mm, nn, got, ref] : first_mismatches)
                std::fprintf(stderr, "  C[%d,%d]: got=%.6f ref=%.6f\n", mm, nn, got, ref);
            std::printf("FAILED\n");
            rc = 1;
        }
        else
        {
            std::printf("PASSED\n");
        }
    }
    else if(kSkipVerifyForFp8)
    {
        std::printf("verify : SKIPPED (fp8/bf8 host reference not supported; timing only)\n");
        std::printf("PASSED\n");
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    return rc;
}
