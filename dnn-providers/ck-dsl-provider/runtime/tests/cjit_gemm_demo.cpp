// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// cjit_gemm_demo.cpp -- end-to-end verification of the C-engine JIT path for
// op 'gemm' on a real GPU, using the SAME generalized ck_dsl::CEngine +
// ck_dsl::Kernel path that CkDslGemmPlan now uses.
//
//   CEngine::build_gemm(POD)            -> .ll text + Manifest + grid + block   [TIME: buildMs]
//   Kernel::from_llvm_ir(...)           -> stage-5 kernel object
//   kernel.ensure_compiled()            -> comgr .ll -> HSACO + module load     [TIME: compileMs]
//   device buffers + warm launch loop   -> hipEventElapsedTime over >=50 iters  [TIME: launchUs]
//   verify vs CPU reference             -> integer-exact (small int inputs)
//
// Build:
//   hipcc -std=c++17 \
//     -I /workspace/rocm-libraries/projects/composablekernel/python/ck_dsl_c/include \
//     -I /workspace/rocm-lib-copy/dnn-providers/ck-dsl-provider/runtime/include \
//     -I /opt/rocm/include \
//     cjit_gemm_demo.cpp -L<libdir> -lckc_core -lamd_comgr -o /tmp/cjit_gemm

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "ck_dsl_runtime/c_engine.hpp"
#include "ck_dsl_runtime/comgr.hpp"
#include "ck_dsl_runtime/kernel.hpp"

using namespace ck_dsl;

// Minimal half<->float helpers (avoid pulling hip fp16 device headers on host).
static uint16_t f2h(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;  // flush subnormals (inputs are small ints)
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}
static float h2f(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        out = sign;  // treat subnormal/zero as zero
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    __builtin_memcpy(&f, &out, 4);
    return f;
}

static void hck(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error %s: %s\n", w, hipGetErrorString(e));
        std::exit(2);
    }
}

int main() {
    // ---- small valid shape: a multiple of the 128x128x32 tile geometry ----
    CEngine::GemmProblem p;
    p.M = 256;
    p.N = 256;
    p.K = 256;
    // demo-proven combo (matches CEngine self-test): compv3 + default epilogue.
    p.pipeline = "compv3";
    p.epilogue = "default";
    p.arch = "gfx950";

    // ===================== TIME 1: .ll generation (buildMs) =====================
    auto t0 = std::chrono::high_resolution_clock::now();
    CEngineResult r = CEngine::build_gemm(p);
    auto t1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf(
        "[build] kernel=%s  ll=%zu bytes  block=%u  grid={%u,%u,%u}  block_m/n/k=%d/%d/%d\n",
        r.manifest.kernel_name.c_str(), r.llvm_ir.size(), r.block, r.grid[0], r.grid[1], r.grid[2],
        r.manifest.block_m, r.manifest.block_n, r.manifest.block_k);
    std::printf("[build] .ll gen time = %.3f ms\n", buildMs);

    // ===================== build the Kernel object (path b) =====================
    std::string isa = Compiler::isa_for(p.arch);  // "amdgcn-amd-amdhsa--gfx950"
    Kernel kernel = Kernel::from_llvm_ir(r.llvm_ir, r.manifest, isa);

    // ===================== TIME 2: comgr compile (compileMs) =====================
    auto c0 = std::chrono::high_resolution_clock::now();
    kernel.ensure_compiled();  // comgr .ll -> HSACO + hipModuleLoad + getFunction
    auto c1 = std::chrono::high_resolution_clock::now();
    double compileMs = std::chrono::duration<double, std::milli>(c1 - c0).count();
    std::printf("[compile] comgr+load time = %.3f ms  (hsaco=%zu bytes)\n", compileMs,
                kernel.hsaco().size());

    // ===================== host data: small integer inputs =====================
    const int M = p.M, N = p.N, K = p.K;
    // RCR layout: A row-major [M,K]; B col-major (stored [N,K] -> B[n,k]); C row-major [M,N].
    std::vector<uint16_t> hA((size_t)M * K), hB((size_t)N * K), hC((size_t)M * N, 0);
    auto rnd = [](int i) { return (float)((i * 1103515245u + 12345u) >> 28 & 0x7); };  // 0..7
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = f2h(rnd((int)i));
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = f2h(rnd((int)(i + 7)));

    // CPU reference (integer-exact: products of small ints, sum < 2^24 fp16-exact
    // in fp32 accum; max term 7*7*256 = 12544 << 2^24).
    std::vector<float> ref((size_t)M * N, 0.f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += h2f(hA[(size_t)m * K + k]) * h2f(hB[(size_t)n * K + k]);
            ref[(size_t)m * N + n] = acc;
        }

    // ===================== device buffers + launch =====================
    void *dA, *dB, *dC;
    hck(hipMalloc(&dA, hA.size() * 2), "malloc A");
    hck(hipMalloc(&dB, hB.size() * 2), "malloc B");
    hck(hipMalloc(&dC, hC.size() * 2), "malloc C");
    hck(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice), "H2D A");
    hck(hipMemcpy(dB, hB.data(), hB.size() * 2, hipMemcpyHostToDevice), "H2D B");
    hck(hipMemset(dC, 0, hC.size() * 2), "memset C");

    std::unordered_map<std::string, void*> ptr_args = {{"A", dA}, {"B", dB}, {"C", dC}};
    std::unordered_map<std::string, uint64_t> scalar_args = {
        {"M", (uint64_t)M}, {"N", (uint64_t)N}, {"K", (uint64_t)K}};

    // warmup + functional launch
    kernel.launch(ptr_args, scalar_args, r.grid, r.block);
    hck(hipDeviceSynchronize(), "sync warmup");

    // copy result back and verify
    hck(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost), "D2H C");
    long bad = 0;
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < hC.size(); ++i) {
        float got = h2f(hC[i]);
        float exp = ref[i];
        double d = std::fabs((double)got - (double)exp);
        if (d > max_abs_diff) max_abs_diff = d;
        // fp16 output: exact for integers up to 2048; above that fp16 quantizes.
        // tolerance = 0.5 ULP of the fp16 value (relative ~2^-11).
        double tol = std::fabs((double)exp) * (1.0 / 1024.0) + 0.5;
        if (d > tol) {
            if (bad < 8)
                std::printf("  diff @%zu: got=%.1f exp=%.1f d=%.3f tol=%.3f\n", i, got, exp, d,
                            tol);
            ++bad;
        }
    }
    bool correct = (bad == 0);
    std::printf("[verify] bad=%ld / %zu   max_abs_diff=%.4f   %s\n", bad, hC.size(), max_abs_diff,
                correct ? "CORRECT" : "INCORRECT");

    // ===================== TIME 3: warm per-launch (launchUs) =====================
    hipEvent_t ev0, ev1;
    hck(hipEventCreate(&ev0), "ev0");
    hck(hipEventCreate(&ev1), "ev1");
    const int iters = 100;
    // a few warmups
    for (int i = 0; i < 10; ++i) kernel.launch(ptr_args, scalar_args, r.grid, r.block);
    hck(hipDeviceSynchronize(), "sync pre-timing");

    hck(hipEventRecord(ev0, nullptr), "rec0");
    for (int i = 0; i < iters; ++i) kernel.launch(ptr_args, scalar_args, r.grid, r.block);
    hck(hipEventRecord(ev1, nullptr), "rec1");
    hck(hipEventSynchronize(ev1), "ev sync");
    float total_ms = 0.f;
    hck(hipEventElapsedTime(&total_ms, ev0, ev1), "elapsed");
    double launchUs = (double)total_ms / iters * 1000.0;
    std::printf("[launch] warm per-launch = %.3f us  (%d iters, total %.3f ms)\n", launchUs, iters,
                total_ms);

    std::printf("\nRESULT op=gemm correct=%d buildMs=%.3f compileMs=%.3f launchUs=%.3f\n",
                correct ? 1 : 0, buildMs, compileMs, launchUs);

    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    return correct ? 0 : 1;
}
