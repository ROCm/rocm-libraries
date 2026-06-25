// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// cjit_conv_demo.cpp -- end-to-end verification of the C-engine JIT path for
// op 'conv' on a real gfx950 GPU, using the GENERALIZED ck_dsl::CEngine +
// ck_dsl::Kernel (the exact path CkDslConvPlan now uses):
//
//   CEngine::build_conv(POD)                       // C engine: .ll gen (TIMED, ms)
//     -> Kernel::from_llvm_ir(ll, manifest, isa)
//     -> k.ensure_compiled()                       // comgr .ll->HSACO (TIMED, ms)
//     -> set up device buffers + launch            // warm per-launch (TIMED, us)
//     -> verify vs CPU NHWC conv reference (small shape, bad==0 within tol)
//
// ABI (recon): A,B,D : ptr<f16,global> ; A_bytes,B_bytes,D_bytes : i32.
//   A = NHWC input [N,Hi,Wi,C]; B = KRSC weights [K,R,S,C]; D = NHWK output
//   [N,Ho,Wo,K]. M = N*Ho*Wo, N_gemm = K, K_gemm = R*S*C. grid_order "NM".
//
// Build:
//   hipcc -std=c++17 \
//     -I .../ck_dsl_c/include -I .../runtime/include -I /opt/rocm/include \
//     cjit_conv_demo.cpp -L<libdir> -lckc_core -lamd_comgr -o /tmp/cjit_conv

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "ck_dsl_runtime/c_engine.hpp"
#include "ck_dsl_runtime/kernel.hpp"

using fp16 = _Float16;
using namespace ck_dsl;

static double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    const char* arch = argc > 1 ? argv[1] : "gfx950";

    // ---- small valid conv shape (geometry baked at build) -----------------
    CEngine::ConvProblem cp;
    cp.N = 2;
    cp.Hi = 14;
    cp.Wi = 14;
    cp.C = 32;
    cp.K = 64;
    cp.R = 3;
    cp.S = 3;
    cp.sH = cp.sW = 1;
    cp.pH = cp.pW = 1;
    cp.dH = cp.dW = 1;
    cp.arch = arch;

    const int Ho = (cp.Hi + 2 * cp.pH - cp.dH * (cp.R - 1) - 1) / cp.sH + 1;
    const int Wo = (cp.Wi + 2 * cp.pW - cp.dW * (cp.S - 1) - 1) / cp.sW + 1;
    const long M = (long)cp.N * Ho * Wo;
    const int N_gemm = cp.K;
    const int K_gemm = cp.R * cp.S * cp.C;

    std::printf(
        "[conv] shape N%d H%d W%d C%d K%d R%d S%d  Ho=%d Wo=%d  M=%ld N_gemm=%d K_gemm=%d\n", cp.N,
        cp.Hi, cp.Wi, cp.C, cp.K, cp.R, cp.S, Ho, Wo, M, N_gemm, K_gemm);

    // ---- (1) C-engine .ll generation (TIMED, ms) --------------------------
    double buildMs = 0;
    CEngineResult r;
    try {
        auto t0 = std::chrono::high_resolution_clock::now();
        r = CEngine::build_conv(cp);
        buildMs = ms_since(t0);
    } catch (const std::exception& e) {
        std::printf("[conv] build_conv FAILED: %s\n", e.what());
        return 2;
    }
    std::printf(
        "[conv] kernel=%s  ll=%zu bytes  block=%u  grid={%u,%u,%u}  "
        "block_m/n/k=%d/%d/%d  sig_has_bytes=%d  buildMs=%.3f\n",
        r.manifest.kernel_name.c_str(), r.llvm_ir.size(), r.block, r.grid[0], r.grid[1], r.grid[2],
        r.manifest.block_m, r.manifest.block_n, r.manifest.block_k,
        r.manifest.sig_has_bytes ? 1 : 0, buildMs);

    // ---- (2) comgr .ll -> HSACO (TIMED, ms) -------------------------------
    Kernel k = Kernel::from_llvm_ir(r.llvm_ir, r.manifest, Compiler::isa_for(arch));
    double compileMs = 0;
    try {
        auto t0 = std::chrono::high_resolution_clock::now();
        k.ensure_compiled();
        compileMs = ms_since(t0);
    } catch (const std::exception& e) {
        std::printf("[conv] ensure_compiled FAILED: %s\n", e.what());
        return 3;
    }
    std::printf("[conv] compiled: hsaco=%zu bytes  compileMs=%.3f\n", k.hsaco().size(), compileMs);

    // ---- host operands ----------------------------------------------------
    std::vector<fp16> A((size_t)cp.N * cp.Hi * cp.Wi * cp.C);     // NHWC
    std::vector<fp16> B((size_t)cp.K * cp.R * cp.S * cp.C);       // KRSC
    std::vector<fp16> D((size_t)cp.N * Ho * Wo * cp.K, fp16(0));  // NHWK
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (auto& v : A) v = (fp16)std::round(dist(rng));
    for (auto& v : B) v = (fp16)std::round(dist(rng));

    // ---- device buffers (byte sizes match recon a/b/d_bytes ABI) ----------
    const uint64_t a_bytes = (uint64_t)cp.N * cp.Hi * cp.Wi * cp.C * sizeof(fp16);
    const uint64_t b_bytes = (uint64_t)cp.K * cp.R * cp.S * cp.C * sizeof(fp16);
    const uint64_t d_bytes = (uint64_t)cp.N * Ho * Wo * cp.K * sizeof(fp16);

    fp16 *Ad, *Bd, *Dd;
    hip_check(hipMalloc(&Ad, a_bytes), "malloc A");
    hip_check(hipMalloc(&Bd, b_bytes), "malloc B");
    hip_check(hipMalloc(&Dd, d_bytes), "malloc D");
    hip_check(hipMemcpy(Ad, A.data(), a_bytes, hipMemcpyHostToDevice), "h2d A");
    hip_check(hipMemcpy(Bd, B.data(), b_bytes, hipMemcpyHostToDevice), "h2d B");
    hip_check(hipMemset(Dd, 0, d_bytes), "memset D");

    auto ptr_args = std::unordered_map<std::string, void*>{{"A", Ad}, {"B", Bd}, {"D", Dd}};
    auto scalar_args = std::unordered_map<std::string, uint64_t>{
        {"A_bytes", a_bytes}, {"B_bytes", b_bytes}, {"D_bytes", d_bytes}};

    // ---- launch once for correctness --------------------------------------
    try {
        k.launch(ptr_args, scalar_args, r.grid, r.block);
        hip_check(hipDeviceSynchronize(), "sync(correctness)");
    } catch (const std::exception& e) {
        std::printf("[conv] launch FAILED: %s\n", e.what());
        hipFree(Ad);
        hipFree(Bd);
        hipFree(Dd);
        return 4;
    }
    hip_check(hipMemcpy(D.data(), Dd, d_bytes, hipMemcpyDeviceToHost), "d2h D");

    // ---- CPU NHWC conv reference ------------------------------------------
    // D[n,ho,wo,k] = sum_{r,s,c} A[n, ho*sH - pH + r*dH, wo*sW - pW + s*dW, c]
    //                            * B[k,r,s,c]    (zero-padded boundary)
    auto Aidx = [&](int n, int h, int w, int c) {
        return (((size_t)n * cp.Hi + h) * cp.Wi + w) * cp.C + c;
    };
    auto Bidx = [&](int kk, int rr, int ss, int c) {
        return (((size_t)kk * cp.R + rr) * cp.S + ss) * cp.C + c;
    };
    auto Didx = [&](int n, int ho, int wo, int kk) {
        return (((size_t)n * Ho + ho) * Wo + wo) * cp.K + kk;
    };

    double worst = 0;
    int bad = 0;
    const float tol = 2.0f;  // fp16 accumulation slack over K_gemm taps
    for (int n = 0; n < cp.N; ++n)
        for (int ho = 0; ho < Ho; ++ho)
            for (int wo = 0; wo < Wo; ++wo)
                for (int kk = 0; kk < cp.K; ++kk) {
                    float acc = 0;
                    for (int rr = 0; rr < cp.R; ++rr) {
                        int hi = ho * cp.sH - cp.pH + rr * cp.dH;
                        if (hi < 0 || hi >= cp.Hi) continue;
                        for (int ss = 0; ss < cp.S; ++ss) {
                            int wi = wo * cp.sW - cp.pW + ss * cp.dW;
                            if (wi < 0 || wi >= cp.Wi) continue;
                            for (int c = 0; c < cp.C; ++c)
                                acc += (float)A[Aidx(n, hi, wi, c)] * (float)B[Bidx(kk, rr, ss, c)];
                        }
                    }
                    float got = (float)D[Didx(n, ho, wo, kk)];
                    float diff = std::fabs(got - acc);
                    worst = std::max(worst, (double)diff);
                    if (diff > tol) {
                        if (bad < 8)
                            std::printf("    mismatch [n%d ho%d wo%d k%d] got=%g ref=%g diff=%g\n",
                                        n, ho, wo, kk, got, acc, diff);
                        ++bad;
                    }
                }
    const long total = M * cp.K;
    bool correct = (bad == 0);
    std::printf("[conv] verify: max_abs_diff=%g bad=%d/%ld tol=%g  %s\n", worst, bad, total, tol,
                correct ? "PASS" : "FAIL");

    // ---- warm per-launch timing (>=50 iters, hipEventElapsedTime us) ------
    double launchUs = 0;
    if (correct) {
        hipEvent_t e0, e1;
        hip_check(hipEventCreate(&e0), "event0");
        hip_check(hipEventCreate(&e1), "event1");
        const int warmup = 10, iters = 100;
        for (int i = 0; i < warmup; ++i) k.launch(ptr_args, scalar_args, r.grid, r.block);
        hip_check(hipDeviceSynchronize(), "sync(warmup)");
        hip_check(hipEventRecord(e0, nullptr), "rec e0");
        for (int i = 0; i < iters; ++i) k.launch(ptr_args, scalar_args, r.grid, r.block);
        hip_check(hipEventRecord(e1, nullptr), "rec e1");
        hip_check(hipEventSynchronize(e1), "sync e1");
        float total_ms = 0;
        hip_check(hipEventElapsedTime(&total_ms, e0, e1), "elapsed");
        launchUs = (double)total_ms * 1000.0 / iters;
        hipEventDestroy(e0);
        hipEventDestroy(e1);
        std::printf("[conv] warm per-launch: %.3f us over %d iters\n", launchUs, iters);
    }

    hipFree(Ad);
    hipFree(Bd);
    hipFree(Dd);

    // ---- machine-readable summary line for the harness --------------------
    std::printf("RESULT op=conv correct=%d buildMs=%.3f compileMs=%.3f launchUs=%.3f bad=%d/%ld\n",
                correct ? 1 : 0, buildMs, compileMs, launchUs, bad, total);
    return correct ? 0 : 1;
}
