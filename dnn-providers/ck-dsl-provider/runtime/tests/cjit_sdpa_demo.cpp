// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// cjit_sdpa_demo.cpp -- end-to-end verification of the GENERALIZED C-engine JIT
// path for op 'sdpa' on the GPU, using the SAME ck_dsl::CEngine + ck_dsl::Kernel
// objects the provider's CkDslAttnPlan uses.
//
//   CEngine::build_sdpa(POD)   -> .ll text + synthesized Manifest + grid/block   (TIME: buildMs)
//   Kernel::from_llvm_ir(...)  -> lazy comgr-compiled kernel
//   k.ensure_compiled()        -> comgr .ll -> HSACO                              (TIME: compileMs)
//   device buffers + k.launch  -> warm per-launch over >=50 iters (hipEvent)     (TIME: launchUs)
//   CPU causal-attention oracle (B=1, fp16) for the verify (within tol)          (correct)
//
// Build:
//   hipcc -std=c++17 \
//     -I <repo>/projects/composablekernel/python/ck_dsl_c/include \
//     -I <provider>/runtime/include -I /opt/rocm/include \
//     cjit_sdpa_demo.cpp -L<libdir> -lckc_core -lamd_comgr -o /tmp/cjit_sdpa
//
// The 18-arg paged-KV ABI (RFC D.5) is fed via the engine-synthesized manifest
// signature; dense BSHD K/V are byte-compatible with the kernel's paged cache
// when S is a multiple of block_size (then S == max_blocks*block_size), with a
// contiguous block_table + single-seq cu_q -- exactly what CkDslAttnPlan does.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "ck_dsl_runtime/c_engine.hpp"
#include "ck_dsl_runtime/kernel.hpp"

using fp16 = __half;
using namespace ck_dsl;

#define HIP_CHECK(e)                                                               \
    do {                                                                           \
        hipError_t _e = (e);                                                       \
        if (_e != hipSuccess) {                                                    \
            std::fprintf(stderr, "HIP %s @%d\n", hipGetErrorString(_e), __LINE__); \
            return 2;                                                              \
        }                                                                          \
    } while (0)

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int main(int argc, char** argv) {
    // Small valid shape: B=1, causal, fp16, S a multiple of block_size so the
    // dense BSHD K/V are byte-identical to the synthesized paged cache.
    const int B = 1;
    const int S = (argc > 1 ? atoi(argv[1]) : 64);
    const int NQH = 16, NKVH = 2, HD = 128;
    const int BLOCK_SIZE = 16;
    const int gqa = NQH / NKVH;
    const float scale = 1.0f / std::sqrt((float)HD);

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::string gcn = prop.gcnArchName;
    if (auto c = gcn.find(':'); c != std::string::npos) gcn = gcn.substr(0, c);
    std::printf("=== C-engine JIT 'sdpa' end-to-end (CEngine + Kernel) ===\n");
    std::printf("device=%s  B=%d S=%d NQH=%d NKVH=%d HD=%d block_size=%d\n", prop.gcnArchName, B, S,
                NQH, NKVH, HD, BLOCK_SIZE);

    if (S % BLOCK_SIZE != 0) {
        std::fprintf(stderr, "S must be a multiple of block_size for byte-compatible dense KV\n");
        return 2;
    }

    // ---- 1) C-engine .ll generation (POD -> CEngineResult). TIME: buildMs ----
    CEngine::SdpaProblem p;
    p.total_q = B * S;
    p.num_seqs = B;
    p.num_query_heads = NQH;
    p.num_kv_heads = NKVH;
    p.head_size = HD;
    p.block_size = BLOCK_SIZE;
    p.max_seqlen_q = S;
    p.max_seqlen_k = S;
    p.dtype = "fp16";
    p.sliding_window = 0;
    p.softcap = 0.0;
    p.arch = gcn.c_str();

    double t0 = now_ms();
    CEngineResult r;
    try {
        r = CEngine::build_sdpa(p);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "build_sdpa FAILED: %s\n", e.what());
        return 3;
    }
    double buildMs = now_ms() - t0;
    std::printf("[build] kernel=%s  ll=%zu bytes  block=%u  grid={%u,%u,%u}  buildMs=%.3f\n",
                r.manifest.kernel_name.c_str(), r.llvm_ir.size(), r.block, r.grid[0], r.grid[1],
                r.grid[2], buildMs);

    // ---- 2) Kernel::from_llvm_ir + comgr compile. TIME: compileMs ----
    Kernel k = Kernel::from_llvm_ir(r.llvm_ir, r.manifest, Compiler::isa_for(gcn));
    double t1 = now_ms();
    try {
        k.ensure_compiled();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ensure_compiled FAILED: %s\n", e.what());
        return 4;
    }
    double compileMs = now_ms() - t1;
    std::printf("[compile] hsaco=%zu bytes  compileMs=%.3f\n", k.hsaco().size(), compileMs);

    // ---- 3) Host data (BSHD: Q[S,NQH,HD], K/V[S,NKVH,HD]) ----
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<fp16> Qh((size_t)S * NQH * HD), Kh((size_t)S * NKVH * HD),
        Vh((size_t)S * NKVH * HD), Oh((size_t)S * NQH * HD, fp16(0));
    for (auto& x : Qh) x = (fp16)(nd(rng) * 0.5f);
    for (auto& x : Kh) x = (fp16)(nd(rng) * 0.5f);
    for (auto& x : Vh) x = (fp16)(nd(rng) * 0.5f);

    void *dQ, *dK, *dV, *dO;
    HIP_CHECK(hipMalloc(&dQ, Qh.size() * 2));
    HIP_CHECK(hipMalloc(&dK, Kh.size() * 2));
    HIP_CHECK(hipMalloc(&dV, Vh.size() * 2));
    HIP_CHECK(hipMalloc(&dO, Oh.size() * 2));
    HIP_CHECK(hipMemcpy(dQ, Qh.data(), Qh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK, Kh.data(), Kh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, Vh.data(), Vh.size() * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dO, 0, Oh.size() * 2));

    // ---- Synthesize paged-KV metadata for 1 seq (CkDslAttnPlan parity) ----
    const int max_blocks = (S + BLOCK_SIZE - 1) / BLOCK_SIZE;  // == S/BLOCK_SIZE
    std::vector<int32_t> bt((size_t)B * max_blocks), sl(B), cu(B + 1, 0);
    for (int b = 0; b < B; ++b) {
        sl[b] = S;
        cu[b + 1] = cu[b] + S;
        for (int j = 0; j < max_blocks; ++j) bt[(size_t)b * max_blocks + j] = b * max_blocks + j;
    }
    auto up = [](const std::vector<int32_t>& v) -> void* {
        void* p = nullptr;
        if (hipMalloc(&p, v.size() * 4) != hipSuccess) return nullptr;
        hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice);
        return p;
    };
    void* dBT = up(bt);
    void* dSL = up(sl);
    void* dCU = up(cu);

    // ---- 4) Pack the 18-arg paged-KV ABI + launch ----
    std::unordered_map<std::string, void*> ptrs = {
        {"output_ptr", dO},       {"query_ptr", dQ},
        {"key_cache_ptr", dK},    {"value_cache_ptr", dV},
        {"sink_ptr", nullptr},    {"block_tables_ptr", dBT},
        {"seq_lens_ptr", dSL},    {"alibi_slopes_ptr", nullptr},
        {"qq_bias_ptr", nullptr}, {"query_start_len_ptr", dCU},
    };
    auto f32 = [](float x) {
        uint32_t b;
        std::memcpy(&b, &x, 4);
        return (uint64_t)b;
    };
    std::unordered_map<std::string, uint64_t> scalars = {
        {"scale", f32(scale)},
        {"k_scale", f32(1.f)},
        {"v_scale", f32(1.f)},
        {"out_scale", f32(1.f)},
        {"softcap", f32(0.f)},
        {"num_seqs", (uint64_t)B},
        {"block_table_stride", (uint64_t)max_blocks},
        {"qq_bias_stride_0", 0},
    };

    // The 2D scalar kernel's native block-id space is (q_tok, q_head, dim) =
    // (total_q, num_query_heads, head_size), exactly what the C engine stashes in
    // grid_explicit. (r.grid is the tiled-kernel block space and does NOT apply
    // to the scalar reference kernel.)
    // After the grid fix, r.grid IS the scalar kernel's (q_tok,q_head,dim) grid
    // directly (== grid_explicit). Launch with r.grid to prove the fix, no workaround.
    std::array<unsigned, 3> launch_grid = r.grid;
    std::printf("[grid] launch_grid={%u,%u,%u} (from r.grid; scalar native q,head,dim)\n",
                launch_grid[0], launch_grid[1], launch_grid[2]);

    // Correctness launch.
    try {
        k.launch(ptrs, scalars, launch_grid, r.block, nullptr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "launch FAILED: %s\n", e.what());
        return 5;
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(Oh.data(), dO, Oh.size() * 2, hipMemcpyDeviceToHost));

    // ---- 5) CPU causal-attention oracle (B=1, GQA, fp16) ----
    auto qv = [&](int s, int h, int d) { return (float)Qh[((size_t)s * NQH + h) * HD + d]; };
    auto kv = [&](int s, int hk, int d) { return (float)Kh[((size_t)s * NKVH + hk) * HD + d]; };
    auto vv = [&](int s, int hk, int d) { return (float)Vh[((size_t)s * NKVH + hk) * HD + d]; };
    double worst = 0;
    int bad = 0;
    std::vector<float> sc(S);
    for (int qi = 0; qi < S; ++qi)
        for (int h = 0; h < NQH; ++h) {
            int hk = h / gqa, lim = qi + 1;
            float mx = -1e30f;
            for (int ki = 0; ki < lim; ++ki) {
                float s = 0;
                for (int d = 0; d < HD; ++d) s += qv(qi, h, d) * kv(ki, hk, d);
                s *= scale;
                sc[ki] = s;
                mx = std::max(mx, s);
            }
            float den = 0;
            for (int ki = 0; ki < lim; ++ki) {
                sc[ki] = std::exp(sc[ki] - mx);
                den += sc[ki];
            }
            for (int d = 0; d < HD; ++d) {
                float acc = 0;
                for (int ki = 0; ki < lim; ++ki) acc += sc[ki] * vv(ki, hk, d);
                acc /= den;
                float got = (float)Oh[((size_t)qi * NQH + h) * HD + d];
                float diff = std::fabs(got - acc);
                worst = std::max(worst, (double)diff);
                if (diff > 0.05f) ++bad;
            }
        }
    bool correct = (bad == 0);
    std::printf("[verify] max_abs_diff=%g bad=%d/%zu  %s\n", worst, bad, Oh.size(),
                correct ? "PASS" : "FAIL");

    // ---- 6) Warm per-launch timing over >=50 iters (hipEventElapsedTime) ----
    const int WARMUP = 10, ITERS = 100;
    for (int i = 0; i < WARMUP; ++i) k.launch(ptrs, scalars, launch_grid, r.block, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t evb, eve;
    HIP_CHECK(hipEventCreate(&evb));
    HIP_CHECK(hipEventCreate(&eve));
    HIP_CHECK(hipEventRecord(evb, nullptr));
    for (int i = 0; i < ITERS; ++i) k.launch(ptrs, scalars, launch_grid, r.block, nullptr);
    HIP_CHECK(hipEventRecord(eve, nullptr));
    HIP_CHECK(hipEventSynchronize(eve));
    float total_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&total_ms, evb, eve));
    double launchUs = (double)total_ms * 1000.0 / ITERS;
    std::printf("[bench] iters=%d  launchUs=%.3f (warm avg)\n", ITERS, launchUs);

    std::printf("\nRESULT correct=%d buildMs=%.3f compileMs=%.3f launchUs=%.3f\n", correct ? 1 : 0,
                buildMs, compileMs, launchUs);

    hipEventDestroy(evb);
    hipEventDestroy(eve);
    hipFree(dQ);
    hipFree(dK);
    hipFree(dV);
    hipFree(dO);
    hipFree(dBT);
    hipFree(dSL);
    hipFree(dCU);
    return correct ? 0 : 1;
}
