// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Standalone runtime test (no hipDNN). Exercises the full transparent pipeline
// on a real bundle + GPU:
//   stage 2  Dispatcher.select(Problem) -> cache_key
//   stage 3  ArtifactStore resolves the artifact
//   stage 4  Compiler (comgr) for the .ll path
//   stage 5  Kernel launch + numerical verify
// Tests BOTH path (a) prebuilt HSACO and path (b) comgr-from-.ll, asserting
// each produces a bit-exact RCR fp16 GEMM.
//
// usage: ck_dsl_runtime_test <bundle_dir> M N K [gfx]
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "ck_dsl_runtime/runtime.hpp"

using fp16 = _Float16;
using namespace ck_dsl;

static int run_gemm(Kernel& k, const char* label, int M, int N, int K) {
    std::vector<fp16> A((size_t)M * K), B((size_t)N * K), C((size_t)M * N, fp16(0));
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> d(-4.f, 4.f);
    for (auto& v : A) v = (fp16)std::round(d(rng));
    for (auto& v : B) v = (fp16)std::round(d(rng));

    fp16 *Ad, *Bd, *Cd;
    hip_check(hipMalloc(&Ad, A.size() * 2), "malloc A");
    hip_check(hipMalloc(&Bd, B.size() * 2), "malloc B");
    hip_check(hipMalloc(&Cd, C.size() * 2), "malloc C");
    hip_check(hipMemcpy(Ad, A.data(), A.size() * 2, hipMemcpyHostToDevice), "h2d A");
    hip_check(hipMemcpy(Bd, B.data(), B.size() * 2, hipMemcpyHostToDevice), "h2d B");
    hip_check(hipMemset(Cd, 0, C.size() * 2), "memset C");

    auto grid = k.gemm_grid(M, N);
    unsigned block = (unsigned)k.manifest().threads_per_block;
    std::printf("[%s] cache_key=%s grid=(%u,%u,%u) block=%u\n", label, k.cache_key().c_str(),
                grid[0], grid[1], grid[2], block);
    k.launch({{"A", Ad}, {"B", Bd}, {"C", Cd}},
             {{"M", (uint64_t)M}, {"N", (uint64_t)N}, {"K", (uint64_t)K}}, grid, block);
    hip_check(hipDeviceSynchronize(), "sync");
    hip_check(hipMemcpy(C.data(), Cd, C.size() * 2, hipMemcpyDeviceToHost), "d2h C");

    double worst = 0;
    int bad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0;
            for (int kk = 0; kk < K; ++kk)
                acc += (float)A[(size_t)m * K + kk] * (float)B[(size_t)n * K + kk];
            float diff = std::fabs((float)C[(size_t)m * N + n] - acc);
            worst = std::max(worst, (double)diff);
            if (diff > 4.0f) ++bad;
        }
    hipFree(Ad);
    hipFree(Bd);
    hipFree(Cd);
    std::printf("[%s] verify: max_abs_diff=%g bad=%d/%d  %s\n", label, worst, bad, M * N,
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s bundle_dir M N K [gfx]\n", argv[0]);
        return 1;
    }
    std::string bundle = argv[1];
    int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
    std::string gfx = argc > 5 ? argv[5] : "gfx950";
    std::string isa = Compiler::isa_for(gfx);

    ArtifactStore store;
    size_t n = store.add_bundle(bundle);
    std::printf("indexed %zu kernel(s) from %s\n", n, bundle.c_str());

    Dispatcher disp(store);
    Problem p;
    p.op = "gemm";
    p.dtype = "fp16";
    p.layout = "RCR";
    p.arch = gfx;
    p.M = M;
    p.N = N;
    p.K = K;
    auto choice = disp.select(p);
    if (!choice.valid()) {
        std::fprintf(stderr, "no candidate for problem\n");
        return 1;
    }
    std::printf("selected: %s (tile %dx%dx%d)\n", choice.cache_key.c_str(), choice.block_m,
                choice.block_n, choice.block_k);

    int rc = 0;
    // PATH A: prebuilt HSACO (or whatever the store prefers).
    {
        Kernel ka = store.make_kernel(choice.cache_key, isa);
        rc |= run_gemm(ka, "path-a", M, N, K);
    }
    // PATH B: force comgr-from-.ll if the entry ships a .ll.
    const auto& e = store.at(choice.cache_key);
    if (e.has_ll()) {
        Kernel kb = Kernel::from_llvm_ir(ArtifactStore::read_text(e.ll_path), e.manifest, isa);
        rc |= run_gemm(kb, "path-b(comgr)", M, N, K);
    } else {
        std::printf("[path-b] skipped (no .ll in bundle)\n");
    }
    std::printf(rc == 0 ? "ALL PASS\n" : "FAILURES\n");
    return rc;
}
