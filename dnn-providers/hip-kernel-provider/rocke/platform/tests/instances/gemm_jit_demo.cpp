// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// gemm_jit_demo.cpp -- flagship end-to-end proof: a real fp16 RCR GEMM built by
// the pure-C rocke engine, JIT-compiled via the runtime comgr path, launched
// on gfx950, and numerically verified against a CPU reference. Zero Python.
//
//   rocke_build_universal_gemm (C) -> .ll -> rocke::Compiler (comgr) -> HSACO
//   -> hipModuleLaunchKernel(gfx950) -> verify C[m,n] = sum_k A[m,k]*B[n,k].

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rocke_runtime/comgr.hpp"

extern "C" {
#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"
}

static void hck(hipError_t e, const char* w)
{
    if(e != hipSuccess)
    {
        std::fprintf(stderr, "HIP %s: %s\n", w, hipGetErrorString(e));
        std::exit(2);
    }
}

int main()
{
    const int M = 256, N = 256, K = 256;

    // ----- Stage 1-3 (pure C): configure GEMM spec + build + lower to .ll ---
    rocke_gemm_universal_spec_t spec = rocke_gemm_universal_spec_default();
    spec.tile.tile_m = 128;
    spec.tile.tile_n = 128;
    spec.tile.tile_k = 32;
    spec.tile.warp_m = 2;
    spec.tile.warp_n = 2;
    spec.tile.warp_k = 1;
    spec.tile.warp_tile_m = 32;
    spec.tile.warp_tile_n = 32;
    spec.tile.warp_tile_k = 16;
    spec.trait.pipeline = "compv3";
    spec.trait.scheduler = "intrawave";
    spec.trait.epilogue = "default";
    spec.data.dtype_a = "fp16";
    spec.data.dtype_b = "fp16";
    spec.data.dtype_c = "fp16";
    spec.data.layout = "RCR";
    spec.block_size = 0; // derived in finalize
    rocke_gemm_universal_spec_finalize(&spec);

    char reason[256] = {0};
    if(!rocke_gemm_universal_is_valid_spec(&spec, "gfx950", reason, sizeof reason))
    {
        std::fprintf(stderr, "spec invalid: %s\n", reason);
        return 1;
    }
    char kname[256] = {0};
    rocke_gemm_universal_kernel_name(&spec, kname, sizeof kname);
    const int block = spec.block_size;
    std::printf("[0] spec ok: %s  block_size=%d  tile=%dx%dx%d\n",
                kname,
                block,
                spec.tile.tile_m,
                spec.tile.tile_n,
                spec.tile.tile_k);

    char* ll = nullptr;
    char err[256] = {0};
    if(rocke_gemm_universal_lower_to_llvm(
           &spec, "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err)
           != ROCKE_OK
       || !ll)
    {
        std::fprintf(stderr, "lower failed: %s\n", err);
        return 1;
    }
    std::string llvm_ir(ll);
    std::free(ll);
    std::printf("[1] C engine emitted GEMM .ll: %zu bytes\n", llvm_ir.size());

    // ----- Stage 4 (runtime JIT): .ll -> HSACO -----------------------------
    auto hsaco = rocke::Compiler::compile(llvm_ir, rocke::Compiler::isa_for("gfx950"));
    std::printf("[2] runtime comgr compiled HSACO: %zu bytes\n", hsaco.size());

    // ----- Stage 5: load + launch ------------------------------------------
    hipModule_t mod;
    hck(hipModuleLoadData(&mod, hsaco.data()), "load");
    hipFunction_t fn;
    hck(hipModuleGetFunction(&fn, mod, kname), "getfn");
    std::printf("[3] loaded + got function\n");

    // RCR: A[M,K] row-major, B[N,K] row-major, C[M,N]. Integer 0/1 entries so
    // the fp32-accumulate->fp16 store is exact and we can demand bad==0.
    std::vector<_Float16> hA(M * K), hB(N * K), hC(M * N, (_Float16)-1);
    for(int m = 0; m < M; ++m)
        for(int k = 0; k < K; ++k)
            hA[m * K + k] = (_Float16)(((m + k) % 2) == 0 ? 1 : 0);
    for(int n = 0; n < N; ++n)
        for(int k = 0; k < K; ++k)
            hB[n * K + k] = (_Float16)(((n + k) % 3) == 0 ? 1 : 0);

    void *dA, *dB, *dC;
    hck(hipMalloc(&dA, hA.size() * 2), "mA");
    hck(hipMalloc(&dB, hB.size() * 2), "mB");
    hck(hipMalloc(&dC, hC.size() * 2), "mC");
    hck(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice), "cA");
    hck(hipMemcpy(dB, hB.data(), hB.size() * 2, hipMemcpyHostToDevice), "cB");

    struct
    {
        void* A;
        void* B;
        void* C;
        int M, N, K;
    } ka{dA, dB, dC, M, N, K};
    size_t ksz = sizeof(ka);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                   &ka,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE,
                   &ksz,
                   HIP_LAUNCH_PARAM_END};
    // grid_order NM -> (N_tiles, M_tiles, 1)
    unsigned gx = N / spec.tile.tile_n, gy = M / spec.tile.tile_m;
    hck(hipModuleLaunchKernel(fn, gx, gy, 1, block, 1, 1, 0, nullptr, nullptr, cfg), "launch");
    hck(hipDeviceSynchronize(), "sync");
    hck(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost), "cC");

    // CPU reference (RCR): C[m,n] = sum_k A[m,k]*B[n,k]
    int bad = 0;
    float c33 = 0, ref33 = 0;
    for(int m = 0; m < M && bad < 8; ++m)
        for(int n = 0; n < N; ++n)
        {
            int acc = 0;
            for(int k = 0; k < K; ++k)
                acc += (int)hA[m * K + k] * (int)hB[n * K + k];
            float got = (float)hC[m * N + n];
            if(m == 3 && n == 3)
            {
                c33 = got;
                ref33 = (float)acc;
            }
            if(got != (float)acc)
            {
                if(bad < 4)
                    std::printf("  mismatch [%d,%d] got %.1f want %d\n", m, n, got, acc);
                ++bad;
            }
        }
    std::printf(
        "[4] GPU GEMM verify: bad=%d/%d  (C[3,3]=%.1f expect %.1f)\n", bad, M * N, c33, ref33);
    std::printf("%s\n", bad == 0 ? "PASS: pure-C GEMM -> comgr JIT -> gfx950 -> correct" : "FAIL");
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    hipModuleUnload(mod);
    return bad == 0 ? 0 : 1;
}
