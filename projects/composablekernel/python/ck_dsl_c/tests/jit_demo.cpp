// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// jit_demo.cpp -- end-to-end proof that the pure-C ck_dsl_c engine wires into
// the RFC's runtime JIT path with ZERO Python:
//
//   ckc_* (C engine)  ->  .ll text  ->  ck_dsl::Compiler (libamd_comgr)  ->
//   HSACO  ->  hipModuleLoadData + launch (gfx950)  ->  numeric verify.
//
// The kernel is a vector-add (C[i] = A[i] + B[i]) built entirely through the
// C IRBuilder API -- no shipped artifact, no Python interpreter on the path.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ck_dsl_runtime/comgr.hpp" // ck_dsl::Compiler (the runtime JIT stage)

extern "C" {
#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
}

static void hck(hipError_t e, const char* what)
{
    if(e != hipSuccess)
    {
        std::fprintf(stderr, "HIP error at %s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

int main()
{
    // ----- Stage 1-3 (in pure C): build kernel + lower to AMDGPU LLVM IR -----
    ckc_ir_builder_t b;
    if(ckc_ir_builder_init(&b, "vadd") != CKC_OK)
    {
        std::fprintf(stderr, "init\n");
        return 1;
    }
    const ckc_type_t* pf32 = ckc_ptr_type(&b, ckc_f32(), "global");
    ckc_value_t* A         = ckc_b_param(&b, "A", pf32, nullptr);
    ckc_value_t* B         = ckc_b_param(&b, "B", pf32, nullptr);
    ckc_value_t* C         = ckc_b_param(&b, "C", pf32, nullptr);
    ckc_value_t* tid       = ckc_b_thread_id_x(&b);
    ckc_value_t* va        = ckc_b_global_load_f32(&b, A, tid, 4);
    ckc_value_t* vb        = ckc_b_global_load_f32(&b, B, tid, 4);
    ckc_value_t* vs        = ckc_b_fadd(&b, va, vb);
    ckc_b_global_store(&b, C, tid, vs, 4);
    if(!ckc_ir_builder_ok(&b))
    {
        std::fprintf(stderr, "build: %s\n", ckc_ir_builder_error(&b));
        return 1;
    }

    char* ll = nullptr;
    if(ckc_lower_kernel_to_llvm(ckc_ir_builder_kernel(&b), CKC_LLVM_FLAVOR_AUTO, "gfx950", &ll) !=
           CKC_OK ||
       !ll)
    {
        std::fprintf(stderr, "lower failed\n");
        return 1;
    }
    std::string llvm_ir(ll);
    std::free(ll);
    ckc_ir_builder_free(&b);
    std::printf("[1] C engine emitted .ll: %zu bytes, %zu lines\n",
                llvm_ir.size(),
                (size_t)std::count(llvm_ir.begin(), llvm_ir.end(), '\n'));

    // ----- Stage 4 (runtime JIT): .ll -> HSACO via libamd_comgr -------------
    std::vector<std::byte> hsaco =
        ck_dsl::Compiler::compile(llvm_ir, ck_dsl::Compiler::isa_for("gfx950"));
    std::printf("[2] runtime comgr compiled HSACO: %zu bytes\n", hsaco.size());

    // ----- Stage 5: load + launch on the GPU -------------------------------
    hipModule_t mod;
    hck(hipModuleLoadData(&mod, hsaco.data()), "hipModuleLoadData");
    hipFunction_t fn;
    hck(hipModuleGetFunction(&fn, mod, "vadd"), "hipModuleGetFunction");
    std::printf("[3] loaded module + got function 'vadd'\n");

    const int N = 256;
    std::vector<float> hA(N), hB(N), hC(N, -1.0f);
    for(int i = 0; i < N; ++i)
    {
        hA[i] = (float)i;
        hB[i] = (float)(2 * i + 1);
    }
    void *dA, *dB, *dC;
    hck(hipMalloc(&dA, N * sizeof(float)), "malloc A");
    hck(hipMalloc(&dB, N * sizeof(float)), "malloc B");
    hck(hipMalloc(&dC, N * sizeof(float)), "malloc C");
    hck(hipMemcpy(dA, hA.data(), N * sizeof(float), hipMemcpyHostToDevice), "H2D A");
    hck(hipMemcpy(dB, hB.data(), N * sizeof(float), hipMemcpyHostToDevice), "H2D B");

    struct
    {
        void* a;
        void* b;
        void* c;
    } kargs{dA, dB, dC};
    size_t ksz  = sizeof(kargs);
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                   &kargs,
                   HIP_LAUNCH_PARAM_BUFFER_SIZE,
                   &ksz,
                   HIP_LAUNCH_PARAM_END};
    hck(hipModuleLaunchKernel(fn, 1, 1, 1, N, 1, 1, 0, nullptr, nullptr, cfg), "launch");
    hck(hipDeviceSynchronize(), "sync");
    hck(hipMemcpy(hC.data(), dC, N * sizeof(float), hipMemcpyDeviceToHost), "D2H C");

    int bad = 0;
    for(int i = 0; i < N; ++i)
        if(hC[i] != hA[i] + hB[i])
        {
            if(bad < 4)
                std::printf("  mismatch i=%d got %f want %f\n", i, hC[i], hA[i] + hB[i]);
            ++bad;
        }
    std::printf("[4] GPU result verify: bad=%d/%d  (sample C[3]=%.1f, expect %.1f)\n",
                bad,
                N,
                hC[3],
                hA[3] + hB[3]);
    std::printf("%s\n",
                bad == 0 ? "PASS: pure-C engine -> comgr JIT -> gfx950 launch -> correct" : "FAIL");

    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    hipModuleUnload(mod);
    return bad == 0 ? 0 : 1;
}
