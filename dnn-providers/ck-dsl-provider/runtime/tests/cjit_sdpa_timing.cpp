// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// cjit_sdpa_timing.cpp -- fine-grained timing of EVERY stage of the SDPA JIT
// pipeline (pure-C engine -> comgr -> HIP launch), zero Python.
//
//   ENGINE   build IR (ckc SSA construct)         |  lower IR -> .ll
//   COMGR    source->BC | BC->reloc | reloc->exe  | (cold incl lib-init vs warm)
//   HIP      hipModuleLoadData | hipModuleGetFunction | cold launch | warm launch
//
// Build:
//   hipcc -std=c++17 -I <ck_dsl_c>/include -I <provider>/runtime/include -I/opt/rocm/include \
//     cjit_sdpa_timing.cpp -L<libdir> -lckc_core -lamd_comgr -o /tmp/cjit_sdpa_timing

#include <amd_comgr/amd_comgr.h>
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

extern "C" {
#include "ckc/instance_attention_unified.h"
#include "ckc/ir.h"
#include "ckc/lower_llvm.h"
}

using fp16 = __half;
using namespace ck_dsl;

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
#define HCK(e)                                                                     \
    do {                                                                           \
        hipError_t _e = (e);                                                       \
        if (_e != hipSuccess) {                                                    \
            std::fprintf(stderr, "HIP %s @%d\n", hipGetErrorString(_e), __LINE__); \
            return 2;                                                              \
        }                                                                          \
    } while (0)

// Timed replica of ck_dsl::Compiler::compile, exposing the 3 comgr actions.
struct ComgrStageTimes {
    double setup, src_to_bc, bc_to_reloc, reloc_to_exe, extract, total;
    size_t hsaco;
};
static std::vector<std::byte> comgr_compile_timed(const std::string& ir, const std::string& isa,
                                                  ComgrStageTimes& t) {
    auto ck = [](amd_comgr_status_t s, const char* w) {
        if (s != AMD_COMGR_STATUS_SUCCESS) {
            const char* m = nullptr;
            amd_comgr_status_string(s, &m);
            std::fprintf(stderr, "comgr %s: %s\n", w, m ? m : "");
            std::abort();
        }
    };
    double a = now_ms();
    amd_comgr_data_set_t in_set{};
    ck(amd_comgr_create_data_set(&in_set), "ds_in");
    amd_comgr_data_t src{};
    ck(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &src), "data");
    ck(amd_comgr_set_data(src, ir.size(), ir.data()), "set_data");
    ck(amd_comgr_set_data_name(src, "kernel.ll"), "name");
    ck(amd_comgr_data_set_add(in_set, src), "add");
    amd_comgr_action_info_t info{};
    ck(amd_comgr_create_action_info(&info), "ai");
    ck(amd_comgr_action_info_set_isa_name(info, isa.c_str()), "isa");
    ck(amd_comgr_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR), "lang");
    const char* opt[] = {"-O3"};
    ck(amd_comgr_action_info_set_option_list(info, opt, 1), "opts");
    double b = now_ms();
    t.setup = b - a;
    amd_comgr_data_set_t bc{};
    ck(amd_comgr_create_data_set(&bc), "ds_bc");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in_set, bc), "src_to_bc");
    double c = now_ms();
    t.src_to_bc = c - b;
    amd_comgr_data_set_t rel{};
    ck(amd_comgr_create_data_set(&rel), "ds_rel");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc, rel),
       "bc_to_reloc");
    double d = now_ms();
    t.bc_to_reloc = d - c;
    amd_comgr_data_set_t exe{};
    ck(amd_comgr_create_data_set(&exe), "ds_exe");
    ck(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, rel, exe),
       "reloc_to_exe");
    double e = now_ms();
    t.reloc_to_exe = e - d;
    size_t n = 0;
    ck(amd_comgr_action_data_count(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, &n), "count");
    amd_comgr_data_t ed{};
    ck(amd_comgr_action_data_get_data(exe, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &ed), "get");
    size_t sz = 0;
    ck(amd_comgr_get_data(ed, &sz, nullptr), "sz");
    std::vector<std::byte> hsaco(sz);
    ck(amd_comgr_get_data(ed, &sz, reinterpret_cast<char*>(hsaco.data())), "rd");
    amd_comgr_release_data(ed);
    amd_comgr_release_data(src);
    amd_comgr_destroy_data_set(in_set);
    amd_comgr_destroy_data_set(bc);
    amd_comgr_destroy_data_set(rel);
    amd_comgr_destroy_data_set(exe);
    amd_comgr_destroy_action_info(info);
    double f = now_ms();
    t.extract = f - e;
    t.total = f - a;
    t.hsaco = sz;
    return hsaco;
}

int main(int argc, char** argv) {
    const int B = 1, S = (argc > 1 ? atoi(argv[1]) : 64), NQH = 16, NKVH = 2, HD = 128,
              BLOCK_SIZE = 16;
    const int gqa = NQH / NKVH;
    const float scale = 1.0f / std::sqrt((float)HD);
    HCK(hipSetDevice(0));
    hipDeviceProp_t prop;
    HCK(hipGetDeviceProperties(&prop, 0));
    std::string gcn = prop.gcnArchName;
    if (auto c = gcn.find(':'); c != std::string::npos) gcn = gcn.substr(0, c);
    std::printf("=== SDPA JIT per-stage timing  device=%s  B=%d S=%d NQH=%d NKVH=%d HD=%d ===\n\n",
                prop.gcnArchName, B, S, NQH, NKVH, HD);

    // ---- problem ----
    ckc_unified_attention_problem_t prob{};
    prob.total_q = B * S;
    prob.num_seqs = B;
    prob.num_query_heads = NQH;
    prob.num_kv_heads = NKVH;
    prob.head_size = HD;
    prob.block_size = BLOCK_SIZE;
    prob.max_seqlen_q = S;
    prob.max_seqlen_k = S;
    prob.dtype = "fp16";
    prob.sliding_window = 0;
    prob.softcap = 0.0;

    // ============================ ENGINE ============================
    // E1: build the SSA KernelDef in C (includes builder init).
    double e0 = now_ms();
    ckc_ir_builder_t bld;
    ckc_kernel_def_t* kd = ckc_build_unified_attention_2d_scalar_new(&bld, &prob, nullptr);
    double e1 = now_ms();
    if (!kd || !ckc_ir_builder_ok(&bld)) {
        std::fprintf(stderr, "build IR failed: %s\n", ckc_ir_builder_error(&bld));
        return 3;
    }
    // E2: lower KernelDef -> AMDGPU .ll text.
    char* ll = nullptr;
    if (ckc_lower_kernel_to_llvm(kd, CKC_LLVM_FLAVOR_AUTO, gcn.c_str(), &ll) != CKC_OK || !ll) {
        std::fprintf(stderr, "lower failed\n");
        return 3;
    }
    double e2 = now_ms();
    std::string llvm_ir(ll);
    std::free(ll);
    ckc_ir_builder_free(&bld);
    // E3: manifest+grid synthesis (via CEngine, the provider path).
    CEngine::SdpaProblem cp;
    cp.total_q = B * S;
    cp.num_seqs = B;
    cp.num_query_heads = NQH;
    cp.num_kv_heads = NKVH;
    cp.head_size = HD;
    cp.block_size = BLOCK_SIZE;
    cp.max_seqlen_q = S;
    cp.max_seqlen_k = S;
    cp.dtype = "fp16";
    cp.sliding_window = 0;
    cp.softcap = 0.0;
    cp.arch = gcn.c_str();
    double m0 = now_ms();
    CEngineResult r = CEngine::build_sdpa(cp);
    double m1 = now_ms();

    std::printf("ENGINE (pure C, no Python):\n");
    std::printf("  E1 build SSA KernelDef ........ %8.3f ms\n", e1 - e0);
    std::printf("  E2 lower IR -> .ll (%zu B) ..... %8.3f ms\n", llvm_ir.size(), e2 - e1);
    std::printf("  E3 manifest+grid synth ........ %8.3f ms  (CEngine::build_sdpa total %.3f)\n",
                (m1 - m0), (m1 - m0));
    std::printf("  engine subtotal (E1+E2) ....... %8.3f ms\n\n", e2 - e0);

    // ============================ COMGR ============================
    // Cold (first ever comgr call -> includes comgr/LLVM library init) then warm.
    ComgrStageTimes cold{}, warm{};
    auto hsaco = comgr_compile_timed(llvm_ir, Compiler::isa_for(gcn), cold);
    comgr_compile_timed(llvm_ir, Compiler::isa_for(gcn), warm);
    std::printf("COMGR (.ll -> HSACO, %zu B):           cold        warm\n", cold.hsaco);
    std::printf("  C1 data setup + action info ... %8.3f   %8.3f ms\n", cold.setup, warm.setup);
    std::printf("  C2 COMPILE_SOURCE_TO_BC ....... %8.3f   %8.3f ms\n", cold.src_to_bc,
                warm.src_to_bc);
    std::printf("  C3 CODEGEN_BC_TO_RELOCATABLE .. %8.3f   %8.3f ms\n", cold.bc_to_reloc,
                warm.bc_to_reloc);
    std::printf("  C4 LINK_RELOC_TO_EXECUTABLE ... %8.3f   %8.3f ms\n", cold.reloc_to_exe,
                warm.reloc_to_exe);
    std::printf("  C5 extract HSACO bytes ........ %8.3f   %8.3f ms\n", cold.extract, warm.extract);
    std::printf("  comgr total ................... %8.3f   %8.3f ms\n\n", cold.total, warm.total);

    // ============================ HIP load/getfn ============================
    double h0 = now_ms();
    hipModule_t mod;
    HCK(hipModuleLoadData(&mod, hsaco.data()));
    double h1 = now_ms();
    hipFunction_t fn;
    HCK(hipModuleGetFunction(&fn, mod, r.manifest.kernel_name.c_str()));
    double h2 = now_ms();
    std::printf("HIP module:\n");
    std::printf("  H1 hipModuleLoadData .......... %8.3f ms\n", h1 - h0);
    std::printf("  H2 hipModuleGetFunction ....... %8.3f ms\n\n", h2 - h1);

    // ---- buffers + 18-arg pack via Kernel (the provider path) ----
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<fp16> Qh((size_t)S * NQH * HD), Kh((size_t)S * NKVH * HD),
        Vh((size_t)S * NKVH * HD), Oh((size_t)S * NQH * HD, fp16(0));
    for (auto& x : Qh) x = (fp16)(nd(rng) * 0.5f);
    for (auto& x : Kh) x = (fp16)(nd(rng) * 0.5f);
    for (auto& x : Vh) x = (fp16)(nd(rng) * 0.5f);
    void *dQ, *dK, *dV, *dO;
    HCK(hipMalloc(&dQ, Qh.size() * 2));
    HCK(hipMalloc(&dK, Kh.size() * 2));
    HCK(hipMalloc(&dV, Vh.size() * 2));
    HCK(hipMalloc(&dO, Oh.size() * 2));
    HCK(hipMemcpy(dQ, Qh.data(), Qh.size() * 2, hipMemcpyHostToDevice));
    HCK(hipMemcpy(dK, Kh.data(), Kh.size() * 2, hipMemcpyHostToDevice));
    HCK(hipMemcpy(dV, Vh.data(), Vh.size() * 2, hipMemcpyHostToDevice));
    HCK(hipMemset(dO, 0, Oh.size() * 2));
    const int max_blocks = S / BLOCK_SIZE;
    std::vector<int32_t> bt(max_blocks), sl(1, S), cu(2, 0);
    sl[0] = S;
    cu[1] = S;
    for (int j = 0; j < max_blocks; ++j) bt[j] = j;
    auto up = [](const std::vector<int32_t>& v) -> void* {
        void* p = nullptr;
        hipMalloc(&p, v.size() * 4);
        hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice);
        return p;
    };
    void *dBT = up(bt), *dSL = up(sl), *dCU = up(cu);
    Kernel k = Kernel::from_hsaco(hsaco, r.manifest);
    k.ensure_compiled();  // from_hsaco => just (re)load+getfn, no comgr
    std::unordered_map<std::string, void*> ptrs = {
        {"output_ptr", dO},       {"query_ptr", dQ},
        {"key_cache_ptr", dK},    {"value_cache_ptr", dV},
        {"sink_ptr", nullptr},    {"block_tables_ptr", dBT},
        {"seq_lens_ptr", dSL},    {"alibi_slopes_ptr", nullptr},
        {"qq_bias_ptr", nullptr}, {"query_start_len_ptr", dCU}};
    auto f32 = [](float x) {
        uint32_t b;
        std::memcpy(&b, &x, 4);
        return (uint64_t)b;
    };
    std::unordered_map<std::string, uint64_t> sca = {{"scale", f32(scale)},
                                                     {"k_scale", f32(1)},
                                                     {"v_scale", f32(1)},
                                                     {"out_scale", f32(1)},
                                                     {"softcap", f32(0)},
                                                     {"num_seqs", (uint64_t)B},
                                                     {"block_table_stride", (uint64_t)max_blocks},
                                                     {"qq_bias_stride_0", 0}};

    // ============================ LAUNCH ============================
    double L0 = now_ms();
    k.launch(ptrs, sca, r.grid, r.block, nullptr);
    HCK(hipDeviceSynchronize());
    double L1 = now_ms();  // cold (pack + first dispatch)
    const int W = 10, IT = 100;
    for (int i = 0; i < W; ++i) k.launch(ptrs, sca, r.grid, r.block, nullptr);
    HCK(hipDeviceSynchronize());
    hipEvent_t a, b2;
    hipEventCreate(&a);
    hipEventCreate(&b2);
    hipEventRecord(a, nullptr);
    for (int i = 0; i < IT; ++i) k.launch(ptrs, sca, r.grid, r.block, nullptr);
    hipEventRecord(b2, nullptr);
    hipEventSynchronize(b2);
    float tot = 0;
    hipEventElapsedTime(&tot, a, b2);
    double warmUs = (double)tot * 1000.0 / IT;
    std::printf("LAUNCH (grid={%u,%u,%u} block=%u):\n", r.grid[0], r.grid[1], r.grid[2], r.block);
    std::printf("  L1 cold launch (pack+1st dispatch+sync) %8.3f ms\n", L1 - L0);
    std::printf("  L2 warm launch (avg/%d, hipEvent) ...... %8.3f us\n\n", IT, warmUs);

    // verify
    HCK(hipMemcpy(Oh.data(), dO, Oh.size() * 2, hipMemcpyDeviceToHost));
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
                float diff = std::fabs((float)Oh[((size_t)qi * NQH + h) * HD + d] - acc);
                worst = std::max(worst, (double)diff);
                if (diff > 0.05f) ++bad;
            }
        }

    double jit_cold = (e2 - e0) + (m1 - m0) + cold.total + (h1 - h0) + (h2 - h1);
    std::printf(
        "=== END-TO-END JIT (cold, one-time) = %.3f ms  [verify bad=%d/%zu max=%.2e %s] ===\n",
        jit_cold, bad, Oh.size(), worst, bad == 0 ? "PASS" : "FAIL");
    std::printf(
        "    breakdown: engine %.2f + comgr %.2f + hip-load %.3f ms ; then warm launch %.1f "
        "us/call\n",
        (e2 - e0) + (m1 - m0), cold.total, (h2 - h0), warmUs);
    hipModuleUnload(mod);
    hipFree(dQ);
    hipFree(dK);
    hipFree(dV);
    hipFree(dO);
    hipFree(dBT);
    hipFree(dSL);
    hipFree(dCU);
    return bad == 0 ? 0 : 1;
}
