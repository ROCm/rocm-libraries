// okl_run.cpp - standalone benchmark for one packaged hipBLASLt kernel.
//
// Compile: /opt/rocm/bin/hipcc -O3 -std=c++17 okl_run.cpp -o okl_run
// Run:     ./okl_run
//
// Kernel: bf16 GEMM TN, solution_index 45732 (heuristic winner for 512^3 bf16 TN
// on gfx942 with ROCm 6.4.3 shipped library). Constants captured via
// TENSILE_DB=0xF0 dump - see kernel-packaging-research.md section 6 and 7.

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(c) do { hipError_t e=(c); if(e){ \
    fprintf(stderr,"HIP error %d at %s:%d: %s\n",e,__FILE__,__LINE__, \
            hipGetErrorString(e)); std::exit(1);} } while(0)

// === solution-specific constants ===
static constexpr const char* SOLUTION_CO_FILE =
    "/opt/rocm-6.4.3/lib/hipblaslt/library/"
    "TensileLibrary_BB_BB_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co";

static constexpr const char* SOLUTION_KERNEL =
    "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x32x128_MI16x16x1_SN_LDSB0_AFC0_"
    "AFEM8_AFEM8_ASEM32_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_"
    "GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA942_IU1_K1_LBSPPA256_LBSPPB256_"
    "LBSPPM0_LPA16_LPB16_LPMn1_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA0_"
    "NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_"
    "SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_"
    "VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1";

static constexpr uint32_t INTERNAL_ARGS  = 0x20080001u;
static constexpr uint32_t INTERNAL_ARGS1 = 0x4c010000u;
static constexpr uint32_t MACRO_TILE_0   = 32;
static constexpr uint32_t MACRO_TILE_1   = 32;
static constexpr uint32_t WORKGROUP_SIZE = 256;

// === problem ===
static constexpr uint32_t M = 512, N = 512, K = 512, BATCH = 1;
// TN: A is KxM (column-major: leading dim K), B is KxN (leading dim K),
// D and C are MxN (leading dim M).
static constexpr uint32_t LDA = K, LDB = K, LDC = M, LDD = M;
static constexpr float    ALPHA = 1.0f, BETA = 0.0f;

static constexpr size_t BF16_BYTES = 2;

static uint16_t fp32_to_bf16(float v) {
    uint32_t u; std::memcpy(&u, &v, 4);
    return uint16_t(u >> 16);
}
static float bf16_to_fp32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f; std::memcpy(&f, &u, 4);
    return f;
}

int main() {
    // ---- 1. Allocate device buffers ----
    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
    size_t bytesA = size_t(K) * M * BF16_BYTES;
    size_t bytesB = size_t(K) * N * BF16_BYTES;
    size_t bytesC = size_t(M) * N * BF16_BYTES;
    size_t bytesD = size_t(M) * N * BF16_BYTES;
    HIP_CHECK(hipMalloc(&dA, bytesA));
    HIP_CHECK(hipMalloc(&dB, bytesB));
    HIP_CHECK(hipMalloc(&dC, bytesC));
    HIP_CHECK(hipMalloc(&dD, bytesD));

    // ---- 2. Deterministic fill ----
    std::vector<uint16_t> hostA(size_t(K) * M), hostB(size_t(K) * N);
    for (size_t i = 0; i < hostA.size(); ++i)
        hostA[i] = fp32_to_bf16(0.01f * float(int(i % 1024) - 512));
    for (size_t i = 0; i < hostB.size(); ++i)
        hostB[i] = fp32_to_bf16(0.02f * float(int(i % 1024) - 512));
    HIP_CHECK(hipMemcpy(dA, hostA.data(), bytesA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hostB.data(), bytesB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, bytesC));
    HIP_CHECK(hipMemset(dD, 0, bytesD));

    // ---- 3. Load the code object and look up the kernel ----
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, SOLUTION_CO_FILE));
    hipFunction_t kernel;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, SOLUTION_KERNEL));

    // ---- 4. Build the 104-byte kernarg buffer ----
    alignas(8) uint8_t kernarg[104];
    auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(kernarg + off, &v, 4); };
    auto put_ptr = [&](size_t off, void* p)    { std::memcpy(kernarg + off, &p, 8); };
    auto put_f32 = [&](size_t off, float v)    { std::memcpy(kernarg + off, &v, 4); };

    uint32_t numWG = ((M + MACRO_TILE_0 - 1) / MACRO_TILE_0) *
                     ((N + MACRO_TILE_1 - 1) / MACRO_TILE_1) * BATCH;

    put_u32(0,   1);
    put_u32(4,   INTERNAL_ARGS);
    put_u32(8,   INTERNAL_ARGS1);
    put_u32(12,  numWG);
    put_u32(16,  M);
    put_u32(20,  N);
    put_u32(24,  BATCH);
    put_u32(28,  K);
    put_ptr(32,  dD);
    put_ptr(40,  dC);
    put_ptr(48,  dA);
    put_ptr(56,  dB);
    put_u32(64,  LDD); put_u32(68, 0);
    put_u32(72,  LDC); put_u32(76, 0);
    put_u32(80,  LDA); put_u32(84, 0);
    put_u32(88,  LDB); put_u32(92, 0);
    put_f32(96,  ALPHA);
    put_f32(100, BETA);

    // ---- 5. Launch ----
    size_t kernarg_size = sizeof(kernarg);
    void* launch_params[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &kernarg_size,
        HIP_LAUNCH_PARAM_END
    };

    uint32_t globalX = numWG * WORKGROUP_SIZE;

    // Matches hipblaslt-bench defaults: --cold_iters 2 --iters 10
    // CPU wall clock + one hipDeviceSynchronize after the hot loop, then
    // total / hot_calls. Equivalent to bench's default (non-`--use_gpu_timer`)
    // path through get_time_us_sync at testing_matmul.hpp:5293-5396
    // and argument_model.hpp:80-94.
    constexpr int COLD_ITERS = 2;
    constexpr int HOT_ITERS  = 10;

    auto launch = [&]() {
        HIP_CHECK(hipExtModuleLaunchKernel(
            kernel, globalX, 1, 1,
            WORKGROUP_SIZE, 1, 1,
            0, nullptr, nullptr, launch_params, nullptr, nullptr));
    };

    for (int i = 0; i < COLD_ITERS; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < HOT_ITERS; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double us_per_iter = total_us / HOT_ITERS;
    double flops  = 2.0 * double(M) * N * K * BATCH;
    double gflops = flops / us_per_iter * 1e-6;  // flops / (us * 1e-6) / 1e9 = flops / us * 1e-3 ... wait
    // gflops = (flops/iter) / (sec/iter) / 1e9
    //        = flops / (us_per_iter * 1e-6) / 1e9
    //        = flops / us_per_iter * 1e-3
    gflops = flops / us_per_iter * 1e-3;

    printf("kernel:    %.80s...\n", SOLUTION_KERNEL);
    printf("problem:   M=%u N=%u K=%u batch=%u  TN  bf16  alpha=%g beta=%g\n",
           M, N, K, BATCH, ALPHA, BETA);
    printf("grid:      %u workgroups x %u threads = %u global threads\n",
           numWG, WORKGROUP_SIZE, globalX);
    printf("iters:     %d hot (after %d cold)\n", HOT_ITERS, COLD_ITERS);
    printf("time:      %.3f us / iter   (total hot window: %.3f us, %d calls, single sync)\n",
           us_per_iter, total_us, HOT_ITERS);
    printf("perf:      %.1f gflops\n", gflops);

    // ---- 6. Spot-check correctness ----
    std::vector<uint16_t> hostD(size_t(M) * N);
    HIP_CHECK(hipMemcpy(hostD.data(), dD, bytesD, hipMemcpyDeviceToHost));
    // TN: D[i,j] = sum_k A[k,i] * B[k,j]
    // Column-major A (KxM): A[k,i] = hostA[i*K + k]
    // Column-major B (KxN): B[k,j] = hostB[j*K + k]
    // Column-major D (MxN): D[i,j] = hostD[j*M + i]
    auto ref_elem = [&](uint32_t i, uint32_t j) {
        double s = 0;
        for (uint32_t k = 0; k < K; ++k) {
            s += double(bf16_to_fp32(hostA[size_t(i) * K + k])) *
                 double(bf16_to_fp32(hostB[size_t(j) * K + k]));
        }
        return float(ALPHA * s + BETA * 0.0 /*C was zeroed*/);
    };
    struct { uint32_t i, j; } probes[] = {{0, 0}, {1, 0}, {0, 1}, {M / 2, N / 2}, {M - 1, N - 1}};
    int fails = 0;
    for (auto p : probes) {
        float got = bf16_to_fp32(hostD[size_t(p.j) * M + p.i]);
        float ref = ref_elem(p.i, p.j);
        double rel = std::abs(got - ref) / (std::abs(ref) + 1e-9f);
        bool ok = rel < 1e-2;  // bf16 has ~3 decimal digits; 1% tolerance
        printf("  D[%4u,%4u]: got=%-12g ref=%-12g rel=%.2e  %s\n",
               p.i, p.j, got, ref, rel, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
    }

    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC)); HIP_CHECK(hipFree(dD));
    HIP_CHECK(hipModuleUnload(module));
    return fails ? 2 : 0;
}
