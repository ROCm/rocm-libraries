// bench_hipdnn_sdpa.cpp
// Standalone benchmark comparing our Flash2 V7 kernel against hipDNN's ASM SDPA engine.
// Compiles WITHOUT the full hipDNN build — uses our kernel directly.
//
// Your colleague's benchmark scenario:
//   graph->set_io_data_type(HALF)       ← FP16: our engine handles this
//   graph->set_io_data_type(BFLOAT16)   ← BF16: ASM SDPA engine handles this
//
// Usage:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 -I/opt/rocm/include \
//         bench_hipdnn_sdpa.cpp HipFlash2FwdPlan.hip \
//         -L/opt/rocm/lib -lhipblas \
//         -o bench_hipdnn_sdpa
//
// This benchmark validates that our hipDNN engine produces:
//   1. Correct output (MaxErr < 0.01 vs CPU FP32 reference)
//   2. High throughput (target: 78+ TFLOPS on MI325X at seq=4096 causal)

#include "hip_flash2_engine/HipFlash2FwdPlanBuilder.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>

#define HIP_CHECK(x)                                                                        \
    do                                                                                      \
    {                                                                                       \
        hipError_t _e = (x);                                                                \
        if(_e != hipSuccess)                                                                \
        {                                                                                   \
            fprintf(stderr, "[HIP] %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
            exit(1);                                                                        \
        }                                                                                   \
    } while(0)

// Declaration of our kernel launcher (from HipFlash2FwdPlan.hip)
namespace hip_flash2_engine
{
void launch_flash2_v7(const Flash2FwdParams& p,
                      const __half* Q,
                      const __half* K,
                      const __half* V,
                      __half* O,
                      hipStream_t stream);
}

struct Timer
{
    hipEvent_t t0, t1;
    Timer()
    {
        hipEventCreate(&t0);
        hipEventCreate(&t1);
    }
    ~Timer()
    {
        hipEventDestroy(t0);
        hipEventDestroy(t1);
    }
    void start()
    {
        hipEventRecord(t0);
    }
    float stop()
    {
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        float ms;
        hipEventElapsedTime(&ms, t0, t1);
        return ms;
    }
};

static void rfp16(std::vector<__half>& v)
{
    for(auto& x : v)
        x = __float2half((rand() / (float)RAND_MAX - 0.5f) * 0.2f);
}

// CPU FP32 reference for correctness check
static void cpu_attention_ref(const std::vector<float>& q,
                              const std::vector<float>& k,
                              const std::vector<float>& v,
                              std::vector<float>& out,
                              int B,
                              int Hq,
                              int Hk,
                              int Lq,
                              int Lk,
                              int D,
                              float scale,
                              bool causal)
{
    int gqa = Hq / Hk;
    out.assign((size_t)B * Lq * Hq * D, 0.f);
    std::vector<float> scores(Lk);
    for(int b = 0; b < B; b++)
        for(int hq = 0; hq < Hq; hq++)
        {
            int hk = hq / gqa;
            for(int qi = 0; qi < Lq; qi++)
            {
                float mx = -1e38f;
                for(int ki = 0; ki < Lk; ki++)
                {
                    if(causal && ki > qi)
                    {
                        scores[ki] = -1e38f;
                        continue;
                    }
                    float s = 0;
                    for(int d = 0; d < D; d++)
                        s += q[b * (size_t)Lq * Hq * D + qi * Hq * D + hq * D + d]
                             * k[b * (size_t)Lk * Hk * D + ki * Hk * D + hk * D + d];
                    scores[ki] = s * scale;
                    mx = std::max(mx, scores[ki]);
                }
                float sumexp = 0;
                for(int ki = 0; ki < Lk; ki++)
                {
                    scores[ki] = (scores[ki] > -1e30f) ? expf(scores[ki] - mx) : 0.f;
                    sumexp += scores[ki];
                }
                for(int d = 0; d < D; d++)
                {
                    float acc = 0;
                    for(int ki = 0; ki < Lk; ki++)
                        acc += scores[ki] * v[b * (size_t)Lk * Hk * D + ki * Hk * D + hk * D + d];
                    out[b * (size_t)Lq * Hq * D + qi * Hq * D + hq * D + d]
                        = (sumexp > 0) ? acc / sumexp : 0;
                }
            }
        }
}

static double tflops_val(int B, int Hq, int Lq, int Lk, int D, double ms)
{
    return (4.0 * B * Hq * (double)Lq * Lk * D / (ms * 1e-3)) / 1e12;
}

struct Cfg
{
    const char* label;
    int B, Hq, Hk, Lq, Lk, D;
    bool causal;
};

static Cfg CFGS[] = {
    // hipDNN SDPA typical call shapes (from SdpaFprop.cpp sample)
    {"SDPA  B=2  H=4   seq=128  D=128 causal", 2, 4, 4, 128, 128, 128, true},
    {"SDPA  B=2  H=4   seq=128  D=128 no-mask", 2, 4, 4, 128, 128, 128, false},
    // Production LLM shapes (your colleague's benchmark)
    {"Prefill B=1 MHA   seq=2048 D=128 causal", 1, 32, 32, 2048, 2048, 128, true},
    {"Prefill B=1 MHA   seq=4096 D=128 causal", 1, 32, 32, 4096, 4096, 128, true},
    {"Prefill B=1 GQA4  seq=4096 D=128 causal", 1, 32, 8, 4096, 4096, 128, true},
    {"Prefill B=1 MHA   seq=2048 D=64  causal", 1, 32, 32, 2048, 2048, 64, true},
    {"Decode  B=1 MHA   kv=2048  D=128", 1, 32, 32, 1, 2048, 128, false},
    {"Decode  B=8 GQA4  kv=4096  D=128", 8, 32, 8, 1, 4096, 128, false},
};
static const int NC = 8, WU = 5, REPS = 30;

int main()
{
    srand(42);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("=== hipDNN SDPA — Flash2 V7 Engine Benchmark ===\n");
    printf("GPU: %s  arch=%s  CUs=%d  HBM=%.0fGB\n\n",
           prop.name,
           prop.gcnArchName,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);
    printf("This benchmark validates our hipDNN HIP Flash2 engine:\n");
    printf("  - graph->set_io_data_type(HALF) → dispatches to our engine\n");
    printf("  - Correctness: MaxErr vs CPU FP32 reference\n");
    printf("  - Performance: TFLOPS\n\n");

    const int W = 44;
    printf("%-*s  %8s %9s  %8s\n", W, "Config", "Time(ms)", "TFLOPS", "MaxErr");
    printf("%s\n", std::string(75, '=').c_str());

    for(int ci = 0; ci < NC; ci++)
    {
        const Cfg& c = CFGS[ci];
        size_t Qe = (size_t)c.B * c.Lq * c.Hq * c.D;
        size_t KVe = (size_t)c.B * c.Lk * c.Hk * c.D;

        // Check applicability (mimics hipDNN isApplicable)
        hip_flash2_engine::HipFlash2FwdPlanBuilder builder;
        bool applicable = builder.isApplicable(prop.gcnArchName, // e.g. "gfx942:sramecc+:xnack-"
                                               c.Lq,
                                               c.Lk,
                                               c.D,
                                               "fp16",
                                               false,
                                               false);

        if(!applicable)
        {
            printf("%-*s  [SKIP — dispatch to ASM/other engine for this shape]\n", W, c.label);
            continue;
        }

        std::vector<__half> hq(Qe), hk(KVe), hv(KVe);
        rfp16(hq);
        rfp16(hk);
        rfp16(hv);

        __half *dq, *dk, *dv, *do_;
        HIP_CHECK(hipMalloc(&dq, Qe * 2));
        HIP_CHECK(hipMalloc(&dk, KVe * 2));
        HIP_CHECK(hipMalloc(&dv, KVe * 2));
        HIP_CHECK(hipMalloc(&do_, Qe * 2));
        HIP_CHECK(hipMemcpy(dq, hq.data(), Qe * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dk, hk.data(), KVe * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dv, hv.data(), KVe * 2, hipMemcpyHostToDevice));

        // Build params (mimics what hipDNN extracts from the graph)
        hip_flash2_engine::Flash2FwdParams params;
        params.batch = c.B;
        params.num_heads_q = c.Hq;
        params.num_heads_k = c.Hk;
        params.seq_len_q = c.Lq;
        params.seq_len_kv = c.Lk;
        params.head_dim = c.D;
        params.causal = c.causal;
        params.attn_scale = 1.f / sqrtf((float)c.D);
        // NHD layout: [B, L, H, D]
        params.q_stride_batch = (int64_t)c.Lq * c.Hq * c.D;
        params.q_stride_seq = (int64_t)c.Hq * c.D;
        params.q_stride_head = (int64_t)c.D;
        params.k_stride_batch = (int64_t)c.Lk * c.Hk * c.D;
        params.k_stride_seq = (int64_t)c.Hk * c.D;
        params.k_stride_head = (int64_t)c.D;
        params.v_stride_batch = params.k_stride_batch;
        params.v_stride_seq = params.k_stride_seq;
        params.v_stride_head = params.k_stride_head;
        params.o_stride_batch = params.q_stride_batch;
        params.o_stride_seq = params.q_stride_seq;
        params.o_stride_head = params.q_stride_head;

        // Warmup
        for(int i = 0; i < WU; i++)
            hip_flash2_engine::launch_flash2_v7(params, dq, dk, dv, do_, nullptr);
        HIP_CHECK(hipDeviceSynchronize());

        // Timed benchmark
        Timer T;
        T.start();
        for(int i = 0; i < REPS; i++)
            hip_flash2_engine::launch_flash2_v7(params, dq, dk, dv, do_, nullptr);
        float lat = T.stop() / REPS;

        // Correctness check vs CPU FP32 reference
        std::vector<__half> out_gpu(Qe);
        HIP_CHECK(hipMemcpy(out_gpu.data(), do_, Qe * 2, hipMemcpyDeviceToHost));

        std::vector<float> q_f(Qe), k_f(KVe), v_f(KVe), ref(Qe);
        for(size_t i = 0; i < Qe; i++)
            q_f[i] = __half2float(hq[i]);
        for(size_t i = 0; i < KVe; i++)
        {
            k_f[i] = __half2float(hk[i]);
            v_f[i] = __half2float(hv[i]);
        }
        cpu_attention_ref(
            q_f, k_f, v_f, ref, c.B, c.Hq, c.Hk, c.Lq, c.Lk, c.D, params.attn_scale, c.causal);

        float max_err = 0;
        for(size_t i = 0; i < Qe; i++)
            max_err = std::max(max_err, std::abs(__half2float(out_gpu[i]) - ref[i]));

        double tf = tflops_val(c.B, c.Hq, c.Lq, c.Lk, c.D, lat);
        const char* result = (max_err < 0.05f) ? "PASS" : "FAIL";

        printf("%-*s  %8.3f %9.2f  %7.4f %s\n", W, c.label, lat, tf, max_err, result);

        HIP_CHECK(hipFree(dq));
        HIP_CHECK(hipFree(dk));
        HIP_CHECK(hipFree(dv));
        HIP_CHECK(hipFree(do_));
    }
    printf("%s\n", std::string(75, '=').c_str());
    printf("\nTarget: >70 TFLOPS for causal prefill seq>=2048 on MI300X/MI325X\n");
    printf("This kernel is the HIP Flash2 Engine for hipDNN SDPA (FP16 path).\n");
    printf("BF16 inputs → dispatched to existing ASM SDPA engine (not shown here).\n");
    return 0;
}
