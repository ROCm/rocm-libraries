// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Is the kernel still correct when seq_len_q is NOT a multiple of qPerCta?
//
// isApplicable() enforces seq_len_q % 64 == 0 because a partial query tile
// makes __syncthreads() divergent. That gate was written when every CTA
// covered 64 queries. The dispatcher now selects 128-, 256- and 384-query
// tiles, and S=2048 is a multiple of 64 but NOT of 384 -- so such a shape
// passes the gate and then runs a partial tile. A wrong tail is silent: no
// error code, no launch failure, just wrong numbers.
//
// Pure HIP, no torch: mixing torch's bundled HIP runtime with another one
// gives hipModuleLoad -> 209 (hipErrorSharedObjectInitFailed).
//
// Checks, in increasing order of strength:
//   1. V constant       -> every output element equals that constant, since
//                          softmax weights sum to 1. Exact.
//   2. Q = 0, noncausal -> every output row equals mean(V). Exact.
//   3. CPU fp32 reference on a reduced head count, both causal and not.
//
// Build (any recent hipcc):
//   hipcc -O2 -std=c++17 partial_tile_probe.cpp -o partial_tile_probe
// Run:
//   ./partial_tile_probe <kernels_dir>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                                            \
    do                                                                                          \
    {                                                                                           \
        const hipError_t _e = (x);                                                              \
        if(_e != hipSuccess)                                                                    \
        {                                                                                       \
            printf("HIP error %d (%s) at line %d\n", (int)_e, hipGetErrorString(_e), __LINE__); \
            return 2;                                                                           \
        }                                                                                       \
    } while(0)

namespace
{

struct Variant
{
    const char* tag;
    unsigned qPerCta;
    unsigned blockDim;
};

// Must match Flash2Dispatch.hpp.
const Variant K_VARIANTS[] = {
    {"w8q3k2", 384, 512},
    {"w8q2k4", 256, 512},
    {"w8q1k4", 128, 512},
    {"w4q1k4", 64, 256},
};

__half f2h(float f)
{
    return __float2half(f);
}
float h2f(__half h)
{
    return __half2float(h);
}

// Deterministic pseudo-random in [-1, 1); no <random> so runs are reproducible
// across libstdc++ versions.
float rnd(unsigned& s)
{
    s = s * 1664525u + 1013904223u;
    return ((float)((s >> 8) & 0xFFFFFF) / 8388608.0f) - 1.0f;
}

struct Buf
{
    void* p = nullptr;
    ~Buf()
    {
        if(p)
            (void)hipFree(p);
    }
};

// Launch: the kernel takes (q,k,v,o, B,Hq,Hk,S,Skv,D, scale, causal, then 12
// int64 strides in BHSD order for q,k,v,o).
hipError_t launch(hipFunction_t fn,
                  void* q,
                  void* k,
                  void* v,
                  void* o,
                  int B,
                  int Hq,
                  int Hk,
                  int S,
                  int Skv,
                  int D,
                  int causal,
                  unsigned qPerCta,
                  unsigned blockDim)
{
    const float scale = 1.0f / std::sqrt((float)D);
    const long long sS = D, sH = (long long)S * D, sB = (long long)Hq * S * D;
    const long long kH = (long long)Skv * D, kB = (long long)Hk * Skv * D;

    void* argv[] = {&q, &k, &v, &o, &B, &Hq, &Hk, &S, &Skv, &D, (void*)&scale, &causal};
    long long st[12] = {sB, sH, sS, kB, kH, sS, kB, kH, sS, sB, sH, sS};
    void* all[24];
    for(int i = 0; i < 12; i++)
        all[i] = argv[i];
    for(int i = 0; i < 12; i++)
        all[12 + i] = &st[i];

    const unsigned gridX = ((unsigned)S + qPerCta - 1) / qPerCta;
    return hipModuleLaunchKernel(
        fn, gridX, (unsigned)B, (unsigned)Hq, blockDim, 1, 1, 0, nullptr, all, nullptr);
}

} // namespace

int main(int argc, char** argv)
{
    const std::string dir = (argc > 1) ? argv[1] : ".";
    const int D = 128;
    int failures = 0, partialRows = 0;

    printf("%-8s %-6s %-9s %-7s %-11s %-11s %-11s\n",
           "variant",
           "S",
           "tile",
           "causal",
           "constV",
           "meanV",
           "vsCPU");

    for(const auto& var : K_VARIANTS)
    {
        const std::string co = dir + "/hip_flash2_fwd_gfx942_" + var.tag + ".co";
        hipModule_t mod;
        if(hipModuleLoad(&mod, co.c_str()) != hipSuccess)
        {
            printf("%-8s (not installed, skipped)\n", var.tag);
            continue;
        }
        hipFunction_t fn;
        HIP_CHECK(hipModuleGetFunction(&fn, mod, "flash2_v7_hipdnn_d128"));

        // Small head count keeps the CPU reference tractable.
        const int B = 1, H = 2;
        for(int S : {1024, 1536, 2048, 2560, 3072, 4096})
        {
            const bool partial = (S % (int)var.qPerCta) != 0;
            const size_t n = (size_t)B * H * S * D;
            std::vector<__half> hq(n), hk(n), hv(n), ho(n);

            Buf dq, dk, dv, doo;
            HIP_CHECK(hipMalloc(&dq.p, n * 2));
            HIP_CHECK(hipMalloc(&dk.p, n * 2));
            HIP_CHECK(hipMalloc(&dv.p, n * 2));
            HIP_CHECK(hipMalloc(&doo.p, n * 2));

            for(int causal = 0; causal <= 1; causal++)
            {
                unsigned seed = 12345u + (unsigned)S + 7u * (unsigned)causal;

                // ---- 1. constant V ------------------------------------------
                for(size_t i = 0; i < n; i++)
                {
                    hq[i] = f2h(rnd(seed));
                    hk[i] = f2h(rnd(seed));
                    hv[i] = f2h(0.5f);
                }
                HIP_CHECK(hipMemcpy(dq.p, hq.data(), n * 2, hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(dk.p, hk.data(), n * 2, hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(dv.p, hv.data(), n * 2, hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(doo.p, 0, n * 2));
                HIP_CHECK(launch(fn,
                                 dq.p,
                                 dk.p,
                                 dv.p,
                                 doo.p,
                                 B,
                                 H,
                                 H,
                                 S,
                                 S,
                                 D,
                                 causal,
                                 var.qPerCta,
                                 var.blockDim));
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(ho.data(), doo.p, n * 2, hipMemcpyDeviceToHost));
                float eConst = 0.f;
                for(size_t i = 0; i < n; i++)
                    eConst = std::fmax(eConst, std::fabs(h2f(ho[i]) - 0.5f));

                // ---- 2. Q = 0, noncausal -> mean(V) --------------------------
                float eMean = -1.f;
                if(!causal)
                {
                    for(size_t i = 0; i < n; i++)
                    {
                        hq[i] = f2h(0.f);
                        hv[i] = f2h(rnd(seed));
                    }
                    HIP_CHECK(hipMemcpy(dq.p, hq.data(), n * 2, hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemcpy(dv.p, hv.data(), n * 2, hipMemcpyHostToDevice));
                    HIP_CHECK(hipMemset(doo.p, 0, n * 2));
                    HIP_CHECK(launch(fn,
                                     dq.p,
                                     dk.p,
                                     dv.p,
                                     doo.p,
                                     B,
                                     H,
                                     H,
                                     S,
                                     S,
                                     D,
                                     0,
                                     var.qPerCta,
                                     var.blockDim));
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemcpy(ho.data(), doo.p, n * 2, hipMemcpyDeviceToHost));
                    eMean = 0.f;
                    for(int h = 0; h < H; h++)
                    {
                        std::vector<float> mean(D, 0.f);
                        for(int s = 0; s < S; s++)
                            for(int d = 0; d < D; d++)
                                mean[d] += h2f(hv[((size_t)h * S + s) * D + d]);
                        for(int d = 0; d < D; d++)
                            mean[d] /= (float)S;
                        for(int s = 0; s < S; s++)
                            for(int d = 0; d < D; d++)
                                eMean = std::fmax(
                                    eMean,
                                    std::fabs(h2f(ho[((size_t)h * S + s) * D + d]) - mean[d]));
                    }
                }

                // ---- 3. CPU fp32 reference ----------------------------------
                for(size_t i = 0; i < n; i++)
                {
                    hq[i] = f2h(rnd(seed));
                    hk[i] = f2h(rnd(seed));
                    hv[i] = f2h(rnd(seed));
                }
                HIP_CHECK(hipMemcpy(dq.p, hq.data(), n * 2, hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(dk.p, hk.data(), n * 2, hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(dv.p, hv.data(), n * 2, hipMemcpyHostToDevice));
                HIP_CHECK(hipMemset(doo.p, 0, n * 2));
                HIP_CHECK(launch(fn,
                                 dq.p,
                                 dk.p,
                                 dv.p,
                                 doo.p,
                                 B,
                                 H,
                                 H,
                                 S,
                                 S,
                                 D,
                                 causal,
                                 var.qPerCta,
                                 var.blockDim));
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipMemcpy(ho.data(), doo.p, n * 2, hipMemcpyDeviceToHost));

                const float scale = 1.0f / std::sqrt((float)D);
                float eRef = 0.f;
                std::vector<float> logits(S), acc(D);
                // Sample rows: the tail rows are the interesting ones, so always
                // include the last few plus a stride across the sequence.
                std::vector<int> rows;
                for(int s = 0; s < S; s += S / 16)
                    rows.push_back(s);
                for(int s = std::max(0, S - 3); s < S; s++)
                    rows.push_back(s);
                for(int h = 0; h < H; h++)
                {
                    for(int qi : rows)
                    {
                        const int kmax = causal ? (qi + 1) : S;
                        float m = -1e30f;
                        for(int kj = 0; kj < kmax; kj++)
                        {
                            float dot = 0.f;
                            for(int d = 0; d < D; d++)
                                dot += h2f(hq[((size_t)h * S + qi) * D + d])
                                       * h2f(hk[((size_t)h * S + kj) * D + d]);
                            logits[kj] = dot * scale;
                            m = std::fmax(m, logits[kj]);
                        }
                        float sum = 0.f;
                        for(int kj = 0; kj < kmax; kj++)
                        {
                            logits[kj] = std::exp(logits[kj] - m);
                            sum += logits[kj];
                        }
                        std::fill(acc.begin(), acc.end(), 0.f);
                        for(int kj = 0; kj < kmax; kj++)
                            for(int d = 0; d < D; d++)
                                acc[d] += logits[kj] * h2f(hv[((size_t)h * S + kj) * D + d]);
                        for(int d = 0; d < D; d++)
                            eRef = std::fmax(
                                eRef,
                                std::fabs(acc[d] / sum - h2f(ho[((size_t)h * S + qi) * D + d])));
                    }
                }

                const bool bad = (eConst > 1e-2f) || (eMean > 1e-2f) || (eRef > 5e-2f);
                if(bad)
                    failures++;
                if(partial)
                    partialRows++;
                printf("%-8s %-6d %-9s %-7s %-11.3e %-11.3e %-11.3e %s\n",
                       var.tag,
                       S,
                       partial ? "PARTIAL" : "exact",
                       causal ? "yes" : "no",
                       eConst,
                       eMean,
                       eRef,
                       bad ? "  <== FAIL" : "");
                fflush(stdout);
            }
        }
        (void)hipModuleUnload(mod);
    }

    printf("\n%s: %d failing configuration(s); %d partial-tile rows exercised\n",
           failures ? "FAIL" : "PASS",
           failures,
           partialRows);
    return failures ? 1 : 0;
}
