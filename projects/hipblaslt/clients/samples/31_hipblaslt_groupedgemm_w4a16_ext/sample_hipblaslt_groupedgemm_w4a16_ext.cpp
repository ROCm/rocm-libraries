// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Grouped w4a16 (MoE experts) with per-group n written on the device:
//   D_g = dequant(A_g) * B_g, g = 0..G-1, m and k shared by all groups
//   A_g: HIP_R_4I, unsigned with zero-points (UNSIGNED_BIAS8 + VEC32_16BF_ZP)
//   B_g: HIP_R_16BF, n_g columns, n_g <= maxN and possibly 0
//
// The problem is set up on the host with n = maxN for every group, which sizes
// the grid. A device kernel then writes the real n_g into the UserArguments
// before each run, so the call needs no host sync and can be graph-captured.
// The zero-point pointer goes in UserArguments::scaleC; the host defaults
// already hold it.

#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CHECK_HIP(x)                                                                         \
    do                                                                                       \
    {                                                                                        \
        hipError_t e = (x);                                                                  \
        if(e != hipSuccess)                                                                  \
        {                                                                                    \
            std::cerr << __FILE__ << ":" << __LINE__ << " " << hipGetErrorString(e) << "\n"; \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while(0)

#define CHECK_LT(x)                                                                        \
    do                                                                                     \
    {                                                                                      \
        hipblasStatus_t s = (x);                                                           \
        if(s != HIPBLAS_STATUS_SUCCESS)                                                    \
        {                                                                                  \
            std::cerr << __FILE__ << ":" << __LINE__ << " hipblaslt status " << s << "\n"; \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while(0)

static uint16_t f32_to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
static float bf16_to_f32(uint16_t h)
{
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float    f;
    std::memcpy(&f, &u, 4);
    return f;
}

__global__ void updateN(uint32_t groups, hipblaslt_ext::UserArguments* args, const int32_t* n)
{
    uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g < groups)
        args[g].n = n[g];
}

constexpr int64_t groupSize   = 32;
constexpr size_t  zpAlignment = 256; // Zero-points start at the next 256 B boundary.

struct Group
{
    std::vector<uint8_t>  aRaw; // nibble per element, [m][k]
    std::vector<uint8_t>  scaleZp; // scales [m][kGroups] bf16, then packed zero-points
    std::vector<uint16_t> b; // [maxN][k]
    void *                dA, *dScale, *dB, *dD;
};

// Returns 0 on pass.
static int runCase(hipblasLtHandle_t           handle,
                   int64_t                     m,
                   int64_t                     k,
                   int64_t                     maxN,
                   const std::vector<int32_t>& nDev,
                   int                         iters,
                   bool                        totalBound = false)
{
    // totalBound: the host problem is n = {R, 1, ..., 1} with R the total rows,
    // which sizes the grid for any split of R rows; each group's buffers then
    // hold only its own n_g columns.
    const int     G       = int(nDev.size());
    auto cols = [&](int g) { return totalBound ? std::max<int64_t>(1, nDev[g]) : maxN; };
    const int64_t kGroups = k / groupSize;
    const size_t  scaleB  = size_t(m) * kGroups * 2;
    const size_t  zpOff   = (scaleB + zpAlignment - 1) / zpAlignment * zpAlignment;
    const size_t  zpBytes = size_t((m + 7) / 8) * kGroups * 4;
    const uint16_t sentinel = f32_to_bf16(1234.0f);

    std::mt19937                          rng(4321);
    std::uniform_int_distribution<int>    nib(0, 15);
    std::uniform_real_distribution<float> sd(0.02f, 0.08f), bd(-1.0f, 1.0f);

    std::vector<Group> grp(G);
    for(int gi = 0; gi < G; gi++)
    {
        auto& g = grp[gi];
        g.aRaw.resize(size_t(m) * k);
        for(auto& v : g.aRaw)
            v = uint8_t(nib(rng));
        g.scaleZp.assign(zpOff + zpBytes, 0);
        auto* s = reinterpret_cast<uint16_t*>(g.scaleZp.data());
        for(size_t i = 0; i < size_t(m) * kGroups; i++)
            s[i] = f32_to_bf16(sd(rng));
        uint8_t* zp = g.scaleZp.data() + zpOff;
        for(int64_t r = 0; r < m; r++)
            for(int64_t q = 0; q < kGroups; q++)
            {
                size_t e = 8 * (size_t(r / 8) * kGroups + q) + (r & 7);
                uint8_t z = uint8_t(nib(rng));
                zp[e / 2] = (e & 1) ? uint8_t((zp[e / 2] & 0x0F) | (z << 4))
                                    : uint8_t((zp[e / 2] & 0xF0) | z);
            }
        g.b.resize(size_t(cols(gi)) * k);
        for(auto& v : g.b)
            v = f32_to_bf16(bd(rng));

        std::vector<uint8_t> packed(size_t(m) * k / 2);
        for(size_t i = 0; i < packed.size(); i++)
            packed[i] = uint8_t(g.aRaw[2 * i] | (g.aRaw[2 * i + 1] << 4));
        std::vector<uint16_t> dInit(size_t(m) * cols(gi), sentinel);

        CHECK_HIP(hipMalloc(&g.dA, packed.size()));
        CHECK_HIP(hipMalloc(&g.dScale, g.scaleZp.size()));
        CHECK_HIP(hipMalloc(&g.dB, g.b.size() * 2));
        CHECK_HIP(hipMalloc(&g.dD, dInit.size() * 2));
        CHECK_HIP(hipMemcpy(g.dA, packed.data(), packed.size(), hipMemcpyHostToDevice));
        CHECK_HIP(
            hipMemcpy(g.dScale, g.scaleZp.data(), g.scaleZp.size(), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(g.dB, g.b.data(), g.b.size() * 2, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(g.dD, dInit.data(), dInit.size() * 2, hipMemcpyHostToDevice));
    }

    // TN: A is k x m (lda = k), B is k x n (ldb = k), D is m x n (ldd = m).
    std::vector<int64_t> vm(G, m), vn(G, maxN), vk(G, k), vb(G, 1), vlda(G, k), vldb(G, k),
        vldc(G, m), vldd(G, m), vsa(G, m * k), vsb(G, maxN * k), vsc(G, m * maxN),
        vsd(G, m * maxN);
    if(totalBound)
    {
        int64_t R = 0;
        for(int32_t v : nDev)
            R += v;
        if(R > maxN)
            return 1;
        std::fill(vn.begin(), vn.end(), 1);
        vn[0] = R;
    }
    float                                   alpha = 1.0f, beta = 0.0f;
    std::vector<hipblaslt_ext::GemmEpilogue> epi(G);
    std::vector<hipblaslt_ext::GemmInputs>   in(G);
    for(int g = 0; g < G; g++)
    {
        epi[g].setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_ZP_EXT);
        in[g].setA(grp[g].dA);
        in[g].setB(grp[g].dB);
        in[g].setC(grp[g].dD);
        in[g].setD(grp[g].dD);
        in[g].setAlpha(&alpha);
        in[g].setBeta(&beta);
        in[g].setScaleA(grp[g].dScale);
    }
    hipblaslt_ext::GemmProblemType ptype(HIPBLAS_OP_T,
                                         HIPBLAS_OP_N,
                                         HIP_R_4I,
                                         HIP_R_16BF,
                                         HIP_R_16BF,
                                         HIP_R_16BF,
                                         HIPBLAS_COMPUTE_32F);
    ptype.setInt4EncodingA(HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT);

    hipblaslt_ext::GroupedGemm gg(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_4I, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
        HIPBLAS_COMPUTE_32F);
    CHECK_LT(gg.setProblem(
        vm, vn, vk, vb, vlda, vldb, vldc, vldd, vsa, vsb, vsc, vsd, epi, in, ptype));

    const size_t                  wsSize = 32 << 20;
    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(wsSize);
    std::vector<hipblasLtMatmulHeuristicResult_t> heur;
    CHECK_LT(gg.algoGetHeuristic(1, pref, heur));
    if(heur.empty())
    {
        std::cerr << "no grouped w4a16 solution\n";
        return 1;
    }

    void *dWs, *dUA, *dN;
    CHECK_HIP(hipMalloc(&dWs, wsSize));
    CHECK_HIP(hipMalloc(&dUA, G * sizeof(hipblaslt_ext::UserArguments)));
    CHECK_HIP(hipMalloc(&dN, G * sizeof(int32_t)));
    // Split-K solutions need workspace; initialize() checks against this bound.
    gg.setMaxWorkspaceBytes(wsSize);
    CHECK_LT(gg.initialize(heur[0].algo, dWs));

    std::vector<hipblaslt_ext::UserArguments> ua(G);
    CHECK_LT(gg.getDefaultValueForDeviceUserArguments(ua.data()));
    CHECK_HIP(hipMemcpy(dUA, ua.data(), G * sizeof(ua[0]), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dN, nDev.data(), G * sizeof(int32_t), hipMemcpyHostToDevice));

    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    auto launch = [&]() {
        updateN<<<(G + 255) / 256, 256, 0, stream>>>(
            G, static_cast<hipblaslt_ext::UserArguments*>(dUA), static_cast<int32_t*>(dN));
        CHECK_LT(gg.run(dUA, stream));
    };
    launch();
    CHECK_HIP(hipStreamSynchronize(stream));

    // Time first, then validate the last run: split-K kernels must leave their
    // synchronizer counters ready for the next call.
    double us = 0;
    if(iters > 0)
    {
        hipEvent_t t0, t1;
        CHECK_HIP(hipEventCreate(&t0));
        CHECK_HIP(hipEventCreate(&t1));
        for(int i = 0; i < 10; i++)
            launch();
        CHECK_HIP(hipEventRecord(t0, stream));
        for(int i = 0; i < iters; i++)
            launch();
        CHECK_HIP(hipEventRecord(t1, stream));
        CHECK_HIP(hipEventSynchronize(t1));
        float ms;
        CHECK_HIP(hipEventElapsedTime(&ms, t0, t1));
        us = 1000.0 * ms / iters;
    }

    // Validate the first n_g columns; the rest must keep the sentinel.
    double maxErr = 0, maxRef = 0;
    bool   untouched = true;
    for(int g = 0; g < G; g++)
    {
        std::vector<uint16_t> got(size_t(m) * cols(g));
        CHECK_HIP(hipMemcpy(got.data(), grp[g].dD, got.size() * 2, hipMemcpyDeviceToHost));
        const auto*    s  = reinterpret_cast<const uint16_t*>(grp[g].scaleZp.data());
        const uint8_t* zp = grp[g].scaleZp.data() + zpOff;
        std::vector<float> deq(size_t(m) * k);
        for(int64_t r = 0; r < m; r++)
            for(int64_t c = 0; c < k; c++)
            {
                const int64_t q = c / groupSize;
                const size_t  e = 8 * (size_t(r / 8) * kGroups + q) + (r & 7);
                const int     z = (zp[e / 2] >> (4 * (e & 1))) & 0xF;
                const float   v = float(int(grp[g].aRaw[size_t(r) * k + c]) - z)
                                * bf16_to_f32(s[size_t(r) * kGroups + q]);
                deq[size_t(r) * k + c] = bf16_to_f32(f32_to_bf16(v));
            }
        for(int64_t j = 0; j < cols(g); j++)
            for(int64_t i = 0; i < m; i++)
            {
                const uint16_t out = got[size_t(j) * m + i];
                if(j >= nDev[g])
                {
                    untouched &= out == sentinel;
                    continue;
                }
                double acc = 0;
                for(int64_t c = 0; c < k; c++)
                    acc += double(deq[size_t(i) * k + c])
                           * bf16_to_f32(grp[g].b[size_t(j) * k + c]);
                maxErr = std::max(maxErr, std::fabs(bf16_to_f32(out) - acc));
                maxRef = std::max(maxRef, std::fabs(acc));
            }
    }
    // A few bf16 ulps of the largest output.
    const bool ok = untouched && maxErr <= std::ldexp(maxRef, -6);


    std::cout << "grouped w4a16 m=" << m << " k=" << k << " G=" << G << " maxN=" << maxN
              << (totalBound ? " host n={R,1,..}" : "")
              << " n={";
    for(int g = 0; g < G; g++)
        std::cout << nDev[g] << (g + 1 < G ? "," : "}");
    std::cout << " max_abs_err=" << maxErr << " max|ref|=" << maxRef
              << " untouched=" << (untouched ? "yes" : "NO");
    if(iters > 0)
        std::cout << " us/call(incl. n update)=" << us;
    std::cout << (ok ? "  PASS\n" : "  FAIL\n");

    CHECK_HIP(hipStreamDestroy(stream));
    for(void* p : {dWs, dUA, dN})
        CHECK_HIP(hipFree(p));
    for(auto& g : grp)
        for(void* p : {g.dA, g.dScale, g.dB, g.dD})
            CHECK_HIP(hipFree(p));
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    // Default: Qwen3.6-35B-A3B MoE decode shapes (8 active experts), with empty groups.
    int iters = argc > 1 ? std::atoi(argv[1]) : 200;

    hipblasLtHandle_t handle;
    CHECK_LT(hipblasLtCreate(&handle));
    int fails = 0;
    // gate_up and down decode: every group n = 1 on the host, two set to 0 on the device.
    fails += runCase(handle, 1024, 2048, 1, {1, 1, 0, 1, 1, 0, 1, 1}, iters);
    fails += runCase(handle, 2048, 512, 1, {1, 1, 0, 1, 1, 0, 1, 1}, iters);
    // All eight at n = 1 (the decode step).
    fails += runCase(handle, 1024, 2048, 1, std::vector<int32_t>(8, 1), iters);
    fails += runCase(handle, 2048, 512, 1, std::vector<int32_t>(8, 1), iters);
    // More groups than the 16 synchronizer slots that grouped split-K kernels need.
    fails += runCase(handle, 1024, 2048, 1, std::vector<int32_t>(20, 1), iters);
    // Up to 2 tokens per expert, with empty groups.
    fails += runCase(handle, 1024, 2048, 2, {2, 0, 1, 2, 2, 0, 1, 2}, iters);
    // A token count whose nearest tile is split-K, with more groups than its 16 slots.
    fails += runCase(handle, 1024, 2048, 1, std::vector<int32_t>(60, 1), iters);
    // Host bound on the total rows only: n = {R, 1, ..., 1} on the host, any
    // split of R rows (empty groups included) on the device.
    fails += runCase(handle, 1024, 2048, 16, {5, 0, 3, 1, 0, 4, 2, 1}, 0, true);
    fails += runCase(handle, 2048, 512, 16, {5, 0, 3, 1, 0, 4, 2, 1}, 0, true);
    {
        // Prefill-like: 64 experts, uneven counts, the last group the largest.
        std::mt19937                       r(7);
        std::uniform_int_distribution<int> c(0, 9);
        std::vector<int32_t>               n(64);
        for(auto& v : n)
            v = c(r) < 3 ? 0 : c(r);
        n.back() = 61;
        int32_t R = 0;
        for(int32_t v : n)
            R += v;
        fails += runCase(handle, 1024, 2048, R, n, 0, true);
        fails += runCase(handle, 2048, 512, R, n, 0, true);
    }
    {
        // vLLM-like prefill on the device path: all 256 experts as groups, about
        // a fifth of them empty, ~3.3k routed rows, host bound {R, 1, ..., 1}. Timed.
        std::mt19937                         r(11);
        std::geometric_distribution<int>     c(1.0 / 16);
        std::uniform_int_distribution<int>   e(0, 4);
        std::vector<int32_t>                 n(256);
        for(auto& v : n)
            v = e(r) == 0 ? 0 : std::min(c(r) + 1, 160);
        int32_t R = 0;
        for(int32_t v : n)
            R += v;
        fails += runCase(handle, 1024, 2048, R, n, iters > 0 ? 20 : 0, true);
        fails += runCase(handle, 2048, 512, R, n, iters > 0 ? 20 : 0, true);
    }
    // Uneven groups below the host bound, including empty ones.
    fails += runCase(handle, 1024, 2048, 32, {3, 0, 17, 1, 0, 32, 8, 5}, 0);
    fails += runCase(handle, 2048, 512, 32, {3, 0, 17, 1, 0, 32, 8, 5}, 0);
    CHECK_LT(hipblasLtDestroy(handle));
    std::cout << (fails ? "FAILED\n" : "done\n");
    return fails ? 1 : 0;
}
