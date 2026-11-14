// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Stream-K parallel reduction test pair: HHS NT, macrotile 64x64x32.
// Main workgroup w owns K split w % F of tile w / F. With F == 1 it applies
// the epilogue to D; otherwise it stores its f32 partial tile into slot w.
// The streamKReduce kernel sums the F slots of one band of a tile in
// ascending order and applies the epilogue.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define NAME c_hip_sk_parallel_Cijk_Ailk_Bjlk_HHS_BH_MT64x64x32_gfx950
#define CAT(a, b) a##b
#define POST(n) CAT(n, _streamKReduce)

constexpr unsigned MT0 = 64, MT1 = 64, DU = 32, THREADS = 256, BANDS = 2;
constexpr unsigned COLS = MT1 * MT0 / THREADS; // columns per thread in main

struct Tile
{
    unsigned b, i0, j0;
};

__device__ Tile tileOf(unsigned t, unsigned M, unsigned N)
{
    unsigned tm = (M + MT0 - 1) / MT0, tn = (N + MT1 - 1) / MT1;
    unsigned r  = t % (tm * tn);
    return {t / (tm * tn), (r % tm) * MT0, (r / tm) * MT1};
}

__device__ void store(__half* D, __half const* C, size_t d, size_t c, float acc, float alpha,
                      float beta)
{
    float v = alpha * acc;
    if(beta != 0.0f)
        v += beta * __half2float(C[c]);
    D[d] = __float2half(v);
}

extern "C" __global__ __launch_bounds__(THREADS) void NAME(
    __half* D, __half const* C, __half const* A, __half const* B, char* W,
    unsigned M, unsigned N, unsigned K, unsigned lda, unsigned strideA, unsigned ldb,
    unsigned strideB, unsigned ldc, unsigned strideC, unsigned ldd, unsigned strideD,
    float alpha, float beta, unsigned F, unsigned numTiles, unsigned wsOffset)
{
    unsigned w = blockIdx.x, split = w % F;
    if(w / F >= numTiles)
        return;
    Tile     t = tileOf(w / F, M, N);
    unsigned i = threadIdx.x % MT0, jl = threadIdx.x / MT0 * COLS;
    unsigned kTiles = (K + DU - 1) / DU;
    unsigned k0 = split * kTiles / F * DU, k1 = min((split + 1) * kTiles / F * DU, K);

    float acc[COLS] = {};
    if(t.i0 + i < M)
    {
        __half const* a = A + size_t(t.b) * strideA + t.i0 + i;
        __half const* b = B + size_t(t.b) * strideB + t.j0 + jl;
        for(unsigned l = k0; l < k1; ++l)
        {
            float x = __half2float(a[size_t(l) * lda]);
            for(unsigned c = 0; c < COLS; ++c)
                if(t.j0 + jl + c < N)
                    acc[c] += x * __half2float(b[size_t(l) * ldb + c]);
        }
    }

    if(F == 1)
    {
        for(unsigned c = 0; c < COLS; ++c)
        {
            unsigned ii = t.i0 + i, jj = t.j0 + jl + c;
            if(ii < M && jj < N)
                store(D, C, size_t(t.b) * strideD + size_t(jj) * ldd + ii,
                      size_t(t.b) * strideC + size_t(jj) * ldc + ii, acc[c], alpha, beta);
        }
        return;
    }
    float* slot = reinterpret_cast<float*>(W + wsOffset) + size_t(w) * MT0 * MT1;
    for(unsigned c = 0; c < COLS; ++c)
        slot[(jl + c) * MT0 + i] = acc[c];
}

extern "C" __global__ __launch_bounds__(THREADS) void POST(NAME)(
    __half* D, __half const* C, char const* W, unsigned M, unsigned N, unsigned ldc,
    unsigned strideC, unsigned ldd, unsigned strideD, float alpha, float beta, unsigned F,
    unsigned numTiles, unsigned wsOffset)
{
    unsigned tile = blockIdx.x / BANDS, band = blockIdx.x % BANDS;
    if(tile >= numTiles)
        return;
    Tile         t     = tileOf(tile, M, N);
    float const* slots = reinterpret_cast<float const*>(W + wsOffset) + size_t(tile) * F * MT0 * MT1;
    constexpr unsigned BAND = MT0 * MT1 / BANDS;
    for(unsigned e = band * BAND + threadIdx.x; e < (band + 1) * BAND; e += THREADS)
    {
        unsigned ii = t.i0 + e % MT0, jj = t.j0 + e / MT0;
        if(ii >= M || jj >= N)
            continue;
        float acc = 0.0f;
        for(unsigned s = 0; s < F; ++s)
            acc += slots[size_t(s) * MT0 * MT1 + e];
        store(D, C, size_t(t.b) * strideD + size_t(jj) * ldd + ii,
              size_t(t.b) * strideC + size_t(jj) * ldc + ii, acc, alpha, beta);
    }
}
