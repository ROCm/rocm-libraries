// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "ConvWinogradNdArgs.h"
#include "ConvWinogradWmma.h"

// Rank-two residue-class F(2,3)^2 with FP32 input, transforms, GEMM and inverse.
constexpr int ND_P = 16;

__device__ inline long long
    ndOffset(const long long* s, long long n, long long c, long long h, long long w)
{
    return n * s[0] + c * s[1] + h * s[2] + w * s[3];
}
__device__ inline void ndOrigin(const KNdArgs& a, int tile, long long& n, long long* o)
{
    long long t = a.tileBegin + tile;
    for(int j = 1; j >= 0; --j)
    {
        const long long q = t % a.axisTiles[j];
        t /= a.axisTiles[j];
        o[j] = q % a.residues[j] + 2 * a.dilation[j] * (q / a.residues[j]);
    }
    n = t;
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradNdFilter(KNdArgs a)
{
    const long long ci = a.ci / a.groups, co = a.co / a.groups;
    const long long total = a.groups * co * ci;
    for(long long q = (long long)blockIdx.x * 256 + threadIdx.x; q < total;
        q += (long long)gridDim.x * 256)
    {
        const long long i = q % ci, o = (q / ci) % co, g = q / (ci * co);
        const long long k = g * co + o;
        const long long c = i;
        float f[9], u[ND_P];
        for(int h = 0; h < 3; ++h)
            for(int w = 0; w < 3; ++w)
            {
                const int hh = a.rotate ? 2 - h : h, ww = a.rotate ? 2 - w : w;
                f[h * 3 + w] = static_cast<const float*>(a.w)[ndOffset(a.wStride, k, c, hh, ww)];
            }
        k5FilterTile<2>(f, u);
        for(int p = 0; p < ND_P; ++p)
            a.u[((g * ND_P + p) * co + o) * ci + i] = u[p];
    }
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradNdInput(KNdArgs a)
{
    const long long ci = a.ci / a.groups;
    const long long total = a.ci * a.tiles;
    for(long long q = (long long)blockIdx.x * 256 + threadIdx.x; q < total;
        q += (long long)gridDim.x * 256)
    {
        const int tile = q % a.tiles;
        const long long ch = q / a.tiles, g = ch / ci, c = ch % ci;
        long long n, origin[2];
        ndOrigin(a, tile, n, origin);
        float patch[ND_P], temp[ND_P], v[ND_P];
        for(int h = 0; h < 4; ++h)
            for(int w = 0; w < 4; ++w)
            {
                const long long y = origin[0] - a.pad[0] + h * a.dilation[0];
                const long long x = origin[1] - a.pad[1] + w * a.dilation[1];
                const bool live = n < a.n && y >= 0 && y < a.input[0] && x >= 0 && x < a.input[1];
                patch[h * 4 + w]
                    = live ? static_cast<const float*>(a.in)[ndOffset(a.inStride, n, ch, y, x)]
                           : 0.0f;
            }
        for(int w = 0; w < 4; ++w)
            K5Wino<2>::bt(patch + w, 4, temp + w, 4);
        for(int h = 0; h < 4; ++h)
            K5Wino<2>::bt(temp + h * 4, 1, v + h * 4, 1);
        for(int p = 0; p < ND_P; ++p)
            a.v[((g * ND_P + p) * ci + c) * a.tiles + tile] = v[p];
    }
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradNdGemm(K5GemmArgs a)
{
    k5GemmWmma<CONV_ND_BM, CONV_ND_BN, CONV_ND_BK>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradNdOutput(KNdArgs a)
{
    const long long co = a.co / a.groups;
    const long long total = a.co * a.tiles;
    for(long long q = (long long)blockIdx.x * 256 + threadIdx.x; q < total;
        q += (long long)gridDim.x * 256)
    {
        const int tile = q % a.tiles;
        const long long ch = q / a.tiles, g = ch / co, c = ch % co;
        long long n, origin[2];
        ndOrigin(a, tile, n, origin);
        float m[ND_P], temp[8], out[4];
        for(int p = 0; p < ND_P; ++p)
            m[p] = a.product[((g * ND_P + p) * co + c) * a.tiles + tile];
        for(int w = 0; w < 4; ++w)
            K5Wino<2>::at(m + w, 4, temp + w, 4);
        for(int h = 0; h < 2; ++h)
            K5Wino<2>::at(temp + h * 4, 1, out + h * 2, 1);
        for(int h = 0; h < 2; ++h)
            for(int w = 0; w < 2; ++w)
            {
                const long long pos[2]
                    = {origin[0] + h * a.dilation[0], origin[1] + w * a.dilation[1]};
                if(n >= a.n || pos[0] >= a.output[0] || pos[1] >= a.output[1])
                    continue;
                bool live = true;
                for(int j = 0; j < 2; ++j)
                {
                    bool axisLive = false;
                    for(int r = 0; r < 3; ++r)
                    {
                        const long long x = pos[j] - a.pad[j] + r * a.dilation[j];
                        axisLive |= x >= 0 && x < a.input[j];
                    }
                    live &= axisLive;
                }
                static_cast<float*>(a.out)[ndOffset(a.outStride, n, ch, pos[0], pos[1])]
                    = live ? out[h * 2 + w] : 0.0f;
            }
    }
}
