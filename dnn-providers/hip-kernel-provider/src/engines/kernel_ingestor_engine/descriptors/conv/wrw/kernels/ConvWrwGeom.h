// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// LDS arithmetic of the backward-weights kernels (K3), shared by the host pack (to gate a
// config on the device's LDS) and by the kernels (which assert their tiles match it). Builtin
// types and constexpr functions only: HIPRTC compiles it as an embedded include.
#pragma once

/// A [position][channel] row pitch (16-bit elements) at least n + 8 whose transpose reads do not
/// conflict in LDS banks (gfx950, measured: a pitch % 128 of 0, 8, 32 or 96 halves the rate).
constexpr int k3TrPitch(int n)
{
    int p = n + 8;
    while(p % 128 == 0 || p % 128 == 8 || p % 128 == 32 || p % 128 == 96)
        p += 8;
    return p;
}

constexpr int k3DivUp(int a, int b)
{
    return (a + b - 1) / b;
}

/// Static LDS bytes of a config. loop 0/1 matrix core (K3Tiles, two buffers), 2 LDS DMA (NS
/// slots), 3 FMA (an R + 1 row dy ring and two x windows). eb: element bytes; wave: lanes per
/// wave; fr: the filter rows a loop-0 block stages, the filter height otherwise.
constexpr long long k3LdsBytes(int loop,
                               int layout,
                               int eb,
                               int bk,
                               int bc,
                               int bq,
                               int fr,
                               int s,
                               int sw,
                               int dw,
                               int wk,
                               int wc,
                               [[maybe_unused]] int tk,
                               [[maybe_unused]] int tc,
                               int vec,
                               int align,
                               int ns)
{
    if(loop == 3)
    {
        const int win = bq + (s - 1) * dw;
        const int nbv = k3DivUp(win, vec) + 1;
        const long long aElems = static_cast<long long>(bq) * (bk + 4);
        const long long bElems = static_cast<long long>(nbv) * vec * (bc + 4);
        return ((fr + 1) * aElems + 2 * bElems) * eb;
    }
    const int waves = wk * wc;
    if(loop == 2 && layout == 2)
    {
        const int na = k3TrPitch(bk) / 8, nb = k3TrPitch(bc) / 8;
        const int win = bq + (s - 1) * dw;
        const int aPasses = k3DivUp(k3DivUp(bq * na, 64), waves);
        const int bPasses = k3DivUp(k3DivUp(win * nb, 64), waves);
        return static_cast<long long>(ns) * (aPasses + bPasses) * waves * 64 * 8 * eb;
    }
    if(loop == 2)
    {
        const int v = 16 / eb;
        const int nqa = bq / v + 1;
        const int win = bq + (s - 1) * dw;
        const int nbv = k3DivUp(win + 2, v) | 1;
        const int aPasses = k3DivUp(k3DivUp(bk * nqa, 64), waves);
        const int bPasses = k3DivUp(k3DivUp(bc * nbv, 64), waves);
        return static_cast<long long>(ns) * (aPasses + bPasses) * waves * 64 * v * eb;
    }
    const int rows = loop == 0 ? fr : 1;
    const bool tr = layout == 2;
    const int kv = layout == 1 ? 4 / eb : 1;
    const int win = (bq - 1) * sw + (s - 1) * dw + 1;
    const int nbv = k3DivUp(win + (align ? vec - 1 : 0), vec);
    const int phRaw = layout == 0 ? nbv * vec / sw : k3DivUp(win, sw);
    const int ph = k3DivUp(phRaw, kv) * kv;
    const int skew = 16 / eb;
    const int lda = tr ? k3TrPitch(bk) : bq + skew;
    const int ldb = tr ? k3TrPitch(bc) : k3DivUp(ph + 2, skew) * skew + skew;
    const long long aElems = static_cast<long long>(tr ? bq : bk) * lda + (align ? vec : 0);
    const long long bElems = static_cast<long long>(rows) * sw * (tr ? ph : bc) * ldb;
    return 2 * (aElems + bElems) * eb;
}
