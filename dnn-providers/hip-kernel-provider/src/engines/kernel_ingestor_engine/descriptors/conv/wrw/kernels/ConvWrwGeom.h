// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// LDS arithmetic of the backward-weights kernels (K3), shared by the host pack (to gate a
// config on the device's LDS) and by the kernels (which assert their tiles match it). Builtin
// types and constexpr functions only: HIPRTC compiles it as an embedded include.
#pragma once

// NOLINTBEGIN(modernize-use-auto,readability-avoid-nested-conditional-operator,readability-braces-around-statements,readability-implicit-bool-conversion,readability-isolate-declaration)
// HIPRTC compiles this file as an embedded include, with no standard library: C arrays,
// macros and C-style code are deliberate, and its arithmetic types are part of the kernels.

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

/// The x tile's row pitch (elements) for rows of ph positions plus two of slack: whole 16-byte
/// units (skew elements). Channels-first at unit width stride: an odd number of them, so the
/// rows a fragment read takes (one per lane) start in different bank groups; an even count put
/// every other row (12 units: dilation 24 at 32 positions) or every row (16 units) on the same
/// banks. Strided tiles keep the even counts (the bump would push some past the LDS), and so do
/// the channels-last tiles, whose transposing stores ran up to 1.5x slower with the bump
/// (output channel counts below a vector: VAE k3 / k4 heads).
constexpr int k3CfPitch(int ph, int skew, int sw, bool channelsFirst)
{
    const int units = k3DivUp(ph + 2, skew) + 1;
    return (sw == 1 && channelsFirst ? units | 1 : units) * skew;
}

/// The matrix-core ring's channels-last tile row pitch (elements of eb bytes) for n positions,
/// with fragment reads of fb bytes a lane: a whole number of fb-wide dword groups, an odd count
/// of them. The rows one read's lanes take then start in distinct bank groups: an 8-byte read
/// of 16 lanes or a 4-byte read of 32 covers the 32 banks once. The transposing stores (a dword
/// a channel row, positions across lanes) stay within two passes.
constexpr int k3ClPitch(int n, int eb, int fb)
{
    const int g = fb >= 16 ? 4 : fb >= 8 ? 2 : 1;
    int d = k3DivUp(k3DivUp(n * eb, 4), g) * g;
    if((d / g) % 2 == 0)
        d += g;
    return d * 4 / eb;
}

/// The generic loop's window row pitch (elements of eb bytes) for wpos positions (a whole number
/// of 16-byte units): at least wpos + 16 bytes, an odd number of 16-byte units (a row stride of
/// 4 mod 8 dwords keeps the fragment reads of consecutive rows off each other's banks).
constexpr int k3WinPitch(int wpos, int eb)
{
    const int unit = 16 / eb;
    int p = wpos + unit;
    if((p / unit) % 2 == 0)
        p += unit;
    return p;
}

// Direct loop (5): rows of c channels in LDS, [column][channel] with the 4-channel groups of
// column x XOR-rotated by 4 g(x), g taking three bits of x (candidate r: bits 3,1,0 / 1,3,0 /
// 0,1,3, masked to c / 16 - 1). A rotation is conflict-free when every half-wave of every
// ds_read_b64_tr_b16 of a 16 x 32 fragment (16-lane groups at columns 8 apart, 4 lanes of
// 4-channel groups per column), at every column shift the loop reads, hits 32 distinct 8-byte
// slots modulo 32.
constexpr int k3dBit(int r, int i)
{
    return r == 0   ? (i == 0   ? 3
                       : i == 1 ? 1
                                : 0)
           : r == 1 ? (i == 0   ? 1
                       : i == 1 ? 3
                                : 0)
                    : (i == 0   ? 0
                       : i == 1 ? 1
                                : 3);
}

constexpr int k3dRot(int r, int c, int x)
{
    const unsigned p = static_cast<unsigned>(x);
    return static_cast<int>(((p >> k3dBit(r, 0)) & 1u) | (((p >> k3dBit(r, 1)) & 1u) << 1)
                            | (((p >> k3dBit(r, 2)) & 1u) << 2))
           & (c / 16 - 1);
}

constexpr bool k3dConflictFree(int r, int c, int shifts)
{
    const int c4 = c / 4;
    for(int cb = 0; cb < c4; cb += 4)
        for(int sh = 0; sh < shifts; ++sh)
            for(int ph = 0; ph < 2; ++ph)
                for(int rd = 0; rd < 2; ++rd)
                {
                    unsigned seen = 0;
                    for(int l = ph * 32; l < ph * 32 + 32; ++l)
                    {
                        const int x = (l / 16) * 8 + rd * 4 + (l % 16) / 4 + sh;
                        const int off = (x * c4 + ((cb + l % 4) ^ (4 * k3dRot(r, c, x)))) % 32;
                        if(seen & (1u << off))
                            return false;
                        seen |= 1u << off;
                    }
                }
    return true;
}

/// The first conflict-free rotation for rows of c channels read at column shifts 0 .. shifts -
/// 1, or -1: c must be a power of two from 32 to 512.
constexpr int k3dRotation(int c, int shifts)
{
    if(c < 32 || c > 512 || (c & (c - 1)) != 0)
        return -1;
    for(int r = 0; r < 3; ++r)
        if(k3dConflictFree(r, c, shifts))
            return r;
    return -1;
}

/// Columns of a direct-loop x row: the strip's 32 plus the taps' reach, rounded up to whole
/// 16-byte DMA rounds of the row (512 elements).
constexpr int k3dXCols(int bc, int s, int dw)
{
    return k3DivUp((32 + (s - 1) * dw) * bc, 512) * 512 / bc;
}

/// 16-byte DMA loads a direct-loop wave issues per row (x and dy rounds over the item's waves).
constexpr int k3dLoads(int bc, int bk, int wq, int s, int dw)
{
    const int nw = 8 / wq;
    return k3DivUp(k3dXCols(bc, s, dw) * bc / 512, nw) + k3DivUp(32 * bk / 512, nw);
}

/// Static LDS bytes: loops 0/1 matrix-core double buffers; 2 DMA slots; 3 FMA dy ring and
/// double x windows; 4 FMA row double buffers; 5 direct ring; 6 small-C row double buffers;
/// 7 packed-row double buffers; 8 generic multi-tap double buffers; 9 zero LDS; 11 one skewed
/// matrix tile; 12 that tile plus 32 bytes per cached position. Loop 10's FP32 staging formula
/// is in ldsBytesOf. eb: element bytes; fr: filter rows, or the generic loop's taps a block.
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
                               int ns,
                               int wq = 0,
                               int fb = 0)
{
    if(loop == 9)
        return 0;
    if(loop == 12)
        return static_cast<long long>(bk + bc) * (bq + 16 / eb) * eb
               + static_cast<long long>(bq) * 32;
    if(loop == 11)
        return static_cast<long long>(bk + bc) * (bq + 16 / eb) * eb;
    if(loop == 8)
    {
        // Generic taps: two buffers of a bk-row dy chunk and fr (taps a block) bc-row x tiles,
        // rows of bq positions padded by 16 bytes.
        // The FMA form (tk > 0): [position][channel] tiles, rows of bk / bc channels + 16
        // bytes, fr tiles of bq positions or (dw > 0) one window of bq + dw.
        if(tk > 0)
        {
            const long long u = 16 / eb;
            const long long xRows = dw > 0 ? bq + dw : static_cast<long long>(fr) * bq;
            return 2 * (static_cast<long long>(bq) * (bk + u) + xRows * (bc + u)) * eb;
        }
        // dw > 0: one shared window of bq + dw positions per x row instead of fr tiles.
        const long long ld = bq + 16 / eb;
        if(dw > 0)
        {
            return 2
                   * (static_cast<long long>(bk) * ld
                      + static_cast<long long>(bc) * k3WinPitch(bq + dw, eb))
                   * eb;
        }
        return 2 * (static_cast<long long>(bk) + static_cast<long long>(fr) * bc) * ld * eb;
    }
    if(loop == 5)
    {
        // ns ring slots of wq items' x and dy rows plus a drain round; the waves_q sum's
        // fp32 staging (wq - 1 items x the item's waves x s fc fk float4 per lane).
        const long long slot
            = static_cast<long long>(wq) * (k3dXCols(bc, s, dw) * bc + 32 * bk) + 512;
        const long long fc = bc / (16 * wc), fk = bk / (16 * wk);
        const long long red
            = wq > 1 ? static_cast<long long>(wq - 1) * (8 / wq) * s * fc * fk * 64 * 16 : 0;
        return ns * slot * eb + red;
    }
    if(loop == 4)
    {
        const int win = (bq - 1) * sw + (s - 1) * dw + 1;
        const int nbv = k3DivUp(win, vec) + (vec > 1 ? 1 : 0);
        const long long aElems = static_cast<long long>(bq) * (bk + 4);
        const long long bElems = static_cast<long long>(fr) * nbv * vec * (bc + 4);
        return 2 * (aElems + bElems) * eb;
    }
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
    // 6: small-C rows, 7: packed rows, both on loop 0's tiles.
    const int rows = loop == 0 || loop == 6 || loop == 7 ? fr : 1;
    // Small-C rows stage channels-last operands into the channels-first tiles.
    layout = loop == 6 ? 0 : layout;
    const bool tr = layout == 2;
    const int kv = layout == 1 ? 4 / eb : 1;
    const int win = (bq - 1) * sw + (s - 1) * dw + 1;
    const int nbv = k3DivUp(win + (align == 1 ? vec - 1 : 0), vec);
    const int phRaw = layout == 0 ? nbv * vec / sw : k3DivUp(win, sw);
    const int ph = k3DivUp(phRaw, kv) * kv;
    const int skew = 16 / eb;
    // fb: the ring's matrix-core fragment bytes on CDNA (k3ClPitch); 0 elsewhere.
    const bool cl = loop == 1 && layout == 1 && fb > 0;
    const int lda = tr ? k3TrPitch(bk) : cl ? k3ClPitch(bq, eb, fb) : bq + skew;
    const int ldb = tr   ? k3TrPitch(bc)
                    : cl ? k3ClPitch(ph + 2, eb, fb)
                         : k3CfPitch(ph, skew, sw, layout == 0);
    const long long aElems = static_cast<long long>(tr ? bq : bk) * lda + (align == 1 ? vec : 0);
    const long long bElems = static_cast<long long>(rows) * sw * (tr ? ph : bc) * ldb;
    // The ring's stages of wq steps: a slot per step.
    const long long slots = loop == 1 && wq > 1 ? wq : 1;
    return 2 * slots * (aElems + bElems) * eb;
}

// NOLINTEND(modernize-use-auto,readability-avoid-nested-conditional-operator,readability-braces-around-statements,readability-implicit-bool-conversion,readability-isolate-declaration)
