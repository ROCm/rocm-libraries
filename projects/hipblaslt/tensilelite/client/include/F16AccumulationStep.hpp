// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <Tensile/LaunchSummationOrder.hpp>

// Self-contained (no HIP types; Tensile/LaunchSummationOrder.hpp includes nothing of Tensile) so that the hardware-free
// unit test compiles the code the reference runs.
// f16 values are passed as their 16-bit patterns.

namespace TensileLite
{
    namespace Client
    {
        // f16 accumulation must reproduce the gfx1250 f16-accumulate 8-bit-float WMMA bit for bit. It accumulates K
        // in steps of 64 consecutive products (the x128 form is two steps); one step takes the f16 accumulator C and
        // the products p = a * b of OCP e4m3 or e5m2 values a and b, each operand in its own format:
        //  - E(p) = e(a) + e(b), e the exponent in the operand's format (a subnormal counts as the format's minimum
        //    normal exponent, -6 for e4m3 and -14 for e5m2); zero products take no part.
        //  - The four quarters of a step (16 consecutive K each) are the groups; a group's products are
        //    truncated toward -inf to a multiple of 2^(Eg - 13), Eg the largest E(p) in the group.
        //  - The group sums and C are truncated toward -inf to a multiple of 2^max(E - 24, eC - 31), E the
        //    largest Eg and eC the exponent of C's leading bit; C counts as 0 when that drops 12 or more of its
        //    low bits.
        //  - The sum is rounded to f16, nearest even, overflowing to +-inf; a magnitude that rounds to 0 gives +0.
        // A NaN operand gives 0xfe3d whatever C is; an e5m2 inf times 0, or +inf and -inf products in one step, give
        // 0xfe00. Otherwise a step whose C is NaN returns C quieted (C | 0x200); inf products give inf of their sign,
        // or 0xfe00 against a C inf of the other sign; a step whose C is infinite returns C, and one without nonzero
        // products returns C, -0 as +0. Steps start at multiples of 64 in K and a short step is zero padded, so the
        // result does not depend on a solution's WMMA K.
        constexpr size_t f16AccumulationBlockK = 64;

        // x toward -inf to a multiple of 2^s (I: int64_t or __int128).
        template <typename I>
        inline I f16AccumulationFloor(I x, int s)
        {
            if(s <= 0)
                return x;
            I m = I(1) << s;
            return x >= 0 ? x - x % m : -((-x + m - 1) / m * m);
        }

        template <typename U>
        inline int f16AccumulationBitLength(U x)
        {
            int n = 0;
            for(; x; x >>= 1)
                n++;
            return n;
        }

        // The f16 nearest to v * 2^w, ties to even, overflowing to +-inf; a magnitude that rounds to 0 gives +0.
        inline uint16_t f16AccumulationRoundUnits(__int128 v, int w)
        {
            unsigned __int128 a   = v < 0 ? (unsigned __int128)(-v) : (unsigned __int128)v;
            int               lsb = std::max(f16AccumulationBitLength(a) - 11 + w, -24);
            int               sh  = lsb - w;
            unsigned __int128 q   = sh > 0 ? a >> sh : a << -sh;
            if(sh > 0)
            {
                unsigned __int128 r = a - (q << sh), half = (unsigned __int128)1 << (sh - 1);
                q += (r > half || (r == half && (q & 1))) ? 1 : 0;
            }
            if(q >= 2048)
            {
                q >>= 1;
                lsb++;
            }
            uint16_t const sign = v < 0 ? 0x8000 : 0;
            if(q == 0)
                return 0;
            if(q < 1024)
                return uint16_t(sign | unsigned(q));
            int const e = lsb + 25;
            return uint16_t(sign | (e >= 31 ? 0x7c00u : (unsigned(e) << 10) | (unsigned(q) - 1024)));
        }

        // The f16 nearest to v * 2^-24, ties to even, overflowing to +-inf.
        inline uint16_t f16AccumulationRound(int64_t v)
        {
            return f16AccumulationRoundUnits(v, -24);
        }

        // An 8-bit float operand format of the WMMA: its minimum normal exponent, which its subnormals count as,
        // and the exponent of its smallest step.
        struct F16AccumulationOperandFormat
        {
            int minExponent;
            int lsbExponent;
        };
        constexpr F16AccumulationOperandFormat f16AccumulationE4M3{-6, -9};
        constexpr F16AccumulationOperandFormat f16AccumulationE5M2{-14, -16};

        // One WMMA step on the n <= 64 products a[k] * b[k] of a block.
        inline uint16_t f16AccumulationStep(uint16_t                     c,
                                            double const*                a,
                                            double const*                b,
                                            size_t                       n,
                                            F16AccumulationOperandFormat fa = f16AccumulationE4M3,
                                            F16AccumulationOperandFormat fb = f16AccumulationE4M3)
        {
            int  eg[4]  = {};
            bool any[4] = {}, inf[2] = {}, invalid = false;
            for(size_t k = 0; k < n; k++)
            {
                if(std::isnan(a[k]) || std::isnan(b[k]))
                    return 0xfe3d;
                if(std::isinf(a[k]) || std::isinf(b[k]))
                {
                    if(a[k] == 0 || b[k] == 0)
                        invalid = true;
                    else
                        inf[std::signbit(a[k]) != std::signbit(b[k])] = true;
                    continue;
                }
                if(a[k] == 0 || b[k] == 0)
                    continue;
                int e = std::max(std::ilogb(a[k]), fa.minExponent) + std::max(std::ilogb(b[k]), fb.minExponent);
                eg[k / 16]  = any[k / 16] ? std::max(eg[k / 16], e) : e;
                any[k / 16] = true;
            }
            int ce = (c >> 10) & 31;
            if(invalid || (inf[0] && inf[1]))
                return 0xfe00;
            if(ce == 31 && (c & 1023))
                return uint16_t(c | 0x200);
            if(inf[0] || inf[1])
                return ce == 31 && bool(c & 0x8000) != inf[1] ? uint16_t(0xfe00) : uint16_t(inf[1] ? 0xfc00 : 0x7c00);
            if(ce == 31)
                return c;
            if(!(any[0] || any[1] || any[2] || any[3]))
                return c == 0x8000 ? uint16_t(0) : c;
            int E = -1000;
            for(int g = 0; g < 4; g++)
                E = any[g] ? std::max(E, eg[g]) : E;
            // Exact in __int128 in units of 2^-64: products are multiples of 2^-32 at the finest (e5m2 x e5m2), the
            // truncation steps 2^-52, and the magnitudes stay below 2^102.
            int const lsb         = fa.lsbExponent + fb.lsbExponent;
            __int128  groupSum[4] = {};
            for(size_t k = 0; k < n; k++)
                if(a[k] != 0 && b[k] != 0)
                    groupSum[k / 16] += f16AccumulationFloor(
                        __int128(std::llround(std::ldexp(a[k], -fa.lsbExponent)))
                            * std::llround(std::ldexp(b[k], -fb.lsbExponent)) * (__int128(1) << (lsb + 64)),
                        eg[k / 16] - 13 + 64);
            int64_t cv = ce == 0 ? int64_t(c & 1023) : int64_t((c & 1023) | 1024) << (ce - 1);
            // Truncation to 2^(s - 24): s = max(E, eC - 7), eC = bitlength(cv) - 25 the exponent of C's leading bit.
            int      s     = cv ? std::max(E, f16AccumulationBitLength(uint64_t(cv)) - 32) : E;
            __int128 total = 0;
            for(int g = 0; g < 4; g++)
                total += f16AccumulationFloor(groupSum[g], s + 40);
            if(s - (ce == 0 ? 0 : ce - 1) < 12)
                total += f16AccumulationFloor(__int128(c & 0x8000 ? -cv : cv) * (__int128(1) << 40), s + 40);
            return f16AccumulationRoundUnits(total, -64);
        }

        inline float f16AccumulationToFloat(uint16_t h)
        {
            uint32_t sign = uint32_t(h & 0x8000) << 16;
            uint32_t e = (h >> 10) & 31, m = h & 1023;
            float    f;
            if(e == 0)
                f = std::ldexp(float(m), -24);
            else
            {
                uint32_t u = e == 31 ? 0x7f800000 | (m << 13) : ((e + 112) << 23) | (m << 13);
                std::memcpy(&f, &u, sizeof(f));
            }
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            u |= sign;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }

        // f32 to f16, nearest even, overflowing to +-inf, as v_cvt_f16_f32.
        inline uint16_t f16AccumulationFromFloat(float f)
        {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            uint16_t sign = uint16_t((u >> 16) & 0x8000);
            uint32_t a    = u & 0x7fffffff;
            if(a > 0x7f800000)
                return sign | 0x7e00;
            if(a >= 0x477ff000)
                return sign | 0x7c00;
            uint32_t e = a >> 23;
            if(e < 102)
                return sign;
            uint32_t q, rem, half;
            if(e < 113)
            {
                uint32_t mant  = (a & 0x7fffff) | 0x800000;
                int      shift = 126 - int(e);
                q              = mant >> shift;
                rem            = mant & ((1u << shift) - 1);
                half           = 1u << (shift - 1);
            }
            else
            {
                uint32_t bits = a - (112u << 23);
                q             = bits >> 13;
                rem           = bits & 0x1fff;
                half          = 0x1000;
            }
            q += (rem > half || (rem == half && (q & 1))) ? 1 : 0;
            return sign | uint16_t(q);
        }

        // The MultipleBuffer reduction: the f16 partials of the split-K slices are added in f32 in slice order
        // and the sum is rounded once to f16.
        inline uint16_t f16AccumulationReduce(uint16_t const* partials, size_t gsu)
        {
            float sum = f16AccumulationToFloat(partials[0]);
            for(size_t g = 1; g < gsu; g++)
                sum += f16AccumulationToFloat(partials[g]);
            return f16AccumulationFromFloat(sum);
        }

        // The StreamK tree fixup: at level L partial i absorbs the sum of partial i + 2^L when i is a multiple of
        // 2^(L+1), in f32; the sum is rounded once to f16.
        inline uint16_t f16AccumulationTreeReduce(uint16_t const* partials, size_t n)
        {
            std::vector<float> sum(n);
            for(size_t i = 0; i < n; i++)
                sum[i] = f16AccumulationToFloat(partials[i]);
            for(size_t step = 1; step < n; step *= 2)
                for(size_t i = 0; i + step < n; i += 2 * step)
                    sum[i] += sum[i + step];
            return f16AccumulationFromFloat(sum[0]);
        }

        // DefaultWGM: the workgroup that workgroup (wg0, wg1) of an n0 x n1 grid becomes. For |wgm| > 1 the grid is
        // walked in blocks of |wgm| columns (rows when wgm < 0), the last block as wide as what is left.
        inline std::pair<size_t, size_t>
            f16AccumulationDefaultWgm(int32_t wgm, size_t n0, size_t n1, size_t wg0, size_t wg1)
        {
            if(wgm > -2 && wgm < 2)
                return {wg0, wg1};
            bool const   rows    = wgm < 0;
            size_t const w       = static_cast<size_t>(rows ? -wgm : wgm);
            size_t const first   = rows ? wg1 : wg0;
            size_t const second  = rows ? wg0 : wg1;
            size_t const nFirst  = rows ? n1 : n0;
            size_t const nSecond = rows ? n0 : n1;
            size_t const block   = second / w;
            size_t const serial  = first + (second % w) * nFirst;
            size_t const width   = block >= nSecond / w ? (nSecond % w ? nSecond % w : w) : w;
            size_t const newFirst  = serial / width;
            size_t const newSecond = serial % width + block * w;
            return rows ? std::pair<size_t, size_t>{newSecond, newFirst}
                        : std::pair<size_t, size_t>{newFirst, newSecond};
        }

        // SpaceFillingCurveWalk: the workgroup of in-batch tile `index` of an n0 x n1 grid. The index is first spread
        // over 8 XCCs (chiplet_transform); with more than one level the grid is cut into the multi-level block and
        // three edge blocks walked with level 0 only. Orders 0 and 1 are both emitted column-major.
        inline std::pair<size_t, size_t> f16AccumulationSpaceFilling(
            uint32_t algo, uint32_t wgmWord, size_t n0, size_t n1, size_t index)
        {
            size_t const levels = algo >> 28;
            if(levels == 0 || levels > 3)
                throw std::runtime_error("f16 accumulation: SpaceFillingAlgo needs 1 to 3 levels");
            auto const order = [&](size_t lvl) { return static_cast<int>((algo >> (4 * lvl)) & 0xF); };

            size_t const total = n0 * n1;
            index              = index % 8 * (total / 8) + std::min(index % 8, total % 8) + index / 8;

            auto const walk = [&](int id, size_t M, size_t N, size_t li) -> std::pair<size_t, size_t> {
                if(id == 0 || id == 1)
                    return {li % M, li / M};
                if(id < 2 || id > 5)
                    throw std::runtime_error("f16 accumulation: unknown SpaceFillingAlgo order");
                static int const hilbert[8][4] = {{0, 1, 3, 2}, {2, 3, 1, 0}, {3, 2, 0, 1}, {1, 0, 2, 3},
                                                  {2, 0, 1, 3}, {3, 1, 0, 2}, {0, 1, 3, 2}, {1, 3, 2, 0}};
                static int const next[8][4]    = {{7, 0, 0, 5}, {4, 1, 1, 6}, {6, 2, 2, 4}, {5, 3, 3, 7},
                                                  {1, 4, 4, 3}, {2, 5, 5, 0}, {3, 6, 6, 1}, {0, 7, 7, 2}};
                static int const morton[3][4]  = {{0, 2, 1, 3}, {0, 1, 2, 3}, {0, 1, 3, 2}};
                size_t y = 0, x = 0;
                int    dir = 0;
                while(M * N > 1)
                {
                    auto const half = [](size_t L) {
                        size_t const floor = size_t(1) << f16AccumulationBitLength(L) >> 1;
                        return L == floor ? L / 2 : floor;
                    };
                    size_t const M1 = half(M), N1 = half(N);
                    size_t const dims[4][2] = {{M1, N1}, {M - M1, N1}, {M1, N - N1}, {M - M1, N - N1}};
                    size_t const offs[4][2] = {{0, 0}, {M1, 0}, {0, N1}, {M1, N1}};
                    int const*   blocks     = id == 2 ? hilbert[dir] : morton[id - 3];
                    for(int j = 0; j < 4; j++)
                    {
                        int const    b    = blocks[j];
                        size_t const size = dims[b][0] * dims[b][1];
                        if(j < 3 && li >= size)
                        {
                            li -= size;
                            continue;
                        }
                        M = dims[b][0];
                        N = dims[b][1];
                        y += offs[b][0];
                        x += offs[b][1];
                        if(id == 2)
                            dir = next[dir][j];
                        break;
                    }
                }
                return {y, x};
            };

            auto const transform = [&](size_t idx, size_t M, size_t N, size_t numLevels, size_t gy, size_t gx) {
                numLevels = std::min(numLevels, levels);
                if(numLevels == 1)
                {
                    auto const l = walk(order(0), M, N, idx % (M * N));
                    return std::pair<size_t, size_t>{gy + l.first, gx + l.second};
                }
                size_t   cumDen = 1, cumX = 1, cumY = 1;
                uint32_t word   = wgmWord;
                for(size_t lvl = 0; lvl < numLevels; lvl++)
                {
                    size_t bM, bN;
                    if(lvl != numLevels - 1)
                    {
                        bM = word & 0xff;
                        bN = (word >> 8) & 0xff;
                    }
                    else
                    {
                        bM = (wgmWord & 0xff) * (numLevels == 3 ? (wgmWord >> 16) & 0xff : 1);
                        bN = ((wgmWord >> 8) & 0xff) * (numLevels == 3 ? (wgmWord >> 24) & 0xff : 1);
                        if(bM == 0 || bN == 0)
                            throw std::runtime_error("f16 accumulation: the SpaceFillingAlgo WGM word has a zero grid");
                        bM = M / bM;
                        bN = N / bN;
                    }
                    if(bM == 0 || bN == 0)
                        throw std::runtime_error("f16 accumulation: the SpaceFillingAlgo WGM word has a zero grid");
                    size_t const li = (lvl == 0 ? idx : idx / cumDen) % (bM * bN);
                    cumDen *= bM * bN;
                    auto const l = walk(order(lvl), bM, bN, li);
                    gx += l.second * cumX;
                    gy += l.first * cumY;
                    if(lvl != numLevels - 1)
                    {
                        cumX *= bN;
                        cumY *= bM;
                        word >>= 16;
                    }
                }
                return std::pair<size_t, size_t>{gy, gx};
            };

            if(levels == 1)
                return transform(index, n0, n1, 1, 0, 0);
            size_t M1 = (wgmWord & 0xff) * (levels == 3 ? (wgmWord >> 16) & 0xff : 1);
            size_t N1 = ((wgmWord >> 8) & 0xff) * (levels == 3 ? (wgmWord >> 24) & 0xff : 1);
            if(M1 == 0 || N1 == 0)
                throw std::runtime_error("f16 accumulation: the SpaceFillingAlgo WGM word has a zero grid");
            size_t const M2 = n0 % M1, N2 = n1 % N1;
            M1 = n0 - M2;
            N1 = n1 - N2;
            if(index < M1 * N1)
                return transform(index, M1, N1, levels, 0, 0);
            index -= M1 * N1;
            if(index < M2 * N1)
                return transform(index, M2, N1, 1, M1, 0);
            index -= M2 * N1;
            if(index < M2 * N2)
                return transform(index, M2, N2, 1, M1, N1);
            return transform(index - M2 * N2, M1, N2, 1, 0, N1);
        }

        // Unroll iterations [start, end) of a tile.
        struct F16AccumulationChunk
        {
            size_t start = 0;
            size_t end   = 0;

            bool operator==(F16AccumulationChunk const& other) const
            {
                return start == other.start && end == other.end;
            }
        };

        // The chunks of linear tile t in K order, which is also the order of the workgroups that run them.
        inline std::vector<F16AccumulationChunk> f16AccumulationStreamKChunks(LaunchSummationOrder const& sk,
                                                                              size_t                        t)
        {
            size_t const                      I = std::max<size_t>(sk.itersPerTile, 1);
            std::vector<F16AccumulationChunk> chunks;
            // n pieces of w iterations, the first I - n * w of them one longer.
            auto pieces = [&](size_t n, size_t w) {
                size_t const rem = I - n * w;
                for(size_t p = 0; p < n; p++)
                {
                    size_t const start = p * w + std::min(p, rem);
                    chunks.push_back({start, start + w + (p < rem ? 1 : 0)});
                }
            };
            bool const split = t + sk.skTiles >= sk.tiles;
            if(sk.persistent == LaunchSummationOrder::Persistent::StreamK && split)
            {
                if(sk.perTileExtraIters && sk.skTiles != 0 && sk.grid % sk.skTiles == 0)
                    pieces(sk.grid / sk.skTiles, sk.skItersPerWG);
                else
                {
                    // The stream-k iterations of all split tiles in a row: workgroup w runs W + 1 of them when
                    // w < E, else W.
                    size_t const W = sk.skItersPerWG, E = sk.extraIters;
                    size_t const tileStart = (t + sk.skTiles - sk.tiles) * I, tileEnd = tileStart + I;
                    size_t const longEnd = E * (W + 1);
                    for(size_t i = tileStart; i < tileEnd;)
                    {
                        size_t end = i < longEnd ? (i / (W + 1) + 1) * (W + 1)
                                                 : longEnd + ((i - longEnd) / W + 1) * W;
                        end        = std::min(end, tileEnd);
                        chunks.push_back({i - tileStart, end - tileStart});
                        i = end;
                    }
                }
            }
            else if(sk.persistent == LaunchSummationOrder::Persistent::StreamKParallel)
                pieces(sk.skSplit, sk.skItersPerWG);
            else if(sk.persistent == LaunchSummationOrder::Persistent::StreamKDynamic && split)
                for(size_t p = 0; p < sk.skSplit; p++)
                    chunks.push_back({p * sk.skItersPerWI, std::min((p + 1) * sk.skItersPerWI, I)});
            if(chunks.empty())
                chunks.push_back({0, I});
            for(auto const& c : chunks)
                if(c.start >= c.end)
                    throw std::runtime_error("f16 accumulation: a StreamK workgroup has no iterations of its tile");
            return chunks;
        }

        // The linear tile index at each workgroup wg0 + wg1 * tiles0 + batch * tiles0 * tiles1. Tile t runs at workgroup
        // (t % tiles0, t / tiles0 % tiles1) of batch t / (tiles0 * tiles1), moved by DefaultWGM with wgm, or by the
        // SpaceFillingAlgo walk with the levels packed in spaceFillingAlgo (4 bits per level, the level count in bits
        // 28-31) and the WGM word.
        inline std::vector<size_t> f16AccumulationTileOrder(LaunchSummationOrder const& sk)
        {
            size_t const        perBatch = sk.tiles0 * sk.tiles1;
            std::vector<size_t> tileOf(sk.tiles, sk.tiles);
            for(size_t t = 0; t < sk.tiles; t++)
            {
                auto const wg = sk.spaceFilling
                                    ? f16AccumulationSpaceFilling(
                                        sk.spaceFillingAlgo, sk.wgmWord, sk.tiles0, sk.tiles1, t % perBatch)
                                    : f16AccumulationDefaultWgm(
                                        sk.wgm, sk.tiles0, sk.tiles1, t % perBatch % sk.tiles0, t % perBatch / sk.tiles0);
                size_t& slot = tileOf.at(wg.first + wg.second * sk.tiles0 + t / perBatch * perBatch);
                if(slot != sk.tiles)
                    throw std::runtime_error("f16 accumulation: two tiles map to one workgroup");
                slot = t;
            }
            return tileOf;
        }

        // What decides the K order of a launch: the solution's DepthU, macro tile and PrefetchGlobalRead, and the
        // LaunchSummationOrder the host computes for the launch.
        struct F16AccumulationLaunch : LaunchSummationOrder
        {
            size_t depthU             = f16AccumulationBlockK;
            size_t macroTile0         = 0;
            size_t macroTile1         = 0;
            int    prefetchGlobalRead = 0;
        };

        // One split-K slice of a tile: its loop runs `iterations` unroll iterations, loop index j covering the
        // DepthU consecutive K of unroll iteration first + j * stride; the loop starts at j = staggerStart and
        // wraps from the last index to 0. A slice with `tail` then runs the K % DepthU remainder.
        struct F16AccumulationSlice
        {
            size_t iterations   = 0;
            size_t first        = 0;
            size_t stride       = 1;
            size_t staggerStart = 0;
            bool   tail         = false;
        };

        // Tile (wg0, wg1) is the kernel's WorkGroup0/WorkGroup1 after its workgroup remapping; numTiles0 is its
        // NumWorkGroups0.
        inline F16AccumulationSlice f16AccumulationSlice(F16AccumulationLaunch const& launch,
                                                         size_t                       sizeK,
                                                         uint32_t                     slice,
                                                         uint32_t                     wg0,
                                                         uint32_t                     wg1,
                                                         uint32_t                     numTiles0)
        {
            uint32_t const numIter = uint32_t(sizeK / launch.depthU);
            uint32_t const gsu     = std::max<uint32_t>(launch.gsu, 1);
            uint32_t const q = numIter / gsu, r = numIter % gsu;

            F16AccumulationSlice s;
            s.iterations = q + (slice < r ? 1 : 0);
            if(gsu > 1 && launch.gsuCoalesced)
                s.first = slice < r ? size_t(q + 1) * slice : size_t(q) * slice + r;
            else if(gsu > 1)
            {
                s.first  = slice;
                s.stride = gsu;
            }
            uint32_t const tailSlice = gsu == 1 ? 0 : !launch.gsuCoalesced ? r : (q == 0 ? r : gsu - 1);
            s.tail                   = sizeK % launch.depthU != 0 && slice == tailSlice;

            // StaggerU fields: count in bits [7:0], stride shift [12:8], mapping [15:13]; 32-bit arithmetic.
            uint32_t const n     = uint32_t(s.iterations);
            uint32_t const shift = launch.staggerUStrideShift & 31;
            uint32_t       cur   = launch.staggerU & 0xff;
            if(launch.prefetchGlobalRead >= 3 && n <= uint32_t(launch.prefetchGlobalRead - 1))
                cur = 0;
            while(n < uint32_t(cur << shift))
                cur >>= 1;
            uint32_t const mask = cur >= 1 ? cur - 1 : 0;
            uint32_t       input;
            switch(launch.staggerUMapping)
            {
            case 0:
                input = wg0;
                break;
            case 1:
                input = wg1;
                break;
            case 3:
                // The serial index is summed into the register that holds the stagger count.
                input = cur + numTiles0 * wg1 + wg0;
                break;
            case 2: // WorkGroup2 only with more than two batch indices
            case 4:
                input = 0xffffffff;
                break;
            default:
                input = cur;
                break;
            }
            uint32_t const start = (mask & input) << shift;
            if(n != 0 && start >= n)
                throw std::runtime_error("f16 accumulation: StaggerU start beyond the slice's loop");
            s.staggerStart = n ? start : 0;
            return s;
        }

        // Calls visit(k, n) for the 64-K steps of one slice of a tile in the kernel's order: n K from k, where
        // only the tail's last step can have n < 64.
        template <typename Visit>
        inline void f16AccumulationForEachStep(F16AccumulationLaunch const& launch,
                                               size_t                       sizeK,
                                               uint32_t                     slice,
                                               uint32_t                     wg0,
                                               uint32_t                     wg1,
                                               uint32_t                     numTiles0,
                                               Visit&&                      visit)
        {
            F16AccumulationSlice s = f16AccumulationSlice(launch, sizeK, slice, wg0, wg1, numTiles0);
            for(size_t t = 0; t < s.iterations; t++)
            {
                size_t it = s.first + (s.staggerStart + t) % s.iterations * s.stride;
                for(size_t k = it * launch.depthU; k < (it + 1) * launch.depthU; k += f16AccumulationBlockK)
                    visit(k, f16AccumulationBlockK);
            }
            if(s.tail)
                for(size_t k = sizeK / launch.depthU * launch.depthU; k < sizeK; k += f16AccumulationBlockK)
                    visit(k, std::min(f16AccumulationBlockK, sizeK - k));
        }

        inline std::vector<size_t> f16AccumulationSliceSteps(F16AccumulationLaunch const& launch,
                                                             size_t                       sizeK,
                                                             uint32_t                     slice,
                                                             uint32_t                     wg0,
                                                             uint32_t                     wg1,
                                                             uint32_t                     numTiles0)
        {
            std::vector<size_t> steps;
            f16AccumulationForEachStep(
                launch, sizeK, slice, wg0, wg1, numTiles0, [&](size_t k, size_t) { steps.push_back(k); });
            return steps;
        }

        // Calls visit(k, n) for the 64-K steps of StreamK chunk c of a tile in the kernel's order. A whole tile runs
        // the unsplit loop with its StaggerU start; any other chunk runs its unroll iterations in order with no
        // stagger, and the chunk that ends the tile also runs the tail.
        template <typename Visit>
        inline void f16AccumulationForEachChunkStep(F16AccumulationLaunch const& launch,
                                                    size_t                       sizeK,
                                                    F16AccumulationChunk         c,
                                                    uint32_t                     wg0,
                                                    uint32_t                     wg1,
                                                    uint32_t                     numTiles0,
                                                    Visit&&                      visit)
        {
            size_t const I = std::max<size_t>(launch.itersPerTile, 1);
            if(c.start == 0 && c.end == I)
            {
                F16AccumulationLaunch whole = launch;
                whole.gsu                   = 1;
                f16AccumulationForEachStep(whole, sizeK, 0, wg0, wg1, numTiles0, visit);
                return;
            }
            size_t const full = sizeK / launch.depthU;
            for(size_t it = c.start; it < std::min(c.end, full); it++)
                for(size_t k = it * launch.depthU; k < (it + 1) * launch.depthU; k += f16AccumulationBlockK)
                    visit(k, f16AccumulationBlockK);
            if(c.end == I && sizeK % launch.depthU != 0)
                for(size_t k = full * launch.depthU; k < sizeK; k += f16AccumulationBlockK)
                    visit(k, std::min(f16AccumulationBlockK, sizeK - k));
        }

        // Equal for two launches that take K in the same order. A launch without split-K, StaggerU or split tiles takes
        // K from 0 upward whatever its other fields are; any other launch is keyed on every field it has.
        inline std::vector<uint64_t> f16AccumulationOrderKey(F16AccumulationLaunch const& l)
        {
            if(l.gsu <= 1 && l.staggerU == 0 && l.persistent <= LaunchSummationOrder::Persistent::DataParallel)
                return {};
            return {l.depthU,
                    l.macroTile0,
                    l.macroTile1,
                    static_cast<uint64_t>(l.prefetchGlobalRead),
                    l.gsu,
                    l.gsuCoalesced,
                    l.staggerU,
                    l.staggerUStrideShift,
                    l.staggerUMapping,
                    static_cast<uint64_t>(l.persistent),
                    l.fixupTree,
                    l.perTileExtraIters,
                    l.tiles,
                    l.itersPerTile,
                    l.grid,
                    l.skTiles,
                    l.skItersPerWG,
                    l.extraIters,
                    l.skSplit,
                    l.skItersPerWI,
                    l.tiles0,
                    l.tiles1,
                    static_cast<uint32_t>(l.wgm),
                    l.spaceFilling,
                    l.spaceFillingAlgo,
                    l.wgmWord};
        }
    } // namespace Client
} // namespace TensileLite
