/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Self-contained (no HIP types) so that the hardware-free unit test compiles the code the reference runs.
// f16 values are passed as their 16-bit patterns.

namespace TensileLite
{
    namespace Client
    {
        // f16 accumulation must reproduce the gfx1250 f16-accumulate fp8 WMMA bit for bit. It accumulates K in
        // steps of 64 consecutive products (the x128 form is two steps); one step takes the f16 accumulator C and
        // the products p = a * b of OCP e4m3 values a and b:
        //  - E(p) = e(a) + e(b), e the fp8 exponent (-6 for a subnormal); zero products take no part.
        //  - The four quarters of a step (16 consecutive K each) are the groups; a group's products are
        //    truncated toward -inf to a multiple of 2^(Eg - 13), Eg the largest E(p) in the group.
        //  - The group sums and C are truncated toward -inf to a multiple of 2^(E - 24), E the largest Eg; C
        //    counts as 0 when that drops 12 or more of its low bits.
        //  - The sum is rounded to f16, nearest even, overflowing to +-inf.
        // A NaN operand gives NaN. Otherwise a step without nonzero products, or whose C is infinite or NaN,
        // leaves C unchanged. Steps start at multiples of 64 in K and a short step is zero padded, so the result
        // does not depend on a solution's WMMA K.
        constexpr size_t f16AccumulationBlockK = 64;

        // x toward -inf to a multiple of 2^s.
        inline int64_t f16AccumulationFloor(int64_t x, int s)
        {
            if(s <= 0)
                return x;
            int64_t m = int64_t(1) << s;
            return x >= 0 ? x - x % m : -((-x + m - 1) / m * m);
        }

        inline int f16AccumulationBitLength(uint64_t x)
        {
            int n = 0;
            for(; x; x >>= 1)
                n++;
            return n;
        }

        // The f16 nearest to v * 2^-24, ties to even, overflowing to +-inf.
        inline uint16_t f16AccumulationRound(int64_t v)
        {
            uint64_t a  = v < 0 ? uint64_t(-v) : uint64_t(v);
            int      sh = std::max(f16AccumulationBitLength(a) - 11, 0);
            uint64_t q  = a >> sh;
            if(sh > 0)
            {
                uint64_t r = a - (q << sh), half = uint64_t(1) << (sh - 1);
                q += (r > half || (r == half && (q & 1))) ? 1 : 0;
            }
            uint64_t mag = q << sh;
            uint16_t u;
            if(mag >= (uint64_t(1) << 40))
                u = 0x7c00;
            else if(mag < 1024)
                u = uint16_t(mag);
            else
            {
                int e = f16AccumulationBitLength(mag) - 10;
                u     = uint16_t((e << 10) | ((mag >> (e - 1)) - 1024));
            }
            return u | (v < 0 ? 0x8000 : 0);
        }

        // One WMMA step on the n <= 64 products a[k] * b[k] of a block.
        inline uint16_t f16AccumulationStep(uint16_t c, double const* a, double const* b, size_t n)
        {
            int  eg[4]  = {};
            bool any[4] = {};
            for(size_t k = 0; k < n; k++)
            {
                if(std::isnan(a[k]) || std::isnan(b[k]))
                    return 0x7e00;
                if(a[k] == 0 || b[k] == 0)
                    continue;
                int e       = std::max(std::ilogb(a[k]), -6) + std::max(std::ilogb(b[k]), -6);
                eg[k / 16]  = any[k / 16] ? std::max(eg[k / 16], e) : e;
                any[k / 16] = true;
            }
            int ce = (c >> 10) & 31;
            if(!(any[0] || any[1] || any[2] || any[3]) || ce == 31)
                return c;
            int E = -1000;
            for(int g = 0; g < 4; g++)
                E = any[g] ? std::max(E, eg[g]) : E;
            // Products in units of 2^-18 and sums in units of 2^-24 are exact in int64.
            int64_t groupSum[4] = {};
            for(size_t k = 0; k < n; k++)
                if(a[k] != 0 && b[k] != 0)
                    groupSum[k / 16] += f16AccumulationFloor(std::llround(a[k] * 512) * std::llround(b[k] * 512),
                                                             eg[k / 16] - 13 + 18);
            int64_t total = 0;
            for(int g = 0; g < 4; g++)
                total += f16AccumulationFloor(groupSum[g] * 64, E);
            int64_t cv = ce == 0 ? int64_t(c & 1023) : int64_t((c & 1023) | 1024) << (ce - 1);
            if(E - (ce == 0 ? 0 : ce - 1) < 12)
                total += f16AccumulationFloor(c & 0x8000 ? -cv : cv, E);
            return f16AccumulationRound(total);
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

        // How a persistent launch splits the unroll iterations of each tile, from the arguments the host packed.
        struct F16AccumulationStreamK
        {
            enum Kind : uint32_t
            {
                None,
                DataParallel, // whole tiles
                Static, // two-tile StreamK with the fixup in the kernel: the last skTiles tiles are split
                Parallel, // static StreamK reduced by the reduction kernel: `split` pieces per tile
                Dynamic // work queue: the last skTiles tiles in `split` pieces of skItersPerWI
            };
            uint32_t kind          = None;
            bool     tree          = false;
            bool     perTileExtras = false;
            size_t   tiles         = 0;
            size_t   itersPerTile  = 1;
            size_t   grid          = 0;
            size_t   skTiles       = 0;
            size_t   skItersPerWG  = 0;
            size_t   extraIters    = 0;
            size_t   split         = 0;
            size_t   skItersPerWI  = 0;
            // Tile t runs at workgroup (t % tiles0, t / tiles0 % tiles1) of batch t / (tiles0 * tiles1), moved by
            // DefaultWGM with wgm, or by the SpaceFillingAlgo walk with the levels packed in spaceFillingAlgo
            // (4 bits per level, the level count in bits 28-31) and the WGM word.
            size_t   tiles0           = 1;
            size_t   tiles1           = 1;
            int32_t  wgm              = 1;
            bool     spaceFilling     = false;
            uint32_t spaceFillingAlgo = 0;
            uint32_t wgmWord          = 0;
        };

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
        inline std::vector<F16AccumulationChunk> f16AccumulationStreamKChunks(F16AccumulationStreamK const& sk,
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
            if(sk.kind == F16AccumulationStreamK::Static && split)
            {
                if(sk.perTileExtras && sk.skTiles != 0 && sk.grid % sk.skTiles == 0)
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
            else if(sk.kind == F16AccumulationStreamK::Parallel)
                pieces(sk.split, sk.skItersPerWG);
            else if(sk.kind == F16AccumulationStreamK::Dynamic && split)
                for(size_t p = 0; p < sk.split; p++)
                    chunks.push_back({p * sk.skItersPerWI, std::min((p + 1) * sk.skItersPerWI, I)});
            if(chunks.empty())
                chunks.push_back({0, I});
            for(auto const& c : chunks)
                if(c.start >= c.end)
                    throw std::runtime_error("f16 accumulation: a StreamK workgroup has no iterations of its tile");
            return chunks;
        }

        // The linear tile index at each workgroup wg0 + wg1 * tiles0 + batch * tiles0 * tiles1.
        inline std::vector<size_t> f16AccumulationTileOrder(F16AccumulationStreamK const& sk)
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

        // What decides the K order of a launch: the solution's DepthU, macro tile and PrefetchGlobalRead, the GSU
        // and StaggerU fields of the internalArgs word the host passed, and how a persistent launch splits tiles.
        struct F16AccumulationLaunch
        {
            size_t   depthU              = f16AccumulationBlockK;
            size_t   macroTile0          = 0;
            size_t   macroTile1          = 0;
            int      prefetchGlobalRead  = 0;
            uint32_t gsu                 = 1;
            bool     gsuCoalesced        = false;
            uint32_t staggerU            = 0;
            uint32_t staggerUStrideShift = 0;
            uint32_t staggerUMapping     = 0;

            F16AccumulationStreamK streamK;
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
            size_t const I = std::max<size_t>(launch.streamK.itersPerTile, 1);
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

        // Equal for two launches exactly when every tile of a problem gets the same order.
        inline std::vector<uint64_t> f16AccumulationOrderKey(F16AccumulationLaunch const& launch)
        {
            bool const split   = launch.gsu > 1;
            bool const stagger = (launch.staggerU & 0xff) != 0;
            auto const& sk     = launch.streamK;
            bool const  skSplit = sk.kind > F16AccumulationStreamK::DataParallel;
            if(!split && !stagger && !skSplit)
                return {};
            uint32_t const        mapping = launch.staggerUMapping;
            bool const            wg0     = stagger && (mapping == 0 || mapping == 3);
            bool const            wg1     = stagger && (mapping == 1 || mapping == 3);
            std::vector<uint64_t> key     = {
                launch.depthU,
                split ? launch.gsu : 1u,
                split && launch.gsuCoalesced,
                stagger ? launch.staggerU & 0xff : 0u,
                stagger ? launch.staggerUStrideShift & 31 : 0u,
                stagger ? mapping : 0u,
                wg0 ? launch.macroTile0 : 0,
                wg1 ? launch.macroTile1 : 0,
                uint64_t(stagger && launch.prefetchGlobalRead >= 3 ? launch.prefetchGlobalRead : 0)};
            if(skSplit)
                key.insert(key.end(),
                           {sk.kind,
                            sk.tree,
                            sk.perTileExtras,
                            sk.tiles,
                            sk.itersPerTile,
                            sk.grid,
                            sk.skTiles,
                            sk.skItersPerWG,
                            sk.extraIters,
                            sk.split,
                            sk.skItersPerWI,
                            sk.tiles0,
                            sk.tiles1,
                            static_cast<uint32_t>(sk.wgm),
                            sk.spaceFilling,
                            sk.spaceFilling ? sk.spaceFillingAlgo : 0u,
                            sk.spaceFilling ? sk.wgmWord : 0u});
            return key;
        }
    } // namespace Client
} // namespace TensileLite
