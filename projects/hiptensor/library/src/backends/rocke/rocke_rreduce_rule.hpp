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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// The rules of rocKE's tuned reduction classes (rreduce): knob validity, the launch geometry, the
// route records of the pack and the untuned choice. Header-only so that hipTensor's planner
// (rocke_plan.cpp) and the C++ pack builder (pack/rocke_pack_builder.cpp: --rreduce-picks, the
// route check of the packs it writes) run the same code; rocKE's statement of the rules is
// library/kernels/common/hiptensor_aot/rreduce_host.py (the functions named in the comments), and
// rocKE's pack cross-check test compares the two choices.

#ifndef HIPTENSOR_ROCKE_RREDUCE_RULE_HPP
#define HIPTENSOR_ROCKE_RREDUCE_RULE_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace hiptensor
{
    namespace rockebackend
    {
        namespace rrule
        {
            constexpr int      kRWave           = 64;
            constexpr int      kRMaxRank        = 8;
            constexpr int64_t  kRGridYZMax      = 65535;
            constexpr uint64_t kRWorkspaceAlign = 256u;

            inline int64_t rCeilDiv(int64_t a, int64_t b)
            {
                return (a + b - 1) / b;
            }

            // FNV-1a 64 of a route key (the pack builder hashes the keys of reduce.tuned.<hash>)
            inline uint64_t rFnv1a64(std::string const& text)
            {
                uint64_t hash = 0xCBF29CE484222325ull;
                for(unsigned char c : text)
                {
                    hash = (hash ^ c) * 0x100000001B3ull;
                }
                return hash;
            }

            // ------------------------------------------- tuned reduction classes (rreduce)
            //
            // rocKE's native strided reduction bodies with the geometry as kernel arguments
            // (library/kernels/common/hiptensor_aot/rreduce.py; the host rules of
            // rreduce_host.py). A class fixes the operator, the data types, the mode and the knobs
            // (block, vector width, threads per output / along the kept run, unroll, thread rows,
            // streaming loads); the split is a launch argument. A problem of rocKE's reduction
            // route table runs with its tuned knobs (pack routes reduce.tuned.<hash>); any other
            // problem with the knobs of the untuned choice (the nearest measured prototypes,
            // pack routes reduce.proto.<i>).

            constexpr char    kRreduce[]           = "rreduce";
            constexpr char    kRreduceFinal[]      = "rreduce_final";
            constexpr int64_t kRreduceFinalLanes[] = {1, 4, 16, 64};
            constexpr int64_t kRreduceFinalTarget  = int64_t(1) << 20;
            constexpr int64_t kRreduceSlotCounts[] = {1, 2, 3, 4, 8};
            constexpr int64_t kI32Limit            = int64_t(1) << 31;
            constexpr int64_t kMaxGridThreads      = (int64_t(1) << 32) - 1;
            // heuristic candidates offered per problem (the first is the rule's choice)
            constexpr size_t kRreduceCandidates = 3;
            // the knobs (block, vec, lanes, unroll, rows) that run every problem of a mode (row:
            // one thread per output; col: one thread per kept element and one reduction row per
            // thread), and the wave count (of a 256-CU device) their split aims for
            constexpr int64_t kRreduceUniversalRow[5] = {256, 1, 1, 4, 0};
            constexpr int64_t kRreduceUniversalCol[5] = {256, 1, 1, 4, 1};
            constexpr int64_t kRreduceUniversalWaves  = 4096;

            // A launch's knobs (rreduce_host.Knobs)
            struct RKnobs
            {
                bool    row    = true;
                int64_t block  = 256;
                int64_t vec    = 1;
                int64_t lanes  = 64;
                int64_t unroll = 1;
                int64_t split  = 1;
                int64_t rows   = 0;
                bool    nt     = false;
            };

            // A reduction in the canonical form of the classes: kept modes (extent, A stride,
            // D stride) and reduced modes (extent, A stride), innermost first; C has D's strides.
            struct RProblem
            {
                std::vector<std::array<int64_t, 3>> kept;
                std::vector<std::array<int64_t, 2>> red;
                std::string                         a, d, op;
                int64_t                             eb = 2; // A's element bytes
            };

            template <typename T>
            inline int64_t rExtents(std::vector<T> const& dims)
            {
                int64_t n = 1;
                for(auto const& x : dims)
                {
                    n *= x[0];
                }
                return n;
            }

            // rreduce_host.mode_of: row when A's innermost mode is reduced
            inline bool rRow(RProblem const& p)
            {
                if(rExtents(p.red) <= 1)
                {
                    return false;
                }
                if(rExtents(p.kept) == 1)
                {
                    return true;
                }
                return std::llabs(p.red[0][1]) < std::llabs(p.kept[0][1]);
            }

            // rreduce_host.col_group: kept modes contiguous in A from the innermost one on
            inline size_t rColGroup(RProblem const& p)
            {
                if(p.kept.empty())
                {
                    return 0;
                }
                size_t n = 1;
                while(n < p.kept.size() && p.kept[n][1] == p.kept[n - 1][1] * p.kept[n - 1][0])
                {
                    n++;
                }
                return n;
            }

            // rreduce_host.chunk_dims: the reduced slots of a row class in `vec` chunks
            inline std::vector<std::array<int64_t, 2>> rChunkDims(RProblem const& p, int64_t vec)
            {
                std::vector<std::array<int64_t, 2>> dims;
                std::array<int64_t, 2> const        first{p.red[0][0] / vec, p.red[0][1] * vec};
                if(first[0] != 1)
                {
                    dims.push_back(first);
                }
                for(size_t i = 1; i < p.red.size(); i++)
                {
                    if(p.red[i][0] != 1)
                    {
                        dims.push_back(p.red[i]);
                    }
                }
                if(dims.empty())
                {
                    dims.push_back({1, 0});
                }
                return dims;
            }

            inline int64_t rSlotCount(size_t n)
            {
                for(int64_t c : kRreduceSlotCounts)
                {
                    if(int64_t(n) <= c)
                    {
                        return c;
                    }
                }
                return -1;
            }

            inline bool rLoadVec(std::string const& a, int64_t v)
            {
                if(a == "f64")
                {
                    return v == 1 || v == 2;
                }
                if(a == "f32")
                {
                    return v == 1 || v == 2 || v == 4;
                }
                return v == 1 || v == 2 || v == 4 || v == 8;
            }

            // rreduce_host.max_vec
            inline int64_t rMaxVec(RProblem const& p, bool row)
            {
                int64_t n = 0, s = 0;
                if(row)
                {
                    n = p.red[0][0];
                    s = p.red[0][1];
                }
                else
                {
                    if(p.kept.empty())
                    {
                        return 1;
                    }
                    n = p.kept[0][0];
                    s = p.kept[0][1];
                }
                if(s != 1)
                {
                    return 1;
                }
                int64_t best = 1;
                for(int64_t v : {1, 2, 4, 8})
                {
                    if(rLoadVec(p.a, v) && n % v == 0)
                    {
                        best = std::max(best, v);
                    }
                }
                return best;
            }

            inline int64_t rRowThreads(RKnobs const& k)
            {
                return k.rows != 0 ? k.rows : k.block / k.lanes;
            }

            // rreduce_host.loop_geometry
            inline void rLoop(RProblem const& p,
                              RKnobs const&   k,
                              int64_t&        nIdx,
                              int64_t&        step,
                              int64_t&        per,
                              int64_t&        nIt)
            {
                if(k.row)
                {
                    nIdx = rExtents(rChunkDims(p, k.vec));
                    step = k.lanes;
                }
                else
                {
                    nIdx = p.red.empty() ? 1 : rExtents(p.red);
                    step = rRowThreads(k);
                }
                per = rCeilDiv(nIdx, k.split);
                nIt = rCeilDiv(rCeilDiv(per, step), k.unroll);
            }

            // rreduce_host.grid_x
            inline int64_t rGridX(RProblem const& p, RKnobs const& k)
            {
                int64_t const nOut = rExtents(p.kept);
                if(k.row)
                {
                    return rCeilDiv(nOut, k.block / k.lanes);
                }
                size_t const ng = rColGroup(p);
                int64_t      m  = 1;
                for(size_t i = 0; i < ng; i++)
                {
                    m *= p.kept[i][0];
                }
                int64_t const tiles = rCeilDiv(m, k.lanes * k.vec);
                int64_t const items = k.block / (k.lanes * rRowThreads(k));
                return rCeilDiv((nOut / m) * tiles, items);
            }

            inline bool rPow2(int64_t x)
            {
                return x > 0 && (x & (x - 1)) == 0;
            }

            // rreduce_host.check_knobs: why `k` cannot run the problem ("" when it can)
            inline std::string rCheck(RProblem const& p, RKnobs const& k, bool streaming)
            {
                for(auto const& x : p.kept)
                {
                    if(x[1] < 0 || x[2] < 0)
                    {
                        return "negative strides";
                    }
                }
                for(auto const& x : p.red)
                {
                    if(x[1] < 0)
                    {
                        return "negative strides";
                    }
                }
                if(p.kept.size() > size_t(kRMaxRank) || p.red.size() > size_t(kRMaxRank))
                {
                    return "more than 8 kept or reduced modes";
                }
                if(k.row != rRow(p))
                {
                    return "mode differs from the problem's";
                }
                if(k.nt && !streaming)
                {
                    return "streaming loads need gfx942 / gfx950";
                }
                if(!rPow2(k.block) || k.block < kRWave || k.block > 1024)
                {
                    return "block";
                }
                if(!rPow2(k.lanes) || k.lanes > k.block)
                {
                    return "lanes";
                }
                if(!rLoadVec(p.a, k.vec) || (k.vec > 1 && k.vec > rMaxVec(p, k.row)))
                {
                    return "vec";
                }
                if(k.unroll != 1 && k.unroll != 2 && k.unroll != 4 && k.unroll != 8)
                {
                    return "unroll";
                }
                if(k.split < 1 || k.split > kRGridYZMax)
                {
                    return "split";
                }
                int64_t const nOut = rExtents(p.kept);
                if(!k.row)
                {
                    if(k.rows != 0 && (!rPow2(k.rows) || k.lanes * k.rows > k.block))
                    {
                        return "rows";
                    }
                    if(p.kept.empty())
                    {
                        return "col mode needs a kept mode";
                    }
                    size_t const ng = rColGroup(p);
                    int64_t      m  = 1;
                    for(size_t i = 0; i < ng; i++)
                    {
                        m *= p.kept[i][0];
                    }
                    if(k.lanes > 1 && (k.lanes / 2) * k.vec >= m)
                    {
                        return "lanes x vec exceed the run";
                    }
                    int64_t const rt    = rRowThreads(k);
                    int64_t const nRows = p.red.empty() ? 1 : rExtents(p.red);
                    if(rt > 1 && rt / 2 >= nRows)
                    {
                        return "more than half the thread rows idle";
                    }
                    if(k.split > std::max<int64_t>(1, rCeilDiv(nRows, rt)))
                    {
                        return "split exceeds the reduction rows per thread row";
                    }
                    int64_t const items = k.block / (k.lanes * rt);
                    int64_t const tiles = rCeilDiv(m, k.lanes * k.vec);
                    if((nOut / m) * tiles + items >= kI32Limit)
                    {
                        return "work items >= 2^31";
                    }
                }
                else
                {
                    if(k.rows != 0)
                    {
                        return "rows is a col knob";
                    }
                    if(p.red.empty())
                    {
                        return "row mode needs a reduced mode";
                    }
                    auto const    dims   = rChunkDims(p, k.vec);
                    int64_t const chunks = rExtents(dims);
                    if(k.split > std::max<int64_t>(1, rCeilDiv(chunks, k.lanes)))
                    {
                        return "split exceeds the chunks per lane";
                    }
                    if(k.lanes > 1 && (k.lanes / 2) >= chunks)
                    {
                        return "lanes exceed the chunks";
                    }
                    if(dims.size() > size_t(kRMaxRank))
                    {
                        return "more than 8 reduced slots";
                    }
                }
                if(nOut + k.block * k.vec >= kI32Limit)
                {
                    return "outputs >= 2^31";
                }
                int64_t nIdx = 0, step = 0, per = 0, nIt = 0;
                rLoop(p, k, nIdx, step, per, nIt);
                int64_t const last = (k.split - 1) * per + (step - 1) + (nIt * k.unroll - 1) * step;
                if(std::max(last, k.split * per) >= kI32Limit)
                {
                    return "reduction indices >= 2^31";
                }
                if(rGridX(p, k) * k.block > kMaxGridThreads)
                {
                    return "launch grid exceeds the HIP limit";
                }
                return "";
            }

            // rreduce_host.needs_idx64
            inline bool rIdx64(RProblem const& p, RKnobs const& k)
            {
                int64_t aSpan = 0, dSpan = 0;
                for(auto const& x : p.kept)
                {
                    aSpan += (x[0] - 1) * x[1];
                    dSpan += (x[0] - 1) * x[2];
                }
                for(auto const& x : p.red)
                {
                    aSpan += (x[0] - 1) * x[1];
                }
                int64_t const nOut  = rExtents(p.kept);
                int64_t const wSpan = k.split > 1 ? (nOut + k.block * k.vec) * k.split : 0;
                return std::max({aSpan, dSpan, wSpan}) >= kI32Limit;
            }

            // rreduce_host.final_lanes
            inline int64_t rFinalLanes(int64_t nOut, int64_t split)
            {
                int64_t lanes = 1;
                while(lanes * 2 <= std::min<int64_t>(kRWave, split)
                      && nOut * lanes * 2 <= kRreduceFinalTarget)
                {
                    lanes *= 2;
                }
                int64_t out = 1;
                for(int64_t l : kRreduceFinalLanes)
                {
                    if(l <= lanes)
                    {
                        out = l;
                    }
                }
                return out;
            }

            // The canonical problem of a tuned route record (rreduce_host.problem_key)
            inline std::string rKey(RProblem const& p)
            {
                std::string k, r;
                for(size_t i = 0; i < p.kept.size(); i++)
                {
                    k += (i ? "," : "") + std::to_string(p.kept[i][0]) + ":"
                         + std::to_string(p.kept[i][1]) + ":" + std::to_string(p.kept[i][2]);
                }
                for(size_t i = 0; i < p.red.size(); i++)
                {
                    r += (i ? "," : "") + std::to_string(p.red[i][0]) + ":"
                         + std::to_string(p.red[i][1]);
                }
                return "a=" + p.a + ";d=" + p.d + ";op=" + p.op + ";k=" + k + ";r=" + r;
            }

            // A tuned route record (rreduce_host.route_record): "v1;mode=...;block=...;..."
            inline bool rParseRecord(std::string const& text, RKnobs& k)
            {
                size_t                start = 0;
                bool                  head  = true;
                int                   seen  = 0;
                std::set<std::string> names;
                while(start <= text.size())
                {
                    size_t      end   = text.find(';', start);
                    std::string field = text.substr(
                        start, end == std::string::npos ? std::string::npos : end - start);
                    if(head)
                    {
                        if(field != "v1")
                        {
                            return false;
                        }
                        head = false;
                    }
                    else
                    {
                        auto const eq = field.find('=');
                        if(eq == std::string::npos)
                        {
                            return false;
                        }
                        auto const name  = field.substr(0, eq);
                        auto const value = field.substr(eq + 1);
                        int64_t    v     = 0;
                        if(!names.insert(name).second)
                        {
                            return false; // a field twice
                        }
                        if(name != "mode")
                        {
                            char* stop = nullptr;
                            v          = std::strtoll(value.c_str(), &stop, 10);
                            if(stop == value.c_str() || *stop != '\0')
                            {
                                return false;
                            }
                        }
                        if(name == "mode")
                        {
                            if(value != "row" && value != "col")
                            {
                                return false;
                            }
                            k.row = value == "row";
                        }
                        else if(name == "block")
                            k.block = v;
                        else if(name == "vec")
                            k.vec = v;
                        else if(name == "lanes")
                            k.lanes = v;
                        else if(name == "unroll")
                            k.unroll = v;
                        else if(name == "split")
                            k.split = v;
                        else if(name == "rows")
                            k.rows = v;
                        else if(name == "nt")
                            k.nt = v != 0;
                        else if(name != "nr" && name != "idx64")
                            return false;
                        seen++;
                    }
                    if(end == std::string::npos)
                    {
                        break;
                    }
                    start = end + 1;
                }
                // mode, block, vec, lanes, unroll, split, rows, nt, nr, idx64: each once
                return seen == 10;
            }

            // rreduce_host.log_q: `x` on a quarter-octave scale (integers only)
            inline int64_t rLogQ(int64_t x)
            {
                if(x <= 1)
                {
                    return 0;
                }
                int64_t const e    = 63 - __builtin_clzll(static_cast<unsigned long long>(x));
                int64_t const frac = e >= 2 ? (x >> (e - 2)) & 3 : (x << (2 - e)) & 3;
                return 4 * e + frac;
            }

            // rreduce_host.features
            inline std::array<int64_t, 5> rFeatures(RProblem const& p)
            {
                int64_t const nOut  = rExtents(p.kept);
                int64_t const nRed  = p.red.empty() ? 1 : rExtents(p.red);
                int64_t       inner = 1;
                if(rRow(p))
                {
                    inner = p.red[0][1] == 1 ? p.red[0][0] : 1;
                }
                else
                {
                    size_t const ng = rColGroup(p);
                    int64_t      m  = 1;
                    for(size_t i = 0; i < ng; i++)
                    {
                        m *= p.kept[i][0];
                    }
                    inner = p.kept[0][1] == 1 ? m : 1;
                }
                return {rLogQ(nOut),
                        rLogQ(nRed),
                        rLogQ(inner),
                        int64_t(p.red.size()),
                        int64_t(p.kept.size())};
            }

            // rreduce_host.feature_distance
            inline int64_t rDistance(std::array<int64_t, 5> const& f,
                                     std::array<int64_t, 5> const& g)
            {
                return 2 * std::llabs(f[0] - g[0]) + 2 * std::llabs(f[1] - g[1])
                       + std::llabs(f[2] - g[2]) + 2 * std::llabs(f[3] - g[3])
                       + 2 * std::llabs(f[4] - g[4]);
            }

            // rreduce_host.Prototype: a measured problem of the untuned choice (pack routes
            // reduce.proto.<i>, reduce.proto.count)
            struct RProto
            {
                bool                   row    = true;
                int64_t                eb     = 0;
                bool                   maxmin = false;
                std::array<int64_t, 5> f{};
                RKnobs                 knobs;
                int64_t                waves = 0;
                int64_t                cus   = 1;
            };

            // A prototype record (rreduce_host.proto_record): "v1;mode=row;eb=2;mm=0;f=a,b,c,d,e;
            // block=..;vec=..;lanes=..;unroll=..;rows=..;nt=..;waves=..;cus=.."
            inline bool rParseProto(std::string const& text, RProto& out)
            {
                size_t                start = 0;
                int                   seen  = 0;
                bool                  head  = true;
                std::set<std::string> names;
                while(start <= text.size())
                {
                    size_t const      end   = text.find(';', start);
                    std::string const field = text.substr(
                        start, end == std::string::npos ? std::string::npos : end - start);
                    if(head)
                    {
                        if(field != "v1")
                        {
                            return false;
                        }
                        head = false;
                    }
                    else
                    {
                        auto const eq = field.find('=');
                        if(eq == std::string::npos)
                        {
                            return false;
                        }
                        auto const name  = field.substr(0, eq);
                        auto const value = field.substr(eq + 1);
                        if(!names.insert(name).second)
                        {
                            return false; // a field twice
                        }
                        if(name == "mode")
                        {
                            if(value != "row" && value != "col")
                            {
                                return false;
                            }
                            out.row = value == "row";
                        }
                        else if(name == "f")
                        {
                            size_t s = 0;
                            for(int i = 0; i < 5; i++)
                            {
                                size_t const e = value.find(',', s);
                                if((i < 4) == (e == std::string::npos))
                                {
                                    return false;
                                }
                                std::string const part = value.substr(
                                    s, e == std::string::npos ? std::string::npos : e - s);
                                char*         stop = nullptr;
                                int64_t const v    = std::strtoll(part.c_str(), &stop, 10);
                                if(part.empty() || *stop != '\0')
                                {
                                    return false;
                                }
                                out.f[i] = v;
                                s        = e + 1;
                            }
                        }
                        else
                        {
                            char*         stop = nullptr;
                            int64_t const v    = std::strtoll(value.c_str(), &stop, 10);
                            if(value.empty() || *stop != '\0')
                            {
                                return false;
                            }
                            if(name == "eb")
                                out.eb = v;
                            else if(name == "mm")
                                out.maxmin = v != 0;
                            else if(name == "block")
                                out.knobs.block = v;
                            else if(name == "vec")
                                out.knobs.vec = v;
                            else if(name == "lanes")
                                out.knobs.lanes = v;
                            else if(name == "unroll")
                                out.knobs.unroll = v;
                            else if(name == "rows")
                                out.knobs.rows = v;
                            else if(name == "nt")
                                out.knobs.nt = v != 0;
                            else if(name == "waves")
                                out.waves = v;
                            else if(name == "cus")
                                out.cus = v;
                            else
                                return false;
                        }
                        seen++;
                    }
                    if(end == std::string::npos)
                    {
                        break;
                    }
                    start = end + 1;
                }
                out.knobs.row = out.row;
                // mode, eb, mm, f, block, vec, lanes, unroll, rows, nt, waves, cus: each once
                return seen == 12 && out.cus > 0;
            }

            inline int64_t rWaves(RProblem const& p, RKnobs const& k)
            {
                return rGridX(p, k) * k.split * k.block / kRWave;
            }

            // rreduce_host.split_workspace: the workspace bytes of the partial results of a split
            // launch (none without a split), rounded up to the workspace alignment
            inline uint64_t rWorkspace(RProblem const& p, int64_t split)
            {
                if(split <= 1)
                {
                    return 0;
                }
                uint64_t const cb = p.a == "f64" || p.d == "f64" ? 8 : 4;
                uint64_t const n  = uint64_t(rExtents(p.kept)) * uint64_t(split) * cb;
                return (n + kRWorkspaceAlign - 1) / kRWorkspaceAlign * kRWorkspaceAlign;
            }

            // rreduce_host.cap_split: `split`, or the largest power of two below it whose partial
            // results fit `limit` bytes (1, no partial results, when none does)
            inline int64_t rCapSplit(RProblem const& p, int64_t split, uint64_t limit)
            {
                while(split > 1 && rWorkspace(p, split) > limit)
                {
                    int64_t s = 1;
                    while(s * 2 < split)
                    {
                        s *= 2;
                    }
                    split = s;
                }
                return split;
            }

            // A tuned reduction class of a pack (the attributes of its kernel)
            struct RClass
            {
                std::string name, mode, a, d, op;
                int64_t     nr = 0, block = 0, vec = 0, lanes = 0, unroll = 0, rows = 0;
                bool        nt = false, una = false, idx64 = false;
            };

            // One launch of the untuned choice: the knobs (with the split) and the class (an index
            // into the classes the choice was given) that runs them
            struct RChoice
            {
                RKnobs knobs;
                size_t cls = 0;
            };

            // rreduce_host.untuned_choices: the prototypes of the problem's mode and element size
            // nearest to it (same MAX / MIN class first, then by feature distance, then table
            // order), each with its knobs, the class that runs them (the fewest reduced slots the
            // problem needs, 32-bit offsets when they fit) and the smallest power-of-two split
            // that reaches the prototype's wave count (scaled to `cus` compute units) whose
            // partial results fit `workspaceLimit` bytes. Up to kRreduceCandidates distinct
            // (class, split) launches, the first the rule's choice; when no prototype's knobs run
            // the problem, the universal knobs of the mode.
            inline std::vector<RChoice> untunedChoices(RProblem const&            p,
                                                       std::vector<RClass> const& classes,
                                                       std::vector<RProto> const& protos,
                                                       bool                       una,
                                                       int64_t                    cus,
                                                       bool                       streaming,
                                                       uint64_t                   workspaceLimit)
            {
                std::vector<RChoice> out;
                bool const           row = rRow(p);
                bool const           mm  = p.op == "max" || p.op == "min";
                if((row && p.red.empty()) || (!row && p.kept.empty()))
                {
                    // one element in all (every mode of extent 1): no class has work for it
                    return out;
                }
                std::vector<size_t> pool;
                for(size_t i = 0; i < classes.size(); i++)
                {
                    auto const& c = classes[i];
                    if(c.a == p.a && c.d == p.d && c.op == p.op && (c.mode == "row") == row
                       && c.una == una)
                    {
                        pool.push_back(i);
                    }
                }
                // the smallest power-of-two split reaching `target` waves (that the knobs admit
                // and whose partial results fit the workspace limit)
                auto grow = [&](RKnobs& k, int64_t target) {
                    while(rWaves(p, k) < target)
                    {
                        RKnobs next = k;
                        next.split  = k.split * 2;
                        if(!rCheck(p, next, streaming).empty()
                           || rWorkspace(p, next.split) > workspaceLimit)
                        {
                            break;
                        }
                        k = next;
                    }
                };
                // the class of the pool that runs `k`: the fewest slots, then 32-bit offsets
                auto classOf = [&](RKnobs const& k, int64_t need) -> int64_t {
                    bool const need64 = rIdx64(p, k);
                    int64_t    best   = -1;
                    for(size_t i : pool)
                    {
                        auto const& c = classes[i];
                        if(c.block != k.block || c.vec != k.vec || c.lanes != k.lanes
                           || c.unroll != k.unroll || c.rows != k.rows || c.nt != k.nt || need <= 0
                           || c.nr < need || (need64 && !c.idx64))
                        {
                            continue;
                        }
                        if(best < 0
                           || std::make_pair(c.nr, c.idx64) < std::make_pair(
                                  classes[size_t(best)].nr, classes[size_t(best)].idx64))
                        {
                            best = int64_t(i);
                        }
                    }
                    return best;
                };
                auto const                                     f = rFeatures(p);
                std::vector<std::tuple<bool, int64_t, size_t>> order;
                for(size_t i = 0; i < protos.size(); i++)
                {
                    if(protos[i].row == row && protos[i].eb == p.eb)
                    {
                        order.emplace_back(protos[i].maxmin != mm, rDistance(f, protos[i].f), i);
                    }
                }
                std::sort(order.begin(), order.end());
                std::set<std::pair<std::string, int64_t>> seen;
                for(auto const& o : order)
                {
                    if(out.size() == kRreduceCandidates)
                    {
                        break;
                    }
                    auto const& proto = protos[std::get<2>(o)];
                    RKnobs      k     = proto.knobs;
                    k.row             = row;
                    k.split           = 1;
                    k.nt              = k.nt && streaming;
                    if(!rCheck(p, k, streaming).empty())
                    {
                        continue;
                    }
                    int64_t const need = rSlotCount(k.row ? rChunkDims(p, k.vec).size()
                                                          : std::max<size_t>(1, p.red.size()));
                    grow(k, proto.waves * cus / proto.cus);
                    int64_t const c = classOf(k, need);
                    if(c < 0 || !seen.emplace(classes[size_t(c)].name, k.split).second)
                    {
                        continue;
                    }
                    out.push_back({k, size_t(c)});
                }
                if(out.empty())
                {
                    // no prototype's knobs run the problem: the universal knobs of the mode (one
                    // thread per output or per kept element, scalar loads), which run every
                    // problem (rreduce_host.UNIVERSAL_KNOBS / UNIVERSAL_WAVES)
                    int64_t const* u = row ? kRreduceUniversalRow : kRreduceUniversalCol;
                    RKnobs         k;
                    k.row              = row;
                    k.block            = u[0];
                    k.vec              = u[1];
                    k.lanes            = u[2];
                    k.unroll           = u[3];
                    k.rows             = row ? 0 : u[4];
                    k.split            = 1;
                    k.nt               = streaming;
                    int64_t const need = rSlotCount(k.row ? rChunkDims(p, k.vec).size()
                                                          : std::max<size_t>(1, p.red.size()));
                    if(need > 0 && rCheck(p, k, streaming).empty())
                    {
                        grow(k, kRreduceUniversalWaves * cus / 256);
                        int64_t const c = classOf(k, need);
                        if(c >= 0)
                        {
                            out.push_back({k, size_t(c)});
                        }
                    }
                }
                return out;
            }

            // The reduction routes of a pack's route table are well formed: reduce.proto.count
            // records reduce.proto.0 .. count - 1, each a prototype record, and every
            // reduce.tuned.<hash> is a route record with its .key, whose FNV-1a 64 is <hash>.
            // Returns what is wrong ("" when nothing is).
            template <typename Map>
            std::string rCheckRoutes(Map const& routes)
            {
                std::string const proto = "reduce.proto.", tuned = "reduce.tuned.";
                int64_t           count = 0;
                auto const        it    = routes.find("reduce.proto.count");
                if(it != routes.end())
                {
                    char* stop = nullptr;
                    count      = std::strtoll(it->second.c_str(), &stop, 10);
                    if(it->second.empty() || *stop != '\0' || count < 0)
                    {
                        return "reduce.proto.count is not a count";
                    }
                }
                int64_t records = 0;
                for(auto const& kv : routes)
                {
                    std::string const& key = kv.first;
                    if(key.rfind(proto, 0) == 0 && key != "reduce.proto.count")
                    {
                        std::string const idx  = key.substr(proto.size());
                        char*             stop = nullptr;
                        int64_t const     i    = std::strtoll(idx.c_str(), &stop, 10);
                        if(idx.empty() || *stop != '\0' || i < 0 || i >= count
                           || std::to_string(i) != idx)
                        {
                            return key + " is not one of the reduce.proto.count records";
                        }
                        RProto r;
                        if(!rParseProto(kv.second, r))
                        {
                            return key + " is not a prototype record: " + kv.second;
                        }
                        records++;
                    }
                    else if(key.rfind(tuned, 0) == 0)
                    {
                        bool const isKey = key.size() > tuned.size() + 4
                                           && key.compare(key.size() - 4, 4, ".key") == 0;
                        std::string const base = isKey ? key.substr(0, key.size() - 4) : key;
                        auto const        rec  = routes.find(base);
                        auto const        kk   = routes.find(base + ".key");
                        if(rec == routes.end() || kk == routes.end())
                        {
                            return base + " lacks its record or its .key";
                        }
                        char hex[17];
                        snprintf(hex,
                                 sizeof(hex),
                                 "%016llx",
                                 static_cast<unsigned long long>(rFnv1a64(kk->second)));
                        if(base.substr(tuned.size()) != hex)
                        {
                            return base + ".key does not hash to its route";
                        }
                        RKnobs k;
                        if(!rParseRecord(rec->second, k))
                        {
                            return base + " is not a route record: " + rec->second;
                        }
                    }
                }
                if(records != count)
                {
                    return "reduce.proto.count is " + std::to_string(count) + ", the routes have "
                           + std::to_string(records) + " prototype records";
                }
                return "";
            }
        } // namespace rrule
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_ROCKE_RREDUCE_RULE_HPP
