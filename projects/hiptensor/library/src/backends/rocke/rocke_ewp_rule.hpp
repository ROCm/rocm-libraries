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

// The rules of rocKE's tuned tiled elementwise classes (ewp): the inputs' kinds, the two runs of
// a layout, the launch arguments of a class, the canonical problem key and the class choice.
// Header-only so that hipTensor's planner (rocke_plan.cpp) and the C++ pack builder
// (pack/rocke_pack_builder.cpp: --ewp-picks) run the same code; rocKE's statement of the rules is
// library/kernels/common/hiptensor_aot/ewp.py (ewp_kinds, ewp_runs, ewp_plan) and ewp_host.py
// (problem_key, choose), and rocKE's pack cross-check test compares the two choices.

#ifndef HIPTENSOR_ROCKE_EWP_RULE_HPP
#define HIPTENSOR_ROCKE_EWP_RULE_HPP

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace hiptensor
{
    namespace rockebackend
    {
        namespace ewprule
        {
            constexpr int64_t kEI32Max    = (int64_t(1) << 31) - 1;
            constexpr int64_t kEGridYZMax = 65535;
            // ewp_host.MIN_CTAS: CTAs the untuned choice wants a launch to have
            constexpr int64_t kEMinCtas = 1024;

            // A canonical mode: extent and the stride of each tensor (A [B] [C] D order).
            struct EMode
            {
                int64_t                extent = 1;
                std::array<int64_t, 4> stride{};
            };

            // A class of the pack (the kernel's attributes).
            struct EClass
            {
                std::string name;
                int64_t     rank  = 0;
                int64_t     arity = 1;
                std::string a, b, c, d; // "" for an absent B / C
                std::string kinds, mode, walk;
                int64_t     to = 0, tt = 0, tw = 1, block = 256, tpc = 1, vec = 0, slots = 8;
            };

            // A launch: argument values by name (i32 bit patterns) and the grid.
            struct ELaunch
            {
                std::map<std::string, int64_t> args;
                std::array<int64_t, 3>         grid{1, 1, 1};
            };

            inline int64_t eCeilDiv(int64_t a, int64_t b)
            {
                return (a + b - 1) / b;
            }

            // FNV-1a 64 of a route key (ewp_host.fnv1a64)
            inline uint64_t eFnv1a64(std::string const& text)
            {
                uint64_t hash = 0xCBF29CE484222325ull;
                for(unsigned char c : text)
                {
                    hash = (hash ^ c) * 0x100000001B3ull;
                }
                return hash;
            }

            inline std::string eHex16(uint64_t v)
            {
                char buf[24];
                std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
                return buf;
            }

            // common.magic_u31 as i32 bit patterns: (m, s); (0, 0) for d == 1
            inline std::pair<int64_t, int64_t> eMagic(int64_t d)
            {
                if(d <= 1)
                {
                    return {0, 0};
                }
                constexpr uint64_t limit = uint64_t(1) << 31;
                for(uint32_t s = 32; s < 64; s++)
                {
                    unsigned __int128 p = static_cast<unsigned __int128>(1) << s;
                    unsigned __int128 m = (p + uint64_t(d) - 1) / uint64_t(d);
                    if(m >= (static_cast<unsigned __int128>(1) << 32))
                    {
                        break;
                    }
                    unsigned __int128 e = m * uint64_t(d) - p;
                    if(e * (limit - 1) < p)
                    {
                        return {
                            static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(m))),
                            int64_t(s - 32)};
                    }
                }
                return {-1, -1};
            }

            inline std::vector<std::string> eNames(int64_t arity)
            {
                if(arity == 1)
                {
                    return {"A", "D"};
                }
                if(arity == 2)
                {
                    return {"A", "C", "D"};
                }
                return {"A", "B", "C", "D"};
            }

            // ewp.ewp_kinds: the kinds of the inputs of a layout ("" when an input is neither)
            inline std::string
                eKinds(std::vector<EMode> const& dims, int tensors, std::string const& mode)
            {
                int const nd = tensors - 1;
                int       o  = -1;
                for(int i = 0; i < int(dims.size()); i++)
                {
                    if(dims[i].stride[nd] == 1)
                    {
                        o = i;
                        break;
                    }
                }
                if(o < 0)
                {
                    return "";
                }
                std::string out;
                if(mode == "a")
                {
                    int64_t const ew = dims[o].extent;
                    int           od = -1;
                    for(int i = 0; i < int(dims.size()); i++)
                    {
                        if(dims[i].stride[nd] == ew && i != o)
                        {
                            od = i;
                            break;
                        }
                    }
                    for(int k = 0; k < nd; k++)
                    {
                        if(dims[o].stride[k] != 1)
                        {
                            return "";
                        }
                        out += od >= 0 && dims[od].stride[k] == ew ? 'V' : 'T';
                    }
                    return out;
                }
                for(int k = 0; k < nd; k++)
                {
                    if(dims[o].stride[k] == 1)
                    {
                        out += 'V';
                        continue;
                    }
                    bool unit = false;
                    for(auto const& m : dims)
                    {
                        unit = unit || m.stride[k] == 1;
                    }
                    if(!unit)
                    {
                        return "";
                    }
                    out += 'T';
                }
                return out;
            }

            inline bool eIn(std::vector<int> const& v, int x)
            {
                for(int y : v)
                {
                    if(y == x)
                    {
                        return true;
                    }
                }
                return false;
            }

            // ewp.ewp_runs: the dims of the two runs (inner first) and the shared stride-1 dim
            // (mode a; -1 otherwise); false when the layout does not fit the class
            inline bool eRuns(EClass const&             c,
                              std::vector<EMode> const& dims,
                              std::vector<int>&         oRun,
                              std::vector<int>&         tRun,
                              int&                      w)
            {
                int const tensors = int(c.arity) + 1;
                int const nd      = tensors - 1;
                int       ki      = -1;
                for(int k = 0; k < nd; k++)
                {
                    if(c.kinds[size_t(k)] == 'T')
                    {
                        ki = k;
                        break;
                    }
                }
                if(ki < 0)
                {
                    return false;
                }
                auto unitDim = [&dims](int k, std::vector<int> const& excl) {
                    for(int i = 0; i < int(dims.size()); i++)
                    {
                        if(dims[i].stride[k] == 1 && !eIn(excl, i))
                        {
                            return i;
                        }
                    }
                    return -1;
                };
                w = -1;
                std::vector<int> excl;
                int              oFirst = -1, tFirst = -1;
                if(c.mode == "a")
                {
                    w = unitDim(nd, {});
                    if(w < 0)
                    {
                        return false;
                    }
                    for(int k = 0; k < tensors; k++)
                    {
                        if(dims[w].stride[k] != 1)
                        {
                            return false;
                        }
                    }
                    excl                = {w};
                    int64_t const ew    = dims[w].extent;
                    auto          first = [&dims, ew](int k, std::vector<int> const& ex) {
                        for(int i = 0; i < int(dims.size()); i++)
                        {
                            if(dims[i].stride[k] == ew && !eIn(ex, i))
                            {
                                return i;
                            }
                        }
                        return -1;
                    };
                    oFirst               = first(nd, excl);
                    std::vector<int> ex2 = excl;
                    if(oFirst >= 0)
                    {
                        ex2.push_back(oFirst);
                    }
                    tFirst = first(ki, ex2);
                }
                else
                {
                    oFirst = unitDim(nd, {});
                    tFirst
                        = unitDim(ki, oFirst >= 0 ? std::vector<int>{oFirst} : std::vector<int>{});
                }
                if(oFirst < 0 || tFirst < 0 || oFirst == tFirst)
                {
                    return false;
                }
                // the run from `first`: alone when its extent is a multiple of the tile, else with
                // the mode contiguous after it in tensor k (not excluded) when the two hold a whole
                // number of tiles
                auto grow = [&dims](int                     first,
                                    int                     k,
                                    int64_t                 tile,
                                    std::vector<int> const& ex,
                                    std::vector<int>&       run) {
                    run.clear();
                    if(dims[first].extent % tile == 0)
                    {
                        run.push_back(first);
                        return true;
                    }
                    int64_t const want = dims[first].extent * dims[first].stride[k];
                    int           nxt  = -1;
                    for(int i = 0; i < int(dims.size()); i++)
                    {
                        if(dims[i].stride[k] == want && i != first && !eIn(ex, i))
                        {
                            nxt = i;
                            break;
                        }
                    }
                    if(nxt < 0 || (dims[first].extent * dims[nxt].extent) % tile != 0)
                    {
                        return false;
                    }
                    run = {first, nxt};
                    return true;
                };
                // the t run first, then the o run; when that fails, the other way
                std::vector<int> ex = excl;
                ex.push_back(oFirst);
                bool ok = grow(tFirst, ki, c.tt, ex, tRun);
                if(ok)
                {
                    std::vector<int> ex3 = excl;
                    ex3.insert(ex3.end(), tRun.begin(), tRun.end());
                    ok = grow(oFirst, nd, c.to, ex3, oRun);
                }
                if(!ok)
                {
                    std::vector<int> ex4 = excl;
                    ex4.push_back(tFirst);
                    if(!grow(oFirst, nd, c.to, ex4, oRun))
                    {
                        return false;
                    }
                    std::vector<int> ex5 = excl;
                    ex5.insert(ex5.end(), oRun.begin(), oRun.end());
                    if(!grow(tFirst, ki, c.tt, ex5, tRun))
                    {
                        return false;
                    }
                }
                return true;
            }

            // ewp.ewp_plan: the launch of class `c` on `dims`; false when the layout does not fit
            inline bool eplan(EClass const& c, std::vector<EMode> const& dims, ELaunch& out)
            {
                int const                tensors = int(c.arity) + 1;
                std::vector<std::string> names   = eNames(c.arity);
                int64_t const            V       = c.vec;
                std::vector<int>         oRun, tRun;
                int                      w = -1;
                if(V <= 0 || !eRuns(c, dims, oRun, tRun, w))
                {
                    return false;
                }
                std::vector<int> used = oRun;
                used.insert(used.end(), tRun.begin(), tRun.end());
                if(w >= 0)
                {
                    used.push_back(w);
                }
                std::vector<int> others;
                for(int i = 0; i < int(dims.size()); i++)
                {
                    if(!eIn(used, i))
                    {
                        others.push_back(i);
                    }
                }
                if(int64_t(others.size()) > c.slots)
                {
                    return false;
                }
                for(int k = 0; k < tensors; k++)
                {
                    int64_t span = 0;
                    for(auto const& m : dims)
                    {
                        if(m.stride[k] < 0)
                        {
                            return false;
                        }
                        span += (m.extent - 1) * m.stride[k];
                    }
                    if(span > kEI32Max)
                    {
                        return false;
                    }
                }
                std::vector<int> all = used;
                all.insert(all.end(), others.begin(), others.end());
                int64_t const eo0 = dims[oRun[0]].extent;
                int64_t const et0 = dims[tRun[0]].extent;
                if(c.mode == "a")
                {
                    if(dims[w].extent % (c.tw * V) != 0)
                    {
                        return false;
                    }
                    // a tile row covers the whole shared mode or at least 128 bytes of it
                    if(c.tw * V != dims[w].extent && c.tw * 16 < 128)
                    {
                        return false;
                    }
                    for(int k = 0; k < tensors; k++)
                    {
                        for(int i : all)
                        {
                            if(i != w && dims[i].stride[k] % V != 0)
                            {
                                return false;
                            }
                        }
                    }
                }
                else
                {
                    if(eo0 % V != 0 || et0 % V != 0)
                    {
                        return false;
                    }
                    for(int k = 0; k < tensors; k++)
                    {
                        bool const direct = k == tensors - 1 || c.kinds[size_t(k)] == 'V';
                        int const  unit   = direct ? oRun[0] : tRun[0];
                        if(dims[unit].stride[k] != 1)
                        {
                            return false;
                        }
                        for(int i : all)
                        {
                            if(i != unit && dims[i].stride[k] % V != 0)
                            {
                                return false;
                            }
                        }
                    }
                }
                auto& v     = out.args;
                auto  magic = [&v](char const* m, char const* s, int64_t d) {
                    auto ms = eMagic(d);
                    v[m]    = ms.first;
                    v[s]    = ms.second;
                };
                int64_t no = 1, nt = 1;
                for(int i : oRun)
                {
                    no *= dims[i].extent;
                }
                for(int i : tRun)
                {
                    nt *= dims[i].extent;
                }
                int64_t const tilesO = no / c.to, tilesT = nt / c.tt;
                int64_t const tilesW = w >= 0 ? dims[w].extent / (c.tw * V) : 1;
                v["eo"]              = eo0;
                magic("om", "os", eo0);
                v["et"] = et0;
                magic("tm", "ts", et0);
                v["nw"] = tilesW;
                magic("wm", "ws", tilesW);
                int64_t const nf = c.walk == "o" ? tilesO : tilesT;
                v["nf"]          = nf;
                magic("fm", "fs", nf);
                int64_t nbatch = 1;
                for(int i : others)
                {
                    nbatch *= dims[i].extent;
                }
                v["nb"]    = nbatch;
                v["ystep"] = std::min<int64_t>(nbatch, kEGridYZMax);
                for(int k = 0; k < tensors; k++)
                {
                    auto const& n     = names[size_t(k)];
                    v["s" + n + "o0"] = dims[oRun[0]].stride[k];
                    v["s" + n + "o1"] = oRun.size() > 1 ? dims[oRun[1]].stride[k] : 0;
                    v["s" + n + "t0"] = dims[tRun[0]].stride[k];
                    v["s" + n + "t1"] = tRun.size() > 1 ? dims[tRun[1]].stride[k] : 0;
                }
                for(int i = 0; i < int(c.slots); i++)
                {
                    bool const    live = i < int(others.size());
                    int64_t const e    = live ? dims[others[size_t(i)]].extent : 1;
                    auto const    si   = std::to_string(i);
                    v["be" + si]       = e;
                    if(i < int(c.slots) - 1)
                    {
                        auto ms      = eMagic(e);
                        v["bm" + si] = ms.first;
                        v["bs" + si] = ms.second;
                    }
                    for(int k = 0; k < tensors; k++)
                    {
                        v["s" + names[size_t(k)] + si]
                            = live && e != 1 ? dims[others[size_t(i)]].stride[k] : 0;
                    }
                }
                int64_t const ntl = tilesW * tilesO * tilesT;
                if(ntl > kEI32Max)
                {
                    return false;
                }
                v["ntl"] = ntl;
                out.grid = {eCeilDiv(ntl, c.tpc), std::min<int64_t>(nbatch, kEGridYZMax), 1};
                return true;
            }

            // ewp_host.problem_key: arity, the data types of A, B, C, D and the canonical modes
            inline std::string eKey(int64_t                           arity,
                                    std::array<std::string, 4> const& dts,
                                    std::vector<EMode> const&         dims)
            {
                std::string k = std::to_string(arity) + "|" + dts[0] + "," + dts[1] + "," + dts[2]
                                + "," + dts[3] + "|";
                int const tensors = int(arity) + 1;
                for(size_t i = 0; i < dims.size(); i++)
                {
                    k += (i ? ";" : "") + std::to_string(dims[i].extent) + ":";
                    for(int t = 0; t < tensors; t++)
                    {
                        k += (t ? "," : "") + std::to_string(dims[i].stride[t]);
                    }
                }
                return k;
            }

            // ewp_host.choose: the tuned class of a tuned problem (`tuned`: the kernel name its
            // route names, "" for none) when it serves the layout; else the first class (in rank
            // order) of the problem's data types whose kinds match the layout, which serves it and
            // launches at least kEMinCtas CTAs; when none does, the serving class with the most
            // CTAs (the first of equals). -1 when no class serves the problem.
            inline int eChoose(std::vector<EClass> const&        classes,
                               std::string const&                tuned,
                               int64_t                           arity,
                               std::array<std::string, 4> const& dts,
                               std::vector<EMode> const&         dims,
                               ELaunch&                          out,
                               bool&                             isTuned)
            {
                isTuned = false;
                if(!tuned.empty())
                {
                    for(int i = 0; i < int(classes.size()); i++)
                    {
                        if(classes[size_t(i)].name == tuned)
                        {
                            ELaunch l;
                            if(eplan(classes[size_t(i)], dims, l))
                            {
                                out     = l;
                                isTuned = true;
                                return i;
                            }
                            break;
                        }
                    }
                }
                int const         tensors = int(arity) + 1;
                std::string const kt      = eKinds(dims, tensors, "t");
                std::string const ka      = eKinds(dims, tensors, "a");
                int               best    = -1;
                int64_t           bestN   = -1;
                ELaunch           bestL;
                for(int i = 0; i < int(classes.size()); i++)
                {
                    auto const& c = classes[size_t(i)];
                    if(c.arity != arity || c.a != dts[0] || c.b != dts[1] || c.c != dts[2]
                       || c.d != dts[3] || c.kinds != (c.mode == "a" ? ka : kt))
                    {
                        continue;
                    }
                    ELaunch l;
                    if(!eplan(c, dims, l))
                    {
                        continue;
                    }
                    int64_t const n = l.grid[0] * l.grid[1] * l.grid[2];
                    if(n >= kEMinCtas)
                    {
                        out = l;
                        return i;
                    }
                    if(best < 0 || n > bestN)
                    {
                        best  = i;
                        bestN = n;
                        bestL = l;
                    }
                }
                if(best >= 0)
                {
                    out = bestL;
                }
                return best;
            }
        } // namespace ewprule
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_ROCKE_EWP_RULE_HPP
