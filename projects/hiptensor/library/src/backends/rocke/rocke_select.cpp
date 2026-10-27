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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <type_traits>

#include "rocke_select.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            using backend::OpKind;
            using backend::Problem;

            // The feature schema (version kSelectSchema). Append only; a change of meaning is a
            // new schema version.
            std::vector<char const*> const kFeatureNames = {
                "op",           "dtype",         "out_dtype", "compute16",  "bilinear", "opnd_op",
                "lg_m",         "lg_n",          "lg_k",      "lg_l",       "lg_mnk",   "modes_m",
                "modes_n",      "modes_k",       "modes_l",   "unit_a",     "unit_b",   "unit_d",
                "lg_inner",     "lg_inner_kept", "route",     "f64route",   "body",     "tile_m",
                "tile_n",       "tile_k",        "block",     "persistent", "streamk",  "kblock",
                "direct",       "steps",         "memsets",   "lg_ws",      "lg_bytes", "lg_flops",
                "intensity_q8", "ctas",          "waves_q8",  "tiles_q8",   "tail_q8",  "pad_q8",
                "split",        "lg_chunk",      "lds",       "scratch",    "cus"};

            int dtypeCode(hiptensorDataType_t type)
            {
                switch(type)
                {
                case HIPTENSOR_R_16F:
                    return 0;
                case HIPTENSOR_R_16BF:
                    return 1;
                case HIPTENSOR_R_32F:
                    return 2;
                case HIPTENSOR_R_64F:
                    return 3;
                case HIPTENSOR_C_32F:
                    return 4;
                case HIPTENSOR_C_64F:
                    return 5;
                default:
                    return 9;
                }
            }

            int64_t ceilDiv(int64_t a, int64_t b)
            {
                return b > 0 ? (a + b - 1) / b : 0;
            }

            // A gradient-boosted tree ensemble over named features. Node n of a tree goes left
            // when x[feature[n]] <= threshold[n]; a child c < 0 is leaf ~c.
            struct Tree
            {
                std::vector<int32_t> feature;
                std::vector<double>  threshold;
                std::vector<int32_t> left, right;
                std::vector<double>  leaf;
            };

            struct Model
            {
                std::string          id;
                std::vector<int32_t> featureIndex; // model feature -> schema index
                std::vector<Tree>    trees;

                double predict(std::vector<int64_t> const& x) const
                {
                    double sum = 0.0;
                    for(auto const& t : trees)
                    {
                        if(t.feature.empty())
                        {
                            sum += t.leaf[0];
                            continue;
                        }
                        int32_t node = 0;
                        while(node >= 0)
                        {
                            auto const value
                                = static_cast<double>(x[featureIndex[t.feature[node]]]);
                            node = value <= t.threshold[node] ? t.left[node] : t.right[node];
                        }
                        sum += t.leaf[~node];
                    }
                    return sum;
                }
            };

            std::vector<std::string> split(std::string const& text, char sep)
            {
                std::vector<std::string> out;
                size_t                   start = 0;
                while(true)
                {
                    auto end = text.find(sep, start);
                    out.push_back(text.substr(start, end - start));
                    if(end == std::string::npos)
                    {
                        return out;
                    }
                    start = end + 1;
                }
            }

            template <typename T>
            bool parseList(std::string const& text, std::vector<T>& out)
            {
                out.clear();
                char const* p   = text.c_str();
                char const* end = p + text.size();
                while(p < end)
                {
                    while(p < end && *p == ' ')
                    {
                        p++;
                    }
                    if(p == end)
                    {
                        break;
                    }
                    char* next = nullptr;
                    if constexpr(std::is_floating_point_v<T>)
                    {
                        out.push_back(std::strtod(p, &next));
                    }
                    else
                    {
                        out.push_back(static_cast<T>(std::strtol(p, &next, 10)));
                    }
                    if(next == p)
                    {
                        return false;
                    }
                    p = next;
                }
                return true;
            }

            uint64_t fnv1a64(std::string const& text)
            {
                uint64_t hash = 0xCBF29CE484222325ull;
                for(unsigned char c : text)
                {
                    hash = (hash ^ c) * 0x100000001B3ull;
                }
                return hash;
            }

            // Parses a model of the pack's route table:
            //   v1;schema=<n>;features=<name>,...;trees=<n>|<tree>|<tree>...
            //   tree: <leaves>;<split features>;<thresholds>;<left>;<right>;<leaf values>
            // (lists space separated; a one-leaf tree has empty node lists).
            bool parseModel(std::string const& cell,
                            std::string const& text,
                            Model&             model,
                            std::string&       why)
            {
                auto parts = split(text, '|');
                auto head  = split(parts[0], ';');
                if(head.size() != 4 || head[0] != "v1")
                {
                    why = "unknown model format";
                    return false;
                }
                if(head[1] != "schema=" + std::to_string(kSelectSchema))
                {
                    why = "model of another feature schema (" + head[1] + ")";
                    return false;
                }
                if(head[2].rfind("features=", 0) != 0 || head[3].rfind("trees=", 0) != 0)
                {
                    why = "malformed model header";
                    return false;
                }
                auto const& names = kFeatureNames;
                for(auto const& name : split(head[2].substr(9), ','))
                {
                    auto it = std::find_if(
                        names.begin(), names.end(), [&name](char const* n) { return name == n; });
                    if(it == names.end())
                    {
                        why = "the model reads feature " + name + ", which this library lacks";
                        return false;
                    }
                    model.featureIndex.push_back(static_cast<int32_t>(it - names.begin()));
                }
                auto const trees = std::strtoul(head[3].c_str() + 6, nullptr, 10);
                if(trees + 1 != parts.size())
                {
                    why = "tree count mismatch";
                    return false;
                }
                auto const nf = static_cast<int32_t>(model.featureIndex.size());
                for(size_t i = 1; i < parts.size(); i++)
                {
                    auto f = split(parts[i], ';');
                    Tree t;
                    if(f.size() != 6 || !parseList(f[1], t.feature) || !parseList(f[2], t.threshold)
                       || !parseList(f[3], t.left) || !parseList(f[4], t.right)
                       || !parseList(f[5], t.leaf))
                    {
                        why = "malformed tree " + std::to_string(i - 1);
                        return false;
                    }
                    auto const leaves = std::strtol(f[0].c_str(), nullptr, 10);
                    auto const nodes  = static_cast<int32_t>(t.feature.size());
                    bool ok = leaves >= 1 && t.leaf.size() == size_t(leaves) && nodes == leaves - 1
                              && t.threshold.size() == size_t(nodes)
                              && t.left.size() == size_t(nodes) && t.right.size() == size_t(nodes);
                    for(int32_t n = 0; ok && n < nodes; n++)
                    {
                        ok = t.feature[n] >= 0 && t.feature[n] < nf;
                        for(auto child : {t.left[n], t.right[n]})
                        {
                            // children come after their parent, so every walk ends
                            ok = ok
                                 && ((child >= 0 && child > n && child < nodes)
                                     || (child < 0 && ~child < leaves));
                        }
                    }
                    if(!ok)
                    {
                        why = "inconsistent tree " + std::to_string(i - 1);
                        return false;
                    }
                    model.trees.push_back(std::move(t));
                }
                char hex[17];
                snprintf(
                    hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(fnv1a64(text)));
                model.id = cell + "@" + hex;
                return true;
            }

            // The model of a cell in a pack, parsed once per process. nullptr with `why` when
            // the pack has none or it does not load.
            std::shared_ptr<Model const>
                modelFor(Pack const& pack, std::string const& cell, std::string& why)
            {
                static std::mutex mutex;
                static std::map<std::pair<Pack const*, std::string>,
                                std::pair<std::shared_ptr<Model const>, std::string>>
                                 cache;
                std::scoped_lock lock(mutex);
                auto             key = std::make_pair(&pack, cell);
                auto             it  = cache.find(key);
                if(it == cache.end())
                {
                    std::shared_ptr<Model const> model;
                    std::string                  reason;
                    auto                         text = pack.route("select.model." + cell);
                    if(text.empty())
                    {
                        reason = "no model for " + cell;
                    }
                    else
                    {
                        auto m = std::make_shared<Model>();
                        if(parseModel(cell, text, *m, reason))
                        {
                            model = std::move(m);
                        }
                        else
                        {
                            reason = "model " + cell + " not used: " + reason;
                        }
                    }
                    it = cache.emplace(key, std::make_pair(model, reason)).first;
                }
                why = it->second.second;
                return it->second.first;
            }
        } // namespace

        int64_t log2Q8(uint64_t x)
        {
            if(x <= 1)
            {
                return 0;
            }
            int p = 63;
            while(((x >> p) & 1u) == 0)
            {
                p--;
            }
            // mantissa in [1, 2) as Q30, squared eight times for eight fraction bits
            uint64_t m = p >= 30 ? (x >> (p - 30)) : (x << (30 - p));
            int64_t  r = p;
            for(int i = 0; i < 8; i++)
            {
                m = (m * m) >> 30;
                r <<= 1;
                if(m >= (uint64_t(2) << 30))
                {
                    m >>= 1;
                    r |= 1;
                }
            }
            return r;
        }

        std::vector<char const*> const& featureNames()
        {
            return kFeatureNames;
        }

        void features(Problem const& problem, Candidate const& candidate, std::vector<int64_t>& out)
        {
            auto const&   f   = candidate.facts;
            auto const&   p   = candidate.program;
            int64_t const cus = std::max(1, problem.device.cuCount);
            out.assign(kFeatureNames.size(), 0);
            size_t i   = 0;
            auto   put = [&out, &i](int64_t v) { out[i++] = v; };
            put(problem.kind == OpKind::Contraction ? 0
                : problem.kind == OpKind::Reduction ? 1
                                                    : 2);
            put(dtypeCode(problem.t[0].type));
            put(dtypeCode(problem.output().type));
            put(problem.compute == HIPTENSOR_COMPUTE_DESC_16F
                        || problem.compute == HIPTENSOR_COMPUTE_DESC_16BF
                    ? 1
                    : 0);
            put(f.bilinear ? 1 : 0);
            put(f.operandOp ? 1 : 0);
            put(log2Q8(uint64_t(f.M)));
            put(log2Q8(uint64_t(f.N)));
            put(log2Q8(uint64_t(f.K)));
            put(log2Q8(uint64_t(f.L)));
            put(log2Q8(uint64_t(f.M) * uint64_t(f.N) * uint64_t(f.K) * uint64_t(f.L)));
            put(f.modesM);
            put(f.modesN);
            put(f.modesK);
            put(f.modesL);
            put(f.unitA);
            put(f.unitB);
            put(f.unitD);
            put(log2Q8(uint64_t(f.inner)));
            put(log2Q8(uint64_t(f.innerKept)));
            put(static_cast<int64_t>(f.route));
            put(f.f64Route ? 1 : 0);
            put(f.body);
            put(f.tileM);
            put(f.tileN);
            put(f.tileK);
            put(f.block);
            put(f.persistent);
            put(f.streamK ? 1 : 0);
            put(f.kBlock ? 1 : 0);
            put(f.direct ? 1 : 0);
            put(static_cast<int64_t>(p.steps.size()));
            put(static_cast<int64_t>(p.memsets.size()));
            put(log2Q8(p.workspaceBytes));
            put(log2Q8(f.bytes));
            put(log2Q8(f.flops));
            put(f.bytes > 0 ? static_cast<int64_t>(std::min<uint64_t>(f.flops, uint64_t(1) << 52)
                                                   * 256 / f.bytes)
                            : 0);
            put(f.ctas);
            put(f.ctas * 256 / cus);
            int64_t tiles = f.ctas;
            if(f.tileM > 0 && f.tileN > 0)
            {
                int64_t const n = f.N * (f.route == Route::CF32 ? 2 : 1);
                tiles           = ceilDiv(f.M, f.tileM) * ceilDiv(n, f.tileN) * f.L;
            }
            put(tiles * 256 / cus);
            put((tiles * 256 / cus) % 256);
            int64_t const volume = f.M * f.N * f.K * f.L;
            put(f.paddedVolume > 0 && volume > 0 ? f.paddedVolume * 256 / volume : 256);
            put(f.split);
            put(log2Q8(uint64_t(std::max<int64_t>(f.chunk, 1))));
            put(f.main != nullptr ? f.main->groupSegment : 0);
            put(f.main != nullptr ? f.main->privateSegment : 0);
            put(cus);
        }

        std::vector<std::string> selectionCells(Problem const& problem)
        {
            auto const type = problem.t[0].type;
            if(problem.kind == OpKind::Contraction)
            {
                bool const c16 = problem.compute == HIPTENSOR_COMPUTE_DESC_16F
                                 || problem.compute == HIPTENSOR_COMPUTE_DESC_16BF;
                char const* cls = type == HIPTENSOR_R_16F || type == HIPTENSOR_R_16BF ? "h"
                                  : type == HIPTENSOR_R_32F ? (c16 ? "h" : "s")
                                  : type == HIPTENSOR_C_32F ? "c"
                                  : type == HIPTENSOR_R_64F ? "d"
                                                            : "z";
                return {std::string("contraction.") + cls, "contraction"};
            }
            if(problem.kind == OpKind::Reduction)
            {
                char const* cls = type == HIPTENSOR_R_64F ? "d" : "s";
                return {std::string("reduction.") + cls, "reduction"};
            }
            return {};
        }

        char const* sourceName(SelectSource source)
        {
            switch(source)
            {
            case SelectSource::Model:
                return "MODEL";
            case SelectSource::Tuned:
                return "TUNED";
            default:
                return "STATIC";
            }
        }

        Selection rankCandidates(Problem const&                problem,
                                 Pack const&                   pack,
                                 std::vector<Candidate> const& candidates,
                                 bool                          useModel)
        {
            Selection  sel;
            auto const n = candidates.size();
            sel.order.resize(n);
            std::iota(sel.order.begin(), sel.order.end(), 0);
            sel.scores.assign(n, std::numeric_limits<double>::quiet_NaN());
            std::stable_sort(sel.order.begin(), sel.order.end(), [&candidates](int a, int b) {
                return candidates[a].staticRank < candidates[b].staticRank;
            });

            // 1. rocKE's tuned route table
            auto tuned = std::find_if(sel.order.begin(), sel.order.end(), [&candidates](int i) {
                return candidates[i].tuned;
            });
            if(tuned != sel.order.end())
            {
                std::rotate(sel.order.begin(), tuned, tuned + 1);
                sel.source = SelectSource::Tuned;
                sel.model  = "tuned route table";
                return sel;
            }

            // 2. the model of the problem's cell
            auto const cells = selectionCells(problem);
            if(cells.empty() || n < 2)
            {
                sel.model = n < 2 ? "one candidate" : "no model for this operation";
                return sel;
            }
            if(!useModel)
            {
                sel.model = "model disabled";
                return sel;
            }
            std::shared_ptr<Model const> model;
            for(auto const& cell : cells)
            {
                std::string why;
                model = modelFor(pack, cell, why);
                if(model != nullptr)
                {
                    break;
                }
                if(sel.model.empty() || why.find("not used") != std::string::npos)
                {
                    sel.model = why;
                }
            }
            if(model == nullptr)
            {
                // 3. the static order
                return sel;
            }
            std::vector<int64_t> x;
            for(size_t i = 0; i < n; i++)
            {
                features(problem, candidates[i], x);
                sel.scores[i] = model->predict(x);
            }
            std::stable_sort(sel.order.begin(), sel.order.end(), [&](int a, int b) {
                if(sel.scores[a] != sel.scores[b])
                {
                    return sel.scores[a] < sel.scores[b];
                }
                return candidates[a].staticRank < candidates[b].staticRank;
            });
            sel.source = SelectSource::Model;
            sel.model  = model->id;
            return sel;
        }
    } // namespace rockebackend
} // namespace hiptensor
