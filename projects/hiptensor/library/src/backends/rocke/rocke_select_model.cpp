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
#include <type_traits>

#include "rocke_select_model.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace selectmodel
        {
            namespace
            {
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

            } // namespace

            // Parses a model of the pack's route table:
            //   v3;schema=<n>;routes=<route>,...;margin=<m>;gemm=<sha256 hex|->;
            //      features=<name>,...;trees=<n>|<tree>...
            //   v2;schema=<n>;routes=<route>,...;margin=<m>;features=<name>,...;trees=<n>|<tree>...
            //   v1;schema=<n>;features=<name>,...;trees=<n>|<tree>|<tree>...
            //   tree: <leaves>;<split features>;<thresholds>;<left>;<right>;<leaf values>
            // (lists space separated; a one-leaf tree has empty node lists). A v1 model scores
            // the routes of the schema-1 corpus (legacyRoute).
            bool parseModel(std::string const& cell,
                            std::string const& text,
                            Model&             model,
                            std::string&       why)
            {
                auto parts = split(text, '|');
                auto head  = split(parts[0], ';');
                if(!((head.size() == 4 && head[0] == "v1") || (head.size() == 6 && head[0] == "v2")
                     || (head.size() == 7 && head[0] == "v3")))
                {
                    why = "unknown model format";
                    return false;
                }
                model.format = head[0][1] - '0';
                // feature schemas are append only: a model of an older schema reads features
                // this library still computes with the same meaning
                bool knownSchema = false;
                for(int s = 1; s <= kSelectSchema; s++)
                {
                    if(head[1] == "schema=" + std::to_string(s))
                    {
                        knownSchema  = true;
                        model.schema = s;
                    }
                }
                if(!knownSchema)
                {
                    why = "model of another feature schema (" + head[1] + ")";
                    return false;
                }
                if(model.format == 3)
                {
                    auto const gemm   = head[4].rfind("gemm=", 0) == 0 ? head[4].substr(5) : "?";
                    bool const hex256 = gemm.size() == 64
                                        && gemm.find_first_not_of("0123456789abcdef")
                                               == std::string::npos;
                    if(gemm != "-" && !hex256)
                    {
                        why = "malformed model header (gemm)";
                        return false;
                    }
                    model.gemm = gemm == "-" ? std::string() : gemm;
                    head.erase(head.begin() + 4);
                }
                if(model.format >= 2)
                {
                    if(head[2].rfind("routes=", 0) != 0)
                    {
                        why = "malformed model header";
                        return false;
                    }
                    for(auto const& r : split(head[2].substr(7), ','))
                    {
                        char*      end   = nullptr;
                        auto const route = std::strtol(r.c_str(), &end, 10);
                        if(r.empty() || *end != '\0' || route < 0 || size_t(route) >= kRoutes)
                        {
                            why = "malformed model routes";
                            return false;
                        }
                        model.routes[size_t(route)] = true;
                    }
                    char* end   = nullptr;
                    bool  valid = head[3].rfind("margin=", 0) == 0 && head[3].size() > 7;
                    if(valid)
                    {
                        model.margin = std::strtod(head[3].c_str() + 7, &end);
                        valid = *end == '\0' && std::isfinite(model.margin) && model.margin >= 0.0;
                    }
                    if(!valid)
                    {
                        why = "malformed model margin";
                        return false;
                    }
                    head.erase(head.begin() + 2, head.begin() + 4);
                }
                else
                {
                    for(size_t r = 0; r < kRoutes; r++)
                    {
                        model.routes[r] = legacyRoute(static_cast<int>(r));
                    }
                }
                if(head[2].rfind("features=", 0) != 0 || head[3].rfind("trees=", 0) != 0)
                {
                    why = "malformed model header";
                    return false;
                }
                auto const& names = featureNames();
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
                    model.readsPrediction = model.readsPrediction || name == "lg_pred_ns";
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

            std::vector<char const*> const& featureNames()
            {
                // The feature schema (version kSelectSchema). Append only; a change of meaning is a
                // new schema version.
                static std::vector<char const*> const names = {"op",
                                                                "dtype",
                                                                "out_dtype",
                                                                "compute16",
                                                                "bilinear",
                                                                "opnd_op",
                                                                "lg_m",
                                                                "lg_n",
                                                                "lg_k",
                                                                "lg_l",
                                                                "lg_mnk",
                                                                "modes_m",
                                                                "modes_n",
                                                                "modes_k",
                                                                "modes_l",
                                                                "unit_a",
                                                                "unit_b",
                                                                "unit_d",
                                                                "lg_inner",
                                                                "lg_inner_kept",
                                                                "route",
                                                                "f64route",
                                                                "body",
                                                                "tile_m",
                                                                "tile_n",
                                                                "tile_k",
                                                                "block",
                                                                "persistent",
                                                                "streamk",
                                                                "kblock",
                                                                "direct",
                                                                "steps",
                                                                "memsets",
                                                                "lg_ws",
                                                                "lg_bytes",
                                                                "lg_flops",
                                                                "intensity_q8",
                                                                "ctas",
                                                                "waves_q8",
                                                                "tiles_q8",
                                                                "tail_q8",
                                                                "pad_q8",
                                                                "split",
                                                                "lg_chunk",
                                                                "lds",
                                                                "scratch",
                                                                "cus",
                                                                // schema 2
                                                                "gett_kind",
                                                                "lg_pred_ns",
                                                                // schema 3
                                                                "jit_mode"};
                return names;
            }

            bool legacyRoute(int route)
            {
                // the ring GETT (7), the direct GETT (8), the tuned reduction classes (10) and
                // the contraction JIT's route (11) came after the schema-1 corpus
                return route != 7 && route != 8 && route != 10 && route != 11;
            }

            double Model::predict(std::vector<int64_t> const& x) const
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
                        auto const value = static_cast<double>(x[featureIndex[t.feature[node]]]);
                        node             = value <= t.threshold[node] ? t.left[node] : t.right[node];
                    }
                    sum += t.leaf[~node];
                }
                return sum;
            }
        } // namespace selectmodel
    } // namespace rockebackend
} // namespace hiptensor
