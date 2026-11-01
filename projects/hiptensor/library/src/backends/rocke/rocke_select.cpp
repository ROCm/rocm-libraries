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
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <tuple>
#include <type_traits>

#include "platform.hpp"
#include "rocke_select.hpp"
#include "rocke_select_model.hpp"
#ifdef HIPTENSOR_ENABLE_ROCKE_JIT
#include "rocke_gett_jit.hpp"
#endif

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            using backend::OpKind;
            using backend::Problem;
            using selectmodel::Model;
            using selectmodel::parseModel;

            static_assert(int(Route::RingGett) == 7 && int(Route::DirectGett) == 8
                              && int(Route::TunedReduction) == 10 && int(Route::JitGett) == 11,
                          "selectmodel::legacyRoute names these Route values");

            bool legacyRoute(Route route)
            {
                return selectmodel::legacyRoute(int(route));
            }

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

            // The text of a cell's model: the pack's route select.model.<cell>, or with
            // HIPTENSOR_ROCKE_SELECT_MODELS=<dir> the file <dir>/<target>/<cell>.model (models
            // being trained or evaluated without a new pack; a cell without a file there has no
            // model). Empty when there is none.
            std::string modelText(Pack const& pack, std::string const& dir, std::string const& cell)
            {
                if(dir.empty())
                {
                    return pack.route("select.model." + cell);
                }
                std::ifstream in(dir + "/" + pack.arch() + "/" + cell + ".model", std::ios::binary);
                if(!in)
                {
                    return {};
                }
                std::string text((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
                while(!text.empty()
                      && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
                {
                    text.pop_back();
                }
                return text;
            }

            // The model of a cell (modelText), parsed once per process and source. nullptr with
            // `why` when there is none or it does not load.
            std::shared_ptr<Model const>
                modelFor(Pack const& pack, std::string const& cell, std::string& why)
            {
                static std::mutex mutex;
                static std::map<std::tuple<Pack const*, std::string, std::string>,
                                std::pair<std::shared_ptr<Model const>, std::string>>
                                  cache;
                auto const        env = getEnvironmentVariable("HIPTENSOR_ROCKE_SELECT_MODELS");
                std::string const dir = env ? *env : std::string();
                std::scoped_lock  lock(mutex);
                auto              key = std::make_tuple(&pack, dir, cell);
                auto              it  = cache.find(key);
                if(it == cache.end())
                {
                    std::shared_ptr<Model const> model;
                    std::string                  reason;
                    auto                         text = modelText(pack, dir, cell);
                    if(text.empty())
                    {
                        reason = "no model for " + cell
                                 + (dir.empty() ? "" : " in HIPTENSOR_ROCKE_SELECT_MODELS");
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
            return selectmodel::featureNames();
        }

        void features(Problem const&        problem,
                      Candidate const&      candidate,
                      std::vector<int64_t>& out,
                      int                   schema)
        {
            auto const&   f   = candidate.facts;
            auto const&   p   = candidate.program;
            int64_t const cus = std::max(1, problem.device.cuCount);
            out.assign(selectmodel::featureNames().size(), 0);
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
            // a JitGett candidate's features are the same before and after its launches are
            // built (the planned launches; no kernel metadata)
            bool const jit = f.route == Route::JitGett;
            put(jit ? int64_t(f.launches) : static_cast<int64_t>(p.steps.size()));
            put(jit ? 0 : static_cast<int64_t>(p.memsets.size()));
            put(log2Q8(p.workspaceBytes));
            // schema 2 had no byte estimate for the JIT route
            uint64_t const bytes = jit && schema < 3 ? 0 : f.bytes;
            put(log2Q8(bytes));
            put(log2Q8(f.flops));
            put(bytes > 0 ? static_cast<int64_t>(std::min<uint64_t>(f.flops, uint64_t(1) << 52)
                                                 * 256 / bytes)
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
            put(f.main != nullptr && !jit ? f.main->groupSegment : 0);
            put(f.main != nullptr && !jit ? f.main->privateSegment : 0);
            put(cus);
            // schema 2
            put(f.gettKind);
            put(log2Q8(uint64_t(std::max<int64_t>(f.predictedNs, 0))));
            // schema 3
            put(f.jitMode ? 1 : 0);
        }

        std::vector<std::string> selectionCells(Problem const& problem, std::string const& tag)
        {
            auto const type = problem.t[0].type;
            if(problem.kind == OpKind::Contraction)
            {
                if(!tag.empty())
                {
                    auto cells = selectionCells(problem);
                    for(auto& cell : cells)
                    {
                        cell += "." + tag;
                    }
                    return cells;
                }
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
            case SelectSource::Rule:
                return "RULE";
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

            // 1. rocKE's tuned route tables (exact problems rocKE measured)
            auto tuned = std::find_if(sel.order.begin(), sel.order.end(), [&candidates](int i) {
                return candidates[i].tuned && candidates[i].facts.route != Route::JitGett;
            });
            if(tuned != sel.order.end())
            {
                std::rotate(sel.order.begin(), tuned, tuned + 1);
                sel.source = SelectSource::Tuned;
                sel.model  = "tuned route table";
                return sel;
            }

            // 2. the model of the problem's cell (an f32 recipe other than the ones the pack has
            // models for: cells no pack has a model for, so the rules)
            auto const cells
                = selectionCells(problem, n > 0 ? candidates[0].facts.selectTag : std::string());
            std::shared_ptr<Model const> model;
            if(cells.empty() || n < 2)
            {
                sel.model = n < 2 ? "one candidate" : "no model for this operation";
            }
            else if(!useModel)
            {
                sel.model = "model disabled";
            }
            else
            {
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
                // a model that reads the planner's predicted time needs the planner's GEMM
                // selection models it was trained with (the features of the JIT candidate)
#ifdef HIPTENSOR_ENABLE_ROCKE_JIT
                bool const jitCandidate
                    = std::any_of(candidates.begin(), candidates.end(), [](Candidate const& c) {
                          return c.facts.route == Route::JitGett;
                      });
                if(model != nullptr && model->readsPrediction && jitCandidate)
                {
                    auto const digest = gettJitModelsDigest(pack);
                    if(model->gemm != digest)
                    {
                        sel.model = "model " + model->id
                                    + " not used: trained with other GEMM selection models of "
                                      "rocKE's tensor GETT planner ("
                                    + (model->gemm.empty() ? std::string("none named")
                                                           : model->gemm.substr(0, 16))
                                    + ", the pack's " + digest.substr(0, 16) + ")";
                        sel.note = sel.model;
                        model    = nullptr;
                    }
                }
#endif
            }
            // the candidates the model scores: the routes it was trained on (without a model,
            // the routes of the schema-1 corpus, so that the rules below order the others)
            auto scored = [&candidates, &model](int i) {
                auto const route = candidates[i].facts.route;
                return model != nullptr ? model->scores(int(route)) : legacyRoute(route);
            };

            // 3. the rules for the routes the model does not score: the route rocKE's tensor
            // GETT planner chooses (JIT) first, then a preferred candidate (the ring GETT, the
            // direct GETT on a target without the GEMM routes, the tuned reduction classes);
            // the others keep the static order after the scored ones
            auto jit = std::find_if(sel.order.begin(), sel.order.end(), [&](int i) {
                return candidates[i].facts.route == Route::JitGett && !scored(i);
            });
            if(jit != sel.order.end())
            {
                std::rotate(sel.order.begin(), jit, jit + 1);
                sel.source = SelectSource::Tuned;
                sel.model  = "rocKE's tensor GETT planner (JIT)";
                return sel;
            }
            std::stable_partition(sel.order.begin(), sel.order.end(), scored);
            auto preferred = std::find_if(sel.order.begin(), sel.order.end(), [&](int i) {
                return candidates[i].preferred && !scored(i);
            });
            if(preferred != sel.order.end())
            {
                auto const route = candidates[*preferred].facts.route;
                std::rotate(sel.order.begin(), preferred, preferred + 1);
                sel.source = SelectSource::Rule;
                sel.model  = route == Route::RingGett         ? "the ring GETT rule"
                             : route == Route::TunedReduction ? "the tuned reduction rule"
                                                              : "the direct GETT rule";
                return sel;
            }
            if(model == nullptr)
            {
                // 4. the static order
                return sel;
            }
            std::vector<int64_t> x;
            for(size_t i = 0; i < n; i++)
            {
                if(scored(static_cast<int>(i)))
                {
                    features(problem, candidates[i], x, model->schema);
                    sel.scores[i] = model->predict(x);
                }
            }
            auto const last = std::find_if_not(sel.order.begin(), sel.order.end(), scored);
            // the static order's first scored candidate (before sorting: the scored candidates
            // keep the static order)
            int const anchor = sel.order.begin() != last ? sel.order.front() : -1;
            std::stable_sort(sel.order.begin(), last, [&](int a, int b) {
                if(sel.scores[a] != sel.scores[b])
                {
                    return sel.scores[a] < sel.scores[b];
                }
                return candidates[a].staticRank < candidates[b].staticRank;
            });
            // the model's best replaces the static order's first only when it is predicted
            // better by more than the model's margin
            if(anchor >= 0 && model->margin > 0.0
               && sel.scores[anchor] - sel.scores[sel.order.front()] <= model->margin)
            {
                auto it = std::find(sel.order.begin(), last, anchor);
                std::rotate(sel.order.begin(), it, it + 1);
            }
            sel.source = SelectSource::Model;
            sel.model  = model->id;
            return sel;
        }
    } // namespace rockebackend
} // namespace hiptensor
