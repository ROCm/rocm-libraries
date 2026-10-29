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
#include <mutex>
#include <string>

#include "data_types.hpp"
#include "logger.hpp"
#include "plan_cache.hpp"
#include "platform.hpp"
#include "rocke_autotune.hpp"
#include "rocke_backend.hpp"
#ifdef HIPTENSOR_ENABLE_ROCKE_JIT
#include "rocke_jit.hpp"
#endif
#include "rocke_plan.hpp"
#include "rocke_runtime.hpp"
#include "rocke_select.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            using backend::OpKind;
            using backend::Problem;

            constexpr uint64_t kUnlimited = std::numeric_limits<uint64_t>::max();

            // Candidates the measured selection times by default, and its repetitions.
            constexpr int kAutotuneTopK = 4;
            constexpr int kAutotuneReps = 5;

            // The prepared program of a plan: launches only.
            class RockePrepared final : public backend::IPrepared
            {
            public:
                RockePrepared(Candidate candidate)
                    : mProgram(std::move(candidate.program))
                {
                    mId.backend   = backend::kRockeBackendName;
                    mId.specKey   = candidate.key + "|" + mProgram.description;
                    mId.kernelAbi = kKernelAbi;
                    mId.uid       = backend::kernelUid(mId.backend, mId.specKey, mId.kernelAbi);
                }

                backend::KernelId const& kernelId() const override
                {
                    return mId;
                }

                uint64_t workspaceSize() const override
                {
                    return mProgram.workspaceBytes;
                }

                hiptensorStatus_t run(backend::PlanContext const&  context,
                                      backend::Scalars const&      scalars,
                                      backend::Buffers const&      buffers,
                                      void*                        workspace,
                                      uint64_t                     workspaceSize,
                                      backend::LaunchConfig const& cfg) const override
                {
                    static_cast<void>(context);
                    return runProgram(
                        mProgram, scalars, buffers, workspace, workspaceSize, cfg.stream);
                }

            private:
                backend::KernelId mId;
                Program           mProgram;
            };

            bool envSet(char const* name)
            {
                auto value = getEnvironmentVariable(name);
                return value && !value->empty() && *value != "0";
            }

            int envInt(char const* name, int fallback)
            {
                auto value = getEnvironmentVariable(name);
                if(!value || value->empty())
                {
                    return fallback;
                }
                if(*value == "all")
                {
                    return std::numeric_limits<int>::max();
                }
                char* end = nullptr;
                long  v   = std::strtol(value->c_str(), &end, 10);
                return end != value->c_str() && v > 0 ? static_cast<int>(v) : fallback;
            }

            // HIPTENSOR_ROCKE_SELECT=static: the static order instead of the pack's model.
            bool useModel()
            {
                auto value = getEnvironmentVariable("HIPTENSOR_ROCKE_SELECT");
                return !(value && *value == "static");
            }

            // HIPTENSOR_ROCKE_F32_RECIPE=native|x3|x4|x6|x9|f64: how contractions of f32 /
            // complex f32 data with the 32F / C32F compute type multiply (rocke_plan.hpp
            // F32Recipe). Unset or empty: native. Another value: native, and the plan log says
            // so (no recipe less accurate than the default is taken by mistake).
            F32Recipe f32Recipe(std::string& note)
            {
                auto value = getEnvironmentVariable("HIPTENSOR_ROCKE_F32_RECIPE");
                if(!value || value->empty())
                {
                    return F32Recipe::Native;
                }
                F32Recipe recipe = F32Recipe::Native;
                if(!parseF32Recipe(*value, recipe))
                {
                    note = "HIPTENSOR_ROCKE_F32_RECIPE=" + *value
                           + " is not one of native, x3, x4, x6, x9, f64: native";
                    return F32Recipe::Native;
                }
                return recipe;
            }

            // Whether plans of `pref` take the JIT's routes (HIPTENSOR_JIT_MODE_DEFAULT in a
            // library built with the JIT).
            bool jitPlan(hiptensorPlanPreference const* pref)
            {
#ifdef HIPTENSOR_ENABLE_ROCKE_JIT
                return pref != nullptr && pref->mJit == HIPTENSOR_JIT_MODE_DEFAULT;
#else
                static_cast<void>(pref);
                return false;
#endif
            }

            // HIPTENSOR_JIT_MAX_MS: the compile budget of one plan (0: none).
            double jitBudgetMs()
            {
                auto v = getEnvironmentVariable("HIPTENSOR_JIT_MAX_MS");
                return v && !v->empty() ? std::atof(v->c_str()) : 0.0;
            }

            CandidateOptions
                candidateOptions(Pack const& pack, bool jit, std::string* note = nullptr)
            {
                CandidateOptions options;
                options.jitGett = jit;
                // HIPTENSOR_ROCKE_CONTRACTION_ROUTE=f64 (debug and test): only the F64 route.
                auto route      = getEnvironmentVariable("HIPTENSOR_ROCKE_CONTRACTION_ROUTE");
                options.f64Only = route && *route == "f64";
                options.disableTuned
                    = getEnvironmentVariable("HIPTENSOR_ROCKE_DISABLE_TUNED").has_value();
                std::string recipeNote;
                options.f32Recipe = f32Recipe(recipeNote);
                if(note != nullptr)
                {
                    *note = recipeNote;
                }
                // the steps of a trinary contraction: the selection of a contraction
                bool const model = useModel();
                options.rankStep = [&pack, model](Problem const&                problem,
                                                  std::vector<Candidate> const& candidates) {
                    return rankCandidates(problem, pack, candidates, model).order;
                };
                return options;
            }

            // The plan-cache identity of a candidate: its key in this pack.
            // The plan-cache identity of a candidate: the pack, the f32 recipe that listed it
            // when not x6 (CandidateFacts::selectTag; a plan-cache line written under one recipe
            // names no candidate of another) and its key.
            uint64_t candidateUid(Pack const& pack, Candidate const& candidate)
            {
                auto const& tag = candidate.facts.selectTag;
                return backend::kernelUid(backend::kRockeBackendName,
                                          pack.packId() + "|" + (tag.empty() ? "" : tag + "|")
                                              + candidate.key,
                                          kKernelAbi);
            }

            std::string hexDigest(uint64_t value)
            {
                char hex[17];
                snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(value));
                return hex;
            }

            std::string jsonString(std::string const& text)
            {
                std::string out = "\"";
                for(char c : text)
                {
                    if(c == '"' || c == '\\')
                    {
                        out += '\\';
                    }
                    out += (static_cast<unsigned char>(c) < 0x20) ? ' ' : c;
                }
                return out + "\"";
            }

            // HIPTENSOR_ROCKE_SELECT_DUMP=<file>: appends one JSON line per candidate (features,
            // the selector's view, whether the plan took it, and the measured time when the
            // measured selection timed it), the training data of the selection models. A measured
            // selection writes its timed candidates; any other plan writes every candidate
            // without a time (the features of a problem pool, the picks of an evaluation).
            void dumpTimings(Problem const&                problem,
                             std::vector<Candidate> const& candidates,
                             Selection const&              sel,
                             AutotuneResult const*         result,
                             int                           reps,
                             int                           chosen)
            {
                auto path = getEnvironmentVariable("HIPTENSOR_ROCKE_SELECT_DUMP");
                if(!path || path->empty())
                {
                    return;
                }
                static std::mutex mutex;
                std::scoped_lock  lock(mutex);
                FILE*             file = std::fopen(path->c_str(), "a");
                if(file == nullptr)
                {
                    return;
                }
                auto tag   = getEnvironmentVariable("HIPTENSOR_ROCKE_SELECT_DUMP_TAG");
                auto cells = selectionCells(
                    problem, candidates.empty() ? std::string() : candidates[0].facts.selectTag);
                auto const&          names = featureNames();
                std::vector<int64_t> x;
                for(size_t i = 0; i < candidates.size(); i++)
                {
                    if(result != nullptr && result->medianUs[i] < 0.0)
                    {
                        continue;
                    }
                    auto const& c         = candidates[i];
                    int         modelRank = static_cast<int>(
                        std::find(sel.order.begin(), sel.order.end(), int(i)) - sel.order.begin());
                    features(problem, c, x);
                    std::string line = "{\"tag\":" + jsonString(tag ? *tag : "") + ",\"problem\":\""
                                       + hexDigest(problem.digest())
                                       + "\",\"arch\":" + jsonString(problem.device.baseArch)
                                       + ",\"cell\":" + jsonString(cells.empty() ? "" : cells[0])
                                       + ",\"key\":" + jsonString(c.key)
                                       + ",\"static_rank\":" + std::to_string(c.staticRank)
                                       + ",\"tuned\":" + (c.tuned ? "1" : "0")
                                       + ",\"rank\":" + std::to_string(modelRank)
                                       + ",\"pick\":" + (int(i) == chosen ? "1" : "0")
                                       + ",\"source\":" + jsonString(sourceName(sel.source))
                                       + ",\"model\":" + jsonString(sel.model);
                    char num[64];
                    if(std::isfinite(sel.scores[i]))
                    {
                        snprintf(num, sizeof(num), "%.17g", sel.scores[i]);
                        line += std::string(",\"score\":") + num;
                    }
                    if(result != nullptr)
                    {
                        snprintf(num, sizeof(num), "%.4f", result->medianUs[i]);
                        line += std::string(",\"us\":") + num
                                + ",\"reps\":" + std::to_string(reps);
                    }
                    line += ",\"f\":{";
                    for(size_t k = 0; k < names.size(); k++)
                    {
                        line += (k ? ",\"" : "\"") + std::string(names[k])
                                + "\":" + std::to_string(x[k]);
                    }
                    line += "}}\n";
                    std::fputs(line.c_str(), file);
                }
                std::fclose(file);
            }

            // The rocKE backend: the kernels of the target's kernel pack. A problem no candidate
            // admits is declined, and the fallback backend serves it.
            class RockeBackend final : public backend::IBackend
            {
            public:
                std::string_view name() const override
                {
                    return backend::kRockeBackendName;
                }

                uint32_t kernelAbi() const override
                {
                    return kKernelAbi;
                }

                backend::Verdict supportsArch(backend::DeviceKey const& device) const override
                {
                    std::string reason;
                    return {packFor(device.baseArch, reason) != nullptr, reason};
                }

                hiptensorStatus_t prepare(Problem const&                       problem,
                                          backend::PlanContext const&          context,
                                          std::unique_ptr<backend::IPrepared>& out,
                                          std::string& reason) const override
                {
                    auto const* pack = packFor(problem.device.baseArch, reason);
                    if(pack == nullptr)
                    {
                        return HIPTENSOR_STATUS_ARCH_MISMATCH;
                    }
                    std::vector<Candidate> candidates;
                    std::string            recipeNote;
                    auto*                  plan = context.plan;
                    auto const*            pref = plan != nullptr ? plan->mPref : nullptr;
                    bool const             jit  = jitPlan(pref);
                    if(auto status = enumerateCandidates(problem,
                                                         *pack,
                                                         candidateOptions(*pack, jit, &recipeNote),
                                                         context.workspaceSizeLimit,
                                                         candidates,
                                                         reason);
                       status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                    auto const sel = rankCandidates(problem, *pack, candidates, useModel());

                    // A JIT candidate's launches are built when it is selected (or timed): its
                    // kernels from the kernel cache, else compiled now. One that cannot be built
                    // (a compiler or builder refusal, the time budget) is not selected; the next
                    // candidate of the order is.
                    double const budget   = jitBudgetMs();
                    double       jitSpent = 0.0;
                    std::string  jitNotes;
                    auto         built = [&](int i) {
                        std::string jitNote, why;
                        auto        status = buildJitCandidate(
                            candidates[i], problem.device, budget, jitSpent, jitNote, why);
                        if(!jitNote.empty())
                        {
                            Logger::instance()->logHeuristics("hiptensorCreatePlan",
                                                              jitNote.c_str());
                        }
                        if(status != HIPTENSOR_STATUS_SUCCESS && jitNotes.empty())
                        {
                            jitNotes = candidates[i].key + " not built: " + why;
                        }
                        return status == HIPTENSOR_STATUS_SUCCESS;
                    };

                    auto*             cache = context.handle != nullptr && pref != nullptr
                                          && pref->mCacheMode == HIPTENSOR_CACHE_MODE_PEDANTIC
                                          && plan->mOpDesc != nullptr
                                                  ? context.handle->getPlanCache()
                                                  : nullptr;
                    PlanCache::HashId hash  = 0;
                    if(cache != nullptr)
                    {
                        hash = cache->getHashID(plan->mOpDesc, backend::kRockeBackendName);
                    }

                    int         chosen = sel.order.front();
                    std::string source = sourceName(sel.source);
                    std::string note   = recipeNote;
                    int const   rank   = pref != nullptr ? pref->mKernelRank : 0;
                    if(rank < 0 || size_t(rank) >= candidates.size())
                    {
                        reason = "KERNEL_RANK " + std::to_string(rank) + " is past the "
                                 + std::to_string(candidates.size()) + " rocKE candidates";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    bool decided = false;
                    if(rank > 0)
                    {
                        chosen  = sel.order[rank];
                        source  = "KERNEL_RANK";
                        decided = true;
                    }
                    if(!decided && cache != nullptr)
                    {
                        auto const uid = cache->querySolutionUidByHash(hash);
                        if(uid != 0)
                        {
                            for(size_t i = 0; i < candidates.size(); i++)
                            {
                                if(candidateUid(*pack, candidates[i]) == uid)
                                {
                                    chosen  = static_cast<int>(i);
                                    source  = "PLAN_CACHE";
                                    decided = true;
                                    break;
                                }
                            }
                            if(!decided)
                            {
                                note += std::string(note.empty() ? "" : "; ")
                                        + "the plan cache line names no candidate of this pack";
                            }
                        }
                    }
                    if(!decided && pref != nullptr
                       && pref->mSelectionAlgorithm == HIPTENSOR_ALGO_DEFAULT_PATIENT
                       && candidates.size() > 1)
                    {
                        int const topK = envInt("HIPTENSOR_ROCKE_AUTOTUNE_TOPK", kAutotuneTopK);
                        int const reps = envInt("HIPTENSOR_ROCKE_AUTOTUNE_REPS", kAutotuneReps);
                        std::vector<int> timed;
                        for(int i : sel.order)
                        {
                            if(timed.size() >= size_t(std::max(topK, 1)))
                            {
                                break;
                            }
                            if(built(i))
                            {
                                timed.push_back(i);
                            }
                        }
                        AutotuneResult result;
                        if(auto status = timed.empty()
                                             ? HIPTENSOR_STATUS_SUCCESS
                                             : autotune(problem, candidates, timed, reps, result);
                           status != HIPTENSOR_STATUS_SUCCESS)
                        {
                            reason = "the measured selection failed";
                            return status;
                        }
                        note = recipeNote.empty() || result.note.empty()
                                   ? recipeNote + result.note
                                   : recipeNote + "; " + result.note;
                        if(result.winner >= 0)
                        {
                            chosen = result.winner;
                            source = "AUTOTUNE";
                            dumpTimings(problem, candidates, sel, &result, reps, chosen);
                            if(cache != nullptr)
                            {
                                cache->addCacheLine(hash, candidateUid(*pack, candidates[chosen]));
                            }
                        }
                    }

                    // the selected candidate's launches (a JIT candidate's), else the next
                    // candidate of the order that has them
                    if(candidates[chosen].program.pending && !built(chosen))
                    {
                        auto pos  = std::find(sel.order.begin(), sel.order.end(), chosen);
                        int  next = -1;
                        for(auto it = pos == sel.order.end() ? sel.order.begin() : pos + 1;
                            it != sel.order.end();
                            ++it)
                        {
                            if(!candidates[*it].program.pending || built(*it))
                            {
                                next = *it;
                                break;
                            }
                        }
                        if(next < 0)
                        {
                            reason = "no rocKE candidate could be built: " + jitNotes;
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                        chosen = next;
                        source += "+NEXT";
                    }
                    if(!jitNotes.empty())
                    {
                        note += std::string(note.empty() ? "" : "; ") + jitNotes;
                    }

                    // the plan's kernel goes to the plan cache (PEDANTIC), as the measured winner
                    // does: later plans of the problem on the handle, or on a handle that reads
                    // the plan cache file, take it without selecting again
                    if(!decided && source != "AUTOTUNE" && cache != nullptr)
                    {
                        cache->addCacheLine(hash, candidateUid(*pack, candidates[chosen]));
                    }
                    if(source != "AUTOTUNE")
                    {
                        dumpTimings(problem, candidates, sel, nullptr, 0, chosen);
                    }
                    logSelection(
                        problem, candidates, sel, chosen, source, note, context.workspaceSizeLimit);
                    if(auto status = loadFunctions(candidates[chosen].program);
                       status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
#ifdef HIPTENSOR_ENABLE_ROCKE_JIT
                    // HIPTENSOR_JIT_MODE_DEFAULT: the elementwise launches of the selected
                    // program become kernels specialized to the problem (never with
                    // HIPTENSOR_JIT_MODE_NONE)
                    if(pref != nullptr && pref->mJit == HIPTENSOR_JIT_MODE_DEFAULT)
                    {
                        auto&     program = candidates[chosen].program;
                        JitReport report;
                        specializeProgram(program, problem.device, report, jitSpent);
                        if(report.launches > 0)
                        {
                            char msg[256];
                            snprintf(msg,
                                     sizeof(msg),
                                     "rocKE JIT: %d of %d launches specialized (compiled %d, "
                                     "kernel cache hits %d, of which from a file %d) in %.3f ms",
                                     report.specialized,
                                     report.launches,
                                     report.compiled,
                                     report.memoryHits,
                                     report.fileHits,
                                     report.milliseconds);
                            std::string line = msg;
                            if(!report.note.empty())
                            {
                                line += "; " + report.note;
                            }
                            Logger::instance()->logHeuristics("hiptensorCreatePlan", line.c_str());
                        }
                        if(report.specialized > 0)
                        {
                            program.description += " jit" + std::to_string(report.specialized);
                        }
                    }
#endif
                    out = std::make_unique<RockePrepared>(std::move(candidates[chosen]));
                    return HIPTENSOR_STATUS_SUCCESS;
                }

#ifdef HIPTENSOR_ENABLE_ROCKE_JIT
                bool hasKernelCache() const override
                {
                    return true;
                }

                hiptensorStatus_t writeKernelCache(backend::DeviceKey const& device,
                                                   char const* fileName) const override
                {
                    return writeKernelCacheFile(fileName, device);
                }

                hiptensorStatus_t readKernelCache(backend::DeviceKey const& device,
                                                  char const*               fileName) override
                {
                    return readKernelCacheFile(fileName, device);
                }
#endif

                hiptensorStatus_t estimateWorkspace(Problem const&                  problem,
                                                    backend::EstimateContext const& context,
                                                    uint64_t&                       estimate,
                                                    std::string& reason) const override
                {
                    auto const* pack = packFor(problem.device.baseArch, reason);
                    if(pack == nullptr)
                    {
                        return HIPTENSOR_STATUS_ARCH_MISMATCH;
                    }
                    std::vector<Candidate> candidates;
                    if(auto status
                       = enumerateCandidates(problem,
                                             *pack,
                                             candidateOptions(*pack, jitPlan(context.planPref)),
                                             kUnlimited,
                                             candidates,
                                             reason);
                       status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                    auto const sel = rankCandidates(problem, *pack, candidates, useModel());
                    // DEFAULT: the workspace of the candidate a plan selects (of every
                    // candidate the measured selection may time); MIN: the least that some
                    // candidate runs with; MAX: what every candidate runs with.
                    auto const& pref  = context.planPref;
                    size_t      count = 1;
                    int const   rank  = pref != nullptr ? pref->mKernelRank : 0;
                    if(rank > 0)
                    {
                        // KERNEL_RANK orders the candidates a plan admits: room for all of
                        // them keeps the order of the estimate
                        count = sel.order.size();
                    }
                    else if(pref != nullptr
                            && pref->mSelectionAlgorithm == HIPTENSOR_ALGO_DEFAULT_PATIENT)
                    {
                        count = static_cast<size_t>(
                            envInt("HIPTENSOR_ROCKE_AUTOTUNE_TOPK", kAutotuneTopK));
                    }
                    if(context.workspacePref == HIPTENSOR_WORKSPACE_MIN)
                    {
                        count = 0;
                    }
                    else if(context.workspacePref == HIPTENSOR_WORKSPACE_MAX)
                    {
                        count = sel.order.size();
                    }
                    if(count == 0)
                    {
                        estimate = kUnlimited;
                        for(auto const& c : candidates)
                        {
                            estimate = std::min(estimate, c.program.workspaceBytes);
                        }
                        return HIPTENSOR_STATUS_SUCCESS;
                    }
                    estimate = 0;
                    for(size_t i = 0; i < std::min(count, sel.order.size()); i++)
                    {
                        estimate
                            = std::max(estimate, candidates[sel.order[i]].program.workspaceBytes);
                    }
                    // a plan of this problem takes the plan cache's kernel: room for it too
                    bool const pedantic
                        = pref == nullptr || pref->mCacheMode == HIPTENSOR_CACHE_MODE_PEDANTIC;
                    auto* cache
                        = context.handle != nullptr ? context.handle->getPlanCache() : nullptr;
                    if(pedantic && cache != nullptr && context.desc != nullptr && rank == 0)
                    {
                        auto const uid = cache->querySolutionUid(
                            const_cast<hiptensorOperationDescriptor_t>(context.desc),
                            backend::kRockeBackendName);
                        for(auto const& c : candidates)
                        {
                            if(uid != 0 && candidateUid(*pack, c) == uid)
                            {
                                estimate = std::max(estimate, c.program.workspaceBytes);
                            }
                        }
                    }
                    return HIPTENSOR_STATUS_SUCCESS;
                }

            private:
                static void logSelection(Problem const&                problem,
                                         std::vector<Candidate> const& candidates,
                                         Selection const&              sel,
                                         int                           chosen,
                                         std::string const&            source,
                                         std::string const&            note,
                                         uint64_t                      workspaceLimit)
                {
                    auto&       logger = Logger::instance();
                    std::string msg
                        = std::string("rocKE selection: Op: ") + backend::opKindName(problem.kind)
                          + ", Candidates: " + std::to_string(candidates.size())
                          + " (workspace limit "
                          + (workspaceLimit == kUnlimited ? std::string("none")
                                                          : std::to_string(workspaceLimit))
                          + ")" + ", Source: " + source + " (" + sel.model + ")"
                          + ", Pick: " + candidates[chosen].key + ", Workspace: "
                          + std::to_string(candidates[chosen].program.workspaceBytes)
                          + ", Launches: ";
                    // the kernel families of the program's launches, in order
                    auto const& steps = candidates[chosen].program.steps;
                    for(size_t i = 0; i < steps.size(); i++)
                    {
                        msg += (i ? "+" : "") + steps[i].kernel->family;
                    }
                    msg += ", Order:";
                    for(size_t i = 0; i < std::min<size_t>(sel.order.size(), 5); i++)
                    {
                        auto const k = sel.order[i];
                        msg += " " + candidates[k].key;
                        if(std::isfinite(sel.scores[k]))
                        {
                            char num[32];
                            snprintf(num, sizeof(num), "(%.3f)", sel.scores[k]);
                            msg += num;
                        }
                    }
                    if(!note.empty())
                    {
                        msg += "; " + note;
                    }
                    logger->logHeuristics("hiptensorCreatePlan", msg.c_str());
                }
            };
        } // namespace

        std::unique_ptr<backend::IBackend> makeRockeBackend()
        {
            return std::make_unique<RockeBackend>();
        }
    } // namespace rockebackend
} // namespace hiptensor
