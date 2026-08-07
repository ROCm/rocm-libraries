// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/UtilsOrigami.hpp>
#include <Tensile/hip/HipHardware.hpp>

namespace TensileLite
{
    /**
     * Returns true when the ExactLogicLibrary selection gate restricts selection
     * to prediction/analytical rows: the prediction library is forced
     * (TENSILE_USE_PREDICTION) or StreamK dynamic scheduling is active. In that
     * mode EqualityMatching/RangeMatching rows are skipped, so the unified union
     * (which includes them) must not be used. This mirrors the effectiveDynamic /
     * predictionLib computation in ExactLogicLibrary::findTopSolutions.
     */
    inline bool predictionLibrarySelectionActive(ContractionProblemGemm const& problem)
    {
        const auto forceDynamic     = Debug::Instance().streamK5ForceMode();
        const bool effectiveDynamic = (forceDynamic == 1)
                                      || (forceDynamic != 0
                                          && problem.getParams().streamKTileSchedulingMode() != 0);
        return Debug::Instance().usePredictionLibrary() || effectiveDynamic;
    }

    /**
     * Draw `numSolutions` solutions from an analytically-ranked union of every
     * candidate library, rather than concatenating each library's independently
     * sorted results (the default findTopSolutions behavior).
     *
     * Selection order:
     *   1. Exact-tuned (EqualityMatching) matches are pinned on top.
     *   2. The remaining slots are filled from the predicate-valid union of all
     *      other libraries, ranked by the Origami analytical model (predicted
     *      runtime, best first).
     *
     * The union is de-duplicated by solution index. Falls back to the library's
     * native findTopSolutions when the analytical model is unavailable (no
     * analytical hardware) or when an unsupported problem is encountered.
     */
    inline SolutionVector<ContractionSolution>
        findTopSolutionsUnified(SolutionLibrary<ContractionProblemGemm> const& library,
                                ContractionProblemGemm const&                  problem,
                                Hardware const&                                hardware,
                                int                                            numSolutions)
    {
        // Defensive: the caller (getSolutions) only invokes this when the analytical
        // model is available, but guard anyway so the function is safe standalone.
        auto const* pAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
        if(!(pAMDGPU && pAMDGPU->analyticalHardware))
            return library.findTopSolutions(problem, hardware, numSolutions);

        const size_t want = numSolutions > 0 ? static_cast<size_t>(numSolutions) : 0;

        SolutionVector<ContractionSolution> rv;
        if(want == 0)
            return rv;

        // 1. De-duplicated union across all libraries. GEMM_TYPE_ONLY is required
        //    so the prediction (analytical) row contributes its kernels; a DEFAULT
        //    search returns an empty set from the prediction / free-size rows and
        //    would drop the entire analytical library from the candidate pool.
        SolutionSet<ContractionSolution> all = library.findAllSolutions(
            problem, hardware, SolutionLibrarySearchType::GEMM_TYPE_ONLY);

        // 2. GEMM_TYPE_ONLY does not apply the per-solution predicates, so filter
        //    here to keep only solutions valid for this problem/hardware. This
        //    mirrors the native findTopSolutions path, whose tuned rows validate
        //    hardware + problem + task predicates (via findBestSolution); the same
        //    task predicate is also enforced downstream at algo-support time.
        auto matchesProblem = [&](std::shared_ptr<ContractionSolution> const& sol) {
            if(!sol)
                return false;
            if(sol->hardwarePredicate && !(*sol->hardwarePredicate)(hardware))
                return false;
            if(sol->problemPredicate && !(*sol->problemPredicate)(problem))
                return false;
            if(sol->taskPredicate)
            {
                Task task(hardware, problem, *sol);
                if(!(*sol->taskPredicate)(task))
                    return false;
            }
            return true;
        };

        // 3. Partition exact-tuned matches from the analytically-ranked rest.
        std::vector<std::shared_ptr<ContractionSolution>> exacts;
        std::vector<std::shared_ptr<ContractionSolution>> rest;
        for(auto const& sol : all)
        {
            if(!matchesProblem(sol))
                continue;
            if(sol->tag == ContractionSolution::MatchingTag::Equal)
                exacts.push_back(sol);
            else
                rest.push_back(sol);
        }

        std::set<int> used;
        auto          pushUnique = [&](std::shared_ptr<ContractionSolution> const& sol) {
            if(rv.size() >= want || !sol)
                return;
            if(used.insert(sol->index).second)
                rv.push_back(sol);
        };

        // 4. Pin exact-tuned matches first.
        for(auto const& sol : exacts)
            pushUnique(sol);

        // 5. Analytically rank and append the remainder.
        if(rv.size() < want && !rest.empty())
        {
            try
            {
                origami::hardware_t ranking_hardware = makeRankingHardware(*pAMDGPU);
                origami::problem_t  origami_problem  = makeOrigamiProblem(problem);

                std::vector<origami::config_t> configs;
                configs.reserve(rest.size());
                for(size_t i = 0; i < rest.size(); ++i)
                    configs.emplace_back(makeOrigamiConfig(*rest[i], static_cast<int>(i)));

                auto ranked = origami::rank_configs(origami_problem, ranking_hardware, configs);

                for(auto const& r : ranked)
                {
                    if(r.config.index >= rest.size())
                        continue;
                    pushUnique(rest[r.config.index]);
                    if(rv.size() >= want)
                        break;
                }
            }
            catch(std::exception const& e)
            {
                // Unsupported dtype or analytical failure: keep the pinned
                // exact-tuned matches and fill the remainder from the library's
                // native ordering (de-duplicated), preserving the guarantee that
                // exacts stay at the top.
                if(Debug::Instance().printPropertyEvaluation())
                    std::cerr << "TensileLite::findTopSolutionsUnified: analytical ranking "
                                 "failed ("
                              << e.what() << "); falling back to findTopSolutions.\n";
                for(auto const& sol : library.findTopSolutions(problem, hardware, numSolutions))
                    pushUnique(sol);
                return rv;
            }
        }

        // 6. Backfill any predicate-valid candidates the analytical model did not
        //    rank. origami::rank_configs may silently drop configs it cannot score
        //    (e.g. LDS- or heuristic-rejected), so append the remaining valid
        //    solutions in union order rather than returning fewer than requested.
        for(auto const& sol : rest)
        {
            if(rv.size() >= want)
                break;
            pushUnique(sol);
        }

        return rv;
    }
} // namespace TensileLite
