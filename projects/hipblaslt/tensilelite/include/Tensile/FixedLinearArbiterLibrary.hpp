// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Utils.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <regex>
#include <string>
#include <vector>

namespace TensileLite
{
    /**
     * Selects one candidate from each child library, then compares those two
     * candidates with the frozen gfx1100 HHS-TN 22-feature linear ranker.
     * The feature contract intentionally matches offline_tournament.py v1;
     * StreamK mode is not a feature and exact score ties choose G0.
     */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct FixedLinearArbiterLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> g0Library;
        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> o3Library;
        std::string                                             modelId;
        std::string                                             featureSchema;
        std::vector<double>                                     weights;
        double                                                  cuCount = 96.0;
        mutable std::atomic<bool>                               lastFindTopRetAll = false;

        static std::string Type()
        {
            return "FixedLinearArbiter";
        }

        std::string type() const override
        {
            return Type();
        }

        std::string description() const override
        {
            return concatenate(Type(), ", model=", modelId);
        }

        std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                       Hardware const& hardware,
                                                       int index) const override
        {
            auto solution = g0Library ? g0Library->getSolutionByIndex(problem, hardware, index)
                                      : nullptr;
            if(solution)
                return solution;
            return o3Library ? o3Library->getSolutionByIndex(problem, hardware, index) : nullptr;
        }

        std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                     Hardware const& hardware,
                                                     double* fitness = nullptr) const override
        {
            auto solutions = findTopSolutions(problem, hardware, 1);
            return solutions.empty() ? nullptr : solutions.front();
        }

        SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                    Hardware const& hardware,
                                                    int numSolutions) const override
        {
            SolutionVector<MySolution> result;
            if(numSolutions <= 0)
                return result;

            // Each child is itself a complete selector. Ask it for its one
            // best solution rather than assuming it implements top-K search
            // (SingleSolutionLibrary intentionally only implements best).
            auto g0Solution
                = g0Library ? g0Library->findBestSolution(problem, hardware) : nullptr;
            auto o3Solution
                = o3Library ? o3Library->findBestSolution(problem, hardware) : nullptr;

            if(!g0Solution && !o3Solution)
            {
                lastFindTopRetAll = true;
                return result;
            }

            auto forced = forceArm();
            std::shared_ptr<MySolution> selected;
            double g0Score = -INFINITY;
            double o3Score = -INFINITY;
            if(forced == "G0")
                selected = g0Solution ? g0Solution : o3Solution;
            else if(forced == "O3")
                selected = o3Solution ? o3Solution : g0Solution;
            else if(!g0Solution)
                selected = o3Solution;
            else if(!o3Solution)
                selected = g0Solution;
            else
            {
                g0Score = score(problem, *g0Solution);
                o3Score = score(problem, *o3Solution);
                selected = o3Score > g0Score ? o3Solution : g0Solution;
            }

            if(Debug::Instance().printPropertyEvaluation())
            {
                std::cout << "FixedLinearArbiter: G0 index="
                          << (g0Solution ? g0Solution->index : -1) << " score=" << g0Score
                          << ", O3 index=" << (o3Solution ? o3Solution->index : -1)
                          << " score=" << o3Score << ", chose="
                          << (selected == o3Solution ? "O3" : "G0") << std::endl;
            }
            selected->tag = MySolution::MatchingTag::FixedLinearArbiter;
            result.push_back(selected);
            lastFindTopRetAll = result.size() < static_cast<size_t>(numSolutions);
            return result;
        }

        SolutionSet<MySolution> findAllSolutions(
            MyProblem const& problem,
            Hardware const& hardware,
            SolutionLibrarySearchType searchType = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> result;
            if(g0Library)
            {
                auto values = g0Library->findAllSolutions(problem, hardware, searchType);
                result.insert(values.begin(), values.end());
            }
            if(o3Library)
            {
                auto values = o3Library->findAllSolutions(problem, hardware, searchType);
                result.insert(values.begin(), values.end());
            }
            return result;
        }

        SolutionSet<MySolution> findAllSolutionsGroupedGemm(
            std::vector<MyProblem> const& problems,
            Hardware const& hardware,
            SolutionLibrarySearchType searchType = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> result;
            return result;
        }

        bool lastFindTopAlreadyRetAll() const override
        {
            return lastFindTopRetAll;
        }

        double score(MyProblem const& problem, MySolution const& solution) const
        {
            auto features = featureVector(problem, solution.KernelName());
            if(weights.size() != features.size())
                throw std::runtime_error("FixedLinearArbiter requires exactly 22 weights");
            double value = 0.0;
            for(size_t i = 0; i < features.size(); ++i)
                value += features[i] * weights[i];
            return value;
        }

        std::array<double, 22> featureVector(MyProblem const& problem,
                                             std::string const& kernelName) const
        {
            std::array<double, 15> kernel{};
            size_t offset = 0;
            parse(kernelName, R"(_MT(\d+)x(\d+)x(\d+)_)", kernel, offset, 3);
            parse(kernelName, R"(_MIWT(\d+)_(\d+)_)", kernel, offset, 2);
            parse(kernelName, R"(_WG(\d+)_(\d+)_1_)", kernel, offset, 2);
            parse(kernelName, R"(_PGR(\d+)_)", kernel, offset, 1);
            parse(kernelName, R"(_PLR(\d+)_)", kernel, offset, 1);
            parse(kernelName, R"(_WGM(\d+)_)", kernel, offset, 1);
            parse(kernelName, R"(_GRVWA(\d+)_)", kernel, offset, 1);
            parse(kernelName, R"(_GRVWB(\d+)_)", kernel, offset, 1);
            parse(kernelName, R"(_LPA(\d+)_)", kernel, offset, 1);
            parse(kernelName, R"(_LPB(\d+)_)", kernel, offset, 1);
            parse(kernelName, R"(_SU(\d+)_)", kernel, offset, 1);

            double m = 1.0, n = 1.0, k = 1.0;
            for(size_t i = 0; i < problem.freeIndicesA().size(); ++i)
                m *= problem.freeSizeA(i);
            for(size_t i = 0; i < problem.freeIndicesB().size(); ++i)
                n *= problem.freeSizeB(i);
            for(size_t i = 0; i < problem.boundIndices().size(); ++i)
                k *= problem.boundSize(i);
            double mt0 = std::max(kernel[0], 1.0);
            double mt1 = std::max(kernel[1], 1.0);
            double du  = std::max(kernel[2], 1.0);
            double tilesM = std::ceil(m / mt0);
            double tilesN = std::ceil(n / mt1);
            double grid = tilesM * tilesN;
            double edge = m * n / (tilesM * mt0 * tilesN * mt1);

            std::array<double, 22> features{};
            features[0] = std::log1p(m);
            features[1] = std::log1p(n);
            features[2] = std::log1p(k);
            features[3] = std::log1p(grid);
            features[4] = edge;
            features[5] = std::fmod(k, du) == 0.0 ? 1.0 : 0.0;
            features[6] = std::min(1.0, grid / cuCount);
            for(size_t i = 0; i < kernel.size(); ++i)
                features[7 + i] = std::log1p(kernel[i]);
            return features;
        }

    private:
        static void parse(std::string const& name,
                          char const* pattern,
                          std::array<double, 15>& values,
                          size_t& offset,
                          size_t groups)
        {
            std::smatch match;
            if(std::regex_search(name, match, std::regex(pattern)))
            {
                for(size_t i = 0; i < groups; ++i)
                    values[offset + i] = std::stod(match[i + 1].str());
                offset += groups;
            }
            else
            {
                values[offset++] = 0.0;
            }
        }

        static std::string forceArm()
        {
            auto value = std::getenv("TENSILE_FIXED_LINEAR_FORCE_ARM");
            return value ? std::string(value) : std::string();
        }
    };
} // namespace TensileLite
