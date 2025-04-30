/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "SolutionIterator.hpp"

#include "ResultReporter.hpp"
#include <Tensile/Debug.hpp>

namespace TensileLite
{
    namespace Client
    {
        std::shared_ptr<SolutionIterator> SolutionIterator::Default(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
            std::shared_ptr<Hardware>                                      hardware,
            po::variables_map const&                                       args)
        {
            bool bestSolution     = args["best-solution"].as<bool>();
            int  gridbasedTopSols = Debug::Instance().getGridbasedTopSols();
            bool printWinnerOnly  = args["PrintWinnersOnly"].as<bool>();

            if(bestSolution)
            {
                return std::make_shared<TopSolutionIterator>(
                    library, hardware, gridbasedTopSols, printWinnerOnly);
            }
            else
            {
                int firstSolutionIdx = args["solution-start-idx"].as<int>();
                int numSolutions     = args["num-solutions"].as<int>();

                return std::make_shared<AllSolutionsIterator>(
                    library,
                    hardware,
                    firstSolutionIdx,
                    numSolutions,
                    printWinnerOnly,
                    AllSolutionsIterator::CreateCriteria(library, hardware, args));
            }
        }

        SolutionIterator::SolutionIterator(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
            std::shared_ptr<Hardware>                                      hardware,
            bool                                                           printWinnerOnly)
            : m_library(library)
            , m_hardware(hardware)
            , m_printWinnerOnly(printWinnerOnly)
        {
        }

        void SolutionIterator::preProblem(ContractionProblem* const problem)
        {
            m_problem = problem;
        }

        bool SolutionIterator::checkSolution(ContractionSolution&    solution,
                                             ContractionProblemGemm& problem)
        {
            if(!(*solution.hardwarePredicate)(*m_hardware))
            {
                m_reporter->report(ResultKey::Validation, "WRONG_HARDWARE");
                if(m_reporter->logAtLevel(LogLevel::Verbose))
                {
                    std::ostringstream msg;
                    solution.hardwarePredicate->debugEval(*m_hardware, msg);
                    msg << std::endl;
                    m_reporter->log(LogLevel::Verbose, msg.str());
                }

                return false;
            }

            // Test if the persistent kernel is eligible for the current hw and solution
            problem.checkPersistentKernelEligibility(solution, *m_hardware);
            if(!(*solution.problemPredicate)(problem))
            {
                m_reporter->report(ResultKey::Validation, "DID_NOT_SATISFY_ASSERTS");
                if(m_reporter->logAtLevel(LogLevel::Verbose) && !m_printWinnerOnly)
                {
                    std::ostringstream msg;
                    solution.problemPredicate->debugEval(problem, msg);
                    msg << std::endl;
                    m_reporter->log(LogLevel::Verbose, msg.str());
                }

                return false;
            }

            if(solution.requiredHostWorkspaceSizePerProblem == static_cast<size_t>(-1))
            {
                solution.requiredHostWorkspaceSizePerProblem
                    = solution.requiredHostSizeGroupedGemmSingle(problem, *m_hardware);
            }
            return true;
        }

        bool SolutionIterator::checkSolution(ContractionSolution& solution)
        {
            if(auto problems = dynamic_cast<ContractionProblemGroupedGemm*>(m_problem))
            {
                for(int idx = 0; idx < problems->gemms.size(); idx++)
                {
                    size_t ws      = solution.requiredWorkspaceSizeGroupedGemm(problems->gemms, *m_hardware);
                    auto&  problem = problems->gemms[idx];
                    problem.setWorkspaceSizeGroupedGemm(ws);
                    problem.setGroupedGemmCount(problems->gemms.size());
                    if(!checkSolution(solution, problem))
                        return false;
                }
            }
            else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
            {
                return checkSolution(solution, *problem);
            }
            else
            {
                throw std::runtime_error(
                    "[SolutionIterator] Failed to cast to any ContractionProblem");
            }

            return true;
        }

        bool SolutionIterator::runCurrentSolution()
        {
            auto solution = getSolution();
            return checkSolution(*solution);
        }

        AllSolutionsIterator::RunCriteria AllSolutionsIterator::CreateCriteria(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
            std::shared_ptr<Hardware>                                      hardware,
            po::variables_map const&                                       args)
        {
            RunCriteria criteria;

            double granThresh = args["granularity-threshold"].as<double>();
            if(granThresh > 0.0)
            {
                criteria.push_back([granThresh](ContractionProblemGemm const& problem,
                                                Hardware const&               hardware,
                                                ContractionSolution const&    solution) {
                    auto projPerf = solution.projectedPerformance(problem, hardware);
                    return projPerf.granularities.totalGranularity >= granThresh;
                });
            }
            return criteria;
        }

        AllSolutionsIterator::AllSolutionsIterator(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
            std::shared_ptr<Hardware>                                      hardware,
            int                                                            firstSolutionIdx,
            int                                                            numSolutions,
            bool                                                           printWinnerOnly,
            RunCriteria                                                    runCriteria)
            : SolutionIterator(library, hardware, printWinnerOnly)
            , m_runCriteria(runCriteria)
        {
            m_firstSolutionIdx = firstSolutionIdx;

            if(m_firstSolutionIdx < 0)
                m_firstSolutionIdx = library->solutions.begin()->first;

            if(numSolutions < 0)
            {
                auto iter         = library->solutions.rbegin();
                m_lastSolutionIdx = iter->first;
            }
            else
            {
                m_lastSolutionIdx = m_firstSolutionIdx + numSolutions - 1;
            }

            m_currentSolutionIdx = m_firstSolutionIdx;
        }

        void AllSolutionsIterator::preProblem(ContractionProblem* const problem)
        {
            SolutionIterator::preProblem(problem);

            m_currentSolutionIdx = m_firstSolutionIdx;
        }

        void AllSolutionsIterator::postProblem() {}

        void AllSolutionsIterator::preSolution(ContractionSolution const& solution)
        {
            m_reporter->report(ResultKey::SolutionLibraryIndex, solution.libraryLogicIndex);
            m_reporter->report(ResultKey::SolutionIndex, m_currentSolutionIdx);
            m_reporter->report(ResultKey::SolutionProgress,
                               concatenate(m_currentSolutionIdx, "/", m_lastSolutionIdx));
        }

        void AllSolutionsIterator::postSolution()
        {
            m_currentSolutionIdx++;
        }

        bool AllSolutionsIterator::moreSolutionsInProblem() const
        {
            return m_currentSolutionIdx <= m_lastSolutionIdx;
        }

        std::shared_ptr<ContractionSolution> AllSolutionsIterator::getSolution()
        {
            auto iter = m_library->solutions.find(m_currentSolutionIdx);
            if(iter == m_library->solutions.end())
                return std::shared_ptr<ContractionSolution>();

            return iter->second;
        }

        bool AllSolutionsIterator::runCurrentSolution()
        {
            auto solution = getSolution();

            if(!checkSolution(*solution))
                return false;

            for(auto const& criterion : m_runCriteria)
            {
                if(auto problem = dynamic_cast<ContractionProblemGroupedGemm*>(m_problem))
                {
                    if(!criterion(problem->gemms[0], *m_hardware, *solution))
                        return false;
                }
                else if(auto problem = dynamic_cast<ContractionProblemGemm*>(m_problem))
                {
                    if(!criterion(*problem, *m_hardware, *solution))
                        return false;
                }
                else
                {
                    std::cout << "Failed to cast problem to any ContractionProblem.";
                    return false;
                }
            }
            return true;
        }

        BestSolutionIterator::BestSolutionIterator(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
            std::shared_ptr<Hardware>                                      hardware,
            bool                                                           printWinnerOnly)
            : SolutionIterator(library, hardware, printWinnerOnly)
        {
        }

        void BestSolutionIterator::preProblem(ContractionProblem* const problem)
        {
            SolutionIterator::preProblem(problem);
            if(auto groupedProblem = dynamic_cast<const ContractionProblemGroupedGemm*>(problem))
            {
                m_currentSolution
                    = m_library->findBestSolution(groupedProblem->gemms[0], *m_hardware);
            }
            else if(auto gemmProblem = dynamic_cast<const ContractionProblemGemm*>(problem))
            {
                m_currentSolution = m_library->findBestSolution(*gemmProblem, *m_hardware);
            }
            else
            {
                throw std::runtime_error(
                    "[BestSolutionIterator] Failed to cast to any ContractionProblem");
            }
            if(m_currentSolution == nullptr)
            {
                m_currentSolution = m_library->solutions.find(0)->second;
            }
            m_usedCurrentSolution = false;
        }

        void BestSolutionIterator::postProblem() {}

        void BestSolutionIterator::preSolution(ContractionSolution const& solution)
        {
            m_reporter->report(ResultKey::SolutionLibraryIndex, solution.libraryLogicIndex);
            m_reporter->report(ResultKey::SolutionIndex, 0);
            m_reporter->report(ResultKey::SolutionProgress, "1/1");
        }

        void BestSolutionIterator::postSolution()
        {
            m_usedCurrentSolution = true;
        }

        bool BestSolutionIterator::moreSolutionsInProblem() const
        {
            return !m_usedCurrentSolution;
        }

        std::shared_ptr<ContractionSolution> BestSolutionIterator::getSolution()
        {
            return m_currentSolution;
        }

        TopSolutionIterator::TopSolutionIterator(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
            std::shared_ptr<Hardware>                                      hardware,
            int                                                            numSolutions,
            bool                                                           printWinnerOnly)
            : SolutionIterator(library, hardware, printWinnerOnly)
            , m_numSolutions(numSolutions)
        {
        }

        void TopSolutionIterator::preProblem(ContractionProblem* const problem)
        {
            SolutionIterator::preProblem(problem);
            if(auto groupedProblem = dynamic_cast<const ContractionProblemGroupedGemm*>(problem))
            {
                m_solutions = m_library->findTopSolutionsGroupedGemm(
                    groupedProblem->gemms, *m_hardware, m_numSolutions);
            }
            else if(auto gemmProblem = dynamic_cast<const ContractionProblemGemm*>(problem))
            {
                m_solutions
                    = m_library->findTopSolutions(*gemmProblem, *m_hardware, m_numSolutions);
            }
            else
            {
                throw std::runtime_error(
                    "[TopSolutionIterator] Failed to cast to any ContractionProblem");
            }
            if(m_solutions.size() == 0)
            {
                m_solutions.push_back(m_library->solutions.find(0)->second);
            }
            m_currentSolutionIdx = 0;
        }

        void TopSolutionIterator::postProblem() {}

        void TopSolutionIterator::preSolution(ContractionSolution const& solution)
        {
            m_reporter->report(ResultKey::SolutionLibraryIndex, solution.libraryLogicIndex);
            m_reporter->report(ResultKey::SolutionIndex, m_currentSolutionIdx);
            m_reporter->report(ResultKey::SolutionProgress,
                               concatenate(m_currentSolutionIdx, "/", m_solutions.size()));
        }

        void TopSolutionIterator::postSolution()
        {
            m_currentSolutionIdx++;
        }

        bool TopSolutionIterator::moreSolutionsInProblem() const
        {
            return m_currentSolutionIdx < m_solutions.size();
        }

        std::shared_ptr<ContractionSolution> TopSolutionIterator::getSolution()
        {
            return m_solutions[m_currentSolutionIdx];
        }
    } // namespace Client
} // namespace TensileLite
