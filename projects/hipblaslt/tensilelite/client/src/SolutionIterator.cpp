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
#include "TimingInstrumentation.hpp"
#include <Tensile/Debug.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/UtilsOrigami.hpp>

#include <origami/simulator/tensilelite/formocast_simulator.hpp>

namespace TensileLite
{
    namespace Client
    {
        std::shared_ptr<SolutionIterator> SolutionIterator::Default(
            std::shared_ptr<MasterSolutionLibrary<ContractionProblemGemm>> library,
            std::shared_ptr<Hardware>                                      hardware,
            po::variables_map const&                                       args)
        {
            bool   bestSolution        = args["best-solution"].as<bool>();
            int    gridbasedTopSols    = Debug::Instance().getGridbasedTopSols();
            bool   printWinnerOnly     = args["PrintWinnersOnly"].as<bool>();
            double predictionThreshold = args["prediction-threshold"].as<double>();

            if(bestSolution)
            {
                return std::make_shared<TopSolutionIterator>(
                    library, hardware, predictionThreshold, gridbasedTopSols, printWinnerOnly);
            }
            else
            {
                int firstSolutionIdx = args["solution-start-idx"].as<int>();
                int numSolutions     = args["num-solutions"].as<int>();

                return std::make_shared<AllSolutionsIterator>(
                    library,
                    hardware,
                    predictionThreshold,
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
                                             ContractionProblemGemm& problem,
                                             bool                    isReportValid)
        {
            if(!(*solution.hardwarePredicate)(*m_hardware))
            {
                if(isReportValid)
                    m_reporter->report(ResultKey::Validation, "WRONG_HARDWARE");
                if(m_reporter->logAtLevel(LogLevel::Verbose))
                {
                    std::ostringstream msg;
                    solution.hardwarePredicate->debugEval(*m_hardware, msg);
                    msg << std::endl;
                    if(isReportValid)
                        m_reporter->log(LogLevel::Verbose, msg.str());
                }

                return false;
            }

            // Test if the persistent kernel is eligible for the current hw and solution
            problem.checkPersistentKernelEligibility(solution, *m_hardware);
            Task task(*m_hardware, problem, solution);
            if(!(*solution.problemPredicate)(problem) || !(*solution.taskPredicate)(task))
            {
                if(isReportValid)
                    m_reporter->report(ResultKey::Validation, "DID_NOT_SATISFY_ASSERTS");
                if(m_reporter->logAtLevel(LogLevel::Verbose) && !m_printWinnerOnly)
                {
                    std::ostringstream msg;
                    solution.problemPredicate->debugEval(problem, msg);
                    msg << std::endl;
                    solution.taskPredicate->debugEval(task, msg);
                    msg << std::endl;
                    if(isReportValid)
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

        static origami::Formocast::ProblemInfo getProblemInfo(ContractionSolution&    solution,
                                                           ContractionProblemGemm& problem)
        {
            origami::Formocast::ProblemInfo problemInfo;
            problemInfo.M = solution.calculateDimensionM(problem);
            problemInfo.N = solution.calculateDimensionN(problem);
            problemInfo.NumBatches = solution.calculateNumBatches(problem);

            problemInfo.K = problem.boundSize(0);
            problemInfo.transA = problem.transA();
            problemInfo.transB = problem.transB();
            problemInfo.bpeA = problem.a().elementBytes();
            problemInfo.bpeB = problem.b().elementBytes();
            problemInfo.bpeD = problem.d().elementBytes();
            problemInfo.bpeCompute = problem.computeTypeElementSize();

            problemInfo.swizzleTensorA = problem.swizzleTensorA();
            problemInfo.swizzleTensorB = problem.swizzleTensorB();

            problemInfo.dataType = datatypeToAnalyticalDatatype(problem.computeInputTypeA());
            return problemInfo;
        }

        static bool isPredictionAvailable(Hardware const& hardware)
        {
            auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            return hipAMDGPU && hipAMDGPU->analyticalHardware;
        }

        static origami::hardware_t::architecture_t getHardware(Hardware const& hardware)
        {
            hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            if(!hipAMDGPU || !hipAMDGPU->analyticalHardware)
                throw std::runtime_error(
                    "[SolutionIterator] analyticalHardware is not available for this GPU");
            return hipAMDGPU->analyticalHardware->arch;
        }

        static origami::Formocast::SizeMapping getSizeMapping(ContractionSolution&    solution,
                                                           ContractionProblemGemm& problem,
                                                           Hardware const&         hardware)
        {
            auto sizeMapping = solution.getSizeMapping();
            origami::Formocast::SizeMapping sm;
            
            sm.waveNum = sizeMapping.waveNum;

            sm.macroTile[0] = sizeMapping.macroTile.x;
            sm.macroTile[1] = sizeMapping.macroTile.y;
            sm.matrixInstruction = sizeMapping.matrixInstruction;

            sm.grvwA = sizeMapping.grvwA;
            sm.grvwB = sizeMapping.grvwB;
            sm.gwvwC = sizeMapping.gwvwC;
            sm.gwvwD = sizeMapping.gwvwD;

            sm.depthU             = sizeMapping.depthU;
            sm.globalSplitU       = solution.calculateAutoGSU(problem, &hardware);

            sm.workGroupMapping   = sizeMapping.workGroupMapping;
            sm.globalAccumulation = sizeMapping.globalAccumulation;

            sm.workGroupMappingXCC                    = sizeMapping.workGroupMappingXCC;
            sm.workGroupMappingXCCGroup               = sizeMapping.workGroupMappingXCCGroup;
            sm.globalSplitUCoalesced                  = sizeMapping.globalSplitUCoalesced;
            sm.globalSplitUWorkGroupMappingRoundRobin = sizeMapping.globalSplitUWorkGroupMappingRoundRobin;

            sm.CUOccupancy            = sizeMapping.CUOccupancy;
            sm.PrefetchGlobalRead     = sizeMapping.PrefetchGlobalRead;
            sm.MathClocksUnrolledLoop = sizeMapping.MathClocksUnrolledLoop;

            sm.DirectToVgprA      = sizeMapping.DirectToVgprA;
            sm.DirectToVgprB      = sizeMapping.DirectToVgprB;
            sm.NumLoadsCoalescedA = sizeMapping.NumLoadsCoalescedA;
            sm.NumLoadsCoalescedB = sizeMapping.NumLoadsCoalescedB;
            sm.VectorWidthA       = sizeMapping.VectorWidthA;
            sm.VectorWidthB       = sizeMapping.VectorWidthB;
            sm.LocalSplitU        = sizeMapping.LocalSplitU;
            sm.DirectToLdsA       = sizeMapping.DirectToLdsA;
            sm.DirectToLdsB       = sizeMapping.DirectToLdsB;

            sm.waveGroup = sizeMapping.waveGroup;
            return sm;
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
            double                                                         predictionThreshold,
            int                                                            firstSolutionIdx,
            int                                                            numSolutions,
            bool                                                           printWinnerOnly,
            RunCriteria                                                    runCriteria)
            : SolutionIterator(library, hardware, printWinnerOnly)
            , m_runCriteria(runCriteria)
            , m_predictionThreshold(predictionThreshold)
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
            if (m_predictionThreshold > 1.0 || !isPredictionAvailable(*m_hardware))
            {
                m_currentSolutionIdx = m_firstSolutionIdx;
            }
            else
            {
                std::vector<std::pair<int,double>>   performance;
                origami::Formocast formocast;
                for (int i = m_firstSolutionIdx; i <= m_lastSolutionIdx; i++)
                {
                    auto iter = m_library->solutions.find(i);
                    if(iter != m_library->solutions.end())
                    {
                        auto solution = iter->second;
                        if(auto gemmProblem = dynamic_cast<ContractionProblemGemm*>(problem))
                        {
                            if(!checkSolution(*solution, *gemmProblem, false))
                                continue;
                            origami::Formocast::PredictedPerformance predPerf;
                            origami::Formocast::ProblemInfo problemInfo = getProblemInfo(*solution, *gemmProblem);
                            origami::Formocast::SizeMapping sizeMapping = getSizeMapping(*solution, *gemmProblem, *m_hardware);
                            auto hwInfo = getHardware(*m_hardware);
                            formocast.setProblem(problemInfo);
                            formocast.setSolution(sizeMapping);
                            formocast.setHardware(hwInfo);
                            predPerf = formocast.predictedPerformance();
                            performance.push_back(std::pair(i,predPerf.microSeconds));
                            m_hitrate[i] = predPerf.hitRate;
                        }
                    }
                }

                auto comp = [](const std::pair<int, double>& e1, const std::pair<int, double>& e2) { return e1.second < e2.second; };
                std::stable_sort(performance.begin(),performance.end(),comp);
                // TODO: This is the simple threshold method.
                // May use the best perf * 1.x as threshold in the future.
                size_t index    = std::min(performance.size() - 1, size_t(performance.size() * m_predictionThreshold));
                auto threshhold = performance[index].second;

                // push content
                if(!m_qSolutionIdx.empty())
                {
                    throw std::runtime_error(
                        "[AllSolutionsIterator::preProblem] Solution queue is not empty");
                }

                for (int i=0; i<performance.size(); i++)
                {
                    if(m_predictionThreshold == 0.0)
                    {   
                        auto bestIdx = 0;
                        m_qSolutionIdx.push(performance[bestIdx]);
                        break;
                    }
                    else if(performance[i].second <= threshhold)
                    {
                        m_qSolutionIdx.push(performance[i]);
                    }
                    else
                    {
                        break;
                    }
                }
                m_currentSolutionIdx = m_qSolutionIdx.front().first;
                m_currentPrediction  = m_qSolutionIdx.front().second;
                m_currentIdx = 0;

                std::cout<<"predict performance is "<<performance[0].second<<" us, idx = "<<performance[0].first<<std::endl;
            }
        }

        void AllSolutionsIterator::postProblem() {}

        void AllSolutionsIterator::preSolution(ContractionSolution* const solution)
        {
            m_reporter->report(ResultKey::SolutionLibraryIndex, solution->libraryLogicIndex);
            m_reporter->report(ResultKey::SolutionIndex, m_currentSolutionIdx);
            if (m_predictionThreshold > 1.0)
            {
                m_reporter->report(ResultKey::SolutionProgress,
                     concatenate(m_currentSolutionIdx, "/", m_lastSolutionIdx));
                
            }
            else
            {
                m_reporter->report(ResultKey::SolutionProgress,
                    concatenate("hitrate,",m_hitrate[m_currentSolutionIdx],",",m_currentSolutionIdx,"->",m_currentPrediction," us, ",m_currentIdx,", ",m_currentSolutionIdx,"/",m_lastSolutionIdx));
            }
        }

        void AllSolutionsIterator::postSolution()
        {
            ScopedTimer timer("post_solution_sol_advance");
            if (m_predictionThreshold > 1.0)
            {
                m_currentSolutionIdx++;
            }
            else
            {
                m_currentIdx++;
                m_qSolutionIdx.pop();
                if(!m_qSolutionIdx.empty())
                {
                    m_currentSolutionIdx = m_qSolutionIdx.front().first;
                    m_currentPrediction  = m_qSolutionIdx.front().second;
                }
            }
        }

        bool AllSolutionsIterator::moreSolutionsInProblem() const
        {
            if (m_predictionThreshold > 1.0)
                return m_currentSolutionIdx <= m_lastSolutionIdx;
            else
                return !m_qSolutionIdx.empty();
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

        void BestSolutionIterator::preSolution(ContractionSolution* const solution)
        {
            m_reporter->report(ResultKey::SolutionLibraryIndex, solution->libraryLogicIndex);
            m_reporter->report(ResultKey::SolutionIndex, 0);
            m_reporter->report(ResultKey::SolutionProgress, "1/1");
        }

        void BestSolutionIterator::postSolution()
        {
            ScopedTimer timer("post_solution_sol_advance");
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
            double                                                         predictionThreshold,
            int                                                            numSolutions,
            bool                                                           printWinnerOnly)
            : SolutionIterator(library, hardware, printWinnerOnly)
            , m_numSolutions(numSolutions)
            , m_predictionThreshold(predictionThreshold)
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
                if(m_predictionThreshold > 1.0)
                {
                    m_solutions
                        = m_library->findTopSolutions(*gemmProblem, *m_hardware, m_numSolutions);
                }
                else
                {
                    if(m_numSolutions != -1)
                    {
                        m_solutions
                            = m_library->findTopSolutions(*gemmProblem, *m_hardware, m_numSolutions);
                    }
                }
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

            if(m_predictionThreshold > 1.0 || !isPredictionAvailable(*m_hardware))
            {
                m_currentSolutionIdx = 0;
            }
            else
            {
                origami::Formocast formocast;
                std::vector<std::pair<int,double>> performance;
                for (int i = 0; i < m_solutions.size(); i++)
                {
                    if(auto gemmProblem = dynamic_cast<ContractionProblemGemm*>(problem))
                    {
                        if(!checkSolution(*m_solutions[i], *gemmProblem, false))
                            continue;
                        origami::Formocast::PredictedPerformance predPerf;
                        origami::Formocast::ProblemInfo problemInfo = getProblemInfo(*m_solutions[i], *gemmProblem);
                        origami::Formocast::SizeMapping sizeMapping = getSizeMapping(*m_solutions[i], *gemmProblem, *m_hardware);
                        auto hwInfo = getHardware(*m_hardware);
                        formocast.setProblem(problemInfo);
                        formocast.setSolution(sizeMapping);
                        formocast.setHardware(hwInfo);
                        predPerf = formocast.predictedPerformance();
                        performance.push_back(std::pair(i,predPerf.microSeconds));
                        m_hitrate[i] = predPerf.hitRate;
                    }
                }

                auto comp = [](const std::pair<int, double>& e1, const std::pair<int, double>& e2) { return e1.second < e2.second; };
                std::stable_sort(performance.begin(),performance.end(),comp);
                size_t index    = std::min(performance.size() - 1, size_t(performance.size() * m_predictionThreshold));
                auto threshhold = performance[index].second;
                // push content
                if(!m_qSolutionIdx.empty())
                {
                    throw std::runtime_error(
                        "[TopSolutionIterator::preProblem] Solution queue is not empty");
                }

                for (int i=0; i<performance.size(); i++)
                {
                    if(performance[i].second <= threshhold)
                    {
                        m_qSolutionIdx.push(performance[i]);
                        break;
                    }
                }
                m_currentSolutionIdx = m_qSolutionIdx.front().first;
                m_currentPrediction  = m_qSolutionIdx.front().second;
            }
        }

        void TopSolutionIterator::postProblem() {}

        void TopSolutionIterator::preSolution(ContractionSolution* const solution)
        {
            m_reporter->report(ResultKey::SolutionLibraryIndex, solution->libraryLogicIndex);
            m_reporter->report(ResultKey::SolutionIndex, m_currentSolutionIdx);
            if(m_predictionThreshold > 1.0)
                m_reporter->report(ResultKey::SolutionProgress,
                               concatenate(m_currentSolutionIdx, "/", m_solutions.size()));
            else    
                m_reporter->report(ResultKey::SolutionProgress,
                               concatenate("hitrate,",m_hitrate[m_currentSolutionIdx],",",m_currentSolutionIdx,"->",m_currentPrediction," us, ",m_currentSolutionIdx,"/",m_solutions.size()));               
        }

        void TopSolutionIterator::postSolution()
        {
            ScopedTimer timer("post_solution_sol_advance");
            if(m_predictionThreshold > 1.0)
            {
                m_currentSolutionIdx++;
            }
            else
            {
                m_qSolutionIdx.pop();
                if(!m_qSolutionIdx.empty())
                {
                    m_currentSolutionIdx = m_qSolutionIdx.front().first;
                    m_currentPrediction  = m_qSolutionIdx.front().second;
                }
            }

        }

        bool TopSolutionIterator::moreSolutionsInProblem() const
        {
            if(m_predictionThreshold > 1.0)
                return m_currentSolutionIdx < m_solutions.size();
            else
                return !m_qSolutionIdx.empty();
        }

        std::shared_ptr<ContractionSolution> TopSolutionIterator::getSolution()
        {
            return m_solutions[m_currentSolutionIdx];
        }
    } // namespace Client
} // namespace TensileLite
