/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <Tensile/MasterSolutionLibrary.hpp>

#include <boost/program_options.hpp>

#include <functional>
#include <vector>

#include "RunListener.hpp"

namespace Tensile
{
    namespace Client
    {
        namespace po = boost::program_options;

        /**
 * Not an iterator by the traditional definition but I can't think of a better
 * name
 */
        class SolutionIterator : public RunListener
        {
        public:
            static std::shared_ptr<SolutionIterator>
                Default(std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
                        std::shared_ptr<Hardware>                                  hardware,
                        po::variables_map const&                                   args);

            virtual bool needMoreBenchmarkRuns() const override
            {
                return false;
            }
            virtual void preBenchmarkRun() override {}
            virtual void postBenchmarkRun() override {}

            virtual bool needMoreRunsInSolution() const override
            {
                return false;
            }

            virtual size_t numWarmupRuns() override
            {
                return 0;
            }
            virtual void setNumWarmupRuns(size_t count) override {}
            virtual void preWarmup() override {}
            virtual void postWarmup() override {}
            virtual void validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                         TimingEvents const&                startEvents,
                                         TimingEvents const&                stopEvents) override
            {
            }

            virtual size_t numSyncs() override
            {
                return 0;
            }
            virtual void setNumSyncs(size_t count) override {}
            virtual void preSyncs() override {}
            virtual void postSyncs() override {}

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            }
            virtual void setNumEnqueuesPerSync(size_t count) override {}
            virtual void preEnqueues() override {}
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents) override
            {
            }
            virtual void validateEnqueues(std::shared_ptr<ContractionInputs> inputs,
                                          TimingEvents const&                startEvents,
                                          TimingEvents const&                stopEvents) override
            {
            }

            virtual void finalizeReport() override {}

            virtual int error() const override
            {
                return 0;
            }

            virtual bool                                 moreSolutionsInProblem() const = 0;
            virtual std::shared_ptr<ContractionSolution> getSolution()                  = 0;
            virtual bool                                 runCurrentSolution();

            virtual void preProblem(ContractionProblem const& problem) override;

        protected:
            SolutionIterator(std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
                             std::shared_ptr<Hardware>                                  hardware);

            virtual bool checkSolution(ContractionSolution const& solution);

            std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> m_library;
            std::shared_ptr<Hardware>                                  m_hardware;
            ContractionProblem                                         m_problem;
        };

        class AllSolutionsIterator : public SolutionIterator
        {
        public:
            using RunCriteria = std::vector<std::function<bool(
                ContractionProblem const&, Hardware const&, ContractionSolution const&)>>;

            static RunCriteria
                CreateCriteria(std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
                               std::shared_ptr<Hardware>                                  hardware,
                               po::variables_map const&                                   args);

            AllSolutionsIterator(std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
                                 std::shared_ptr<Hardware> hardware,
                                 int                       firstSolutionIdx,
                                 int                       numSolutions,
                                 RunCriteria               runCriteria = RunCriteria());

            virtual void preProblem(ContractionProblem const& problem) override;
            virtual void postProblem() override;

            virtual void preSolution(ContractionSolution const& solution) override;
            virtual void postSolution() override;

            virtual bool                                 moreSolutionsInProblem() const override;
            virtual std::shared_ptr<ContractionSolution> getSolution() override;
            virtual bool                                 runCurrentSolution() override;

        private:
            int m_firstSolutionIdx;
            int m_lastSolutionIdx;

            int m_currentSolutionIdx;

            RunCriteria m_runCriteria;
        };

        class BestSolutionIterator : public SolutionIterator
        {
        public:
            BestSolutionIterator(std::shared_ptr<MasterSolutionLibrary<ContractionProblem>> library,
                                 std::shared_ptr<Hardware> hardware);

            virtual void preProblem(ContractionProblem const& problem) override;
            virtual void postProblem() override;

            virtual void preSolution(ContractionSolution const& solution) override;
            virtual void postSolution() override;

            virtual bool                                 moreSolutionsInProblem() const override;
            virtual std::shared_ptr<ContractionSolution> getSolution() override;

        private:
            std::shared_ptr<ContractionSolution> m_currentSolution;
            bool                                 m_usedCurrentSolution = false;
        };
    } // namespace Client
} // namespace Tensile
