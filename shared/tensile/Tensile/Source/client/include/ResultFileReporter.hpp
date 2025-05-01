/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "CSVStackFile.hpp"
#include "ResultReporter.hpp"

#include <boost/program_options.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        namespace po = boost::program_options;

        class ResultFileReporter : public ResultReporter
        {
        public:
            static std::shared_ptr<ResultFileReporter> Default(po::variables_map const& args);

            ResultFileReporter(std::string const& filename,
                               bool               exportExtraCols,
                               bool               mergeSameProblems,
                               PerformanceMetric  performanceMetric);

            virtual void reportValue_string(std::string const& key,
                                            std::string const& value) override;
            virtual void reportValue_uint(std::string const& key, uint64_t value) override;
            virtual void reportValue_int(std::string const& key, int64_t value) override;
            virtual void reportValue_double(std::string const& key, double value) override;
            virtual void reportValue_sizes(std::string const&         key,
                                           std::vector<size_t> const& value) override;

            virtual void postProblem() override;
            virtual void postSolution() override;

            void finalizeReport() override;

        private:
            template <typename T>
            void reportValue(std::string const& key, T const& value);
            void mergeRow(std::unordered_map<std::string, std::string>& newRow);

            CSVStackFile      m_output;
            std::string       m_solutionName;
            bool              m_invalidSolution = false;
            bool              m_extraCol;
            bool              m_mergeSameProblems;
            PerformanceMetric m_performanceMetric;
            // for extra columns
            std::string m_winnerSolution;
            int64_t     m_currSolutionIdx   = -1;
            int64_t     m_winnerSolutionIdx = -1;
            double      m_fastestGflops     = -1.0;
            double      m_fasterTimeUS      = -1.0;
            // for logging winner GFx clock in CSV file
            uint16_t m_currGfxClock         = 0;
            uint16_t m_winnerGfxClock       = 0;
            uint16_t m_currPower            = 0;
            uint16_t m_winnerPower          = 0;
            uint16_t m_currTemperatureHot   = 0;
            uint16_t m_winnerTemperatureHot = 0;

            // for merge rows
            int64_t                                                         m_currProbID = -1;
            std::map<int64_t, std::unordered_map<std::string, std::string>> m_probMap;
        };
    } // namespace Client
} // namespace Tensile
