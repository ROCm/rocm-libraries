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

#include <ResultFileReporter.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        std::shared_ptr<ResultFileReporter>
            ResultFileReporter::Default(po::variables_map const& args)
        {
            return std::make_shared<ResultFileReporter>(
                args["results-file"].as<std::string>(),
                args["csv-export-extra-cols"].as<bool>(),
                args["csv-merge-same-problems"].as<bool>(),
                args["performance-metric"].as<PerformanceMetric>());
        }

        ResultFileReporter::ResultFileReporter(std::string const& filename,
                                               bool               exportExtraCols,
                                               bool               mergeSameProblems,
                                               PerformanceMetric  performanceMetric)
            : m_output(filename)
            , m_extraCol(exportExtraCols)
            , m_mergeSameProblems(mergeSameProblems)
            , m_performanceMetric(performanceMetric)
        {
            if(m_performanceMetric == PerformanceMetric::CUEfficiency)
                m_output.setHeaderForKey(ResultKey::ProblemIndex, "GFlopsPerCU");
            else // Default to 'DeviceEfficiency' benchmarking if CUEfficiency not specified
                m_output.setHeaderForKey(ResultKey::ProblemIndex, "GFlops");
        }

        template <typename T>
        void ResultFileReporter::reportValue(std::string const& key, T const& value)
        {
            std::string valueStr = boost::lexical_cast<std::string>(value);

            if(key == ResultKey::Validation)
            {
                if(valueStr != "PASSED" && valueStr != "NO_CHECK")
                {
                    m_output.setValueForKey(m_solutionName, -1.0);
                    m_invalidSolution = true;
                }
            }
            else if(key == ResultKey::SolutionName)
            {
                m_solutionName = valueStr;
                m_output.setHeaderForKey(valueStr, valueStr);
            }
            else if(key == ResultKey::TimeUS)
            {
                // cascade from BenchmarkTimer, Time-US first
                ++m_currSolutionIdx;
                if(!m_invalidSolution)
                {
                    double timeUS = std::stod(valueStr);
                    if(m_fasterTimeUS < 0 || m_fasterTimeUS > timeUS)
                    {
                        m_fasterTimeUS = timeUS;
                    }
                }
            }
            else if((key == ResultKey::SpeedGFlops
                     && m_performanceMetric == PerformanceMetric::DeviceEfficiency)
                    || (key == ResultKey::SpeedGFlopsPerCu
                        && m_performanceMetric == PerformanceMetric::CUEfficiency))
            {
                // cascade from BenchmarkTimer, SpeedGFlops or SpeedGFlopsPerCU second
                if(!m_invalidSolution)
                {
                    m_output.setValueForKey(m_solutionName, value);

                    double gflops = std::stod(valueStr);
                    if(m_fastestGflops < gflops)
                    {
                        m_winnerSolution       = m_solutionName;
                        m_winnerSolutionIdx    = m_currSolutionIdx;
                        m_fastestGflops        = gflops;
                        m_winnerGfxClock       = m_currGfxClock;
                        m_winnerPower          = m_currPower;
                        m_winnerTemperatureHot = m_currTemperatureHot;
                    }
                }
            }
            else if(key == ResultKey::GfxFrequency)
            {
                m_currGfxClock = static_cast<uint16_t>(std::stoi(valueStr));
            }
            else if(key == ResultKey::Power)
            {
                m_currPower = static_cast<uint16_t>(std::stoi(valueStr));
            }
            else if(key == ResultKey::TemperatureHot)
            {
                m_currTemperatureHot = static_cast<uint16_t>(std::stoi(valueStr));
            }
            else
            {
                m_output.setValueForKey(key, value);
            }
        }

        void ResultFileReporter::reportValue_string(std::string const& key,
                                                    std::string const& value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_uint(std::string const& key, uint64_t value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_int(std::string const& key, int64_t value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_double(std::string const& key, double value)
        {
            reportValue(key, value);
        }

        void ResultFileReporter::reportValue_sizes(std::string const&         key,
                                                   std::vector<size_t> const& value)
        {
            if(key == ResultKey::ProblemSizes)
            {
                for(size_t i = 0; i < value.size(); i++)
                {
                    std::string key = concatenate("Size", static_cast<char>('I' + i));
                    m_output.setHeaderForKey(key, key);
                    m_output.setValueForKey(key, value[i]);
                }

                // Values for these come separately.
                m_output.setHeaderForKey(ResultKey::LDD, "LDD");
                m_output.setHeaderForKey(ResultKey::LDC, "LDC");
                m_output.setHeaderForKey(ResultKey::LDA, "LDA");
                m_output.setHeaderForKey(ResultKey::LDB, "LDB");
                m_output.setHeaderForKey(ResultKey::GfxFrequency, "WinnerFreq");
                if(m_extraCol)
                {
                    m_output.setHeaderForKey(ResultKey::FastestGFlops, "WinnerGFlops");
                    m_output.setHeaderForKey(ResultKey::TimeUS, "WinnerTimeUS");
                    m_output.setHeaderForKey(ResultKey::SolutionWinnerIdx, "WinnerIdx");
                    m_output.setHeaderForKey(ResultKey::SolutionWinner, "WinnerName");
                }
                m_output.setHeaderForKey(ResultKey::TotalFlops, "TotalFlops");
                m_output.setHeaderForKey(ResultKey::Power, "WinnerPower");
                m_output.setHeaderForKey(ResultKey::TemperatureHot, "WinnerTemperature");
            }
        }

        void ResultFileReporter::mergeRow(std::unordered_map<std::string, std::string>& newRow)
        {
            m_currProbID = std::stoull(newRow[ResultKey::ProblemIndex]);
            if(m_probMap.count(m_currProbID) == 0)
            {
                m_probMap[m_currProbID] = newRow;
                return;
            }

            auto& oldRow = m_probMap[m_currProbID];
            for(auto& oldRowIter : oldRow)
            {
                const std::string& key = oldRowIter.first;
                if(key.compare(ResultKey::ProblemIndex) == 0
                   || key.find("Size") != std::string::npos || key.compare(ResultKey::LDD) == 0
                   || key.compare(ResultKey::LDC) == 0 || key.compare(ResultKey::LDA) == 0
                   || key.compare(ResultKey::LDB) == 0 || key.compare(ResultKey::TotalFlops) == 0)
                {
                    // these data should be the same for same problem
                    assert(oldRowIter.second == newRow[key]);
                }
                else if(key.compare(ResultKey::FastestGFlops) == 0)
                {
                    // if new row is better, update, dummy guard for -1 and empty str
                    int64_t oldFastest
                        = (oldRowIter.second.empty()) ? 0 : std::stoll(oldRowIter.second);
                    int64_t newFastest = (newRow[key].empty()) ? 0 : std::stoll(newRow[key]);
                    if(newFastest > oldFastest)
                    {
                        oldRow[ResultKey::FastestGFlops]     = newRow[ResultKey::FastestGFlops];
                        oldRow[ResultKey::TimeUS]            = newRow[ResultKey::TimeUS];
                        oldRow[ResultKey::SolutionWinnerIdx] = newRow[ResultKey::SolutionWinnerIdx];
                        oldRow[ResultKey::SolutionWinner]    = newRow[ResultKey::SolutionWinner];
                        oldRow[ResultKey::GfxFrequency]      = newRow[ResultKey::GfxFrequency];
                        oldRow[ResultKey::Power]             = newRow[ResultKey::Power];
                        oldRow[ResultKey::TemperatureHot]    = newRow[ResultKey::TemperatureHot];
                    }
                }
                else if(key.compare(ResultKey::TimeUS) == 0
                        || key.compare(ResultKey::SolutionWinnerIdx) == 0
                        || key.compare(ResultKey::SolutionWinner) == 0
                        || key.compare(ResultKey::GfxFrequency) == 0
                        || key.compare(ResultKey::Power) == 0
                        || key.compare(ResultKey::TemperatureHot) == 0)
                {
                    // skip, we update these together with FastestGFlops
                    continue;
                }
                else
                {
                    // these are gflops for each solution
                    // if new row is better, update. Dummy guard for -1 and empty str
                    int64_t oldFastest
                        = (oldRowIter.second.empty()) ? 0 : std::stoll(oldRowIter.second);
                    int64_t newFastest = (newRow[key].empty()) ? 0 : std::stoll(newRow[key]);
                    if(newFastest > oldFastest)
                    {
                        oldRow[key] = newRow[key];
                    }
                }
            }
        }

        void ResultFileReporter::postProblem()
        {
            if(m_extraCol)
            {
                // update winner
                m_output.setValueForKey(ResultKey::FastestGFlops, m_fastestGflops);
                m_output.setValueForKey(ResultKey::TimeUS, m_fasterTimeUS);
                m_output.setValueForKey(ResultKey::SolutionWinnerIdx, m_winnerSolutionIdx);
                m_output.setValueForKey(ResultKey::SolutionWinner, m_winnerSolution);
            }
            m_output.setValueForKey(ResultKey::GfxFrequency, m_winnerGfxClock);
            m_output.setValueForKey(ResultKey::Power, m_winnerPower);
            m_output.setValueForKey(ResultKey::TemperatureHot, m_winnerTemperatureHot);

            // reset
            m_winnerSolution       = "";
            m_currSolutionIdx      = -1;
            m_winnerSolutionIdx    = -1;
            m_fastestGflops        = -1.0;
            m_fasterTimeUS         = -1.0;
            m_currGfxClock         = 0;
            m_winnerGfxClock       = 0;
            m_currGfxClock         = 0;
            m_winnerGfxClock       = 0;
            m_currPower            = 0;
            m_winnerPower          = 0;
            m_currTemperatureHot   = 0;
            m_winnerTemperatureHot = 0;

            if(!m_mergeSameProblems)
            {
                m_output.writeCurrentRow();
            }
            else
            {
                std::unordered_map<std::string, std::string> curRow;
                m_output.readCurrentRow(curRow);
                m_output.clearCurrentRow();
                // for (auto & field : curRow )
                //     std::cout << "key:" << field.first << ", value:" << field.second << std::endl;
                this->mergeRow(curRow);
            }
        }

        void ResultFileReporter::postSolution()
        {
            m_solutionName    = "";
            m_invalidSolution = false;
        }

        void ResultFileReporter::finalizeReport()
        {
            if(m_mergeSameProblems)
            {
                for(auto& probIter : m_probMap)
                {
                    auto& single_row = probIter.second;
                    for(auto& field : single_row)
                    {
                        m_output.setValueForKey(field.first, field.second);
                    }
                    m_output.writeCurrentRow();
                }
            }
        }
    } // namespace Client
} // namespace Tensile
