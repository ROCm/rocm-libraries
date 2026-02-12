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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "Profiler.hpp"

#include "ResultReporter.hpp"

namespace TensileLite
{
    namespace Client
    {
        std::shared_ptr<Profiler> Profiler::Default(po::variables_map const& args)
        {
            int deviceIdx = args["device-idx"].as<int>();
            std::vector<std::string> counters = args["rocprof-counter"].as<std::vector<std::string>>();
            return std::make_shared<Profiler>(deviceIdx, counters);
        }

        Profiler::Profiler(int deviceIdx, std::vector<std::string> counters)
        {
            m_counterNames = std::set<std::string>(counters.begin(), counters.end());
            auto& rocprof = TensileLite::Client::RocProfiler::getInstance();
            rocprof.initialize(deviceIdx, counters, this);
            rocprof.start();
        }

        Profiler::~Profiler()
        {
            TensileLite::Client::RocProfiler::getInstance().stop();
        }

        void Profiler::preSolution(ContractionSolution* const solution)
        {
            m_currentSolutionIdx = solution->index;
            m_currentDone = false;
        }

        void Profiler::postSolution()
        {
            if (m_currentDone)
            {
                auto counters = TensileLite::Client::RocProfiler::getInstance().fetch(m_currentSolutionIdx);
                m_reporter->report(ResultKey::RocProfCounter, counters);
            }
            else
            {
                m_reporter->report(ResultKey::RocProfCounter, "");
            }
        }

        void Profiler::preProfiler()
        {
          if (!m_currentDone)
              TensileLite::Client::RocProfiler::getInstance().enable();
        }

        void Profiler::postProfiler()
        {
            TensileLite::Client::RocProfiler::getInstance().disable();
            m_currentDone = true;
        }

        void Profiler::postProblem()
        {
            m_solutionIdx2DispatchId.clear();
            m_dispatchId2ProfileInfo.clear();
        }
    }
}
