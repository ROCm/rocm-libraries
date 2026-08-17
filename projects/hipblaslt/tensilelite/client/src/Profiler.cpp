// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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
            auto& rocprof = RocProfiler::getInstance();
            rocprof.initialize(deviceIdx, this);
            rocprof.start();
        }

        Profiler::~Profiler()
        {
            TensileLite::Client::RocProfiler::getInstance().stop();
        }

        void Profiler::preSolution(ContractionSolution* const solution)
        {
            m_currentSolutionIdx = solution->index;
            auto& rocprof = RocProfiler::getInstance();
            m_currentKernelId = rocprof.getKernelId(solution->kernelName);
            m_currentDone = false;
        }

        void Profiler::postSolution()
        {
            if (m_currentDone)
            {
                auto counters = RocProfiler::getInstance().fetch(m_currentSolutionIdx);
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
                RocProfiler::getInstance().enable();
        }

        void Profiler::postProfiler()
        {
            RocProfiler::getInstance().disable();
            m_currentDone = true;
        }

        void Profiler::postProblem()
        {
            m_solutionIdx2DispatchId.clear();
            m_dispatchId2ProfileInfo.clear();
        }
    }
}
