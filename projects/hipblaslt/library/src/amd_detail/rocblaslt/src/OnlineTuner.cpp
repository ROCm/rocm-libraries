// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OnlineTuner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace rocblaslt
{
    namespace
    {
        constexpr int c_defaultRepeats = 3;

        // Slack over the candidates * repeats launches exploration needs. A
        // caller is free to ignore the rotation selectCandidate() asks for, so
        // without a ceiling a problem could stay in the exploring state, and on
        // the write lock, for the life of the process.
        constexpr int c_visitBudgetFactor = 8;

        constexpr const char* c_tracePrefix = "[hipblaslt-online-tune]";

        int envInt(const char* name, int defaultValue)
        {
            const char* env = std::getenv(name);
            if(!env)
                return defaultValue;

            return static_cast<int>(std::strtol(env, nullptr, 0));
        }

        int positionOf(const std::vector<int>& rankedSolutionIndices, int solutionIndex)
        {
            if(solutionIndex < 0)
                return -1;

            for(size_t i = 0; i < rankedSolutionIndices.size(); ++i)
                if(rankedSolutionIndices[i] == solutionIndex)
                    return static_cast<int>(i);

            return -1;
        }

        float median(std::vector<float> samples)
        {
            std::sort(samples.begin(), samples.end());

            const size_t count = samples.size();
            if(count % 2)
                return samples[count / 2];

            return 0.5f * (samples[count / 2 - 1] + samples[count / 2]);
        }

        std::ostringstream traceLine(const char* event, size_t problemKey)
        {
            std::ostringstream msg;
            msg << std::fixed << std::setprecision(3);
            msg << c_tracePrefix << " event=" << event << " key=" << problemKey;
            return msg;
        }
    }

    OnlineTuner::OnlineTuner()
    {
        const int topK = envInt("HIPBLASLT_ORIGAMI_ONLINE_TUNE_TOP_K", 0);

        m_topK    = topK > 0 ? topK : 0;
        m_repeats = std::max(envInt("HIPBLASLT_ORIGAMI_ONLINE_TUNE_REPEATS", c_defaultRepeats), 1);
        m_verbose = std::getenv("HIPBLASLT_ORIGAMI_ONLINE_TUNE_VERBOSE") != nullptr;
        m_enabled = m_topK >= 2;
    }

    // The pooled events are deliberately not destroyed. This is a function-local
    // static, so its destructor competes with the HIP runtime's own teardown, and
    // hipEventDestroy() after that teardown faults. The process is exiting, so
    // leaking the handles costs nothing.
    OnlineTuner::~OnlineTuner() {}

    int OnlineTuner::selectCandidateImpl(size_t                  problemKey,
                                         const std::vector<int>& rankedSolutionIndices)
    {
        if(rankedSolutionIndices.empty())
            return -1;

        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_problems.find(problemKey);
            if(iter != m_problems.end() && iter->second.m_resolved)
                return positionOf(rankedSolutionIndices, iter->second.m_winner);
        }

        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

        ProblemState& state = m_problems[problemKey];
        if(state.m_candidates.empty())
            registerProblem(problemKey, state, rankedSolutionIndices);

        if(!state.m_resolved)
        {
            ++state.m_calls;

            const bool sampled = nextCandidate(state) < 0 && state.m_pendingCandidate < 0;
            const bool gaveUp  = state.m_calls > visitBudget(state);

            if(sampled || gaveUp)
                resolve(problemKey, state);
        }

        if(state.m_resolved)
            return positionOf(rankedSolutionIndices, state.m_winner);

        const int candidate = nextCandidate(state);
        if(candidate < 0)
            return -1;

        return positionOf(rankedSolutionIndices, state.m_candidates[candidate]);
    }

    void OnlineTuner::harvestPendingImpl(size_t problemKey)
    {
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_problems.find(problemKey);
            if(iter == m_problems.end() || iter->second.m_pendingCandidate < 0)
                return;
        }

        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

        auto iter = m_problems.find(problemKey);
        if(iter == m_problems.end() || iter->second.m_pendingCandidate < 0)
            return;

        ProblemState& state = iter->second;

        // Queried under the write lock rather than the shared one: another thread
        // could otherwise harvest this pair and start a new measurement in
        // between, and recycling events the GPU is still writing to would corrupt
        // that measurement.
        if(hipEventQuery(state.m_pendingStop) != hipSuccess)
            return;

        const int candidate = state.m_pendingCandidate;

        float      milliseconds = 0.0f;
        const bool timed
            = hipEventElapsedTime(&milliseconds, state.m_pendingStart, state.m_pendingStop)
                  == hipSuccess
              && std::isfinite(milliseconds) && milliseconds > 0.0f;

        releaseEvents(state.m_pendingStart, state.m_pendingStop);
        state.m_pendingStart     = nullptr;
        state.m_pendingStop      = nullptr;
        state.m_pendingCandidate = -1;

        if(timed)
            state.m_samples[candidate].push_back(milliseconds * 1000.0f);

        if(m_verbose)
        {
            std::ostringstream msg = traceLine(timed ? "sample" : "drop", problemKey);
            msg << " cand=" << candidate << " sol=" << state.m_candidates[candidate];
            if(timed)
                msg << " us=" << milliseconds * 1000.0f;
            msg << "\n";
            std::cerr << msg.str();
        }

        if(!state.m_resolved && nextCandidate(state) < 0)
            resolve(problemKey, state);
    }

    bool OnlineTuner::beginMeasurementImpl(size_t      problemKey,
                                           int         solutionIndex,
                                           hipEvent_t& start,
                                           hipEvent_t& stop)
    {
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_problems.find(problemKey);
            if(iter == m_problems.end() || measurableCandidate(iter->second, solutionIndex) < 0)
                return false;
        }

        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

        auto iter = m_problems.find(problemKey);
        if(iter == m_problems.end())
            return false;

        ProblemState& state     = iter->second;
        const int     candidate = measurableCandidate(state, solutionIndex);
        if(candidate < 0)
            return false;

        if(!acquireEvents(start, stop))
            return false;

        state.m_pendingCandidate = candidate;
        state.m_pendingStart     = start;
        state.m_pendingStop      = stop;
        ++state.m_issued[candidate];

        return true;
    }

    void OnlineTuner::registerProblem(size_t                  problemKey,
                                      ProblemState&           state,
                                      const std::vector<int>& rankedSolutionIndices)
    {
        const size_t count
            = std::min(rankedSolutionIndices.size(), static_cast<size_t>(m_topK));

        state.m_candidates.assign(rankedSolutionIndices.begin(),
                                  rankedSolutionIndices.begin() + count);
        state.m_samples.resize(count);
        state.m_issued.assign(count, 0);

        // Nothing to compare against, so the problem is born resolved with no
        // winner and every later call takes the read-only path.
        if(count < 2)
            state.m_resolved = true;

        if(m_verbose)
        {
            std::ostringstream msg = traceLine("register", problemKey);
            msg << " candidates=" << count << " repeats=" << m_repeats << " sols=";
            for(size_t i = 0; i < count; ++i)
                msg << (i ? "," : "") << state.m_candidates[i];
            msg << "\n";
            std::cerr << msg.str();
        }
    }

    int OnlineTuner::nextCandidate(const ProblemState& state) const
    {
        int next = -1;

        for(size_t i = 0; i < state.m_issued.size(); ++i)
        {
            if(state.m_issued[i] >= m_repeats)
                continue;

            if(next < 0 || state.m_issued[i] < state.m_issued[next])
                next = static_cast<int>(i);
        }

        return next;
    }

    int OnlineTuner::measurableCandidate(const ProblemState& state, int solutionIndex) const
    {
        if(state.m_resolved || state.m_pendingCandidate >= 0)
            return -1;

        for(size_t i = 0; i < state.m_candidates.size(); ++i)
            if(state.m_candidates[i] == solutionIndex && state.m_issued[i] < m_repeats)
                return static_cast<int>(i);

        return -1;
    }

    int OnlineTuner::visitBudget(const ProblemState& state) const
    {
        return static_cast<int>(state.m_candidates.size()) * m_repeats * c_visitBudgetFactor;
    }

    void OnlineTuner::resolve(size_t problemKey, ProblemState& state)
    {
        int   winner      = -1;
        int   winnerIndex = -1;
        float winnerScore = 0.0f;

        for(size_t i = 0; i < state.m_samples.size(); ++i)
        {
            if(state.m_samples[i].empty())
                continue;

            const float score = median(state.m_samples[i]);
            if(winnerIndex < 0 || score < winnerScore)
            {
                winnerIndex = static_cast<int>(i);
                winnerScore = score;
                winner      = state.m_candidates[i];
            }
        }

        state.m_winner   = winner;
        state.m_resolved = true;

        if(m_verbose)
        {
            const size_t samples = winnerIndex < 0 ? 0 : state.m_samples[winnerIndex].size();

            std::ostringstream msg = traceLine("winner", problemKey);
            msg << " cand=" << winnerIndex << " sol=" << winner << " us=" << winnerScore
                << " samples=" << samples << " calls=" << state.m_calls << "\n";
            std::cerr << msg.str();
        }
    }

    bool OnlineTuner::acquireEvents(hipEvent_t& start, hipEvent_t& stop)
    {
        hipEvent_t events[2] = {nullptr, nullptr};

        for(int i = 0; i < 2; ++i)
        {
            if(!m_events.empty())
            {
                events[i] = m_events.back();
                m_events.pop_back();
                continue;
            }

            if(hipEventCreateWithFlags(&events[i], hipEventDefault) != hipSuccess)
            {
                releaseEvents(events[0], nullptr);
                return false;
            }
        }

        start = events[0];
        stop  = events[1];

        return true;
    }

    void OnlineTuner::releaseEvents(hipEvent_t start, hipEvent_t stop)
    {
        if(start)
            m_events.push_back(start);

        if(stop)
            m_events.push_back(stop);
    }
} // namespace rocblaslt
