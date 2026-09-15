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

            if(state.m_calls > visitBudget(state))
                state.m_gaveUp = true;

            const bool allIssued = nextCandidate(state) < 0;

            if((allIssued || state.m_gaveUp) && state.m_pending.empty())
                resolve(problemKey, state);
        }

        if(state.m_resolved)
            return positionOf(rankedSolutionIndices, state.m_winner);

        // Given up but still draining: promoting a candidate nothing is going to
        // measure would only perturb the caller's own ordering.
        if(state.m_gaveUp)
            return -1;

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
            if(iter == m_problems.end() || iter->second.m_pending.empty())
                return;
        }

        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

        auto iter = m_problems.find(problemKey);
        if(iter == m_problems.end() || iter->second.m_pending.empty())
            return;

        ProblemState& state = iter->second;

        size_t outstanding = 0;

        for(size_t i = 0; i < state.m_pending.size(); ++i)
        {
            const PendingMeasurement pending = state.m_pending[i];

            // Queried under the write lock rather than the shared one: another
            // thread could otherwise harvest this pair and start a new
            // measurement in between, and recycling events the GPU is still
            // writing to would corrupt that measurement.
            //
            // Records the GPU has not finished are compacted to the front, which
            // keeps the launch order the rest of this loop reports in.
            if(hipEventQuery(pending.m_stop) != hipSuccess)
            {
                state.m_pending[outstanding++] = pending;
                continue;
            }

            // A pair the caller never recorded still queries as complete, and the
            // pool hands recycled pairs back swapped, so it reads a negative
            // elapsed time. The sign test is what keeps a missed measurement from
            // looking like an impossibly fast kernel and winning every median.
            float      milliseconds = 0.0f;
            const bool timed
                = hipEventElapsedTime(&milliseconds, pending.m_start, pending.m_stop) == hipSuccess
                  && std::isfinite(milliseconds) && milliseconds > 0.0f;

            releaseEvents(pending.m_start, pending.m_stop);

            if(timed)
                state.m_samples[pending.m_candidate].push_back(milliseconds * 1000.0f);

            if(m_verbose)
            {
                std::ostringstream msg = traceLine(timed ? "sample" : "drop", problemKey);
                msg << " cand=" << pending.m_candidate
                    << " sol=" << state.m_candidates[pending.m_candidate];
                if(timed)
                    msg << " us=" << milliseconds * 1000.0f;
                msg << "\n";
                std::cerr << msg.str();
            }
        }

        state.m_pending.resize(outstanding);

        if(!state.m_resolved && state.m_pending.empty()
           && (state.m_gaveUp || nextCandidate(state) < 0))
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
            if(iter == m_problems.end())
            {
                // A key the selection hook never registered. The two hooks
                // disagreeing is otherwise indistinguishable from the feature
                // being switched off, so it gets its own event.
                if(m_verbose)
                {
                    std::ostringstream msg = traceLine("miss", problemKey);
                    msg << " sol=" << solutionIndex << "\n";
                    std::cerr << msg.str();
                }

                return false;
            }

            if(measurableCandidate(iter->second, solutionIndex) < 0)
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

        // Checked here rather than in measurableCandidate() so the refusal can be
        // counted: exploration stalling behind a full in-flight set is the
        // failure this scheme exists to avoid, and declined= on the winner line
        // is how a recurrence would show up.
        if(static_cast<int>(state.m_pending.size()) >= inFlightCap(state))
        {
            ++state.m_declined;
            return false;
        }

        if(!acquireEvents(start, stop))
            return false;

        state.m_pending.push_back({start, stop, candidate});
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
        else
            state.m_pending.reserve(static_cast<size_t>(inFlightCap(state)));

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
        if(state.m_resolved || state.m_gaveUp)
            return -1;

        for(size_t i = 0; i < state.m_candidates.size(); ++i)
            if(state.m_candidates[i] == solutionIndex && state.m_issued[i] < m_repeats)
                return static_cast<int>(i);

        return -1;
    }

    // The rotation issues at most repeats() launches per candidate, so a whole
    // exploration is also the most that can ever be outstanding at once. Capping
    // at exactly that lets a caller enqueue every sample before the GPU retires
    // any of them, which is what stops exploration advancing at queue-drain rate.
    int OnlineTuner::inFlightCap(const ProblemState& state) const
    {
        return static_cast<int>(state.m_candidates.size()) * m_repeats;
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
                << " samples=" << samples << " calls=" << state.m_calls
                << " declined=" << state.m_declined
                << " gaveup=" << (state.m_gaveUp ? 1 : 0) << "\n";
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
