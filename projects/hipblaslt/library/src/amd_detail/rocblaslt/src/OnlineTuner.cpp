// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OnlineTuner.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace rocblaslt
{
    namespace
    {
        constexpr const char* c_tracePrefix = "[hipblaslt-online-tune]";

        // The candidate selectCandidate() last offered this thread, which only
        // the next launch on this thread may be timed for. Per thread, because
        // selection and launch are the same hipblasLtMatmul call on one thread,
        // so another problem's launch elsewhere cannot claim it. m_state names
        // the exploration; m_problems never erases, so the address is stable.
        struct ThreadOffer
        {
            const void* m_state     = nullptr;
            int         m_candidate = -1;
        };

        thread_local ThreadOffer t_offer;

        int positionOf(const std::vector<int>& rankedSolutionIndices, int solutionIndex)
        {
            if(solutionIndex < 0)
                return -1;

            for(size_t i = 0; i < rankedSolutionIndices.size(); ++i)
                if(rankedSolutionIndices[i] == solutionIndex)
                    return static_cast<int>(i);

            return -1;
        }

        std::ostringstream traceLine(const char* event, const TensileLite::ProblemOverride& key)
        {
            std::ostringstream msg;
            msg << std::fixed << std::setprecision(3);
            msg << c_tracePrefix << " event=" << event
                << " key=" << std::hash<TensileLite::ProblemOverride>{}(key);
            return msg;
        }
    }

    void OnlineTuner::applyConfig()
    {
        const OnlineTunerConfig config = onlineTunerConfig();

        m_topK        = std::clamp(config.topK, 0, c_maxTopK);
        m_repeats     = std::clamp(config.repeats, 1, c_maxRepeats);
        m_coldCalls   = std::max(config.coldCalls, 0);
        m_livePairCap = std::max(config.livePairCap, 1);

        m_minImprovementPct = std::clamp(config.minImprovementPct, 0, 100);
        m_trace       = config.trace;

        m_livePairCapReached = config.livePairCapReached;

        m_enabled = config.enabled && m_topK >= 2;
    }

    OnlineTuner::OnlineTuner()
    {
        applyConfig();
    }

    void OnlineTuner::reloadForTest()
    {
        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        for(auto& [key, state] : m_problems)
            for(const PendingMeasurement& pending : state.m_pending)
                retireEvents(pending.m_start, pending.m_stop, pending.m_device);

        for(auto& [device, pool] : m_pairs)
            for(const EventPair& pair : pool)
                retireEvents(pair.m_start, pair.m_stop, device);

        m_problems.clear();
        m_pairs.clear();
        m_holders.clear();
        m_livePairs.clear();
        m_capReported.clear();
        t_offer = {};

        applyConfig();
    }

    int OnlineTuner::selectCandidate(const Key&              key,
                                     size_t                  problemHash,
                                     size_t                  workspaceBytes,
                                     const std::vector<int>& rankedSolutionIndices,
                                     bool                    mayProbe,
                                     Record*                 record)
    {
        if(rankedSolutionIndices.empty())
            return -1;

        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_problems.find(key);
            if(iter != m_problems.end())
            {
                ProblemState& state = iter->second;

                if(state.m_resolved)
                    return servedPosition(state, rankedSolutionIndices, record);

                // Another problem under a key being explored is turned away
                // without the write lock, as the exclusive path below would.
                if(!state.m_candidates.empty() && state.m_problemHash != problemHash
                   && state.m_coldCalls >= m_coldCalls)
                {
                    if(mayProbe)
                        t_offer.m_candidate = -1;
                    return -1;
                }
            }
        }

        // A captured graph never comes back through selection, so a probe would
        // never be measured; it does not spend the budget.
        if(!mayProbe)
            return -1;

        // Whatever this thread was offered before is stale.
        t_offer.m_candidate = -1;

        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        ProblemState& state = m_problems[key];

        // Guards above the dispatch-counter reset turn away visits that are not
        // this exploration's; those below it, visits that do not advance it.
        const bool cold = state.m_candidates.empty() && state.m_coldCalls < m_coldCalls;

        if(cold)
            ++state.m_coldCalls;
        else if(state.m_candidates.empty())
            registerProblem(key, problemHash, workspaceBytes, state, rankedSolutionIndices);
        else if(!state.m_resolved && state.m_problemHash != problemHash)
            return -1; // another problem under the key

        state.m_dispatchesSinceSelection.store(0, std::memory_order_relaxed);

        if(cold)
            return -1;

        if(!state.m_resolved)
        {
            // Stops at the budget, so it cannot overflow.
            if(!state.m_gaveUp)
                ++state.m_calls;

            if(state.m_calls > visitBudget(state))
                state.m_gaveUp = true;

            const bool allIssued = nextCandidate(state) < 0;

            if((allIssued || state.m_gaveUp) && state.m_pending.empty())
                resolve(key, state);
        }

        if(state.m_resolved)
            return servedPosition(state, rankedSolutionIndices, record);

        // Given up but still draining.
        if(state.m_gaveUp)
            return -1;

        const int candidate = nextCandidate(state);
        if(candidate < 0)
            return -1;

        const int position = positionOf(rankedSolutionIndices, state.m_candidates[candidate]);

        if(position >= 0)
            t_offer = {&state, candidate};

        return position;
    }

    // The first call served the winner writes the row, whichever problem under
    // the key it is, so a key whose registrant stopped dispatching still gets
    // one. Any such call can run the winner: it is in its ranking.
    int OnlineTuner::servedPosition(ProblemState&           state,
                                    const std::vector<int>& rankedSolutionIndices,
                                    Record*                 record)
    {
        const int position = positionOf(rankedSolutionIndices, state.m_winner);

        if(position >= 0 && record
           && !state.m_recordClaimed.exchange(true, std::memory_order_relaxed))
        {
            record->us             = state.m_winnerUs;
            record->complete       = state.m_complete;
            record->candidates     = static_cast<int>(state.m_candidates.size());
            record->workspaceBytes = state.m_workspace;
        }

        return position;
    }

    bool OnlineTuner::settled(const Key& key)
    {
        std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

        const auto iter = m_problems.find(key);
        return iter != m_problems.end() && iter->second.m_resolved
               && (iter->second.m_winner < 0
                   || iter->second.m_recordClaimed.load(std::memory_order_relaxed));
    }

    void OnlineTuner::harvestPending(const Key& key)
    {
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_problems.find(key);
            if(iter == m_problems.end() || iter->second.m_resolved)
                return;

            ProblemState& state = iter->second;

            const int unseen
                = state.m_dispatchesSinceSelection.fetch_add(1, std::memory_order_relaxed) + 1;

            const bool exhausted = unseen > abandonBudget(state);

            if(state.m_pending.empty() && !exhausted)
                return;
        }

        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        auto iter = m_problems.find(key);
        if(iter == m_problems.end() || iter->second.m_resolved)
            return;

        ProblemState& state = iter->second;

        if(state.m_dispatchesSinceSelection.load(std::memory_order_relaxed) > abandonBudget(state))
            state.m_gaveUp = true;

        if(state.m_pending.empty() && !state.m_gaveUp)
            return;

        size_t outstanding = 0;

        for(size_t i = 0; i < state.m_pending.size(); ++i)
        {
            const PendingMeasurement pending = state.m_pending[i];

            // Under the write lock, so no other harvest can hand the pair out
            // in between.
            float             us = 0.0f;
            const PairReading reading
                = pending.m_launched ? readPair(pending, us) : PairReading::Outstanding;
            if(reading == PairReading::Outstanding)
                state.m_pending[outstanding++] = pending;
            else
                settlePair(key, state, pending, reading, us, "sample");
        }

        state.m_pending.resize(outstanding);

        if(!state.m_resolved && state.m_pending.empty()
           && (state.m_gaveUp || nextCandidate(state) < 0))
            resolve(key, state);
    }

    // The sign test is load-bearing: a pair its launch did not record still
    // reads as complete, but every way of missing a record reads non-positive
    // (fresh and pooled pairs are inverted).
    //
    // One hipEventElapsedTime, which answers hipErrorNotReady while either
    // event is outstanding and never waits. Unlike hipEventQuery, it neither
    // checks nor touches the stream an event was last recorded on, so reading a
    // pair cannot invalidate a capture in progress, on any thread, or reach a
    // stream the caller has since destroyed. It runs inside the caller's
    // hipblasLtMatmul, so an error it causes is cleared only when the thread
    // had none before.
    OnlineTuner::PairReading OnlineTuner::readPair(const PendingMeasurement& pending,
                                                   float&                    us) const
    {
        const hipError_t before = hipPeekAtLastError();

        float milliseconds = 0.0f;
        us                 = 0.0f;
        const hipError_t elapsed
            = hipEventElapsedTime(&milliseconds, pending.m_start, pending.m_stop);

        if(elapsed == hipErrorNotReady)
            return PairReading::Outstanding;

        if(elapsed != hipSuccess)
        {
            if(before == hipSuccess)
                static_cast<void>(hipGetLastError());
            return PairReading::Rejected;
        }

        if(!std::isfinite(milliseconds) || milliseconds <= 0.0f)
            return PairReading::Rejected;

        us = milliseconds * 1000.0f;
        return PairReading::Sample;
    }

    // Only a pair that produced a sample is known to have been recorded, so only
    // it is pooled. Nothing here throws (see registerProblem() and
    // acquireEvents()), so a harvest cannot stop half way through its compaction
    // and leave a pooled pair listed.
    void OnlineTuner::settlePair(const Key&                key,
                                 ProblemState&             state,
                                 const PendingMeasurement& pending,
                                 PairReading               reading,
                                 float                     us,
                                 const char*               event)
    {
        const bool timed = reading == PairReading::Sample;
        if(timed)
        {
            state.m_samples[pending.m_candidate].push_back(us);
            recycleEvents(pending.m_start, pending.m_stop, pending.m_device);
        }
        else
            retireEvents(pending.m_start, pending.m_stop, pending.m_device);

        if(m_trace)
            try
            {
                std::ostringstream msg = traceLine(timed ? event : "drop", key);
                msg << " cand=" << pending.m_candidate
                    << " sol=" << state.m_candidates[pending.m_candidate];
                if(timed)
                    msg << " us=" << us;
                msg << " device=" << pending.m_device;
                m_trace(__func__, msg.str());
            }
            catch(...)
            {
            }
    }

    // At the cap with nothing pooled, pool the finished pairs other problems
    // hold on device, crediting their samples, and retire those that read as
    // rejected, as their own harvest would, so one-shot shapes cannot hold the
    // cap for good. Nothing is resolved here, and requester's own harvest has
    // just run.
    void OnlineTuner::reclaimFinishedPairs(int device, const ProblemState* requester)
    {
        size_t listed = 0;

        for(size_t h = 0; h < m_holders.size(); ++h)
        {
            Problems::value_type* const holder = m_holders[h];
            ProblemState&               state  = holder->second;

            if(state.m_pending.empty())
            {
                state.m_listed = false;
                continue;
            }

            m_holders[listed++] = holder;

            if(&state == requester)
                continue;

            size_t kept = 0;

            for(size_t i = 0; i < state.m_pending.size(); ++i)
            {
                const PendingMeasurement pending = state.m_pending[i];

                float             us      = 0.0f;
                const PairReading reading = pending.m_device == device && pending.m_launched
                                                ? readPair(pending, us)
                                                : PairReading::Outstanding;
                if(reading == PairReading::Outstanding)
                    state.m_pending[kept++] = pending;
                else
                    settlePair(holder->first, state, pending, reading, us, "reclaim");
            }

            state.m_pending.resize(kept);

            if(kept == 0)
            {
                state.m_listed = false;
                --listed;
            }
        }

        m_holders.resize(listed);
    }

    bool OnlineTuner::beginMeasurement(
        const Key& key, int solutionIndex, hipStream_t stream, hipEvent_t& start, hipEvent_t& stop)
    {
        // Only a launch this thread was offered is timed, so any other needs no lock.
        if(t_offer.m_candidate < 0)
            return false;

        // Nor a launch on a thread holding an error it has not read: recording
        // the pair could replace that error with its own. The offer stays.
        if(hipPeekAtLastError() != hipSuccess)
            return false;

        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_problems.find(key);
            if(iter == m_problems.end() || measurableCandidate(iter->second, solutionIndex) < 0)
                return false;
        }

        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        auto iter = m_problems.find(key);
        if(iter == m_problems.end())
            return false;

        ProblemState& state     = iter->second;
        const int     candidate = measurableCandidate(state, solutionIndex);
        if(candidate < 0)
            return false;

        // Declined, the launch runs untimed and the offer stays.
        int            device   = 0;
        const Acquired acquired = acquireEvents(stream, start, stop, device, &state);
        if(acquired != Acquired::Pair)
        {
            const bool firstAtCap
                = acquired == Acquired::AtCap && m_capReported.insert(device).second;
            const int cap = livePairCap();
            lock.unlock();

            if(firstAtCap && m_livePairCapReached)
                m_livePairCapReached(device, cap);

            return false;
        }

        t_offer.m_candidate = -1;

        // Listed first: the append can throw, and m_pending is reserved.
        if(!state.m_listed)
        {
            m_holders.push_back(&*iter);
            state.m_listed = true;
        }

        state.m_pending.push_back({start, stop, candidate, device});
        ++state.m_issued[candidate];

        return true;
    }

    OnlineTuner::Measurement
        OnlineTuner::measure(const Key& key, int solutionIndex, hipStream_t stream)
    {
        Measurement measurement;
        if(beginMeasurement(key, solutionIndex, stream, measurement.m_start, measurement.m_stop))
        {
            measurement.m_tuner = this;
            measurement.m_key   = &key;
        }
        return measurement;
    }

    // Under the write lock, so a harvest that sees the mark also sees the
    // launch's records. Nothing removes an entry before it is marked.
    void OnlineTuner::endMeasurement(const Key& key, hipEvent_t start)
    {
        if(!start)
            return;

        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        auto iter = m_problems.find(key);
        if(iter == m_problems.end())
            return;

        for(PendingMeasurement& pending : iter->second.m_pending)
            if(pending.m_start == start && !pending.m_launched)
            {
                pending.m_launched = true;
                return;
            }
    }

    void OnlineTuner::registerProblem(const Key&              key,
                                      size_t                  problemHash,
                                      size_t                  workspaceBytes,
                                      ProblemState&           state,
                                      const std::vector<int>& rankedSolutionIndices)
    {
        const size_t count = std::min(rankedSolutionIndices.size(), static_cast<size_t>(m_topK));

        state.m_problemHash = problemHash;
        state.m_workspace   = workspaceBytes;
        state.m_candidates.assign(rankedSolutionIndices.begin(),
                                  rankedSolutionIndices.begin() + count);
        state.m_samples.resize(count);
        state.m_issued.assign(count, 0);

        // A candidate takes at most m_repeats samples, so crediting one never
        // allocates.
        for(std::vector<float>& samples : state.m_samples)
            samples.reserve(static_cast<size_t>(m_repeats));

        // A single candidate is measured too, so its key gets a row and replays.
        state.m_pending.reserve(static_cast<size_t>(inFlightCap(state)));

        if(m_trace)
        {
            std::ostringstream msg = traceLine("register", key);
            msg << " candidates=" << count << " repeats=" << m_repeats
                << " cold=" << state.m_coldCalls << " sols=";
            for(size_t i = 0; i < count; ++i)
                msg << (i ? "," : "") << state.m_candidates[i];
            m_trace(__func__, msg.str());
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

    // Only the launch selectCandidate() offered this thread; matching on the
    // solution index alone would accept another problem's own pick under the
    // key. The issued count is re-checked because another thread may have taken
    // the last repeat.
    int OnlineTuner::measurableCandidate(const ProblemState& state, int solutionIndex) const
    {
        if(state.m_resolved || state.m_gaveUp)
            return -1;

        if(t_offer.m_candidate < 0 || t_offer.m_state != &state)
            return -1;

        if(state.m_candidates[t_offer.m_candidate] != solutionIndex
           || state.m_issued[t_offer.m_candidate] >= m_repeats)
            return -1;

        return t_offer.m_candidate;
    }

    int OnlineTuner::inFlightCap(const ProblemState& state) const
    {
        return static_cast<int>(state.m_candidates.size()) * m_repeats;
    }

    int OnlineTuner::visitBudget(const ProblemState& state) const
    {
        const int candidates
            = state.m_candidates.empty() ? m_topK : static_cast<int>(state.m_candidates.size());

        return candidates * m_repeats * c_visitBudgetFactor;
    }

    int OnlineTuner::abandonBudget(const ProblemState& state) const
    {
        return visitBudget(state) * c_abandonBudgetFactor;
    }

    void OnlineTuner::resolve(const Key& key, ProblemState& state)
    {
        int   winner      = -1;
        int   winnerIndex = -1;
        float winnerScore = 0.0f;

        state.m_complete = true;

        for(size_t i = 0; i < state.m_samples.size(); ++i)
        {
            if(state.m_samples[i].empty())
            {
                state.m_complete = false;
                continue;
            }

            // Timing noise only makes a launch slower, so the smallest sample wins.
            const float candidateScore
                = *std::min_element(state.m_samples[i].begin(), state.m_samples[i].end());
            if(winnerIndex < 0 || candidateScore < winnerScore)
            {
                winnerIndex = static_cast<int>(i);
                winnerScore = candidateScore;
                winner      = state.m_candidates[i];
            }
        }

        // A challenger replaces candidate 0 only by the configured margin.
        if(winnerIndex > 0 && m_minImprovementPct > 0 && !state.m_samples[0].empty())
        {
            const float incumbent
                = *std::min_element(state.m_samples[0].begin(), state.m_samples[0].end());
            if(winnerScore > incumbent * (1.0f - m_minImprovementPct / 100.0f))
            {
                winnerIndex = 0;
                winnerScore = incumbent;
                winner      = state.m_candidates[0];
            }
        }

        state.m_winner   = winner;
        state.m_winnerUs = winnerIndex < 0 ? 0.0f : winnerScore;
        state.m_resolved = true;

        if(m_trace)
        {
            const size_t samples = winnerIndex < 0 ? 0 : state.m_samples[winnerIndex].size();

            std::ostringstream msg = traceLine("winner", key);
            msg << " cand=" << winnerIndex << " sol=" << winner << " us=" << winnerScore
                << " samples=" << samples << " calls=" << state.m_calls
                << " unseen=" << state.m_dispatchesSinceSelection.load(std::memory_order_relaxed)
                << " gaveup=" << (state.m_gaveUp ? 1 : 0)
                << " complete=" << (state.m_complete ? 1 : 0);
            m_trace(__func__, msg.str());
        }

        releaseExploration(state);
    }

    // m_problems never erases, so a resolved problem gives back its buffers.
    // The candidate list stays: an empty one marks a new problem.
    void OnlineTuner::releaseExploration(ProblemState& state)
    {
        std::vector<std::vector<float>>().swap(state.m_samples);
        std::vector<PendingMeasurement>().swap(state.m_pending);
    }

    // Pairs are pooled per current device; a stream on another device is
    // refused only when a new pair would be created. A new pair is recorded
    // inverted once before it is handed out, like a pooled pair, so readPair()
    // never reads an event HIP has not seen recorded. Like readPair(), an error
    // a failed HIP call leaves is cleared only when the thread had none before.
    OnlineTuner::Acquired OnlineTuner::acquireEvents(hipStream_t         stream,
                                                     hipEvent_t&         start,
                                                     hipEvent_t&         stop,
                                                     int&                device,
                                                     const ProblemState* requester)
    {
        const hipError_t before = hipPeekAtLastError();

        const auto failed = [before] {
            if(before == hipSuccess)
                static_cast<void>(hipGetLastError());
            return Acquired::Failed;
        };

        if(hipGetDevice(&device) != hipSuccess)
            return failed();

        std::vector<EventPair>& pool = m_pairs[device];
        int&                    live = m_livePairs[device];

        if(pool.empty() && live >= livePairCap())
            reclaimFinishedPairs(device, requester);

        if(!pool.empty())
        {
            const EventPair pair = pool.back();
            pool.pop_back();

            start = pair.m_start;
            stop  = pair.m_stop;

            return Acquired::Pair;
        }

        if(live >= livePairCap())
            return Acquired::AtCap;

        // hipStreamGetDevice cannot take hipStreamLegacy, which, like the null
        // stream, is the current device's.
        int streamDevice = device;
        if(stream != nullptr && stream != hipStreamLegacy
           && hipStreamGetDevice(stream, &streamDevice) != hipSuccess)
            return failed();

        if(streamDevice != device)
            return Acquired::Failed;

        // The pool never holds more than the live pairs, so recycling never
        // allocates.
        pool.reserve(static_cast<size_t>(live) + 1);

        EventPair pair;
        if(hipEventCreateWithFlags(&pair.m_start, hipEventDefault) != hipSuccess)
            return failed();

        if(hipEventCreateWithFlags(&pair.m_stop, hipEventDefault) != hipSuccess)
        {
            static_cast<void>(hipEventDestroy(pair.m_start));
            return failed();
        }

        if(hipEventRecord(pair.m_stop, stream) != hipSuccess
           || hipEventRecord(pair.m_start, stream) != hipSuccess)
        {
            static_cast<void>(hipEventDestroy(pair.m_start));
            static_cast<void>(hipEventDestroy(pair.m_stop));
            return failed();
        }

        ++live;

        start = pair.m_start;
        stop  = pair.m_stop;

        return Acquired::Pair;
    }

    // Pooled inverted; see readPair().
    void OnlineTuner::recycleEvents(hipEvent_t start, hipEvent_t stop, int device)
    {
        if(start && stop)
            m_pairs[device].push_back({stop, start});
    }

    // A rejected pair is destroyed, not pooled: pooling would invert it back to
    // an orientation that reads its old duration as a positive time.
    void OnlineTuner::retireEvents(hipEvent_t start, hipEvent_t stop, int device)
    {
        if(start)
            static_cast<void>(hipEventDestroy(start));

        if(stop)
            static_cast<void>(hipEventDestroy(stop));

        int& live = m_livePairs[device];
        if(live > 0)
            --live;
    }
} // namespace rocblaslt
