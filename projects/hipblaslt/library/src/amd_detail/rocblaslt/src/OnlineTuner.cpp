// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OnlineTuner.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace rocblaslt
{
    namespace
    {
        constexpr const char* c_tracePrefix = "[hipblaslt-online-tune]";

        /**
         * The offer selectCandidate() last made to this thread: the candidate
         * the next launch on this thread may be timed for.
         *
         * Held per thread rather than on ProblemState because it pairs one
         * selection visit with the launch it asked for, and a problem can have
         * one such pairing outstanding per thread in the library. m_registration
         * ties it to one registration of the problem, so it can never match a
         * later candidate list under the same key.
         *
         * Every selection visit on this thread drops the offer, and only a visit
         * answered with a candidate renews it. On the measured path selection
         * and launch are the same hipblasLtMatmul call on the same thread, so
         * the offer a launch claims is the one its own call made, and no other
         * thread can take it. See the scope note on OnlineTuner.
         *
         * m_candidate < 0 means nothing is offered.
         */
        struct ThreadOffer
        {
            size_t m_key          = 0;
            size_t m_registration = 0;
            int    m_candidate    = -1;
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

        float minimum(const std::vector<float>& samples)
        {
            return *std::min_element(samples.begin(), samples.end());
        }

        std::ostringstream traceLine(const char* event, size_t problemKey)
        {
            std::ostringstream msg;
            msg << std::fixed << std::setprecision(3);
            msg << c_tracePrefix << " event=" << event << " key=" << problemKey;
            return msg;
        }
    }

    // Read once, like the tuning mode itself, so a setting changed after the
    // first lookup has no effect.
    void OnlineTuner::applyConfig()
    {
        const OnlineTunerConfig config = onlineTunerConfig();

        m_topK        = std::max(config.topK, 0);
        m_repeats     = std::max(config.repeats, 1);
        m_coldCalls   = std::max(config.coldCalls, 0);
        m_livePairCap = std::max(config.livePairCap, 1);
        m_verbose     = config.verbose;

        m_livePairCapReached = config.livePairCapReached;

        // One candidate is nothing to choose between.
        m_enabled = config.enabled && m_topK >= 2;
    }

    OnlineTuner::OnlineTuner()
    {
        applyConfig();
    }

    void OnlineTuner::reloadForTest()
    {
        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        // Slots first, so nothing can be reached through the table once the
        // pool it points into is gone.
        for(std::atomic<const Resolution*>& slot : m_resolutions)
            slot.store(nullptr, std::memory_order_release);

        // Nothing could reach these pairs again, and they would still count
        // against livePairCap().
        for(auto& [key, state] : m_problems)
            for(const PendingMeasurement& pending : state.m_pending)
                retireEvents(pending.m_start, pending.m_stop, pending.m_device);

        m_problems.clear();
        m_holders.clear();
        m_capReported.clear();
        m_resolutionPool.clear();
        m_winnerPool.clear();
        m_misses.store(0, std::memory_order_relaxed);

        applyConfig();
    }

    // The pooled events are deliberately not destroyed: this is a
    // function-local static, so its destructor can run after the HIP runtime's
    // teardown, where hipEventDestroy() faults.
    OnlineTuner::~OnlineTuner() { }

    /**
     * Four guards turn a visit away here: resolved, capturing or retained,
     * cold, and merged sibling. They answer two questions, and the code is laid
     * out in two steps to match:
     *
     * 1. Whose exploration is this? Only the problem that registered the
     *    candidate list owns it. A visit that is not its own writes nothing --
     *    not the abandonment counter, not the offer, not the budget.
     *
     * 2. Does this visit advance the rotation? An own visit always resets the
     *    abandonment counter, because the registrant is still asking. Only an
     *    own visit answered with a candidate spends a visit of the budget and
     *    carries an offer; a cold visit does neither.
     *
     * No write to the problem's state happens above marker A or between A and
     * B, except registerProblem(), which creates the registrant. Dropping this
     * thread's offer is above both, because the offer belongs to the thread,
     * not to the problem. A new guard goes above marker A if it says "not this
     * exploration's", and between A and B if it says "not a rotation visit".
     */
    int OnlineTuner::selectCandidateImpl(size_t                  problemKey,
                                         size_t                  problem,
                                         const std::vector<int>& rankedSolutionIndices,
                                         bool                    capturing,
                                         bool                    retained)
    {
        if(rankedSolutionIndices.empty())
            return -1;

        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_problems.find(problemKey);
            if(iter != m_problems.end() && iter->second.m_resolved)
                return positionOf(rankedSolutionIndices, iter->second.m_winner);
        }

        // A capture is never offered a probe, but is served a winner by the
        // resolved branch above. A captured graph never comes back through
        // selection, so a probe baked into it would be replayed for the life of
        // the graph, and the measurement hook refuses a capturing stream, so
        // the probe would buy no sample either. Returning before the visit is
        // counted keeps a captured shape from spending a budget it cannot use.
        //
        // A heuristic query is refused for the same two reasons: the caller
        // keeps the algo and dispatches it itself, and a caller-supplied algo
        // is never measured. See the scope note on OnlineTuner.
        if(capturing || retained)
            return -1;

        // This thread is about to dispatch something, so whatever it was
        // offered before is stale, whichever problem the dispatch belongs to --
        // including a merged sibling turned away below. Renewed at the bottom
        // only if this visit is answered with a candidate. Below the resolved
        // branch, which must stay a load and a compare.
        t_offer.m_candidate = -1;

        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        ProblemState& state = m_problems[problemKey];

        // ------------------------------------------------------------------
        // Step 1. Whose exploration is this?
        // ------------------------------------------------------------------

        // No candidate list yet and still inside the cold visits: the caller's
        // ordering is left alone, so exploration starts on a device the problem
        // has already warmed. There is no registrant yet, so the visit is its
        // own.
        const bool cold = state.m_candidates.empty() && state.m_coldCalls < m_coldCalls;

        if(cold)
            ++state.m_coldCalls;
        else if(state.m_candidates.empty())
        {
            // First sight past the cold visits; whoever registers owns the
            // exploration.
            registerProblem(problemKey, problem, state, rankedSolutionIndices);
        }
        // A merged sibling: not this exploration's to sample, so it is offered
        // nothing and spends none of the budget, which heavy sibling traffic
        // would otherwise exhaust. It is served the winner once there is one.
        else if(!state.m_resolved && state.m_problem != problem)
            return -1;

        // ------------------------------------------------------------------
        // MARKER A. Past here the visit is this exploration's own.
        // ------------------------------------------------------------------

        // The registrant is still asking; see m_dispatchesSinceSelection.
        state.m_dispatchesSinceSelection.store(0, std::memory_order_relaxed);

        if(cold)
            return -1;

        // ------------------------------------------------------------------
        // MARKER B. Past here the visit advances the rotation.
        // ------------------------------------------------------------------

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

        // Given up but still draining: promoting a candidate nothing will
        // measure would only perturb the caller's ordering.
        if(state.m_gaveUp)
            return -1;

        const int candidate = nextCandidate(state);
        if(candidate < 0)
            return -1;

        const int position = positionOf(rankedSolutionIndices, state.m_candidates[candidate]);

        // A candidate absent from the list in hand is not promoted, so no
        // launch of it is coming and an offer would only invite a wrong claim.
        if(position >= 0)
            t_offer = {problemKey, state.m_registration, candidate};

        return position;
    }

    void OnlineTuner::harvestPendingImpl(size_t problemKey)
    {
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_problems.find(problemKey);
            if(iter == m_problems.end())
                return;

            ProblemState& state = iter->second;

            // Every dispatch of the problem reaches this hook, so this is where
            // a problem whose registrant stopped visiting can be seen.
            const int unseen
                = state.m_dispatchesSinceSelection.fetch_add(1, std::memory_order_relaxed) + 1;

            const bool exhausted = !state.m_resolved && unseen > abandonBudget(state);

            if(state.m_pending.empty() && !exhausted)
                return;
        }

        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        auto iter = m_problems.find(problemKey);
        if(iter == m_problems.end() || iter->second.m_resolved)
            return;

        ProblemState& state = iter->second;

        // Giving up still drains the samples already paid for.
        if(state.m_dispatchesSinceSelection.load(std::memory_order_relaxed) > abandonBudget(state))
            state.m_gaveUp = true;

        if(state.m_pending.empty() && !state.m_gaveUp)
            return;

        size_t outstanding = 0;

        for(size_t i = 0; i < state.m_pending.size(); ++i)
        {
            const PendingMeasurement pending = state.m_pending[i];

            // Read under the write lock: otherwise another thread could harvest
            // this pair and hand it to a new measurement in between. A pair
            // whose launch has not returned is not read at all; see m_launched.
            // Unfinished pairs are compacted to the front in launch order.
            float             milliseconds = 0.0f;
            const PairReading reading
                = pending.m_launched ? readPair(pending, milliseconds) : PairReading::Outstanding;
            if(reading == PairReading::Outstanding)
            {
                state.m_pending[outstanding++] = pending;
                continue;
            }

            // Only a pair that produced a sample is known to have been
            // recorded, so only that pair may be handed out again.
            const bool timed = reading == PairReading::Sample;
            if(timed)
            {
                recycleEvents(pending.m_start, pending.m_stop, pending.m_device);
                state.m_samples[pending.m_candidate].push_back(milliseconds * 1000.0f);
            }
            else
                retireEvents(pending.m_start, pending.m_stop, pending.m_device);

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

    // What a pending pair whose launch has returned says now; used by both the
    // harvest and reclaimFinishedPairs().
    //
    // The sign test is load-bearing. A pair the launch did not record still
    // queries as complete, and every way of missing a record reads as
    // non-positive: a fresh pair still holds the inverted records
    // acquireEvents() made, a pooled pair is inverted, and a launch that
    // recorded start and then failed reads an older stop against a newer start.
    // A launch whose start record failed records no stop either.
    //
    // This runs on the caller's thread inside its hipblasLtMatmul, so it must
    // not leave an error in the thread's HIP state. Every pair it reads has
    // been recorded, so hipEventElapsedTime does not fail with
    // hipErrorInvalidHandle, and hipErrorNotReady does not stick. Any other
    // error from hipEventElapsedTime is cleared, but only when the thread had
    // none before the call: one that was already there is the caller's. An
    // error from hipEventQuery other than hipErrorNotReady is deliberately
    // left alone, and the pair stays outstanding: with the capture mode
    // relaxed, only a device fault can cause one, and that is the caller's to
    // see.
    //
    // HIP refuses hipEventQuery while a global-mode stream capture is in
    // progress anywhere in the process, and the refusal invalidates that
    // capture, whether it is this call's or another thread's. Nothing queried
    // here was recorded into a capture, so the query runs with this thread's
    // capture mode exchanged to relaxed, and the caller's mode is put back
    // straight after.
    OnlineTuner::PairReading OnlineTuner::readPair(const PendingMeasurement& pending,
                                                   float&                    milliseconds) const
    {
        const hipError_t before = hipPeekAtLastError();

        hipStreamCaptureMode mode = hipStreamCaptureModeRelaxed;
        if(hipThreadExchangeStreamCaptureMode(&mode) != hipSuccess)
        {
            if(before == hipSuccess)
                static_cast<void>(hipGetLastError());
            return PairReading::Outstanding;
        }

        const hipError_t query = hipEventQuery(pending.m_stop);
        static_cast<void>(hipThreadExchangeStreamCaptureMode(&mode));

        if(query != hipSuccess)
            return PairReading::Outstanding;

        milliseconds = 0.0f;
        const hipError_t elapsed
            = hipEventElapsedTime(&milliseconds, pending.m_start, pending.m_stop);

        if(elapsed != hipSuccess && elapsed != hipErrorNotReady && before == hipSuccess)
            static_cast<void>(hipGetLastError());

        const bool timed
            = elapsed == hipSuccess && std::isfinite(milliseconds) && milliseconds > 0.0f;

        return timed ? PairReading::Sample : PairReading::Rejected;
    }

    /**
     * @brief Read back the finished launches other problems hold pairs for on
     * device, and pool those pairs.
     *
     * A problem's pairs otherwise come back only when the problem is dispatched
     * again. Shapes dispatched once would keep their pairs for good, and once
     * they held the whole cap nothing new would be measured on the device; a
     * rotation through more shapes than the cap would starve the same way.
     *
     * Called from acquireEvents() only when the device is at the cap with
     * nothing pooled, so it costs nothing below the cap. It walks m_holders,
     * not every problem seen, and its cost is spread over the pairs it hands
     * back.
     *
     * A sample read back here is credited to the problem that took it, as its
     * own harvest would have done; resolving stays with that problem's own
     * visits. requester is skipped because its harvest has just run.
     *
     * Only a pair that reads as a sample is taken. A pair whose launch has not
     * returned is not read, a stop the GPU has not reached reads as
     * outstanding, and a pair its launch did not record reads non-positive and
     * is left for its own harvest to retire. Nothing here destroys a pair.
     * Runs under the write lock, like the harvest, and makes the same HIP
     * calls.
     */
    void OnlineTuner::reclaimFinishedPairs(int device, size_t requester)
    {
        size_t listed = 0;

        for(size_t h = 0; h < m_holders.size(); ++h)
        {
            const size_t key   = m_holders[h];
            const auto   found = m_problems.find(key);
            if(found == m_problems.end())
                continue;

            ProblemState& state = found->second;

            if(state.m_pending.empty())
            {
                state.m_listed = false;
                continue;
            }

            m_holders[listed++] = key;

            if(key == requester)
                continue;

            size_t kept = 0;

            for(size_t i = 0; i < state.m_pending.size(); ++i)
            {
                const PendingMeasurement pending = state.m_pending[i];

                // The m_launched test is defence in depth here: an unmarked
                // pair reads as a sample only once both of its records have
                // landed, and pooling it then is harmless.
                float milliseconds = 0.0f;
                if(pending.m_device != device || !pending.m_launched
                   || readPair(pending, milliseconds) != PairReading::Sample)
                {
                    state.m_pending[kept++] = pending;
                    continue;
                }

                recycleEvents(pending.m_start, pending.m_stop, pending.m_device);
                state.m_samples[pending.m_candidate].push_back(milliseconds * 1000.0f);

                if(m_verbose)
                {
                    std::ostringstream msg = traceLine("reclaim", key);
                    msg << " cand=" << pending.m_candidate
                        << " sol=" << state.m_candidates[pending.m_candidate]
                        << " us=" << milliseconds * 1000.0f << " device=" << device << "\n";
                    std::cerr << msg.str();
                }
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

    bool OnlineTuner::beginMeasurementImpl(size_t      problemKey,
                                           int         solutionIndex,
                                           hipStream_t stream,
                                           hipEvent_t& start,
                                           hipEvent_t& stop)
    {
        {
            std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

            auto iter = m_problems.find(problemKey);
            if(iter == m_problems.end())
            {
                // The two hooks disagreeing on the key would otherwise look
                // like the feature being off, so it is counted and traced.
                m_misses.fetch_add(1, std::memory_order_relaxed);

                if(m_verbose)
                {
                    std::ostringstream msg = traceLine("miss", problemKey);
                    msg << " sol=" << solutionIndex << "\n";
                    std::cerr << msg.str();
                }

                return false;
            }

            if(measurableCandidate(iter->second, problemKey, solutionIndex) < 0)
                return false;
        }

        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        auto iter = m_problems.find(problemKey);
        if(iter == m_problems.end())
            return false;

        ProblemState& state     = iter->second;
        const int     candidate = measurableCandidate(state, problemKey, solutionIndex);
        if(candidate < 0)
            return false;

        // Checked here rather than in measurableCandidate() so the refusal is
        // counted, as declined= on the winner trace line.
        if(static_cast<int>(state.m_pending.size()) >= inFlightCap(state))
        {
            ++state.m_declined;
            return false;
        }

        // Declined too when the device is at livePairCap() with every pair in
        // flight, or when HIP will not create an event. The launch then runs
        // untimed, and the offer stays for this thread's next launch.
        int            device   = 0;
        const Acquired acquired = acquireEvents(stream, start, stop, device, problemKey);
        if(acquired != Acquired::Pair)
        {
            ++state.m_declined;

            // Reported once per device, after the lock is released.
            const bool firstAtCap
                = acquired == Acquired::AtCap && m_capReported.insert(device).second;
            const int cap = livePairCap();
            lock.unlock();

            if(firstAtCap && m_livePairCapReached)
                m_livePairCapReached(device, cap);

            return false;
        }

        // The offer buys one sample. It is this thread's, so clearing it needs
        // no lock.
        t_offer.m_candidate = -1;

        // Listed before the entry is added: the append can throw, and an entry
        // added before it would never be marked launched. m_pending is
        // reserved, so adding the entry cannot throw.
        if(!state.m_listed)
        {
            m_holders.push_back(problemKey);
            state.m_listed = true;
        }

        state.m_pending.push_back({start, stop, candidate, device});
        ++state.m_issued[candidate];

        return true;
    }

    // The entry is still on the pending list: nothing removes an entry before
    // it is marked here, and the problem cannot resolve while it holds one.
    // Under the write lock like every other change to the list, so a harvest
    // that sees the mark also sees the launch's records. One more exclusive
    // acquisition per measured launch, which is at most topK() * repeats() per
    // problem.
    void OnlineTuner::endMeasurementImpl(size_t problemKey, hipEvent_t start)
    {
        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        auto iter = m_problems.find(problemKey);
        if(iter == m_problems.end())
            return;

        for(PendingMeasurement& pending : iter->second.m_pending)
            if(pending.m_start == start && !pending.m_launched)
            {
                pending.m_launched = true;
                return;
            }
    }

    void OnlineTuner::registerProblem(size_t                  problemKey,
                                      size_t                  problem,
                                      ProblemState&           state,
                                      const std::vector<int>& rankedSolutionIndices)
    {
        const size_t count = std::min(rankedSolutionIndices.size(), static_cast<size_t>(m_topK));

        // Whoever registers owns the exploration; a key covers problems that
        // run differently, and letting a later one take over would mix their
        // samples.
        state.m_problem = problem;

        state.m_registration = ++m_registrations;

        state.m_candidates.assign(rankedSolutionIndices.begin(),
                                  rankedSolutionIndices.begin() + count);
        state.m_samples.resize(count);
        state.m_issued.assign(count, 0);

        // Nothing to compare against, so the problem is born resolved with no
        // winner.
        if(count < 2)
        {
            state.m_resolved = true;
            publishResolution(problemKey, state);
            releaseExploration(state);
        }
        else
            state.m_pending.reserve(static_cast<size_t>(inFlightCap(state)));

        if(m_verbose)
        {
            std::ostringstream msg = traceLine("register", problemKey);
            msg << " candidates=" << count << " repeats=" << m_repeats
                << " cold=" << state.m_coldCalls << " sols=";
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

    // Only the launch selectCandidate() offered this thread, for this key and
    // registration. Matching the solution index against the candidate list
    // instead would accept a merged sibling's own pick. The issued count is
    // re-checked because another thread may have taken the candidate's last
    // repeat since the offer was made.
    int OnlineTuner::measurableCandidate(const ProblemState& state,
                                         size_t              problemKey,
                                         int                 solutionIndex) const
    {
        if(state.m_resolved || state.m_gaveUp)
            return -1;

        if(t_offer.m_candidate < 0 || t_offer.m_key != problemKey
           || t_offer.m_registration != state.m_registration)
            return -1;

        if(state.m_candidates[t_offer.m_candidate] != solutionIndex
           || state.m_issued[t_offer.m_candidate] >= m_repeats)
            return -1;

        return t_offer.m_candidate;
    }

    // A whole exploration can be outstanding at once, so a caller can enqueue
    // every sample before the GPU finishes any of them.
    int OnlineTuner::inFlightCap(const ProblemState& state) const
    {
        return static_cast<int>(state.m_candidates.size()) * m_repeats;
    }

    int OnlineTuner::visitBudget(const ProblemState& state) const
    {
        // Before registration the configured depth stands in for the candidate
        // list; zero would end a problem on its first cold visit.
        const int candidates
            = state.m_candidates.empty() ? m_topK : static_cast<int>(state.m_candidates.size());

        return candidates * m_repeats * c_visitBudgetFactor;
    }

    // Counted in dispatches, not in visits of the registrant; see
    // ProblemState::m_dispatchesSinceSelection.
    int OnlineTuner::abandonBudget(const ProblemState& state) const
    {
        return visitBudget(state) * c_abandonBudgetFactor;
    }

    // Timing noise on a shared GPU only makes a launch slower, so the smallest
    // sample is the best estimate of a kernel's time.
    float OnlineTuner::score(const std::vector<float>& samples) const
    {
        return minimum(samples);
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

            const float candidateScore = score(state.m_samples[i]);
            if(winnerIndex < 0 || candidateScore < winnerScore)
            {
                winnerIndex = static_cast<int>(i);
                winnerScore = candidateScore;
                winner      = state.m_candidates[i];
            }
        }

        state.m_winner   = winner;
        state.m_winnerUs = winnerIndex < 0 ? 0.0f : winnerScore;
        state.m_resolved = true;

        publishResolution(problemKey, state);

        if(m_verbose)
        {
            const size_t samples = winnerIndex < 0 ? 0 : state.m_samples[winnerIndex].size();

            std::ostringstream msg = traceLine("winner", problemKey);
            msg << " cand=" << winnerIndex << " sol=" << winner << " us=" << winnerScore
                << " samples=" << samples << " calls=" << state.m_calls
                << " unseen=" << state.m_dispatchesSinceSelection.load(std::memory_order_relaxed)
                << " declined=" << state.m_declined << " gaveup=" << (state.m_gaveUp ? 1 : 0)
                << "\n";
            std::cerr << msg.str();
        }

        releaseExploration(state);
    }

    /**
     * @brief Free the buffers a problem needed only while exploring.
     *
     * Nothing reads the samples or the pending list once the problem is
     * resolved. The candidate list stays, because an empty one is how
     * selectCandidateImpl() recognises a new problem. m_problems never erases
     * (pinWinner() looks the registrant up, and lock-free readers of a
     * Resolution cannot be waited for), so this keeps a retired problem small.
     * Swapping with an empty vector releases the capacity, which clear() keeps.
     */
    void OnlineTuner::releaseExploration(ProblemState& state)
    {
        std::vector<std::vector<float>>().swap(state.m_samples);
        std::vector<PendingMeasurement>().swap(state.m_pending);
    }

    /**
     * @brief Make a resolved problem visible to the lock-free lookup.
     *
     * Runs under the write lock where a problem becomes resolved, so the entry
     * is complete before the release store publishes it; resolution() pairs it
     * with an acquire load. Entries live as long as the tuner, because a reader
     * holding one takes no lock and cannot be waited for. The pool is not
     * capped at c_resolutionSlots: the table is direct-mapped, so a slot is
     * shared rather than spent, and a cap would leave every later problem on
     * the locked path. Growth moves the unique_ptrs, not the entries.
     */
    void OnlineTuner::publishResolution(size_t problemKey, const ProblemState& state)
    {
        auto entry        = std::make_unique<Resolution>();
        entry->m_key      = problemKey;
        entry->m_winner   = state.m_winner;
        entry->m_winnerUs = state.m_winnerUs;

        const Resolution* published = entry.get();
        m_resolutionPool.push_back(std::move(entry));

        m_resolutions[problemKey & (c_resolutionSlots - 1)].store(published,
                                                                  std::memory_order_release);
    }

    // The write lock makes "at most once" exact; callers may check pinned()
    // first to stay off the lock, which is what the steady state does.
    const OnlineTuner::PinnedWinner*
        OnlineTuner::pinWinner(const Resolution&                                        resolved,
                               const std::shared_ptr<TensileLite::ContractionSolution>& solution,
                               size_t                                                   problem,
                               size_t                              requiredWorkspace,
                               size_t                              searchWorkspace,
                               const TensileLite::ProblemOverride& tuningKey)
    {
        if(!solution)
            return nullptr;

        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        // Not capped, like the resolution pool: a problem refused a winner
        // object would take the full lookup path on every call.
        if(resolved.m_pinned.load(std::memory_order_relaxed))
            return nullptr;

        // Only the problem that registered was sampled, so only it gets the
        // record and its key a row. A merged sibling can serve the winner from
        // its own ranking but did not measure it. m_problems never erases, so
        // a resolved key is always found.
        const auto registered = m_problems.find(resolved.m_key);
        if(registered == m_problems.end() || registered->second.m_problem != problem)
            return nullptr;

        auto entry               = std::make_unique<PinnedWinner>();
        entry->solution          = solution;
        entry->problem           = problem;
        entry->requiredWorkspace = requiredWorkspace;
        entry->searchWorkspace   = searchWorkspace;
        entry->tuningKey         = tuningKey;

        const PinnedWinner* published = entry.get();
        m_winnerPool.push_back(std::move(entry));

        resolved.m_pinned.store(published, std::memory_order_release);

        return published;
    }

    // A pair is only handed out on the device that created it: recording an
    // event on another device's stream is refused with
    // hipErrorInvalidResourceHandle, and the pair would keep its old
    // timestamps. A stream on another device than the current one is declined
    // before anything is created.
    //
    // A new pair is created only while the device holds fewer than
    // livePairCap() live pairs; at the cap, reclaimFinishedPairs() tries to
    // free one first.
    //
    // A new pair is recorded once, stop and then start, on the launch's stream
    // before it is handed out. That puts it in the state of a pooled pair --
    // recorded, and reading non-positive until a launch records it again -- so
    // readPair() never reads a pair HIP has not seen recorded, which would
    // leave hipErrorInvalidHandle in the caller's thread. This costs two
    // records per pair created, at most livePairCap() per device.
    OnlineTuner::Acquired OnlineTuner::acquireEvents(
        hipStream_t stream, hipEvent_t& start, hipEvent_t& stop, int& device, size_t requester)
    {
        if(hipGetDevice(&device) != hipSuccess)
            return Acquired::Failed;

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

        int streamDevice = -1;
        if(hipStreamGetDevice(stream, &streamDevice) != hipSuccess || streamDevice != device)
            return Acquired::Failed;

        EventPair pair;
        if(hipEventCreateWithFlags(&pair.m_start, hipEventDefault) != hipSuccess)
            return Acquired::Failed;

        if(hipEventCreateWithFlags(&pair.m_stop, hipEventDefault) != hipSuccess)
        {
            // Never counted as live, so destroyed rather than retired.
            static_cast<void>(hipEventDestroy(pair.m_start));
            return Acquired::Failed;
        }

        const hipError_t before = hipPeekAtLastError();
        if(hipEventRecord(pair.m_stop, stream) != hipSuccess
           || hipEventRecord(pair.m_start, stream) != hipSuccess)
        {
            static_cast<void>(hipEventDestroy(pair.m_start));
            static_cast<void>(hipEventDestroy(pair.m_stop));

            // Cleared only if the caller had no error first; see readPair().
            if(before == hipSuccess)
                static_cast<void>(hipGetLastError());
            return Acquired::Failed;
        }

        ++live;

        start = pair.m_start;
        stop  = pair.m_stop;

        return Acquired::Pair;
    }

    int OnlineTuner::livePairs(int device)
    {
        std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

        const auto found = m_livePairs.find(device);
        return found == m_livePairs.end() ? 0 : found->second;
    }

    // Pooled inverted, so reusing the pair without recording it reads the
    // negative of its last duration and is rejected by the sign test. device
    // is the one the pair was created on, which need not be the harvesting
    // thread's current device.
    void OnlineTuner::recycleEvents(hipEvent_t start, hipEvent_t stop, int device)
    {
        if(start && stop)
            m_pairs[device].push_back({stop, start});
    }

    // A rejected pair is destroyed rather than pooled: pooling it would invert
    // it a second time, and a pair in its original orientation that no launch
    // re-records reads its old duration as a plausible positive time.
    // hipEventDestroy() defers the release of a recorded event rather than
    // blocking, and its failure is not reported to the caller.
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
