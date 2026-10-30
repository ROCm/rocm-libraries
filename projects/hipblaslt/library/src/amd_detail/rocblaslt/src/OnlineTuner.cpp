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
         * The offer selectCandidate() last made to THIS THREAD.
         *
         * Not a field of ProblemState, and that is the whole point: the offer
         * is a pairing between one selection visit and the launch it asked
         * for, and a problem can have one of those outstanding per thread in
         * the library. See the offer paragraph on ProblemState for what the
         * per-problem slot this replaces could and could not express.
         *
         * m_candidate < 0 is "nothing offered" and is the only empty state; a
         * key of zero is a perfectly ordinary hash. m_registration is the
         * serial OnlineTuner stamped on the registration the offer was made
         * from, so a stale offer can never be matched against a later
         * candidate list that happens to reuse the key.
         *
         * ITS LIFETIME IS ONE RULE: dropped by any selection visit this thread
         * makes and renewed only where that visit is answered with a candidate.
         * A visit is the thread saying it is about to dispatch, so whatever it
         * was offered before is stale however the guards answer this one -- in
         * particular when they turn it away as a merged sibling, because the
         * sibling's launch is the very next thing the thread does.
         *
         * On the dispatch path that makes the pairing exact. selectCandidate()
         * and beginMeasurement() are the same hipblasLtMatmul call on this
         * thread with nothing of this tuner's in between, so the offer the
         * launch claims is the one this call's own visit made, and no caller on
         * any thread can take it. That is the only path there is: a launch of
         * an algo the caller supplied, which is what a heuristic query hands
         * out, goes past both measurement hooks, and a heuristic query is never
         * offered anything to begin with. See the scope note on OnlineTuner.
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

    // Pulled from the link rather than read here, so that the knobs the tuning
    // stack already owns are the only ones there are; see onlineTunerConfig().
    // Read once, like the tuning mode itself, so a knob set after the first
    // lookup has no effect.
    void OnlineTuner::applyConfig()
    {
        const OnlineTunerConfig config = onlineTunerConfig();

        m_topK      = std::max(config.topK, 0);
        m_repeats   = std::max(config.repeats, 1);
        m_coldCalls   = std::max(config.coldCalls, 0);
        m_livePairCap = std::max(config.livePairCap, 1);
        m_verbose     = config.verbose;

        m_livePairCapReached = config.livePairCapReached;

        // One candidate is nothing to choose between, so a depth below two is
        // the feature switched off however the mode is set.
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

        // The pairs the dropped problems still had outstanding would otherwise
        // be unreachable and still count against livePairCap().
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

    // The pooled events are deliberately not destroyed. This is a function-local
    // static, so its destructor competes with the HIP runtime's own teardown, and
    // hipEventDestroy() after that teardown faults. The process is exiting, so
    // leaking the handles costs nothing.
    OnlineTuner::~OnlineTuner() { }

    /**
     * @brief The rule this function keeps, stated once.
     *
     * Four guards turn a visit away here -- resolved, capturing or retained,
     * cold, merged sibling -- and each is argued where it stands. What none of
     * them says, and what two earlier defects were, is how they relate to the
     * state this function mutates. Both were a mutation sitting above a guard
     * that should have refused it: the give-up counter reset above the
     * merged-sibling guard, so
     * a sibling exempt from spending the budget still cleared the counter
     * exploration needs in order to end; and the offer dropped above the same
     * guard, so a sibling cancelled an offer the registrant had not consumed.
     *
     * THE RULE IS IN TWO PARTS, BECAUSE THE GUARDS ANSWER TWO QUESTIONS.
     *
     * 1. WHOSE EXPLORATION IS THIS? Exploration state belongs to the one
     *    problem that registered the candidate list. Before there is a
     *    registrant nobody owns it and every visit is its own; afterwards only
     *    the registrant's is. A visit that is not its own writes NOTHING -- not
     *    the counter, not the offer, not the budget -- and is answered from
     *    what it finds.
     *
     * 2. DOES THIS VISIT ADVANCE THE ROTATION? An own visit always says the
     *    library is still being asked about this shape, and so always resets
     *    the abandonment counter. Only an own visit that goes on to be answered
     *    with a candidate spends a visit of the budget and carries an offer. A
     *    cold visit is an own visit that does neither: it is the library being
     *    asked, which is what the counter measures, without a probe going out,
     *    which is what the budget pays for. That is why the two answers are not
     *    the same answer.
     *
     * The code below is laid out in exactly those two steps, with a marked line
     * for each and no write to this problem's state above either of them --
     * except registerProblem(), which creates the registrant rather than
     * spending anything of its. The one write that is above both is the drop of
     * THIS THREAD's offer, which belongs to no problem and is explained where
     * it stands.
     *
     * A FIFTH GUARD GOES ABOVE MARKER A IF IT SAYS "NOT THIS EXPLORATION'S",
     * AND BETWEEN A AND B IF IT SAYS "NOT A ROTATION VISIT". There is nowhere
     * else to put it, which is the property both defects above lacked.
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

        // A capture is not a dispatch, and what it writes down is permanent.
        //
        // Deliberately below the resolved branch above and not above it. A
        // resolved problem has a measured winner, and a graph is exactly where
        // that answer belongs: it is the same kernel cache mode would replay,
        // chosen the same way, and baking it in is the outcome the feature
        // exists to produce. What must not be baked in is a PROBE. Exploration
        // deliberately dispatches a candidate it does not believe is best, on
        // the understanding that the next visit reconsiders -- and a captured
        // graph never comes back through selection, so the next visit never
        // arrives. The caller replays a knowingly non-optimal kernel for the
        // life of the graph with no path to correction.
        //
        // It is also pure loss for the exploration: the measurement hook
        // refuses a capturing stream, correctly, since an event recorded into
        // a capture can be neither queried nor read back. So a probe issued
        // here buys no sample. Returning before the visit is counted is what
        // stops a captured shape spending a budget it can never pay for -- the
        // same reason a merged sibling spends none below.
        //
        // A heuristic query is refused for both reasons too, and so returns
        // here with nothing registered, counted or offered. Its answer is an
        // algo the caller keeps and dispatches itself, as often and for as long
        // as it likes, so a probe promoted into it can be just as permanent.
        // And it can never be measured: the launch of an algo the caller
        // supplied goes past both measurement hooks in runContractionProblem,
        // because nothing the library can see pairs that launch with the visit
        // that produced it -- the algo is a public type and carries no offer,
        // and the thread that dispatches it may have made any number of other
        // visits in between. A query is still served a measured winner, by the
        // resolved branch above, once dispatches of the problem have found one.
        // This is the heuristic half of the scope note on OnlineTuner.
        if(capturing || retained)
            return -1;

        // Dropped before anything else can hand out a new one. An offer is good
        // for the single launch the visit it was made to is about to issue, and
        // this is a different visit; one this thread never dispatched must not
        // be left lying around for its next dispatch to claim -- including a
        // merged sibling's dispatch, which is the next thing this thread does
        // when the guards below turn it away.
        //
        // ABOVE BOTH MARKERS, AND NOT AN EXCEPTION TO THEM: the offer is this
        // THREAD's and not this exploration's, so dropping it is not writing
        // another problem's state. The thread is about to dispatch something;
        // whatever it was last offered is stale whoever that dispatch belongs
        // to. Renewed at the bottom, and only where this visit is answered with
        // a candidate. See ThreadOffer.
        //
        // Deliberately below the resolved branch, which is the lock-free steady
        // state and must stay a load and a compare.
        t_offer.m_candidate = -1;

        std::unique_lock<std::shared_timed_mutex> lock(m_mutex);

        ProblemState& state = m_problems[problemKey];

        // ------------------------------------------------------------------
        // Step 1. Whose exploration is this?
        // ------------------------------------------------------------------

        // No candidate list yet, so this problem is still inside its cold
        // visits: the caller's own ordering is left alone and nothing is
        // offered an event pair, which is what makes exploration start on a
        // device this problem has already warmed. There is no registrant yet
        // either, so the sibling guard below has nothing to compare against
        // and this visit is its own.
        const bool cold = state.m_candidates.empty() && state.m_coldCalls < m_coldCalls;

        if(cold)
            ++state.m_coldCalls;
        else if(state.m_candidates.empty())
        {
            // First sight past the cold visits. Whoever registers owns the
            // exploration; see registerProblem().
            registerProblem(problemKey, problem, state, rankedSolutionIndices);
        }
        // A problem the key merges with the one being explored. Its launches
        // are not this exploration's to sample, so it is offered nothing -- and
        // it spends none of the visit budget either, which a sibling
        // dispatching often enough could otherwise exhaust, ending exploration
        // early on whatever samples had been taken by then. It takes the branch
        // below once there is a winner, like every other caller.
        //
        // Above marker A, and that placement is the whole rule: a visit that
        // is not this exploration's writes none of this exploration's state.
        // Reached only when there is a registrant to compare against -- the two
        // branches above are the cases where there is not.
        else if(!state.m_resolved && state.m_problem != problem)
            return -1;

        // ------------------------------------------------------------------
        // MARKER A. Past here the visit is this exploration's own.
        // ------------------------------------------------------------------

        // Selection has visited, so the run of dispatches the registrant had
        // not made ends here. Only an unbroken run of them says the library has
        // stopped being asked by the one caller whose visits can finish this.
        state.m_dispatchesSinceSelection.store(0, std::memory_order_relaxed);

        // An own visit that does not advance the rotation: still the library
        // being asked, which is all marker A's counter measures, but no probe
        // goes out so no visit of the budget is spent and no offer is carried.
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

        // Given up but still draining: promoting a candidate nothing is going to
        // measure would only perturb the caller's own ordering.
        if(state.m_gaveUp)
            return -1;

        const int candidate = nextCandidate(state);
        if(candidate < 0)
            return -1;

        const int position = positionOf(rankedSolutionIndices, state.m_candidates[candidate]);

        // Recorded only where the caller is really being asked for it. A
        // candidate absent from the list in hand is not promoted, so no launch
        // of it is coming and an offer would only be there for something else
        // to claim.
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

            // This hook is the only one every dispatch of the problem reaches,
            // so it is where a problem selection has stopped visiting can be
            // seen at all. See ProblemState::m_dispatchesSinceSelection.
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

        // Re-read under the write lock, which is where it can be acted on. The
        // drain below still runs: samples already paid for are not thrown away
        // for the sake of giving up, exactly as the selection half does it.
        if(state.m_dispatchesSinceSelection.load(std::memory_order_relaxed) > abandonBudget(state))
            state.m_gaveUp = true;

        if(state.m_pending.empty() && !state.m_gaveUp)
            return;

        size_t outstanding = 0;

        for(size_t i = 0; i < state.m_pending.size(); ++i)
        {
            const PendingMeasurement pending = state.m_pending[i];

            // Queried under the write lock rather than the shared one: another
            // thread could otherwise harvest this pair and start a new
            // measurement in between, and recycling events the GPU is still
            // writing to would corrupt that measurement.
            //
            // A pair whose launch has not returned is not read at all. It may
            // not be recorded yet, and would then be rejected and destroyed
            // under the launch about to record it. See m_launched.
            //
            // Records the GPU has not finished are compacted to the front, which
            // keeps the launch order the rest of this loop reports in.
            float             milliseconds = 0.0f;
            const PairReading reading
                = pending.m_launched ? readPair(pending, milliseconds) : PairReading::Outstanding;
            if(reading == PairReading::Outstanding)
            {
                state.m_pending[outstanding++] = pending;
                continue;
            }

            // Only a pair that produced a sample is known to have been
            // recorded, so only that pair is safe to hand out again.
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

    // What a pending pair says now, read the same way by a problem's own
    // harvest and by reclaimFinishedPairs().
    //
    // A pair the caller never recorded still queries as complete, so the
    // elapsed time is the only evidence that the launch happened. Every way of
    // missing one reads as non-positive, which is why the sign test is
    // load-bearing and not defensive: a fresh pair still holds the inverted
    // records acquireEvents() made when it created it, a pooled pair is
    // inverted so it reads the negative of its last duration, and a launch
    // that recorded start and then failed reads an older stop against a newer
    // start. A launch whose start record failed records no stop either --
    // SolutionAdapter::launchKernel skips it -- so that case is one of the
    // first two and never a fresh stop against a stale start.
    //
    // Runs inside the caller's hipblasLtMatmul, on the caller's thread, so it
    // must not leave an error in that thread's HIP state for the caller's own
    // hipGetLastError() to find and blame on its work. Every pair this reads
    // has been recorded, which is what keeps hipEventElapsedTime from failing
    // here at all: on a pair never recorded it fails with
    // hipErrorInvalidHandle, and that error outlives every later successful
    // call until the caller reads it. hipErrorNotReady, from a start the GPU
    // has not reached, does not stick. Anything else it might still return is
    // this library's to clear -- but only when the thread had no error before
    // the call, since one that was already there is the caller's.
    //
    // hipEventQuery is the one call here that HIP refuses while a stream
    // capture in global mode is in progress anywhere in the process, and the
    // refusal invalidates that capture: the caller's own, when this call is
    // being captured, or another thread's. Nothing this queries was recorded
    // into a capture -- beginMeasurement() is never reached for a capturing
    // stream -- so the query is safe, and it runs with this thread's capture
    // mode exchanged to relaxed, which tells HIP exactly that. The caller's
    // mode is put back before anything else runs.
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
     * A problem's pairs otherwise only come back through a harvest of its own
     * key, which needs that key to be dispatched again. A shape dispatched
     * fewer times than an exploration takes never is, so without this its pair
     * stays taken for the life of the process, and once such shapes hold the
     * whole of livePairCap() nothing new is ever measured on the device again.
     * Rotating through more shapes than the cap allows starves the same way,
     * only slower: a shape waiting for a pair spends its visit budget on
     * declined launches and resolves with no winner.
     *
     * Called only from acquireEvents(), and only when the device is at the cap
     * with nothing pooled, so below the cap it costs nothing. A pass walks
     * m_holders rather than every problem the tuner has seen, and drops the
     * problems it finds holding nothing, so what it walks is bounded by the
     * live pairs plus the problems handed a pair since the previous pass. Its
     * cost is spread over the pairs it hands back: a pass that pools many
     * pairs is followed by as many acquisitions that need no pass. It repeats
     * on every acquisition only while all of the device's pairs are in flight,
     * which takes a host running a long way ahead of its GPU.
     *
     * Each sample a pass reads back is credited to the problem that took it,
     * exactly as that problem's own harvest would have done: the sample is
     * kept, and the problem still resolves on its own visits if it comes back.
     * Nothing is resolved here; that stays with the visits that own the
     * exploration. requester is skipped because its own harvest has just run.
     *
     * Only a pair that reads as a sample is taken, which is what keeps this
     * away from launches that are not finished. A pair handed out by
     * beginMeasurement() whose launch has not returned -- another thread can
     * be between the two right now, outside the lock -- is not read at all.
     * A stop the GPU has not reached queries as outstanding. A pair whose
     * launch returned without recording it reads non-positive, by the
     * invariant the sign test relies on, and is left for its own problem's
     * harvest to retire. Nothing here retires or destroys a pair.
     *
     * Under the write lock, like the harvest, for the same reason: the events
     * are queried and the pending lists rewritten together, and nothing else
     * can take or pool a pair in between. The HIP calls are the harvest's
     * own two, so no lock is held across anything the harvest did not
     * already hold it across.
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
                // A key the selection hook never registered. The two hooks
                // disagreeing is otherwise indistinguishable from the feature
                // being switched off, so it gets its own event and a count.
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

        // Checked here rather than in measurableCandidate() so the refusal can be
        // counted: exploration stalling behind a full in-flight set is the
        // failure this scheme exists to avoid, and declined= on the winner line
        // is how a recurrence would show up.
        if(static_cast<int>(state.m_pending.size()) >= inFlightCap(state))
        {
            ++state.m_declined;
            return false;
        }

        // Declined the same way when the device is at livePairCap() and every
        // pair it holds is still in flight, which is the cap doing its job, or
        // when HIP will not create an event. Either way the launch goes ahead
        // untimed, and the offer is left for the next launch this thread's
        // visit makes.
        int            device   = 0;
        const Acquired acquired = acquireEvents(stream, start, stop, device, problemKey);
        if(acquired != Acquired::Pair)
        {
            ++state.m_declined;

            // Said once per device, after the lock is let go. Otherwise the
            // only sign of a launch going untimed at the cap is declined= on
            // the winner trace line, which reads exactly like a shape that was
            // measured.
            const bool firstAtCap
                = acquired == Acquired::AtCap && m_capReported.insert(device).second;
            const int cap = livePairCap();
            lock.unlock();

            if(firstAtCap && m_livePairCapReached)
                m_livePairCapReached(device, cap);

            return false;
        }

        // Spent. The offer buys one sample, so a second launch claiming the
        // same one would issue a candidate twice on a single visit of it, and a
        // launch arriving after the caller's own is by definition not the one
        // selection asked for. This thread's offer and no other's, which is why
        // nothing here needs the lock the surrounding state does.
        t_offer.m_candidate = -1;

        state.m_pending.push_back({start, stop, candidate, device});
        ++state.m_issued[candidate];

        if(!state.m_listed)
        {
            state.m_listed = true;
            m_holders.push_back(problemKey);
        }

        return true;
    }

    // The pair is still on the problem's pending list: nothing removes an
    // entry before it is marked here, and the problem cannot resolve while it
    // holds one. Under the write lock like every other change to that list,
    // so a harvest that sees the mark also sees the launch's records, which
    // were made before this call.
    //
    // Costs one more exclusive acquisition per measured launch, which happens
    // at most topK() * repeats() times per problem and never once it is
    // resolved, or with the tuner off.
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

        // Whoever registers owns the exploration, for the same reason the first
        // caller to offer a winner object owns that record: a key covers
        // problems that rank and run differently, and letting a later one take
        // it over would only make the two take turns discarding each other's
        // samples.
        state.m_problem = problem;

        // Names this candidate list, so an offer outstanding on some thread
        // from an earlier one cannot be claimed against it. See ThreadOffer.
        state.m_registration = ++m_registrations;

        state.m_candidates.assign(rankedSolutionIndices.begin(),
                                  rankedSolutionIndices.begin() + count);
        state.m_samples.resize(count);
        state.m_issued.assign(count, 0);

        // Nothing to compare against, so the problem is born resolved with no
        // winner and every later call takes the read-only path.
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

    // The launch selectCandidate() asked THIS THREAD for, and nothing else.
    // Searching the candidate list for the index instead would accept a merged
    // sibling's own pick, which is a different problem's launch wearing a
    // candidate's number; see the offer paragraph on ProblemState.
    //
    // The key and the registration are both tested, so an offer made for
    // another problem, or from a candidate list this one has since replaced,
    // is not an offer here. The issued count is re-tested because
    // nextCandidate() picked the offer, and another thread exploring the same
    // problem can have taken that candidate's last repeat in between -- a
    // dropped sample, never a sample over the repeat count.
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
        // Before registration there is no candidate list to size the budget
        // from, and the configured depth is what that list is going to be. A
        // budget of zero here would end a problem on the first visit of its
        // cold phase.
        const int candidates
            = state.m_candidates.empty() ? m_topK : static_cast<int>(state.m_candidates.size());

        return candidates * m_repeats * c_visitBudgetFactor;
    }

    // How long a problem no caller is exploring may stay unresolved, counted in
    // dispatches rather than in visits of the registrant. A separate and much
    // larger ceiling than visitBudget() on purpose: see
    // ProblemState::m_dispatchesSinceSelection and c_abandonBudgetFactor.
    int OnlineTuner::abandonBudget(const ProblemState& state) const
    {
        return visitBudget(state) * c_abandonBudgetFactor;
    }

    // GPU timing noise is one-sided: contention, clock excursions and cache
    // state make a launch slower than the kernel's floor, never faster, so the
    // smallest of a candidate's samples is its best estimate with the rest of
    // the machine taken out. Not a knob: a measurement shared with whatever
    // else the device is doing has no typical value worth reporting, and the
    // alternative was measured not to help winner stability either way.
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
     * @brief Give back the heap a problem needed only while it was exploring.
     *
     * The samples and the in-flight record exist to pick a winner, and once one
     * is picked nothing reads them again: every path that would is behind a
     * m_resolved test, and resolve() runs once. What is left is the problem's
     * answer -- its winner, its score, and the candidate list -- which stays,
     * because an empty candidate list is how selectCandidateImpl() recognises a
     * problem it has never seen and would re-register a resolved one.
     *
     * Worth doing because m_problems never erases, by design: pinWinner() looks
     * the registering problem back up to decide whose key a row may be written
     * under, and a reader of a published Resolution takes no lock and so cannot
     * be waited for. A process that explores a lot of shapes therefore keeps
     * every ProblemState it ever made, and clear() alone would keep their
     * buffers too -- it is defined to leave capacity alone. Swapping against an
     * empty vector is what actually hands the pages back.
     *
     * For the default depth that is the larger part of a retired problem: the
     * sample arrays and a pending record reserved at the in-flight cap come to
     * a few hundred bytes against a hundred or so for what remains.
     */
    void OnlineTuner::releaseExploration(ProblemState& state)
    {
        std::vector<std::vector<float>>().swap(state.m_samples);
        std::vector<PendingMeasurement>().swap(state.m_pending);
    }

    /**
     * @brief Make a resolved problem visible to the lock-free lookup.
     *
     * Called only where a problem becomes resolved -- resolve(), and the
     * single-candidate case in registerProblem() -- both of which run under the
     * write lock, so the entry is fully built before anything can see it. The
     * release store is what orders those writes ahead of the pointer, pairing
     * with the acquire load in resolution().
     *
     * Entries are kept alive for the life of the tuner rather than recycled,
     * because a reader holding one takes no lock and so cannot be waited for.
     *
     * The pool is unbounded for that reason and not by oversight. It used to
     * stop at c_resolutionSlots, on the reasoning that a full table can hold no
     * more -- but the table is direct-mapped, so a slot is not spent, it is
     * shared. Refusing to publish past the 4096th resolution meant every
     * problem resolved after that kept a null resolution() for ever and so took
     * the full locked path on every dispatch, including problems whose slot was
     * free. One Resolution is a key, an int, a float and two atomics; the
     * m_problems entry the same problem already has, and never gives back,
     * costs more. See c_resolutionSlots for the boundary that remains.
     *
     * Growth moves the unique_ptrs, not the objects, so a pointer already
     * published stays good.
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

    // The write lock is what makes "at most once" true rather than merely
    // likely: two threads resolving the same problem would otherwise both see
    // an empty record and both fill it. The load inside the lock is the
    // authoritative one; a caller is free to check pinned() first to stay off
    // the lock entirely, which is what the steady state does.
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

        // Unbounded for the same reason as the resolution pool, and it has to
        // move with it: a problem refused a winner object here keeps a null
        // pinned(), which is what sends resolvedOnlineTuningAnswer() and
        // promoteResolvedOnlineTuningWinner() back down the full path. Capping
        // one and not the other would have left the cliff in place.
        if(resolved.m_pinned.load(std::memory_order_relaxed))
            return nullptr;

        // The record describes the problem that was measured, and that problem
        // is the one that registered the exploration -- it is the only one
        // beginMeasurement() took a sample for. Reaching here says the winner
        // is in this caller's ranking, which is true of every merged sibling
        // that can serve it and says nothing about who measured it, so the
        // caller is matched against the registration instead. Refusing one
        // costs it the shortcut and its own ranking still answers; what it
        // buys is that the key a row is written under belongs to the problem
        // the time was taken on. See PinnedWinner::tuningKey.
        //
        // The state is in m_problems, which never erases, and the resolution
        // was published from it, so the lookup finds it whenever the key is
        // one this tuner resolved.
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

    // A pair is only ever handed back to the device that created it. An event
    // is owned by the device current when hipEventCreateWithFlags() ran, and
    // recording it onto a stream belonging to another one is refused with
    // hipErrorInvalidResourceHandle -- measured on gfx950, and not something
    // hipSetDevice around the record can repair.
    //
    // Nothing downstream would report that cleanly either. A refused record
    // leaves the pair holding its previous timestamps, and hipEventQuery still
    // says complete, so hipEventElapsedTime returns a stale duration. The
    // inversion recycleEvents() applies is what saves the measurement: the
    // stale value comes back negative and the sign test drops it. The sample
    // is wrong only in sign, by luck of the ordering, and a pair pooled in its
    // natural orientation would have returned a plausible positive number from
    // a different device's kernel.
    //
    // A new pair is only created while the device holds fewer than
    // livePairCap() live ones. Past that, a measurement can still start on a
    // pair some harvest handed back, or one reclaimFinishedPairs() reads back
    // from another problem's finished launch, and otherwise it does not start.
    //
    // A new pair is recorded once before it is handed out, stop first and then
    // start, on the stream the launch is about to use. That puts it in the
    // state every pooled pair is already in -- recorded, and reading
    // non-positive until a launch records it again -- so the sign test needs
    // no case for a pair HIP has never seen recorded. Reading such a pair is
    // the one way hipEventElapsedTime fails with an error that stays in the
    // caller's thread (see readPair()), and a launch can leave one: its start
    // record swallowed, or the launch failing before it records anything. The
    // two records cost a pair's creation, which happens at most livePairCap()
    // times per device, and nothing on any later use.
    //
    // A stream on another device than the current one could not take the
    // records, and neither could the launch; such a call is declined before
    // anything is created rather than left to fail.
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
            // Pooling a lone handle would let it be paired with one from an
            // unrelated measurement, whose timestamps say nothing about either.
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

            // This library's error, not the caller's, unless the caller's was
            // there first; see readPair().
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

    // Inverted on purpose, and only ever reached for a pair a launch really did
    // record: reusing it without recording it then reads the negative of the
    // duration it last measured, which harvestPendingImpl()'s sign test drops.
    //
    // device is the one the pair was created on, carried through
    // PendingMeasurement rather than read from hipGetDevice() here, since the
    // thread harvesting is not necessarily the thread that measured.
    void OnlineTuner::recycleEvents(hipEvent_t start, hipEvent_t stop, int device)
    {
        if(start && stop)
            m_pairs[device].push_back({stop, start});
    }

    // Retiring instead of pooling is what closes the window a rejected sample
    // used to leave open: a pair that came back to the pool after a rejection
    // would be inverted twice, and a pair in its original orientation that no
    // launch has re-recorded reads its old duration back as a positive time
    // that no guard here can distinguish from a real one.
    //
    // hipEventDestroy() does not block on a recorded-but-unretired event, it
    // defers the release, so this stays off the critical path. Its own failure
    // is not actionable and must not reach the caller's matmul.
    //
    // device is the one the pair was created on, whose livePairCap() count
    // the pair was charged to.
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
