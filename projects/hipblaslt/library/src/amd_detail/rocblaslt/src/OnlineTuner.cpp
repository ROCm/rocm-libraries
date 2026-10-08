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
        // Slack over the candidates * repeats launches exploration needs. A
        // caller is free to ignore the rotation selectCandidate() asks for, so
        // without a ceiling a problem could stay in the exploring state, and on
        // the write lock, for the life of the process.
        constexpr int c_visitBudgetFactor = 8;

        // Slack over the visit budget before a problem NOBODY is exploring is
        // given up on. Deliberately much larger, because what it bounds is a
        // different thing: see ProblemState::m_dispatchesSinceSelection. The
        // counter it caps is advanced by every dispatch under the key,
        // including a merged sibling's, and only the registrant's own visits
        // reset it -- so this has to leave room for however much sibling
        // traffic separates two of the registrant's visits, or heavy sibling
        // traffic would end a healthy exploration early. Overshooting costs an
        // abandoned problem a bounded run of cheap locked dispatches before it
        // short-circuits for the life of the process; undershooting costs a
        // live exploration its samples.
        constexpr int c_abandonBudgetFactor = 16;

        constexpr const char* c_tracePrefix = "[hipblaslt-online-tune]";

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
        m_coldCalls = std::max(config.coldCalls, 0);
        m_verbose   = config.verbose;

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

        m_problems.clear();
        m_resolutionPool.clear();
        m_winnerPool.clear();

        applyConfig();
    }

    // The pooled events are deliberately not destroyed. This is a function-local
    // static, so its destructor competes with the HIP runtime's own teardown, and
    // hipEventDestroy() after that teardown faults. The process is exiting, so
    // leaking the handles costs nothing.
    OnlineTuner::~OnlineTuner() {}

    /**
     * @brief The rule this function keeps, stated once.
     *
     * Five guards turn a visit away here -- resolved, capturing, cold,
     * retained-on-first-sight, merged sibling -- and each is argued where it
     * stands. What none of them says, and what two earlier defects
     * were, is how they relate to the state this function
     * mutates. Both were a mutation sitting above a guard that should have
     * refused it: the give-up counter reset above the merged-sibling guard, so
     * a sibling exempt from spending the budget still cleared the counter
     * exploration needs in order to end; and the offer dropped above the same
     * guard, so a sibling cancelled an offer the registrant had not consumed.
     * This change is the first of the two; the offer is still a slot per
     * problem below and is still dropped above both markers.
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
     *    cold visit and a retained caller's first sight are own visits that do
     *    neither: they are the library being asked, which is what the counter
     *    measures, without a probe going out, which is what the budget pays
     *    for. That is why the two answers are not the same answer.
     *
     * The code below is laid out in exactly those two steps, with a marked line
     * for each and no write to this problem's state above either of them --
     * except registerProblem(), which creates the registrant rather than
     * spending anything of its, and the offer drop, which is the offer defect
     * and is marked where it stands.
     *
     * A SIXTH GUARD GOES ABOVE MARKER A IF IT SAYS "NOT THIS EXPLORATION'S",
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
        if(capturing)
            return -1;

        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

        ProblemState& state = m_problems[problemKey];

        // Dropped before anything else can hand out a new one. An offer is good
        // for the single launch the visit it was made to is about to issue, and
        // this is a different visit; one the caller never dispatched must not be
        // left lying around for the next arrival to claim.
        //
        // KNOWN VIOLATION OF THE RULE ABOVE, and the one thing left above the
        // markers that is not registration: a merged sibling's visit drops an
        // offer that is not its to drop, so it cancels one the registrant has
        // not consumed. The slot cannot express the pairing it is for --
        // the offer defect -- so moving this line is not the fix and the fix is
        // not this change's.
        state.m_offered = -1;

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

        // The first sight of a problem by a caller that keeps the answer. Read
        // before registerProblem() runs, because registering is precisely what
        // stops the NEXT such query being a first sight.
        const bool deferred = state.m_candidates.empty() && !cold && retained;

        if(cold)
            ++state.m_coldCalls;
        else if(state.m_candidates.empty())
        {
            // The first sight of a problem by a caller that keeps the answer.
            //
            // A probe is a kernel the rotation does not believe is best,
            // dispatched because the next visit reconsiders it. A caller
            // holding the answer decides for itself whether there is a next
            // visit, and the one that fetches a heuristic once and dispatches
            // the algo for the life of the process never comes back -- so the
            // probe it was handed is permanent, exactly as a captured graph's
            // is, and there is no later visit to correct it.
            //
            // Which kind of caller this is cannot be known on the first query
            // and does not have to be: asking again for the same problem is
            // the evidence, and it is evidence the caller supplies by its own
            // behaviour. So the first query registers the problem and is
            // answered by the ranking it brought, and a second one is an
            // ordinary visit. A caller that queries per call therefore still
            // drives a full exploration, one visit later than before; a caller
            // that queries once is never offered anything it cannot take back.
            //
            // Registering here rather than returning before it is what makes
            // the second query an ordinary visit instead of another first one,
            // and it is also what gives the shape a state the measurement hook
            // can bring to an end if the second query never comes.
            //
            // Deliberately not a blanket refusal of every heuristic query, the
            // way a capture is refused. A capture is known not to come back; a
            // heuristic query is not, and refusing all of them would withdraw
            // online tuning from every application that asks the library for a
            // kernel per call -- which is half of what the feature is for.
            //
            // The refusal itself is the `deferred` branch past marker A, not a
            // return from here: a first sight is still the library being
            // asked, so it still says the shape has not been abandoned.
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
        // is not this exploration's writes none of this exploration's state. Reached only when there is a registrant
        // to compare against -- the two branches above are the cases where
        // there is not.
        else if(!state.m_resolved && state.m_problem != problem)
            return -1;

        // ------------------------------------------------------------------
        // MARKER A. Past here the visit is this exploration's own.
        // ------------------------------------------------------------------

        // Selection has visited, so the run of dispatches the registrant had
        // not made ends here. Only an unbroken run of them says the library has
        // stopped being asked by the one caller whose visits can finish this.
        state.m_dispatchesSinceSelection.store(0, std::memory_order_relaxed);

        // Own visits that do not advance the rotation: still the library being
        // asked, which is all marker A's counter measures, but no probe goes
        // out so no visit of the budget is spent and no offer is carried.
        if(cold || deferred)
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
            state.m_offered = candidate;

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

        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

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
            // Records the GPU has not finished are compacted to the front, which
            // keeps the launch order the rest of this loop reports in.
            if(hipEventQuery(pending.m_stop) != hipSuccess)
            {
                state.m_pending[outstanding++] = pending;
                continue;
            }

            // A pair the caller never recorded still queries as complete, so
            // the elapsed time is the only evidence that the launch happened.
            // Every way of missing one reads as non-positive, which is why the
            // sign test is load-bearing and not defensive: a fresh pair fails
            // with hipErrorInvalidHandle, a pooled pair is inverted so it reads
            // the negative of its last duration, and a launch that recorded
            // start and then failed reads an older stop against a newer start.
            float      milliseconds = 0.0f;
            const bool timed
                = hipEventElapsedTime(&milliseconds, pending.m_start, pending.m_stop) == hipSuccess
                  && std::isfinite(milliseconds) && milliseconds > 0.0f;

            // Only a pair that produced a sample is known to have been
            // recorded, so only that pair is safe to hand out again.
            if(timed)
            {
                recycleEvents(pending.m_start, pending.m_stop, pending.m_device);
                state.m_samples[pending.m_candidate].push_back(milliseconds * 1000.0f);
            }
            else
                retireEvents(pending.m_start, pending.m_stop);

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

        int device = 0;
        if(!acquireEvents(start, stop, device))
            return false;

        // Spent. The offer buys one sample, so a second launch claiming the
        // same one would issue a candidate twice on a single visit of it, and a
        // launch arriving after the caller's own is by definition not the one
        // selection asked for.
        state.m_offered = -1;

        state.m_pending.push_back({start, stop, candidate, device});
        ++state.m_issued[candidate];

        return true;
    }

    void OnlineTuner::registerProblem(size_t                  problemKey,
                                      size_t                  problem,
                                      ProblemState&           state,
                                      const std::vector<int>& rankedSolutionIndices)
    {
        const size_t count
            = std::min(rankedSolutionIndices.size(), static_cast<size_t>(m_topK));

        // Whoever registers owns the exploration, for the same reason the first
        // caller to offer a winner object owns that record: a key covers
        // problems that rank and run differently, and letting a later one take
        // it over would only make the two take turns discarding each other's
        // samples.
        state.m_problem = problem;

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

    // The launch selectCandidate() asked for on this visit, and nothing else.
    // Searching the candidate list for the index instead would accept a merged
    // sibling's own pick, which is a different problem's launch wearing a
    // candidate's number; see ProblemState::m_offered. The issued count is
    // re-tested because nextCandidate() picked the offer, and under contention
    // another visit's offer can have overwritten it in between -- a dropped
    // sample, never a double-issued one.
    int OnlineTuner::measurableCandidate(const ProblemState& state, int solutionIndex) const
    {
        if(state.m_resolved || state.m_gaveUp || state.m_offered < 0)
            return -1;

        if(state.m_candidates[state.m_offered] != solutionIndex
           || state.m_issued[state.m_offered] >= m_repeats)
            return -1;

        return state.m_offered;
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

        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

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
        // the time was taken on. See PinnedWinner::m_tuningKey.
        //
        // The state is in m_problems, which never erases, and the resolution
        // was published from it, so the lookup finds it whenever the key is
        // one this tuner resolved.
        const auto registered = m_problems.find(resolved.m_key);
        if(registered == m_problems.end() || registered->second.m_problem != problem)
            return nullptr;

        auto entry                 = std::make_unique<PinnedWinner>();
        entry->m_solution          = solution;
        entry->m_problem           = problem;
        entry->m_requiredWorkspace = requiredWorkspace;
        entry->m_searchWorkspace   = searchWorkspace;
        entry->m_tuningKey         = tuningKey;

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
    bool OnlineTuner::acquireEvents(hipEvent_t& start, hipEvent_t& stop, int& device)
    {
        if(hipGetDevice(&device) != hipSuccess)
            return false;

        std::vector<EventPair>& pool = m_pairs[device];

        if(!pool.empty())
        {
            const EventPair pair = pool.back();
            pool.pop_back();

            start = pair.m_start;
            stop  = pair.m_stop;

            return true;
        }

        EventPair pair;
        if(hipEventCreateWithFlags(&pair.m_start, hipEventDefault) != hipSuccess)
            return false;

        if(hipEventCreateWithFlags(&pair.m_stop, hipEventDefault) != hipSuccess)
        {
            // Pooling a lone handle would let it be paired with one from an
            // unrelated measurement, whose timestamps say nothing about either.
            retireEvents(pair.m_start, nullptr);
            return false;
        }

        start = pair.m_start;
        stop  = pair.m_stop;

        return true;
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
    void OnlineTuner::retireEvents(hipEvent_t start, hipEvent_t stop)
    {
        if(start)
            static_cast<void>(hipEventDestroy(start));

        if(stop)
            static_cast<void>(hipEventDestroy(stop));
    }
} // namespace rocblaslt
