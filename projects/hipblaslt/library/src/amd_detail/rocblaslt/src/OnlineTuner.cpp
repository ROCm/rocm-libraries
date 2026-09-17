// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OnlineTuner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

        // Provenance markers on the register and winner lines: whether a
        // candidate came from the Origami ranking or from a reserved slot.
        constexpr char c_rankedSource   = 'o';
        constexpr char c_equalitySource = 'e';
        constexpr char c_noSource       = '-';

        int envInt(const char* name, int defaultValue)
        {
            const char* env = std::getenv(name);
            if(!env)
                return defaultValue;

            return static_cast<int>(std::strtol(env, nullptr, 0));
        }

        // An unrecognised value keeps the default rather than failing the
        // process, which is why the winner line reports the statistic it
        // actually scored with: that trace field, not the environment, is what
        // an A/B arm should be identified by.
        OnlineTuner::Statistic envStatistic(const char*            name,
                                            OnlineTuner::Statistic defaultValue)
        {
            const char* env = std::getenv(name);
            if(!env)
                return defaultValue;

            if(!std::strcmp(env, "min"))
                return OnlineTuner::Statistic::Min;

            if(!std::strcmp(env, "median"))
                return OnlineTuner::Statistic::Median;

            return defaultValue;
        }

        const char* statisticName(OnlineTuner::Statistic statistic)
        {
            return statistic == OnlineTuner::Statistic::Min ? "min" : "median";
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

        float minimum(const std::vector<float>& samples)
        {
            return *std::min_element(samples.begin(), samples.end());
        }

        bool sameTile(const EqualitySlotCandidate& a, const EqualitySlotCandidate& b)
        {
            return a.m_tileM == b.m_tileM && a.m_tileN == b.m_tileN && a.m_depthU == b.m_depthU;
        }

        // How far a tile's aspect ratio is from the problem's, as the ratio of
        // the two cross products: no logarithm, no signed compare, and exactly
        // 1 when MT_N / MT_M equals n / m. Every term is at least one.
        double aspectDistance(const EqualitySlotCandidate& candidate, size_t m, size_t n)
        {
            const double tileTerm    = static_cast<double>(candidate.m_tileN * m);
            const double problemTerm = static_cast<double>(candidate.m_tileM * n);

            return std::max(tileTerm, problemTerm) / std::min(tileTerm, problemTerm);
        }

        std::ostringstream traceLine(const char* event, size_t problemKey)
        {
            std::ostringstream msg;
            msg << std::fixed << std::setprecision(3);
            msg << c_tracePrefix << " event=" << event << " key=" << problemKey;
            return msg;
        }
    }

    std::vector<size_t>
        orderEqualitySlotCandidates(const std::vector<EqualitySlotCandidate>& pool,
                                    const std::vector<EqualitySlotCandidate>& ranked,
                                    size_t                                    problemM,
                                    size_t                                    problemN)
    {
        const size_t m = std::max<size_t>(1, problemM);
        const size_t n = std::max<size_t>(1, problemN);

        std::vector<size_t> order;
        std::vector<double> distance(pool.size());
        order.reserve(pool.size());

        for(size_t i = 0; i < pool.size(); ++i)
        {
            distance[i] = aspectDistance(pool[i], m, n);

            const bool offered
                = std::any_of(ranked.begin(), ranked.end(), [&](const EqualitySlotCandidate& r) {
                      return r.m_solutionIndex == pool[i].m_solutionIndex || sameTile(r, pool[i]);
                  });

            if(!offered)
                order.push_back(i);
        }

        std::stable_sort(order.begin(), order.end(), [&distance](size_t a, size_t b) {
            return distance[a] < distance[b];
        });

        return order;
    }

    OnlineTuner::OnlineTuner()
    {
        const int topK = envInt("HIPBLASLT_ORIGAMI_ONLINE_TUNE_TOP_K", 0);

        m_topK = topK > 0 ? topK : 0;
        m_repeats
            = std::max(envInt("HIPBLASLT_ORIGAMI_ONLINE_TUNE_REPEATS", c_defaultRepeats), 1);
        m_verbose   = std::getenv("HIPBLASLT_ORIGAMI_ONLINE_TUNE_VERBOSE") != nullptr;
        m_statistic = envStatistic("HIPBLASLT_ORIGAMI_ONLINE_TUNE_STAT", Statistic::Median);
        m_enabled   = m_topK >= 2;

        // Clamped rather than rejected so no value of the knob can produce an
        // exploration set with nothing from the ranking left in it to beat.
        m_equalitySlots
            = std::min(std::max(envInt("HIPBLASLT_ORIGAMI_ONLINE_TUNE_EQUALITY_SLOTS", 0), 0),
                       std::max(m_topK - 1, 0));

        // Read only to label the trace with the arm it was produced under. The
        // merge itself happens at library load, in the Prediction library, and
        // this says nothing about whether any kernel was actually merged -- a
        // problem type with no Equality row is unaffected by the knob.
        m_poolMerged = envInt("TENSILE_MERGE_EQUALITY_POOL", 0) != 0;
    }

    // The pooled events are deliberately not destroyed. This is a function-local
    // static, so its destructor competes with the HIP runtime's own teardown, and
    // hipEventDestroy() after that teardown faults. The process is exiting, so
    // leaking the handles costs nothing.
    OnlineTuner::~OnlineTuner() {}

    int OnlineTuner::selectCandidateImpl(size_t                      problemKey,
                                         const std::vector<int>&     rankedSolutionIndices,
                                         const std::vector<uint8_t>& equalitySourced)
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
            registerProblem(problemKey, state, rankedSolutionIndices, equalitySourced);

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
                recycleEvents(pending.m_start, pending.m_stop);
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

        if(!acquireEvents(start, stop))
            return false;

        state.m_pending.push_back({start, stop, candidate});
        ++state.m_issued[candidate];

        return true;
    }

    void OnlineTuner::registerProblem(size_t                      problemKey,
                                      ProblemState&               state,
                                      const std::vector<int>&     rankedSolutionIndices,
                                      const std::vector<uint8_t>& equalitySourced)
    {
        const size_t count
            = std::min(rankedSolutionIndices.size(), static_cast<size_t>(m_topK));

        state.m_candidates.assign(rankedSolutionIndices.begin(),
                                  rankedSolutionIndices.begin() + count);
        state.m_samples.resize(count);
        state.m_issued.assign(count, 0);

        // The caller's flags cover its whole ranked list, which may be longer
        // than the exploration window; anything past the window is not a
        // candidate and so has no provenance to record. A shorter list than
        // the window leaves the rest unflagged rather than reading past it.
        // Left empty when nothing is flagged, so the common case stores nothing.
        const size_t flagged = std::min(count, equalitySourced.size());

        if(std::any_of(equalitySourced.begin(),
                       equalitySourced.begin() + flagged,
                       [](uint8_t flag) { return flag != 0; }))
        {
            state.m_equalitySource.assign(count, 0);
            std::copy(equalitySourced.begin(),
                      equalitySourced.begin() + flagged,
                      state.m_equalitySource.begin());
        }

        // Nothing to compare against, so the problem is born resolved with no
        // winner and every later call takes the read-only path.
        if(count < 2)
        {
            state.m_resolved = true;
            publishResolution(problemKey, state);
        }
        else
            state.m_pending.reserve(static_cast<size_t>(inFlightCap(state)));

        if(m_verbose)
        {
            std::ostringstream msg = traceLine("register", problemKey);
            msg << " candidates=" << count << " repeats=" << m_repeats << " sols=";
            for(size_t i = 0; i < count; ++i)
                msg << (i ? "," : "") << state.m_candidates[i];
            msg << " eqslots=" << m_equalitySlots << " eqmerge=" << (m_poolMerged ? 1 : 0)
                << " src=";
            for(size_t i = 0; i < count; ++i)
                msg << (i ? "," : "")
                    << (fromEquality(state, static_cast<int>(i)) ? c_equalitySource
                                                                 : c_rankedSource);
            msg << "\n";
            std::cerr << msg.str();
        }
    }

    bool OnlineTuner::fromEquality(const ProblemState& state, int candidate) const
    {
        return candidate >= 0 && static_cast<size_t>(candidate) < state.m_equalitySource.size()
               && state.m_equalitySource[candidate] != 0;
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

    // GPU timing noise is one-sided: contention, clock excursions and cache
    // state make a launch slower than the kernel's floor, never faster. Min
    // therefore estimates what the candidate can achieve and median estimates
    // what it typically achieves under whatever else the machine is doing.
    float OnlineTuner::score(const std::vector<float>& samples) const
    {
        if(m_statistic == Statistic::Min)
            return minimum(samples);

        return median(samples);
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
        state.m_resolved = true;

        publishResolution(problemKey, state);

        if(m_verbose)
        {
            const size_t samples = winnerIndex < 0 ? 0 : state.m_samples[winnerIndex].size();

            std::ostringstream msg = traceLine("winner", problemKey);
            msg << " cand=" << winnerIndex << " sol=" << winner << " us=" << winnerScore
                << " samples=" << samples << " calls=" << state.m_calls
                << " declined=" << state.m_declined
                << " gaveup=" << (state.m_gaveUp ? 1 : 0)
                << " stat=" << statisticName(m_statistic) << " eqslots=" << m_equalitySlots
                << " eqmerge=" << (m_poolMerged ? 1 : 0) << " src="
                << (winnerIndex < 0 ? c_noSource
                                    : (fromEquality(state, winnerIndex) ? c_equalitySource
                                                                        : c_rankedSource))
                << "\n";
            std::cerr << msg.str();
        }
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
     * Once the pool is full, resolved problems keep working and simply keep
     * taking the locked path.
     */
    void OnlineTuner::publishResolution(size_t problemKey, const ProblemState& state)
    {
        if(m_resolutionPool.size() >= c_resolutionSlots)
            return;

        auto entry      = std::make_unique<Resolution>();
        entry->m_key    = problemKey;
        entry->m_winner = state.m_winner;

        const Resolution* published = entry.get();
        m_resolutionPool.push_back(std::move(entry));

        m_resolutions[problemKey & (c_resolutionSlots - 1)].store(published,
                                                                  std::memory_order_release);
    }

    bool OnlineTuner::acquireEvents(hipEvent_t& start, hipEvent_t& stop)
    {
        if(!m_pairs.empty())
        {
            const EventPair pair = m_pairs.back();
            m_pairs.pop_back();

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
    void OnlineTuner::recycleEvents(hipEvent_t start, hipEvent_t stop)
    {
        if(start && stop)
            m_pairs.push_back({stop, start});
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
