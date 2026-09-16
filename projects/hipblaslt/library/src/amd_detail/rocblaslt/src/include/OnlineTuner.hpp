// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace rocblaslt
{
    /**
 * @brief Explore-then-cache kernel selection on top of the Origami ranking.
 *
 * The first topK() * repeats() times a problem is seen, each of the top-K
 * ranked candidates is dispatched in turn and timed on the GPU; once every
 * candidate has been sampled the measured winner is pinned for every later
 * call with that problem. statistic() decides how a candidate's repeats are
 * reduced to the one score the winner is chosen on.
 *
 * Timing is deferred-read. beginMeasurement() hands back an event pair for the
 * caller to wrap the launch with, and the elapsed time is only read on a later
 * visit to the same problem, once hipEventQuery() reports the stop event
 * complete. A whole exploration may be outstanding at once, so a caller that
 * enqueues the same problem back to back without synchronising still explores
 * at dispatch rate rather than at queue-drain rate. No entry point here ever
 * waits on the GPU.
 *
 * Every entry point is a branch on a member flag when
 * HIPBLASLT_ORIGAMI_ONLINE_TUNE_TOP_K is unset, so the feature costs nothing
 * when it is off.
 */
    class OnlineTuner
    {
    public:
        /**
     * @brief How a candidate's repeated samples are reduced to one score.
     *
     * Median reports the typical time under whatever interference the run
     * carries; Min reports the best time the candidate was seen to achieve.
     * Selected by HIPBLASLT_ORIGAMI_ONLINE_TUNE_STAT.
     */
        enum class Statistic : uint8_t
        {
            Median = 0,
            Min    = 1
        };

        static OnlineTuner& getInstance()
        {
            static OnlineTuner gInstance;
            return gInstance;
        }

        // copy constructor
        OnlineTuner(const OnlineTuner&) = delete;
        // assignment operator
        OnlineTuner& operator=(const OnlineTuner&) = delete;

        bool enabled() const
        {
            return m_enabled;
        }

        int topK() const
        {
            return m_topK;
        }

        int repeats() const
        {
            return m_repeats;
        }

        Statistic statistic() const
        {
            return m_statistic;
        }

        /**
     * @brief Pick which of the ranked candidates should run next.
     *
     * Registers the problem's candidate list on first sight. Returns the
     * position within rankedSolutionIndices to promote to the front, or -1 to
     * leave the caller's ordering alone.
     */
        int selectCandidate(size_t problemKey, const std::vector<int>& rankedSolutionIndices)
        {
            if(!m_enabled)
                return -1;

            return selectCandidateImpl(problemKey, rankedSolutionIndices);
        }

        /**
     * @brief Read back every outstanding measurement the GPU has finished.
     * Leaves the unfinished ones outstanding.
     */
        void harvestPending(size_t problemKey)
        {
            if(!m_enabled)
                return;

            harvestPendingImpl(problemKey);
        }

        /**
     * @brief Claim an event pair for the imminent launch of solutionIndex.
     *
     * On true, start and stop must both be recorded around that launch,
     * otherwise the sample is dropped on the next harvest and the pair is
     * retired rather than pooled again. On false neither argument is touched
     * and nothing is owed. Problems that selectCandidate() has not registered
     * are never measured.
     */
        bool beginMeasurement(size_t      problemKey,
                              int         solutionIndex,
                              hipEvent_t& start,
                              hipEvent_t& stop)
        {
            if(!m_enabled)
                return false;

            return beginMeasurementImpl(problemKey, solutionIndex, start, stop);
        }

    private:
        // A pooled pair, kept whole so two handles from different measurements
        // can never be paired up with each other. Pairs in the pool obey one
        // invariant, which is what makes an unrecorded reuse detectable:
        // a pair is only ever returned to the pool after a launch recorded it
        // and the sample was accepted, and it is returned inverted, so reusing
        // it without recording it reads a negative elapsed time.
        struct EventPair
        {
            hipEvent_t m_start = nullptr;
            hipEvent_t m_stop  = nullptr;
        };

        // One launch the GPU has not been observed to finish yet.
        struct PendingMeasurement
        {
            hipEvent_t m_start     = nullptr;
            hipEvent_t m_stop      = nullptr;
            int        m_candidate = -1;
        };

        // m_pending holds the outstanding measurements in launch order, capped
        // at inFlightCap(). m_gaveUp means exploration ran past its visit
        // budget: no further launch is measured, but m_pending is still drained
        // before a winner is picked, so samples already paid for are not thrown
        // away. m_declined counts launches refused because the cap was full.
        struct ProblemState
        {
            std::vector<int>                m_candidates;
            std::vector<std::vector<float>> m_samples;
            std::vector<int>                m_issued;
            std::vector<PendingMeasurement> m_pending;
            int                             m_calls    = 0;
            int                             m_declined = 0;
            int                             m_winner   = -1;
            bool                            m_gaveUp   = false;
            bool                            m_resolved = false;
        };

        OnlineTuner();
        ~OnlineTuner();

        int  selectCandidateImpl(size_t problemKey, const std::vector<int>& rankedSolutionIndices);
        void harvestPendingImpl(size_t problemKey);
        bool beginMeasurementImpl(size_t      problemKey,
                                  int         solutionIndex,
                                  hipEvent_t& start,
                                  hipEvent_t& stop);

        // The remainder require m_mutex, held shared where they only read and
        // exclusively where they mutate state or the event pool.
        void registerProblem(size_t                  problemKey,
                             ProblemState&           state,
                             const std::vector<int>& rankedSolutionIndices);
        int  nextCandidate(const ProblemState& state) const;
        int  measurableCandidate(const ProblemState& state, int solutionIndex) const;
        int  inFlightCap(const ProblemState& state) const;
        int  visitBudget(const ProblemState& state) const;
        float score(const std::vector<float>& samples) const;
        void  resolve(size_t problemKey, ProblemState& state);
        bool  acquireEvents(hipEvent_t& start, hipEvent_t& stop);
        void  recycleEvents(hipEvent_t start, hipEvent_t stop);
        void  retireEvents(hipEvent_t start, hipEvent_t stop);

        bool      m_enabled   = false;
        int       m_topK      = 0;
        int       m_repeats   = 0;
        bool      m_verbose   = false;
        Statistic m_statistic = Statistic::Median;

        std::unordered_map<size_t, ProblemState> m_problems;
        std::vector<EventPair>                   m_pairs;
        std::shared_timed_mutex                  m_mutex;
    };
} // namespace rocblaslt
