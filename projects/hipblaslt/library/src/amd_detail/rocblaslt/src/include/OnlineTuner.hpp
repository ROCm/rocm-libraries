// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// For the tuning file's problem key. Touches no device, solution library or
// logger, so the tuner stays compilable and testable on its own.
#include "TuningCacheStore.hpp"

// Only carried through a shared_ptr, so the tuner needs no solution library.
namespace TensileLite
{
    class ContractionSolution;
}

// Defined only by the tuner's unit tests, which read private state.
struct OnlineTunerTestAccess;

namespace rocblaslt
{
    /**
     * @brief The tuner's settings, supplied by whoever links it.
     *
     * The tuner reads no environment itself. Inside hipBLASLt the settings come
     * from HIPBLASLT_TUNING_MODE and the HIPBLASLT_TUNING_* variables, read
     * through the library's secure accessor; see onlineTunerConfig().
     */
    struct OnlineTunerConfig
    {
        // Off by default; then every entry point costs one branch.
        bool enabled = false;

        // Ranked-prefix depth. Below two there is nothing to choose between,
        // and the tuner stays off.
        int topK = 0;

        // Timed launches per candidate.
        int repeats = 1;

        // Visits of a problem that run the ranking's own pick untimed before
        // exploration starts. Counted in visits, not launches.
        int coldCalls = 0;

        /**
         * The default live-pair cap.
         *
         * A fixed number of pairs rather than a multiple of topK * repeats: a
         * pair costs a little host memory and two runtime handles whatever the
         * search depth, and scaling the cap by the depth would let a deep
         * search lift it by orders of magnitude. Sized to
         * OnlineTuner::c_resolutionSlots, the number of distinct problems a
         * process is expected to explore, so one-shot shapes alone do not run
         * a process inside that expectation out of pairs.
         */
        static constexpr int c_defaultLivePairCap = 4096;

        // The most event pairs kept alive on one device; see
        // OnlineTuner::livePairCap(). Below one counts as one.
        int livePairCap = c_defaultLivePairCap;

        bool verbose = false;

        /**
         * Called once per device, with no lock held, the first time a launch
         * there goes untimed because every pair on the device is in flight at
         * livePairCap(). Optional. A callback because the tuner links no
         * logger; hipBLASLt passes TensileLite::noteOnlineEventPairCap().
         */
        void (*livePairCapReached)(int device, int cap) = nullptr;
    };

    /**
     * @brief The configuration the tuner is built with.
     *
     * Defined by whoever links the tuner, because the mapping needs the tuning
     * mode and TuningPolicy, which live with the secure environment accessor.
     * Called from the singleton's constructor, on first use.
     */
    OnlineTunerConfig onlineTunerConfig();

    /**
     * @brief Explore-then-cache kernel selection on top of the ranked
     * candidates.
     *
     * The first topK() * repeats() times a problem is dispatched, each of the
     * top-K ranked candidates is launched in turn and timed on the GPU. Once
     * every candidate has been sampled, the one with the smallest sample is
     * pinned for every later call with that problem.
     *
     * Timing is read back lazily. beginMeasurement() hands out an event pair
     * for the caller to record around the launch, and the elapsed time is read
     * on a later visit of the same problem, once the stop event has completed.
     * A whole exploration can be outstanding at once, so exploration runs at
     * dispatch rate, and no entry point ever waits on the GPU.
     *
     * SCOPE: WHICH CALLS ARE TUNED. This is the authoritative statement; the
     * hooks in tensile_host.cpp and the comments elsewhere refer here.
     *
     *  - Tuned: hipblasLtMatmul with algo == nullptr. The library chooses the
     *    kernel (SolutionLookup::PerDispatch) and launches it in the same call
     *    on the same thread, so a probe offered at selection is paired exactly
     *    with the launch that measures it.
     *  - Not tuned: hipblasLtMatmul with an algo the caller supplies, from any
     *    source. It is launched as given, never measured and never harvested.
     *  - Not tuned: hipblasLtMatmulAlgoGetHeuristic (SolutionLookup::Heuristic).
     *    It is answered with the measured winner if there is one and its normal
     *    ranking otherwise, is never offered a probe, and registers nothing. So
     *    the GetHeuristic-then-hipblasLtMatmul(algo) flow is not tuned; it is
     *    served only what null-algo dispatches of the problem, or the cache
     *    file, provide.
     *  - Not tuned: the hipblaslt_ext GEMM and grouped GEMM interfaces, and
     *    problems tuningCacheKeyable() refuses.
     *
     * The line is drawn there because a launch of a caller-supplied algo
     * cannot be paired with the selection visit that produced it (see t_offer
     * in OnlineTuner.cpp), and a probe handed to a caller that keeps the algo
     * would never be measured. With info logging on, the first heuristic query
     * and the first caller-supplied algo of a process each log one
     * online-not-tuned line; see TensileLite::noteOnlineScopeEvent().
     * docs/reference/env-variables.rst states the same scope for users.
     */
    class OnlineTuner
    {
    public:
        /**
         * @brief The pinned winner itself, so a resolved call need not fetch a
         * ranking only to pick it out.
         *
         * problem is the problem the winner was measured on, at the granularity
         * the solution library caches rankings at: the problem that registered
         * the exploration (pinWinner() enforces this). A caller whose problem
         * matches would get the very ranking the winner was recorded in, so it
         * can use the winner directly; any other caller under the same
         * resolution key has to check its own ranking, because a key
         * deliberately covers problems that rank differently.
         *
         * requiredWorkspace is that ranking's workspace answer for the winner,
         * so only the caller's own allocation remains to be checked.
         *
         * tuningKey is the same problem in the key the tuning file uses. One
         * resolution key can cover several file keys, and only the measured
         * problem's key may have a row written under it.
         *
         * Built once, at pin time: it costs a pass over the problem and a
         * device query.
         */
        struct PinnedWinner
        {
            std::shared_ptr<TensileLite::ContractionSolution> solution;
            size_t                                            problem           = 0;
            size_t                                            requiredWorkspace = 0;

            // The workspace limit the candidates were filtered by, recorded as
            // a tune row records it: a winner chosen under a small limit says
            // nothing about the kernels the limit ruled out.
            size_t searchWorkspace = 0;

            TensileLite::ProblemOverride tuningKey;
        };

        /**
         * @brief Everything a problem needs once its winner is pinned.
         *
         * Published once under the write lock with a release store, and read
         * without a lock through an acquire load; the key and winner never
         * change afterwards. pinned() is published the same way, at most once.
         * position() is only a hint, checked by callers against the list in
         * hand, so it is relaxed.
         */
        class Resolution
        {
            friend class OnlineTuner;

        public:
            /// The pinned solution index, or -1 if exploration measured nothing.
            int winner() const
            {
                return m_winner;
            }

            /// The winner's score in microseconds, or zero if nothing was
            /// measured. This is the time a recorded row reports.
            float winnerTimeUs() const
            {
                return m_winnerUs;
            }

            /// The winner object, or nullptr until a caller has offered one.
            const PinnedWinner* pinned() const
            {
                return m_pinned.load(std::memory_order_acquire);
            }

            int position() const
            {
                return m_position.load(std::memory_order_relaxed);
            }

            void setPosition(int position) const
            {
                m_position.store(position, std::memory_order_relaxed);
            }

        private:
            size_t                                   m_key      = 0;
            int                                      m_winner   = -1;
            float                                    m_winnerUs = 0.0f;
            mutable std::atomic<int>                 m_position{-1};
            mutable std::atomic<const PinnedWinner*> m_pinned{nullptr};
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

        // Slack over the topK * repeats launches exploration needs, so a
        // caller that ignores the requested rotation cannot keep a problem
        // exploring, and on the write lock, for the life of the process.
        static constexpr int c_visitBudgetFactor = 8;

        // Slack over the visit budget before a problem nobody is exploring is
        // given up on. Much larger, because the counter it bounds also counts
        // merged siblings' dispatches; see
        // ProblemState::m_dispatchesSinceSelection.
        static constexpr int c_abandonBudgetFactor = 16;

        // Size of the resolved table. It is direct-mapped over the key and
        // never rehashes, which is what lets resolution() run without a lock,
        // and its entries live as long as the tuner. Past roughly this many
        // resolved keys, a colliding key evicts the earlier one, which then
        // takes the locked path for good; nothing it is told is wrong. The
        // resolution and winner pools are not capped at this number.
        static constexpr size_t c_resolutionSlots = 4096;

        /**
         * @brief Re-read the configuration and forget every problem. Tests
         * only, like TuningModeSingleton::reloadForTest.
         *
         * Drops the published resolutions, so it must only be called with no
         * other thread inside the library. Outstanding pairs are retired;
         * pooled pairs are kept, since they belong to a device and not to a
         * configuration.
         */
        void reloadForTest();

        int topK() const
        {
            return m_topK;
        }

        int repeats() const
        {
            return m_repeats;
        }

        int coldCalls() const
        {
            return m_coldCalls;
        }

        /**
         * @brief The most event pairs, pooled and outstanding, kept alive on
         * one device.
         *
         * A problem's pairs come back on its next visit, so a shape dispatched
         * once would otherwise keep its pair for the life of the process. At
         * the cap with nothing pooled, the finished launches other problems
         * hold pairs for are read back and their pairs pooled
         * (reclaimFinishedPairs()). Only when every pair on the device is still
         * in flight does beginMeasurement() decline, and the launch then runs
         * as an untuned one would.
         */
        int livePairCap() const
        {
            return m_livePairCap;
        }

        /// Pairs created on device and not yet retired. Takes the lock, so it
        /// is for tests and diagnostics, not the dispatch path.
        int livePairs(int device);

        /// Launches the measurement hook saw under a key selection never
        /// registered, i.e. the two hooks disagreeing on the key. Counted since
        /// construction or the last reloadForTest().
        uint64_t misses() const
        {
            return m_misses.load(std::memory_order_relaxed);
        }

        /**
         * @brief The resolution of a problem, or nullptr.
         *
         * One acquire load and one compare, with no lock. nullptr is not an
         * answer: the problem may still be exploring or its slot may hold
         * another key, and the caller takes the locked path through
         * selectCandidate().
         */
        const Resolution* resolution(size_t problemKey) const
        {
            const Resolution* entry = m_resolutions[problemKey & (c_resolutionSlots - 1)].load(
                std::memory_order_acquire);

            return entry && entry->m_key == problemKey ? entry : nullptr;
        }

        /**
         * @brief Record the winner object of a resolved problem, so later calls
         * need not fetch a ranking to find it.
         *
         * Installed at most once, and only when problem (in selectCandidate()'s
         * terms) is the problem that registered the exploration: only its
         * launches were sampled, so only its tuningKey may be written. Returns
         * the record this call installed, or nullptr; whoever records the
         * winner outside the process must use that return value, since the
         * caller's own pinned() check is taken without the lock. Cannot change
         * which kernel any call runs.
         */
        const PinnedWinner*
            pinWinner(const Resolution&                                        resolved,
                      const std::shared_ptr<TensileLite::ContractionSolution>& solution,
                      size_t                                                   problem,
                      size_t                                                   requiredWorkspace,
                      size_t                                                   searchWorkspace,
                      const TensileLite::ProblemOverride&                      tuningKey);

        /**
         * @brief Pick which of the ranked candidates should run next.
         *
         * Registers the problem's candidate list on first sight. Returns the
         * position within rankedSolutionIndices to promote to the front, or -1
         * to leave the caller's ordering alone.
         *
         * problem identifies the caller at the granularity the solution library
         * caches rankings at. A resolution key covers several such problems;
         * the first to register owns the exploration, and the others are left
         * their own ordering until there is a winner. See
         * ProblemState::m_problem.
         *
         * capturing (the dispatch is being recorded into a HIP graph) and
         * retained (a heuristic query, whose answer the caller keeps) are never
         * offered a probe: they are served the winner if there is one, and
         * otherwise their own ordering, with nothing registered or counted.
         */
        int selectCandidate(size_t                  problemKey,
                            size_t                  problem,
                            const std::vector<int>& rankedSolutionIndices,
                            bool                    capturing = false,
                            bool                    retained  = false)
        {
            if(!m_enabled)
                return -1;

            return selectCandidateImpl(
                problemKey, problem, rankedSolutionIndices, capturing, retained);
        }

        /**
         * @brief Read back every outstanding measurement the GPU has finished.
         * Leaves the unfinished ones outstanding.
         */
        void harvestPending(size_t problemKey)
        {
            if(!m_enabled)
                return;

            // A resolved problem has nothing outstanding and never acquires
            // more, so this is harvestPendingImpl()'s early return without the
            // lock.
            if(resolution(problemKey))
                return;

            harvestPendingImpl(problemKey);
        }

        /**
         * @brief Claim an event pair for the imminent launch of solutionIndex.
         *
         * Succeeds only for the launch selectCandidate() last offered this
         * thread for this problem; see t_offer in OnlineTuner.cpp. On true,
         * start and stop are to be recorded around the launch on stream, and
         * endMeasurement() called once the launch has returned, whether or not
         * it recorded anything. A pair the launch did not record is dropped by
         * the next harvest. On false nothing is touched and nothing is owed.
         *
         * A pair created for this call is first recorded once on stream,
         * inverted, so every pair the tuner reads back has been recorded; see
         * acquireEvents().
         */
        bool beginMeasurement(size_t      problemKey,
                              int         solutionIndex,
                              hipStream_t stream,
                              hipEvent_t& start,
                              hipEvent_t& stop)
        {
            if(!m_enabled)
                return false;

            // measurableCandidate() refuses every launch of a resolved problem,
            // so this is what the locked path would return.
            if(resolution(problemKey))
                return false;

            return beginMeasurementImpl(problemKey, solutionIndex, stream, start, stop);
        }

        /**
         * @brief Hand back the pair beginMeasurement() gave a launch, once the
         * launch has returned. start is the start event it was given.
         *
         * Until then nothing reads, pools or destroys the pair, whichever
         * thread harvests the problem. A pair not yet recorded reads like one
         * whose launch failed to record it, and only the latter may be
         * destroyed. See PendingMeasurement::m_launched.
         */
        void endMeasurement(size_t problemKey, hipEvent_t start)
        {
            if(!m_enabled || !start)
                return;

            endMeasurementImpl(problemKey, start);
        }

    private:
        // A pooled pair, kept whole so handles from different measurements are
        // never paired. A pair is pooled only after a launch recorded it and
        // its sample was accepted, and it is pooled inverted, so reusing it
        // without recording it reads a negative elapsed time.
        struct EventPair
        {
            hipEvent_t m_start = nullptr;
            hipEvent_t m_stop  = nullptr;
        };

        // A launch not yet observed to finish. m_device is the device the pair
        // was created on, so it goes back to that device's pool.
        //
        // m_launched is set by endMeasurement() once the launch has returned.
        // The launch records the pair outside the lock, so until then the pair
        // is not read at all: a harvest by another dispatch of the problem
        // would reject it and destroy it just before it is recorded.
        struct PendingMeasurement
        {
            hipEvent_t m_start     = nullptr;
            hipEvent_t m_stop      = nullptr;
            int        m_candidate = -1;
            int        m_device    = 0;
            bool       m_launched  = false;
        };

        // One problem's exploration.
        //
        // m_pending holds the outstanding measurements in launch order, at most
        // inFlightCap(). m_gaveUp stops new measurements once a budget is
        // spent; m_pending is still drained before a winner is picked.
        // m_declined counts launches refused because a cap was full.
        // m_coldCalls counts the visits before registration, separately from
        // m_calls, so the cold phase cannot spend the visit budget.
        //
        // The resolution key merges problems that differ in epilogue, bias,
        // strides or scaling, all of which change a kernel's runtime, so
        // samples are comparable only if they all come from one problem. Two
        // things ensure it:
        //
        //  - m_problem is the problem that registered, at the solution
        //    library's ranking granularity (equal to PinnedWinner::problem).
        //    Only its visits explore. A merged sibling is offered nothing,
        //    spends none of the budget, installs no record, and is served the
        //    winner through its own ranking.
        //  - The offer: beginMeasurement() times only the launch selection last
        //    offered on the calling thread. A sibling still dispatches its own
        //    pick, often one of this problem's candidates, so matching on the
        //    solution index alone would take its sample. The offer is held per
        //    thread rather than here because one problem can have an offer
        //    outstanding on every thread in the library; see t_offer in
        //    OnlineTuner.cpp. On the measured path selection and launch are
        //    the same call on the same thread, so the pairing is exact. A
        //    caller-supplied algo cannot be paired that way, which is why it is
        //    out of scope.
        //
        // m_registration is a process-wide serial, so an offer names one
        // registration and not just a key. A problem registers once and is
        // never erased, so this matters only across reloadForTest().
        //
        // m_dispatchesSinceSelection ends a problem whose registrant has
        // stopped dispatching while merged siblings go on dispatching under
        // the key; otherwise it would never resolve, and every sibling dispatch
        // would keep paying for the locked path. It is advanced by every
        // dispatch at the harvest hook and reset only by the registrant's own
        // visits, since a sibling's visits do not move the exploration on. It
        // is bounded by abandonBudget() rather than visitBudget() because it
        // counts sibling traffic too. Atomic because it is advanced under the
        // shared lock.
        struct ProblemState
        {
            std::vector<int>                m_candidates;
            std::vector<std::vector<float>> m_samples;
            std::vector<int>                m_issued;
            std::vector<PendingMeasurement> m_pending;
            size_t                          m_problem      = 0;
            size_t                          m_registration = 0;
            int                             m_coldCalls    = 0;
            int                             m_calls        = 0;
            int                             m_declined     = 0;
            int                             m_winner       = -1;
            float                           m_winnerUs     = 0.0f;
            bool                            m_gaveUp       = false;
            bool                            m_resolved     = false;
            bool                            m_listed       = false;
            std::atomic<int>                m_dispatchesSinceSelection{0};
        };

        OnlineTuner();
        ~OnlineTuner();

        /// Read onlineTunerConfig() into the members. Shared by the
        /// constructor and reloadForTest().
        void applyConfig();

        int  selectCandidateImpl(size_t                  problemKey,
                                 size_t                  problem,
                                 const std::vector<int>& rankedSolutionIndices,
                                 bool                    capturing,
                                 bool                    retained);
        void harvestPendingImpl(size_t problemKey);
        bool beginMeasurementImpl(size_t      problemKey,
                                  int         solutionIndex,
                                  hipStream_t stream,
                                  hipEvent_t& start,
                                  hipEvent_t& stop);
        void endMeasurementImpl(size_t problemKey, hipEvent_t start);

        // The remainder require m_mutex, held shared where they only read and
        // exclusively where they mutate state or the event pool.
        void  registerProblem(size_t                  problemKey,
                              size_t                  problem,
                              ProblemState&           state,
                              const std::vector<int>& rankedSolutionIndices);
        // What readPair() finds: the GPU has not reached the stop yet, the
        // pair holds an accepted sample, or it holds nothing a launch is known
        // to have recorded.
        enum class PairReading
        {
            Outstanding,
            Sample,
            Rejected,
        };

        PairReading readPair(const PendingMeasurement& pending, float& milliseconds) const;
        void        reclaimFinishedPairs(int device, size_t requester);
        int   nextCandidate(const ProblemState& state) const;
        int   measurableCandidate(const ProblemState& state,
                                  size_t              problemKey,
                                  int                 solutionIndex) const;
        int   inFlightCap(const ProblemState& state) const;
        int   visitBudget(const ProblemState& state) const;
        int   abandonBudget(const ProblemState& state) const;
        float score(const std::vector<float>& samples) const;
        void  resolve(size_t problemKey, ProblemState& state);
        void  releaseExploration(ProblemState& state);
        void  publishResolution(size_t problemKey, const ProblemState& state);
        // What acquireEvents() came back with. AtCap is the device holding
        // livePairCap() pairs with every one still in flight; Failed is HIP
        // refusing to create an event or report the device.
        enum class Acquired
        {
            Pair,
            AtCap,
            Failed,
        };

        Acquired acquireEvents(
            hipStream_t stream, hipEvent_t& start, hipEvent_t& stop, int& device, size_t requester);
        void  recycleEvents(hipEvent_t start, hipEvent_t stop, int device);
        void  retireEvents(hipEvent_t start, hipEvent_t stop, int device);

        bool m_enabled   = false;
        int  m_topK      = 0;
        int  m_repeats   = 0;
        int  m_coldCalls = 0;
        int  m_livePairCap = 1;
        bool m_verbose   = false;

        // Source of ProblemState::m_registration. Never reset, so an offer left
        // on a thread before reloadForTest() cannot match a later registration.
        size_t m_registrations = 0;

        std::atomic<uint64_t> m_misses{0};

        // The pools are keyed by device, because hipEventRecord refuses an
        // event created on another device. All the containers below are
        // guarded by m_mutex.
        std::unordered_map<size_t, ProblemState>        m_problems;
        std::unordered_map<int, std::vector<EventPair>> m_pairs;
        std::shared_timed_mutex                         m_mutex;

        // The problems that may hold a pending pair, which is all
        // reclaimFinishedPairs() has to walk. A problem is listed (m_listed)
        // the first time it is handed a pair and dropped by the first pass
        // that finds it holding none. Outside a pass this costs a flag test
        // per measured launch and one append per problem; a pass walks the
        // problems holding pairs, which the cap bounds, plus those listed
        // since the previous pass.
        std::vector<size_t> m_holders;

        // Live pairs per device, pooled and outstanding: what livePairCap()
        // bounds.
        std::unordered_map<int, int> m_livePairs;

        // Devices that have already reported a launch untimed at the cap.
        std::unordered_set<int> m_capReported;
        void (*m_livePairCapReached)(int device, int cap) = nullptr;

        // Never freed while the tuner lives, so a lock-free reader of a
        // published entry cannot be handed a dangling one.
        std::vector<std::unique_ptr<Resolution>> m_resolutionPool;
        std::atomic<const Resolution*>           m_resolutions[c_resolutionSlots] = {};

        // Likewise: at most one per resolved problem. The solutions are owned
        // by the library, so this holds a reference count, not a kernel.
        std::vector<std::unique_ptr<PinnedWinner>> m_winnerPool;

        friend struct ::OnlineTunerTestAccess;
    };
} // namespace rocblaslt
