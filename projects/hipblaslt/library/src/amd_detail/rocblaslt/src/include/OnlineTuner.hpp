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
#include <vector>

// The tuning file's problem key, so a pinned winner can carry the key a cache
// row is written under. Nothing in that header touches a device, the solution
// library or the logger, which is why including it leaves the tuner compilable
// and testable on its own.
#include "TuningCacheStore.hpp"

// Carried through a shared_ptr and never dereferenced here, so the declaration
// is all this needs. Keeping the solution library's headers out is what lets
// the tuner be compiled and tested without one.
namespace TensileLite
{
    class ContractionSolution;
}

namespace rocblaslt
{
    /**
 * @brief The tuner's knobs, as whoever links it supplies them.
 *
 * The tuner reads no environment of its own. Inside hipBLASLt these come from
 * HIPBLASLT_TUNING_MODE and the HIPBLASLT_TUNING_* search settings, read
 * through the same secure accessor the rest of the tuning stack uses, which is
 * only reachable from the library and not from here.
 */
    struct OnlineTunerConfig
    {
        // Off unless the link says otherwise, so the feature stays off by
        // default and every entry point costs one branch on a member flag.
        bool enabled = false;

        // Ranked-prefix depth. Below two there is nothing to choose between,
        // and the tuner stays off however the mode is set.
        int topK = 0;

        // Timed launches per candidate.
        int repeats = 1;

        // Visits of a problem that dispatch the ranking's own pick and are
        // sampled by nothing, so exploration begins on a device the problem has
        // already warmed. Counted in visits, not in launches: each one is a
        // whole call that goes untouched.
        int coldCalls = 0;

        bool verbose = false;
    };

    /**
 * @brief The configuration the tuner is built with.
 *
 * Defined by whoever links the tuner rather than read here, because the
 * mapping needs the tuning mode and TuningPolicy, which live where the secure
 * environment accessor does. Called once, from the singleton's constructor,
 * which is the first use of the tuner and so is after everything it reads can
 * be constructed.
 */
    OnlineTunerConfig onlineTunerConfig();

    /**
 * @brief Explore-then-cache kernel selection on top of the Origami ranking.
 *
 * The first topK() * repeats() times a problem is seen, each of the top-K
 * ranked candidates is dispatched in turn and timed on the GPU; once every
 * candidate has been sampled the measured winner is pinned for every later
 * call with that problem. A candidate's repeats are reduced to one score by
 * taking the smallest of them.
 *
 * Timing is deferred-read. beginMeasurement() hands back an event pair for the
 * caller to wrap the launch with, and the elapsed time is only read on a later
 * visit to the same problem, once hipEventQuery() reports the stop event
 * complete. A whole exploration may be outstanding at once, so a caller that
 * enqueues the same problem back to back without synchronising still explores
 * at dispatch rate rather than at queue-drain rate. No entry point here ever
 * waits on the GPU.
 *
 * Every entry point is a branch on a member flag when the configuration does
 * not switch the tuner on, so the feature costs nothing when it is off.
 */
    class OnlineTuner
    {
    public:
        /**
     * @brief The winner itself, for a caller that would otherwise have to ask
     * the library for a ranked list only to pick one entry out of it.
     *
     * m_problem identifies the problem the winner was offered for, at the
     * granularity the solution library caches its rankings at rather than the
     * granularity resolution() is keyed on. A caller whose problem matches it
     * would be handed the very ranking the winner was recorded in, so it can
     * use the winner without fetching that ranking to look for it; one whose
     * problem does not match has to go and ask, because a key deliberately
     * covers problems that rank differently.
     *
     * m_requiredWorkspace is that same ranking's workspace answer for the
     * winner, which depends only on the solution and the problem and so is
     * fixed alongside them. Only the caller's own allocation still varies, and
     * comparing against it is all that remains of the filter.
     *
     * m_tuningKey is the same problem again, in the key the tuning file is
     * written and replayed under. It is here and not on the Resolution for the
     * reason the two keys differ at all: resolution() is keyed on the fields
     * the prediction model ranks on, which deliberately merges problems that
     * differ in epilogue, bias, strides or scaling, while a file row
     * deliberately separates exactly those. One resolution key can therefore
     * cover several distinct file keys, and this is the one that belongs to the
     * problem this record describes -- the problem whose ranking held the
     * winner where it was recorded, whose hash m_problem is, and whose
     * workspace answer m_requiredWorkspace is. A record is written once and
     * never overwritten, so a merged sibling is served the winner at run time
     * wherever its own ranking confirms it, but is never the key a row is
     * written under, because nothing measured it.
     *
     * Built at pin time and nowhere else: it costs a pass over forty-odd fields
     * and a cached device query, which is affordable once per resolved problem
     * and would not be affordable per call.
     */
        struct PinnedWinner
        {
            std::shared_ptr<TensileLite::ContractionSolution> m_solution;
            size_t                                            m_problem           = 0;
            size_t                                            m_requiredWorkspace = 0;

            // The workspace the candidate list was filtered by, which is the
            // limit of the caller whose ranking the winner was recorded in.
            // Recorded for the same reason a tune row records one: a winner
            // chosen from candidates a small allocation could cover says
            // nothing about the kernels it ruled out.
            size_t m_searchWorkspace = 0;

            TensileLite::ProblemOverride m_tuningKey;
        };

        /**
     * @brief Everything a problem still needs once its winner is pinned.
     *
     * Published once, while the write lock is held, and read from then on
     * without taking anything: the key and the winner never change afterwards,
     * so the release store that publishes the pointer and the acquire load that
     * finds it are the whole synchronisation.
     *
     * pinned() is filled in later, by the first caller to offer the winner
     * object, and is published the same way for the same reason -- a release
     * store under the write lock against an acquire load taking none. It is
     * written at most once, so a reader either sees nothing or sees a record
     * that is complete and will never change, and copying the shared_ptr out of
     * it is a read of an object nothing mutates.
     *
     * position() is the exception and is only a hint -- where the caller found
     * the winner in the list it was offered last time. Callers check it against
     * the list in hand before acting on it, so a wrong value costs one trip down
     * the full path and can never change which kernel is chosen. Nothing is
     * ordered against it, hence relaxed.
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

            /**
             * The winner's score, in microseconds, or zero when exploration
             * measured nothing.
             *
             * Published with the winner and never changed afterwards, so it is
             * read the same way: it is the number a recorded row reports, and
             * the only evidence in the file that the kernel was measured here
             * rather than predicted.
             */
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
     * @brief The pinned winner for a problem, or nullptr while it is still
     * being explored.
     *
     * One acquire load and one compare, taking no lock and doing no work that
     * grows with topK(). A caller holding a Resolution already has its answer,
     * so it can skip the candidate filtering and the search that dominate the
     * per-call cost of a problem there is nothing left to learn about.
     *
     * The table is direct-mapped over the key and a colliding key evicts, so
     * nullptr is not an answer -- it only means the caller must go through
     * selectCandidate() and take the lock, which is what every call did before.
     * Nothing here can be wrong, only absent.
     */
        const Resolution* resolution(size_t problemKey) const
        {
            const Resolution* entry = m_resolutions[problemKey & (c_resolutionSlots - 1)].load(
                std::memory_order_acquire);

            return entry && entry->m_key == problemKey ? entry : nullptr;
        }

        /**
     * @brief Offer a resolved problem the winner object itself, so later calls
     * need not ask the library for a ranking to find it in.
     *
     * Taken from the first caller to offer one and ignored afterwards. A key can
     * cover problems that rank differently, and the one recorded here is the one
     * the winner was measured on; letting a later problem overwrite it would
     * only make the two take turns invalidating each other's record.
     *
     * problem is the caller's own identifier for the problem the winner was
     * offered for, and is handed back unexamined through pinned(). Calling this
     * cannot change which kernel is chosen for any problem: a caller that does
     * not recognise what comes back has lost nothing but the shortcut.
     *
     * tuningKey is that same problem in the tuning file's key, carried for
     * whoever records the winner; see PinnedWinner::m_tuningKey.
     *
     * Returns the record this call installed, or null when it installed
     * nothing -- because another caller got there first, because there was
     * nothing to offer, or because the pool is full. Whoever records the
     * winner outside this process has to know which caller's key was taken,
     * and this is the only answer that is not a race: the check above is a
     * hint taken without the lock, so two callers can both reach here and only
     * one of them leaves a record behind.
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

            // A resolved problem has nothing outstanding and can never acquire
            // any: resolve() only runs with m_pending empty, and beginMeasurement
            // refuses every launch afterwards. So this is the same early return
            // harvestPendingImpl() makes, reached without the lock.
            if(resolution(problemKey))
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

            // measurableCandidate() refuses every launch once the problem is
            // resolved, so this returns what the locked path would. A resolved
            // problem was registered to get here, so skipping that path cannot
            // swallow the miss event either.
            if(resolution(problemKey))
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
        //
        // m_coldCalls counts the visits spent before the candidate list was
        // registered at all, which is the only state a problem still inside
        // coldCalls() carries. It is deliberately not m_calls: the visit budget
        // bounds how long exploration may run, and a problem that has not
        // started exploring must not be able to exhaust it.
        struct ProblemState
        {
            std::vector<int>                m_candidates;
            std::vector<std::vector<float>> m_samples;
            std::vector<int>                m_issued;
            std::vector<PendingMeasurement> m_pending;
            int                             m_coldCalls = 0;
            int                             m_calls     = 0;
            int                             m_declined  = 0;
            int                             m_winner    = -1;
            float                           m_winnerUs  = 0.0f;
            bool                            m_gaveUp    = false;
            bool                            m_resolved  = false;
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
        void  publishResolution(size_t problemKey, const ProblemState& state);
        bool  acquireEvents(hipEvent_t& start, hipEvent_t& stop);
        void  recycleEvents(hipEvent_t start, hipEvent_t stop);
        void  retireEvents(hipEvent_t start, hipEvent_t stop);

        bool m_enabled   = false;
        int  m_topK      = 0;
        int  m_repeats   = 0;
        int  m_coldCalls = 0;
        bool m_verbose   = false;

        std::unordered_map<size_t, ProblemState> m_problems;
        std::vector<EventPair>                   m_pairs;
        std::shared_timed_mutex                  m_mutex;

        // The resolved table is deliberately not the problem map: it is fixed
        // in size and never rehashes, which is what lets resolution() follow a
        // published entry with no lock at all. The entries themselves outlive
        // the process, so a reader can never be handed a dangling one.
        //
        // A power of two so the index is a mask, and far larger than the number
        // of distinct problems a process is expected to see; overshooting only
        // costs the table's own footprint, while a collision costs one problem
        // the locked path it used to take anyway.
        static constexpr size_t c_resolutionSlots = 4096;

        std::vector<std::unique_ptr<Resolution>> m_resolutionPool;
        std::atomic<const Resolution*>           m_resolutions[c_resolutionSlots] = {};

        // Kept alive for the same reason and bounded by the same slot count:
        // at most one per resolved problem, taken from the first caller to
        // offer one. The solutions themselves are owned by the library, which
        // outlives the tuner, so what this holds is a reference count and not
        // the kernel.
        std::vector<std::unique_ptr<PinnedWinner>> m_winnerPool;
    };
} // namespace rocblaslt
