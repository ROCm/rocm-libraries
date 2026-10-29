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

// What a resolved problem has stopped holding is not something it can be asked,
// so the test that guards releaseExploration() reaches in. Declared here and
// defined only by that test; nothing in the library names it.
struct OnlineTunerTestAccess;

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
     * pinWinner() holds that last sentence up rather than leaving it to the
     * order callers happen to arrive in. A record is installed only for the
     * problem that registered the exploration, which is also the only problem
     * whose launches were sampled, so m_problem here is ProblemState::m_problem
     * and m_tuningKey is that problem's file key and no other's.
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

        /**
     * @brief Re-read the configuration and forget everything learned under
     * the old one. Tests only, like TuningModeSingleton::reloadForTest.
     *
     * The configuration is read once, in the constructor of a
     * process-lifetime singleton, and that constructor runs on the process's
     * first dispatch -- long before a test that sets the tuning variables in
     * its own setup can have set them. Without this a single-process test
     * suite can only ever observe online tuning as whatever the first
     * dispatch of the binary happened to configure, which is off.
     *
     * Drops the published resolutions along with the problems. They are
     * normally kept for the life of the process precisely so that a reader
     * following a published pointer without a lock can never be handed a
     * dangling one, so this must only be called with no other thread inside
     * the library.
     *
     * The event pool is deliberately left alone. A pooled pair belongs to a
     * device and to the HIP runtime, not to a configuration, and destroying
     * events is what the destructor already declines to do. The pairs the
     * dropped problems still had outstanding are retired, since nothing could
     * reach them again and they would go on counting against livePairCap().
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
         * @brief The most event pairs the tuner keeps alive on one device,
         * pooled and outstanding together.
         *
         * A pair is only handed back by a later harvest of the problem that
         * took it, so a shape dispatched once and never again keeps its pair
         * for the life of the process; with no cold visits that is the
         * ordinary life of a one-shot shape. The cap is what bounds that.
         * Once a device holds this many and none is pooled, no new
         * measurement starts there: beginMeasurement() declines, and the
         * launch runs exactly as an untuned one would.
         *
         * Scaled by topK() * repeats(), which is one whole exploration and
         * the most a single problem can have outstanding, so the configured
         * depth is never what the cap starves; c_livePairExplorations of
         * those is room for that many problems to have their entire
         * exploration in flight on one device at once. At the defaults that
         * is a few hundred pairs, a small, fixed number of runtime handles
         * however many shapes the process goes through.
         */
        int livePairCap() const
        {
            return c_livePairExplorations * m_topK * m_repeats;
        }

        /**
         * @brief Launches the measurement hook asked about under a key the
         * selection hook never registered.
         *
         * The two hooks derive the key separately, so this is how a
         * disagreement between them shows, rather than as a problem that is
         * quietly never measured. Counted since construction or the last
         * reloadForTest().
         */
        uint64_t misses() const
        {
            return m_misses.load(std::memory_order_relaxed);
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
     * offered for, in the same terms selectCandidate() takes it, and is checked
     * against the problem that registered the exploration rather than merely
     * carried. A caller the resolution key merges with that problem measured
     * none of it, so it has no key worth writing a row under; it is handed the
     * winner through its own ranking and leaves no record behind. What comes
     * back is handed on through pinned(), where a later caller compares its own
     * identifier against it. Calling this cannot change which kernel is chosen
     * for any problem: a caller that does not recognise what comes back has
     * lost nothing but the shortcut.
     *
     * tuningKey is that same problem in the tuning file's key, carried for
     * whoever records the winner; see PinnedWinner::m_tuningKey.
     *
     * Returns the record this call installed, or null when it installed
     * nothing -- because this caller is not the problem that was measured,
     * because another caller got there first, or because there was nothing to
     * offer. Whoever records the winner outside
     * this process has to know which caller's key was taken, and this is the
     * only answer that is not a race: the check above is a hint taken without
     * the lock, so two callers can both reach here and only one of them leaves
     * a record behind.
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
     *
     * problem identifies the caller at the granularity the solution library
     * caches its rankings at, and is what decides whose exploration this is.
     * One resolution key covers several such problems by design; the first to
     * register owns the exploration and every other is left its own ordering
     * until there is a measured winner to serve it. See
     * ProblemState::m_problem.
     *
     * capturing says the caller's dispatch is being recorded into a HIP graph
     * rather than executed. Such a caller is never offered a probe candidate;
     * it is served a measured winner if there is one, and otherwise left its
     * own ordering. See selectCandidateImpl() for why.
     *
     * retained says the caller keeps this answer and dispatches it itself,
     * rather than being asked again on its next dispatch -- a heuristic query.
     * Such a caller is never the one a problem's first probe is offered to,
     * because nothing here can know whether it will ever ask again. Asking a
     * second time is what says it will, and from then on it is an ordinary
     * visit. See selectCandidateImpl() for why that, and not a blanket
     * refusal.
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
     *
     * Nor is any dispatch other than the one selectCandidate() last asked THIS
     * THREAD for. The solution index alone does not identify it: a problem the
     * resolution key merges with the one being explored is left its own
     * ordering, and its own pick is very often a candidate of that exploration
     * that still owes samples, so timing it would credit one problem with a
     * launch of another. The offer is held on the thread selection answered
     * rather than on the problem, because one problem can have as many offers
     * outstanding as there are threads in the library; see the offer paragraph
     * on ProblemState for what that does and does not close.
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

        // One launch the GPU has not been observed to finish yet. m_device is
        // the device the pair was created on, carried so recycleEvents() can
        // put it back where it came from.
        struct PendingMeasurement
        {
            hipEvent_t m_start     = nullptr;
            hipEvent_t m_stop      = nullptr;
            int        m_candidate = -1;
            int        m_device    = 0;
        };

        // m_pending holds the outstanding measurements in launch order, capped
        // at inFlightCap(). m_gaveUp means exploration ran past its visit
        // budget: no further launch is measured, but m_pending is still drained
        // before a winner is picked, so samples already paid for are not thrown
        // away. m_declined counts launches refused because a cap was full:
        // this problem's inFlightCap(), or its device's livePairCap().
        //
        // m_coldCalls counts the visits spent before the candidate list was
        // registered at all, which is the only state a problem still inside
        // coldCalls() carries. It is deliberately not m_calls: the visit budget
        // bounds how long exploration may run, and a problem that has not
        // started exploring must not be able to exhaust it.
        //
        // m_problem and the offer are what make the samples below comparable
        // with one another, which is the one thing resolve() assumes and the
        // key alone does not give. The resolution key deliberately merges
        // problems differing in epilogue, bias, strides or scaling -- all of
        // which move a kernel's runtime -- so an unattributed sample set is a
        // set of measurements of different things, and picking the smallest of
        // them says which sibling dispatched rather than which kernel is
        // faster. The two answer that in the two places it can go wrong, and
        // each is needed:
        //
        // m_problem is the problem that registered the candidate list, at the
        // granularity the solution library caches its rankings at -- the same
        // quantity PinnedWinner::m_problem carries, and equal to it by
        // construction. Exploration belongs to whoever registered: a merged
        // sibling is offered no candidate, spends none of the visit budget and
        // installs no record, and is served the measured winner through its own
        // ranking once there is one.
        //
        // The offer is the candidate selectCandidate() last asked a caller to
        // dispatch, and the only launch beginMeasurement() will time. Refusing
        // the sibling a candidate is not enough on its own, because it still
        // dispatches its own ranking's pick, and that is very often a candidate
        // of this problem's that still owes samples -- matching on the solution
        // index alone would take the sample anyway.
        //
        // IT IS DELIBERATELY NOT A FIELD HERE. What it has to express is a
        // pairing between two events -- this selection visit, that launch --
        // and the number of those outstanding under one resolution key at any
        // moment is the number of threads in the library, not one.
        // hipblasLtMatmul is documented as callable from many threads and
        // streams, so a single slot per problem is both losable, because the
        // next visit overwrites it, and claimable by a launch it was never
        // made to. That is a cardinality mismatch and not a window that can be
        // narrowed. The offer therefore lives on the thread selection answered
        // -- see t_offer in OnlineTuner.cpp -- and m_registration below is what
        // ties it to one registration of one problem.
        //
        // What that delivers, exactly, because the previous version of this
        // comment claimed a pairing the code did not keep. On the dispatch path
        // (hipblasLtMatmul with algo == nullptr) selection and launch are the
        // same library call on the same thread with nothing of this tuner's in
        // between, and every selection visit a thread makes drops the offer it
        // was holding, so the offer a launch claims is the one that call's own
        // visit made and no caller on any thread can take it. That is exact.
        //
        // On the heuristic path it is not. The application holds the algo and
        // dispatches it when and where it likes, so a launch can arrive with no
        // selection visit of that thread in between, and the library has
        // nothing to pair with beyond "this thread was offered this candidate
        // for this registration". A merged sibling dispatched from that same
        // thread, with a caller-supplied algo, can still claim the offer. It
        // takes one thread holding two heuristic answers under one resolution
        // key and dispatching the older of them, and it costs one sample of one
        // candidate. Closing it needs the offer carried ON the algo, which is a
        // public type and so a decision and not a fix.
        //
        // m_queriedOnly says nothing but heuristic queries has visited this
        // registration as its owner. The identity is the solution library's,
        // and a query and a dispatch of one problem do not share it -- the
        // query ranks under a dummy beta and the preference's workspace
        // ceiling -- so while it holds, the first dispatch to arrive with no
        // sample taken takes the registration over rather than being turned
        // away as a merged sibling. See the merged-sibling guard in
        // selectCandidateImpl().
        //
        // m_registration is a process-wide serial taken at registerProblem(),
        // so an offer names one registration and not merely one key. Without it
        // an offer outstanding across reloadForTest(), or across a problem that
        // resolved and was re-registered, could be matched against a candidate
        // list it was never made from.
        //
        // m_dispatchesSinceSelection is how a problem the library is never
        // asked about again still comes to an end.
        //
        // m_calls only moves where selection visits, and selection is only
        // reached while the library is the one choosing the kernel. A caller
        // dispatching an algo it fetched once goes straight past it -- so such
        // a problem sits one visit short of its budget for the life of the
        // process: m_gaveUp never becomes true, nextCandidate() keeps naming a
        // candidate nothing will ever issue, and neither resolve condition
        // fires. Nothing is wrong with the answers it gives, but resolution()
        // returns null for ever, and that is what the two entry points check
        // before taking the lock. Every one of those dispatches therefore pays
        // two shared acquisitions and two map lookups to be told there is
        // nothing to do, for a shape that can never be tuned.
        //
        // Counted at the measurement hook, which is the one every dispatch
        // reaches, and reset by a visit of the problem that owns the
        // exploration -- so it measures the run of dispatches THE REGISTRANT
        // has not made, not the run nobody has made. The distinction is the
        // whole mechanism: a merged sibling is exempt from spending the visit
        // budget, so letting its visits reset this counter let it hold an
        // exploration the registrant had abandoned open for ever, taking the
        // exclusive lock on every dispatch -- the very failure this counter was
        // added to end.
        //
        // Bounded by abandonBudget() and not by visitBudget(), because it is no
        // longer counting an exploration's own progress. It counts everyone's
        // dispatches, including a busy sibling's, so a ceiling tight enough to
        // bound progress would end a healthy exploration early on whatever
        // samples sibling traffic let it take. The two questions want two
        // ceilings: visitBudget() bounds how many visits the registrant may
        // spend, abandonBudget() bounds how long a problem nobody is exploring
        // may stay unresolved. Past it the problem gives up, publishes a
        // winner-less resolution and both entry points short-circuit without a
        // lock, which is what the design already does for every other terminal
        // state.
        //
        // Atomic because the measurement hook advances it under the shared
        // lock; the exclusive lock excludes that, so the reset needs nothing
        // further.
        //
        // Deliberately not a second identity check at the measurement hook:
        // that hook holds the same Tensile problem with a different
        // workspaceSize, the caller's own allocation rather than the ceiling
        // the heuristic ranked under, which is the whole reason the resolution
        // key exists (see onlineTuningProblemKey). A hash taken there would
        // disagree with the registered one for every caller whose preference
        // and allocation differ, and nothing of that caller's would ever be
        // measured. The offer costs one compare and does not care.
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
            bool                            m_queriedOnly  = false;
            std::atomic<int>                m_dispatchesSinceSelection{0};
        };

        OnlineTuner();
        ~OnlineTuner();

        /// Read onlineTunerConfig() into the member flags. Constructor-shaped
        /// work, named so reloadForTest cannot drift from it.
        void applyConfig();

        int  selectCandidateImpl(size_t                  problemKey,
                                 size_t                  problem,
                                 const std::vector<int>& rankedSolutionIndices,
                                 bool                    capturing,
                                 bool                    retained);
        void harvestPendingImpl(size_t problemKey);
        bool beginMeasurementImpl(size_t      problemKey,
                                  int         solutionIndex,
                                  hipEvent_t& start,
                                  hipEvent_t& stop);

        // The remainder require m_mutex, held shared where they only read and
        // exclusively where they mutate state or the event pool.
        void registerProblem(size_t                  problemKey,
                             size_t                  problem,
                             ProblemState&           state,
                             const std::vector<int>& rankedSolutionIndices);
        static bool unmeasured(const ProblemState& state);
        int         nextCandidate(const ProblemState& state) const;
        int  measurableCandidate(const ProblemState& state,
                                 size_t              problemKey,
                                 int                 solutionIndex) const;
        int  inFlightCap(const ProblemState& state) const;
        int  visitBudget(const ProblemState& state) const;
        int  abandonBudget(const ProblemState& state) const;
        float score(const std::vector<float>& samples) const;
        void  resolve(size_t problemKey, ProblemState& state);
        void  releaseExploration(ProblemState& state);
        void  publishResolution(size_t problemKey, const ProblemState& state);
        bool  acquireEvents(hipEvent_t& start, hipEvent_t& stop, int& device);
        void  recycleEvents(hipEvent_t start, hipEvent_t stop, int device);
        void  retireEvents(hipEvent_t start, hipEvent_t stop, int device);

        bool m_enabled   = false;
        int  m_topK      = 0;
        int  m_repeats   = 0;
        int  m_coldCalls = 0;
        bool m_verbose   = false;

        // Stamped onto every registration so a thread-held offer names one
        // registration of one problem and not merely one key. Written under the
        // write lock; never reset, including by reloadForTest(), which is the
        // point -- an offer a test's earlier problem left on some thread must
        // not match the problem that replaces it.
        size_t m_registrations = 0;

        std::atomic<uint64_t> m_misses{0};

        // Keyed by device, because a HIP event belongs to the device that
        // created it: hipEventRecord onto another device's stream is refused
        // with hipErrorInvalidResourceHandle. One flat pool would hand device
        // B a pair device A created as soon as both explore in the same
        // process, which is the ordinary shape of a multi-GPU application.
        //
        // The map is only ever read and written under m_mutex held
        // exclusively, so the vectors it owns cannot move under a reader.
        std::unordered_map<size_t, ProblemState>        m_problems;
        std::unordered_map<int, std::vector<EventPair>> m_pairs;
        std::shared_timed_mutex                         m_mutex;

        // Pairs created on each device and not yet retired, whether pooled or
        // outstanding; what livePairCap() bounds. Same lock as m_pairs.
        std::unordered_map<int, int> m_livePairs;

        // The resolved table is deliberately not the problem map: it is fixed
        // in size and never rehashes, which is what lets resolution() follow a
        // published entry with no lock at all. The entries themselves outlive
        // the process, so a reader can never be handed a dangling one.
        //
        // A power of two so the index is a mask, and far larger than the number
        // of distinct problems a process is expected to see; overshooting only
        // costs the table's own footprint, 32KiB.
        //
        // KNOWN BOUNDARY. This is the one limit the fast path has. Past roughly
        // this many live resolved keys, keys start landing in occupied slots
        // and the later one evicts the earlier, which costs the evicted problem
        // the fast path for good -- publishing happens once, at the moment a
        // problem resolves, and nothing republishes it. The degradation is
        // gradual and proportional, each evicted problem falling back to the
        // locked path it took while it was still being explored, and nothing it
        // is told is wrong; resolution() answers on the key, so a slot holding
        // someone else's entry reads as absent. The pools below are
        // deliberately NOT capped at this number: capping them turned the same
        // boundary into a cliff, where the 4097th problem to resolve and every
        // one after it was refused a published entry outright.
        static constexpr size_t c_resolutionSlots = 4096;

        // See livePairCap().
        static constexpr int c_livePairExplorations = 16;

        std::vector<std::unique_ptr<Resolution>> m_resolutionPool;
        std::atomic<const Resolution*>           m_resolutions[c_resolutionSlots] = {};

        // Kept alive for the same reason: at most one per resolved problem,
        // taken from the first caller to offer one. The solutions themselves
        // are owned by the library, which outlives the tuner, so what this
        // holds is a reference count and not the kernel.
        std::vector<std::unique_ptr<PinnedWinner>> m_winnerPool;

        friend struct ::OnlineTunerTestAccess;
    };
} // namespace rocblaslt
