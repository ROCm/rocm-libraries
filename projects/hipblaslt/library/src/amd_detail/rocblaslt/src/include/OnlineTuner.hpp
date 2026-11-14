// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "TuningCacheStore.hpp"

// Defined only by the tuner's unit tests, which read private state.
struct OnlineTunerTestAccess;

namespace rocblaslt
{
    /**
     * The tuner's settings. The tuner reads no environment itself; see
     * onlineTunerConfig().
     */
    struct OnlineTunerConfig
    {
        bool enabled   = false;
        int  topK      = 0; // ranked-prefix depth; below two the tuner stays off
        int  repeats   = 1; // timed launches per candidate
        int  coldCalls = 0; // untimed visits of a problem before exploration

        // How much faster than candidate 0, the heuristic's pick, in percent,
        // another candidate must measure to win; 0 lets the fastest win.
        int minImprovementPct = 0;

        // Not scaled by the search depth.
        static constexpr int c_defaultLivePairCap = 4096;

        int livePairCap = c_defaultLivePairCap;

        // Where the per-event trace goes; null turns it off. Called with the
        // tuner's lock held, so it must not throw or call back into the tuner.
        void (*trace)(const char* func, const std::string& message) = nullptr;

        // Called once per device, with no lock held, when a launch there goes
        // untimed because every pair is in flight.
        void (*livePairCapReached)(int device, int cap) = nullptr;
    };

    /** Defined by whoever links the tuner; called by the constructor and reloadForTest(). */
    OnlineTunerConfig onlineTunerConfig();

    /**
     * @brief Explore-then-cache kernel selection on top of the ranked
     * candidates.
     *
     * The first topK() * repeats() dispatches of a problem launch each of the
     * top-K candidates in turn, timed with an event pair that is read back on a
     * later visit once the stop event has completed, so nothing waits on the
     * GPU. The candidate with the smallest sample wins and is written to the
     * tuning file.
     *
     * A problem is its tuning-cache key, the key tune searches and the file
     * stores: each key is explored once and gets one row, and later calls
     * replay that row without entering the tuner.
     *
     * Which problems are explored is onlineTunable() in tensile_host.cpp;
     * env-variables.rst describes it for users. Only hipblasLtMatmul with
     * algo == nullptr reaches the tuner: such a call selects and launches on
     * one thread, so a probe offered at selection is paired exactly with the
     * launch that measures it (t_offer in OnlineTuner.cpp). A caller-supplied
     * algo cannot be paired that way, and a heuristic query's caller keeps the
     * algo, so a probe handed to either would never be measured.
     */
    class OnlineTuner
    {
    public:
        using Key = TensileLite::ProblemOverride;

        static OnlineTuner& getInstance()
        {
            // Never destroyed: no member destructor runs at exit, possibly after
            // the HIP runtime's teardown, and pooled events stay reachable.
            static OnlineTuner* const gInstance = new OnlineTuner;
            return *gInstance;
        }

        OnlineTuner(const OnlineTuner&)            = delete;
        OnlineTuner& operator=(const OnlineTuner&) = delete;

        bool enabled() const
        {
            return m_enabled;
        }

        // Slack over the topK * repeats visits exploration needs, so a caller
        // that ignores the rotation cannot keep a problem exploring for ever.
        static constexpr int c_visitBudgetFactor = 8;

        // Larger, because the counter it bounds also counts the dispatches of
        // other problems under the key; see ProblemState::m_dispatchesSinceSelection.
        static constexpr int c_abandonBudgetFactor = 16;

        // Settings above these are clamped, so the budgets fit in an int.
        static constexpr int c_maxTopK    = 4096;
        static constexpr int c_maxRepeats = 1024;
        static_assert(int64_t(c_maxTopK) * c_maxRepeats * c_visitBudgetFactor
                              * c_abandonBudgetFactor
                          <= std::numeric_limits<int>::max() / 2,
                      "the abandonment budget must fit in an int");

        /**
         * Re-read the configuration and forget every problem. Tests only, with
         * no other thread inside the library.
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
         * The most event pairs, pooled and outstanding, kept alive on one
         * device. At the cap, other problems' finished pairs are reclaimed;
         * only when every pair is in flight does a launch go untimed.
         */
        int livePairCap() const
        {
            return m_livePairCap;
        }

        /** What the one call that writes a key's row is told to record. */
        struct Record
        {
            float  us             = -1.0f; // the winner's time; negative on every other call
            bool   complete       = false; // whether every candidate was measured
            int    candidates     = 0; // how many candidates were explored
            size_t workspaceBytes = 0; // the workspace of the problem that explored them
        };

        /**
         * @brief Pick which of the ranked candidates should run next: the
         * position to promote to the front, or -1 to leave the order alone.
         *
         * Each key is explored once, by the first exact problem to register
         * under it; calls under the key that differ in what it leaves out
         * (beta, workspace) are not probed. A call that may not probe, such as
         * a capture, registers nothing. Once the key resolves, every call is
         * served the winner, and the first of them, whichever problem it is,
         * has record filled in.
         */
        int selectCandidate(const Key&              key,
                            size_t                  problemHash,
                            size_t                  workspaceBytes,
                            const std::vector<int>& rankedSolutionIndices,
                            bool                    mayProbe,
                            Record*                 record);

        /**
         * Whether selection has nothing left to do for key: it resolved with no
         * winner, or a call has claimed its winner's row, after which replay
         * serves every call the winner suits.
         */
        bool settled(const Key& key);

        /** Read back every outstanding measurement the GPU has finished. */
        void harvestPending(const Key& key);

        /**
         * @brief Claim an event pair for the imminent launch of solutionIndex.
         *
         * Succeeds only for the launch selectCandidate() last offered this
         * thread, and only while the thread holds no unread HIP error, which a
         * record could replace. On true, record start and stop around the
         * launch on stream, then call endMeasurement() once the launch has
         * returned, whether or not it recorded anything.
         */
        bool beginMeasurement(const Key&  key,
                              int         solutionIndex,
                              hipStream_t stream,
                              hipEvent_t& start,
                              hipEvent_t& stop);

        /**
         * Hand back the pair beginMeasurement() gave a launch, once the launch
         * has returned. Until then nothing reads, pools or destroys the pair.
         */
        void endMeasurement(const Key& key, hipEvent_t start);

        /**
         * A launch's event pair from beginMeasurement(), handed back by
         * endMeasurement() when the handle goes. Without a pair, start() and
         * stop() are null and the launch runs untimed.
         */
        class Measurement
        {
        public:
            Measurement() = default;
            Measurement(Measurement&& other) noexcept
                : m_tuner(other.m_tuner)
                , m_key(other.m_key)
                , m_start(std::exchange(other.m_start, nullptr))
                , m_stop(std::exchange(other.m_stop, nullptr))
            {
            }
            Measurement& operator=(Measurement&&) = delete;

            ~Measurement()
            {
                if(m_start)
                    m_tuner->endMeasurement(*m_key, m_start);
            }

            hipEvent_t start() const
            {
                return m_start;
            }

            hipEvent_t stop() const
            {
                return m_stop;
            }

        private:
            friend class OnlineTuner;

            OnlineTuner* m_tuner = nullptr;
            const Key*   m_key   = nullptr;
            hipEvent_t   m_start = nullptr;
            hipEvent_t   m_stop  = nullptr;
        };

        /**
         * beginMeasurement() for the launch about to run on this thread. key
         * must outlive the handle.
         */
        Measurement measure(const Key& key, int solutionIndex, hipStream_t stream);

    private:
        // Pooled only after an accepted sample, and pooled inverted, so reusing
        // it without recording it reads a negative elapsed time.
        struct EventPair
        {
            hipEvent_t m_start = nullptr;
            hipEvent_t m_stop  = nullptr;
        };

        // m_launched is set by endMeasurement(); until then the pair is not
        // read, because the launch records it outside the lock.
        struct PendingMeasurement
        {
            hipEvent_t m_start     = nullptr;
            hipEvent_t m_stop      = nullptr;
            int        m_candidate = -1;
            int        m_device    = 0;
            bool       m_launched  = false;
        };

        // m_problemHash is the problem that registered; only its visits explore,
        // since a key covers calls whose kernels run differently (beta 0 or 1).
        // m_dispatchesSinceSelection counts every unreplayed dispatch under the
        // key since the registrant's last visit, so a problem whose registrant
        // stopped dispatching still resolves. Atomics: written under the shared
        // lock.
        struct ProblemState
        {
            std::vector<int>                m_candidates;
            std::vector<std::vector<float>> m_samples;
            std::vector<int>                m_issued;
            std::vector<PendingMeasurement> m_pending;
            size_t                          m_problemHash = 0;
            size_t                          m_workspace   = 0;
            int                             m_coldCalls   = 0;
            int                             m_calls       = 0;
            int                             m_winner      = -1;
            float                           m_winnerUs    = 0.0f;
            bool                            m_complete    = false;
            bool                            m_gaveUp      = false;
            bool                            m_resolved    = false;
            bool                            m_listed      = false;
            std::atomic<bool>               m_recordClaimed{false};
            std::atomic<int>                m_dispatchesSinceSelection{0};
        };

        using Problems = std::unordered_map<Key, ProblemState>;

        enum class PairReading
        {
            Outstanding,
            Sample,
            Rejected,
        };

        enum class Acquired
        {
            Pair,
            AtCap,
            Failed,
        };

        OnlineTuner();
        ~OnlineTuner() = delete; // see getInstance()

        void applyConfig();

        // The remainder require m_mutex, held shared where they only read and
        // exclusively where they mutate state or the event pool.
        void        registerProblem(const Key&              key,
                                    size_t                  problemHash,
                                    size_t                  workspaceBytes,
                                    ProblemState&           state,
                                    const std::vector<int>& rankedSolutionIndices);
        int         servedPosition(ProblemState&           state,
                                   const std::vector<int>& rankedSolutionIndices,
                                   Record*                 record);
        PairReading readPair(const PendingMeasurement& pending, float& us) const;
        void        settlePair(const Key&                key,
                               ProblemState&             state,
                               const PendingMeasurement& pending,
                               PairReading               reading,
                               float                     us,
                               const char*               event);
        void        reclaimFinishedPairs(int device, const ProblemState* requester);
        int         nextCandidate(const ProblemState& state) const;
        int         measurableCandidate(const ProblemState& state, int solutionIndex) const;
        int         inFlightCap(const ProblemState& state) const;
        int         visitBudget(const ProblemState& state) const;
        int         abandonBudget(const ProblemState& state) const;
        void        resolve(const Key& key, ProblemState& state);
        void        releaseExploration(ProblemState& state);
        Acquired    acquireEvents(hipStream_t         stream,
                                  hipEvent_t&         start,
                                  hipEvent_t&         stop,
                                  int&                device,
                                  const ProblemState* requester);
        void        recycleEvents(hipEvent_t start, hipEvent_t stop, int device);
        void        retireEvents(hipEvent_t start, hipEvent_t stop, int device);

        bool m_enabled     = false;
        int  m_topK        = 0;
        int  m_repeats     = 0;
        int  m_coldCalls   = 0;
        int  m_livePairCap = 1;

        int m_minImprovementPct = 0;

        void (*m_trace)(const char* func, const std::string& message) = nullptr;

        // Pools are keyed by device: hipEventRecord refuses another device's
        // event. All containers below are guarded by m_mutex.
        Problems                                        m_problems;
        std::unordered_map<int, std::vector<EventPair>> m_pairs;
        std::shared_timed_mutex                         m_mutex;

        // The problems that may hold a pending pair (ProblemState::m_listed),
        // which is all reclaimFinishedPairs() walks.
        std::vector<Problems::value_type*> m_holders;

        std::unordered_map<int, int> m_livePairs;

        std::unordered_set<int> m_capReported;
        void (*m_livePairCapReached)(int device, int cap) = nullptr;

        friend struct ::OnlineTunerTestAccess;
    };
} // namespace rocblaslt
