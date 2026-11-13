// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the online tuner's bookkeeping, against a private copy of the tuner:
// OnlineTuner.cpp is compiled into this binary with the onlineTunerConfig() at the
// bottom of this file. See clients/tests/src/CMakeLists.txt.

#include "OnlineTuner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <string>
#include <thread>
#include <vector>

using Tuner = rocblaslt::OnlineTuner;
using Key   = Tuner::Key;

/**
 * Declared by OnlineTuner.hpp and defined only here. Read-only access to state
 * the public surface does not expose.
 */
struct OnlineTunerTestAccess
{
    // Samples a problem has taken, over all its candidates.
    static size_t samples(const Tuner& tuner, const Key& key)
    {
        const auto found = tuner.m_problems.find(key);
        if(found == tuner.m_problems.end())
            return 0;

        size_t count = 0;
        for(const std::vector<float>& candidate : found->second.m_samples)
            count += candidate.size();
        return count;
    }

    // Whether a key has resolved, and to what: the winner's solution index (-1 if
    // nothing was measured) and its time. Takes the lock, so other threads may be
    // inside the tuner.
    static bool resolved(Tuner& tuner, const Key& key, int* winner = nullptr, float* us = nullptr)
    {
        std::shared_lock<std::shared_timed_mutex> lock(tuner.m_mutex);

        const auto found = tuner.m_problems.find(key);
        if(found == tuner.m_problems.end() || !found->second.m_resolved)
            return false;

        if(winner)
            *winner = found->second.m_winner;
        if(us)
            *us = found->second.m_winnerUs;
        return true;
    }

    // Resolves a registered problem as if its candidates had measured samples.
    static void resolveWith(Tuner& tuner, const Key& key, std::vector<std::vector<float>> samples)
    {
        std::unique_lock<std::shared_timed_mutex> lock(tuner.m_mutex);

        auto& state     = tuner.m_problems.at(key);
        state.m_samples = std::move(samples);
        tuner.resolve(key, state);
    }

    // Pairs created on a device and not yet retired, pooled or outstanding.
    static int livePairs(const Tuner& tuner, int device)
    {
        const auto found = tuner.m_livePairs.find(device);
        return found == tuner.m_livePairs.end() ? 0 : found->second;
    }
};

namespace
{
    // What this test's onlineTunerConfig() hands the singleton.
    constexpr int kTopK    = 3;
    constexpr int kRepeats = 2;

    // Small, so a case reaches the cap with a handful of shapes.
    constexpr int kLivePairCap = 16;

    // 0 unless a case sets it before reloadForTest(); the fixture puts it back.
    int minImprovementPct = 0;

    // Host-only cases record no event, so exploration can only end at the visit
    // budget. Bounded well above it, so a tuner that never gives up fails instead of
    // hanging.
    constexpr int kCallCeiling = 4096;

    // visitBudget() bounds the visits the registrant may spend; abandonBudget()
    // bounds how long a problem nobody is exploring stays unresolved.
    constexpr int kVisitBudget   = kTopK * kRepeats * Tuner::c_visitBudgetFactor;
    constexpr int kAbandonBudget = kVisitBudget * Tuner::c_abandonBudgetFactor;

    // Well past the registrant's budget and well short of the abandonment ceiling.
    constexpr int kEarlyEnough = 512;

    Tuner& tuner()
    {
        return Tuner::getInstance();
    }

    // A tuning-cache key, told apart by its m.
    Key keyOf(size_t id)
    {
        Key key;
        key.m = id;
        return key;
    }

    /**
     * Stands in for the exact problem identity the library passes alongside the
     * key. Varying it under a fixed key makes another problem under the key: one
     * the key leaves undistinguished (beta 0 or 1, workspace) that ranks or runs
     * differently.
     */
    size_t identityOf(const Key& key, int variant = 0)
    {
        return key.m * 1000003u + static_cast<size_t>(variant);
    }

    // A dispatch's selection visit. record is filled in on the call that writes
    // the row.
    int visit(const Key&              key,
              size_t                  problem,
              const std::vector<int>& ranked,
              Tuner::Record*          record    = nullptr,
              size_t                  workspace = 0)
    {
        return tuner().selectCandidate(key, problem, workspace, ranked, /*mayProbe=*/true, record);
    }

    // A selection that may not probe, such as a capture's: never offered one.
    int query(const Key& key, size_t problem, const std::vector<int>& ranked)
    {
        return tuner().selectCandidate(key, problem, 0, ranked, /*mayProbe=*/false, nullptr);
    }

    bool resolved(const Key& key, int* winner = nullptr, float* us = nullptr)
    {
        return OnlineTunerTestAccess::resolved(tuner(), key, winner, us);
    }

    int winnerOf(const Key& key)
    {
        int winner = -2;
        static_cast<void>(resolved(key, &winner));
        return winner;
    }

    int positionOf(const std::vector<int>& ranked, int solution)
    {
        return static_cast<int>(std::find(ranked.begin(), ranked.end(), solution) - ranked.begin());
    }

    // What the livePairCapReached callback was told, in order: (device, cap).
    std::vector<std::pair<int, int>>& capReports()
    {
        static std::vector<std::pair<int, int>> reports;
        return reports;
    }

    void recordCapReached(int device, int cap)
    {
        capReports().emplace_back(device, cap);
    }

    // Every case starts and ends with a fresh tuner and an empty pool, so none
    // depends on another.
    class OnlineTuner : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            minImprovementPct = 0;
            tuner().reloadForTest();
            capReports().clear();
        }

        void TearDown() override
        {
            minImprovementPct = 0;
            tuner().reloadForTest();
        }
    };

    class OnlineTuner_multi_gpu : public OnlineTuner
    {
    };

    // Offers the same ranking for the same problem until it resolves. Returns the
    // number of visits, or -1 if it never resolved.
    int exploreToResolution(const Key& key, size_t problem, const std::vector<int>& ranked)
    {
        for(int calls = 1; calls <= kCallCeiling; ++calls)
        {
            visit(key, problem, ranked);
            if(resolved(key))
                return calls;
        }

        return -1;
    }

    // With nothing ever measured, exploration gives up at the visit budget rather
    // than holding the write lock for the life of the process.
    TEST_F(OnlineTuner, ExplorationGivesUpAtTheVisitBudget)
    {
        const Key              key = keyOf(0x1004);
        const std::vector<int> ranked{21, 22, 23, 24};

        ASSERT_TRUE(tuner().enabled());
        ASSERT_EQ(tuner().topK(), kTopK);
        ASSERT_EQ(tuner().repeats(), kRepeats);
        ASSERT_EQ(tuner().livePairCap(), kLivePairCap);

        // An empty ranking (xf32's first lookup) registers nothing: registered, it
        // would resolve at once with no winner.
        EXPECT_EQ(visit(key, identityOf(key), {}), -1);
        EXPECT_FALSE(resolved(key)) << "an empty ranking registered the key";

        // A key selection never registered is refused, not measured blind.
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, 21, nullptr, start, stop));
        EXPECT_EQ(start, nullptr);

        EXPECT_EQ(visit(key, identityOf(key), ranked), 0);
        EXPECT_FALSE(resolved(key));
        EXPECT_FALSE(tuner().settled(key));

        const int calls = exploreToResolution(key, identityOf(key), ranked);
        ASSERT_GT(calls, 0) << "exploration never ended";
        EXPECT_GT(calls, kTopK * kRepeats);
        EXPECT_LE(calls, kVisitBudget) << "the visit budget did not bound it";

        EXPECT_EQ(winnerOf(key), -1);
        EXPECT_TRUE(tuner().settled(key)) << "a key resolved with no winner is not settled";
        EXPECT_EQ(visit(key, identityOf(key), ranked), -1);

        EXPECT_FALSE(tuner().beginMeasurement(key, 21, nullptr, start, stop));
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(stop, nullptr);

        tuner().harvestPending(key);
        EXPECT_EQ(winnerOf(key), -1);
    }

    // A candidate replaces candidate 0, the heuristic's pick, only if it measured at
    // least the minimum improvement faster. By default the fastest wins.
    TEST_F(OnlineTuner, AChallengerMustBeatCandidateZeroByTheMinimumImprovement)
    {
        const std::vector<int> ranked{301, 302, 303};

        // Candidate 1 is 2% faster than candidate 0, candidate 2 1% faster.
        const std::vector<std::vector<float>> close{{100.0f, 104.0f}, {98.0f, 99.0f}, {99.0f}};
        // Candidate 2 is 5% faster.
        const std::vector<std::vector<float>> clear{{100.0f}, {98.0f}, {95.0f}};

        const auto winnerFor = [&](const Key& key, const std::vector<std::vector<float>>& samples) {
            EXPECT_EQ(visit(key, identityOf(key), ranked), 0);
            OnlineTunerTestAccess::resolveWith(tuner(), key, samples);
            return winnerOf(key);
        };

        EXPECT_EQ(winnerFor(keyOf(0x9001), close), 302) << "the fastest did not win by default";

        minImprovementPct = 3;
        tuner().reloadForTest();

        EXPECT_EQ(winnerFor(keyOf(0x9002), close), 301)
            << "a candidate less than 3% faster replaced candidate 0";
        EXPECT_EQ(winnerFor(keyOf(0x9003), clear), 303)
            << "a candidate 5% faster did not replace candidate 0";

        float us = 0.0f;
        ASSERT_TRUE(resolved(keyOf(0x9002), nullptr, &us));
        EXPECT_EQ(us, 100.0f) << "the kept candidate 0 is recorded with another's time";
    }

    // Another problem under the key, here beta 1 where the owner runs beta 0, neither
    // ends an exploration early nor keeps one alive. Its dispatches spend none of the
    // owner's visit budget and each owner visit restarts the abandonment count, but
    // an owner that stops visiting is given up on once that count runs out.
    TEST_F(OnlineTuner, AnAbandonedExplorationEndsLateAndNeverEarly)
    {
        const Key key = keyOf(0x1019);

        const size_t owner = identityOf(key, 0);
        const size_t other = identityOf(key, 1);

        const std::vector<int> ranked{201, 202, 203};

        ASSERT_EQ(visit(key, owner, ranked), 0);

        // Twice the window between the two ceilings, split by an owner visit.
        ASSERT_GT(kEarlyEnough, kVisitBudget);
        ASSERT_LT(kEarlyEnough, kAbandonBudget);
        ASSERT_GT(2 * kEarlyEnough, kAbandonBudget);

        for(int half = 0; half < 2; ++half)
        {
            for(int call = 0; call < kEarlyEnough; ++call)
            {
                ASSERT_EQ(visit(key, other, ranked), -1)
                    << "another problem was handed a probe candidate on call " << call;
                tuner().harvestPending(key);
            }

            EXPECT_FALSE(resolved(key))
                << "another problem's dispatches ended the exploration early";

            // Still owed its first candidate, and the visit restarts the count.
            EXPECT_EQ(visit(key, owner, ranked), 0);
        }

        // The owner stops visiting while the other problem keeps dispatching.
        int otherCalls = 0;
        for(; otherCalls < kCallCeiling && !resolved(key); ++otherCalls)
        {
            visit(key, other, ranked);
            tuner().harvestPending(key);
        }

        ASSERT_TRUE(resolved(key)) << "another problem held an abandoned exploration open for ever";
        EXPECT_EQ(winnerOf(key), -1) << "nothing was measured, so there is no winner";
        EXPECT_GT(otherCalls, kVisitBudget) << "the owner's visit budget ended it";
        EXPECT_LE(otherCalls, kAbandonBudget + 2) << "the abandonment budget did not bound it";

        EXPECT_EQ(visit(key, owner, ranked), -1);
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], nullptr, start, stop));
    }

    // ----------------------------------------------------------------------
    // Cases that need a device: a candidate's issue count only moves when a launch
    // is offered an event pair, and the pool is itself under test.
    // ----------------------------------------------------------------------

    int deviceCount()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess ? count : 0;
    }

    // A stream and a scrap buffer to bracket between a measurement's two events:
    // back-to-back records can read exactly 0.0 ms, which the sign test rejects.
    class Device
    {
    public:
        explicit Device(int id)
            : m_id(id)
        {
            EXPECT_EQ(hipSetDevice(id), hipSuccess);
            EXPECT_EQ(hipStreamCreate(&m_stream), hipSuccess);
            EXPECT_EQ(hipMalloc(&m_scratch, c_scratchBytes), hipSuccess);
        }

        ~Device()
        {
            static_cast<void>(hipSetDevice(m_id));
            static_cast<void>(hipFree(m_scratch));
            static_cast<void>(hipStreamDestroy(m_stream));
        }

        Device(const Device&)            = delete;
        Device& operator=(const Device&) = delete;

        int id() const
        {
            return m_id;
        }
        hipStream_t stream() const
        {
            return m_stream;
        }

        // What the launch does for the library: record both ends around some work,
        // return, and leave the pair for the next harvest.
        void bracket(const Key& key, hipEvent_t start, hipEvent_t stop) const
        {
            EXPECT_EQ(hipEventRecord(start, m_stream), hipSuccess);
            EXPECT_EQ(hipMemsetAsync(m_scratch, 0, c_scratchBytes, m_stream), hipSuccess);
            EXPECT_EQ(hipEventRecord(stop, m_stream), hipSuccess);
            tuner().endMeasurement(key, start);
            EXPECT_EQ(hipStreamSynchronize(m_stream), hipSuccess);
        }

    private:
        static constexpr size_t c_scratchBytes = 4u << 20;

        int         m_id      = 0;
        hipStream_t m_stream  = nullptr;
        void*       m_scratch = nullptr;
    };

    // One whole visit: take the offer, time the launch, read it back. Returns the
    // position offered.
    int sampleOneVisit(const Key&              key,
                       size_t                  problem,
                       const std::vector<int>& ranked,
                       const Device&           device)
    {
        const int position = visit(key, problem, ranked);
        if(position < 0)
            return position;

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_TRUE(tuner().beginMeasurement(key, ranked[position], device.stream(), start, stop))
            << "the tuner refused the launch it had just asked for";

        if(start && stop)
        {
            device.bracket(key, start, stop);
            tuner().harvestPending(key);
        }

        return position;
    }

    // One dispatch as the library makes it: harvest, select, time the launch. The
    // pair stays outstanding until the key's next dispatch. Returns whether the
    // launch was timed.
    bool dispatchOnce(const Key&              key,
                      size_t                  problem,
                      const std::vector<int>& ranked,
                      const Device&           device)
    {
        tuner().harvestPending(key);

        const int position = visit(key, problem, ranked);
        if(position < 0)
            return false;

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        if(!tuner().beginMeasurement(key, ranked[position], device.stream(), start, stop))
            return false;

        device.bracket(key, start, stop);
        return true;
    }

    // A stream held on a host-visible value until release(), so what is recorded
    // on it stays in flight for as long as a case needs. Like Device, it brackets
    // some work, so a pair never reads exactly 0.0 ms.
    class HeldStream
    {
    public:
        static bool supported(int id)
        {
            int canWait = 0;
            return hipDeviceGetAttribute(&canWait, hipDeviceAttributeCanUseStreamWaitValue, id)
                       == hipSuccess
                   && canWait != 0;
        }

        explicit HeldStream(int id)
        {
            EXPECT_EQ(hipSetDevice(id), hipSuccess);
            EXPECT_EQ(hipStreamCreate(&m_stream), hipSuccess);
            EXPECT_EQ(hipMalloc(&m_scratch, c_scratchBytes), hipSuccess);
            EXPECT_EQ(hipHostMalloc(reinterpret_cast<void**>(&m_go), sizeof(*m_go), 0), hipSuccess);
            if(m_go)
            {
                *m_go = 0;
                EXPECT_EQ(hipStreamWaitValue32(m_stream, m_go, 1, hipStreamWaitValueEq),
                          hipSuccess);
            }
        }

        ~HeldStream()
        {
            release();
            static_cast<void>(hipStreamDestroy(m_stream));
            static_cast<void>(hipFree(m_scratch));
            static_cast<void>(hipHostFree(m_go));
        }

        HeldStream(const HeldStream&)            = delete;
        HeldStream& operator=(const HeldStream&) = delete;

        // A launch's two records, behind the wait, and its return.
        void bracket(const Key& key, hipEvent_t start, hipEvent_t stop) const
        {
            EXPECT_EQ(hipEventRecord(start, m_stream), hipSuccess);
            EXPECT_EQ(hipMemsetAsync(m_scratch, 0, c_scratchBytes, m_stream), hipSuccess);
            EXPECT_EQ(hipEventRecord(stop, m_stream), hipSuccess);
            tuner().endMeasurement(key, start);
        }

        void release()
        {
            if(m_go && !m_released)
                __atomic_store_n(m_go, 1u, __ATOMIC_SEQ_CST);
            m_released = true;
            EXPECT_EQ(hipStreamSynchronize(m_stream), hipSuccess);
        }

    private:
        static constexpr size_t c_scratchBytes = 4u << 20;

        hipStream_t m_stream   = nullptr;
        void*       m_scratch  = nullptr;
        uint32_t*   m_go       = nullptr;
        bool        m_released = false;
    };

    // The offer names one candidate and buys one sample of it, for the owner's
    // next launch on the thread it was made on. Another problem's own pick is often
    // a candidate the exploration still owes, so matching a launch on the solution
    // index alone would credit one problem with the other's launch.
    TEST_F(OnlineTuner, OnlyTheOwnersOfferedLaunchIsSampled)
    {
        const Key key = keyOf(0x100a);

        const size_t owner = identityOf(key, 0);
        const size_t other = identityOf(key, 1);

        const std::vector<int> ranked{81, 82, 83};

        ASSERT_EQ(visit(key, owner, ranked), 0);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[1], nullptr, start, stop))
            << "a candidate the rotation did not ask for was timed";

        ASSERT_EQ(visit(key, other, ranked), -1);
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], nullptr, start, stop))
            << "another problem's launch was timed as a sample of the owner";
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(stop, nullptr);

        // Had either launch been credited, the rotation would have moved on.
        ASSERT_EQ(visit(key, owner, ranked), 0)
            << "a launch that was not offered advanced the rotation";

        if(deviceCount() < 1)
            return;

        const Device device(0);

        // Another problem on another thread, whose own first choice is the offered
        // solution, cannot claim the owner's offer, nor cancel it.
        std::thread([&] {
            EXPECT_EQ(visit(key, other, ranked), -1);
            hipEvent_t stolenStart = nullptr;
            hipEvent_t stolenStop  = nullptr;
            EXPECT_FALSE(
                tuner().beginMeasurement(key, ranked[0], device.stream(), stolenStart, stolenStop))
                << "another thread took the sample the owner's offer was made for";
        }).join();

        ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
            << "the owner was refused the launch its own offer named";
        device.bracket(key, start, stop);
        tuner().harvestPending(key);
        EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), 1u);

        // Another problem's visit between every one of the owner's, from another
        // thread, does not starve the owner of a winner.
        for(int visits = 0; visits < kCallCeiling && !resolved(key); ++visits)
        {
            const int position = visit(key, owner, ranked);

            std::thread([&] { EXPECT_EQ(visit(key, other, ranked), -1); }).join();

            if(position < 0)
                continue;

            ASSERT_TRUE(
                tuner().beginMeasurement(key, ranked[position], device.stream(), start, stop))
                << "another problem's visit cancelled the owner's offer on visit " << visits;
            device.bracket(key, start, stop);
            tuner().harvestPending(key);
        }

        int   winner = -1;
        float us     = 0.0f;
        ASSERT_TRUE(resolved(key, &winner, &us)) << "exploration never ended";
        EXPECT_GE(winner, 0) << "other problems' traffic starved the owner of a winner";
        EXPECT_GT(us, 0.0f);
    }

    // A captured graph replays its kernel without coming back through selection,
    // so a probe handed to it would never be measured. A selection that may not
    // probe is never offered one, registers nothing and spends no budget, and is
    // served the winner. Here it is another problem under the key, so a
    // registration would hold off the dispatches.
    TEST_F(OnlineTuner, ASelectionThatMayNotProbeIsNeverProbedAndIsServedTheWinner)
    {
        const Key              key        = keyOf(0x100d);
        const size_t           dispatched = identityOf(key, 0);
        const size_t           queried    = identityOf(key, 1);
        const std::vector<int> ranked{101, 102, 103};

        for(int call = 0; call < 512; ++call)
            ASSERT_EQ(query(key, queried, ranked), -1) << "a probe was offered on call " << call;

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], nullptr, start, stop));

        // The dispatches still register, and the budget is untouched.
        ASSERT_EQ(visit(key, dispatched, ranked), 0)
            << "the dispatch was held off by a registration the query made";
        for(int call = 0; call < 512; ++call)
            ASSERT_EQ(query(key, queried, ranked), -1) << "a probe was offered on call " << call;
        EXPECT_FALSE(resolved(key)) << "the queries spent the budget";
        EXPECT_EQ(visit(key, dispatched, ranked), 0) << "a query was credited";

        if(deviceCount() < 1)
            return;

        const Device device(0);

        // One visit short of a full pass leaves the last candidate owed.
        for(int candidate = 0; candidate < kTopK - 1; ++candidate)
            ASSERT_EQ(sampleOneVisit(key, dispatched, ranked, device), candidate);
        EXPECT_EQ(query(key, queried, ranked), -1) << "the owed probe was promoted";
        EXPECT_EQ(visit(key, dispatched, ranked), kTopK - 1)
            << "the query consumed the candidate it was refused";

        for(int call = 0; call < kCallCeiling && !resolved(key); ++call)
            sampleOneVisit(key, dispatched, ranked, device);

        const int winner = winnerOf(key);
        ASSERT_GE(winner, 0) << "exploration never ended, or nothing was measured";
        EXPECT_EQ(query(key, queried, ranked), positionOf(ranked, winner))
            << "denied the measured winner";
    }

    // The row is written once, by the first call served the winner, whichever
    // problem under the key that is: here the owner stops dispatching once its key
    // resolves and another problem records the row. A single candidate is measured
    // and recorded like any other. complete says whether every candidate was timed.
    TEST_F(OnlineTuner, TheRowIsClaimedOnceByTheFirstCallServedTheWinner)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        const Key              key   = keyOf(0x100c);
        const size_t           owner = identityOf(key, 0);
        const size_t           other = identityOf(key, 1);
        const std::vector<int> ranked{94, 95, 96};

        for(int call = 0; call < kCallCeiling && !resolved(key); ++call)
            sampleOneVisit(key, owner, ranked, device);

        int   winner = -1;
        float us     = 0.0f;
        ASSERT_TRUE(resolved(key, &winner, &us)) << "exploration never ended";
        ASSERT_GE(winner, 0) << "nothing was measured";
        EXPECT_FALSE(tuner().settled(key)) << "a key settled before its row was claimed";

        // The row describes the exploration, not the problem that claims it.
        Tuner::Record record;
        EXPECT_EQ(visit(key, other, ranked, &record, /*workspace=*/1u << 20),
                  positionOf(ranked, winner));
        EXPECT_FLOAT_EQ(record.us, us) << "the first call served the winner did not claim the row";
        EXPECT_TRUE(tuner().settled(key)) << "a key whose row was claimed is not settled";
        EXPECT_TRUE(record.complete) << "every candidate was timed";
        EXPECT_EQ(record.candidates, 3);
        EXPECT_EQ(record.workspaceBytes, 0u) << "the row took the claimer's workspace";

        record = {};
        EXPECT_EQ(visit(key, owner, ranked, &record), positionOf(ranked, winner));
        EXPECT_LT(record.us, 0.0f) << "a second call was told to write the row";

        // One candidate.
        const Key              single = keyOf(0x101B);
        const std::vector<int> one{221};

        int visits = 0;
        for(; visits < kCallCeiling && !resolved(single); ++visits)
            ASSERT_EQ(sampleOneVisit(single, identityOf(single), one, device), 0) << visits;

        ASSERT_TRUE(resolved(single, &winner, &us)) << "exploration never ended";
        EXPECT_EQ(winner, 221);
        EXPECT_GT(us, 0.0f) << "the only candidate was not measured";
        EXPECT_EQ(visits, kRepeats) << "a single candidate took longer than its repeats";

        record = {};
        EXPECT_EQ(visit(single, identityOf(single), one, &record), 0);
        EXPECT_FLOAT_EQ(record.us, us) << "the single candidate's row was not claimed";
        EXPECT_EQ(record.candidates, 1);

        // Given up after one candidate was timed: its winner, marked incomplete.
        const Key partial = keyOf(0x101C);
        ASSERT_EQ(sampleOneVisit(partial, identityOf(partial), ranked, device), 0);
        ASSERT_GT(exploreToResolution(partial, identityOf(partial), ranked), 0);
        ASSERT_EQ(winnerOf(partial), ranked[0]);

        record = {};
        EXPECT_EQ(visit(partial, identityOf(partial), ranked, &record), 0);
        EXPECT_GT(record.us, 0.0f);
        EXPECT_FALSE(record.complete) << "a winner of one timed candidate in three was complete";
    }

    // ----------------------------------------------------------------------
    // The harvest's sign test.
    //
    // A pair queries as complete whether or not a launch recorded it, so a sample
    // is kept only when the elapsed time is positive. Pairs are pooled inverted and
    // only after a sample, so every way of missing a record reads non-positive. A
    // rejected pair is retired, not pooled: pooling it would invert it twice.
    // ----------------------------------------------------------------------

    // Each way a launch can fail to record its pair, on a pooled pair and on a
    // fresh one, is dropped, credits nothing, leaves no HIP error and retires the
    // pair.
    TEST_F(OnlineTuner, AnUnrecordedPairIsNeverASample)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id = device.id();

        const std::vector<int> ranked{301, 302, 303};
        size_t                 next = 0x1100;

        // One measurement of a new key, recorded as record() says, then harvested.
        // Returns the pair it was handed.
        auto measure = [&](const std::function<void(hipEvent_t, hipEvent_t)>& record) {
            const Key key = keyOf(next++);
            EXPECT_EQ(visit(key, identityOf(key), ranked), 0);

            std::pair<hipEvent_t, hipEvent_t> pair{nullptr, nullptr};
            EXPECT_TRUE(
                tuner().beginMeasurement(key, ranked[0], device.stream(), pair.first, pair.second));

            record(pair.first, pair.second);
            tuner().endMeasurement(key, pair.first);
            EXPECT_EQ(hipStreamSynchronize(device.stream()), hipSuccess);

            const int liveBefore = OnlineTunerTestAccess::livePairs(tuner(), id);
            tuner().harvestPending(key);

            EXPECT_EQ(hipGetLastError(), hipSuccess)
                << "the harvest left an error in the caller's HIP state";
            EXPECT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), liveBefore - 1)
                << "a rejected pair was not retired";
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), 0u)
                << "a pair no launch recorded was taken as a sample";
            return pair;
        };

        auto recordNothing   = [](hipEvent_t, hipEvent_t) {};
        auto recordStartOnly = [&](hipEvent_t start, hipEvent_t) {
            EXPECT_EQ(hipEventRecord(start, device.stream()), hipSuccess);
        };

        for(const auto& record : {std::function<void(hipEvent_t, hipEvent_t)>(recordNothing),
                                  std::function<void(hipEvent_t, hipEvent_t)>(recordStartOnly)})
        {
            // The pool starts empty, so one real sample leaves exactly its pair there.
            const Key  warm  = keyOf(next++);
            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            ASSERT_EQ(visit(warm, identityOf(warm), ranked), 0);
            ASSERT_TRUE(tuner().beginMeasurement(warm, ranked[0], device.stream(), start, stop));
            device.bracket(warm, start, stop);
            tuner().harvestPending(warm);
            ASSERT_EQ(OnlineTunerTestAccess::samples(tuner(), warm), 1u);

            // That pair comes back inverted.
            const auto pooled = measure(record);
            EXPECT_EQ(pooled.first, stop) << "the pair was not pooled inverted";
            EXPECT_EQ(pooled.second, start) << "the pair was not pooled inverted";

            // The pool is empty again, so a fresh pair, holding only the inverted
            // records made at creation.
            measure(record);
        }
    }

    // A device never holds more than livePairCap() pairs. One-shot shapes that hold
    // the whole cap with finished launches do not stop a shape that keeps coming back
    // from being explored: at the cap finished pairs are read back and reused, and
    // each one-shot shape keeps its sample. Only with every pair in flight does a
    // launch run untimed, reported once per device; nothing in flight is read back
    // early, and reading in-flight pairs leaves no HIP error.
    TEST_F(OnlineTuner, LiveEventPairsAreCappedReclaimedAndReportedOnce)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";
        if(!HeldStream::supported(0))
            GTEST_SKIP() << "the device cannot hold a stream on a host value";

        const Device device(0);
        const int    id  = device.id();
        const int    cap = tuner().livePairCap();

        const std::vector<int> ranked{271, 272, 273};
        size_t                 next = 0x71000;

        for(int shape = 0; shape < cap; ++shape, ++next)
            ASSERT_TRUE(dispatchOnce(keyOf(next), identityOf(keyOf(next)), ranked, device))
                << shape;
        ASSERT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap);

        const Key repeated = keyOf(0x72000);
        int       visits   = 0;
        for(; !resolved(repeated) && visits < kCallCeiling; ++visits)
            EXPECT_TRUE(dispatchOnce(repeated, identityOf(repeated), ranked, device) || visits > 0)
                << "the first launch of a new shape at the cap was not measured";

        ASSERT_TRUE(resolved(repeated));
        EXPECT_GE(winnerOf(repeated), 0) << "a shape that kept coming back was never measured";
        EXPECT_EQ(visits, kTopK * kRepeats + 1) << "exploration took longer than an uncapped one";
        for(size_t shape = 0x71000; shape < 0x71000 + static_cast<size_t>(cap); ++shape)
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), keyOf(shape)), 1u)
                << "shape " << shape << ": the sample its launch bought was not kept";
        EXPECT_LE(OnlineTunerTestAccess::livePairs(tuner(), id), cap);
        EXPECT_TRUE(capReports().empty()) << "reaching the cap and reclaiming was reported";

        // Every pair in flight, twice over.
        for(int hold = 0; hold < 2; ++hold)
        {
            HeldStream       held(id);
            std::vector<Key> timed;

            for(int shape = 0; shape < cap + 3; ++shape, ++next)
            {
                const Key key = keyOf(next);
                ASSERT_EQ(visit(key, identityOf(key), ranked), 0);

                hipEvent_t start = nullptr;
                hipEvent_t stop  = nullptr;
                if(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
                {
                    held.bracket(key, start, stop);
                    timed.push_back(key);
                    continue;
                }

                EXPECT_EQ(start, nullptr) << "a declined launch was handed an event";
                EXPECT_EQ(hipGetLastError(), hipSuccess)
                    << "reading pairs still in flight left an error in the caller's HIP state";
            }

            EXPECT_EQ(timed.size(), static_cast<size_t>(cap))
                << "launches in flight held more or fewer live event pairs than the cap";
            EXPECT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap);
            ASSERT_EQ(capReports().size(), 1u) << "hold " << hold;
            EXPECT_EQ(capReports()[0], std::make_pair(id, cap));

            held.release();
            for(const Key& key : timed)
            {
                tuner().harvestPending(key);
                EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), 1u)
                    << "key " << key.m << ": a launch still in flight was read back early";
            }
        }
    }

    // beginMeasurement() hands a pair out under the lock and the launch records it
    // after the lock is released. Until then nothing may take the pair: neither a
    // reclaim at the cap, which would hand one pair to two launches, nor a harvest
    // of the same shape on another thread, which would retire (destroy) a handle
    // about to be recorded.
    TEST_F(OnlineTuner, APairNotYetRecordedIsNeitherReclaimedNorRetired)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id  = device.id();
        const int    cap = tuner().livePairCap();

        const std::vector<int> ranked{251, 252, 253};

        // The whole cap pooled, then handed out for launches that have not recorded.
        std::vector<Key> filled;
        for(size_t next = 0x74000; filled.size() < static_cast<size_t>(cap); ++next)
        {
            filled.push_back(keyOf(next));
            ASSERT_TRUE(dispatchOnce(filled.back(), identityOf(filled.back()), ranked, device));
        }
        for(const Key& key : filled)
            tuner().harvestPending(key);

        struct Taken
        {
            Key        key;
            hipEvent_t start;
            hipEvent_t stop;
        };
        std::vector<Taken> taken;
        for(int shape = 0; shape < cap; ++shape)
        {
            Taken pair{keyOf(0x75000 + static_cast<size_t>(shape)), nullptr, nullptr};
            ASSERT_EQ(visit(pair.key, identityOf(pair.key), ranked), 0);
            ASSERT_TRUE(tuner().beginMeasurement(
                pair.key, ranked[0], device.stream(), pair.start, pair.stop));
            taken.push_back(pair);
        }
        ASSERT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap);

        // At the cap with nothing pooled, so the reclaim runs and must take nothing.
        const Key late = keyOf(0x76000);
        ASSERT_EQ(visit(late, identityOf(late), ranked), 0);
        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_FALSE(tuner().beginMeasurement(late, ranked[0], device.stream(), start, stop))
            << "a pair handed out for a launch not yet recorded was taken back";
        EXPECT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap)
            << "a pair handed out for a launch not yet recorded was retired";

        for(const Taken& pair : taken)
        {
            device.bracket(pair.key, pair.start, pair.stop);
            tuner().harvestPending(pair.key);
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), pair.key), 1u)
                << "key " << pair.key.m << " lost the launch it had a pair for";
        }

        // Two threads dispatching one shape: one holds a pooled pair it has not
        // recorded while the other harvests the shape. Twice.
        const Key    shared  = keyOf(0x79000);
        const size_t problem = identityOf(shared);
        for(size_t round = 1; round <= 2; ++round)
        {
            SCOPED_TRACE("round " + std::to_string(round));

            const int position = visit(shared, problem, ranked);
            ASSERT_GE(position, 0);
            ASSERT_TRUE(
                tuner().beginMeasurement(shared, ranked[position], device.stream(), start, stop));

            // Whatever the pair holds is complete, so a harvest that read it would get
            // an answer rather than "outstanding".
            ASSERT_EQ(hipStreamSynchronize(device.stream()), hipSuccess);

            const int live = OnlineTunerTestAccess::livePairs(tuner(), id);
            std::thread([&] { tuner().harvestPending(shared); }).join();

            // Asserted before the record, which on a destroyed pair would fault.
            ASSERT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), live)
                << "another dispatch retired a pair whose launch had not recorded it";

            device.bracket(shared, start, stop);
            std::thread([&] { tuner().harvestPending(shared); }).join();

            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), shared), round)
                << "the launch's sample was lost";
        }
    }

    // At the cap, a pair whose launch returned without recording it is retired by
    // the reclaim, as its own harvest would, even though its shape never comes
    // back: rejected pairs of shapes not seen again must not hold the cap.
    TEST_F(OnlineTuner, AtTheCapRejectedPairsOfShapesNotSeenAgainAreRetired)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);
        const int    id  = device.id();
        const int    cap = tuner().livePairCap();

        const std::vector<int> ranked{331, 332, 333};

        for(int shape = 0; shape < cap; ++shape)
        {
            const Key key = keyOf(0x7A000 + static_cast<size_t>(shape));
            ASSERT_EQ(visit(key, identityOf(key), ranked), 0);

            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop));
            tuner().endMeasurement(key, start);
        }
        ASSERT_EQ(hipStreamSynchronize(device.stream()), hipSuccess);
        ASSERT_EQ(OnlineTunerTestAccess::livePairs(tuner(), id), cap);

        const Key late = keyOf(0x7B000);
        ASSERT_EQ(visit(late, identityOf(late), ranked), 0);

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;
        EXPECT_TRUE(tuner().beginMeasurement(late, ranked[0], device.stream(), start, stop))
            << "rejected pairs of shapes not seen again held the cap";
        EXPECT_TRUE(capReports().empty()) << "the launch was reported as untimed at the cap";
        EXPECT_LE(OnlineTunerTestAccess::livePairs(tuner(), id), cap);
        EXPECT_EQ(hipGetLastError(), hipSuccess)
            << "the reclaim left an error in the caller's HIP state";

        if(start && stop)
        {
            device.bracket(late, start, stop);
            tuner().harvestPending(late);
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), late), 1u);
        }
    }

    // A pair is read back on a later call, which may be captured, on the stream
    // the pair was recorded on. Reading it must leave the capture valid, whether
    // that call is on the capturing thread or on another one, and still take the
    // sample.
    TEST_F(OnlineTuner, ReadingAPairBackLeavesACaptureOfItsStreamValid)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const std::vector<int> ranked{311, 312, 313};

        void* target = nullptr;
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_EQ(hipMalloc(&target, 4096), hipSuccess);

        for(const bool otherThread : {false, true})
        {
            SCOPED_TRACE(otherThread ? "read on another thread" : "read on the capturing thread");

            // A stream of its own, so one way failing cannot fail the other.
            const Device device(0);

            const Key key = keyOf(otherThread ? 0x1201 : 0x1200);
            ASSERT_EQ(visit(key, identityOf(key), ranked), 0);

            hipEvent_t start = nullptr;
            hipEvent_t stop  = nullptr;
            ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop));
            device.bracket(key, start, stop);

            ASSERT_EQ(hipStreamBeginCapture(device.stream(), hipStreamCaptureModeGlobal),
                      hipSuccess);
            EXPECT_EQ(hipMemsetAsync(target, 0, 4096, device.stream()), hipSuccess);

            if(otherThread)
                std::thread([&] { tuner().harvestPending(key); }).join();
            else
                tuner().harvestPending(key);

            hipGraph_t       graph = nullptr;
            const hipError_t ended = hipStreamEndCapture(device.stream(), &graph);
            const hipError_t left  = hipGetLastError();

            EXPECT_EQ(ended, hipSuccess) << hipGetErrorName(ended);
            EXPECT_EQ(left, hipSuccess) << hipGetErrorName(left);
            EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), 1u)
                << "the finished launch was not read back";

            if(graph)
            {
                hipGraphExec_t exec = nullptr;
                EXPECT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
                if(exec)
                {
                    EXPECT_EQ(hipGraphLaunch(exec, device.stream()), hipSuccess);
                    EXPECT_EQ(hipStreamSynchronize(device.stream()), hipSuccess);
                    static_cast<void>(hipGraphExecDestroy(exec));
                }
                static_cast<void>(hipGraphDestroy(graph));
            }
        }

        static_cast<void>(hipFree(target));
    }

    // The tuner runs inside the caller's hipblasLtMatmul and leaves its HIP error
    // state as it found it. A thread holding an error it has not read is not
    // measured, since a record could replace that error, and its offer stands. On
    // a clean thread, a pair that cannot be made leaves no error behind.
    TEST_F(OnlineTuner, TheCallersHipErrorStateIsLeftAsItWas)
    {
        if(deviceCount() < 1)
            GTEST_SKIP() << "no HIP device";

        const Device device(0);

        const std::vector<int> ranked{321, 322, 323};

        hipEvent_t start = nullptr;
        hipEvent_t stop  = nullptr;

        // The pool is empty, so a pair would be made for this stream, which HIP no
        // longer knows.
        const Key gone = keyOf(0x1210);
        ASSERT_EQ(visit(gone, identityOf(gone), ranked), 0);

        hipStream_t destroyed = nullptr;
        ASSERT_EQ(hipStreamCreate(&destroyed), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(destroyed), hipSuccess);
        ASSERT_EQ(hipGetLastError(), hipSuccess);

        EXPECT_FALSE(tuner().beginMeasurement(gone, ranked[0], destroyed, start, stop));
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(hipGetLastError(), hipSuccess)
            << "a pair that could not be made left an error in the caller's HIP state";

        // An error the caller has not read yet.
        const Key key = keyOf(0x1211);
        ASSERT_EQ(visit(key, identityOf(key), ranked), 0);

        ASSERT_NE(hipSetDevice(deviceCount()), hipSuccess) << "an out-of-range device was accepted";
        const hipError_t planted = hipPeekAtLastError();
        ASSERT_NE(planted, hipSuccess) << "the caller's error did not stick";

        EXPECT_FALSE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
            << "a launch was timed while the caller held an unread error";
        EXPECT_EQ(start, nullptr);
        EXPECT_EQ(hipGetLastError(), planted) << "the caller's error was replaced or cleared";

        // Once it is read, the same launch is timed.
        ASSERT_TRUE(tuner().beginMeasurement(key, ranked[0], device.stream(), start, stop))
            << "the declined launch lost its offer";
        device.bracket(key, start, stop);
        tuner().harvestPending(key);
        EXPECT_EQ(OnlineTunerTestAccess::samples(tuner(), key), 1u);
    }

    // The pool is keyed by device: hipEventRecord rejects another device's event
    // with hipErrorInvalidResourceHandle, which the launch would then carry.
    TEST_F(OnlineTuner_multi_gpu, APooledEventPairIsNotHandedToAnotherDevice)
    {
        if(deviceCount() < 2)
            GTEST_SKIP() << "needs two HIP devices";

        const Device zero(0);
        const Device one(1);

        const Key              keyZero = keyOf(0x1010);
        const Key              keyOne  = keyOf(0x1011);
        const std::vector<int> ranked{131, 132, 133};

        // Device 0 pools one pair.
        ASSERT_EQ(hipSetDevice(zero.id()), hipSuccess);
        hipEvent_t startZero = nullptr;
        hipEvent_t stopZero  = nullptr;
        ASSERT_EQ(visit(keyZero, identityOf(keyZero), ranked), 0);
        ASSERT_TRUE(
            tuner().beginMeasurement(keyZero, ranked[0], zero.stream(), startZero, stopZero));
        zero.bracket(keyZero, startZero, stopZero);
        tuner().harvestPending(keyZero);

        // Device 1 now explores a shape of its own.
        ASSERT_EQ(hipSetDevice(one.id()), hipSuccess);
        hipEvent_t startOne = nullptr;
        hipEvent_t stopOne  = nullptr;
        ASSERT_EQ(visit(keyOne, identityOf(keyOne), ranked), 0);
        ASSERT_TRUE(tuner().beginMeasurement(keyOne, ranked[0], one.stream(), startOne, stopOne));

        for(hipEvent_t given : {startOne, stopOne})
            for(hipEvent_t pooled : {startZero, stopZero})
                EXPECT_NE(given, pooled) << "device 1 was handed device 0's pair";

        // Whatever device 1 was given must be recordable there.
        EXPECT_EQ(hipEventRecord(startOne, one.stream()), hipSuccess)
            << "the pair handed to device 1 cannot be recorded there";
        EXPECT_EQ(hipEventRecord(stopOne, one.stream()), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(one.stream()), hipSuccess);

        ASSERT_EQ(hipSetDevice(zero.id()), hipSuccess);
    }

} // namespace

// The configuration this file's private copy of the tuner is built from, so the
// tests fix it to known values whatever the environment says.
rocblaslt::OnlineTunerConfig rocblaslt::onlineTunerConfig()
{
    OnlineTunerConfig config;
    config.enabled     = true;
    config.topK        = kTopK;
    config.repeats     = kRepeats;
    config.livePairCap = kLivePairCap;

    config.minImprovementPct  = minImprovementPct;
    config.livePairCapReached = recordCapReached;
    return config;
}
