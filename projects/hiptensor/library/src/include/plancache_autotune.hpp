/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "data_types.hpp"
#include "plan_cache.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    enum class AutotuneOps
    {
        Autotune_Contraction,
        Autotune_ContractionTrinary,
        Autotune_Permutation,
        Autotune_BinaryOp,
        Autotune_TrinaryOp,
        Autotune_Redution
    };

    class PlancacheAutotuneMgr;

    // One execution's part in Plan Cache autotuning: the solution the execution runs and,
    // during incremental autotuning, the candidate it reserved. Executions that share a plan
    // each get their own trial, so none of them runs or records another one's candidate.
    // A trial with a reserved candidate that is destroyed before save() or fail() counts as a
    // failed run, so an execution that returns early does not stall its round.
    template <typename T>
    class AutotuneTrial
    {
    public:
        AutotuneTrial()                                = default;
        AutotuneTrial(AutotuneTrial const&)            = delete;
        AutotuneTrial& operator=(AutotuneTrial const&) = delete;
        AutotuneTrial(AutotuneTrial&& other) noexcept
        {
            *this = std::move(other);
        }
        AutotuneTrial& operator=(AutotuneTrial&& other) noexcept
        {
            if(this != &other)
            {
                if(mRound != 0)
                    fail();
                mType     = other.mType;
                mHandle   = std::exchange(other.mHandle, nullptr);
                mPlan     = std::exchange(other.mPlan, nullptr);
                mSolution = std::exchange(other.mSolution, nullptr);
                mRound    = std::exchange(other.mRound, 0);
                mCandidate = other.mCandidate;
                mDirect    = other.mDirect;
            }
            return *this;
        }
        ~AutotuneTrial()
        {
            if(mRound != 0)
                fail();
        }

        // The solution to run: the reserved candidate, the cached solution or the plan's
        // solution. nullptr when the plan has no solution yet.
        T* solution() const
        {
            return mSolution;
        }

        // Records a successful run of solution taking time.
        void save(T* solution, float time);

        // Records a failed run. The failed candidate is not retried.
        void fail();

    private:
        friend class PlancacheAutotuneMgr;

        AutotuneOps       mType     = AutotuneOps::Autotune_Contraction;
        hiptensorHandle_t mHandle   = nullptr; // nullptr when there is nothing to record
        hiptensorPlan_t   mPlan     = nullptr;
        T*                mSolution = nullptr;
        uint64_t          mRound    = 0; // round of the reserved candidate, 0 if none
        uint32_t          mCandidate = 0; // index of the reserved candidate in its round
        bool              mDirect = false; // no round to join: save() caches the solution that ran
    };

    class PlancacheAutotuneMgr : public LazySingleton<PlancacheAutotuneMgr>
    {
    private:
        // Incremental autotune state of one problem
        // The candidates and the size of a round are taken from the plan that starts it and
        // kept for the whole round: plans of the same problem may differ in their counts, and
        // calls through any of them reserve and complete the same round.
        // A round is closed when the problem gets a cached kernel some other way: later calls
        // use that kernel and reserve nothing, so the round ends once its pending candidates
        // are recorded.
        struct AutotuneState
        {
            bool               mClosed = false;
            uint64_t           mRound = 0; //Identifies the round, see mNextRound
            std::vector<void*> mCandidates; //Candidates the round tries, in order
            uint32_t           mSize          = 0; //Number of calls the round records
            uint32_t           mNextCandidate = 0; //Index of the next candidate to hand out
            uint32_t           mPending       = 0; //Candidates handed out but not recorded yet
            uint32_t           mCallCount = 0; //Number of calls (timed or failed) recorded so far
            float              mBestTime  = std::numeric_limits<float>::max();
            uint32_t           mBestIndex = std::numeric_limits<uint32_t>::max();
            void*              mBestSolution = nullptr; //Fastest solution timed so far
        };

        // Autotune state is tracked per operation, plan cache, problem and set of candidates, so
        // concurrent calls on different problems or handles do not mix their timings. The
        // problem hash leaves out the unary operators, which restrict the candidates, so plans
        // of one problem can select from different kernels; they get separate rounds. An entry
        // is created on first use and erased once its result is written to the plan cache.
        using AutotuneKey = std::tuple<AutotuneOps, std::uintptr_t, PlanCache::HashId, std::size_t>;
        std::map<AutotuneKey, AutotuneState> mAutotuneStates;

        // Every round gets a new number, so a trial cannot be recorded in a later round of
        // the same problem.
        uint64_t mNextRound = 1;

        // The manager is process-wide and every execute call goes through it,
        // so all access to the members above must hold this lock.
        std::mutex mMutex;

        //Singleton: only one instance
        PlancacheAutotuneMgr() {}
        PlancacheAutotuneMgr(PlancacheAutotuneMgr const&)            = delete;
        PlancacheAutotuneMgr(PlancacheAutotuneMgr&&)                 = delete;
        PlancacheAutotuneMgr& operator=(PlancacheAutotuneMgr const&) = delete;
        PlancacheAutotuneMgr& operator=(PlancacheAutotuneMgr&&)      = delete;

        // Identifies the set of candidates of a plan, whatever their order.
        static std::size_t candidateSetId(const hiptensorPlan_t plan)
        {
            std::uint64_t id = 0;
            for(auto* candidate : plan->mPref->mCandidates)
            {
                // Mix each address (splitmix64 finalizer) and add, so the order does not matter.
                auto x = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(candidate));
                x      = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
                x      = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
                id += x ^ (x >> 31);
            }
            return static_cast<std::size_t>(id);
        }

        static AutotuneKey makeKey(const AutotuneOps       autotuneType,
                                   const hiptensorHandle_t handle,
                                   const hiptensorPlan_t   plan)
        {
            auto* planCache = handle->getPlanCache();
            return {autotuneType,
                    reinterpret_cast<std::uintptr_t>(planCache),
                    planCache->getHashID(plan->mOpDesc),
                    candidateSetId(plan)};
        }

        // Closes the rounds of a problem that now has a cached kernel. Must hold mMutex.
        void closeRoundsLocked(const AutotuneOps       autotuneType,
                               std::uintptr_t          planCacheId,
                               const PlanCache::HashId hashID)
        {
            auto it = mAutotuneStates.lower_bound({autotuneType, planCacheId, hashID, 0});
            while(it != mAutotuneStates.end() && std::get<0>(it->first) == autotuneType
                  && std::get<1>(it->first) == planCacheId && std::get<2>(it->first) == hashID)
            {
                if(it->second.mPending == 0)
                {
                    it = mAutotuneStates.erase(it);
                }
                else
                {
                    it->second.mClosed = true;
                    ++it;
                }
            }
        }

        void closeRounds(const AutotuneOps       autotuneType,
                         PlanCache const*        planCache,
                         const PlanCache::HashId hashID)
        {
            std::scoped_lock lock(mMutex);
            closeRoundsLocked(autotuneType, reinterpret_cast<std::uintptr_t>(planCache), hashID);
        }

        // Runs the plan cache's kernel for the problem, if it is one of the plan's candidates
        // (a plan created from a cache hit may have no candidate list); otherwise keeps the
        // plan's own solution.
        template <typename T>
        static void useCachedSolution(AutotuneTrial<T>&     trial,
                                      const hiptensorPlan_t plan,
                                      PlanCache::Uid        Uid)
        {
            //If the plan already uses the cached solution, skip searching the candidates
            if(trial.mSolution != nullptr && trial.mSolution->uid() == Uid)
                return;

            //If there is a solution already in the Plan Cache, directly set solution from Plan Cache
            auto candidates = std::vector<T*>(plan->mPref->mCandidates.size());
            std::transform(plan->mPref->mCandidates.begin(),
                           plan->mPref->mCandidates.end(),
                           candidates.begin(),
                           [](auto* p) { return (T*)p; });

            if(auto* cached = findSolutionByUid(candidates, Uid))
            {
                plan->mPref->mSolution.store(cached);
                trial.mSolution = cached;
            }
        }

        // Number of calls a round records. A plan with candidates tries each of them once, up
        // to mIncrementalCount; only the calls that reserved a candidate are recorded. A plan
        // without candidates (e.g. elementwise) records mIncrementalCount calls of its solution.
        static uint32_t roundSize(const hiptensorPlan_t plan)
        {
            auto count = static_cast<std::size_t>(std::max(plan->mPref->mIncrementalCount, 0));
            if(!plan->mPref->mCandidates.empty())
            {
                count = std::min(count, plan->mPref->mCandidates.size());
            }
            return static_cast<uint32_t>(count);
        }

        // Records one autotune call: a successful run of solution taking time, or a failed run
        // when solution is nullptr. A failed run still advances the round (so a candidate that
        // cannot run does not stall autotuning) but is never selected. Once the round has
        // recorded all its calls, the fastest successful solution is added to the Plan Cache.
        template <typename T>
        void recordAutotune(AutotuneTrial<T> const& trial, T* solution, float time)
        {
            auto key = makeKey(trial.mType, trial.mHandle, trial.mPlan);

            std::scoped_lock lock(mMutex);

            // No state means the round already completed (or was never started by
            // setAutotune): do not start a new round that would never be completed.
            auto it = mAutotuneStates.find(key);
            if(it == mAutotuneStates.end())
                return;

            auto& state = it->second;
            if(trial.mRound != 0)
            {
                // The trial's round already completed and a new one started.
                if(trial.mRound != state.mRound)
                    return;
                state.mPending--;
            }
            else if(!state.mCandidates.empty())
            {
                // The call ran the plan's solution, which is not a candidate of this round.
                return;
            }

            //For Plan Cache autotuning, remember the best performace solution.
            //Untimed runs all take 0 ms: a tie goes to the earlier candidate, which ranked
            //faster in the plan's kernel selection, whatever order the calls finish in.
            auto index
                = trial.mRound != 0 ? trial.mCandidate : std::numeric_limits<uint32_t>::max();
            if(solution != nullptr
               && (time < state.mBestTime || (time == state.mBestTime && index < state.mBestIndex)))
            {
                state.mBestTime     = time;
                state.mBestIndex    = index;
                state.mBestSolution = solution;
            }

            //Count the number of calling this function for Plan Cache autotuning
            state.mCallCount++;

            //When every call of the round is recorded, add the best solution to the Plan Cache.
            //Waiting for the pending candidates keeps a slow candidate from being dropped.
            //A closed round ends with its pending candidates and keeps the cached kernel.
            if(state.mPending == 0 && (state.mCallCount >= state.mSize || state.mClosed))
            {
                bool const cache
                    = state.mCallCount >= state.mSize && state.mBestSolution != nullptr;
                if(cache)
                {
                    trial.mHandle->getPlanCache()->addCacheLine(
                        std::get<2>(key), static_cast<T*>(state.mBestSolution)->uid());
                }

                //Reset for the next round of autotuning
                mAutotuneStates.erase(it);

                //The problem has a cached kernel now: its other rounds get no more calls.
                if(cache)
                {
                    closeRoundsLocked(std::get<0>(key), std::get<1>(key), std::get<2>(key));
                }
            }
        }

        template <typename T>
        friend class AutotuneTrial;

    public:
        friend std::unique_ptr<PlancacheAutotuneMgr> std::make_unique<PlancacheAutotuneMgr>();

        //Picks the solution an execution runs and returns it in a trial, to be saved after the
        //run. The plan's solution is not changed for incremental autotuning: executions sharing
        //the plan each run the candidate in their own trial.
        template <typename T>
        AutotuneTrial<T> setAutotune(const AutotuneOps       autotuneType,
                                     const hiptensorHandle_t handle,
                                     const hiptensorPlan_t   plan)
        {
            AutotuneTrial<T> trial;
            if(plan == nullptr || plan->mPref == nullptr)
                return trial;
            trial.mSolution = static_cast<T*>(plan->mPref->mSolution.load());

            if(handle == nullptr)
                return trial;
            if(handle->getPlanCache() == nullptr)
                return trial;
            if(plan->mPref->mCacheMode != HIPTENSOR_CACHE_MODE_PEDANTIC)
                return trial;

            trial.mType   = autotuneType;
            trial.mHandle = handle;
            trial.mPlan   = plan;

            // Look for solution from memory cache (Plan Cache) or do autotuning for Plan Cache.
            // The plan cache has its own lock, and the plan's solution is atomic, so this
            // lookup does not take mMutex: holding it here would serialize every call.
            auto Uid = handle->getPlanCache()->querySolutionUid(plan->mOpDesc);
            if(Uid > 0ull)
            {
                useCachedSolution(trial, plan, Uid);
            }
            else if(plan->mPref->mAutotuneMode == HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
            {
                auto key = makeKey(autotuneType, handle, plan);

                std::scoped_lock lock(mMutex);

                // Another call may have completed a round and cached its kernel since the
                // lookup above. Completing a round caches under mMutex, so this check and the
                // round below cannot miss it.
                Uid = handle->getPlanCache()->querySolutionUid(plan->mOpDesc);
                if(Uid > 0ull)
                {
                    useCachedSolution(trial, plan, Uid);
                    return trial;
                }

                //If there is no solution in the Plan Cache, we may need to do autotuning based on the number of calling this function.
                //Candidates are handed out by a separate counter so that concurrent calls, which
                //all start before any of them is recorded, each try a different candidate.
                auto it = mAutotuneStates.find(key);
                if(it == mAutotuneStates.end())
                {
                    // A plan that tries nothing (an incremental count of 0) joins a round of
                    // another plan but does not start one: save() caches the solution that ran.
                    auto size = roundSize(plan);
                    if(size == 0)
                    {
                        trial.mDirect = true;
                        return trial;
                    }

                    AutotuneState state;
                    state.mRound      = mNextRound++;
                    state.mCandidates = plan->mPref->mCandidates;
                    state.mSize       = size;
                    it                = mAutotuneStates.emplace(key, std::move(state)).first;
                }

                auto& state = it->second;
                if(!state.mClosed && !state.mCandidates.empty()
                   && state.mNextCandidate < state.mSize)
                {
                    //If the number of calling this function is less than the number of different candidates to explore,
                    //we are doing the autotuning to find the best solution
                    // TODO: need a ranked solution array to explore from fastest to slowest
                    auto* candidate = state.mCandidates[state.mNextCandidate];

                    // Rounds are kept apart by their set of candidates, so the candidate is one
                    // of the plan's; this only guards against two sets with the same id.
                    auto const& own = plan->mPref->mCandidates;
                    if(std::find(own.begin(), own.end(), candidate) != own.end())
                    {
                        trial.mSolution  = static_cast<T*>(candidate);
                        trial.mRound     = state.mRound;
                        trial.mCandidate = state.mNextCandidate;
                        state.mNextCandidate++;
                        state.mPending++;
                    }
                }
            }
            return trial;
        }

        //Drop the autotune state of a plan cache that is being destroyed, so that a new
        //plan cache allocated at the same address does not inherit its unfinished rounds.
        void releasePlanCache(hiptensor::PlanCache const* planCache)
        {
            if(planCache == nullptr)
                return;

            auto id = reinterpret_cast<std::uintptr_t>(planCache);

            std::scoped_lock lock(mMutex);
            std::erase_if(mAutotuneStates,
                          [id](auto const& entry) { return std::get<1>(entry.first) == id; });
        }
    };

    //Check and save autotune results to Plan Cache.
    //solution is the kernel that produced time.
    template <typename T>
    void AutotuneTrial<T>::save(T* solution, float time)
    {
        if(mHandle == nullptr)
            return;
        if(solution == nullptr)
        {
            fail();
            return;
        }

        if(mPlan->mPref->mAutotuneMode != HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL || mDirect)
        {
            //There is no autotuning, or it tries no candidate (an incremental count of 0):
            //add the solution that ran to the Plan Cache.
            auto* planCache = mHandle->getPlanCache();
            auto  hashID    = planCache->getHashID(mPlan->mOpDesc);
            planCache->addCacheLine(hashID, solution->uid());

            //Later calls use this kernel and reserve no more candidates: close any open round.
            //Caching before closing means a round created after the close sees the cache hit.
            PlancacheAutotuneMgr::instance()->closeRounds(mType, planCache, hashID);
        }
        else
        {
            PlancacheAutotuneMgr::instance()->recordAutotune(*this, solution, time);
        }
        mHandle = nullptr;
        mRound  = 0;
    }

    //Record a failed execution during incremental autotuning. The candidate that failed
    //is skipped: the round moves on to the next candidate instead of retrying it forever.
    template <typename T>
    void AutotuneTrial<T>::fail()
    {
        if(mHandle == nullptr)
            return;

        if(mPlan->mPref->mAutotuneMode == HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
        {
            PlancacheAutotuneMgr::instance()->recordAutotune<T>(*this, nullptr, 0.0f);
        }
        mHandle = nullptr;
        mRound  = 0;
    }

} // namespace hiptensor
