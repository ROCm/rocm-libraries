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

#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
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

    class PlancacheAutotuneMgr : public LazySingleton<PlancacheAutotuneMgr>
    {
    private:
        // Incremental autotune state of one problem
        struct AutotuneState
        {
            uint32_t mNextCandidate = 0; //Index of the next candidate to hand out
            uint32_t mCallCount     = 0; //Number of calls (timed or failed) so far
            float    mBestTime      = std::numeric_limits<float>::max();
            void*    mBestSolution  = nullptr; //Fastest solution timed so far
        };

        // Autotune state is tracked per operation, plan cache and problem, so concurrent calls
        // on different problems or handles do not mix their timings. An entry is created on
        // first use and erased once its result is written to the plan cache.
        using AutotuneKey = std::tuple<AutotuneOps, std::uintptr_t, PlanCache::HashId>;
        std::map<AutotuneKey, AutotuneState> mAutotuneStates;

        // The manager is process-wide and every execute call goes through it,
        // so all access to the map above must hold this lock.
        std::mutex mMutex;

        //Singleton: only one instance
        PlancacheAutotuneMgr() {}
        PlancacheAutotuneMgr(PlancacheAutotuneMgr const&)            = delete;
        PlancacheAutotuneMgr(PlancacheAutotuneMgr&&)                 = delete;
        PlancacheAutotuneMgr& operator=(PlancacheAutotuneMgr const&) = delete;
        PlancacheAutotuneMgr& operator=(PlancacheAutotuneMgr&&)      = delete;

        static AutotuneKey makeKey(const AutotuneOps       autotuneType,
                                   const hiptensorHandle_t handle,
                                   const hiptensorPlan_t   plan)
        {
            auto* planCache = handle->getPlanCache();
            return {autotuneType,
                    reinterpret_cast<std::uintptr_t>(planCache),
                    planCache->getHashID(plan->mOpDesc)};
        }

        // Records one autotune call: a successful run of solution taking time, or a failed run
        // when solution is nullptr. A failed run still advances the round (so a candidate that
        // cannot run does not stall autotuning) but is never selected. Once the round has seen
        // mIncrementalCount calls, the fastest successful solution is added to the Plan Cache.
        template <typename T>
        void recordAutotune(const AutotuneOps       autotuneType,
                            T*                      solution,
                            float                   time,
                            const hiptensorHandle_t handle,
                            const hiptensorPlan_t   plan)
        {
            auto key = makeKey(autotuneType, handle, plan);

            std::scoped_lock lock(mMutex);

            // No state means the round already completed (or was never started by
            // setAutotune): do not start a new round that would never be completed.
            auto it = mAutotuneStates.find(key);
            if(it == mAutotuneStates.end())
                return;

            //For Plan Cache autotuning, remember the best performace solution
            auto& state = it->second;
            if(solution != nullptr && time < state.mBestTime)
            {
                state.mBestTime     = time;
                state.mBestSolution = solution;
            }

            //Count the number of calling this function for Plan Cache autotuning
            state.mCallCount++;

            //When the number of calling this function is equal to the number of different candidates to explore,
            //add the best solution to the Plan Cache.
            if(state.mCallCount >= plan->mPref->mIncrementalCount)
            {
                if(state.mBestSolution != nullptr)
                {
                    handle->getPlanCache()->addCacheLine(
                        std::get<2>(key), static_cast<T*>(state.mBestSolution)->uid());
                }

                //Reset for the next round of autotuning
                mAutotuneStates.erase(it);
            }
        }

    public:
        friend std::unique_ptr<PlancacheAutotuneMgr> std::make_unique<PlancacheAutotuneMgr>();

        //Set current solution to try in autotune
        template <typename T>
        void setAutotune(const AutotuneOps       autotuneType,
                         const hiptensorHandle_t handle,
                         const hiptensorPlan_t   plan)
        {
            if(handle == nullptr || plan == nullptr)
                return;
            if(handle->getPlanCache() == nullptr)
                return;
            if(plan->mPref == nullptr)
                return;
            if(plan->mPref->mCacheMode != HIPTENSOR_CACHE_MODE_PEDANTIC)
                return;

            // Look for solution from memory cache (Plan Cache) or do autotuning for Plan Cache.
            // The plan cache has its own lock, and the plan's solution is atomic, so this
            // lookup does not take mMutex: holding it here would serialize every call.
            auto Uid = handle->getPlanCache()->querySolutionUid(plan->mOpDesc);
            if(Uid > 0ull)
            {
                //If the plan already uses the cached solution, skip searching the candidates
                auto* current = static_cast<T*>(plan->mPref->mSolution.load());
                if(current != nullptr && current->uid() == Uid)
                    return;

                //If there is a solution already in the Plan Cache, directly set solution from Plan Cache
                auto candidates = std::vector<T*>(plan->mPref->mCandidates.size());
                std::transform(plan->mPref->mCandidates.begin(),
                               plan->mPref->mCandidates.end(),
                               candidates.begin(),
                               [](auto* p) { return (T*)p; });

                //Keep the current solution if the cached one is not among the candidates
                //(e.g. a plan created from a cache hit may have no candidate list)
                if(auto* cached = findSolutionByUid(candidates, Uid))
                    plan->mPref->mSolution.store(cached);
            }
            else if(plan->mPref->mAutotuneMode == HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
            {
                auto key = makeKey(autotuneType, handle, plan);

                std::scoped_lock lock(mMutex);

                //If there is no solution in the Plan Cache, we may need to do autotuning based on the number of calling this function.
                //Candidates are handed out by a separate counter so that concurrent calls, which
                //all start before any of them is recorded, each try a different candidate.
                auto&    state = mAutotuneStates[key];
                uint32_t next  = state.mNextCandidate;
                if(next < plan->mPref->mIncrementalCount)
                {
                    //If the number of calling this function is less than the number of different candidates to explore,
                    //we are doing the autotuning to find the best solution
                    if(next < plan->mPref->mCandidates.size())
                    {
                        plan->mPref->mSolution.store(plan->mPref->mCandidates[next]);
                        state.mNextCandidate++;
                    }
                    // TODO: need a ranked solution array to explore from fastest to slowest
                }
            }
        }

        //Check and save autotune results to Plan Cache.
        //solution is the kernel that produced time. Callers pass the solution they ran rather
        //than letting this re-read the plan, which another thread sharing the plan may have
        //changed in the meantime.
        template <typename T>
        void saveAutotune(const AutotuneOps       autotuneType,
                          T*                      solution,
                          float                   time,
                          const hiptensorHandle_t handle,
                          const hiptensorPlan_t   plan)
        {
            if(solution == nullptr || handle == nullptr || plan == nullptr)
                return;
            if(handle->getPlanCache() == nullptr)
                return;
            if(plan->mPref == nullptr)
                return;
            if(plan->mPref->mCacheMode != HIPTENSOR_CACHE_MODE_PEDANTIC)
                return;

            if(plan->mPref->mAutotuneMode != HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
            {
                //There is no autotuning: add the solution that ran to the Plan Cache.
                auto hashID = handle->getPlanCache()->getHashID(plan->mOpDesc);
                handle->getPlanCache()->addCacheLine(hashID, solution->uid());
                return;
            }

            recordAutotune(autotuneType, solution, time, handle, plan);
        }

        //Record a failed execution during incremental autotuning. The candidate that failed
        //is skipped: the round moves on to the next candidate instead of retrying it forever.
        template <typename T>
        void failAutotune(const AutotuneOps       autotuneType,
                          const hiptensorHandle_t handle,
                          const hiptensorPlan_t   plan)
        {
            if(handle == nullptr || plan == nullptr)
                return;
            if(handle->getPlanCache() == nullptr)
                return;
            if(plan->mPref == nullptr)
                return;
            if(plan->mPref->mCacheMode != HIPTENSOR_CACHE_MODE_PEDANTIC)
                return;
            if(plan->mPref->mAutotuneMode != HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
                return;

            recordAutotune<T>(autotuneType, nullptr, 0.0f, handle, plan);
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

} // namespace hiptensor
