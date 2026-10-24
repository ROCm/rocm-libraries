// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_find_plan.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_find_plan.h
 * @brief cuDNN v8 `time_sorted_plan` and `EngineConfigGenerator::cudnnFindPlan*` over the shim.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <set>
#include <utility>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfigGenerator.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_get_plan.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_shim.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_find_plan.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,misc-const-correctness,modernize-avoid-c-arrays,performance-unnecessary-value-param,readability-avoid-nested-conditional-operator):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

template <CudnnFindSamplingTechnique samplingTechnique>
auto time_sorted_plan(cudnnHandle_t handle,
                      executionPlans_t plans,
                      const VariantPack& variantPack,
                      uint64_t max_plans_to_evaluate = 1000) -> executionPlans_t
{
    executionPlans_t time_sorted_plans;

    auto plan_cmp = [](const ExecutionPlan& a, const ExecutionPlan& b) {
        return a.getExecutionTime() < b.getExecutionTime();
    };
    // A multiset keeps plans whose times tie; upstream's std::set dropped all but the first.
    std::multiset<std::reference_wrapper<ExecutionPlan>, decltype(plan_cmp)> timed_execution_plans(
        plan_cmp);

    const int maxIterCount
        = (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE) ? 1
          : (samplingTechnique == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE)
              ? 3
              : 100;
    const float threshhold = 0.95f;
    uint64_t successful_plan_count = 0;
    detail::PlanTimer timer(handle);
    static_cast<void>(detail::deviceSynchronize());

    for(auto& plan : plans)
    {
        float time_ms = 0.0f;
        float final_time_ms = 0.0f;
        float min_time_ms = std::numeric_limits<float>::max();

        auto warmup_status
            = detail::execute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());
        if(warmup_status != CUDNN_STATUS_SUCCESS)
        {
            CUDNN_FE_LOG_LABEL_ENDL("Plan " << plan.getTag() << " failed with "
                                            << to_string(warmup_status));
            continue;
        }
        successful_plan_count++;
        static_cast<void>(detail::deviceSynchronize());

        float time_run_ms[3] = {0.0f, 0.0f, 0.0f};
        for(int i = 0; i < maxIterCount; i++)
        {
            static_cast<void>(timer.start());

            detail::execute(handle, plan.get_raw_desc(), variantPack.get_raw_desc());

            static_cast<void>(timer.stop());
            static_cast<void>(timer.elapsedMs(&time_ms));

            if constexpr(samplingTechnique
                         == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE)
            {
                final_time_ms = std::min(min_time_ms, time_ms);
                if(time_ms / min_time_ms < threshhold)
                {
                    min_time_ms = final_time_ms;
                }
                else
                {
                    break;
                }
            }
            else if constexpr(samplingTechnique
                              == CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE)
            {
                time_run_ms[i] = time_ms;
                if(i == maxIterCount - 1)
                {
                    auto min_of_first_two = std::min(time_run_ms[0], time_run_ms[1]);
                    auto max_of_first_two = std::max(time_run_ms[0], time_run_ms[1]);
                    final_time_ms
                        = std::max(min_of_first_two, std::min(max_of_first_two, time_run_ms[2]));
                }
            }
            else
            {
                final_time_ms = time_ms;
            }
        }
        CUDNN_FE_LOG_LABEL_ENDL("Plan " << plan.getTag() << " took " << std::setw(10)
                                        << final_time_ms);
        plan.setExecutionTime(final_time_ms);
        timed_execution_plans.insert(plan);
        if(successful_plan_count >= max_plans_to_evaluate)
        {
            CUDNN_FE_LOG_LABEL_ENDL("Successfully profiled " << max_plans_to_evaluate << "plans.");
            break;
        }
    }

    for(ExecutionPlan& plan : timed_execution_plans)
    {
        time_sorted_plans.emplace_back(std::move(plan));
    }

    CUDNN_FE_LOG_LABEL_ENDL("Auto-tuning returns " << time_sorted_plans.size() << " plans.");

    return time_sorted_plans;
}

template <CudnnFindSamplingTechnique samplingTechnique>
auto EngineConfigGenerator::cudnnFindPlan(cudnnHandle_t handle,
                                          cudnn_frontend::OperationGraph& opGraph,
                                          const cudnn_frontend::VariantPack& variantPack)
    -> executionPlans_t
{
    executionPlans_t plans = cudnnGetPlan(handle, opGraph);
    return time_sorted_plan<samplingTechnique>(handle, std::move(plans), variantPack);
}

template <CudnnFindSamplingTechnique samplingTechnique>
auto EngineConfigGenerator::cudnnFindPlan(cudnnHandle_t handle,
                                          cudnn_frontend::OperationGraph& opGraph,
                                          const cudnn_frontend::VariantPack& variantPack,
                                          Predicate pred) -> executionPlans_t
{
    executionPlans_t plans = cudnnGetPlan(handle, opGraph, pred);
    return time_sorted_plan<samplingTechnique>(handle, std::move(plans), variantPack);
}

template <CudnnFindSamplingTechnique samplingTechnique>
auto EngineConfigGenerator::cudnnFindPlanAndCache(cudnnHandle_t handle,
                                                  cudnn_frontend::OperationGraph& opGraph,
                                                  const cudnn_frontend::VariantPack& variantPack,
                                                  cudnn_frontend::ExecutionPlanCache& cache,
                                                  Predicate pred) -> cudnn_frontend::ExecutionPlan
{
    auto sorted_plans = cudnnFindPlan<samplingTechnique>(handle, opGraph, variantPack, pred);
    // Upstream calls front() on an empty list when no plan ran.
    if(sorted_plans.empty())
    {
        set_error_and_throw_exception(
            nullptr, CUDNN_STATUS_NOT_SUPPORTED, "cudnnFindPlanAndCache: no execution plan ran");
        return ExecutionPlanBuilder_v8().build();
    }
    if(cache.is_fastest_plan_stable(opGraph, sorted_plans.front().getTag()))
    {
        cache.add_plan_to_cache(opGraph, sorted_plans.front());
    }
    return sorted_plans.front();
}

// NOLINTEND(readability-identifier-naming,misc-const-correctness,modernize-avoid-c-arrays,performance-unnecessary-value-param,readability-avoid-nested-conditional-operator)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
