// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_EngineConfigGenerator.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_EngineConfigGenerator.h
 * @brief cuDNN v8 `EngineConfigGenerator` and plan `filter` over the shim's v0.x builders.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_EngineConfigGenerator.h at the new tag, then update this "
              "assert.");

// NOLINTBEGIN(readability-identifier-naming,performance-for-range-copy,performance-unnecessary-value-param,readability-container-size-empty,readability-implicit-bool-conversion):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

using executionPlans_t = std::vector<cudnn_frontend::ExecutionPlan>;
using Predicate = std::function<bool(const cudnn_frontend::ExecutionPlan& plan)>;
using GeneratorSource
    = std::function<cudnn_frontend::EngineConfigList(cudnn_frontend::OperationGraph&)>;

enum class CudnnFindSamplingTechnique
{
    CUDNN_FIND_SAMPLE_ONCE,
    CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE,
    CUDNN_FIND_SAMPLE_TILL_STABLE
};

class EngineConfigGenerator
{
private:
    std::vector<GeneratorSource> engine_config_generators;

public:
    EngineConfigGenerator(const int sourceSize, const GeneratorSource* sources)
    {
        for(int i = 0; i < sourceSize; i++)
        {
            engine_config_generators.push_back(sources[i]);
        }
    };

    auto generate_engine_config(cudnn_frontend::OperationGraph& opGraph)
        -> cudnn_frontend::EngineConfigList
    {
        cudnn_frontend::EngineConfigList engine_configs;
        for(auto fn : engine_config_generators)
        {
            cudnn_frontend::EngineConfigList new_engine_config = fn(opGraph);
            CUDNN_FE_LOG_LABEL_ENDL("Called engine config generator and produced "
                                    << new_engine_config.size() << " configs.");
            std::copy(new_engine_config.begin(),
                      new_engine_config.end(),
                      std::back_inserter(engine_configs));
            new_engine_config.clear();
        }
        return engine_configs;
    }

    auto cudnnGetPlan(cudnnHandle_t handle,
                      cudnn_frontend::OperationGraph& opGraph,
                      Predicate pred,
                      size_t max_plans = 1000) -> executionPlans_t;
    auto cudnnGetPlan(cudnnHandle_t handle,
                      cudnn_frontend::OperationGraph& opGraph,
                      size_t max_plans = 1000) -> executionPlans_t;

    template <CudnnFindSamplingTechnique samplingTechnique>
    auto cudnnFindPlan(cudnnHandle_t handle,
                       cudnn_frontend::OperationGraph& opGraph,
                       const cudnn_frontend::VariantPack& variantPack,
                       Predicate pred) -> executionPlans_t;

    template <CudnnFindSamplingTechnique samplingTechnique>
    auto cudnnFindPlan(cudnnHandle_t handle,
                       cudnn_frontend::OperationGraph& opGraph,
                       const cudnn_frontend::VariantPack& variantPack) -> executionPlans_t;

    template <CudnnFindSamplingTechnique samplingTechnique>
    auto cudnnFindPlanAndCache(
        cudnnHandle_t handle,
        cudnn_frontend::OperationGraph& opGraph,
        const cudnn_frontend::VariantPack& variantPack,
        cudnn_frontend::ExecutionPlanCache& cache,
        Predicate pred = [](const cudnn_frontend::ExecutionPlan&) {
            return false;
        }) -> cudnn_frontend::ExecutionPlan;
};

inline auto filter(Predicate pred, executionPlans_t& plans) -> executionPlans_t
{
    executionPlans_t filtered_plans;
    for(auto& plan : plans)
    {
        CUDNN_FE_LOG_LABEL("Filtered ");
        if(!pred(plan))
        {
            CUDNN_FE_LOG("and Added ");
            filtered_plans.emplace_back(std::move(plan));
        }
        if(filtered_plans.size())
        {
            CUDNN_FE_LOG(filtered_plans.back().getTag());
        }
    }
    CUDNN_FE_LOG_LABEL_ENDL("Filtered plans count " << filtered_plans.size());
    return filtered_plans;
}

// NOLINTEND(readability-identifier-naming,performance-for-range-copy,performance-unnecessary-value-param,readability-container-size-empty,readability-implicit-bool-conversion)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
