// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_get_plan.h
//     Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_get_plan.h
 * @brief cuDNN v8 `EngineConfigGenerator::cudnnGetPlan` over the shim's v0.x builders.
 */

#pragma once

#include <cstddef>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfigGenerator.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_get_plan.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,performance-unnecessary-value-param):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

auto inline EngineConfigGenerator::cudnnGetPlan(cudnnHandle_t handle,
                                                OperationGraph& opGraph,
                                                size_t max_plans) -> executionPlans_t
{
    executionPlans_t plans;
    for(auto& engine_config : generate_engine_config(opGraph))
    {
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        try
        {
#endif
            plans.push_back(ExecutionPlanBuilder()
                                .setHandle(handle)
                                .setEngineConfig(engine_config, opGraph.getTag())
                                .build());
            CUDNN_FE_LOG_LABEL_ENDL("Added plan " << plans.back().getTag() << " "
                                                  << to_string(plans.back().get_status()));
#ifndef NV_CUDNN_DISABLE_EXCEPTION
        }
        catch(cudnnException& e)
        {
            CUDNN_FRONTEND_UNUSED(e);
            continue;
        }
#endif
        if(plans.size() >= max_plans)
        {
            break;
        }
    }
    return plans;
}

auto inline EngineConfigGenerator::cudnnGetPlan(cudnnHandle_t handle,
                                                OperationGraph& opGraph,
                                                Predicate pred,
                                                size_t max_plans) -> executionPlans_t
{
    executionPlans_t plans = cudnnGetPlan(handle, opGraph, max_plans);
    return filter(pred, plans);
}

// NOLINTEND(readability-identifier-naming,performance-unnecessary-value-param)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
