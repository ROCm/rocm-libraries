// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>

#include "harness/Benchmark.hpp"
#include "harness/BenchmarkReport.hpp"
#include "harness/PlanWorkspace.hpp"
#include "harness/VariantPack.hpp"

namespace hipdnn_integration_tests
{

/// Times the engine of the plan `graph` already built, on the tensors in
/// `variantPack`. Graph::autotune() does the timing, limited to that one engine,
/// so the warmup, the stall gate, the retries and the statistics are the same as
/// in autotune.
inline hipdnn_frontend::AutotuneResult benchmarkGraph(hipdnn_frontend::graph::Graph& graph,
                                                      hipdnnHandle_t handle,
                                                      VariantPack& variantPack,
                                                      const BenchmarkSettings& settings)
{
    hipdnn_frontend::AutotuneResult failed;

    int64_t engineId = -1;
    if(auto status = graph.get_execution_plan_engine_id(engineId); !status.is_good())
    {
        failed.errorMessage = status.get_message();
        return failed;
    }

    auto workspace = allocatePlanWorkspace(graph, failed.errorMessage);
    if(!workspace.has_value())
    {
        return failed;
    }

    hipdnn_frontend::AutotuneConfig config;
    config.strategy = hipdnn_frontend::AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = settings.timedIterations;
    config.warmupIterations = settings.warmupIterations;
    config.engineIdFilter = {engineId};

    std::vector<hipdnn_frontend::AutotuneResult> results;
    const auto status = graph.autotune(handle, variantPack, workspace->get(), config, {}, &results);
    return selectEngineResult(engineId, results, status);
}

} // namespace hipdnn_integration_tests
