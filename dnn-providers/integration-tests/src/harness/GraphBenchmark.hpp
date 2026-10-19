// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>

#include "harness/Benchmark.hpp"
#include "harness/PlanWorkspace.hpp"
#include "harness/VariantPack.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"

namespace hipdnn_integration_tests
{

/// Times the plan `graph` already built, on the inputs and outputs in `variantPack`.
inline BenchmarkResult benchmarkGraph(hipdnn_frontend::graph::Graph& graph,
                                      hipdnnHandle_t handle,
                                      VariantPack& variantPack,
                                      const BenchmarkSettings& settings)
{
    BenchmarkResult failed;

    int64_t engineId = -1;
    if(auto status = graph.get_execution_plan_engine_id(engineId); !status.is_good())
    {
        failed.error = status.get_message();
        return failed;
    }

    auto workspace = allocatePlanWorkspace(graph, failed.error);
    if(!workspace.has_value())
    {
        return failed;
    }

    auto result = runBenchmarkLoop(
        settings,
        [&] { return graph.execute(handle, variantPack, workspace->get()); },
        [&](hipdnn_frontend::ExecutionTiming& timing) {
            return graph.execute_timed_ext(handle, variantPack, workspace->get(), timing);
        });

    // Plugin engines are not in the static name registry, so ask the loaded table first.
    result.engine = hipdnn_data_sdk::utilities::engineNameOrHex(engineId);
    for(const auto& engine : bundle::LoadedEngineTable::get().all())
    {
        if(engine.id == engineId)
        {
            result.engine = engine.name;
        }
    }
    return result;
}

} // namespace hipdnn_integration_tests
