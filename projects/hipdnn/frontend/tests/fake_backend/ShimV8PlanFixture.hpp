// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ShimV8BuilderPlanFixture plus a config's dense engine index, from the engine
// the mock hipDNN ENGINECFG behind it was made for, and an INSTANT-heuristics
// config generator.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfigGenerator.h>

#include "ShimV8BuilderPlanFixture.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace hipdnn_shim_test
{

class ShimV8PlanFixture : public ShimV8BuilderPlanFixture
{
protected:
    // A config the mock did not make fails the test and reads as -1.
    std::int64_t indexOf(cudnnBackendDescriptor_t config) const
    {
        const std::optional<std::int64_t> id = knownEngineIdOf(config);
        if(!id)
        {
            ADD_FAILURE() << "not a mock hipDNN engine config";
            return -1;
        }
        return denseIndex(enumerableEngineIds(), *id);
    }

    std::vector<std::int64_t> indicesOf(const v8fe::EngineConfigList& configs) const
    {
        std::vector<std::int64_t> indices;
        indices.reserve(configs.size());
        for(const v8fe::ManagedOpaqueDescriptor& config : configs)
        {
            indices.push_back(indexOf(config->get_backend_descriptor()));
        }
        return indices;
    }

    static v8fe::EngineConfigList heuristicSource(v8fe::OperationGraph& graph)
    {
        v8fe::EngineHeuristics heuristics = v8fe::EngineHeuristicsBuilder()
                                                .setOperationGraph(graph)
                                                .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                                .build();
        return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    }

    static v8fe::EngineConfigGenerator heuristicGenerator()
    {
        const std::array<v8fe::GeneratorSource, 1> sources = {heuristicSource};
        return {static_cast<int>(sources.size()), sources.data()};
    }
};

} // namespace hipdnn_shim_test
