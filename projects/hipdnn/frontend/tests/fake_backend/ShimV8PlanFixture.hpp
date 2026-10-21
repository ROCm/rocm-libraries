// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ShimV8BuilderPlanFixture plus a config's dense engine index, from the engine
// the mock hipDNN ENGINECFG behind it was made for.
#pragma once

#include "ShimV8BuilderPlanFixture.hpp"

#include <cstdint>
#include <vector>

namespace hipdnn_shim_test
{

class ShimV8PlanFixture : public ShimV8BuilderPlanFixture
{
protected:
    std::int64_t indexOf(cudnnBackendDescriptor_t config) const
    {
        return denseIndex(enumerableEngineIds(), engineIdOf(config));
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
};

} // namespace hipdnn_shim_test
