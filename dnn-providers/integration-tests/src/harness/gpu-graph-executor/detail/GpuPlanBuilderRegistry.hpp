// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <variant>

#include "GpuPlanRegistrySignatureKey.hpp"
#include "IGpuGraphNodePlanBuilder.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

using GpuPlanRegistryMap = std::unordered_map<GpuPlanRegistrySignatureKey,
                                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                              GpuPlanRegistrySignatureKeyHash,
                                              GpuPlanRegistrySignatureKeyEqual>;

class GpuPlanBuilderRegistry
{
public:
    GpuPlanBuilderRegistry() = default;

    GpuPlanBuilderRegistry(const GpuPlanBuilderRegistry&) = delete;
    GpuPlanBuilderRegistry& operator=(const GpuPlanBuilderRegistry&) = delete;
    GpuPlanBuilderRegistry(GpuPlanBuilderRegistry&&) = delete;
    GpuPlanBuilderRegistry& operator=(GpuPlanBuilderRegistry&&) = delete;

    const IGpuGraphNodePlanBuilder& getPlanBuilder(const GpuPlanRegistrySignatureKey& key)
    {
        initializeRegistry();

        auto it = _registry.find(key);
        if(it != _registry.end())
        {
            return *it->second;
        }

        std::ostringstream oss;
        oss << "No GPU plan builder registered for signature key: " << key;
        throw std::runtime_error(oss.str());
    }

private:
    void initializeRegistry()
    {
        if(!_initialized)
        {
            _initialized = true;
            registerBuildersForVariant(GpuPlanRegistrySignatureKey{});
        }
    }

    template <class T>
    void registerBuilder()
    {
        auto builders = T::getPlanBuilders();
        for(auto& [key, builder] : builders)
        {
            _registry[key] = std::move(builder);
        }
    }

    template <class... Ts>
    void registerBuildersForVariant([[maybe_unused]] std::variant<Ts...> var)
    {
        (registerBuilder<Ts>(), ...);
    }

    bool _initialized = false;
    GpuPlanRegistryMap _registry;
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
