// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdPlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormTrainPlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionBwdPlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionFwdPlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PointwisePlan.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanRegistrySignatureKey.hpp>

namespace hipdnn_sdk::test_utilities
{

typedef std::unordered_map<PlanRegistrySignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           PlanRegistrySignatureKeyHash,
                           PlanRegistrySignatureKeyEqual>
    PlanRegistryMap;

class PlanBuilderRegistry
{
public:
    PlanBuilderRegistry() = default;

    PlanBuilderRegistry(const PlanBuilderRegistry&) = delete;
    PlanBuilderRegistry& operator=(const PlanBuilderRegistry&) = delete;
    PlanBuilderRegistry(PlanBuilderRegistry&&) = delete;
    PlanBuilderRegistry& operator=(PlanBuilderRegistry&&) = delete;

    const IGraphNodePlanBuilder& getPlanBuilder(const PlanRegistrySignatureKey& key)
    {
        initializeRegistry();

        auto it = _registry.find(key);
        if(it != _registry.end())
        {
            return *it->second;
        }

        throw std::runtime_error("No plan builder registered for the given signature key.");
    }

private:
    void initializeRegistry()
    {
        if(!_initialized)
        {
            _initialized = true;
            initializePlanBuilders();
        }
    }

    void initializePlanBuilders()
    {
        registerBuildersForVariant(PlanRegistrySignatureKey{});
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
    PlanRegistryMap _registry;
};
}
