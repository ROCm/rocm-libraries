// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_test_sdk/detail/BatchnormBwdPlan.hpp>
#include <hipdnn_test_sdk/detail/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_test_sdk/detail/BatchnormFwdInferenceWithVariancePlan.hpp>
#include <hipdnn_test_sdk/detail/BatchnormTrainPlan.hpp>
#include <hipdnn_test_sdk/detail/ConvolutionBwdPlan.hpp>
#include <hipdnn_test_sdk/detail/ConvolutionFwdPlan.hpp>
#include <hipdnn_test_sdk/detail/MatmulPlan.hpp>
#include <hipdnn_test_sdk/detail/PlanRegistrySignatureKey.hpp>
#include <hipdnn_test_sdk/detail/PointwisePlan.hpp>

#include <hipdnn_data_sdk/logging/Logger.hpp>

namespace hipdnn_test_sdk::detail
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

        HIPDNN_LOG_INFO("Looking up plan builder for signature key: {}", key);

        auto it = _registry.find(key);
        if(it != _registry.end())
        {
            return *it->second;
        }

        throw std::runtime_error(
            fmt::format("No plan builder registered for signature key: {}", key));
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
