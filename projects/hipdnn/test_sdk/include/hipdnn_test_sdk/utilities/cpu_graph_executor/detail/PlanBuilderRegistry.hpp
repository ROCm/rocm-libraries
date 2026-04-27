// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormBwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceWithVariancePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormTrainPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionBwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionFwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormFpropPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MatmulPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanRegistrySignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwisePlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormFwdPlan.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ReductionPlan.hpp>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <sstream>

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

        HIPDNN_SDK_LOG_INFO("Looking up plan builder for signature key: " << key);

        auto it = _registry.find(key);
        if(it != _registry.end())
        {
            return *it->second;
        }

        std::ostringstream oss;
        oss << "No plan builder registered for signature key: " << key;
        throw std::runtime_error(oss.str());
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

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
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
} // namespace hipdnn_test_sdk::detail
