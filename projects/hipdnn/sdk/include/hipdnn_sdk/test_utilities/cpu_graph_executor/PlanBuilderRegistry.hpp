// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdPlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanRegistrySignatureKey.hpp>

namespace hipdnn_sdk::test_utilities
{

/*
 * Eventually we may wish to centalize all the supported signature arrays for all ops in another file
 * once we have a significant number of ops supported.
*/
constexpr std::array<BatchnormFwdInferenceSignatureKey, 2>
    ALL_SUPPORTED_BATCHNORM_FWD_INFERENCE_SIGNATURES
    = {BatchnormFwdInferenceSignatureKey(hipdnn_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_sdk::data_objects::DataType::FLOAT),
       BatchnormFwdInferenceSignatureKey(hipdnn_sdk::data_objects::DataType::HALF,
                                         hipdnn_sdk::data_objects::DataType::HALF,
                                         hipdnn_sdk::data_objects::DataType::HALF)};

constexpr std::array<BatchnormBwdSignatureKey, 1> ALL_SUPPORTED_BATCHNORM_BWD_SIGNATURES = {
    BatchnormBwdSignatureKey(hipdnn_sdk::data_objects::DataType::FLOAT,
                             hipdnn_sdk::data_objects::DataType::FLOAT,
                             hipdnn_sdk::data_objects::DataType::FLOAT) /*,
       BatchnormBwdSignatureKey(hipdnn_sdk::data_objects::DataType::HALF,
                                hipdnn_sdk::data_objects::DataType::HALF, // half is causing static cast errors in the ref impl...
                                hipdnn_sdk::data_objects::DataType::HALF)*/
};

class PlanBuilderRegistry
{
public:
    IGraphNodePlanBuilder* getPlanBuilder(const PlanRegistrySignatureKey& key)
    {
        initializeRegistry();

        auto it = _registry.find(key);
        if(it != _registry.end())
        {
            return it->second.get();
        }
        return nullptr;
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
        registerBatchnormFwdInferencePlanBuilders(
            std::make_index_sequence<ALL_SUPPORTED_BATCHNORM_FWD_INFERENCE_SIGNATURES.size()>{});

        registerBatchnormBwdPlanBuilders(
            std::make_index_sequence<ALL_SUPPORTED_BATCHNORM_BWD_SIGNATURES.size()>{});
    }

    template <std::size_t... Is>
    void registerBatchnormFwdInferencePlanBuilders(
        [[maybe_unused]] std::index_sequence<Is...> sequence)
    {
        ((_registry[ALL_SUPPORTED_BATCHNORM_FWD_INFERENCE_SIGNATURES[Is]]
          = std::make_unique<BatchnormFwdInferencePlanBuilder<
              ALL_SUPPORTED_BATCHNORM_FWD_INFERENCE_SIGNATURES[Is].inputDataType,
              ALL_SUPPORTED_BATCHNORM_FWD_INFERENCE_SIGNATURES[Is].scaleBiasDataType,
              ALL_SUPPORTED_BATCHNORM_FWD_INFERENCE_SIGNATURES[Is].meanVarianceDataType>>()),
         ...);
    }

    template <std::size_t... Is>
    void registerBatchnormBwdPlanBuilders([[maybe_unused]] std::index_sequence<Is...> sequence)
    {
        ((_registry[ALL_SUPPORTED_BATCHNORM_BWD_SIGNATURES[Is]]
          = std::make_unique<BatchnormBwdPlanBuilder<
              ALL_SUPPORTED_BATCHNORM_BWD_SIGNATURES[Is].inputDataType,
              ALL_SUPPORTED_BATCHNORM_BWD_SIGNATURES[Is].scaleBiasDataType,
              ALL_SUPPORTED_BATCHNORM_BWD_SIGNATURES[Is].meanVarianceDataType>>()),
         ...);
    }

    bool _initialized = false;
    std::unordered_map<PlanRegistrySignatureKey,
                       std::unique_ptr<IGraphNodePlanBuilder>,
                       PlanRegistrySignatureKeyHash,
                       PlanRegistrySignatureKeyEqual>
        _registry;
};

}
