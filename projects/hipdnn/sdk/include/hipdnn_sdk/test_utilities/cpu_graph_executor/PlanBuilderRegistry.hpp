// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdPlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormTrainPlan.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanRegistrySignatureKey.hpp>

namespace hipdnn_sdk::test_utilities
{

/*
 * Eventually we may wish to centalize all the supported signature arrays for all ops in another file
 * once we have a significant number of ops supported.
*/

constexpr std::array<BatchnormFwdInferenceSignatureKey, 3>
    ALL_SUPPORTED_BATCHNORM_FWD_INFERENCE_SIGNATURES
    = {BatchnormFwdInferenceSignatureKey(hipdnn_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_sdk::data_objects::DataType::FLOAT),
       BatchnormFwdInferenceSignatureKey(hipdnn_sdk::data_objects::DataType::HALF,
                                         hipdnn_sdk::data_objects::DataType::HALF,
                                         hipdnn_sdk::data_objects::DataType::HALF),
       BatchnormFwdInferenceSignatureKey(hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                         hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                         hipdnn_sdk::data_objects::DataType::BFLOAT16)};

constexpr std::array<BatchnormBwdSignatureKey, 3> ALL_SUPPORTED_BATCHNORM_BWD_SIGNATURES
    = {BatchnormBwdSignatureKey(hipdnn_sdk::data_objects::DataType::FLOAT,
                                hipdnn_sdk::data_objects::DataType::FLOAT,
                                hipdnn_sdk::data_objects::DataType::FLOAT),
       BatchnormBwdSignatureKey(hipdnn_sdk::data_objects::DataType::HALF,
                                hipdnn_sdk::data_objects::DataType::HALF,
                                hipdnn_sdk::data_objects::DataType::HALF),
       BatchnormBwdSignatureKey(hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                hipdnn_sdk::data_objects::DataType::BFLOAT16)};

constexpr std::array<BatchnormTrainSignatureKey, 3> ALL_SUPPORTED_BATCHNORM_TRAIN_SIGNATURES
    = {BatchnormTrainSignatureKey(hipdnn_sdk::data_objects::DataType::FLOAT,
                                  hipdnn_sdk::data_objects::DataType::FLOAT,
                                  hipdnn_sdk::data_objects::DataType::FLOAT),
       BatchnormTrainSignatureKey(hipdnn_sdk::data_objects::DataType::HALF,
                                  hipdnn_sdk::data_objects::DataType::HALF,
                                  hipdnn_sdk::data_objects::DataType::HALF),
       BatchnormTrainSignatureKey(hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                  hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                  hipdnn_sdk::data_objects::DataType::BFLOAT16)};

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
        registerBatchnormFwdInferencePlanBuilders(
            std::make_index_sequence<ALL_SUPPORTED_BATCHNORM_FWD_INFERENCE_SIGNATURES.size()>{});

        registerBatchnormBwdPlanBuilders(
            std::make_index_sequence<ALL_SUPPORTED_BATCHNORM_BWD_SIGNATURES.size()>{});

        registerBatchnormTrainPlanBuilders(
            std::make_index_sequence<ALL_SUPPORTED_BATCHNORM_TRAIN_SIGNATURES.size()>{});
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

    template <std::size_t... Is>
    void registerBatchnormTrainPlanBuilders([[maybe_unused]] std::index_sequence<Is...> sequence)
    {
        ((_registry[ALL_SUPPORTED_BATCHNORM_TRAIN_SIGNATURES[Is]]
          = std::make_unique<BatchnormTrainPlanBuilder<
              ALL_SUPPORTED_BATCHNORM_TRAIN_SIGNATURES[Is].inputDataType,
              ALL_SUPPORTED_BATCHNORM_TRAIN_SIGNATURES[Is].scaleBiasDataType,
              ALL_SUPPORTED_BATCHNORM_TRAIN_SIGNATURES[Is].meanVarianceDataType>>()),
         ...);
    }

    bool _initialized = false;
    PlanRegistryMap _registry;
};
}
