// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferenceSignatureKey.hpp>

namespace hipdnn_sdk::test_utilities
{

/*
 * For each new op we add to our Plan registry we need to update this variant key to support it.
 * This way, we can have a single registry for all operations which simplifies the graph executor.
 * Each key must have a hashSelf() and equal() method to support hashing and equality comparison.
 * 
 * Additionally for new new key: 
 * - we need to update the CpuReferenceGraphExecutor::buildSignatureKey to 
 *   properly build the key so we can look up the plan builder in the registry.
 * - Add a templated plan builder function similar to registerBatchnormFwdInferencePlanBuilders
 *   to the registry class and call it in initializePlanBuilders().
 * - A constexpr array of all supported signatures for the new op.
 * 
*/
using Key = std::variant<BatchnormFwdInferenceSignatureKey /*, OtherKeyTypes...*/>;

struct KeyHash
{
    std::size_t operator()(Key const& k) const noexcept
    {
        return std::visit([](auto const& x) { return x.hashSelf(); }, k);
    }
};

struct KeyEqual
{
    bool operator()(Key const& a, Key const& b) const noexcept
    {
        if(a.index() != b.index())
        {
            return false; // different concrete types
        }
        return std::visit([](auto const& x, auto const& y) { return x.equal(y); }, a, b);
    }
};

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

class PlanBuilderRegistry
{
public:
    IGraphNodePlanBuilder* getPlanBuilder(const Key& key)
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

    bool _initialized = false;
    std::unordered_map<Key, std::unique_ptr<IGraphNodePlanBuilder>, KeyHash, KeyEqual> _registry;
};

}
