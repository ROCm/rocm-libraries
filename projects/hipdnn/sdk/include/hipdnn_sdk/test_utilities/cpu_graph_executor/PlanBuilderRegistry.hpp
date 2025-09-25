// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormSignatureRegistryKey.hpp>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferencePlan.hpp>

namespace hipdnn_sdk::test_utilities
{

using Key = std::variant<BatchnormSignatureRegistryKey /*, OtherKeyTypes...*/>;

struct KeyHash
{
    std::size_t operator()(Key const& k) const noexcept
    {
        return std::visit([](auto const& x) { return x.hash_self(); }, k);
    }
};

struct KeyEqual
{
    bool operator()(Key const& a, Key const& b) const noexcept
    {
        if(a.index() != b.index())
            return false; // different concrete types
        return std::visit([](auto const& x, auto const& y) { return x.equal(y); }, a, b);
    }
};

inline std::unordered_map<Key, std::unique_ptr<IGraphNodePlanBuilder>, KeyHash, KeyEqual>&
    planBuilderRegistry()
{
    static std::unordered_map<Key, std::unique_ptr<IGraphNodePlanBuilder>, KeyHash, KeyEqual>
        registry;
    return registry;
}

// //TODO FIX DOCUMENTATION FOR THIS.
template <std::size_t... Is>
void registerBatchnormFwdInferencePlanBuilders(std::index_sequence<Is...>)
{
    ((planBuilderRegistry()[ALL_SUPPORTED_BATCHNORM_SIGNATURES[Is]]
      = std::make_unique<BatchnormFwdInferencePlanBuilder<
          ALL_SUPPORTED_BATCHNORM_SIGNATURES[Is].inputDataType,
          ALL_SUPPORTED_BATCHNORM_SIGNATURES[Is].scaleBiasDataType,
          ALL_SUPPORTED_BATCHNORM_SIGNATURES[Is].meanVarianceDataType>>()),
     ...);
}

inline void initializeBatchnormRegistry()
{
    registerBatchnormFwdInferencePlanBuilders(
        std::make_index_sequence<ALL_SUPPORTED_BATCHNORM_SIGNATURES.size()>{});
}

struct BatchnormRegistryInitializer
{
    BatchnormRegistryInitializer()
    {
        initializeBatchnormRegistry();
    }
};

inline BatchnormRegistryInitializer _batchnormRegistryInitializer;

}
