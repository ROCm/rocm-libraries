// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdSignatureKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferenceSignatureKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormTrainSignatureKey.hpp>

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
using PlanRegistrySignatureKey = std::variant<BatchnormFwdInferenceSignatureKey,
                                              BatchnormBwdSignatureKey,
                                              BatchnormTrainSignatureKey>;

struct PlanRegistrySignatureKeyHash
{
    std::size_t operator()(const PlanRegistrySignatureKey& k) const noexcept
    {
        return std::visit([](auto const& x) { return x.hashSelf(); }, k);
    }
};

struct PlanRegistrySignatureKeyEqual
{
    template <typename T, typename U>
    bool operator()([[maybe_unused]] const T& a, [[maybe_unused]] const U& b) const noexcept
    {
        return false;
    }

    template <typename T>
    bool operator()(const T& a, const T& b) const noexcept
    {
        return a == b;
    }

    bool operator()(const PlanRegistrySignatureKey& a,
                    const PlanRegistrySignatureKey& b) const noexcept
    {
        return std::visit(*this, a, b);
    }
};

}
