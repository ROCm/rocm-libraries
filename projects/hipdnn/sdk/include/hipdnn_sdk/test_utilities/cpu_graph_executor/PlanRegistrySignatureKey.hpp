// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdSignatureKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormFwdInferenceSignatureKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormTrainSignatureKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionBwdSignatureKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionFwdSignatureKey.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PointwiseSignatureKey.hpp>

namespace hipdnn_sdk::test_utilities
{

/*
 * For each new op we add to our Plan registry we need to update this variant key to support it.
 * This way, we can have a single registry for all operations which simplifies the graph executor.
 * Each key must have a 
 *  - hashSelf() method
 *  - equality operator
 *  - hash operator
 *  - Constructor to build the key from a data_object::Node && tensorMap
 *  - A static method getPlanBuilders() which returns a map of keys to plan builders for that key
 * 
*/
using PlanRegistrySignatureKey = std::variant<BatchnormFwdInferenceSignatureKey,
                                              BatchnormBwdSignatureKey,
                                              BatchnormTrainSignatureKey,
                                              ConvolutionFwdSignatureKey,
                                              ConvolutionBwdSignatureKey,
                                              PointwiseSignatureKey>;

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
