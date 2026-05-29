// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>
#include <variant>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceWithVarianceSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormTrainSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizeSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionWrwSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormFpropSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MatmulSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwiseSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ReductionSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdSignatureKey.hpp>

namespace hipdnn_test_sdk::detail
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
                                              BatchnormFwdInferenceWithVarianceSignatureKey,
                                              BatchnormBwdSignatureKey,
                                              BatchnormTrainSignatureKey,
                                              BlockScaleDequantizeSignatureKey,
                                              ConvolutionFwdSignatureKey,
                                              ConvolutionBwdSignatureKey,
                                              ConvolutionWrwSignatureKey,
                                              LayernormFpropSignatureKey,
                                              MatmulSignatureKey,
                                              PointwiseSignatureKey,
                                              RMSNormFwdSignatureKey,
                                              RMSNormBwdSignatureKey,
                                              SdpaBwdSignatureKey,
                                              SdpaFwdSignatureKey,
                                              ReductionSignatureKey>;

struct PlanRegistrySignatureKeyHash
{
    std::size_t operator()(const PlanRegistrySignatureKey& k) const
    {
        return std::visit([](const auto& x) { return x.hashSelf(); }, k);
    }
};

struct PlanRegistrySignatureKeyEqual
{
    template <typename T, typename U>
    bool operator()([[maybe_unused]] const T& a, [[maybe_unused]] const U& b) const
    {
        return false;
    }

    template <typename T>
    bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }

    // NOLINTNEXTLINE(readability-redundant-casting)
    bool operator()(const PlanRegistrySignatureKey& a, const PlanRegistrySignatureKey& b) const
    {
        return std::visit(*this, a, b);
    }
};

inline std::ostream& operator<<(std::ostream& os, const PlanRegistrySignatureKey& key)
{
    std::visit([&os](const auto& arg) { os << arg; }, key);
    return os;
}

} // namespace hipdnn_test_sdk::detail
